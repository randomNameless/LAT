import torch
from torch.optim.optimizer import Optimizer
import logging

INIT_TYPES = ["instruction", "suffix"]

LABEL_PAD_TOKEN_ID = -100  # [CHANGED] HF/TRL 标准 padding label id


class NoAttack:
    def __init__(
        self,
        embed_weights,
    ) -> None:
        """
        No Attack
        """
        self.embed_weights = embed_weights
        self.vocab_size = self.embed_weights.shape[0]
        self.embedding_size = self.embed_weights.shape[1]
        self.label_pad_token_id = LABEL_PAD_TOKEN_ID  # [CHANGED]
        return

    def attack(self, model, input_ids, target_ids, attention_mask, **kwargs):
        # kwargs ignored for NoAttack (keeps interface compatible)  # [CHANGED]
        all_losses = []
        affirmative_responses = []

        adv_perturbation, adv_perturbation_mask = self.init_perturbation(
            input_ids, target_ids, attention_mask
        )
        input_embeds = self.get_embeddings(input_ids)

        return (
            input_embeds.detach(),
            adv_perturbation.detach(),
            adv_perturbation_mask.detach(),
            all_losses,
            affirmative_responses,
        )

    def get_adv_embeddings(self, input_embeds, adv_perturbation, adv_perturbation_mask):
        return input_embeds

    def apply_perturbation_to_batch(  # [CHANGED]
        self,
        input_ids,
        labels,
        attention_mask,
        adv_perturbation,
        adv_perturbation_mask,
    ):
        # NoAttack: do nothing
        return self.get_embeddings(input_ids)

    def init_perturbation(self, input_ids, target_ids, attention_mask):
        # [CHANGED] prompt tokens are those with labels == -100 and attention_mask == 1
        prompt_mask = (target_ids == self.label_pad_token_id) & attention_mask.to(torch.bool)
        input_mask = prompt_mask

        batch_size, num_input_tokens = input_ids.shape
        dtype = self.embed_weights.dtype

        adv_perturbation = torch.zeros(
            (batch_size, num_input_tokens, self.embedding_size),
            device=input_ids.device,
            dtype=dtype,
        )

        adv_perturbation_mask = torch.zeros(
            (batch_size, num_input_tokens),
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        adv_perturbation_mask[input_mask] = 1

        return adv_perturbation, adv_perturbation_mask.unsqueeze(-1)

    def get_one_hot(self, ids):
        device = self.embed_weights.device
        batch_size, num_tokens = ids.shape

        # Adjusting IDs less than 0 to 0
        ids = torch.where(ids < 0, torch.tensor(0, device=device, dtype=ids.dtype), ids)

        one_hot = torch.zeros(
            batch_size,
            num_tokens,
            self.vocab_size,
            device=device,
            dtype=self.embed_weights.dtype,
        )
        one_hot.scatter_(2, ids.unsqueeze(2), 1)
        return one_hot

    def get_embeddings(self, ids):
        one_hot = self.get_one_hot(ids)
        embeddings = (one_hot @ self.embed_weights).data
        return embeddings


class EmbeddingSpaceAttack:
    def __init__(
        self,
        embed_weights,
        response_key,
        tokenizer,
        iters=8,
        opt_config=None,
        eps=0.01,
        init_type="instruction",
        suffix_tokens=10,
        relative_lr=False,
        debug=0,
        *args,
        **kwargs,
    ) -> None:
        """
        Initializes the EmbeddingAttack class.
        """

        self.embed_weights = embed_weights
        self.tokenizer = tokenizer
        self.vocab_size = self.embed_weights.shape[0]
        self.embedding_size = self.embed_weights.shape[1]
        self.embedding_norm = torch.norm(embed_weights, p=2, dim=-1).mean()
        self.iters = iters
        self.opt_config = opt_config
        self.label_pad_token_id = LABEL_PAD_TOKEN_ID  # [CHANGED]

        # [CHANGED] define eps before relative_lr uses it
        self.eps = eps * self.embedding_norm

        if relative_lr and self.opt_config is not None:
            # [CHANGED] use already-defined self.eps
            self.opt_config["lr"] = self.opt_config["lr"] * self.eps

        logging.info(
            f"L2 norm of embedding weights equals {self.embedding_norm} eps multiplier is: {eps} using eps: {self.eps}"
        )

        if init_type not in INIT_TYPES:
            ValueError(f"init_type must be in {INIT_TYPES} and not {self.init_type}")
        self.init_type = init_type

        self.suffix_tokens = suffix_tokens
        self.debug = debug
        self.loss_fct = torch.nn.CrossEntropyLoss()

    def attack(
        self,
        model,
        input_ids,
        target_ids,
        attention_mask,
        contrast_input_ids=None,          # [CHANGED]
        contrast_target_ids=None,         # [CHANGED]
        contrast_attention_mask=None,     # [CHANGED]
        contrast_weight=1.0,              # [CHANGED]
    ):
        disable_model_gradients(model)

        best_loss = torch.inf
        all_losses = []
        affirmative_responses = []

        adv_perturbation, adv_perturbation_mask = self.init_perturbation(
            input_ids, target_ids, attention_mask
        )

        input_embeds = self.get_embeddings(input_ids)
        target_one_hot = self.get_one_hot(target_ids)

        attention_mask = self.get_attention_mask(input_ids, attention_mask)
        loss_mask = self.get_loss_mask(target_ids)

        # [CHANGED] contrast preparation (toward/safe)
        use_contrast = (
            contrast_input_ids is not None
            and contrast_target_ids is not None
            and contrast_attention_mask is not None
        )
        if use_contrast:
            contrast_target_one_hot = self.get_one_hot(contrast_target_ids)
            contrast_attention_mask = self.get_attention_mask(contrast_input_ids, contrast_attention_mask)
            contrast_loss_mask = self.get_loss_mask(contrast_target_ids)

        opt = self.init_opt([adv_perturbation])

        if self.debug > 2:
            self.debug_shapes(
                input_embeds,
                target_one_hot,
                adv_perturbation,
                adv_perturbation_mask,
                attention_mask,
                loss_mask,
            )

        for i in range(self.iters):
            opt.zero_grad()

            adv_embeds = self.get_adv_embeddings(input_embeds, adv_perturbation, adv_perturbation_mask)
            logits, loss_away = self.calc_loss(
                i, model, adv_embeds, target_one_hot, attention_mask, loss_mask
            )

            # [CHANGED] inner-loop contrast term: maximize safe CE while minimizing harmful CE
            if use_contrast:
                contrast_embeds = self.apply_perturbation_to_batch(
                    contrast_input_ids,
                    contrast_target_ids,
                    contrast_attention_mask,
                    adv_perturbation,
                    adv_perturbation_mask,
                )
                _, loss_contrast = self.calc_loss(
                    i,
                    model,
                    contrast_embeds,
                    contrast_target_one_hot,
                    contrast_attention_mask,
                    contrast_loss_mask,
                    log_debug=False,
                )
                loss = loss_away - contrast_weight * loss_contrast
            else:
                loss = loss_away

            loss.backward()
            all_losses.append(loss.item())
            opt.step()

            if self.init_type == "instruction":
                adv_perturbation = self.project_l2(adv_perturbation)
            elif self.init_type == "suffix":
                adv_perturbation = self.project_simplex(adv_perturbation)

            num_affirmative_responses = self.get_num_affirmative_responses(target_ids, logits)
            affirmative_responses.append(num_affirmative_responses)

            if loss < best_loss:
                best_loss = loss
                self.best_adv_perturbation = adv_perturbation

            if self.debug > 2:
                self.debug_norm(adv_perturbation)

        if self.debug > 0:
            self.debug_loss(best_loss)
        if self.debug > 2:
            self.debug_output(target_ids, logits, attention_mask)

        enable_model_gradients(model)

        return (
            input_embeds.detach(),
            adv_perturbation.detach(),
            adv_perturbation_mask.detach(),
            all_losses,
            affirmative_responses,
        )

    def calc_loss(self, i, model, input_embeds, target_one_hot, attention_mask, loss_mask, log_debug=True):
        output = model(inputs_embeds=input_embeds, attention_mask=attention_mask)
        logits = output.logits

        logits_loss_mask = logits[:, :-1][loss_mask]
        target_one_hot_loss_mask = target_one_hot[:, 1:][loss_mask]
        loss = self.loss_fct(logits_loss_mask, target_one_hot_loss_mask)

        if i == 0:
            self.logits_benign = logits
            self.loss_benign = loss

        if self.debug > 1 and log_debug:
            self.debug_iter_loss(i, loss)

        return logits, loss

    def project_l2(self, adv_perturbation):
        norm = torch.norm(adv_perturbation, p=2, dim=-1, keepdim=True)
        mask = (norm > self.eps).squeeze()
        if torch.any(mask):
            with torch.no_grad():
                if len(mask.shape) == 1:
                    mask = mask.unsqueeze(0)
                adv_perturbation[mask, :] = adv_perturbation[mask, :] / norm[mask] * self.eps

        return adv_perturbation

    def project_simplex(self, adv_perturbation):
        raise NotImplementedError("Simplex projection not implemented yet")

    def get_one_hot(self, ids):
        device = self.embed_weights.device
        batch_size, num_tokens = ids.shape

        ids = torch.where(ids < 0, torch.tensor(0, device=device, dtype=ids.dtype), ids)

        one_hot = torch.zeros(
            batch_size,
            num_tokens,
            self.vocab_size,
            device=device,
            dtype=self.embed_weights.dtype,
        )
        one_hot.scatter_(2, ids.unsqueeze(2), 1)
        return one_hot

    def get_embeddings(self, ids):
        one_hot = self.get_one_hot(ids)
        embeddings = (one_hot @ self.embed_weights).data
        return embeddings

    def get_adv_embeddings(self, input_embeds, adv_perturbation, adv_perturbation_mask):
        masked_perturbation = adv_perturbation * adv_perturbation_mask
        adv_embeds = input_embeds + masked_perturbation
        return adv_embeds

    def apply_perturbation_to_batch(  # [CHANGED]
        self,
        input_ids,
        labels,
        attention_mask,
        adv_perturbation,
        adv_perturbation_mask,
    ):
        """
        Apply the (away-optimized) perturbation to another batch (e.g., toward/safe) by aligning
        prompt positions using (labels == -100) and right-aligning the prompt tokens.
        """
        input_embeds = self.get_embeddings(input_ids)
        out = input_embeds.clone()

        away_prompt_mask = adv_perturbation_mask.squeeze(-1).to(torch.bool)
        other_prompt_mask = (labels == self.label_pad_token_id) & attention_mask.to(torch.bool)

        B = out.shape[0]
        B2 = adv_perturbation.shape[0]
        Bm = min(B, B2)

        out = out[:Bm]
        other_prompt_mask = other_prompt_mask[:Bm]
        away_prompt_mask = away_prompt_mask[:Bm]
        adv_perturbation = adv_perturbation[:Bm]

        for b in range(Bm):
            idxA = torch.nonzero(away_prompt_mask[b], as_tuple=False).squeeze(-1)
            idxB = torch.nonzero(other_prompt_mask[b], as_tuple=False).squeeze(-1)
            if idxA.numel() == 0 or idxB.numel() == 0:
                continue
            L = min(idxA.numel(), idxB.numel())
            idxA = idxA[-L:]
            idxB = idxB[-L:]
            out[b, idxB, :] = out[b, idxB, :] + adv_perturbation[b, idxA, :]

        return out

    def get_loss_slice_start_and_end(self, input_embeds):
        input_len = input_embeds.shape[1]

        if self.init_type == "instruction":
            start = input_len - 1
            end = -1
        elif self.init_type == "suffix":
            start = input_len + self.suffix_tokens
            end = -1
        return start, end

    def get_attention_mask(self, input_ids, attention_mask):
        if self.init_type == "instruction":
            return attention_mask
        elif self.init_type == "suffix":
            len_input = input_ids.shape[1]
            input_mask = attention_mask[:, :len_input]
            adversarial_mask = torch.ones(
                (attention_mask.shape[0], self.suffix_tokens),
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            target_mask = attention_mask[:, len_input:]
            attention_mask = torch.hstack([input_mask, adversarial_mask, target_mask])
            return attention_mask

    def get_loss_mask(self, target_ids):
        # [CHANGED] target tokens are those != -100
        target_mask = target_ids != self.label_pad_token_id
        if self.init_type == "instruction":
            return target_mask[:, 1:]
        elif self.init_type == "suffix":
            padding_for_suffix = torch.zeros(
                (target_mask.shape[0], self.suffix_tokens),
                dtype=target_ids.dtype,
                device=target_ids.device,
            )
            loss_mask = torch.hstack([padding_for_suffix, target_mask])
        return loss_mask

    def init_perturbation(self, input_ids, target_ids, attention_mask):
        # [CHANGED] prompt positions: labels == -100 & attention_mask==1
        prompt_mask = (target_ids == self.label_pad_token_id) & attention_mask.to(torch.bool)
        input_mask = prompt_mask

        batch_size, num_input_tokens = input_ids.shape
        dtype = self.embed_weights.dtype

        if self.init_type == "instruction":
            adv_perturbation = torch.zeros(
                (batch_size, num_input_tokens, self.embedding_size),
                device=input_ids.device,
                dtype=dtype,
            )

            adv_perturbation_mask = torch.zeros(
                (batch_size, num_input_tokens),
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
            adv_perturbation_mask[input_mask] = 1
        elif self.init_type == "suffix":
            adv_perturbation = torch.randn(
                (batch_size, num_input_tokens + self.suffix_tokens, self.embedding_size),
                device=input_ids.device,
                dtype=dtype,
            )
            adv_perturbation = self.project_simplex(adv_perturbation)

            adv_perturbation_mask = torch.zeros(
                (batch_size, num_input_tokens + self.suffix_tokens),
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
            num_false = torch.sum(input_mask, dim=1)
            row_indices = torch.arange(adv_perturbation_mask.shape[0])
            col_indices = num_false[:, None] + torch.arange(self.suffix_tokens)
            col_indices = torch.clip(col_indices, 0, adv_perturbation_mask.shape[1] - 1)
            adv_perturbation_mask[row_indices[:, None], col_indices] = True

        adv_perturbation.requires_grad = True
        adv_perturbation_mask = adv_perturbation_mask.unsqueeze(2)

        self.best_adv_perturbation = adv_perturbation

        return adv_perturbation, adv_perturbation_mask

    def init_opt(self, parameters):
        if self.opt_config is None:
            self.opt_config = {"type": "sign", "lr": 0.01}
            logging.info(f"No opt_config specified using default opt_config: {self.opt_config}")

        optimizer_type = self.opt_config["type"]
        if optimizer_type == "adam":
            opt = torch.optim.Adam(parameters, lr=self.opt_config["lr"])
        elif optimizer_type == "sign":
            opt = SignSGD(parameters, lr=self.opt_config["lr"])
        elif optimizer_type == "rms":
            opt = torch.optim.RMSprop(parameters, lr=self.opt_config["lr"])
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")  # [CHANGED]

        return opt

    def get_num_affirmative_responses(self, target_ids, logits):
        target_ids_clone = target_ids.clone()
        target_ids_clone = target_ids_clone[:, 1:]
        output_ids = torch.argmax(logits, dim=-1)
        output_ids = output_ids[:, :-1]

        # [CHANGED] ignore prompt/pad labels
        input_mask = target_ids_clone == self.label_pad_token_id
        output_ids[input_mask] = 0
        target_ids_clone[input_mask] = 0

        affirmative_responses = output_ids == target_ids_clone
        affirmative_responses_sum = affirmative_responses.all(dim=-1).sum().item()

        return affirmative_responses_sum

    def debug_norm(self, adv_perturbation):
        if self.init_type == "instruction":
            norm = torch.norm(adv_perturbation, p=2, dim=-1).max()
            logging.info(f"Debugging ESA | L2 Norm max adversarial perturbation: {norm}")

    def debug_loss(self, loss_adversarial):
        logging.info(
            f"Debugging ESA | Benign loss: {self.loss_benign.item()} | Best Adversarial loss {loss_adversarial.item()}"
        )

    def debug_iter_loss(self, i, loss_adversarial):
        logging.info(f"Debugging ESA | i: {i} | Adversarial loss: {loss_adversarial.item()}")

    def debug_shapes(
        self,
        input_embeds,
        target_one_hot,
        adv_perturbation,
        adv_perturbation_mask,
        attention_mask,
        loss_mask,
    ):
        logging.info(
            "====== Debugging ESA Adversarial Attack Shapes ======\n"
            f"input_embeds: {input_embeds.shape}\n"
            f"adv_perturbation: {adv_perturbation.shape}\n"
            f"adv_perturbation_mask: {adv_perturbation_mask.shape}\n"
            f"target_one_hot: {target_one_hot.shape}\n"
            f"attention_mask: {attention_mask.shape}\n"
            f"loss_mask: {loss_mask.shape}"
        )

    def debug_output(self, target_ids, logits, attention_mask):
        with torch.no_grad():
            attention_mask = attention_mask.to(dtype=torch.bool)

            target_ids = target_ids[0][1:]
            target_mask = target_ids != self.label_pad_token_id  # [CHANGED]
            only_targets = target_ids[target_mask]
            original_text = self.tokenizer.decode(only_targets, skip_special_tokens=True)

            output_ids_benign = torch.argmax(self.logits_benign, dim=-1)
            output_ids_benign = output_ids_benign[0][:-1][target_mask]
            generated_text_benign = self.tokenizer.decode(output_ids_benign, skip_special_tokens=True)

            output_ids_adv = torch.argmax(logits, dim=-1)
            output_ids_adv = output_ids_adv[0][:-1][target_mask]
            generated_text_adv = self.tokenizer.decode(output_ids_adv, skip_special_tokens=True)
            logging.info(
                "===== Debugging ESA Original text ====\n"
                f"{original_text}\n"
                "===== Debugging ESA Generated text benign ====\n"
                f"{generated_text_benign}\n"
                "===== Debugging ESA Generated text adversarial ====\n"
                f"{generated_text_adv}"
            )


def disable_model_gradients(model):
    for name, param in model.named_parameters():
        if param.requires_grad and "lora" not in name:
            raise ValueError(f"Non-Lora Parameter {name} requires grad")
        param.requires_grad = False


def enable_model_gradients(model, only_train_lora=True):
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True


class SignSGD(Optimizer):
    def __init__(self, params, lr=0.01):
        defaults = dict(lr=lr)
        super(SignSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    sign = torch.sign(grad)
                    p.add_(other=sign, alpha=-group["lr"])

        return loss
