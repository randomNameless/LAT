import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_from_disk, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm


def find_last_assistant(messages: List[Dict[str, Any]]) -> Optional[int]:
    last_idx = None
    for i, m in enumerate(messages):
        role = m.get("role", None)
        if role == "assistant":
            last_idx = i
    return last_idx


def safe_apply_chat_template(tokenizer, messages: List[Dict[str, Any]], add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    parts = []
    for m in messages:
        role = m.get("role", "unknown")
        content = m.get("content", "")
        parts.append(f"{role.upper()}: {content}\n")
    if add_generation_prompt:
        parts.append("ASSISTANT: ")
    return "".join(parts)


def build_text_pair(tokenizer, messages: List[Dict[str, Any]]) -> Optional[Tuple[str, str]]:
    if not isinstance(messages, list) or len(messages) == 0:
        return None

    has_role = any(isinstance(m, dict) and "role" in m for m in messages)
    if not has_role:
        return None

    a_idx = find_last_assistant(messages)
    if a_idx is None:
        return None

    prompt_msgs = messages[:a_idx]
    full_msgs = messages[:a_idx + 1]

    prompt_text = safe_apply_chat_template(tokenizer, prompt_msgs, add_generation_prompt=True)
    full_text = safe_apply_chat_template(tokenizer, full_msgs, add_generation_prompt=False)
    return prompt_text, full_text


def truncate_left(input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor, max_len: int):
    if input_ids.size(1) <= max_len:
        return input_ids, attention_mask, labels, 0
    cut = input_ids.size(1) - max_len
    input_ids = input_ids[:, cut:]
    attention_mask = attention_mask[:, cut:]
    labels = labels[:, cut:]
    return input_ids, attention_mask, labels, cut


def collate_batch(examples: List[Dict[str, Any]], pad_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(e["input_ids"].numel() for e in examples)
    bsz = len(examples)

    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)

    for i, e in enumerate(examples):
        L = e["input_ids"].numel()
        input_ids[i, :L] = e["input_ids"]
        attention_mask[i, :L] = e["attention_mask"]
        labels[i, :L] = e["labels"]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@torch.no_grad()
def eval_adapter(
    base_model: str,
    adapter_dir: str,
    data_dir: str,
    max_samples: int,
    batch_size: int,
    max_seq_len: int,
    dtype: str,
    device: str,
    log_every: int = 50,  # NEW: log every N batches
) -> Dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if os.path.isdir(data_dir):
        ds = load_from_disk(data_dir)
    else:
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    if max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]

    # NOTE: device_map=None means full model on one device.
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map=None,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    model.to(device)

    total_loss_sum = 0.0
    total_tokens = 0
    n_used = 0
    n_skipped = 0

    buf: List[Dict[str, Any]] = []

    pbar = tqdm(total=len(ds), desc=f"Utility eval [{os.path.basename(adapter_dir)}]", unit="ex")
    batch_count = 0

    for i in range(len(ds)):
        ex = ds[i]
        messages = ex.get("messages", None)
        pair = build_text_pair(tokenizer, messages)
        if pair is None:
            n_skipped += 1
            pbar.update(1)
            continue

        prompt_text, full_text = pair

        prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        full_enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
        input_ids_1d = full_enc["input_ids"][0]
        attn_1d = full_enc["attention_mask"][0]

        prompt_len = int(prompt_ids.numel())
        labels_1d = input_ids_1d.clone()
        labels_1d[:prompt_len] = -100

        input_ids = input_ids_1d.unsqueeze(0)
        attention_mask = attn_1d.unsqueeze(0)
        labels = labels_1d.unsqueeze(0)
        input_ids, attention_mask, labels, _cut = truncate_left(input_ids, attention_mask, labels, max_seq_len)

        input_ids = input_ids[0]
        attention_mask = attention_mask[0]
        labels = labels[0]

        buf.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})
        pbar.update(1)

        if len(buf) >= batch_size:
            batch = collate_batch(buf, tokenizer.pad_token_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss

            active = (batch["labels"] != -100)
            tokens = int(active.sum().item())
            total_loss_sum += float(loss.item()) * max(tokens, 1)
            total_tokens += tokens
            n_used += len(buf)

            buf = []
            batch_count += 1

            if log_every > 0 and (batch_count % log_every == 0):
                mean_loss_so_far = total_loss_sum / max(total_tokens, 1)
                pbar.set_postfix(
                    used=n_used,
                    skipped=n_skipped,
                    toks=total_tokens,
                    mean_nll=f"{mean_loss_so_far:.4f}",
                )

    # flush remainder
    if len(buf) > 0:
        batch = collate_batch(buf, tokenizer.pad_token_id)
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss

        active = (batch["labels"] != -100)
        tokens = int(active.sum().item())
        total_loss_sum += float(loss.item()) * max(tokens, 1)
        total_tokens += tokens
        n_used += len(buf)

    pbar.close()

    mean_loss = total_loss_sum / max(total_tokens, 1)
    ppl = math.exp(mean_loss) if mean_loss < 50 else float("inf")

    return {
        "base_model": base_model,
        "adapter": adapter_dir,
        "data_dir": data_dir,
        "max_samples": max_samples,
        "batch_size": batch_size,
        "max_seq_len": max_seq_len,
        "dtype": dtype,
        "n_rows_total": int(len(ds)),
        "n_rows_used": int(n_used),
        "n_rows_skipped_no_assistant": int(n_skipped),
        "assistant_tokens_scored": int(total_tokens),
        "mean_nll_loss": float(mean_loss),
        "perplexity": float(ppl),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_a", required=True)
    ap.add_argument("--adapter_b", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_json", default="./asr_compare/utility_ultrachat/report.json")
    ap.add_argument("--max_samples", type=int, default=2000)  # set 0 to use all
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log_every", type=int, default=50, help="Update tqdm postfix every N batches")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    print(f"[INFO] Evaluating adapter A: {args.adapter_a}", flush=True)
    res_a = eval_adapter(
        base_model=args.base_model,
        adapter_dir=args.adapter_a,
        data_dir=args.data_dir,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        dtype=args.dtype,
        device=args.device,
        log_every=args.log_every,
    )

    print(f"[INFO] Evaluating adapter B: {args.adapter_b}", flush=True)
    res_b = eval_adapter(
        base_model=args.base_model,
        adapter_dir=args.adapter_b,
        data_dir=args.data_dir,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        dtype=args.dtype,
        device=args.device,
        log_every=args.log_every,
    )

    out = {"A": res_a, "B": res_b}
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("[DONE] Wrote:", args.out_json, flush=True)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()

