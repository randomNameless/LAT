import torch
from torch.utils.data import DataLoader

import data
import model_utils


MODEL_PATH = "google/gemma-2b-it"      # 跟你训练用的一致
MODEL_NAME = "gemma-2b-it"            # 跟 get_model_name 输出一致
DATA_PATH  = "../../../data/"         # 跟 cfg.dataset.data_path 一致

def main():
    model, tokenizer = model_utils.load_model_and_tokenizer(
        MODEL_PATH,
        bnb_config=None,              # 只是看数据，不需要4bit
        padding_side="left",
        dtype="bf16",
    )

    formatting_func, collator, response_key = data.get_prompt_formatting_func_and_collator(
        MODEL_NAME, tokenizer
    )
    print("response_key =", response_key)

    train_data, _ = data.load_adversarial_training_data(
        data_path=DATA_PATH,
        utility_data="None",          # 先关掉utility，避免 dataset_id=2 混进来影响你观察
        probabilities=[1.0, 0.0],
        model_name=MODEL_NAME,
        tokenizer=tokenizer,
        stopping_strategy="first_exhausted",
        diverse_safe_answers=False,
        restricted_trainingset_size=20,
    )

    dl = DataLoader(train_data, batch_size=2, collate_fn=collator, shuffle=False)
    batch = next(iter(dl))

    # batch keys 一般会有: input_ids, attention_mask, labels, dataset_id (以及dpo的话 logps)
    print("batch keys:", list(batch.keys()))
    print("dataset_id:", batch["dataset_id"].tolist())

    labels = batch["labels"][0].cpu()
    input_ids = batch["input_ids"][0].cpu()

    print("\nlabels[:120] =", labels[:120].tolist())
    print("num -100 =", int((labels == -100).sum().item()), "/", labels.numel())

    labeled = labels[labels != -100]
    print("\ndecoded labeled part:\n", tokenizer.decode(labeled, skip_special_tokens=False))

    print("\ndecoded full input_ids:\n", tokenizer.decode(input_ids, skip_special_tokens=False))

    # 顺便把 “labels里第一个非-100的位置” 打出来，确认assistant回复从哪开始
    idx = (labels != -100).nonzero(as_tuple=True)[0]
    if len(idx) > 0:
        print("\nfirst labeled index =", int(idx[0].item()))
    else:
        print("\nWARNING: no labeled tokens found (all -100)")

if __name__ == "__main__":
    main()
