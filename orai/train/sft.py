"""SFT：Qwen3.5-4B-Base + bf16 LoRA，学「AI 版本 → 真人原文」。

python -m orai.train.sft --out outputs/sft-v1
"""

from __future__ import annotations

import argparse
import json

from orai.train.format import END, STYLE_BY_DOMAIN, build_prompt


def load_rows(path: str):
    rows = []
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        rows.append({"prompt": build_prompt(d["ai"], STYLE_BY_DOMAIN[d["domain"]]),
                     "completion": d["human"].strip() + END})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="models/Qwen3.5-4B-Base-text")
    ap.add_argument("--data", default="data/pairs/train.jsonl")
    ap.add_argument("--out", default="outputs/sft-v1")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--quant", choices=["8bit", "4bit", "16bit"], default="8bit",
                    help="基座精度。12 GB 卡上 4B 的 bf16 LoRA 放不下，默认 8-bit")
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()

    from unsloth import FastLanguageModel  # 必须在 transformers / trl 之前导入
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.base, max_seq_length=args.max_len,
        load_in_4bit=args.quant == "4bit", load_in_8bit=args.quant == "8bit",
        load_in_16bit=args.quant == "16bit", full_finetuning=False,
    )
    # 全部线性层：注意力、MLP，以及 Gated DeltaNet 的投影层
    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.rank, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407,
    )
    ds = Dataset.from_list(load_rows(args.data)).shuffle(seed=0)
    print("train rows", len(ds))

    trainer = SFTTrainer(
        model=model, processing_class=tok, train_dataset=ds,
        args=SFTConfig(
            output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
            per_device_train_batch_size=args.bs, gradient_accumulation_steps=args.grad_accum,
            lr_scheduler_type="cosine", warmup_ratio=0.05, weight_decay=0.0,
            optim="adamw_8bit", bf16=True, logging_steps=5, save_strategy="epoch",
            max_length=args.max_len, completion_only_loss=True, report_to="none", seed=0,
            dataset_num_proc=1, max_steps=args.max_steps,
        ),
    )
    trainer.train()
    model.save_pretrained(args.out + "/final")
    tok.save_pretrained(args.out + "/final")
    print("saved", args.out + "/final")


if __name__ == "__main__":
    main()
