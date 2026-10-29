"""离线迭代 DPO（12 GB 显存下代替在线 RL）。一轮分四步，每步单独跑，避免模型同时占显存：

1. merge  ：把 SFT LoRA 合进基座，作为本轮的策略初始化和参考模型
   python -m orai.train.dpo_iter merge --adapter outputs/sft-v1/final --out models/orai-sft-v1-merged
2. sample ：对训练输入采样 K 个候选
   python -m orai.train.dpo_iter sample --model models/orai-sft-v1-merged --k 6 --limit 1200 --out data/dpo/r1.cands.jsonl
3. score  ：裁判团打分，组成偏好对
   python -m orai.train.dpo_iter score --cands data/dpo/r1.cands.jsonl --out data/dpo/r1.pairs.jsonl
4. train  ：DPO 训练新的 LoRA
   python -m orai.train.dpo_iter train --model models/orai-sft-v1-merged --pairs data/dpo/r1.pairs.jsonl --out outputs/dpo-r1
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from orai.train.format import END, STYLE_BY_DOMAIN, build_prompt


def cmd_merge(args):
    from unsloth import FastLanguageModel

    model, tok = FastLanguageModel.from_pretrained(model_name=args.adapter, max_seq_length=1536,
                                                   load_in_4bit=False, load_in_16bit=True)
    model.save_pretrained_merged(args.out, tok, save_method="merged_16bit")
    print("merged ->", args.out)


def cmd_sample(args):
    from orai.infer.generate import Rewriter

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.limit]
    rw = Rewriter(args.model)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for style in ("书面", "口语"):
            sub = [r for r in rows if STYLE_BY_DOMAIN[r["domain"]] == style]
            for k in range(0, len(sub), 32):
                part = sub[k:k + 32]
                cands = rw.generate([r["ai"] for r in part], style=style, n=args.k, temperature=args.temperature)
                for r, c in zip(part, cands):
                    f.write(json.dumps({"id": r["id"], "style": style, "ai": r["ai"], "human": r["human"],
                                        "cands": c}, ensure_ascii=False) + "\n")
                f.flush()
                print(f"{style} {k + len(part)}/{len(sub)}", flush=True)


def cmd_score(args):
    from orai.infer.judge import Judge

    judge = Judge(with_classifier=True)
    rng = random.Random(0)
    n_pairs, n_human = 0, 0
    with open(args.out, "w", encoding="utf-8") as f:
        for line in open(args.cands, encoding="utf-8"):
            d = json.loads(line)
            ranked = judge.rank(d["ai"], d["cands"])
            ok = [r for r in ranked if r["ok"]]
            if not ok:
                continue
            chosen = ok[0]
            # rejected：AI 分最高的候选，或者没过语义门槛的候选
            rejected = max(ranked, key=lambda r: (not r["ok"], r["ai_score"]))
            if rejected is chosen or rejected["ai_score"] - chosen["ai_score"] < args.margin and rejected["ok"]:
                continue
            prompt = build_prompt(d["ai"], d["style"])
            f.write(json.dumps({"prompt": prompt, "chosen": chosen["text"] + END, "rejected": rejected["text"] + END,
                                "gap": rejected["ai_score"] - chosen["ai_score"]}, ensure_ascii=False) + "\n")
            n_pairs += 1
            # 参考 MASH：一部分直接用真人原文作为 chosen，把策略往真人分布上拉
            if rng.random() < args.human_ratio:
                f.write(json.dumps({"prompt": prompt, "chosen": d["human"] + END, "rejected": rejected["text"] + END,
                                    "gap": -1}, ensure_ascii=False) + "\n")
                n_human += 1
    print(f"pairs {n_pairs} + human-chosen {n_human} -> {args.out}")


def cmd_train(args):
    from unsloth import FastLanguageModel, PatchDPOTrainer

    PatchDPOTrainer()
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    model, tok = FastLanguageModel.from_pretrained(model_name=args.model, max_seq_length=1536,
                                                   load_in_4bit=False, load_in_16bit=True)
    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.rank, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407)
    rows = [json.loads(l) for l in open(args.pairs, encoding="utf-8")]
    ds = Dataset.from_list([{k: r[k] for k in ("prompt", "chosen", "rejected")} for r in rows]).shuffle(seed=0)
    print("dpo pairs", len(ds))
    trainer = DPOTrainer(
        model=model, ref_model=None, processing_class=tok, train_dataset=ds,
        args=DPOConfig(output_dir=args.out, beta=args.beta, learning_rate=args.lr, num_train_epochs=1,
                       per_device_train_batch_size=1, gradient_accumulation_steps=8, lr_scheduler_type="cosine",
                       warmup_ratio=0.05, optim="adamw_8bit", bf16=True, logging_steps=5, save_strategy="no",
                       max_length=1536, report_to="none", seed=0),
    )
    trainer.train()
    model.save_pretrained(args.out + "/final")
    tok.save_pretrained(args.out + "/final")
    print("saved", args.out + "/final")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("merge"); m.add_argument("--adapter", required=True); m.add_argument("--out", required=True)
    s = sp.add_parser("sample")
    s.add_argument("--model", required=True); s.add_argument("--out", required=True)
    s.add_argument("--data", default="data/pairs/train.jsonl"); s.add_argument("--k", type=int, default=6)
    s.add_argument("--limit", type=int, default=1200); s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--seed", type=int, default=1)
    c = sp.add_parser("score")
    c.add_argument("--cands", required=True); c.add_argument("--out", required=True)
    c.add_argument("--margin", type=float, default=0.05); c.add_argument("--human-ratio", type=float, default=0.3)
    t = sp.add_parser("train")
    t.add_argument("--model", required=True); t.add_argument("--pairs", required=True); t.add_argument("--out", required=True)
    t.add_argument("--beta", type=float, default=0.1); t.add_argument("--lr", type=float, default=5e-6)
    t.add_argument("--rank", type=int, default=32)
    a = ap.parse_args()
    {"merge": cmd_merge, "sample": cmd_sample, "score": cmd_score, "train": cmd_train}[a.cmd](a)
