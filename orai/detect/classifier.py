"""有监督 AI 文本分类器（D2）：chinese-roberta-wwm-ext + 二分类头。

训练：python -m orai.detect.classifier train
数据：HC3-Chinese（human_answers vs chatgpt_answers）+ 我们自己的配对训练集（原文 vs AI 版本）。
评测集里的原文和 AI 版本不参与训练。
"""

from __future__ import annotations

import argparse
import json
import random
import re

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

from orai.collect.build_human import normalize_punct


def load_training_texts(pairs_path: str, hc3_path: str, hc3_limit: int, seed: int):
    rng = random.Random(seed)
    rows = []
    for line in open(pairs_path, encoding="utf-8"):
        d = json.loads(line)
        rows.append((d["human"], 0))
        rows.append((d["ai"], 1))
    hc3 = []
    for line in open(hc3_path, encoding="utf-8"):
        d = json.loads(line)
        for a in d.get("human_answers") or []:
            if isinstance(a, str) and 80 <= len(a) <= 600:
                hc3.append((normalize_punct(a), 0))
        for a in d.get("chatgpt_answers") or []:
            if isinstance(a, str) and 80 <= len(a) <= 600:
                # 去掉 ChatGPT 回答里的分点编号，避免分类器只学格式
                hc3.append((normalize_punct(re.sub(r"\n\s*\d+[.、]", "", a)), 1))
    rng.shuffle(hc3)
    rows += hc3[:hc3_limit]
    rng.shuffle(rows)
    return rows


class Classifier:
    def __init__(self, path="models/orai-cls", device="cuda"):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval().half()
        self.device = device

    @torch.no_grad()
    def prob_ai(self, texts: list[str], batch_size=32) -> list[float]:
        out = []
        for k in range(0, len(texts), batch_size):
            enc = self.tok(texts[k:k + batch_size], truncation=True, max_length=512, padding=True,
                           return_tensors="pt").to(self.device)
            out.extend(torch.softmax(self.model(**enc).logits.float(), -1)[:, 1].tolist())
        return out


def train(args):
    rows = load_training_texts(args.pairs, args.hc3, args.hc3_limit, args.seed)
    print("train rows", len(rows), "ai ratio", sum(y for _, y in rows) / len(rows))
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=2).cuda()

    def collate(batch):
        enc = tok([t for t, _ in batch], truncation=True, max_length=384, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([y for _, y in batch])
        return enc

    dl = DataLoader(rows, batch_size=args.bs, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(dl) * args.epochs
    sch = get_cosine_schedule_with_warmup(opt, int(0.06 * steps), steps)
    model.train()
    step = 0
    for _ in range(args.epochs):
        for batch in dl:
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
            step += 1
            if step % 50 == 0:
                print(f"step {step}/{steps} loss {loss.item():.4f}", flush=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--pairs", default="data/pairs/train.jsonl")
    ap.add_argument("--hc3", default="data/raw/hc3_all.jsonl")
    ap.add_argument("--hc3-limit", type=int, default=12000)
    ap.add_argument("--base", default="models/chinese-roberta-wwm-ext")
    ap.add_argument("--out", default="models/orai-cls")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=0)
    train(ap.parse_args())
