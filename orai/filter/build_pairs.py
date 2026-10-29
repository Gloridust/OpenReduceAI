"""合并子 agent 的 AI 版本和真人原文，打语义分并过滤，写出 data/pairs/{train,eval}.jsonl。

python -m orai.filter.build_pairs [--no-filter]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from orai.detect.stats import cheat_flags

# 批次号 → 生成它的模型（见 orai/aiify/README.md）
def gen_model(batch_name: str) -> str:
    split, k = batch_name.split("_")
    if split == "eval":
        return "claude-opus"
    k = int(k)
    if k >= 40:  # 口语批次（CLUE 知乎/点评）：40-51 sonnet，52-63 haiku
        return "claude-sonnet" if k < 52 else "claude-haiku"
    if k >= 24:  # 追加批次：24-31 sonnet，32-39 haiku
        return "claude-sonnet" if k < 32 else "claude-haiku"
    return "claude-sonnet" if k < 8 else "claude-haiku" if k < 16 else "claude-fable"


def load_outputs(outdir: Path) -> dict[str, str]:
    got = {}
    for p in sorted(outdir.glob("part_*.jsonl")):
        for line in open(p, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                got[d["id"]] = d["ai_text"].strip()
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="data/aiify/batches")
    ap.add_argument("--outs", default="data/aiify/out")
    ap.add_argument("--human", nargs="+", default=["data/human/paragraphs.jsonl", "data/human/colloquial.jsonl"])
    ap.add_argument("--dst", default="data/pairs")
    ap.add_argument("--min-sim", type=float, default=0.80)
    ap.add_argument("--min-fact", type=float, default=0.8)
    ap.add_argument("--no-filter", action="store_true")
    args = ap.parse_args()

    meta = {}
    for hp in args.human:
        for l in open(hp, encoding="utf-8"):
            d = json.loads(l)
            meta[d["id"]] = d
    # 子 agent 有意删改过内容的条目（见各批次回报），不进训练
    ex_path = Path("data/aiify/exclude_ids.txt")
    exclude = set(ex_path.read_text().split()) if ex_path.exists() else set()
    rows = []
    for bf in sorted(Path(args.batches).glob("*.jsonl")):
        od = Path(args.outs) / bf.stem
        if not od.exists():
            continue
        got = load_outputs(od)
        for line in open(bf, encoding="utf-8"):
            b = json.loads(line)
            if b["id"] not in got or b["id"] in exclude:
                continue
            m = meta[b["id"]]
            rows.append({"id": b["id"], "split": bf.stem.split("_")[0], "batch": bf.stem,
                         "gen_model": gen_model(bf.stem), "source": m["source"], "domain": m["domain"],
                         "op": b["op"], "human": b["text"], "ai": got[b["id"]]})
    print("collected", len(rows))

    from orai.filter.semantic import SemanticScorer
    sc = SemanticScorer()
    metrics = sc.score([r["human"] for r in rows], [r["ai"] for r in rows])
    kept = {"train": [], "eval": []}
    drop = {}
    for r, mt in zip(rows, metrics):
        r["meta"] = mt
        reason = None
        if cheat_flags(r["ai"], r["human"]):
            reason = "cheat_flags"
        elif mt["sim"] < args.min_sim:
            reason = "low_sim"
        elif mt["fact_recall"] < args.min_fact:
            reason = "fact_lost"
        elif not 0.6 <= mt["len_ratio"] <= 1.6:
            reason = "length"
        if reason and not args.no_filter and r["split"] == "train":
            drop[reason] = drop.get(reason, 0) + 1
            continue
        kept[r["split"]].append(r)
    Path(args.dst).mkdir(parents=True, exist_ok=True)
    for split, rs in kept.items():
        with open(Path(args.dst) / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("kept", {k: len(v) for k, v in kept.items()}, "dropped", drop)


if __name__ == "__main__":
    main()
