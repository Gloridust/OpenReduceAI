"""按「像人程度」筛训练目标，并调整来源配比，让训练目标整体偏口语。

每个来源只保留 ai_rank 最低（最像人）的一部分：书面语料（学术摘要、新闻）保留得少，口语语料几乎全留。
ai_rank 来自 orai/filter/human_likeness.py（Fast-DetectGPT + 套话密度 + 口语标记，排名平均）。

python -m orai.filter.select_targets
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

KEEP = {"csl": 0.4, "thucnews": 0.5, "hc3_human": 0.7, "clue_zhihu": 1.0, "clue_review": 1.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/pairs/train.jsonl")
    ap.add_argument("--scores", nargs="+", default=["data/pairs/train.hl.json", "data/human/colloquial.hl.json"])
    ap.add_argument("--out", default="data/pairs/train_sel.jsonl")
    args = ap.parse_args()

    hl = {}
    for p in args.scores:
        hl.update(json.loads(Path(p).read_text()))
    rows = [json.loads(l) for l in open(args.src, encoding="utf-8")]
    by = collections.defaultdict(list)
    for r in rows:
        if r["id"] in hl:
            r["hl"] = hl[r["id"]]
            by[r["source"]].append(r)
    out = []
    for src, rs in by.items():
        rs.sort(key=lambda r: r["hl"]["ai_rank"])
        k = round(len(rs) * KEEP.get(src, 0.5))
        out += rs[:k]
        print(f"{src}: {len(rs)} -> {k}")
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("total", len(out), "->", args.out)


if __name__ == "__main__":
    main()
