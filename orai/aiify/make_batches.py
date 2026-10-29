"""把真人段落分成批次，每条随机分配一种「AI 润色」操作，交给子 agent 生成 AI 版本。

输出：data/aiify/batches/<split>_<NNN>.jsonl，每行 {id, op, instruction, text}
"""

import argparse
import json
import random
from pathlib import Path

# 真实用户最常对 AI 说的几类话。domain 不同，措辞略有区别。
OPS = {
    "polish_formal": (0.25, {
        "academic": "请润色这段文字，使其更正式、规范，更符合学术论文的表达。",
        "qa": "请润色这段回答，使其更专业、规范。",
        "news": "请润色这段文字，使其更正式、规范。",
        "review": "请润色这段评价，使其表达更规范、更有条理。",
        "*": "请润色这段文字，使其更正式、规范。",
    }),
    "polish_fluent": (0.25, {
        "*": "请优化这段文字的逻辑和表达，让它更流畅、更有条理。",
    }),
    "rewrite": (0.2, {
        "*": "请用你自己的话把这段文字重新表述一遍。",
    }),
    "expand": (0.15, {
        "*": "请适当扩写这段文字，让论述更完整、更有说服力。",
    }),
    "from_points": (0.15, {
        "*": "请先提炼这段文字的要点，再根据要点重新写成一段完整、连贯的话（只输出最后这段话）。",
    }),
}


def pick_op(rng: random.Random):
    r, acc = rng.random(), 0.0
    for name, (w, _) in OPS.items():
        acc += w
        if r <= acc:
            return name
    return "rewrite"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/human/paragraphs.jsonl")
    ap.add_argument("--outdir", default="data/aiify/batches")
    ap.add_argument("--train-batches", type=int, default=24)
    ap.add_argument("--eval-batches", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--eval-batch-size", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--offset", type=int, default=0, help="从段落池第几条开始取（追加批次时用）")
    ap.add_argument("--start-k", type=int, default=0, help="追加批次的起始编号")
    ap.add_argument("--eval-start-k", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in open(args.src, encoding="utf-8")]
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    i = args.offset
    plan = [("eval", args.eval_start_k + k, args.eval_batch_size) for k in range(args.eval_batches)]
    plan += [("train", args.start_k + k, args.batch_size) for k in range(args.train_batches)]
    for split, k, size in plan:
        chunk = rows[i:i + size]
        i += size
        with open(out / f"{split}_{k:03d}.jsonl", "w", encoding="utf-8") as f:
            for r in chunk:
                op = pick_op(rng)
                inst = OPS[op][1].get(r["domain"], OPS[op][1].get("*")) or OPS[op][1]["*"]
                f.write(json.dumps({"id": r["id"], "op": op, "instruction": inst, "text": r["text"]},
                                   ensure_ascii=False) + "\n")
    print(f"used {i} paragraphs, {len(plan)} batches -> {out}")


if __name__ == "__main__":
    main()
