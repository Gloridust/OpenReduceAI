"""给真人文本打「像人程度」，用于筛训练目标。

不依赖单个检测器：综合 Fast-DetectGPT（D1，0.8B 打分）、AI 套话密度、口语标记密度三项，取排名平均。
评测用的检测器（Binoculars、第三方 zhv3）刻意不参与，避免自己给自己打分。

python -m orai.filter.human_likeness data/human/colloquial.jsonl            # 写 <文件>.hl.json
python -m orai.filter.human_likeness data/pairs/train.jsonl --field human
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from orai.detect.stats import ai_phrase_density

COLLOQUIAL = re.compile(
    r"我|你|俺|咱|吧|呢|啊|呀|嘛|哈|啦|哦|噢|嗯|呗|咋|挺|蛮|超级|贼|特别|觉得|感觉|其实|反正|说实话|"
    r"真的|就是|然后|还是|有点|好像|估计|不过|毕竟|当时|结果|居然|竟然|简直|算了|哎|唉|！")


def colloquial_density(t: str) -> float:
    return len(COLLOQUIAL.findall(t)) * 100 / max(len(t), 1)


def ranks(xs: list[float]) -> list[float]:
    """0 最好（最像人）到 1 最差。"""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    for k, i in enumerate(order):
        r[i] = k / max(len(xs) - 1, 1)
    return r


def score_texts(texts: list[str]) -> list[dict]:
    from orai.detect.lm import LMDetector

    lm = LMDetector(instruct=None).score(texts)
    fdg = [s.fast_detectgpt for s in lm]
    phrase = [ai_phrase_density(t) for t in texts]
    coll = [colloquial_density(t) for t in texts]
    r_fdg, r_ph, r_co = ranks(fdg), ranks(phrase), ranks([-c for c in coll])
    return [{"fdg": f, "lp_std": s.lp_std, "phrase": p, "colloq": c, "ai_rank": (a + b + d) / 3}
            for f, s, p, c, a, b, d in zip(fdg, lm, phrase, coll, r_fdg, r_ph, r_co)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--field", default="text")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.path, encoding="utf-8")]
    sc = score_texts([r[args.field] for r in rows])
    out = Path(args.path).with_suffix(".hl.json")
    out.write_text(json.dumps({r["id"]: s for r, s in zip(rows, sc)}, ensure_ascii=False))
    print("wrote", out, len(sc))


if __name__ == "__main__":
    main()
