"""评测。生成和打分分两步跑，因为 4B 改写模型和裁判团同时放不进 12 GB 显存。

生成：python -m orai.eval.run_eval gen --adapter outputs/sft-v1/final --name sft-v1 --n 4
打分：python -m orai.eval.run_eval score --names sft-v1 [dpo-v1 ...]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

EVAL = "data/pairs/eval.jsonl"
OUT = Path("outputs/eval")


def load_eval(limit=None):
    rows = [json.loads(l) for l in open(EVAL, encoding="utf-8")]
    return rows[:limit] if limit else rows


def cmd_gen(args):
    from orai.infer.generate import Rewriter
    from orai.train.format import STYLE_BY_DOMAIN

    rows = load_eval(args.limit)
    rw = Rewriter(args.adapter, quant=args.quant)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{args.name}.cands.jsonl", "w", encoding="utf-8") as f:
        for style in ("书面", "口语"):
            sub = [r for r in rows if STYLE_BY_DOMAIN[r["domain"]] == style]
            cands = rw.generate([r["ai"] for r in sub], style=style, n=args.n, temperature=args.temperature,
                                batch_size=args.batch_size)
            for r, c in zip(sub, cands):
                f.write(json.dumps({"id": r["id"], "cands": c}, ensure_ascii=False) + "\n")
    print("wrote", OUT / f"{args.name}.cands.jsonl")


def summarize(name, dets, sems, th):
    def mean(k, xs):
        v = [x[k] for x in xs if x[k] == x[k]]
        return statistics.mean(v) if v else float("nan")
    s = {"system": name, "n": len(dets)}
    for k in ("fdg", "cls", "ext", "bino", "lp_std", "phrase", "sent_cv"):
        s[k] = mean(k, dets)
    s["flag_fdg"] = statistics.mean(d["fdg"] > th["fdg"] for d in dets)
    s["flag_cls"] = statistics.mean(d["cls"] > 0.5 for d in dets)
    s["flag_ext"] = statistics.mean(d["ext"] > 0.5 for d in dets)
    s["flag_bino"] = statistics.mean(d["bino"] < th["bino"] for d in dets)
    s["cheat"] = statistics.mean(bool(d["cheat"]) for d in dets)
    if sems:
        for k in ("sim", "nli_fwd", "nli_bwd", "fact_recall", "len_ratio"):
            s[k] = mean(k, sems)
    return s


def best_threshold(h, a, higher_is_ai=True):
    """在「真人 vs AI 输入」上取使准确率最高的阈值，作为该检测器的判 AI 线。"""
    best, bt = -1, 0.0
    for t in sorted(h + a):
        if higher_is_ai:
            acc = sum(x <= t for x in h) + sum(x > t for x in a)
        else:
            acc = sum(x >= t for x in h) + sum(x < t for x in a)
        if acc > best:
            best, bt = acc, t
    return bt, best / (len(h) + len(a))


def cmd_score(args):
    from orai.infer.judge import Judge

    rows = load_eval(args.limit)
    by_id = {r["id"]: r for r in rows}
    judge = Judge(with_classifier=True, with_binoculars=True, with_external=True)
    det_h = judge.detect([r["human"] for r in rows])
    det_a = judge.detect([r["ai"] for r in rows])
    th = {}
    th["fdg"], acc_f = best_threshold([d["fdg"] for d in det_h], [d["fdg"] for d in det_a], True)
    th["bino"], acc_b = best_threshold([d["bino"] for d in det_h], [d["bino"] for d in det_a], False)
    print(f"阈值 fdg={th['fdg']:.3f}(acc {acc_f:.3f}) bino={th['bino']:.4f}(acc {acc_b:.3f})")

    table = [summarize("真人原文", det_h, judge.sem.score([r["ai"] for r in rows], [r["human"] for r in rows]), th),
             summarize("AI 输入", det_a, None, th)]
    for name in args.names:
        cands = {json.loads(l)["id"]: json.loads(l)["cands"] for l in open(OUT / f"{name}.cands.jsonl")}
        ids = [r["id"] for r in rows if r["id"] in cands]
        first = [cands[i][0] for i in ids]
        srcs = [by_id[i]["ai"] for i in ids]
        table.append(summarize(f"{name} 单次采样", judge.detect(first), judge.sem.score(srcs, first), th))
        best = []
        for i in ids:
            ranked = judge.rank(by_id[i]["ai"], cands[i])
            best.append(ranked[0]["text"])
        table.append(summarize(f"{name} 多采样重排", judge.detect(best), judge.sem.score(srcs, best), th))
        with open(OUT / f"{name}.best.jsonl", "w", encoding="utf-8") as f:
            for i, b in zip(ids, best):
                f.write(json.dumps({"id": i, "ai": by_id[i]["ai"], "out": b, "human": by_id[i]["human"]},
                                   ensure_ascii=False) + "\n")

    report = {"thresholds": th, "threshold_acc": {"fdg": acc_f, "bino": acc_b}, "table": table}
    (OUT / f"report-{'-'.join(args.names) or 'base'}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    cols = ["system", "flag_ext", "flag_bino", "flag_fdg", "flag_cls", "ext", "bino", "fdg", "cls", "lp_std", "phrase",
            "sim", "nli_fwd", "fact_recall", "len_ratio"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for s in table:
        print("| " + " | ".join(s[c] if isinstance(s.get(c), str) else f"{s.get(c, float('nan')):.3f}" for c in cols) + " |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gen")
    g.add_argument("--adapter", required=True)
    g.add_argument("--name", required=True)
    g.add_argument("--n", type=int, default=4)
    g.add_argument("--temperature", type=float, default=0.9)
    g.add_argument("--limit", type=int)
    g.add_argument("--quant", default="16bit")
    g.add_argument("--batch-size", type=int, default=2)
    s = sp.add_parser("score")
    s.add_argument("--names", nargs="*", default=[])
    s.add_argument("--limit", type=int)
    a = ap.parse_args()
    {"gen": cmd_gen, "score": cmd_score}[a.cmd](a)
