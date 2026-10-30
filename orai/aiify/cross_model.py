"""换一家模型重写评测集的 AI 版本，检验换了 AI 腔之后效果还在不在。

评测集原本的 AI 版本全是 Claude 写的。这里对同一批真人原文、同一条 instruction，改由别家模型
（默认 DeepSeek，经 pi 调用，密钥由 pi 自己管理）来写，输出和 data/pairs/eval.jsonl 同结构的文件，
只替换 ai 字段。可以断点续跑。

python -m orai.aiify.cross_model --out data/pairs/eval_deepseek.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SYSTEM = "只输出处理后的正文，不要开场白、标题、分点或说明。"


def call_pi(model: str, prompt: str) -> str:
    provider, name = model.split("/", 1)
    r = subprocess.run(["pi", "-p", "--no-session", "--no-tools", "--no-extensions", "--no-skills",
                        "--provider", provider, "--model", name, "--system-prompt", SYSTEM, prompt],
                       capture_output=True, text=True, timeout=600, cwd="/tmp")
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(r.stderr[-500:])
    return r.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/pairs/eval.jsonl")
    ap.add_argument("--batches", default="data/aiify/batches")
    ap.add_argument("--out", default="data/pairs/eval_deepseek.jsonl")
    ap.add_argument("--model", default="deepseek/deepseek-flash:off")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    inst = {}
    for p in Path(args.batches).glob("eval_*.jsonl"):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            inst[r["id"]] = r["instruction"]
    rows = [json.loads(l) for l in open(args.src, encoding="utf-8")]
    out = Path(args.out)
    done = {json.loads(l)["id"] for l in open(out, encoding="utf-8")} if out.exists() else set()
    todo = [r for r in rows if r["id"] not in done]

    def work(r):
        for _ in range(3):
            try:
                return r, call_pi(args.model, f"{inst[r['id']]}\n\n{r['human']}")
            except Exception as e:  # 网络抖动重试，三次都失败就跳过，下次续跑再补
                err = e
        print("skip", r["id"], err, flush=True)
        return r, None

    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as ex:
        for k, (r, text) in enumerate(ex.map(work, todo)):
            if text:
                f.write(json.dumps({**r, "ai": text, "gen_model": args.model}, ensure_ascii=False) + "\n")
                f.flush()
            print(k, len(text or ""), flush=True)


if __name__ == "__main__":
    main()
