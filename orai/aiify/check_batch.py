"""检查子 agent 的输出是否完整：python3 orai/aiify/check_batch.py <batch.jsonl> <输出目录>

输出目录里可以有多个 part_*.jsonl，每行 {"id": ..., "ai_text": ...}。
打印缺失的 id、坏行、长度异常的条目；全部通过时打印 OK 并以 0 退出。
"""

import json
import sys
from pathlib import Path


def main():
    batch, outdir = Path(sys.argv[1]), Path(sys.argv[2])
    want = {}
    for line in open(batch, encoding="utf-8"):
        d = json.loads(line)
        want[d["id"]] = d["text"]
    got, bad = {}, []
    for p in sorted(outdir.glob("part_*.jsonl")):
        for n, line in enumerate(open(p, encoding="utf-8"), 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                got[d["id"]] = d["ai_text"]
            except Exception as e:  # noqa: BLE001
                bad.append(f"{p.name}:{n} {e}")
    missing = [k for k in want if k not in got or not str(got[k]).strip()]
    extra = [k for k in got if k not in want]
    weird = []
    for k, v in got.items():
        if k in want:
            ratio = len(v) / max(len(want[k]), 1)
            if ratio < 0.6 or ratio > 1.6:
                weird.append(f"{k} 长度比 {ratio:.2f}")
    for name, items in (("坏行", bad), ("缺失", missing), ("多余 id", extra), ("长度异常", weird)):
        if items:
            print(f"{name} {len(items)}:", *items[:30], sep="\n  ")
    if bad or missing or extra or weird:
        sys.exit(1)
    print(f"OK {len(got)}/{len(want)}")


if __name__ == "__main__":
    main()
