"""从 2022-11 之前的公开语料里抽真人段落，统一标点，写成 data/human/paragraphs.jsonl。

语料：CSL（学术摘要，2010-2020）、HC3-Chinese 的 human_answers（问答，口语）、THUCNews（新闻，2005-2011）。
"""

import argparse
import gzip
import hashlib
import json
import random
import re
from pathlib import Path

CJK = r"一-鿿"
RAW = Path("data/raw")


def normalize_punct(text: str) -> str:
    """CSL 等老语料常用半角标点，统一成中文全角，避免模型学到「半角标点」这种取巧特征。"""
    t = text.replace("\r", "").replace("　", " ")
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(rf"(?<=[{CJK}”）》」])\s*,\s*", "，", t)
    t = re.sub(rf"\s*,\s*(?=[{CJK}“（《「])", "，", t)
    t = re.sub(rf"(?<=[{CJK}”）》」])\s*\.(?!\d)\s*", "。", t)
    t = re.sub(rf"(?<=[{CJK}”）》」])\s*;\s*", "；", t)
    t = re.sub(rf"(?<=[{CJK}”）》」])\s*:\s*", "：", t)
    t = re.sub(rf"(?<=[{CJK}”）》」])\s*\?\s*", "？", t)
    t = re.sub(rf"(?<=[{CJK}”）》」])\s*!\s*", "！", t)
    t = re.sub(rf"\(\s*(?=[{CJK}])", "（", t)
    t = re.sub(rf"(?<=[{CJK}])\s*\)", "）", t)
    # 成对的半角双引号换成中文引号
    parts = t.split('"')
    if len(parts) % 2 == 1 and len(parts) > 1:
        t = "".join(p + ("“" if i % 2 == 0 else "”") for i, p in enumerate(parts[:-1])) + parts[-1]
    t = re.sub(rf"(?<=[{CJK}，。；：？！“”（）])\s+(?=[{CJK}，。；：？！“”（）])", "", t)
    return t


def cjk_ratio(t: str) -> float:
    return len(re.findall(rf"[{CJK}]", t)) / max(len(t), 1)


def ok(t: str, lo: int, hi: int) -> bool:
    if not (lo <= len(t) <= hi):
        return False
    if cjk_ratio(t) < 0.6:
        return False
    if re.search(r"http|www\.|@|\[\d+\]|&[a-z]+;|�", t):
        return False
    if not re.search(r"[。！？]$", t):
        return False
    return True


def load_csl(n: int, rng: random.Random):
    rows = []
    with gzip.open(RAW / "csl.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            t = normalize_punct(d["abstract"])
            if ok(t, 150, 520):
                rows.append({"source": "csl", "domain": "academic", "topic": d.get("discipline", ""), "text": t})
    rng.shuffle(rows)
    return rows[:n]


def load_hc3(n: int, rng: random.Random):
    rows = []
    with open(RAW / "hc3_all.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            for a in d.get("human_answers") or []:
                if not isinstance(a, str):
                    continue
                t = normalize_punct(a)
                if ok(t, 120, 520):
                    rows.append({"source": "hc3_human", "domain": "qa", "topic": d.get("source", ""), "text": t})
    rng.shuffle(rows)
    return rows[:n]


def load_thucnews(n: int, rng: random.Random):
    import pyarrow as pa

    with pa.memory_map(str(RAW / "thucnews_train.arrow")) as src:
        try:
            table = pa.ipc.open_stream(src).read_all()
        except pa.ArrowInvalid:
            table = pa.ipc.open_file(src).read_all()
    col = "text" if "text" in table.column_names else table.column_names[0]
    rows = []
    for doc in table.column(col).to_pylist():
        # 新闻按段切，再把相邻短段合并到 150-500 字
        buf = ""
        for para in re.split(r"[\n\r]+|\s{2,}", doc):
            para = normalize_punct(para)
            if not para:
                continue
            buf = para if not buf else buf + para
            if len(buf) >= 150:
                if ok(buf, 150, 520):
                    rows.append({"source": "thucnews", "domain": "news", "topic": "", "text": buf})
                buf = ""
    rng.shuffle(rows)
    return rows[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csl", type=int, default=7000)
    ap.add_argument("--hc3", type=int, default=3000)
    ap.add_argument("--news", type=int, default=2000)
    ap.add_argument("--out", default="data/human/paragraphs.jsonl")
    ap.add_argument("--seed", type=int, default=20261029)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = load_csl(args.csl, rng) + load_hc3(args.hc3, rng) + load_thucnews(args.news, rng)
    seen, out = set(), []
    for r in rows:
        key = hashlib.md5(r["text"][:60].encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        r["id"] = f"{r['source']}-{key[:10]}"
        out.append(r)
    rng.shuffle(out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by = {}
    for r in out:
        by[r["source"]] = by.get(r["source"], 0) + 1
    print(f"wrote {len(out)} paragraphs -> {args.out}", by)


if __name__ == "__main__":
    main()
