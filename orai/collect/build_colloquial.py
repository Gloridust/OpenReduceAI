"""从 CLUECorpusSmall（2019 年以前）抽口语化真人段落：知乎问答（webtext2019zh）和点评/商品评论（comments2019）。

新闻、百科、广告、公告按关键词排除；剩下的按「口语标记密度」排序取前若干条。
输出 data/human/colloquial.jsonl，字段与 paragraphs.jsonl 一致（domain = qa / review，风格都是「口语」）。

python orai/collect/build_colloquial.py --zhihu 3000 --review 1500
"""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from build_human import CJK, cjk_ratio, normalize_punct

COLLOQUIAL = re.compile(
    r"我|你|俺|咱|吧|呢|啊|呀|嘛|哈|啦|哦|噢|嗯|呗|咋|挺|蛮|超级|贼|特别|觉得|感觉|其实|反正|说实话|"
    r"真的|就是|然后|还是|有点|好像|估计|不过|毕竟|当时|结果|居然|竟然|简直|算了|哎|唉|。。|～|！")
NEWS = re.compile(r"记者|讯|日电|据了解|新华社|报道|通讯员|本报|发布会|记者从|获悉|原标题|责任编辑")
WIKI = re.compile(r"^[^，。]{1,15}(是|，)(一|位于|中国|我国|指|由)|位于|海拔|主演|出品|别名|学名|属于.{0,4}(科|属)")
NSFW = re.compile(r"性交|做爱|强奸|性欲|性幻想|自慰|手淫|裸照|嫖|约炮|色情|黄片|阴茎|阴道|乳房|射精|处女|破处|性侵|自杀|自残")
ADS = re.compile(r"预订|热线|电话|微信|QQ|qq|http|www|\.com|优惠|促销|包邮|加盟|代理|报名|咨询|喊单|招聘|免费领|点击|二维码|【|】|#")


def windows(text: str, lo: int, hi: int):
    """把长回答切成以句号等结尾、长度在 [lo, hi] 的连续句子窗口。"""
    sents = re.split(r"(?<=[。！？])", text)
    buf = ""
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if len(buf) + len(s) > hi:
            if len(buf) >= lo:
                yield buf
            buf = s if len(s) <= hi else ""
        else:
            buf += s
    if len(buf) >= lo:
        yield buf


def score(t: str) -> float:
    return len(COLLOQUIAL.findall(t)) * 100 / max(len(t), 1)


def good(t: str) -> bool:
    if cjk_ratio(t) < 0.7 or NEWS.search(t) or ADS.search(t) or WIKI.search(t) or NSFW.search(t):
        return False
    if re.search(r"(.{3,})\1\1|�|&#|\d{6,}", t):
        return False
    return bool(re.search(rf"[{CJK}。！？…）”]$", t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw/clue")
    ap.add_argument("--zhihu", type=int, default=3000)
    ap.add_argument("--review", type=int, default=1500)
    ap.add_argument("--min-score", type=float, default=4.0, help="每百字口语标记数下限")
    ap.add_argument("--out", default="data/human/colloquial.jsonl")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    zhihu, review, seen = [], [], set()
    for p in sorted(Path(args.src).glob("chunk_*.txt")):
        lines = p.read_text(encoding="utf-8", errors="ignore").split("\n")[1:-1]  # 首尾行是被截断的
        for raw in lines:
            line = normalize_punct(raw)
            q = line.find("？")
            if 0 < q < 80 and len(line) > 300:
                # 知乎：标题？描述＋回答。跳过标题，从回答部分切窗口
                for w in windows(line[q + 1:], 150, 450):
                    if good(w) and score(w) >= args.min_score:
                        zhihu.append(w)
            elif 120 <= len(line) <= 450 and good(line) and score(line) >= args.min_score + 1:
                review.append(line)

    rng = random.Random(args.seed)
    out = []
    for name, pool, n in (("clue_zhihu", zhihu, args.zhihu), ("clue_review", review, args.review)):
        rng.shuffle(pool)
        pool.sort(key=score, reverse=True)
        picked = 0
        for t in pool:
            key = hashlib.md5(t[:60].encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            out.append({"id": f"{name}-{key[:10]}", "source": name,
                        "domain": "qa" if name == "clue_zhihu" else "review", "topic": "", "text": t})
            picked += 1
            if picked >= n:
                break
        print(name, "候选", len(pool), "选中", picked)
    rng.shuffle(out)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", len(out), "->", args.out)


if __name__ == "__main__":
    main()
