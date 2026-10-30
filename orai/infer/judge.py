"""裁判团：给候选打「像 AI 的程度」和「意思是否保住」，并据此挑最优候选。

奖励检测器（D1 Fast-DetectGPT、D2 分类器）参与挑选；D3 Binoculars 只用于评测，不在这里用。
"""

from __future__ import annotations

from orai.detect.stats import ai_phrase_density, cheat_flags, sentence_len_cv


class Judge:
    def __init__(self, with_classifier=True, with_binoculars=False, with_external=False):
        from orai.detect.lm import LMDetector
        from orai.filter.semantic import SemanticScorer

        self.lm = LMDetector(instruct="models/Qwen3.5-0.8B-text" if with_binoculars else None)
        self.sem = SemanticScorer()
        self.cls = None
        if with_classifier:
            from orai.detect.classifier import Classifier
            self.cls = Classifier()
        self.ext = None
        if with_external:  # 第三方检测器只用于评测
            from orai.detect.third_party import ExternalDetector
            self.ext = ExternalDetector()

    def detect(self, texts: list[str]) -> list[dict]:
        lm = self.lm.score(texts)
        nan = [float("nan")] * len(texts)
        cls = self.cls.prob_ai(texts) if self.cls else nan
        ext = self.ext.prob_ai(texts) if self.ext else nan
        return [{"fdg": s.fast_detectgpt, "bino": s.binoculars, "lp_mean": s.lp_mean, "lp_std": s.lp_std,
                 "cls": c, "ext": e, "phrase": ai_phrase_density(t), "sent_cv": sentence_len_cv(t),
                 "cheat": cheat_flags(t)} for s, c, e, t in zip(lm, cls, ext, texts)]

    @staticmethod
    def ai_score(d: dict) -> float:
        """越低越像人。Fast-DetectGPT 分数大致在 [-2, 6]，压到 0-1 后与分类器概率平均。"""
        fdg = min(max((d["fdg"] + 1) / 6, 0.0), 1.0)
        return 0.5 * fdg + 0.5 * d["cls"] + 0.05 * d["phrase"]

    @staticmethod
    def passes(m: dict, min_sim=0.82, min_fact=0.9) -> bool:
        return (m["sim"] >= min_sim and m["fact_recall"] >= min_fact and 0.7 <= m["len_ratio"] <= 1.4
                and m["nli_fwd"] >= 0.3 and m.get("added", 0) == 0)

    def rank(self, src: str, cands: list[str]) -> list[dict]:
        """按「先过语义门槛，再比 AI 分」排序；返回带分数的候选列表，最优在前。"""
        det = self.detect(cands)
        sem = self.sem.score([src] * len(cands), cands)
        rows = []
        for c, d, m in zip(cands, det, sem):
            d["cheat"] = cheat_flags(c, src)
            ok = self.passes(m) and not d["cheat"]
            rows.append({"text": c, "det": d, "sem": m, "ok": ok, "ai_score": self.ai_score(d)})
        rows.sort(key=lambda r: (not r["ok"], r["ai_score"]))
        return rows
