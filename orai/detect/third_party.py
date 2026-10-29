"""第三方中文 AIGC 检测器（D4，只用于评测，不参与训练和候选挑选）。

yuchuantian/AIGC_detector_zhv3（BERT 二分类，Apache-2.0），label 1 = AI（在本项目评测集上实测核对过）。
"""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ExternalDetector:
    def __init__(self, path="models/AIGC_detector_zhv3", device="cuda"):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval().half()
        self.device = device

    @torch.no_grad()
    def prob_ai(self, texts: list[str], batch_size=32) -> list[float]:
        out = []
        for k in range(0, len(texts), batch_size):
            enc = self.tok(texts[k:k + batch_size], truncation=True, max_length=512, padding=True,
                           return_tensors="pt").to(self.device)
            out.extend(torch.softmax(self.model(**enc).logits.float(), -1)[:, 1].tolist())
        return out
