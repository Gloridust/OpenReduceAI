"""训练和推理共用的提示格式。两边必须完全一致。"""

SYSTEM = "把下面这段文字改写成人自然写出来的样子。意思、数字、术语和引用都不能变，不要加原文没有的内容。风格：{style}"

# 训练集里，学术、新闻的原文偏书面，问答偏口语，所以按领域给风格标签；推理时用户可以自选
STYLE_BY_DOMAIN = {"academic": "书面", "news": "书面", "qa": "口语", "review": "口语"}
END = "<|im_end|>"


def build_prompt(ai_text: str, style: str = "书面") -> str:
    return (f"<|im_start|>system\n{SYSTEM.format(style=style)}<|im_end|>\n"
            f"<|im_start|>user\n{ai_text.strip()}<|im_end|>\n"
            f"<|im_start|>assistant\n")
