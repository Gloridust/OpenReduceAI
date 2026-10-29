# OpenReduceAI

在单张 12 GB 显卡上微调的中文「去 AI 腔」改写小模型。基座 Qwen3.5-4B-Base，bf16 LoRA。

核心做法：拿 ChatGPT 之前（2022-11 以前）的真人中文文本当目标，先让大模型把它「AI 润色」一遍，
再训练小模型把 AI 腔还原成真人写法。之后用本地检测器打分做迭代 DPO。设计细节见 [docs/PLAN.md](docs/PLAN.md)，
实测结果见 [docs/RESULTS.md](docs/RESULTS.md)。

> 定位是写作辅助：去掉套话、让文字更像人写的。请遵守你所在学校、期刊、平台对 AI 使用的规定。

## 目录

```
orai/collect/   真人语料抽取（CSL、HC3-Chinese 人类回答、THUCNews），统一标点
orai/aiify/     「AI 润色」批次划分；由 Claude 子 agent 逐条改写（AGENT_PROMPT.md），check_batch.py 校验
orai/filter/    语义相似度、NLI、硬信息保留、作弊字符检查，组成 data/pairs/{train,eval}.jsonl
orai/detect/    裁判团：Fast-DetectGPT、Binoculars、有监督分类器、套话密度、句长起伏
orai/train/     sft.py（LoRA SFT）、dpo_iter.py（merge → sample → score → train）
orai/infer/     generate.py（采样）、judge.py（候选重排）、cli.py（txt / docx 文稿改写）
orai/eval/      run_eval.py（生成、打分、出表）
```

## 环境

```bash
~/miniconda3/envs/torch/bin/python -m venv .venv
.venv/bin/pip install uv
UV_INDEX_URL=https://pypi.org/simple .venv/bin/uv pip install --python .venv/bin/python \
  unsloth trl peft datasets jieba datasketch sentence-transformers pyyaml python-docx pyarrow
```

模型放在 `models/`（不入库）：`Qwen3.5-4B-Base`、`Qwen3.5-0.8B-Base`、`Qwen3.5-0.8B`、
`chinese-roberta-wwm-ext`、`Erlangshen-Roberta-110M-NLI`、`bge-base-zh-v1.5`。

## 流程

```bash
python orai/collect/build_human.py               # 真人段落池
python orai/aiify/make_batches.py                # 分批，交给子 agent 生成 data/aiify/out/
.venv/bin/python -m orai.filter.build_pairs      # 组对 + 过滤
.venv/bin/python -m orai.detect.classifier train # 训练检测分类器 D2
.venv/bin/python -m orai.train.sft --out outputs/sft-v1
.venv/bin/python -m orai.train.dpo_iter merge  --adapter outputs/sft-v1/final --out models/orai-sft-v1-merged
.venv/bin/python -m orai.train.dpo_iter sample --model models/orai-sft-v1-merged --out data/dpo/r1.cands.jsonl
.venv/bin/python -m orai.train.dpo_iter score  --cands data/dpo/r1.cands.jsonl --out data/dpo/r1.pairs.jsonl
.venv/bin/python -m orai.train.dpo_iter train  --model models/orai-sft-v1-merged --pairs data/dpo/r1.pairs.jsonl --out outputs/dpo-r1
.venv/bin/python -m orai.eval.run_eval gen --adapter outputs/dpo-r1/final --name dpo-r1
.venv/bin/python -m orai.eval.run_eval score --names sft-v1 dpo-r1
```

改写文稿：

```bash
.venv/bin/python -m orai.infer.cli 论文.docx --adapter outputs/dpo-r1/final --style 书面
```

显存紧张时先停掉占显存的常驻服务；训练建议放进限内存的 scope：
`systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0 .venv/bin/python -m orai.train.sft`。

## 数据与许可

CSL、THUCNews、HC3-Chinese 等语料只用于研究。原始语料和生成数据都不入库、不分发。
