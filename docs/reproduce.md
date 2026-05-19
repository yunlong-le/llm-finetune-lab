# 复现说明

本项目分为本地轻量检查和 GPU 训练两类复现方式。

## 本地轻量检查

本地不需要 GPU，主要用于检查配置、样例数据、公开文档和图表生成脚本。

```bash
python -m pip install -r requirements.txt
python scripts/check_public_text.py
python scripts/check_links.py
python -m src.lab train --config configs/lora.yaml --dry-run
python -m src.lab train --config configs/dpo.yaml --dry-run
```

## GPU 训练

GPU 环境需要提前准备模型和数据。以本项目的远程实验为例，模型和数据路径如下：

```text
/root/autodl-tmp/llm-finetune-lab/model/Qwen3-0.6B-Base
/root/autodl-tmp/llm-finetune-lab/data/ultrachat_200k
/root/autodl-tmp/llm-finetune-lab/finetuned/Qwen3-0.6B-SFT
```

LoRA 仅助手回答损失实验：

```bash
python -m src.lab train --config configs/lora_ultrachat_assistant.yaml
```

QLoRA 仅助手回答损失实验：

```bash
python -m src.lab train --config configs/qlora_ultrachat_assistant.yaml
```

DPO 链路验证实验：

```bash
python -m src.lab train --config configs/dpo_ultrachat.yaml
```

## 图表生成

如果已经有 TensorBoard 标量导出文件，可以重新生成图表：

```bash
python scripts/generate_figures.py
```

如果已经有多个 `summary.json`，可以重新生成运行汇总：

```bash
python scripts/summarize_runs.py --runs-dir reports/runs_v2 --out-dir reports
```

## 注意事项

- `assistant_only_loss` 依赖训练兼容的 chat template，本项目在训练时注入 TRL 内置 Qwen3 训练模板。
- UltraChat 样本会先按字符长度过滤，避免截断后没有 assistant token 可计算 loss。
- DPO 当前配置是链路验证，不代表完整偏好对齐实验。
