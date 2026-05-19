# llm-finetune-lab

这是一个围绕小规模开源大模型微调实践整理的实验仓库。项目把 SFT、LoRA、QLoRA、DPO 放到同一套训练入口、配置格式、日志记录和报告生成流程里，重点观察它们在单卡资源约束下的工程差异。

仓库保留了从训练入口、数据转换、日志记录到结果复盘的完整实验脉络。后续继续扩展样本、步数或模型时，可以沿着同一套记录方式追加实验。

## 项目关注点

大模型微调看起来常常只是几行 trainer 代码，但真正跑起来后，很多问题都藏在细节里。本仓库主要关注这些问题：

- SFT、LoRA、QLoRA、DPO 能否使用统一入口和配置组织起来。
- LoRA 与 QLoRA 在小模型场景下的显存、速度、loss 曲线差异是否明显。
- DPO 如何从已有 SFT checkpoint 出发，完成偏好数据训练链路。
- chat template 是否支持只在 assistant 回答部分计算 loss。
- 数据被截断后是否仍保留可训练的 assistant token。
- TensorBoard 日志、结构化 summary、图表和文字复盘如何形成一个可持续追加的实验记录。
- 短训练结果应该如何谨慎解释，避免把流程验证夸大成模型能力提升。

## 技术栈选择

主线采用 `TRL + PEFT`。这样 SFT 和 DPO 可以分别使用 `trl.SFTTrainer`、`trl.DPOTrainer`，LoRA/QLoRA 则统一由 `peft.LoraConfig` 注入 adapter。

显存优化采用 `bitsandbytes` 4-bit NF4、bf16、gradient accumulation 和 gradient checkpointing。`Accelerate` 作为 GPU 启动入口，避免把设备管理散落在训练脚本里。

Unsloth 保留为可选配置，用于后续做速度和显存对照。它没有进入主依赖路径，因为我希望基础实验先依赖更通用的 Hugging Face 生态，降低复现门槛。

| 路线 | 核心实现 | 主要用途 | 当前进度 |
| --- | --- | --- | --- |
| SFT | `trl.SFTTrainer` | 建立监督微调基线，验证数据和模板链路 | 已完成短训练 |
| LoRA SFT | `SFTTrainer + peft.LoraConfig` | 训练 adapter，观察低成本微调效果 | 已完成 40 步与 120 步训练 |
| QLoRA SFT | `PEFT + bitsandbytes 4-bit NF4` | 4-bit 加载 base model 后训练 adapter | 已完成短训练 |
| DPO | `trl.DPOTrainer` | 从 SFT checkpoint 出发验证偏好优化 | 已完成短训练 |
| Unsloth SFT | `Unsloth + PEFT` | 作为后续加速对照 | 配置已预留 |

## 代码结构

训练入口集中在 `src/lab.py`。它负责读取 YAML 配置、执行 dry run、打印环境信息，并把 SFT/DPO 分发到不同训练函数。

数据处理集中在 `src/data.py`。目前支持两种数据来源：仓库内置的小型 JSONL 样例，以及本地已有的多轮对话数据副本。SFT 类数据会被整理成标准 `messages` 结构；DPO 数据会被整理成 `prompt / chosen / rejected` 结构。

训练逻辑集中在 `src/training.py`。SFT、LoRA、QLoRA 复用同一个训练函数，通过配置决定是否启用 adapter、是否使用 4-bit 加载、是否只计算 assistant 回答 loss。DPO 单独使用偏好训练入口，但共享模型路径、日志路径和训练参数组织方式。

```text
configs/        SFT、LoRA、QLoRA、DPO、Unsloth 的实验配置
data/           小样例数据和数据格式示例
docs/           公开学习笔记与实验复盘
reports/        训练曲线、资源对比和结果摘要
scripts/        文本检查、链接检查、图表生成和同步脚本
src/            数据处理、训练入口和公共训练逻辑
```

## 关键实现细节

### 1. 统一配置

不同路线都使用 YAML 描述实验参数，训练入口只关心配置里的 `task`、`model`、`dataset`、`training`、`peft` 等字段。这样新增一次 LoRA 或 QLoRA 实验时，不需要改训练代码，只需要复制并调整配置。

### 2. LoRA 与 QLoRA 的可比性

LoRA 和 QLoRA 使用相同的 adapter 配置，包括 `r`、`alpha`、`dropout` 和 `target_modules`。这样两者差异主要来自 base model 的加载方式，而不是因为 adapter 结构不同导致比较失真。

### 3. assistant-only loss

普通多轮对话样本里既有用户输入，也有助手回答。如果 loss 计算到了用户问题上，训练目标就不够干净。这个仓库在 Qwen3 类模型上注入 TRL 内置训练模板，使 `assistant_only_loss` 可以正确识别助手回答区域。

第一次开启该设置时，评估 loss 出现过 `nan`。排查后发现部分样本过长，被截断后没有留下可计算的 assistant token。后续通过过滤较短对话恢复正常。这也是我把数据过滤逻辑纳入公开训练入口的原因。

### 4. 日志和图表

每个 GPU run 都会保留结构化 summary，并从 TensorBoard scalar 中生成图片。README 中展示的是训练损失、DPO reward 指标、资源占用和已有 SFT 长训练日志。图表不是装饰，而是用来回答两个问题：训练有没有真的跑起来，以及不同路线的资源取舍是否清楚。

## 第一轮实验结果

第一批 GPU 实验在单张 RTX 5090 上完成。实验规模刻意控制得较小，目标是验证链路、记录资源和观察趋势，而不是把短训练包装成完整模型能力评测。

| 实验 | 步数 | 数据规模 | 耗时 | 峰值显存 | 最后记录 |
| --- | ---: | --- | ---: | ---: | --- |
| SFT smoke | 30 | 192 train / 24 eval | 40.7s | 6463 MiB | train loss 1.614 |
| LoRA smoke | 40 | 256 train / 32 eval | 63.3s | 3031 MiB | train loss 1.710 |
| QLoRA smoke | 40 | 256 train / 32 eval | 75.8s | 3060 MiB | train loss 1.838 |
| DPO smoke | 30 | 128 train / 16 eval | 49.5s | 11225 MiB | train loss 0.641 |
| LoRA 120 steps | 120 | 1024 train / 64 eval | 183.7s | 3031 MiB | eval loss 1.680 |
| LoRA assistant-only | 40 | 256 train / 32 eval | 59.4s | 2135 MiB | eval loss 1.838 |
| QLoRA assistant-only | 40 | 256 train / 32 eval | 75.6s | 2135 MiB | eval loss 1.964 |
| LoRA assistant-only 120 | 120 | 1024 train / 64 eval | 179.9s | 2203 MiB | eval loss 1.621 |

从资源上看，LoRA 和 QLoRA 都能把小规模实验的峰值显存控制在约 3 GiB，adapter 产物约 58 MiB，明显小于完整 checkpoint。SFT 和 DPO 的显存占用更高，其中 DPO 需要同时处理 policy/reference 相关计算，短训练下峰值显存达到 11225 MiB。

从速度上看，QLoRA 在这组 0.6B 模型实验中没有比 LoRA 更快。4-bit 加载降低了 base model 显存压力，但量化相关开销也会影响吞吐。因此我的判断是：QLoRA 更适合模型规模或 batch size 已经明显压迫显存的场景；在小模型短训练里，它的优势不一定体现在速度上。

## TensorBoard 图表

<p align="center">
  <strong>四条路线训练损失对比</strong><br>
  <img src="reports/figures/smoke_train_loss.png" width="760" alt="SFT、LoRA、QLoRA、DPO 训练损失对比">
</p>

<p align="center">
  <strong>LoRA 120 步训练曲线</strong><br>
  <img src="reports/figures/lora_120_loss.png" width="760" alt="LoRA 120步损失曲线">
</p>

<p align="center">
  <strong>LoRA / QLoRA 仅助手回答损失</strong><br>
  <img src="reports/figures/assistant_only_train_loss.png" width="760" alt="LoRA 和 QLoRA 仅助手回答损失对比">
</p>

<p align="center">
  <strong>DPO 偏好指标</strong><br>
  <img src="reports/figures/dpo_reward_metrics.png" width="760" alt="DPO reward 指标">
</p>

<p align="center">
  <strong>资源占用对比</strong><br>
  <img src="reports/figures/resource_comparison.png" width="760" alt="四条路线资源占用对比">
</p>

<p align="center">
  <strong>已有 SFT 1000 步日志</strong><br>
  <img src="reports/figures/existing_sft_loss.png" width="760" alt="已有 SFT 1000步训练日志">
</p>

## 生成行为对比

除了 loss 和显存，我还用固定中文 prompt 做了定性对比。解码方式为 greedy generation，`max_new_tokens=160`。这组样例只用于观察行为变化，不作为严格评测结论。

| Prompt | 基础模型 | 已有 SFT checkpoint | 本轮 LoRA / QLoRA / DPO 观察 |
| --- | --- | --- | --- |
| 请用两句话解释 LoRA 的作用。 | 出现复读，并夹杂异常片段。 | 能解释 LoRA 是低秩矩阵微调方法，并提到减少资源消耗。 | LoRA assistant-only 能给出更简短且基本正确的解释；QLoRA 40 步仍有复读；DPO 接近已有 SFT。 |
| 请严格输出 JSON，抽取一句话的关键词。 | 复读输入，并夹杂异常片段。 | 基本复述输入，没有稳定满足 JSON 约束。 | LoRA 仍不稳定；QLoRA 出现接近 JSON 的关键词列表但前缀不干净；DPO 没有稳定解决结构化输出。 |
| 我今天学习状态不好，请用温和但不夸张的语气鼓励我。 | 复读输入，并夹杂 assistant 标记。 | 能给出正常、温和的鼓励。 | LoRA 能回应但有重复倾向；QLoRA 有特殊标记残留；DPO 输出接近已有 SFT。 |

这个对比很有用：base model 和 SFT checkpoint 的差距在自然中文问答上比较明显；本轮 LoRA 短训练已经能在概念解释上看到一些改善，但 120 步仍不足以稳定处理 JSON 这类强格式输出；QLoRA 路线跑通了，但短训练下生成质量波动更大；DPO 在当前小规模构造偏好数据上更像流程验证，而不是稳定的偏好对齐结论。

## 结果复盘

LoRA 的价值在这组实验里比较清楚：它能以较低显存和较小产物体积完成可训练 adapter，适合快速迭代数据清洗、模板设计和超参数试验。

QLoRA 的价值不是“必然更快”，而是在显存更紧张时保留训练可能性。对于 0.6B 这类小模型，4-bit 的收益没有充分释放，反而会被额外开销抵消一部分。

DPO 的关键不只是 trainer 能跑起来，而是偏好数据是否可信。本仓库当前 DPO 数据更偏流程验证，所以只记录 reward 指标和链路现象，不把它写成模型已经完成偏好对齐。

assistant-only loss 是我认为最值得保留的工程细节之一。它让训练目标从“复现整段对话文本”变成“学习助手应该如何回答”，也暴露了样本截断和模板标记这两个很实际的问题。

已有 1000 步 SFT 日志说明更长训练确实能形成更平滑的 loss 下降轨迹。本仓库第一轮新增实验更偏横向比较，后续如果扩展，应该优先在同一数据切分下增加 LoRA/QLoRA 步数，再做更系统的生成评测。

## 复现方式

安装依赖：

```bash
python -m pip install -r requirements.txt
```

本地轻量检查：

```bash
python scripts/check_public_text.py
python scripts/check_links.py
python -m src.lab env
python -m src.lab train --config configs/sft.yaml --dry-run
python -m src.lab train --config configs/lora.yaml --dry-run
python -m src.lab train --config configs/dpo.yaml --dry-run
```

GPU 训练示例：

```bash
accelerate launch -m src.lab train --config configs/lora.yaml
accelerate launch -m src.lab train --config configs/qlora.yaml
accelerate launch -m src.lab train --config configs/dpo.yaml
```

重新生成报告图表：

```bash
python scripts/generate_figures.py
python scripts/summarize_runs.py --runs-dir reports/runs_v2 --out-dir reports
```

## 当前边界

这不是一个追求大规模榜单成绩的仓库。第一轮实验的样本数和训练步数都很小，更适合用来说明工程流程、资源差异和常见问题。

生成样例是定性 sanity check，不是严格 benchmark。结构化输出、多轮稳定性和事实性还需要更系统的评测集。

短训练中的 loss 下降只能说明训练链路有效，不能直接等价为真实任务质量提升。尤其是 DPO，偏好数据规模和质量会决定最终效果。

本仓库更适合作为一个持续扩展的微调实验底座：先把路线跑通、把日志留清楚，再逐步扩大样本、训练步数和评测维度。
