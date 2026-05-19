# 学习笔记

这份笔记记录本项目中用到的微调概念和工程取舍。内容来自公开资料、开源文档和实际实验整理，重点放在“能复现、能解释、能对比”的实践层面。

## 微调路线

- **SFT**：使用指令和回答样本继续训练模型，是建立指令跟随能力和任务格式适配能力的常见基线。
- **LoRA**：冻结大部分基础模型参数，只训练低秩 adapter，适合低成本迭代和多版本实验。
- **QLoRA**：在 LoRA 基础上，用 4-bit 方式加载基础模型，进一步降低显存压力。
- **DPO**：使用 chosen/rejected 偏好样本直接优化模型偏好，不需要单独训练奖励模型。

## 实验中更容易踩坑的点

- 数据列名和消息格式会影响 TRL 是否把数据识别成 conversational dataset。
- chat template 不只是推理格式，也会影响训练时哪些 token 被计入 loss。
- `assistant_only_loss` 需要训练模板显式标记 assistant 生成区域，否则 TRL 无法构造正确的 loss mask。
- 显存占用不只来自模型参数，还来自 activation、gradient、optimizer state、序列长度和 batch 策略。
- 小规模实验适合验证链路和比较资源，不适合直接宣称模型能力大幅提升。

## 当前默认选择

- 使用 Qwen3-0.6B 级别模型做主实验，降低排错成本。
- 优先使用 bf16、gradient checkpointing 和 gradient accumulation。
- adapter 先单独保存，只有在推理或交付需要时再考虑合并。
- 训练日志和配置必须保留，实验结论要能回到原始 run 复查。

## DPO 数据构造说明

当前 DPO 短训练使用的是流程验证数据：从 SFT 数据里取 prompt 和原始 assistant 回答作为 chosen，再构造一个简短、信息不足的 rejected。这个做法的价值是验证 DPO 训练链路、日志和指标，不适合作为真实偏好对齐效果的最终证明。

后续如果要提升 DPO 结论可信度，需要换成真实 chosen/rejected 偏好数据，并补充人工样例检查。
