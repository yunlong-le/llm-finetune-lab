# 库选型说明

本项目采用当前开源生态中常见的微调组合，但每个库只承担相对明确的职责，避免把实验写成几个互不相关的脚本。

## TRL

TRL 负责高层训练器。SFT 路线使用 `SFTTrainer`，DPO 路线使用 `DPOTrainer`。这样可以把注意力集中在数据格式、配置和实验对比上，而不是重复实现训练循环。

## PEFT

PEFT 负责参数高效微调。LoRA 和 QLoRA 都通过 `LoraConfig` 明确 adapter 的 rank、alpha、dropout 和 target modules，便于复查 trainable parameter 的范围。

## bitsandbytes

bitsandbytes 只用于 QLoRA 路线。实验中使用 4-bit NF4 加载基础模型，再叠加 LoRA adapter 训练，用来观察显存和速度的实际变化。

## Accelerate

Accelerate 作为启动和设备管理层。当前目标是单卡可复现，但保留 `accelerate launch` 这种启动方式，后续扩展到更复杂环境时不需要重写入口。

## Unsloth

Unsloth 暂时作为可选对照路线。它适合后续比较速度、显存和工程复杂度，但主线仍然保持在 TRL + PEFT，方便读者直接理解训练过程。
