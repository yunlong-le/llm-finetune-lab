# 实验报告

本目录保存训练结果摘要、TensorBoard 标量转出的图像，以及后续的生成样例对比。

当前已有图像：

- `figures/smoke_train_loss.png`：SFT、LoRA、QLoRA 短训练损失对比
- `figures/lora_120_loss.png`：LoRA 120 步训练和评估损失曲线
- `figures/assistant_only_train_loss.png`：LoRA 与 QLoRA 仅助手回答损失对比
- `figures/dpo_reward_metrics.png`：DPO reward margin 与 reward accuracy
- `figures/resource_comparison.png`：四条路线显存和耗时对比
- `figures/existing_sft_loss.png`：已有 1000 步 SFT 日志曲线
- `generation_compare_20260519.md`：固定 prompt 生成样例对比
- `run_summary.csv` / `run_summary_table.md`：由远程 `summary.json` 自动汇总得到的运行表
- `runs_v2/*/summary.json`：从远程训练目录抽取的原始运行摘要，只包含指标和路径，不包含模型权重
