# Chat Template 训练笔记

TRL 的 `assistant_only_loss` 依赖 chat template 中的 `{% generation %}` 标记。这个标记告诉 trainer 哪一段 token 属于 assistant 回答，从而只在回答部分计算 loss。

本项目里遇到的问题是：模型自带的 Qwen3 chat template 可以用于推理，但缺少训练标记。直接启用 `assistant_only_loss` 会报错。

处理方式：

1. 训练时从 TRL 读取内置的 `qwen3_training_chat_template`。
2. 将它注入到 tokenizer 的 `chat_template`。
3. 对训练和评估样本做长度过滤，避免样本被截断后没有 assistant token 可用于计算 loss。

这个问题单独记录下来，是因为它很容易被误判成模型或显存问题。实际上它属于“数据格式和训练模板没有对齐”。
