from __future__ import annotations

from typing import Any

from src.data import load_jsonl_rows, load_training_dataset, require_file, validate_dataset, validate_rows


def run_training(config: dict[str, Any], dry_run: bool = False) -> None:
    track = config["experiment"]["track"]
    data_config = config["data"]

    data_source = data_config.get("source", "jsonl")
    if data_source == "jsonl":
        require_file(data_config["train_file"])
        if data_config.get("eval_file"):
            require_file(data_config["eval_file"])

    if dry_run:
        if data_source == "jsonl":
            validate_rows(load_jsonl_rows(data_config["train_file"]), track=track)
            if data_config.get("eval_file"):
                validate_rows(load_jsonl_rows(data_config["eval_file"]), track=track)
        else:
            print(f"dry run skipped local data load for source={data_source}")
        print(f"dry run ok: {config['experiment']['name']} ({track})")
        return

    dataset = load_training_dataset(data_config)
    validate_dataset(dataset["train"], track=track)
    if "eval" in dataset:
        validate_dataset(dataset["eval"], track=track)

    if track in {"sft", "lora", "qlora", "unsloth_sft"}:
        run_sft_like(config, dataset)
    elif track == "dpo":
        run_dpo(config, dataset)
    else:
        raise ValueError(f"Unsupported track: {track}")


def run_sft_like(config: dict[str, Any], dataset) -> None:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    model_config = config["model"]
    train_config = config["training"]
    quant_config = config.get("quantization")

    tokenizer = AutoTokenizer.from_pretrained(model_config["name_or_path"], trust_remote_code=True)
    assistant_only_loss = bool(train_config.get("assistant_only_loss", False))
    if assistant_only_loss:
        from trl.chat_template_utils import qwen3_training_chat_template

        tokenizer.chat_template = qwen3_training_chat_template
    model_kwargs: dict[str, Any] = {"trust_remote_code": True}

    if quant_config and quant_config.get("load_in_4bit"):
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_config.get("quant_type", "nf4"),
            bnb_4bit_use_double_quant=bool(quant_config.get("double_quant", False)),
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(model_config["name_or_path"], **model_kwargs)

    if quant_config and quant_config.get("load_in_4bit"):
        model = prepare_model_for_kbit_training(model)

    if config["experiment"]["track"] in {"lora", "qlora"}:
        lora_config = config["lora"]
        model = get_peft_model(
            model,
            LoraConfig(
                r=int(lora_config["r"]),
                lora_alpha=int(lora_config["alpha"]),
                lora_dropout=float(lora_config["dropout"]),
                bias="none",
                target_modules=list(lora_config["target_modules"]),
                task_type="CAUSAL_LM",
            ),
        )

    args = SFTConfig(
        output_dir=train_config["output_dir"],
        per_device_train_batch_size=int(train_config["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(train_config["gradient_accumulation_steps"]),
        learning_rate=float(train_config["learning_rate"]),
        max_steps=int(train_config["max_steps"]),
        warmup_ratio=float(train_config["warmup_ratio"]),
        logging_steps=int(train_config["logging_steps"]),
        eval_steps=int(train_config["eval_steps"]),
        save_steps=int(train_config["save_steps"]),
        bf16=bool(train_config.get("bf16", False)),
        gradient_checkpointing=bool(train_config.get("gradient_checkpointing", False)),
        max_length=int(config["data"]["max_length"]),
        assistant_only_loss=assistant_only_loss,
        report_to=["tensorboard"],
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("eval"),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(train_config["output_dir"])


def run_dpo(config: dict[str, Any], dataset) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    model_config = config["model"]
    train_config = config["training"]

    tokenizer = AutoTokenizer.from_pretrained(model_config["name_or_path"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_config["name_or_path"], trust_remote_code=True)
    ref_model = AutoModelForCausalLM.from_pretrained(model_config["name_or_path"], trust_remote_code=True)

    args = DPOConfig(
        output_dir=train_config["output_dir"],
        per_device_train_batch_size=int(train_config["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(train_config["gradient_accumulation_steps"]),
        learning_rate=float(train_config["learning_rate"]),
        max_steps=int(train_config["max_steps"]),
        warmup_ratio=float(train_config["warmup_ratio"]),
        logging_steps=int(train_config["logging_steps"]),
        eval_steps=int(train_config["eval_steps"]),
        save_steps=int(train_config["save_steps"]),
        bf16=bool(train_config.get("bf16", False)),
        gradient_checkpointing=bool(train_config.get("gradient_checkpointing", False)),
        beta=float(config["dpo"]["beta"]),
        report_to=["tensorboard"],
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("eval"),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(train_config["output_dir"])
