from __future__ import annotations

from pathlib import Path
from typing import Any

import json


def load_jsonl_dataset(train_file: str, eval_file: str | None = None):
    from datasets import load_dataset

    data_files: dict[str, str] = {"train": train_file}
    if eval_file:
        data_files["eval"] = eval_file
    return load_dataset("json", data_files=data_files)


def compact_message_chars(row: dict[str, Any]) -> int:
    return sum(len(message.get("content") or "") for message in row["messages"])


def has_user_and_assistant(row: dict[str, Any]) -> bool:
    messages = row.get("messages") or []
    has_user = any(message.get("role") == "user" and message.get("content") for message in messages)
    has_assistant = any(
        message.get("role") == "assistant" and message.get("content") for message in messages
    )
    return has_user and has_assistant


def take_short_conversations(split, size: int, max_chars: int):
    from datasets import Dataset

    rows: list[dict[str, Any]] = []
    for row in split:
        row = dict(row)
        if has_user_and_assistant(row) and compact_message_chars(row) <= max_chars:
            rows.append({"messages": row["messages"]})
        if len(rows) >= size:
            break
    if len(rows) < size:
        raise RuntimeError(f"Only found {len(rows)} usable rows, requested {size}")
    return Dataset.from_list(rows)


def load_ultrachat_sft_dataset(data_config: dict[str, Any]):
    from datasets import load_dataset

    dataset = load_dataset(data_config["path"])
    max_chars = int(data_config.get("max_chars", 1800))
    train = take_short_conversations(dataset["train_sft"], int(data_config["train_size"]), max_chars)
    eval_ds = take_short_conversations(dataset["test_sft"], int(data_config["eval_size"]), max_chars)
    return {"train": train, "eval": eval_ds}


def first_turn(row: dict[str, Any]) -> tuple[str, str]:
    user = ""
    assistant = ""
    for message in row.get("messages") or []:
        if not user and message.get("role") == "user":
            user = message.get("content") or ""
        if not assistant and message.get("role") == "assistant":
            assistant = message.get("content") or ""
    return user, assistant


def load_ultrachat_dpo_dataset(data_config: dict[str, Any]):
    from datasets import Dataset, load_dataset

    dataset = load_dataset(data_config["path"])
    max_chars = int(data_config.get("max_chars", 1800))
    rejected_text = data_config.get(
        "rejected_text",
        "I do not have enough information to answer that well.",
    )

    def convert(split, size: int):
        rows: list[dict[str, Any]] = []
        for row in split:
            row = dict(row)
            if compact_message_chars(row) > max_chars:
                continue
            user, assistant = first_turn(row)
            if not user or not assistant:
                continue
            rows.append(
                {
                    "prompt": [{"role": "user", "content": user}],
                    "chosen": [{"role": "assistant", "content": assistant}],
                    "rejected": [{"role": "assistant", "content": rejected_text}],
                }
            )
            if len(rows) >= size:
                break
        if len(rows) < size:
            raise RuntimeError(f"Only found {len(rows)} usable DPO rows, requested {size}")
        return Dataset.from_list(rows)

    return {
        "train": convert(dataset["train_sft"], int(data_config["train_size"])),
        "eval": convert(dataset["test_sft"], int(data_config["eval_size"])),
    }


def load_training_dataset(config: dict[str, Any]):
    source = config.get("source", "jsonl")
    if source == "jsonl":
        return load_jsonl_dataset(config["train_file"], config.get("eval_file"))
    if source == "ultrachat_local":
        return load_ultrachat_sft_dataset(config)
    if source == "ultrachat_dpo":
        return load_ultrachat_dpo_dataset(config)
    raise ValueError(f"Unsupported data source: {source}")


def load_jsonl_rows(path: str, limit: int = 20) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def validate_sft_row(row: dict[str, Any]) -> None:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("SFT rows require a non-empty messages list")
    for message in messages:
        if message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported role: {message.get('role')}")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError("Each message requires non-empty content")


def validate_dpo_row(row: dict[str, Any]) -> None:
    prompt = row.get("prompt")
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if not isinstance(prompt, list) or not isinstance(chosen, list) or not isinstance(rejected, list):
        raise ValueError("DPO rows require prompt, chosen, and rejected message lists")
    for message_list in [prompt, chosen, rejected]:
        for message in message_list:
            if message.get("role") not in {"system", "user", "assistant"}:
                raise ValueError(f"Unsupported role: {message.get('role')}")


def validate_rows(rows, track: str) -> None:
    validator = validate_dpo_row if track == "dpo" else validate_sft_row
    for idx, row in enumerate(rows):
        try:
            validator(dict(row))
        except Exception as exc:
            raise ValueError(f"Invalid row at index {idx}: {exc}") from exc


def validate_dataset(dataset, track: str, limit: int = 20) -> None:
    validate_rows(dataset.select(range(min(len(dataset), limit))), track=track)


def require_file(path: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(path)
