from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return data


def command_env(_: argparse.Namespace) -> None:
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["cuda_device"] = torch.cuda.get_device_name(0)
            report["cuda_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3,
                2,
            )
    except Exception as exc:
        report["torch_error"] = str(exc)

    for package_name in ["transformers", "datasets", "trl", "peft", "bitsandbytes", "accelerate"]:
        try:
            module = __import__(package_name)
            report[package_name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            report[f"{package_name}_error"] = str(exc)

    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_train(args: argparse.Namespace) -> None:
    from src.training import run_training

    config = load_config(args.config)
    run_training(config=config, dry_run=args.dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-finetune-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    env_parser = subparsers.add_parser("env", help="print environment information")
    env_parser.set_defaults(func=command_env)

    train_parser = subparsers.add_parser("train", help="run an experiment")
    train_parser.add_argument("--config", required=True, help="path to a YAML config")
    train_parser.add_argument("--dry-run", action="store_true", help="validate config without loading the model")
    train_parser.set_defaults(func=command_train)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
