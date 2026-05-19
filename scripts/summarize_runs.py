from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def collect_summaries(root: Path) -> list[dict]:
    summaries: list[dict] = []
    for path in sorted(root.glob("*/summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        result = data.get("train_result", {})
        summaries.append(
            {
                "run_id": data["run_id"],
                "track": data["track"],
                "steps": data["max_steps"],
                "train_size": data["train_size"],
                "eval_size": data["eval_size"],
                "runtime_sec": round(float(result.get("train_runtime", data["train_runtime_sec"])), 1),
                "peak_memory_mib": data.get("peak_memory_mb", ""),
                "train_loss": round(float(result.get("train_loss", 0.0)), 3),
                "assistant_only_loss": data.get("assistant_only_loss", False),
            }
        )
    return summaries


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 训练运行汇总",
        "",
        "| Run ID | 路线 | 步数 | 训练样本 | 评估样本 | 耗时(s) | 峰值显存(MiB) | 训练损失 | 仅助手回答损失 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {run_id} | {track} | {steps} | {train_size} | {eval_size} | {runtime_sec} | "
            "{peak_memory_mib} | {train_loss} | {assistant_only_loss} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs_v2", help="directory containing run subdirectories")
    parser.add_argument("--out-dir", default="reports", help="output directory")
    args = parser.parse_args()

    rows = collect_summaries(Path(args.runs_dir))
    if not rows:
        raise SystemExit(f"No summary.json files found under {args.runs_dir}")
    out_dir = Path(args.out_dir)
    write_csv(rows, out_dir / "run_summary.csv")
    write_markdown(rows, out_dir / "run_summary_table.md")
    print(f"wrote {len(rows)} rows to {out_dir}")


if __name__ == "__main__":
    main()
