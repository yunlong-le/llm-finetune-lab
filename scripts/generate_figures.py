from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / ".local" / "remote_scalars.json"
OUT = ROOT / "reports" / "figures"

FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
]


RUN_LABELS = {
    "sft_20260519_103927": "SFT 30步",
    "lora_20260519_102936": "LoRA 40步",
    "qlora_20260519_103425": "QLoRA 40步",
    "dpo_20260519_103752": "DPO 30步",
    "lora_20260519_104246": "LoRA 120步",
    "lora_20260519_112459": "LoRA 助手损失 40步",
    "qlora_20260519_112634": "QLoRA 助手损失 40步",
    "lora_20260519_112838": "LoRA 助手损失 120步",
}


def load_data():
    try:
        text = DATA.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = DATA.read_text(encoding="utf-16")
    return json.loads(text)


def setup_font():
    for font_path in FONT_CANDIDATES:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            prop = font_manager.FontProperties(fname=str(font_path))
            plt.rcParams["font.family"] = prop.get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def series(payload, run_id: str, tag: str):
    for key, tags in payload["scalars"].items():
        if run_id in key and tag in tags:
            values = tags[tag]
            return [v["step"] for v in values], [v["value"] for v in values]
    return [], []


def save_current(name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def plot_smoke_loss(payload):
    plt.figure(figsize=(9.5, 5.2))
    for run_id in ["sft_20260519_103927", "lora_20260519_102936", "qlora_20260519_103425"]:
        steps, values = series(payload, run_id, "train/loss")
        if steps:
            plt.plot(steps, values, marker="o", linewidth=2, label=RUN_LABELS[run_id])
    plt.title("SFT / LoRA / QLoRA 训练损失对比")
    plt.xlabel("训练步数")
    plt.ylabel("训练损失")
    plt.grid(alpha=0.28)
    plt.legend()
    return save_current("smoke_train_loss.png")


def plot_lora_120(payload):
    run_id = "lora_20260519_112838"
    plt.figure(figsize=(9.5, 5.2))
    steps, values = series(payload, run_id, "train/loss")
    if steps:
        plt.plot(steps, values, marker="o", linewidth=2, label="训练损失")
    eval_steps, eval_values = series(payload, run_id, "eval/loss")
    if eval_steps:
        plt.plot(eval_steps, eval_values, marker="s", linewidth=2.5, label="评估损失")
    plt.title("LoRA 120步仅助手回答损失曲线")
    plt.xlabel("训练步数")
    plt.ylabel("损失")
    plt.grid(alpha=0.28)
    plt.legend()
    return save_current("lora_120_loss.png")


def plot_dpo_rewards(payload):
    run_id = "dpo_20260519_103752"
    plt.figure(figsize=(9.5, 5.2))
    for tag, label in [
        ("train/rewards/margins", "训练 reward margin"),
        ("eval/rewards/margins", "评估 reward margin"),
        ("train/rewards/accuracies", "训练 reward accuracy"),
        ("eval/rewards/accuracies", "评估 reward accuracy"),
    ]:
        steps, values = series(payload, run_id, tag)
        if steps:
            plt.plot(steps, values, marker="o", linewidth=2, label=label)
    plt.title("DPO 偏好指标")
    plt.xlabel("训练步数")
    plt.ylabel("指标值")
    plt.grid(alpha=0.28)
    plt.legend()
    return save_current("dpo_reward_metrics.png")


def plot_memory_runtime(payload):
    summaries = payload["summaries"]
    run_ids = ["sft_20260519_103927", "lora_20260519_112459", "qlora_20260519_112634", "dpo_20260519_103752"]
    labels = ["SFT", "LoRA助手损失", "QLoRA助手损失", "DPO"]
    memory = [summaries[r]["peak_memory_mb"] / 1024 for r in run_ids]
    runtime = [summaries[r]["train_result"]["train_runtime"] for r in run_ids]

    fig, ax1 = plt.subplots(figsize=(9.5, 5.2))
    ax2 = ax1.twinx()
    x = range(len(run_ids))
    ax1.bar([i - 0.18 for i in x], memory, width=0.36, label="峰值显存 GiB", color="#4C78A8")
    ax2.bar([i + 0.18 for i in x], runtime, width=0.36, label="训练耗时 秒", color="#F58518")
    ax1.set_xticks(list(x), labels)
    ax1.set_ylabel("峰值显存 (GiB)")
    ax2.set_ylabel("训练耗时 (秒)")
    ax1.set_title("四条路线资源占用对比")
    ax1.grid(axis="y", alpha=0.25)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    return save_current("resource_comparison.png")


def plot_assistant_loss(payload):
    plt.figure(figsize=(9.5, 5.2))
    for run_id in ["lora_20260519_112459", "qlora_20260519_112634"]:
        steps, values = series(payload, run_id, "train/loss")
        if steps:
            plt.plot(steps, values, marker="o", linewidth=2, label=RUN_LABELS[run_id])
    plt.title("LoRA / QLoRA 仅助手回答损失对比")
    plt.xlabel("训练步数")
    plt.ylabel("训练损失")
    plt.grid(alpha=0.28)
    plt.legend()
    return save_current("assistant_only_train_loss.png")


def plot_old_sft(payload):
    plt.figure(figsize=(9.5, 5.2))
    for key, tags in payload["scalars"].items():
        if key.endswith("logs/Qwen3-0.6B-SFT") and "Loss" in tags:
            vals = tags["Loss"]
            steps = [v["step"] for v in vals]
            values = [v["value"] for v in vals]
            plt.plot(steps, values, marker="o", linewidth=1.8, alpha=0.85)
    plt.title("已有 SFT 1000步训练日志")
    plt.xlabel("训练步数")
    plt.ylabel("损失")
    plt.grid(alpha=0.28)
    return save_current("existing_sft_loss.png")


def main():
    setup_font()
    payload = load_data()
    paths = [
        plot_smoke_loss(payload),
        plot_lora_120(payload),
        plot_dpo_rewards(payload),
        plot_memory_runtime(payload),
        plot_assistant_loss(payload),
        plot_old_sft(payload),
    ]
    for path in paths:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
