from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

from stemtune.benchmark_mcqa import parse_seeds
from stemtune.smoke_mcqa import (
    DEFAULT_DATASET_ID,
    DEFAULT_MODEL_ID,
    evaluate_prompts,
    load_examples,
    load_generator,
    make_plain_prompt,
    make_support_prompt,
    pick_device,
    summarize,
)


BUDGET_LABELS = {
    "plain": "Question only",
    "support_24": "24-word support",
    "support_48": "48-word support",
    "support_full": "Full support",
}
BUDGET_ORDER = ["plain", "support_24", "support_48", "support_full"]
BUDGET_COLORS = {
    "plain": "#1d4ed8",
    "support_24": "#0ea5e9",
    "support_48": "#14b8a6",
    "support_full": "#0f766e",
}


def truncate_words(text: str, limit: int) -> str:
    words = text.split()
    return " ".join(words[:limit])


def prompts_for_budget(examples, budget_name: str) -> list[str]:
    if budget_name == "plain":
        return [make_plain_prompt(example) for example in examples]
    if budget_name == "support_full":
        return [make_support_prompt(example, example.support) for example in examples]

    word_limit = int(budget_name.split("_")[1])
    return [make_support_prompt(example, truncate_words(example.support, word_limit)) for example in examples]


def save_predictions_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "seed",
        "condition",
        "example_id",
        "question",
        "correct_letter",
        "prediction",
        "is_correct",
        "is_valid",
        "latency_s",
        "raw_output",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def save_report_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# MCQA Support Budget Study",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- Dataset: `{payload['dataset_id']}`",
        f"- Device: `{payload['device']}`",
        f"- Seeds: `{', '.join(str(seed) for seed in payload['seeds'])}`",
        f"- Examples per seed: `{payload['limit']}`",
        "",
        "## Aggregate Results",
        "",
        "| Condition | Mean Accuracy | Mean Latency (s) | Gain vs Plain |",
        "|---|---:|---:|---:|",
    ]
    plain_accuracy = payload["aggregate"]["plain"]["accuracy_mean"]
    for condition in BUDGET_ORDER:
        summary = payload["aggregate"][condition]
        lines.append(
            f"| {BUDGET_LABELS[condition]} | {summary['accuracy_mean']:.3f} | {summary['avg_latency_s_mean']:.3f} | {summary['accuracy_mean'] - plain_accuracy:+.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)


def save_plot(path: Path, payload: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("#f8fafc")
    labels = [BUDGET_LABELS[item] for item in BUDGET_ORDER]
    colors = [BUDGET_COLORS[item] for item in BUDGET_ORDER]

    accuracies = [payload["aggregate"][item]["accuracy_mean"] for item in BUDGET_ORDER]
    acc_errors = [payload["aggregate"][item]["accuracy_stdev"] for item in BUDGET_ORDER]
    latencies = [payload["aggregate"][item]["avg_latency_s_mean"] for item in BUDGET_ORDER]
    plain_accuracy = payload["aggregate"]["plain"]["accuracy_mean"]
    gains = [value - plain_accuracy for value in accuracies]

    for ax in axes:
        ax.set_facecolor("#f8fafc")
        style_axes(ax)

    axes[0].bar(labels, accuracies, yerr=acc_errors, capsize=5, color=colors)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Accuracy vs Evidence Budget", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Mean accuracy")
    axes[0].tick_params(axis="x", rotation=18)

    axes[1].plot(labels, latencies, marker="o", linewidth=2.5, color="#b45309")
    axes[1].fill_between(labels, latencies, color="#fde68a", alpha=0.45)
    axes[1].set_title("Latency Cost of More Context", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Mean latency per example (s)")
    axes[1].tick_params(axis="x", rotation=18)

    axes[2].bar(labels, gains, color=colors)
    axes[2].axhline(0.0, color="#0f172a", linewidth=1)
    axes[2].set_title("Accuracy Gain vs Question Only", fontsize=13, fontweight="bold")
    axes[2].set_ylabel("Accuracy delta")
    axes[2].tick_params(axis="x", rotation=18)

    fig.suptitle("STEMTune MCQA Support Budget Study", fontsize=16, fontweight="bold", color="#0f172a")
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def run_budget_study(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or pick_device()
    seeds = parse_seeds(args.seeds)
    tokenizer, model = load_generator(args.model_id, device)

    all_rows = []
    per_seed = []
    for seed in seeds:
        examples = load_examples(limit=args.limit, seed=seed)
        seed_summary = {"seed": seed}
        for condition in BUDGET_ORDER:
            rows = evaluate_prompts(
                condition,
                examples,
                prompts_for_budget(examples, condition),
                tokenizer,
                model,
                device,
                args.max_new_tokens,
            )
            for row in rows:
                row["seed"] = seed
            all_rows.extend(rows)
            seed_summary[condition] = summarize(condition, rows)
        per_seed.append(seed_summary)

    aggregate = {}
    for condition in BUDGET_ORDER:
        accuracies = [row[condition]["accuracy"] for row in per_seed]
        latencies = [row[condition]["avg_latency_s"] for row in per_seed]
        aggregate[condition] = {
            "accuracy_mean": statistics.mean(accuracies) if accuracies else 0.0,
            "accuracy_stdev": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
            "avg_latency_s_mean": statistics.mean(latencies) if latencies else 0.0,
            "avg_latency_s_stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        }

    payload = {
        "model_id": args.model_id,
        "dataset_id": DEFAULT_DATASET_ID,
        "device": device,
        "limit": args.limit,
        "seeds": seeds,
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    save_predictions_csv(output_dir / "predictions.csv", all_rows)
    save_summary_json(output_dir / "summary.json", payload)
    save_report_markdown(output_dir / "report.md", payload)
    save_plot(output_dir / "study.png", payload)
    return payload


def build_budget_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Study how much support text is needed for MCQA grounding.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--seeds", default="7,11,13,17,23")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output-dir", default="docs/results/mcqa_support_budget_qwen25_0p5b")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_budget_parser()
    args = parser.parse_args(argv)
    payload = run_budget_study(args)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
