from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from stemtune.benchmark_mcqa import parse_seeds
from stemtune.smoke_mcqa import (
    DEFAULT_DATASET_ID,
    McqaExample,
    evaluate_condition,
    evaluate_prompts,
    load_examples,
    load_generator,
    make_support_prompt,
    pick_device,
    summarize,
)


DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
]

CONDITION_ORDER = ["plain", "shuffled", "grounded"]
CONDITION_LABELS = {
    "plain": "Question only",
    "shuffled": "Mismatched support",
    "grounded": "Correct support",
}
CONDITION_COLORS = {
    "plain": "#1d4ed8",
    "shuffled": "#94a3b8",
    "grounded": "#0f766e",
}


def parse_models(raw: str) -> list[str]:
    if not raw.strip():
        return list(DEFAULT_MODELS)
    models = [item.strip() for item in raw.split(",") if item.strip()]
    if not models:
        raise ValueError("At least one model is required.")
    return models


def model_slug(model_id: str) -> str:
    return model_id.split("/")[-1].lower().replace(".", "p")


def shuffled_support_prompts(examples: list[McqaExample]) -> list[str]:
    supports = [example.support for example in examples]
    rotated = supports[1:] + supports[:1]
    return [make_support_prompt(example, support_text) for example, support_text in zip(examples, rotated)]


def summarize_model(model_id: str, per_seed: list[dict]) -> dict:
    aggregate: dict[str, dict[str, float]] = {}
    for condition in CONDITION_ORDER:
        accuracies = [row[condition]["accuracy"] for row in per_seed]
        valid_rates = [row[condition]["valid_rate"] for row in per_seed]
        latencies = [row[condition]["avg_latency_s"] for row in per_seed]
        aggregate[condition] = {
            "accuracy_mean": statistics.mean(accuracies) if accuracies else 0.0,
            "accuracy_stdev": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
            "valid_rate_mean": statistics.mean(valid_rates) if valid_rates else 0.0,
            "valid_rate_stdev": statistics.stdev(valid_rates) if len(valid_rates) > 1 else 0.0,
            "avg_latency_s_mean": statistics.mean(latencies) if latencies else 0.0,
            "avg_latency_s_stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        }

    grounded_gains = [row["grounded"]["accuracy"] - row["plain"]["accuracy"] for row in per_seed]
    shuffled_gains = [row["shuffled"]["accuracy"] - row["plain"]["accuracy"] for row in per_seed]
    return {
        "model_id": model_id,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "grounded_gain_mean": statistics.mean(grounded_gains) if grounded_gains else 0.0,
        "grounded_gain_stdev": statistics.stdev(grounded_gains) if len(grounded_gains) > 1 else 0.0,
        "shuffled_gain_mean": statistics.mean(shuffled_gains) if shuffled_gains else 0.0,
        "shuffled_gain_stdev": statistics.stdev(shuffled_gains) if len(shuffled_gains) > 1 else 0.0,
    }


def release_model(tokenizer, model) -> None:
    del tokenizer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def save_predictions_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "model_id",
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


def save_model_summary_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "model_id",
        "plain_accuracy_mean",
        "shuffled_accuracy_mean",
        "grounded_accuracy_mean",
        "grounded_gain_mean",
        "shuffled_gain_mean",
        "plain_latency_mean",
        "shuffled_latency_mean",
        "grounded_latency_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            aggregate = row["aggregate"]
            writer.writerow(
                {
                    "model_id": row["model_id"],
                    "plain_accuracy_mean": aggregate["plain"]["accuracy_mean"],
                    "shuffled_accuracy_mean": aggregate["shuffled"]["accuracy_mean"],
                    "grounded_accuracy_mean": aggregate["grounded"]["accuracy_mean"],
                    "grounded_gain_mean": row["grounded_gain_mean"],
                    "shuffled_gain_mean": row["shuffled_gain_mean"],
                    "plain_latency_mean": aggregate["plain"]["avg_latency_s_mean"],
                    "shuffled_latency_mean": aggregate["shuffled"]["avg_latency_s_mean"],
                    "grounded_latency_mean": aggregate["grounded"]["avg_latency_s_mean"],
                }
            )


def save_summary_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)


def save_plot(path: Path, payload: dict) -> None:
    if len(payload["models"]) == 1:
        return save_single_model_plot(path, payload)

    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    fig.patch.set_facecolor("#f8fafc")
    for ax in axes.flat:
        ax.set_facecolor("#f8fafc")
        style_axes(ax)

    models = [row["model_id"].split("/")[-1] for row in payload["models"]]
    x = list(range(len(models)))
    width = 0.24

    ax = axes[0, 0]
    for idx, condition in enumerate(CONDITION_ORDER):
        means = [row["aggregate"][condition]["accuracy_mean"] for row in payload["models"]]
        errors = [row["aggregate"][condition]["accuracy_stdev"] for row in payload["models"]]
        positions = [value + (idx - 1) * width for value in x]
        ax.bar(
            positions,
            means,
            width=width,
            color=CONDITION_COLORS[condition],
            yerr=errors,
            capsize=5,
            label=CONDITION_LABELS[condition],
        )
    ax.set_xticks(x, models, rotation=12, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy by Model and Evidence Condition", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean accuracy")
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))

    ax = axes[0, 1]
    grounded = [row["grounded_gain_mean"] for row in payload["models"]]
    shuffled = [row["shuffled_gain_mean"] for row in payload["models"]]
    grounded_err = [row["grounded_gain_stdev"] for row in payload["models"]]
    shuffled_err = [row["shuffled_gain_stdev"] for row in payload["models"]]
    ax.bar([value - width / 2 for value in x], shuffled, width=width, color=CONDITION_COLORS["shuffled"], yerr=shuffled_err, capsize=5, label="Mismatched vs plain")
    ax.bar([value + width / 2 for value in x], grounded, width=width, color=CONDITION_COLORS["grounded"], yerr=grounded_err, capsize=5, label="Correct vs plain")
    ax.axhline(0.0, color="#0f172a", linewidth=1)
    ax.set_xticks(x, models, rotation=12, ha="right")
    ax.set_title("Evidence Helps Only When It Is Relevant", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy delta")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    for condition in CONDITION_ORDER:
        xs = [row["aggregate"][condition]["avg_latency_s_mean"] for row in payload["models"]]
        ys = [row["aggregate"][condition]["accuracy_mean"] for row in payload["models"]]
        ax.scatter(xs, ys, s=100, color=CONDITION_COLORS[condition], label=CONDITION_LABELS[condition], alpha=0.95)
        for x_value, y_value, label in zip(xs, ys, models):
            ax.annotate(label, (x_value, y_value), textcoords="offset points", xytext=(6, 6), fontsize=9, color="#334155")
    ax.set_title("Latency vs Accuracy Frontier", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean latency per example (s)")
    ax.set_ylabel("Mean accuracy")
    ax.set_ylim(0.55, 1.02)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1, 1]
    for index, row in enumerate(payload["models"]):
        gains = [seed_row["grounded"]["accuracy"] - seed_row["plain"]["accuracy"] for seed_row in row["per_seed"]]
        ax.scatter(
            [index] * len(gains),
            gains,
            color=CONDITION_COLORS["grounded"],
            s=60,
            alpha=0.9,
        )
        ax.vlines(index, min(gains), max(gains), color="#64748b", linewidth=2)
        ax.hlines(row["grounded_gain_mean"], index - 0.18, index + 0.18, color="#0f172a", linewidth=3)
    ax.axhline(0.0, color="#0f172a", linewidth=1)
    ax.set_xticks(x, models, rotation=12, ha="right")
    ax.set_title("Grounding Gain Distribution Across Seeds", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy delta")

    fig.suptitle("STEMTune MCQA Evidence Study", fontsize=18, fontweight="bold", color="#0f172a")
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_single_model_plot(path: Path, payload: dict) -> None:
    plt.style.use("default")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("#f8fafc")
    model = payload["models"][0]
    seed_labels = [str(row["seed"]) for row in model["per_seed"]]

    for ax in axes:
        ax.set_facecolor("#f8fafc")
        style_axes(ax)

    axes[0].plot(
        seed_labels,
        [row["plain"]["accuracy"] for row in model["per_seed"]],
        marker="o",
        linewidth=2.5,
        color=CONDITION_COLORS["plain"],
        label=CONDITION_LABELS["plain"],
    )
    axes[0].plot(
        seed_labels,
        [row["shuffled"]["accuracy"] for row in model["per_seed"]],
        marker="o",
        linewidth=2.5,
        color=CONDITION_COLORS["shuffled"],
        label=CONDITION_LABELS["shuffled"],
    )
    axes[0].plot(
        seed_labels,
        [row["grounded"]["accuracy"] for row in model["per_seed"]],
        marker="o",
        linewidth=2.5,
        color=CONDITION_COLORS["grounded"],
        label=CONDITION_LABELS["grounded"],
    )
    axes[0].set_ylim(0.55, 1.02)
    axes[0].set_title("Accuracy by Seed and Evidence Condition", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(frameon=False, loc="lower right")

    aggregate = model["aggregate"]
    labels = [CONDITION_LABELS[item] for item in CONDITION_ORDER]
    means = [aggregate[item]["accuracy_mean"] for item in CONDITION_ORDER]
    errors = [aggregate[item]["accuracy_stdev"] for item in CONDITION_ORDER]
    colors = [CONDITION_COLORS[item] for item in CONDITION_ORDER]
    axes[1].bar(labels, means, yerr=errors, capsize=6, color=colors)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Aggregate Accuracy", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Mean +/- stdev")
    axes[1].tick_params(axis="x", rotation=15)

    grounded_gains = [row["grounded"]["accuracy"] - row["plain"]["accuracy"] for row in model["per_seed"]]
    shuffled_gains = [row["shuffled"]["accuracy"] - row["plain"]["accuracy"] for row in model["per_seed"]]
    axes[2].bar(seed_labels, shuffled_gains, color=CONDITION_COLORS["shuffled"], label="Mismatched vs plain", alpha=0.95)
    axes[2].bar(seed_labels, grounded_gains, color=CONDITION_COLORS["grounded"], label="Correct vs plain", alpha=0.95)
    axes[2].axhline(0.0, color="#0f172a", linewidth=1)
    axes[2].set_title("Gain Relative to Question Only", fontsize=13, fontweight="bold")
    axes[2].set_ylabel("Accuracy delta")
    axes[2].legend(frameon=False, loc="upper right")

    fig.suptitle("STEMTune MCQA Evidence Study", fontsize=16, fontweight="bold", color="#0f172a")
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_report_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# MCQA Evidence Study",
        "",
        f"- Dataset: `{payload['dataset_id']}`",
        f"- Device: `{payload['device']}`",
        f"- Models: `{', '.join(payload['model_ids'])}`",
        f"- Seeds: `{', '.join(str(seed) for seed in payload['seeds'])}`",
        f"- Examples per seed: `{payload['limit']}`",
        "",
        "## Model Summary",
        "",
        "| Model | Plain | Mismatched Support | Correct Support | Correct Gain | Mismatched Gain |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["models"]:
        model_name = row["model_id"].split("/")[-1]
        aggregate = row["aggregate"]
        lines.append(
            f"| {model_name} | {aggregate['plain']['accuracy_mean']:.3f} | {aggregate['shuffled']['accuracy_mean']:.3f} | {aggregate['grounded']['accuracy_mean']:.3f} | {row['grounded_gain_mean']:+.3f} | {row['shuffled_gain_mean']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Correct support should lift accuracy materially above the question-only baseline.",
            "- Mismatched support should not produce the same gain; otherwise the effect could be explained by prompt length alone.",
            "- The latency frontier helps decide whether the accuracy gain is worth the added context cost for deployment.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_study(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or pick_device()
    seeds = parse_seeds(args.seeds)
    model_ids = parse_models(args.models)

    all_predictions = []
    all_models = []
    for model_id in model_ids:
        tokenizer, model = load_generator(model_id, device)
        per_seed = []
        try:
            for seed in seeds:
                examples = load_examples(limit=args.limit, seed=seed)
                plain_rows = evaluate_condition("plain", examples, tokenizer, model, device, args.max_new_tokens)
                shuffled_rows = evaluate_prompts(
                    "shuffled",
                    examples,
                    shuffled_support_prompts(examples),
                    tokenizer,
                    model,
                    device,
                    args.max_new_tokens,
                )
                grounded_rows = evaluate_condition("grounded", examples, tokenizer, model, device, args.max_new_tokens)

                for bucket in (plain_rows, shuffled_rows, grounded_rows):
                    for row in bucket:
                        row["seed"] = seed
                        row["model_id"] = model_id
                        all_predictions.append(row)

                per_seed.append(
                    {
                        "seed": seed,
                        "plain": summarize("plain", plain_rows),
                        "shuffled": summarize("shuffled", shuffled_rows),
                        "grounded": summarize("grounded", grounded_rows),
                    }
                )
        finally:
            release_model(tokenizer, model)

        all_models.append(summarize_model(model_id, per_seed))

    payload = {
        "dataset_id": DEFAULT_DATASET_ID,
        "device": device,
        "limit": args.limit,
        "seeds": seeds,
        "model_ids": model_ids,
        "models": all_models,
        "total_examples": len(model_ids) * len(seeds) * args.limit,
    }

    save_predictions_csv(output_dir / "predictions.csv", all_predictions)
    save_model_summary_csv(output_dir / "model_summary.csv", all_models)
    save_summary_json(output_dir / "summary.json", payload)
    save_report_markdown(output_dir / "report.md", payload)
    save_plot(output_dir / "study.png", payload)
    return payload


def build_study_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a multi-model MCQA evidence study with ablations.")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--seeds", default="7,11,13")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output-dir", default="docs/results/mcqa_evidence_study")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_study_parser()
    args = parser.parse_args(argv)
    payload = run_study(args)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
