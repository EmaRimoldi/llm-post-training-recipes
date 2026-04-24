from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

from stemtune.smoke_mcqa import (
    DEFAULT_DATASET_ID,
    DEFAULT_MODEL_ID,
    evaluate_condition,
    load_examples,
    load_generator,
    pick_device,
    summarize,
)


def parse_seeds(raw: str) -> list[int]:
    seeds = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            seeds.append(int(item))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def enrich_rows(rows: list[dict], seed: int) -> list[dict]:
    return [{**row, "seed": seed} for row in rows]


def aggregate_metric(per_seed: list[dict], condition: str, metric: str) -> tuple[float, float]:
    values = [row[condition][metric] for row in per_seed]
    mean = statistics.mean(values) if values else 0.0
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, stdev


def save_rows_csv(path: Path, rows: list[dict]) -> None:
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


def save_seed_summary_csv(path: Path, per_seed: list[dict]) -> None:
    fieldnames = [
        "seed",
        "plain_accuracy",
        "grounded_accuracy",
        "accuracy_gain",
        "plain_valid_rate",
        "grounded_valid_rate",
        "plain_avg_latency_s",
        "grounded_avg_latency_s",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_seed:
            plain = row["plain"]
            grounded = row["grounded"]
            writer.writerow(
                {
                    "seed": row["seed"],
                    "plain_accuracy": plain["accuracy"],
                    "grounded_accuracy": grounded["accuracy"],
                    "accuracy_gain": grounded["accuracy"] - plain["accuracy"],
                    "plain_valid_rate": plain["valid_rate"],
                    "grounded_valid_rate": grounded["valid_rate"],
                    "plain_avg_latency_s": plain["avg_latency_s"],
                    "grounded_avg_latency_s": grounded["avg_latency_s"],
                }
            )


def save_summary_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def save_report_markdown(path: Path, payload: dict) -> None:
    aggregate = payload["aggregate"]
    lines = [
        "# MCQA Grounding Benchmark",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- Dataset: `{payload['dataset_id']}`",
        f"- Device: `{payload['device']}`",
        f"- Seeds: `{', '.join(str(seed) for seed in payload['seeds'])}`",
        f"- Examples per seed: `{payload['limit']}`",
        "",
        "## Aggregate Results",
        "",
        "| Condition | Mean Accuracy | Mean Valid Rate | Mean Latency (s) |",
        "|---|---:|---:|---:|",
        f"| Plain question-only | {aggregate['plain']['accuracy_mean']:.3f} | {aggregate['plain']['valid_rate_mean']:.3f} | {aggregate['plain']['avg_latency_s_mean']:.3f} |",
        f"| Grounded with support passage | {aggregate['grounded']['accuracy_mean']:.3f} | {aggregate['grounded']['valid_rate_mean']:.3f} | {aggregate['grounded']['avg_latency_s_mean']:.3f} |",
        "",
        f"- Mean accuracy gain: `{aggregate['accuracy_gain_mean']:+.3f}`",
        f"- Accuracy gain std: `{aggregate['accuracy_gain_stdev']:.3f}`",
        f"- Total evaluated examples: `{payload['total_examples']}`",
        "",
        "## Per-Seed Accuracy",
        "",
        "| Seed | Plain | Grounded | Gain |",
        "|---:|---:|---:|---:|",
    ]
    for row in payload["per_seed"]:
        lines.append(
            f"| {row['seed']} | {row['plain']['accuracy']:.3f} | {row['grounded']['accuracy']:.3f} | {row['grounded']['accuracy'] - row['plain']['accuracy']:+.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_plot(path: Path, payload: dict) -> None:
    seeds = [str(item["seed"]) for item in payload["per_seed"]]
    plain_acc = [item["plain"]["accuracy"] for item in payload["per_seed"]]
    grounded_acc = [item["grounded"]["accuracy"] for item in payload["per_seed"]]
    gains = [grounded - plain for plain, grounded in zip(plain_acc, grounded_acc)]

    aggregate = payload["aggregate"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    x = range(len(seeds))
    axes[0].plot(x, plain_acc, marker="o", color="#2563eb", label="Plain")
    axes[0].plot(x, grounded_acc, marker="o", color="#059669", label="Grounded")
    axes[0].set_xticks(list(x), seeds)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Accuracy by Seed")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(frameon=False)

    axes[1].bar(seeds, gains, color="#f59e0b")
    axes[1].axhline(0.0, color="#111827", linewidth=1)
    axes[1].set_ylim(min(-0.05, min(gains) - 0.05), max(0.05, max(gains) + 0.05))
    axes[1].set_title("Grounding Gain by Seed")
    axes[1].set_ylabel("Accuracy delta")

    means = [aggregate["plain"]["accuracy_mean"], aggregate["grounded"]["accuracy_mean"]]
    errors = [aggregate["plain"]["accuracy_stdev"], aggregate["grounded"]["accuracy_stdev"]]
    axes[2].bar(["Plain", "Grounded"], means, yerr=errors, capsize=6, color=["#2563eb", "#059669"])
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Aggregate Accuracy")
    axes[2].set_ylabel("Mean +/- stdev")

    fig.suptitle("STEMTune MCQA Grounding Benchmark", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_benchmark(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or pick_device()
    seeds = parse_seeds(args.seeds)
    tokenizer, model = load_generator(args.model_id, device)

    all_rows = []
    per_seed = []
    for seed in seeds:
        examples = load_examples(limit=args.limit, seed=seed)
        plain_rows = enrich_rows(
            evaluate_condition("plain", examples, tokenizer, model, device, args.max_new_tokens),
            seed,
        )
        grounded_rows = enrich_rows(
            evaluate_condition("grounded", examples, tokenizer, model, device, args.max_new_tokens),
            seed,
        )
        all_rows.extend(plain_rows)
        all_rows.extend(grounded_rows)
        per_seed.append(
            {
                "seed": seed,
                "plain": summarize("plain", plain_rows),
                "grounded": summarize("grounded", grounded_rows),
            }
        )

    plain_accuracy_mean, plain_accuracy_stdev = aggregate_metric(per_seed, "plain", "accuracy")
    grounded_accuracy_mean, grounded_accuracy_stdev = aggregate_metric(per_seed, "grounded", "accuracy")
    plain_valid_mean, plain_valid_stdev = aggregate_metric(per_seed, "plain", "valid_rate")
    grounded_valid_mean, grounded_valid_stdev = aggregate_metric(per_seed, "grounded", "valid_rate")
    plain_latency_mean, plain_latency_stdev = aggregate_metric(per_seed, "plain", "avg_latency_s")
    grounded_latency_mean, grounded_latency_stdev = aggregate_metric(per_seed, "grounded", "avg_latency_s")
    gains = [row["grounded"]["accuracy"] - row["plain"]["accuracy"] for row in per_seed]

    payload = {
        "model_id": args.model_id,
        "dataset_id": DEFAULT_DATASET_ID,
        "device": device,
        "limit": args.limit,
        "seeds": seeds,
        "total_examples": len(seeds) * args.limit,
        "per_seed": per_seed,
        "aggregate": {
            "plain": {
                "accuracy_mean": plain_accuracy_mean,
                "accuracy_stdev": plain_accuracy_stdev,
                "valid_rate_mean": plain_valid_mean,
                "valid_rate_stdev": plain_valid_stdev,
                "avg_latency_s_mean": plain_latency_mean,
                "avg_latency_s_stdev": plain_latency_stdev,
            },
            "grounded": {
                "accuracy_mean": grounded_accuracy_mean,
                "accuracy_stdev": grounded_accuracy_stdev,
                "valid_rate_mean": grounded_valid_mean,
                "valid_rate_stdev": grounded_valid_stdev,
                "avg_latency_s_mean": grounded_latency_mean,
                "avg_latency_s_stdev": grounded_latency_stdev,
            },
            "accuracy_gain_mean": statistics.mean(gains) if gains else 0.0,
            "accuracy_gain_stdev": statistics.stdev(gains) if len(gains) > 1 else 0.0,
        },
    }

    save_rows_csv(output_dir / "predictions.csv", all_rows)
    save_seed_summary_csv(output_dir / "seed_summary.csv", per_seed)
    save_summary_json(output_dir / "summary.json", payload)
    save_report_markdown(output_dir / "report.md", payload)
    save_plot(output_dir / "benchmark.png", payload)
    return payload


def build_benchmark_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark question-only vs grounded MCQA prompts across multiple seeds.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--seeds", default="7,11,13,17,23")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output-dir", default="docs/results/mcqa_grounding_qwen25_0p5b")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_benchmark_parser()
    args = parser.parse_args(argv)
    payload = run_benchmark(args)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
