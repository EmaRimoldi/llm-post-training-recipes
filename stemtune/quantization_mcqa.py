from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from stemtune.benchmark_mcqa import parse_seeds
from stemtune.smoke_mcqa import DEFAULT_DATASET_ID, DEFAULT_MODEL_ID, evaluate_condition, load_examples, load_generator, summarize


def quantize_model_dynamic_int8(model):
    torch.backends.quantized.engine = "qnnpack"
    return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def save_predictions_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "variant",
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


def aggregate_variant(per_seed: list[dict], key: str) -> dict:
    accuracies = [row[key]["accuracy"] for row in per_seed]
    valid_rates = [row[key]["valid_rate"] for row in per_seed]
    latencies = [row[key]["avg_latency_s"] for row in per_seed]
    return {
        "accuracy_mean": statistics.mean(accuracies) if accuracies else 0.0,
        "accuracy_stdev": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
        "valid_rate_mean": statistics.mean(valid_rates) if valid_rates else 0.0,
        "valid_rate_stdev": statistics.stdev(valid_rates) if len(valid_rates) > 1 else 0.0,
        "avg_latency_s_mean": statistics.mean(latencies) if latencies else 0.0,
        "avg_latency_s_stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
    }


def save_summary_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def save_report_markdown(path: Path, payload: dict) -> None:
    fp = payload["aggregate"]["full_precision"]
    q = payload["aggregate"]["dynamic_int8"]
    lines = [
        "# MCQA Quantization Retention Study",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- Dataset: `{payload['dataset_id']}`",
        f"- Device: `{payload['device']}`",
        f"- Condition: `{payload['condition']}`",
        f"- Seeds: `{', '.join(str(seed) for seed in payload['seeds'])}`",
        f"- Examples per seed: `{payload['limit']}`",
        "",
        "| Variant | Mean Accuracy | Mean Valid Rate | Mean Latency (s) |",
        "|---|---:|---:|---:|",
        f"| Full precision | {fp['accuracy_mean']:.3f} | {fp['valid_rate_mean']:.3f} | {fp['avg_latency_s_mean']:.3f} |",
        f"| Dynamic int8 | {q['accuracy_mean']:.3f} | {q['valid_rate_mean']:.3f} | {q['avg_latency_s_mean']:.3f} |",
        "",
        f"- Accuracy delta: `{q['accuracy_mean'] - fp['accuracy_mean']:+.3f}`",
        f"- Latency delta: `{q['avg_latency_s_mean'] - fp['avg_latency_s_mean']:+.3f}` seconds",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_plot(path: Path, payload: dict) -> None:
    fp = payload["aggregate"]["full_precision"]
    q = payload["aggregate"]["dynamic_int8"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("#f8fafc")
    colors = ["#1d4ed8", "#7c3aed"]

    metrics = [
        ("Accuracy", [fp["accuracy_mean"], q["accuracy_mean"]], [fp["accuracy_stdev"], q["accuracy_stdev"]], (0, 1.05)),
        ("Valid Rate", [fp["valid_rate_mean"], q["valid_rate_mean"]], [fp["valid_rate_stdev"], q["valid_rate_stdev"]], (0, 1.05)),
        ("Avg Latency (s)", [fp["avg_latency_s_mean"], q["avg_latency_s_mean"]], [fp["avg_latency_s_stdev"], q["avg_latency_s_stdev"]], None),
    ]

    for ax, (title, values, errors, ylim) in zip(axes, metrics):
        ax.bar(["FP", "INT8"], values, yerr=errors, capsize=6, color=colors)
        ax.set_title(title, fontsize=13, fontweight="bold")
        if ylim:
            ax.set_ylim(*ylim)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_facecolor("#f8fafc")

    fig.suptitle("STEMTune MCQA Quantization Retention Study", fontsize=16, fontweight="bold", color="#0f172a")
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def run_quantization_study(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_seeds(args.seeds)
    tokenizer, model = load_generator(args.model_id, "cpu")
    quantized_model = quantize_model_dynamic_int8(model)

    all_rows = []
    per_seed = []
    for seed in seeds:
        examples = load_examples(limit=args.limit, seed=seed)
        fp_rows = evaluate_condition(args.condition, examples, tokenizer, model, "cpu", args.max_new_tokens)
        q_rows = evaluate_condition(args.condition, examples, tokenizer, quantized_model, "cpu", args.max_new_tokens)
        for row in fp_rows:
            row["seed"] = seed
            row["variant"] = "full_precision"
            all_rows.append(row)
        for row in q_rows:
            row["seed"] = seed
            row["variant"] = "dynamic_int8"
            all_rows.append(row)
        per_seed.append(
            {
                "seed": seed,
                "full_precision": summarize(args.condition, fp_rows),
                "dynamic_int8": summarize(args.condition, q_rows),
            }
        )

    payload = {
        "model_id": args.model_id,
        "dataset_id": DEFAULT_DATASET_ID,
        "device": "cpu",
        "condition": args.condition,
        "limit": args.limit,
        "seeds": seeds,
        "per_seed": per_seed,
        "aggregate": {
            "full_precision": aggregate_variant(per_seed, "full_precision"),
            "dynamic_int8": aggregate_variant(per_seed, "dynamic_int8"),
        },
    }
    save_predictions_csv(output_dir / "predictions.csv", all_rows)
    save_summary_json(output_dir / "summary.json", payload)
    save_report_markdown(output_dir / "report.md", payload)
    save_plot(output_dir / "study.png", payload)
    return payload


def build_quantization_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test whether CPU dynamic int8 quantization preserves MCQA behavior.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--condition", choices=["plain", "grounded"], default="plain")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--seeds", default="7,11,13")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output-dir", default="docs/results/mcqa_quantization_retention")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_quantization_parser()
    args = parser.parse_args(argv)
    payload = run_quantization_study(args)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
