from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from stemtune.benchmark_mcqa import parse_seeds
from stemtune.smoke_mcqa import (
    DEFAULT_DATASET_ID,
    DEFAULT_MODEL_ID,
    build_example,
    evaluate_condition,
    evaluate_prompts,
    load_generator,
    make_support_prompt,
    summarize,
)


def build_corpus(train_limit: int, eval_records) -> list[str]:
    train = load_dataset(DEFAULT_DATASET_ID, split="train").select(range(train_limit))
    corpus = [record["support"].strip() for record in train]
    corpus.extend(record["support"].strip() for record in eval_records)
    return corpus


def retrieve_supports(eval_records, corpus: list[str]) -> tuple[list[str], float]:
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(corpus)
    retrieved = []
    hits = 0
    for record in eval_records:
        query = " ".join(
            [
                record["question"],
                record["correct_answer"],
                record["distractor1"],
                record["distractor2"],
                record["distractor3"],
            ]
        )
        scores = linear_kernel(vectorizer.transform([query]), matrix).ravel()
        index = int(scores.argmax())
        support = corpus[index]
        retrieved.append(support)
        if support == record["support"].strip():
            hits += 1
    return retrieved, hits / len(eval_records) if eval_records else 0.0


def save_predictions_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "condition",
        "seed",
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
    aggregate = payload["aggregate"]
    lines = [
        "# MCQA Retrieval Study",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- Dataset: `{payload['dataset_id']}`",
        f"- Device: `{payload['device']}`",
        f"- Corpus size: `{payload['corpus_size']}`",
        f"- Retrieval hit@1: `{payload['retrieval_hit_rate']:.3f}`",
        "",
        "| Condition | Mean Accuracy | Mean Latency (s) |",
        "|---|---:|---:|",
        f"| Question only | {aggregate['plain']['accuracy_mean']:.3f} | {aggregate['plain']['avg_latency_s_mean']:.3f} |",
        f"| Retrieved support | {aggregate['retrieved']['accuracy_mean']:.3f} | {aggregate['retrieved']['avg_latency_s_mean']:.3f} |",
        f"| Oracle support | {aggregate['oracle']['accuracy_mean']:.3f} | {aggregate['oracle']['avg_latency_s_mean']:.3f} |",
        "",
        f"- Retrieved gain vs plain: `{aggregate['retrieved']['accuracy_mean'] - aggregate['plain']['accuracy_mean']:+.3f}`",
        f"- Oracle gain vs plain: `{aggregate['oracle']['accuracy_mean'] - aggregate['plain']['accuracy_mean']:+.3f}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_condition(per_seed: list[dict], key: str) -> dict:
    accuracies = [row[key]["accuracy"] for row in per_seed]
    latencies = [row[key]["avg_latency_s"] for row in per_seed]
    return {
        "accuracy_mean": statistics.mean(accuracies) if accuracies else 0.0,
        "accuracy_stdev": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
        "avg_latency_s_mean": statistics.mean(latencies) if latencies else 0.0,
        "avg_latency_s_stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
    }


def save_plot(path: Path, payload: dict) -> None:
    aggregate = payload["aggregate"]
    labels = ["Question only", "Retrieved", "Oracle"]
    values = [
        aggregate["plain"]["accuracy_mean"],
        aggregate["retrieved"]["accuracy_mean"],
        aggregate["oracle"]["accuracy_mean"],
    ]
    errors = [
        aggregate["plain"]["accuracy_stdev"],
        aggregate["retrieved"]["accuracy_stdev"],
        aggregate["oracle"]["accuracy_stdev"],
    ]
    latencies = [
        aggregate["plain"]["avg_latency_s_mean"],
        aggregate["retrieved"]["avg_latency_s_mean"],
        aggregate["oracle"]["avg_latency_s_mean"],
    ]
    colors = ["#1d4ed8", "#0f766e", "#059669"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("#f8fafc")

    axes[0].bar(labels, values, yerr=errors, capsize=6, color=colors)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Accuracy", fontsize=13, fontweight="bold")

    axes[1].bar(labels, latencies, color=colors)
    axes[1].set_title("Average Latency", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Seconds")

    axes[2].bar(["Hit@1"], [payload["retrieval_hit_rate"]], color="#b45309")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Retriever Quality", fontsize=13, fontweight="bold")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_facecolor("#f8fafc")
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("STEMTune MCQA Retrieval Study", fontsize=16, fontweight="bold", color="#0f172a")
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def run_rag_study(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_seeds(args.seeds)
    tokenizer, model = load_generator(args.model_id, args.device)

    all_rows = []
    per_seed = []
    hit_rates = []
    for seed in seeds:
        eval_records = list(load_dataset(DEFAULT_DATASET_ID, split="validation").shuffle(seed=seed).select(range(args.limit)))
        corpus = build_corpus(args.corpus_size, eval_records)
        retrieved_supports, hit_rate = retrieve_supports(eval_records, corpus)
        hit_rates.append(hit_rate)
        rng = random.Random(seed)
        examples = [build_example(record, rng, index) for index, record in enumerate(eval_records)]

        plain_rows = evaluate_condition("plain", examples, tokenizer, model, args.device, args.max_new_tokens)
        oracle_rows = evaluate_condition("grounded", examples, tokenizer, model, args.device, args.max_new_tokens)
        retrieved_rows = evaluate_prompts(
            "retrieved",
            examples,
            [make_support_prompt(example, support) for example, support in zip(examples, retrieved_supports)],
            tokenizer,
            model,
            args.device,
            args.max_new_tokens,
        )
        for bucket in (plain_rows, retrieved_rows, oracle_rows):
            for row in bucket:
                row["seed"] = seed
                all_rows.append(row)
        per_seed.append(
            {
                "seed": seed,
                "plain": summarize("plain", plain_rows),
                "retrieved": summarize("retrieved", retrieved_rows),
                "oracle": summarize("oracle", oracle_rows),
                "retrieval_hit_rate": hit_rate,
            }
        )

    payload = {
        "model_id": args.model_id,
        "dataset_id": DEFAULT_DATASET_ID,
        "device": args.device,
        "limit": args.limit,
        "seeds": seeds,
        "corpus_size": args.corpus_size + args.limit,
        "retrieval_hit_rate": statistics.mean(hit_rates) if hit_rates else 0.0,
        "per_seed": per_seed,
        "aggregate": {
            "plain": aggregate_condition(per_seed, "plain"),
            "retrieved": aggregate_condition(per_seed, "retrieved"),
            "oracle": aggregate_condition(per_seed, "oracle"),
        },
    }
    save_predictions_csv(output_dir / "predictions.csv", all_rows)
    save_summary_json(output_dir / "summary.json", payload)
    save_report_markdown(output_dir / "report.md", payload)
    save_plot(output_dir / "study.png", payload)
    return payload


def build_rag_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a simple MCQA RAG study with TF-IDF retrieval.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--seeds", default="7,11,13")
    parser.add_argument("--corpus-size", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output-dir", default="docs/results/mcqa_rag_retrieval")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_rag_parser()
    args = parser.parse_args(argv)
    payload = run_rag_study(args)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
