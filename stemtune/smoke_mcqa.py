from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_DATASET_ID = "allenai/sciq"


LETTER_PATTERN = re.compile(r"\b([A-D])\b")


@dataclass
class McqaExample:
    example_id: str
    question: str
    support: str
    choices: list[str]
    correct_letter: str
    correct_text: str


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    if device in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def format_choices(choices: list[str]) -> str:
    letters = ["A", "B", "C", "D"]
    return "\n".join(f"{letter}. {choice}" for letter, choice in zip(letters, choices))


def make_plain_prompt(example: McqaExample) -> str:
    return (
        "Answer the following science multiple-choice question.\n"
        "Return only the final answer letter: A, B, C, or D.\n\n"
        f"Question:\n{example.question}\n\n"
        f"Choices:\n{format_choices(example.choices)}\n"
    )


def make_grounded_prompt(example: McqaExample) -> str:
    return make_support_prompt(example, example.support)


def make_support_prompt(example: McqaExample, support_text: str) -> str:
    return (
        "Answer the following science multiple-choice question using the support passage.\n"
        "Return only the final answer letter: A, B, C, or D.\n\n"
        f"Support passage:\n{support_text}\n\n"
        f"Question:\n{example.question}\n\n"
        f"Choices:\n{format_choices(example.choices)}\n"
    )


def build_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You solve science multiple-choice questions carefully and output a single answer letter.",
        },
        {"role": "user", "content": prompt},
    ]


def parse_prediction(text: str) -> str | None:
    match = LETTER_PATTERN.search(text.upper())
    if match:
        return match.group(1)
    cleaned = text.strip().upper()
    if cleaned[:1] in {"A", "B", "C", "D"}:
        return cleaned[:1]
    return None


def build_example(record: dict, rng: random.Random, index: int) -> McqaExample:
    options = [
        record["correct_answer"],
        record["distractor1"],
        record["distractor2"],
        record["distractor3"],
    ]
    rng.shuffle(options)
    correct_text = record["correct_answer"]
    correct_letter = "ABCD"[options.index(correct_text)]
    return McqaExample(
        example_id=str(record.get("id", index)),
        question=record["question"].strip(),
        support=record["support"].strip(),
        choices=options,
        correct_letter=correct_letter,
        correct_text=correct_text,
    )


def load_examples(limit: int, seed: int) -> list[McqaExample]:
    dataset = load_dataset(DEFAULT_DATASET_ID, split="validation")
    subset = dataset.shuffle(seed=seed).select(range(limit))
    rng = random.Random(seed)
    return [build_example(record, rng, index) for index, record in enumerate(subset)]


def load_generator(model_id: str, device: str):
    dtype = pick_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model.to(device)
    model.eval()
    return tokenizer, model


def generate_letter(tokenizer, model, device: str, prompt: str, max_new_tokens: int) -> tuple[str, str | None, float]:
    messages = build_messages(prompt)
    if getattr(tokenizer, "chat_template", None):
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        rendered = (
            "System: You solve science multiple-choice questions carefully and output a single answer letter.\n\n"
            f"User: {prompt}\n\nAssistant:"
        )
    inputs = tokenizer(rendered, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    latency = time.perf_counter() - start
    generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return text, parse_prediction(text), latency


def evaluate_prompts(
    condition_name: str,
    examples: list[McqaExample],
    prompts: list[str],
    tokenizer,
    model,
    device: str,
    max_new_tokens: int,
):
    rows = []
    for example, prompt in zip(examples, prompts):
        raw_text, prediction, latency = generate_letter(tokenizer, model, device, prompt, max_new_tokens)
        rows.append(
            {
                "condition": condition_name,
                "example_id": example.example_id,
                "question": example.question,
                "correct_letter": example.correct_letter,
                "prediction": prediction,
                "is_correct": prediction == example.correct_letter,
                "is_valid": prediction is not None,
                "latency_s": latency,
                "raw_output": raw_text,
            }
        )
    return rows


def evaluate_condition(
    condition_name: str,
    examples: list[McqaExample],
    tokenizer,
    model,
    device: str,
    max_new_tokens: int,
):
    prompt_builder = make_plain_prompt if condition_name == "plain" else make_grounded_prompt
    prompts = [prompt_builder(example) for example in examples]
    return evaluate_prompts(condition_name, examples, prompts, tokenizer, model, device, max_new_tokens)


def summarize(condition_name: str, rows: list[dict]) -> dict:
    total = len(rows)
    correct = sum(1 for row in rows if row["is_correct"])
    valid = sum(1 for row in rows if row["is_valid"])
    latencies = [row["latency_s"] for row in rows]
    return {
        "condition": condition_name,
        "num_examples": total,
        "accuracy": correct / total if total else 0.0,
        "valid_rate": valid / total if total else 0.0,
        "avg_latency_s": statistics.mean(latencies) if latencies else 0.0,
        "median_latency_s": statistics.median(latencies) if latencies else 0.0,
    }


def save_rows_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
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
    plain = payload["summary"]["plain"]
    grounded = payload["summary"]["grounded"]
    lines = [
        "# MCQA Smoke Test Report",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- Dataset: `{payload['dataset_id']}`",
        f"- Device: `{payload['device']}`",
        f"- Examples: `{payload['limit']}`",
        "",
        "## Summary",
        "",
        "| Condition | Accuracy | Valid Rate | Avg Latency (s) |",
        "|---|---:|---:|---:|",
        f"| Plain question-only | {plain['accuracy']:.3f} | {plain['valid_rate']:.3f} | {plain['avg_latency_s']:.3f} |",
        f"| Grounded with support passage | {grounded['accuracy']:.3f} | {grounded['valid_rate']:.3f} | {grounded['avg_latency_s']:.3f} |",
        "",
        f"- Accuracy delta: `{grounded['accuracy'] - plain['accuracy']:+.3f}`",
        f"- Valid rate delta: `{grounded['valid_rate'] - plain['valid_rate']:+.3f}`",
        f"- Latency delta: `{grounded['avg_latency_s'] - plain['avg_latency_s']:+.3f}` seconds",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_plot(path: Path, summaries: list[dict]) -> None:
    names = ["Plain", "Grounded"]
    accuracy = [item["accuracy"] for item in summaries]
    valid_rate = [item["valid_rate"] for item in summaries]
    latency = [item["avg_latency_s"] for item in summaries]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    colors = ["#2563eb", "#059669"]

    axes[0].bar(names, accuracy, color=colors)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Accuracy")
    axes[0].set_ylabel("Score")

    axes[1].bar(names, valid_rate, color=colors)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Valid Answer Rate")

    axes[2].bar(names, latency, color=colors)
    axes[2].set_title("Average Latency")
    axes[2].set_ylabel("Seconds")

    fig.suptitle("STEMTune MCQA Smoke Test", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_smoke_test(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or pick_device()
    examples = load_examples(limit=args.limit, seed=args.seed)
    tokenizer, model = load_generator(args.model_id, device)

    plain_rows = evaluate_condition("plain", examples, tokenizer, model, device, args.max_new_tokens)
    grounded_rows = evaluate_condition("grounded", examples, tokenizer, model, device, args.max_new_tokens)
    all_rows = plain_rows + grounded_rows

    summary = {
        "plain": summarize("plain", plain_rows),
        "grounded": summarize("grounded", grounded_rows),
    }
    payload = {
        "model_id": args.model_id,
        "dataset_id": DEFAULT_DATASET_ID,
        "device": device,
        "limit": args.limit,
        "seed": args.seed,
        "summary": summary,
    }

    save_rows_csv(output_dir / "predictions.csv", all_rows)
    save_summary_json(output_dir / "summary.json", payload)
    save_report_markdown(output_dir / "report.md", payload)
    save_plot(output_dir / "comparison.png", [summary["plain"], summary["grounded"]])
    return payload


def build_smoke_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small MCQA smoke test with grounding vs question-only prompts.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output-dir", default="artifacts/evals/smoke_mcqa")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_smoke_parser()
    args = parser.parse_args(argv)
    payload = run_smoke_test(args)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
