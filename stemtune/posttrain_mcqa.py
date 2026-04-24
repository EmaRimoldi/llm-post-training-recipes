from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from stemtune.smoke_mcqa import DEFAULT_DATASET_ID, DEFAULT_MODEL_ID, build_example, build_messages, pick_device


DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
CONTRACT_PATTERN = re.compile(
    r"^<final>\s*choice=([A-D])\s*source=question_only\s*</final>$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class TrainingRecord:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


class SupervisedDataset(Dataset):
    def __init__(self, rows: list[TrainingRecord]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        row = self.rows[index]
        return {
            "input_ids": row.input_ids,
            "attention_mask": row.attention_mask,
            "labels": row.labels,
        }


def load_split_examples(split: str, limit: int, seed: int):
    dataset = load_dataset(DEFAULT_DATASET_ID, split=split)
    subset = dataset.shuffle(seed=seed).select(range(limit))
    rng = random.Random(seed)
    return [build_example(record, rng, index) for index, record in enumerate(subset)]


def format_choices(choices: list[str]) -> str:
    return "\n".join(f"{letter}. {choice}" for letter, choice in zip("ABCD", choices))


def make_contract_prompt(example) -> str:
    return (
        "Answer the following science multiple-choice question.\n"
        "Return exactly this machine-readable format and nothing else:\n"
        "<final>\nchoice=<A|B|C|D>\nsource=question_only\n</final>\n\n"
        f"Question:\n{example.question}\n\n"
        f"Choices:\n{format_choices(example.choices)}\n"
    )


def make_contract_target(example) -> str:
    return f"<final>\nchoice={example.correct_letter}\nsource=question_only\n</final>"


def render_chat(tokenizer, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    lines = []
    for message in messages:
        role = message["role"].capitalize()
        lines.append(f"{role}: {message['content']}")
    if add_generation_prompt:
        lines.append("Assistant:")
    return "\n\n".join(lines)


def make_training_record(tokenizer, example, max_length: int) -> TrainingRecord:
    prompt = make_contract_prompt(example)
    prompt_messages = build_messages(prompt)
    full_messages = prompt_messages + [{"role": "assistant", "content": make_contract_target(example)}]

    prompt_text = render_chat(tokenizer, prompt_messages, add_generation_prompt=True)
    full_text = render_chat(tokenizer, full_messages, add_generation_prompt=False)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length)

    input_ids = list(full["input_ids"])
    attention_mask = list(full["attention_mask"])
    labels = list(input_ids)
    prompt_length = min(len(prompt_ids), len(labels) - 1)
    labels[:prompt_length] = [-100] * prompt_length
    return TrainingRecord(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


def collate_batch(tokenizer, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
    max_len = max(len(item["input_ids"]) for item in batch)
    padded = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        padded["input_ids"].append(item["input_ids"] + [tokenizer.pad_token_id] * pad_len)
        padded["attention_mask"].append(item["attention_mask"] + [0] * pad_len)
        padded["labels"].append(item["labels"] + [-100] * pad_len)
    return {key: torch.tensor(value, dtype=torch.long) for key, value in padded.items()}


def pick_training_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.float16
    return torch.float32


def load_training_model(model_id: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=pick_training_dtype(device))
    model.to(device)
    return tokenizer, model


def parse_contract(text: str) -> tuple[bool, str | None]:
    match = CONTRACT_PATTERN.search(text.strip())
    if not match:
        return False, None
    return True, match.group(1).upper()


def generate_contract(tokenizer, model, device: str, prompt: str, max_new_tokens: int) -> tuple[str, bool, str | None, float]:
    messages = build_messages(prompt)
    rendered = render_chat(tokenizer, messages, add_generation_prompt=True)
    inputs = tokenizer(rendered, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    start = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
    end = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
    if start and end:
        start.record()
    else:
        import time

        started = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    if start and end:
        end.record()
        torch.cuda.synchronize()
        latency = start.elapsed_time(end) / 1000
    else:
        latency = time.perf_counter() - started
    generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    is_valid, prediction = parse_contract(text)
    return text, is_valid, prediction, latency


def evaluate_phase(phase: str, examples, tokenizer, model, device: str, max_new_tokens: int) -> list[dict]:
    rows = []
    for example in examples:
        raw_text, is_valid, prediction, latency = generate_contract(
            tokenizer,
            model,
            device,
            make_contract_prompt(example),
            max_new_tokens,
        )
        rows.append(
            {
                "phase": phase,
                "example_id": example.example_id,
                "question": example.question,
                "correct_letter": example.correct_letter,
                "prediction": prediction,
                "is_correct": is_valid and prediction == example.correct_letter,
                "is_valid": is_valid,
                "latency_s": latency,
                "raw_output": raw_text,
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    correct = sum(1 for row in rows if row["is_correct"])
    valid = sum(1 for row in rows if row["is_valid"])
    latencies = [row["latency_s"] for row in rows]
    return {
        "num_examples": total,
        "accuracy": correct / total if total else 0.0,
        "valid_rate": valid / total if total else 0.0,
        "avg_latency_s": statistics.mean(latencies) if latencies else 0.0,
        "median_latency_s": statistics.median(latencies) if latencies else 0.0,
    }


def train_adapter(args: argparse.Namespace, tokenizer, model, train_examples):
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[item.strip() for item in args.target_modules.split(",") if item.strip()],
    )
    model = get_peft_model(model, config)
    model.train()

    dataset = SupervisedDataset([make_training_record(tokenizer, example, args.max_length) for example in train_examples])
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda rows: collate_batch(tokenizer, rows),
    )
    optimizer = AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate)

    history = []
    for epoch in range(args.epochs):
        epoch_losses = []
        optimizer.zero_grad()
        for step, batch in enumerate(dataloader, start=1):
            batch = {key: value.to(args.device_name) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            if torch.isnan(loss):
                raise RuntimeError("Training loss became NaN. Try running on CPU or lowering the learning rate.")
            loss.backward()
            epoch_losses.append(float(loss.detach().cpu()) * args.gradient_accumulation_steps)
            if step % args.gradient_accumulation_steps == 0 or step == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
        history.append({"epoch": epoch + 1, "loss_mean": statistics.mean(epoch_losses) if epoch_losses else 0.0})
    model.eval()
    return history, model


def save_rows_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "phase",
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
    baseline = payload["summary"]["baseline"]
    adapted = payload["summary"]["adapted"]
    lines = [
        "# MCQA Post-Training Smoke Test",
        "",
        "This smoke test measures contract-compliant task behavior, not just raw answer accuracy.",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- Dataset: `{payload['dataset_id']}`",
        f"- Device: `{payload['device']}`",
        f"- Train examples: `{payload['train_limit']}`",
        f"- Eval examples: `{payload['eval_limit']}`",
        "",
        "| Phase | Strict Accuracy | Contract Valid Rate | Avg Latency (s) |",
        "|---|---:|---:|---:|",
        f"| Baseline | {baseline['accuracy']:.3f} | {baseline['valid_rate']:.3f} | {baseline['avg_latency_s']:.3f} |",
        f"| After tiny LoRA post-training | {adapted['accuracy']:.3f} | {adapted['valid_rate']:.3f} | {adapted['avg_latency_s']:.3f} |",
        "",
        f"- Strict accuracy gain: `{adapted['accuracy'] - baseline['accuracy']:+.3f}`",
        f"- Contract-valid gain: `{adapted['valid_rate'] - baseline['valid_rate']:+.3f}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_plot(path: Path, payload: dict) -> None:
    baseline = payload["summary"]["baseline"]
    adapted = payload["summary"]["adapted"]
    history = payload["training_history"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("#f8fafc")
    colors = ["#1d4ed8", "#0f766e"]

    axes[0].bar(["Baseline", "Post-trained"], [baseline["accuracy"], adapted["accuracy"]], color=colors)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Strict Accuracy", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Accuracy")

    axes[1].bar(["Baseline", "Post-trained"], [baseline["valid_rate"], adapted["valid_rate"]], color=colors)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Contract Valid Rate", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Valid outputs")

    axes[2].plot([item["epoch"] for item in history], [item["loss_mean"] for item in history], marker="o", color="#b45309")
    axes[2].set_title("Training Loss", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_facecolor("#f8fafc")

    fig.suptitle("STEMTune MCQA Post-Training Smoke Test", fontsize=16, fontweight="bold", color="#0f172a")
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def run_posttrain_smoke(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or pick_device()
    if device == "mps":
        device = "cpu"
    args.device_name = device
    train_examples = load_split_examples("train", args.train_limit, args.seed)
    eval_examples = load_split_examples("validation", args.eval_limit, args.seed + 101)

    tokenizer, model = load_training_model(args.model_id, device)
    baseline_rows = evaluate_phase("baseline", eval_examples, tokenizer, model, device, args.max_new_tokens)
    training_history, adapted_model = train_adapter(args, tokenizer, model, train_examples)
    adapted_rows = evaluate_phase("adapted", eval_examples, tokenizer, adapted_model, device, args.max_new_tokens)

    payload = {
        "model_id": args.model_id,
        "dataset_id": DEFAULT_DATASET_ID,
        "device": device,
        "train_limit": args.train_limit,
        "eval_limit": args.eval_limit,
        "seed": args.seed,
        "training_history": training_history,
        "summary": {
            "baseline": summarize(baseline_rows),
            "adapted": summarize(adapted_rows),
        },
    }

    all_rows = baseline_rows + adapted_rows
    save_rows_csv(output_dir / "predictions.csv", all_rows)
    save_summary_json(output_dir / "summary.json", payload)
    save_report_markdown(output_dir / "report.md", payload)
    save_plot(output_dir / "comparison.png", payload)
    if args.save_adapter:
        adapted_model.save_pretrained(output_dir / "adapter")
        tokenizer.save_pretrained(output_dir / "adapter")
    return payload


def build_posttrain_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a tiny LoRA post-training smoke test on SciQ MCQA.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--train-limit", type=int, default=32)
    parser.add_argument("--eval-limit", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default=",".join(DEFAULT_TARGET_MODULES))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-dir", default="docs/results/mcqa_posttrain_smoke")
    parser.add_argument("--save-adapter", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_posttrain_parser()
    args = parser.parse_args(argv)
    payload = run_posttrain_smoke(args)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
