from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from stemtune.posttrain_mcqa import (
    DEFAULT_TARGET_MODULES,
    build_messages,
    evaluate_phase,
    load_split_examples,
    make_contract_prompt,
    make_contract_target,
    render_chat,
    summarize,
)
from stemtune.smoke_mcqa import DEFAULT_DATASET_ID, DEFAULT_MODEL_ID, pick_device


@dataclass
class PreferenceRecord:
    prompt_ids: list[int]
    chosen_ids: list[int]
    rejected_ids: list[int]


class PreferenceDataset(Dataset):
    def __init__(self, rows: list[PreferenceRecord]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        row = self.rows[index]
        return {
            "prompt_ids": row.prompt_ids,
            "chosen_ids": row.chosen_ids,
            "rejected_ids": row.rejected_ids,
        }


def make_rejected_target(example) -> str:
    return example.correct_letter


def build_preference_record(tokenizer, example, max_length: int) -> PreferenceRecord:
    prompt = make_contract_prompt(example)
    prompt_ids = tokenizer(render_chat(tokenizer, build_messages(prompt), add_generation_prompt=True), add_special_tokens=False)["input_ids"]
    chosen_ids = tokenizer(make_contract_target(example), add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
    rejected_ids = tokenizer(make_rejected_target(example), add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
    return PreferenceRecord(prompt_ids=prompt_ids, chosen_ids=chosen_ids, rejected_ids=rejected_ids)


def collate_preferences(tokenizer, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
    def pad(sequences: list[list[int]], pad_value: int) -> torch.Tensor:
        max_len = max(len(sequence) for sequence in sequences)
        return torch.tensor([sequence + [pad_value] * (max_len - len(sequence)) for sequence in sequences], dtype=torch.long)

    return {
        "prompt_ids": pad([item["prompt_ids"] for item in batch], tokenizer.pad_token_id),
        "chosen_ids": pad([item["chosen_ids"] for item in batch], tokenizer.pad_token_id),
        "rejected_ids": pad([item["rejected_ids"] for item in batch], tokenizer.pad_token_id),
    }


def load_model_pair(model_id: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    ref = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    policy.to(device)
    ref.to(device)
    ref.eval()
    for parameter in ref.parameters():
        parameter.requires_grad = False
    return tokenizer, policy, ref


def sequence_logprob(model, prompt_ids: torch.Tensor, continuation_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    input_ids = torch.cat([prompt_ids, continuation_ids], dim=1)
    attention_mask = (input_ids != pad_token_id).long()
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    labels = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    prompt_len = prompt_ids.size(1)
    continuation_mask = (labels != pad_token_id).float()
    continuation_mask[:, : prompt_len - 1] = 0.0
    return (token_log_probs * continuation_mask).sum(dim=1)


def train_dpo(args: argparse.Namespace, tokenizer, policy, ref, train_examples):
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[item.strip() for item in args.target_modules.split(",") if item.strip()],
    )
    policy = get_peft_model(policy, config)
    policy.train()

    dataset = PreferenceDataset([build_preference_record(tokenizer, example, args.max_length) for example in train_examples])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda rows: collate_preferences(tokenizer, rows))
    optimizer = AdamW((parameter for parameter in policy.parameters() if parameter.requires_grad), lr=args.learning_rate)

    history = []
    beta = args.beta
    for epoch in range(args.epochs):
        losses = []
        optimizer.zero_grad()
        for step, batch in enumerate(dataloader, start=1):
            batch = {key: value.to(args.device_name) for key, value in batch.items()}
            chosen_logps = sequence_logprob(policy, batch["prompt_ids"], batch["chosen_ids"], tokenizer.pad_token_id)
            rejected_logps = sequence_logprob(policy, batch["prompt_ids"], batch["rejected_ids"], tokenizer.pad_token_id)
            with torch.no_grad():
                ref_chosen_logps = sequence_logprob(ref, batch["prompt_ids"], batch["chosen_ids"], tokenizer.pad_token_id)
                ref_rejected_logps = sequence_logprob(ref, batch["prompt_ids"], batch["rejected_ids"], tokenizer.pad_token_id)
            logits = beta * ((chosen_logps - rejected_logps) - (ref_chosen_logps - ref_rejected_logps))
            loss = (-F.logsigmoid(logits).mean()) / args.gradient_accumulation_steps
            loss.backward()
            losses.append(float(loss.detach().cpu()) * args.gradient_accumulation_steps)
            if step % args.gradient_accumulation_steps == 0 or step == len(dataloader):
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
        history.append({"epoch": epoch + 1, "loss_mean": statistics.mean(losses) if losses else 0.0})
    policy.eval()
    return history, policy


def save_predictions_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "phase",
        "example_id",
        "question",
        "correct_letter",
        "prediction",
        "letter_prediction",
        "letter_correct",
        "is_correct",
        "is_valid",
        "weighted_score",
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
        "# MCQA DPO Smoke Test",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- Dataset: `{payload['dataset_id']}`",
        f"- Device: `{payload['device']}`",
        f"- Train examples: `{payload['train_limit']}`",
        f"- Eval examples: `{payload['eval_limit']}`",
        "",
        "| Phase | Letter Accuracy | Contract Valid Rate | Strict Accuracy | Weighted Score |",
        "|---|---:|---:|---:|---:|",
        f"| Baseline | {baseline['letter_accuracy']:.3f} | {baseline['valid_rate']:.3f} | {baseline['strict_accuracy']:.3f} | {baseline['weighted_score']:.3f} |",
        f"| After tiny DPO | {adapted['letter_accuracy']:.3f} | {adapted['valid_rate']:.3f} | {adapted['strict_accuracy']:.3f} | {adapted['weighted_score']:.3f} |",
        "",
        f"- Letter accuracy gain: `{adapted['letter_accuracy'] - baseline['letter_accuracy']:+.3f}`",
        f"- Contract-valid gain: `{adapted['valid_rate'] - baseline['valid_rate']:+.3f}`",
        f"- Weighted score gain: `{adapted['weighted_score'] - baseline['weighted_score']:+.3f}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_plot(path: Path, payload: dict) -> None:
    baseline = payload["summary"]["baseline"]
    adapted = payload["summary"]["adapted"]
    history = payload["training_history"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("#f8fafc")
    colors = ["#1d4ed8", "#0f766e"]

    axes[0].bar(["Baseline", "DPO"], [baseline["letter_accuracy"], adapted["letter_accuracy"]], color=colors)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Letter Accuracy", fontsize=13, fontweight="bold")

    axes[1].bar(["Baseline", "DPO"], [baseline["valid_rate"], adapted["valid_rate"]], color=colors)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Contract Valid Rate", fontsize=13, fontweight="bold")

    axes[2].bar(["Baseline", "DPO"], [baseline["weighted_score"], adapted["weighted_score"]], color=colors)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Weighted Score", fontsize=13, fontweight="bold")

    axes[3].plot([item["epoch"] for item in history], [item["loss_mean"] for item in history], marker="o", color="#b45309")
    axes[3].set_title("DPO Loss", fontsize=13, fontweight="bold")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_facecolor("#f8fafc")

    fig.suptitle("STEMTune MCQA DPO Smoke Test", fontsize=16, fontweight="bold", color="#0f172a")
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def run_dpo_smoke(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or pick_device()
    if device == "mps":
        device = "cpu"
    args.device_name = device

    train_examples = load_split_examples("train", args.train_limit, args.seed)
    eval_examples = load_split_examples("validation", args.eval_limit, args.seed + 101)

    tokenizer, policy, ref = load_model_pair(args.model_id, device)
    baseline_rows = evaluate_phase("baseline", eval_examples, tokenizer, policy, device, args.max_new_tokens)
    history, adapted_policy = train_dpo(args, tokenizer, policy, ref, train_examples)
    adapted_rows = evaluate_phase("adapted", eval_examples, tokenizer, adapted_policy, device, args.max_new_tokens)

    payload = {
        "model_id": args.model_id,
        "dataset_id": DEFAULT_DATASET_ID,
        "device": device,
        "train_limit": args.train_limit,
        "eval_limit": args.eval_limit,
        "seed": args.seed,
        "training_history": history,
        "summary": {
            "baseline": summarize(baseline_rows),
            "adapted": summarize(adapted_rows),
        },
    }
    save_predictions_csv(output_dir / "predictions.csv", baseline_rows + adapted_rows)
    save_summary_json(output_dir / "summary.json", payload)
    save_report_markdown(output_dir / "report.md", payload)
    save_plot(output_dir / "comparison.png", payload)
    return payload


def build_dpo_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a tiny DPO smoke test on SciQ MCQA preferences.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--train-limit", type=int, default=32)
    parser.add_argument("--eval-limit", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default=",".join(DEFAULT_TARGET_MODULES))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-dir", default="docs/results/mcqa_dpo_smoke")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_dpo_parser()
    args = parser.parse_args(argv)
    payload = run_dpo_smoke(args)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
