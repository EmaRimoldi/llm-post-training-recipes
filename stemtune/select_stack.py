#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stemtune.scaffold import TASK_BLUEPRINTS, create_project_scaffold
from stemtune.smoke_mcqa import run_smoke_test


CATALOG_PATH = Path(__file__).with_name("model_catalog.json")


TASK_GUIDE = {
    "sft": {
        "summary": "Supervised fine-tuning for domain adaptation from labeled QA data.",
        "when_to_use": [
            "You have supervised question-answer pairs or instruction-response examples.",
            "You want the simplest reliable baseline before adding preference optimization.",
            "You need to adapt a base or instruct model to a specific STEM domain.",
        ],
        "recipes": ["training/sft/m2", "training/sft/m3"],
        "next_steps": [
            "Prepare task data under datasets/external/ or datasets/builders/.",
            "Start with a LoRA-style SFT run on the recommended base or instruct model.",
            "Promote the result to MCQA or DPO only after SFT quality is acceptable.",
        ],
    },
    "mcqa": {
        "summary": "Multiple-choice adaptation for answer-accuracy driven STEM tasks.",
        "when_to_use": [
            "The output space is a fixed set of candidate answers.",
            "The main metric is answer accuracy rather than open-ended generation quality.",
            "You want a compact path from domain data to benchmarkable QA behavior.",
        ],
        "recipes": ["training/mcqa", "datasets/builders/mcqa"],
        "next_steps": [
            "Normalize choices and answer indices with datasets/builders/mcqa/.",
            "Fine-tune on compact multiple-choice prompts before introducing preference optimization.",
            "Use this path when the target metric is answer accuracy rather than open-ended style.",
        ],
    },
    "dpo": {
        "summary": "Preference optimization on chosen/rejected pairs after SFT.",
        "when_to_use": [
            "You already have a usable SFT or instruct baseline.",
            "You can build chosen/rejected preference pairs.",
            "You want ranking-style alignment gains instead of raw next-token learning only.",
        ],
        "recipes": ["training/dpo", "datasets/builders/dpo"],
        "next_steps": [
            "Start from an already reasonable SFT or instruct baseline.",
            "Build chosen/rejected preference pairs with datasets/builders/dpo/.",
            "Use DPO after behavior is mostly correct and you want ranking-style alignment gains.",
        ],
    },
    "quantization": {
        "summary": "Compression and QLoRA workflows for cheaper local serving.",
        "when_to_use": [
            "The aligned model quality is good enough but serving cost is too high.",
            "You want local deployment on smaller hardware.",
            "You need a calibrated compression path after establishing a stronger reference model.",
        ],
        "recipes": ["training/quantization/m2", "training/quantization/m3"],
        "next_steps": [
            "Pick the smallest model that still solves the task before compressing.",
            "Use calibration data from datasets/calibration/ for reproducible experiments.",
            "Treat quantization as a deployment step after you establish a strong reference model.",
        ],
    },
    "rag": {
        "summary": "Retrieval-aware training when external documents matter.",
        "when_to_use": [
            "The task depends on external documents or changing knowledge.",
            "Grounding matters more than pure parametric memorization.",
            "You want better knowledge freshness and evidence-aware behavior.",
        ],
        "recipes": ["training/rag", "retrieval/knowledge_base"],
        "next_steps": [
            "Prepare a retrieval corpus first in retrieval/knowledge_base/.",
            "Use RAG when knowledge freshness or document grounding matters more than pure parametric memorization.",
            "Prefer longer-context or tool-friendly instruct models if the serving stack is retrieval-heavy.",
        ],
    },
}


def load_catalog() -> list[dict]:
    with CATALOG_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def has_long_context(model: dict) -> bool:
    strengths = set(model["strengths"])
    return "long_context" in strengths or "128k_context" in strengths


def score_model(model: dict, args: argparse.Namespace) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []

    if args.gpu_memory_gb >= model["min_vram_gb"]:
        score += 3
        reasons.append(f"fits the declared VRAM tier (needs about {model['min_vram_gb']}GB or more)")
    else:
        score -= 100
        reasons.append(f"does not fit the declared VRAM tier (needs about {model['min_vram_gb']}GB or more)")

    if args.task in model["good_for"]:
        score += 4
        reasons.append(f"is a good match for `{args.task}`")

    if args.task == "rag" and has_long_context(model):
        score += 2
        reasons.append("has long-context value for retrieval-heavy setups")

    if args.prefer_long_context and has_long_context(model):
        score += 2
        reasons.append("matches the long-context preference")

    if args.prefer_tool_use and "tool_use" in model["strengths"]:
        score += 2
        reasons.append("matches the tool-use preference")

    if args.prefer_multilingual and "multilingual" in model["strengths"]:
        score += 2
        reasons.append("matches the multilingual preference")

    if args.gpu_memory_gb <= 16 and model["params_b"] <= 1:
        score += 2
        reasons.append("is sized for cheap experimentation")

    if args.gpu_memory_gb >= 24 and 4 <= model["params_b"] <= 8 and args.task in {"sft", "mcqa", "dpo"}:
        score += 2
        reasons.append("sits in the most practical post-training tier for a single strong GPU")

    if args.task == "quantization" and model["params_b"] <= 8:
        score += 2
        reasons.append("is a practical candidate for compression experiments")

    if args.task == "rag" and args.gpu_memory_gb >= 48 and model["params_b"] >= 20:
        score += 2
        reasons.append("uses the available memory budget for a stronger retrieval-oriented model")

    return score, reasons


def rank_models(args: argparse.Namespace) -> list[tuple[int, dict, list[str]]]:
    catalog = load_catalog()
    ranked = []
    for model in catalog:
        score, reasons = score_model(model, args)
        ranked.append((score, model, reasons))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def recommend(args: argparse.Namespace) -> dict:
    ranked = rank_models(args)
    best_score, best_model, reasons = ranked[0]
    return {
        "task": args.task,
        "gpu_memory_gb": args.gpu_memory_gb,
        "recommended_model": best_model,
        "score": best_score,
        "why": reasons,
        "task_guide": TASK_GUIDE[args.task],
        "alternatives": [
            {
                "id": model["id"],
                "family": model["family"],
                "score": score,
            }
            for score, model, _ in ranked[1:3]
            if score > -50
        ],
    }


def list_models(args: argparse.Namespace) -> dict:
    ranked = rank_models(args) if getattr(args, "task", None) else None
    models = []
    if ranked is None:
        for model in load_catalog():
            models.append(model)
    else:
        for score, model, reasons in ranked:
            enriched = dict(model)
            enriched["score"] = score
            enriched["why"] = reasons
            models.append(enriched)
    return {
        "task": getattr(args, "task", None),
        "gpu_memory_gb": getattr(args, "gpu_memory_gb", None),
        "models": models,
    }


def task_details(task: str) -> dict:
    guide = TASK_GUIDE[task]
    return {
        "task": task,
        "summary": guide["summary"],
        "when_to_use": guide["when_to_use"],
        "recipes": guide["recipes"],
        "next_steps": guide["next_steps"],
    }


def all_tasks() -> dict:
    return {
        "tasks": [
            {"task": name, "summary": guide["summary"]}
            for name, guide in TASK_GUIDE.items()
        ]
    }


def render_text(result: dict) -> str:
    model = result["recommended_model"]
    guide = result["task_guide"]
    lines = [
        "STEMTune recommendation",
        f"Task: {result['task']}",
        f"GPU memory budget: {result['gpu_memory_gb']}GB",
        "",
        f"Recommended model: {model['id']}",
        f"Family: {model['family']}",
        f"Variant: {model['variant']}",
        f"Why:",
    ]
    lines.extend([f"- {reason}" for reason in result["why"]])
    lines.append("")
    lines.append("Start here:")
    lines.extend([f"- {recipe}" for recipe in guide["recipes"]])
    lines.append("")
    lines.append("Next steps:")
    lines.extend([f"- {step}" for step in guide["next_steps"]])
    if result["alternatives"]:
        lines.append("")
        lines.append("Alternatives:")
        lines.extend([f"- {item['id']} ({item['score']})" for item in result["alternatives"]])
    lines.append("")
    lines.append(f"Reference: {model['reference_url']}")
    return "\n".join(lines)


def render_model_list(result: dict) -> str:
    lines = ["STEMTune model catalog"]
    if result["task"] and result["gpu_memory_gb"] is not None:
        lines.append(f"Ranked for task `{result['task']}` with {result['gpu_memory_gb']}GB")
    else:
        lines.append("Unranked catalog")
    lines.append("")
    for model in result["models"]:
        line = f"- {model['id']} | family={model['family']} | min_vram={model['min_vram_gb']}GB"
        if "score" in model:
            line += f" | score={model['score']}"
        lines.append(line)
        if "why" in model:
            lines.append(f"  why: {', '.join(model['why'][:2])}")
    return "\n".join(lines)


def render_task(result: dict) -> str:
    lines = [
        f"STEMTune task guide: {result['task']}",
        result["summary"],
        "",
        "Use when:",
    ]
    lines.extend([f"- {item}" for item in result["when_to_use"]])
    lines.append("")
    lines.append("Start here:")
    lines.extend([f"- {item}" for item in result["recipes"]])
    lines.append("")
    lines.append("Next steps:")
    lines.extend([f"- {item}" for item in result["next_steps"]])
    return "\n".join(lines)


def render_tasks(result: dict) -> str:
    lines = ["STEMTune tasks", ""]
    lines.extend([f"- {item['task']}: {item['summary']}" for item in result["tasks"]])
    return "\n".join(lines)


def build_recommend_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "recommend",
        help="Recommend a model and recipe path for a task and hardware budget.",
    )
    parser.add_argument("--task", choices=sorted(TASK_GUIDE.keys()), required=True)
    parser.add_argument("--gpu-memory-gb", type=int, required=True)
    parser.add_argument("--prefer-long-context", action="store_true")
    parser.add_argument("--prefer-tool-use", action="store_true")
    parser.add_argument("--prefer-multilingual", action="store_true")
    parser.add_argument("--output", choices=["text", "json"], default="text")
    parser.set_defaults(handler=handle_recommend)


def build_list_models_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "list-models",
        help="List the curated open-source models, optionally ranked for a task.",
    )
    parser.add_argument("--task", choices=sorted(TASK_GUIDE.keys()))
    parser.add_argument("--gpu-memory-gb", type=int)
    parser.add_argument("--prefer-long-context", action="store_true")
    parser.add_argument("--prefer-tool-use", action="store_true")
    parser.add_argument("--prefer-multilingual", action="store_true")
    parser.add_argument("--output", choices=["text", "json"], default="text")
    parser.set_defaults(handler=handle_list_models)


def build_show_task_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "show-task",
        help="Explain when to use a task recipe and where to start in the repo.",
    )
    parser.add_argument("task", choices=sorted(TASK_GUIDE.keys()))
    parser.add_argument("--output", choices=["text", "json"], default="text")
    parser.set_defaults(handler=handle_show_task)


def build_list_tasks_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "list-tasks",
        help="List the alignment tasks covered by STEMTune.",
    )
    parser.add_argument("--output", choices=["text", "json"], default="text")
    parser.set_defaults(handler=handle_list_tasks)


def build_init_project_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "init-project",
        help="Generate a project scaffold with configs, manifests, evaluation gates, and runbook.",
    )
    parser.add_argument("--name", required=True, help="Human-readable project name.")
    parser.add_argument("--task", choices=sorted(TASK_BLUEPRINTS.keys()), required=True)
    parser.add_argument("--base-model", required=True, help="Any open-source base or instruct model ID.")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where the new project folder should be created. Defaults to the current directory.",
    )
    parser.add_argument("--hf-namespace", help="Your own Hugging Face username or organization.")
    parser.add_argument("--model-repo-name", help="Optional explicit model repo name.")
    parser.add_argument("--dataset-repo-name", help="Optional explicit dataset repo name.")
    parser.add_argument("--kb-repo-name", help="Optional explicit knowledge-base repo name.")
    parser.add_argument("--private-hub-repos", action="store_true")
    parser.add_argument("--force", action="store_true", help="Allow writing into an existing non-empty directory.")
    parser.add_argument("--output", choices=["text", "json"], default="text")
    parser.set_defaults(handler=handle_init_project)


def build_smoke_mcqa_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "smoke-mcqa",
        help="Run a small MCQA smoke test and compare question-only vs grounded prompts.",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output-dir", default="artifacts/evals/smoke_mcqa")
    parser.add_argument("--output", choices=["text", "json"], default="text")
    parser.set_defaults(handler=handle_smoke_mcqa)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select open-source models and alignment recipes for STEMTune tasks."
    )
    subparsers = parser.add_subparsers(dest="command")
    build_recommend_parser(subparsers)
    build_list_models_parser(subparsers)
    build_show_task_parser(subparsers)
    build_list_tasks_parser(subparsers)
    build_init_project_parser(subparsers)
    build_smoke_mcqa_parser(subparsers)
    return parser


def handle_recommend(args: argparse.Namespace) -> str:
    result = recommend(args)
    if args.output == "json":
        return json.dumps(result, indent=2)
    return render_text(result)


def handle_list_models(args: argparse.Namespace) -> str:
    if args.task and args.gpu_memory_gb is None:
        raise SystemExit("--gpu-memory-gb is required when ranking models for a task.")
    result = list_models(args)
    if args.output == "json":
        return json.dumps(result, indent=2)
    return render_model_list(result)


def handle_show_task(args: argparse.Namespace) -> str:
    result = task_details(args.task)
    if args.output == "json":
        return json.dumps(result, indent=2)
    return render_task(result)


def handle_list_tasks(args: argparse.Namespace) -> str:
    result = all_tasks()
    if args.output == "json":
        return json.dumps(result, indent=2)
    return render_tasks(result)


def handle_init_project(args: argparse.Namespace) -> str:
    target_dir, files_written = create_project_scaffold(args)
    result = {
        "project_dir": str(target_dir),
        "files_written": files_written,
        "task": args.task,
        "base_model": args.base_model,
    }
    if args.output == "json":
        return json.dumps(result, indent=2)

    lines = [
        "STEMTune project scaffold created",
        f"Project directory: {target_dir}",
        f"Task: {args.task}",
        f"Base model: {args.base_model}",
        "",
        "Generated files:",
    ]
    lines.extend([f"- {path}" for path in files_written])
    lines.extend(
        [
            "",
            "Next steps:",
            "- Edit configs/dataset.json with your own source data.",
            "- Edit configs/training.json with your training settings and model choices.",
            "- Edit configs/evaluation.json with the metrics and promotion thresholds you care about.",
            "- Edit configs/publish.json and .env.example with your own Hugging Face namespace.",
            "- Put project-specific conversion scripts under scripts/.",
        ]
    )
    return "\n".join(lines)


def handle_smoke_mcqa(args: argparse.Namespace) -> str:
    payload = run_smoke_test(args)
    if args.output == "json":
        return json.dumps(payload, indent=2)

    plain = payload["summary"]["plain"]
    grounded = payload["summary"]["grounded"]
    lines = [
        "STEMTune MCQA smoke test",
        f"Model: {payload['model_id']}",
        f"Device: {payload['device']}",
        f"Examples: {payload['limit']}",
        "",
        f"Plain accuracy: {plain['accuracy']:.3f}",
        f"Grounded accuracy: {grounded['accuracy']:.3f}",
        f"Accuracy delta: {grounded['accuracy'] - plain['accuracy']:+.3f}",
        f"Plain avg latency: {plain['avg_latency_s']:.3f}s",
        f"Grounded avg latency: {grounded['avg_latency_s']:.3f}s",
        "",
        f"Artifacts: {args.output_dir}",
    ]
    return "\n".join(lines)


def normalize_argv(argv: list[str]) -> list[str]:
    commands = {"recommend", "list-models", "show-task", "list-tasks", "init-project", "smoke-mcqa", "-h", "--help"}
    if not argv:
        return argv
    if argv[0] in commands:
        return argv
    return ["recommend", *argv]


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    normalized_argv = normalize_argv(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(normalized_argv)

    if not hasattr(args, "handler"):
        parser.print_help()
    else:
        print(args.handler(args))


if __name__ == "__main__":
    main()
