from __future__ import annotations

import json
import re
from pathlib import Path


TASK_BLUEPRINTS = {
    "sft": {
        "summary": "Supervised fine-tuning from labeled instruction or QA pairs.",
        "target_output": "A task-adapted instruct or base model checkpoint.",
        "default_record_schema": {
            "prompt": "prompt",
            "response": "response",
            "system": "system",
        },
        "knowledge_base_enabled": False,
        "recipe_paths": ["training/sft/m2", "training/sft/m3"],
    },
    "mcqa": {
        "summary": "Multiple-choice adaptation for answer-accuracy driven tasks.",
        "target_output": "A model specialized for fixed-choice question answering.",
        "default_record_schema": {
            "question": "question",
            "choices": "choices",
            "answer": "answer",
            "explanation": "explanation",
        },
        "knowledge_base_enabled": False,
        "recipe_paths": ["training/mcqa", "datasets/builders/mcqa"],
    },
    "dpo": {
        "summary": "Preference optimization from chosen/rejected pairs.",
        "target_output": "A preference-aligned model checkpoint.",
        "default_record_schema": {
            "prompt": "prompt",
            "chosen": "chosen",
            "rejected": "rejected",
        },
        "knowledge_base_enabled": False,
        "recipe_paths": ["training/dpo", "datasets/builders/dpo"],
    },
    "quantization": {
        "summary": "Compression and adapter-based low-memory deployment workflows.",
        "target_output": "A quantized model artifact and calibration manifest.",
        "default_record_schema": {
            "prompt": "prompt",
            "response": "response",
        },
        "knowledge_base_enabled": False,
        "recipe_paths": ["training/quantization/m2", "training/quantization/m3"],
    },
    "rag": {
        "summary": "Retrieval-aware training and grounded answer generation.",
        "target_output": "A model and document index that can serve grounded responses.",
        "default_record_schema": {
            "question": "question",
            "context": "context",
            "answer": "answer",
        },
        "knowledge_base_enabled": True,
        "recipe_paths": ["training/rag", "retrieval/knowledge_base"],
    },
}


PROJECT_DIRS = [
    "data/raw",
    "data/interim",
    "data/processed",
    "knowledge_base/raw",
    "knowledge_base/processed",
    "knowledge_base/index",
    "configs",
    "artifacts/models",
    "artifacts/evals",
    "scripts",
]


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "stemtune-project"


def default_repo_name(slug: str, suffix: str) -> str:
    return f"{slug}-{suffix}"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    ensure_parent(path)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def touch_keep(path: Path) -> None:
    ensure_parent(path)
    path.write_text("", encoding="utf-8")


def build_project_spec(args) -> dict:
    task = TASK_BLUEPRINTS[args.task]
    slug = slugify(args.name)
    hf_namespace = args.hf_namespace or "your-hf-namespace"
    model_repo = args.model_repo_name or default_repo_name(slug, "model")
    dataset_repo = args.dataset_repo_name or default_repo_name(slug, "dataset")
    kb_repo = args.kb_repo_name or default_repo_name(slug, "kb")

    return {
        "schema_version": 1,
        "project": {
            "name": args.name,
            "slug": slug,
            "task": args.task,
            "summary": task["summary"],
            "base_model": args.base_model,
            "target_output": task["target_output"],
        },
        "paths": {
            "raw_data_dir": "data/raw",
            "interim_data_dir": "data/interim",
            "processed_data_dir": "data/processed",
            "knowledge_base_raw_dir": "knowledge_base/raw",
            "knowledge_base_processed_dir": "knowledge_base/processed",
            "knowledge_base_index_dir": "knowledge_base/index",
            "artifacts_model_dir": "artifacts/models",
            "artifacts_eval_dir": "artifacts/evals",
        },
        "huggingface": {
            "namespace": hf_namespace,
            "model_repo_id": f"{hf_namespace}/{model_repo}",
            "dataset_repo_id": f"{hf_namespace}/{dataset_repo}",
            "knowledge_base_repo_id": f"{hf_namespace}/{kb_repo}",
            "private_repos": args.private_hub_repos,
        },
        "recipes": {
            "recommended_paths": task["recipe_paths"],
        },
    }


def build_dataset_config(spec: dict) -> dict:
    task = TASK_BLUEPRINTS[spec["project"]["task"]]
    return {
        "project_slug": spec["project"]["slug"],
        "task": spec["project"]["task"],
        "input_sources": [
            {
                "name": "replace-me",
                "source_type": "local_jsonl",
                "path": "data/raw/train.jsonl",
                "split": "train",
            }
        ],
        "record_schema": task["default_record_schema"],
        "validation_source": {
            "path": "data/raw/validation.jsonl",
            "optional": True,
        },
        "output_dataset_path": "data/processed/train.jsonl",
        "notes": [
            "Replace the input source definitions with your own datasets or Hub datasets.",
            "Normalize all fields here before adapting the training recipes.",
        ],
    }


def build_knowledge_base_config(spec: dict) -> dict:
    enabled = TASK_BLUEPRINTS[spec["project"]["task"]]["knowledge_base_enabled"]
    return {
        "project_slug": spec["project"]["slug"],
        "enabled": enabled,
        "source_documents": [
            {
                "name": "replace-me",
                "source_type": "local_files",
                "path": "knowledge_base/raw",
                "glob": "**/*",
            }
        ],
        "document_schema": {
            "id": "doc_id",
            "title": "title",
            "text": "text",
            "source_url": "source_url",
        },
        "chunking": {
            "strategy": "recursive",
            "chunk_size": 1024,
            "chunk_overlap": 128,
        },
        "index_output_dir": "knowledge_base/index",
        "hub_repo_id": spec["huggingface"]["knowledge_base_repo_id"],
    }


def build_training_config(spec: dict) -> dict:
    task = spec["project"]["task"]
    return {
        "project_slug": spec["project"]["slug"],
        "task": task,
        "base_model": spec["project"]["base_model"],
        "dataset_path": "data/processed/train.jsonl",
        "validation_path": "data/processed/validation.jsonl",
        "output_dir": "artifacts/models",
        "training": {
            "strategy": "lora",
            "learning_rate": 2e-4,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "num_train_epochs": 3,
            "bf16": True,
        },
        "publishing": {
            "push_to_hub": True,
            "model_repo_id": spec["huggingface"]["model_repo_id"],
        },
        "recipe_entrypoints": TASK_BLUEPRINTS[task]["recipe_paths"],
    }


def build_publish_config(spec: dict) -> dict:
    return {
        "project_slug": spec["project"]["slug"],
        "namespace": spec["huggingface"]["namespace"],
        "private_repos": spec["huggingface"]["private_repos"],
        "resources": {
            "model_repo_id": spec["huggingface"]["model_repo_id"],
            "dataset_repo_id": spec["huggingface"]["dataset_repo_id"],
            "knowledge_base_repo_id": spec["huggingface"]["knowledge_base_repo_id"],
        },
        "required_env": [
            "HF_TOKEN",
            "HF_USERNAME",
            "HF_MODEL_REPO_ID",
            "HF_DATASET_REPO_ID",
            "HF_KB_REPO_ID",
        ],
    }


def build_evaluation_config(spec: dict) -> dict:
    task = spec["project"]["task"]
    base_metrics = {
        "sft": ["exact_match", "rougeL", "manual_review"],
        "mcqa": ["accuracy", "macro_f1"],
        "dpo": ["win_rate", "judge_preference_rate", "manual_review"],
        "quantization": ["accuracy_delta", "latency", "memory_footprint"],
        "rag": ["answer_faithfulness", "context_recall", "answer_accuracy"],
    }
    return {
        "project_slug": spec["project"]["slug"],
        "task": task,
        "eval_split_path": "data/processed/validation.jsonl",
        "metrics": base_metrics[task],
        "promotion_gates": {
            "minimum_metrics": {
                metric: "set-me"
                for metric in base_metrics[task]
            },
            "notes": [
                "Use promotion gates to decide whether the current model is good enough to publish or compress.",
                "This file is meant to become the control point between training stages.",
            ],
        },
        "artifacts": {
            "predictions_path": "artifacts/evals/predictions.jsonl",
            "report_path": "artifacts/evals/report.json",
        },
    }


def render_project_readme(spec: dict) -> str:
    task = spec["project"]["task"]
    paths = spec["recipes"]["recommended_paths"]
    lines = [
        f"# {spec['project']['name']}",
        "",
        f"Task: `{task}`",
        f"Base model: `{spec['project']['base_model']}`",
        "",
        spec["project"]["summary"],
        "",
        "## Folder Intent",
        "",
        "- `data/raw/`: unmodified source data you own or downloaded yourself.",
        "- `data/interim/`: normalized intermediate artifacts before training.",
        "- `data/processed/`: final train/eval assets consumed by training scripts.",
        "- `knowledge_base/`: raw documents, processed chunks, and retrieval indexes.",
        "- `configs/`: user-owned manifests for data, KB, training, evaluation, and publishing.",
        "- `artifacts/`: models and evaluation outputs produced by your runs.",
        "- `scripts/`: your custom conversion or orchestration scripts.",
        "",
        "## Start Here",
        "",
        "1. Edit `configs/dataset.json` with your source data layout.",
        "2. If retrieval matters, edit `configs/knowledge_base.json` with your document sources.",
        "3. Edit `configs/training.json` with your base model and training parameters.",
        "4. Edit `configs/evaluation.json` with your quality gates before publishing or compressing.",
        "5. Edit `configs/publish.json` and `.env.example` with your own Hub namespace.",
        "6. Reuse the following recipe folders from STEMTune:",
    ]
    lines.extend([f"- `{path}`" for path in paths])
    return "\n".join(lines) + "\n"


def render_scripts_readme() -> str:
    return "\n".join(
        [
            "# Custom Scripts",
            "",
            "Put project-specific ingestion, normalization, or orchestration scripts here.",
            "",
            "Typical additions:",
            "",
            "- `prepare_dataset.py`: convert raw source records to `data/processed/`.",
            "- `build_kb.py`: chunk and index your document corpus.",
            "- `run_training.sh`: wrap the training recipe you chose from STEMTune.",
            "- `evaluate_model.py`: compute metrics defined in `configs/evaluation.json`.",
        ]
    ) + "\n"


def render_env_example(spec: dict) -> str:
    namespace = spec["huggingface"]["namespace"]
    return "\n".join(
        [
            f"HF_USERNAME={namespace}",
            "HF_TOKEN=replace-me",
            f"HF_MODEL_REPO_ID={spec['huggingface']['model_repo_id']}",
            f"HF_DATASET_REPO_ID={spec['huggingface']['dataset_repo_id']}",
            f"HF_KB_REPO_ID={spec['huggingface']['knowledge_base_repo_id']}",
            "WANDB_PROJECT=stemtune",
        ]
    ) + "\n"


def render_runbook(spec: dict) -> str:
    task = spec["project"]["task"]
    lines = [
        f"# {spec['project']['name']} Runbook",
        "",
        "## End-to-End Automation Flow",
        "",
        "1. Put your raw source data in `data/raw/`.",
        "2. Normalize the dataset using `configs/dataset.json` and your own helper scripts in `scripts/`.",
    ]
    if TASK_BLUEPRINTS[task]["knowledge_base_enabled"]:
        lines.append("3. Build and index your document corpus using `configs/knowledge_base.json`.")
        lines.append("4. Select the model and recipe path with `python -m stemtune --task rag --gpu-memory-gb <budget>`.")
        lines.append("5. Run the selected retrieval/training recipe and save outputs in `artifacts/models/`.")
        lines.append("6. Evaluate the run against `configs/evaluation.json` before promotion or publication.")
        lines.append("7. Publish datasets, KB assets, and model artifacts using `configs/publish.json` and your own namespace.")
    else:
        lines.append(f"3. Select the model and recipe path with `python -m stemtune --task {task} --gpu-memory-gb <budget>`.")
        lines.append("4. Run the selected training recipe and save outputs in `artifacts/models/`.")
        lines.append("5. Evaluate the run against `configs/evaluation.json` before promotion or publication.")
        lines.append("6. Publish datasets and model artifacts using `configs/publish.json` and your own namespace.")
    lines.extend(
        [
            "",
            "## Why This Exists",
            "",
            "This scaffold keeps your project separate from any repository-specific assets or account assumptions.",
            "You own the model choice, data sources, Hub namespace, evaluation gates, and orchestration layer.",
        ]
    )
    return "\n".join(lines) + "\n"


def create_project_scaffold(args) -> tuple[Path, list[str]]:
    spec = build_project_spec(args)
    target_dir = Path(args.output_dir).expanduser().resolve() / spec["project"]["slug"]

    if target_dir.exists() and any(target_dir.iterdir()) and not args.force:
        raise FileExistsError(
            f"Target directory '{target_dir}' already exists and is not empty. Use --force to overwrite files."
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    for relative_dir in PROJECT_DIRS:
        (target_dir / relative_dir).mkdir(parents=True, exist_ok=True)

    files_written = []

    write_json(target_dir / "stemtune.project.json", spec)
    files_written.append("stemtune.project.json")

    write_json(target_dir / "configs/dataset.json", build_dataset_config(spec))
    files_written.append("configs/dataset.json")

    write_json(target_dir / "configs/knowledge_base.json", build_knowledge_base_config(spec))
    files_written.append("configs/knowledge_base.json")

    write_json(target_dir / "configs/training.json", build_training_config(spec))
    files_written.append("configs/training.json")

    write_json(target_dir / "configs/publish.json", build_publish_config(spec))
    files_written.append("configs/publish.json")

    write_json(target_dir / "configs/evaluation.json", build_evaluation_config(spec))
    files_written.append("configs/evaluation.json")

    write_text(target_dir / "README.md", render_project_readme(spec))
    files_written.append("README.md")

    write_text(target_dir / "scripts/README.md", render_scripts_readme())
    files_written.append("scripts/README.md")

    write_text(target_dir / ".env.example", render_env_example(spec))
    files_written.append(".env.example")

    write_text(target_dir / "runbook.md", render_runbook(spec))
    files_written.append("runbook.md")

    for keep_path in [
        "data/raw/.gitkeep",
        "data/interim/.gitkeep",
        "data/processed/.gitkeep",
        "knowledge_base/raw/.gitkeep",
        "knowledge_base/processed/.gitkeep",
        "knowledge_base/index/.gitkeep",
        "artifacts/models/.gitkeep",
        "artifacts/evals/.gitkeep",
    ]:
        touch_keep(target_dir / keep_path)
        files_written.append(keep_path)

    return target_dir, files_written
