from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
save_directory_hf = PROJECT_ROOT / "quantization" / "qwen_llm_compressor_a8w8"
os.makedirs(save_directory_hf, exist_ok=True)


MODEL_ID = "Qwen/Qwen3-0.6B-Base"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

from datasets import load_dataset

NUM_CALIBRATION_SAMPLES=512
MAX_SEQUENCE_LENGTH=2048


dataset = load_dataset(
    "json",
    data_files={"train": str(PROJECT_ROOT / "data" / "calibration" / "gpqa_only_calibration_data.jsonl")},
)["train"]

def tokenize_function(examples):
    return tokenizer(examples["text"], padding=False, truncation=True, max_length=MAX_SEQUENCE_LENGTH)

tokenized_dataset = dataset.map(tokenize_function, remove_columns=dataset.column_names)

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

# Configure the quantization algorithms to run.
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]

# Apply quantization.
oneshot(
    model=model,
    dataset=tokenized_dataset,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save to disk compressed.
model.save_pretrained(str(save_directory_hf), save_compressed=True)
tokenizer.save_pretrained(str(save_directory_hf))
