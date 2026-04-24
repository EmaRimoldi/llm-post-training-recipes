from datasets import load_dataset
from huggingface_hub import HfApi

# === Config ===
DATASET_ID = "VinceEPFL/mcqa-reasoned"            # <-- cambia con il tuo
OUTPUT_DATASET_ID = "VinceEPFL/mcqa-reasoned"  # o stesso per overwrite
SPLIT = "train"
LETTERS = ["A", "B", "C", "D"]

# === 1. Load ===
ds = load_dataset(DATASET_ID, split=SPLIT)

# === 2. Fix: convert dict→list + correct index→letter ===
def fix_choices_and_correct(example):
    # Fix choices (from dict to list)
    if isinstance(example["choices"], dict):
        example["choices"] = [example["choices"].get(l, "") for l in LETTERS]
    
    # Fix correct (from int to letter)
    if isinstance(example["correct"], int):
        if 0 <= example["correct"] < len(LETTERS):
            example["correct"] = LETTERS[example["correct"]]
        else:
            example["correct"] = "A"  # fallback
    return example

ds_fixed = ds.map(fix_choices_and_correct)

# === 3. (Optional) check ===
def check_correct_format(example):
    if example["correct"] not in LETTERS:
        print("⚠️ Bad correct value:", example["correct"])
    return example

ds_checked = ds_fixed.map(check_correct_format)

# === 4. Push to Hugging Face Hub ===
ds_checked.push_to_hub(OUTPUT_DATASET_ID, private=True)
print(f"✅ Pushed to {OUTPUT_DATASET_ID}")
