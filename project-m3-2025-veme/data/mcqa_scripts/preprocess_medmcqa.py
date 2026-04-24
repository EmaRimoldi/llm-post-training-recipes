from datasets import load_dataset
from huggingface_hub import login
import os

# Login to HuggingFace (make sure you've run 'huggingface-cli login' first)
login()

print("Loading MedMCQA dataset...")
dataset = load_dataset("openlifescienceai/medmcqa")

def convert_to_standard_format(example):
    """Convert MedMCQA to standard MCQA format"""
    # Get the 4 choices
    choices = [
        example["opa"],
        example["opb"], 
        example["opc"],
        example["opd"]
    ]
    
    # Convert answer index to letter (handle both 0-based and 1-based)
    idx = int(example["cop"])
    if idx > 3:  
        idx = idx - 1
    answer = ["A", "B", "C", "D"][idx]
    
    return {
        "question": example["question"],
        "choices": choices,
        "answer": answer
    }

# Process all splits
processed_dataset = {}
for split in dataset.keys():
    print(f"\nProcessing {split} split...")
    
    # Filter to single-choice questions only
    filtered = dataset[split].filter(
        lambda ex: ex.get("choice_type", "single") == "single"
    )
    print(f"Filtered from {len(dataset[split])} to {len(filtered)} single-choice questions")
    
    # Convert to standard format
    processed = filtered.map(
        convert_to_standard_format,
        remove_columns=filtered.column_names
    )
    
    processed_dataset[split] = processed
    
    # Show sample
    if len(processed) > 0:
        print(f"Sample from {split}:")
        sample = processed[0]
        print(f"  Question: {sample['question'][:100]}...")
        print(f"  Choices: {sample['choices']}")
        print(f"  Answer: {sample['answer']}")

# Upload to HuggingFace
HF_USERNAME = "VinceEPFL"  # Change this to your username
DATASET_NAME = "medmcqa-standard"
repo_id = f"{HF_USERNAME}/{DATASET_NAME}"

print(f"\nUploading to {repo_id}...")
for split, data in processed_dataset.items():
    data.push_to_hub(repo_id, split=split)

print(f"\n✅ Done! Dataset uploaded to: https://huggingface.co/datasets/{repo_id}")
print(f"\nNow update your training script to use: '{repo_id}'")