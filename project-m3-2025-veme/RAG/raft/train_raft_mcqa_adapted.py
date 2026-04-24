import os
import torch
import random
import numpy as np # Import numpy for np.random.seed
import logging
import wandb # Keep import, but make it optional
import gc
import sys # For sys.exit()

from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from huggingface_hub import HfApi, HfFolder, Repository # For Hugging Face Hub operations

# --- Suppress warnings for cleaner output ---
import warnings
warnings.filterwarnings("ignore")


def require_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Set the {name} environment variable before running this script.")
    return value

# --- Configuration ---
MODEL_NAME = "EmaRimoldi/rag-mcqa-model"
OUTPUT_PATH = "./qwen3_0.6B_mcqa_sciq_raft" # Directory where the fine-tuned model will be saved locally
DOC_MAX_TOKENS = 512
NUM_NEGATIVES = 5
LETTERS = ["A", "B", "C", "D"]

# Hugging Face Hub details
HF_REPO_NAME = os.getenv("HF_MODEL_REPO_ID", "EmaRimoldi/mnlp-raft-qwen-SciQ")
HF_TOKEN = require_env_var("HF_TOKEN")

# --- Wandb Configuration (New Section) ---
# Set to True if you want to enable Wandb, False otherwise.
USE_WANDB = True

# --- Set seed for reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED) # Use np.random.seed for numpy reproducibility
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Logging setup ---
# Ensure the log file directory exists
log_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(log_dir, "output_rag_sciq_raft.log")
os.makedirs(log_dir, exist_ok=True) # Create directory if it doesn't exist

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

logger.info(f"CUDA available: {torch.cuda.is_available()}")
logger.info(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    logger.info(f"Device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    # --- GPU memory logging ---
    def log_gpu_usage(tag=""):
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"[{tag}] GPU Memory - Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Total: {total:.2f} GB")
    log_gpu_usage("Startup")
else:
    # Define a dummy log_gpu_usage if CUDA is not available
    def log_gpu_usage(tag=""):
        logger.info(f"[{tag}] GPU Memory - N/A (CUDA not available)")


# --- Initialize Wandb (Conditional) ---
wandb_config = {
    "model_name": MODEL_NAME,
    "num_negatives": NUM_NEGATIVES,
    "doc_max_tokens": DOC_MAX_TOKENS,
    "learning_rate": 2e-5,
    "epochs": 3,
    "batch_size": 2,
    "gradient_accumulation_steps": 2,
}

if USE_WANDB:
    try:
        wandb.init(
            project="raft-finetuning-SciQ",
            entity="emanuelerimoldi7-epfl", # Your entity name
            config=wandb_config
        )
        logger.info("Initialized Weights & Biases")
    except Exception as e:
        logger.warning(f"Could not initialize Wandb. Continuing without it. Error: {e}", exc_info=True)
        USE_WANDB = False # Disable Wandb if initialization fails (e.g., no API key)
else:
    logger.info("Wandb is disabled by configuration (USE_WANDB = False).")


# --- Load data ---
logger.info("Loading SciQ dataset...")
try:
    raw_dataset = load_dataset("allenai/sciq", split="train")
    all_supports = raw_dataset["support"]
    logger.info(f"Loaded {len(raw_dataset)} examples.")
except Exception as e:
    logger.error(f"Error loading SciQ dataset: {e}", exc_info=True)
    if USE_WANDB:
        wandb.finish()
    sys.exit(1)


# --- Load model & tokenizer ---
logger.info(f"Loading tokenizer & model: {MODEL_NAME}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer pad_token set to eos_token.")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))
    logger.info("Tokenizer and model loaded and ready.")
    log_gpu_usage("After model load")
except Exception as e:
    logger.error(f"Error loading tokenizer or model {MODEL_NAME}: {e}", exc_info=True)
    if USE_WANDB:
        wandb.finish()
    sys.exit(1)


# --- Preprocessing ---
def preprocess_rag_sciq(example):
    q = example["question"]
    # Ensure options are always present before shuffling
    distractors = [example["distractor3"], example["distractor1"], example["distractor2"]]
    corr = example["correct_answer"]
    pos = example["support"]

    # Handle cases where all_supports might not have enough unique supports
    available_distractors = [s for s in all_supports if s != pos]
    if len(available_distractors) < NUM_NEGATIVES:
        negs = random.sample(available_distractors, len(available_distractors))
        if len(negs) < NUM_NEGATIVES: # Pad with empty strings if not enough unique negatives
            negs.extend([""] * (NUM_NEGATIVES - len(negs)))
        logger.warning(f"Not enough unique negative documents for example. Found {len(available_distractors)}, needed {NUM_NEGATIVES}.")
    else:
        negs = random.sample(available_distractors, NUM_NEGATIVES)

    prompt = f"The following are multiple choice questions about knowledge and skills in advanced master-level STEM courses.\n\n{q}\n"
    options = distractors + [corr]
    random.shuffle(options) # Shuffle options for proper multiple choice format
    
    # Track the correct answer's letter after shuffling
    answer_letter_for_this_example = ""
    for letter, opt in zip(LETTERS, options):
        opt_text = opt.strip()
        if opt_text.lower().startswith(f"{letter.lower()}."):
            opt_text = opt_text[2:].strip()
        prompt += f"{letter}. {opt_text}\n"
        if opt == corr: # Identify the correct answer's letter after shuffling
            answer_letter_for_this_example = letter

    prompt += "Answer:\n\nRelevant Documents:\n"

    docs = [pos] + negs
    random.shuffle(docs) # Shuffle documents to mix positive and negative

    for i, doc in enumerate(docs):
        doc_tokens = tokenizer.encode(doc, truncation=True, max_length=DOC_MAX_TOKENS, add_special_tokens=False)
        truncated_doc = tokenizer.decode(doc_tokens)
        prompt += f"Document {i}:::\n{truncated_doc}\n\n"

    answer_text = f" {answer_letter_for_this_example}. {corr.strip()}"
    full_prompt = prompt + answer_text + tokenizer.eos_token

    tokenized_output = tokenizer(
        full_prompt,
        padding="max_length",
        truncation=True,
        max_length=4096,
        return_tensors="pt" # Return PyTorch tensors directly
    )

    input_ids = tokenized_output["input_ids"].squeeze(0) # Remove batch dimension
    attention_mask = tokenized_output["attention_mask"].squeeze(0) # Remove batch dimension

    prompt_only = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_len = len(prompt_only)
    
    labels = input_ids.clone() # Start with a copy of input_ids for labels
    labels[:prompt_len] = -100 # Mask out prompt tokens
    
    # Ensure labels also have max_length and padding with -100
    if len(labels) < 4096:
        padding = torch.full((4096 - len(labels),), -100, dtype=torch.long)
        labels = torch.cat([labels, padding])
    elif len(labels) > 4096:
        labels = labels[:4096]

    return {
        "input_ids": input_ids.tolist(), # Convert back to list for dataset map
        "attention_mask": attention_mask.tolist(), # Convert back to list
        "labels": labels.tolist() # Convert back to list
    }

logger.info("Applying preprocessing to dataset...")
log_gpu_usage("Before preprocessing")
try:
    tokenized_dataset = raw_dataset.map(preprocess_rag_sciq, remove_columns=raw_dataset.column_names, num_proc=os.cpu_count())
    logger.info("Preprocessing complete.")
    log_gpu_usage("After preprocessing")
except Exception as e:
    logger.error(f"Error during dataset preprocessing: {e}", exc_info=True)
    if USE_WANDB:
        wandb.finish()
    sys.exit(1)


# --- Split Dataset ---
logger.info("Splitting dataset into train and evaluation sets...")
if len(tokenized_dataset) > 100:
    split = tokenized_dataset.train_test_split(test_size=0.05, seed=SEED)
    train_ds, eval_ds = split["train"], split["test"]
    logger.info(f"Dataset split into {len(train_ds)} training and {len(eval_ds)} evaluation examples.")
else:
    train_ds, eval_ds = tokenized_dataset, None
    logger.warning("Not enough data for an evaluation split. Training on the full dataset.")

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
logger.info("Data collator ready.")


# --- Training arguments ---
logger.info("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_PATH,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2, # Keep for potential eval_ds or future use
    gradient_accumulation_steps=2,
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=1,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    fp16=torch.cuda.is_available(), # Use fp16 if CUDA is available
    # --- IMPORTANT Wandb Integration Change ---
    report_to=["wandb"] if USE_WANDB else "none",
    remove_unused_columns=False, # Set to False when using DataCollatorForLanguageModeling
    logging_dir=f"{OUTPUT_PATH}/logs",
    run_name="raft_finetuning_sciq",
    eval_strategy="steps" if eval_ds else "no", # Only evaluate if eval_ds exists
    load_best_model_at_end=True if eval_ds else False, # Load best model only if evaluating
)

# --- Trainer ---
logger.info("Setting up Trainer...")
log_gpu_usage("Before trainer setup")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds, # Pass eval_ds even if None, Trainer handles it
    tokenizer=tokenizer,
    data_collator=collator
)
log_gpu_usage("After trainer setup")

# --- Training ---
logger.info("--- Starting training ---")
log_gpu_usage("Before training")
try:
    trainer.train()
    log_gpu_usage("After training")
    logger.info("--- Training complete ---")

    # --- Save ---
    final_dir = os.path.join(OUTPUT_PATH, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir) # Saves model and tokenizer
    tokenizer.save_pretrained(final_dir) # Ensure tokenizer is saved separately, though save_model usually handles it
    logger.info(f"Model saved locally to {final_dir}")

except Exception as e:
    logger.error(f"An error occurred during training: {e}", exc_info=True)
    if USE_WANDB:
        wandb.finish()
    sys.exit(1)

# --- Evaluate ---
if eval_ds:
    logger.info("--- Starting evaluation ---")
    metrics = trainer.evaluate()
    logger.info(f"Evaluation results: {metrics}")
else:
    logger.info("Skipping evaluation: no eval dataset available.")

# --- Finish the Wandb run (Conditional) ---
if USE_WANDB:
    wandb.finish()
    logger.info("Wandb run finished.")

# Clear CUDA cache if possible to free up memory before push
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    log_gpu_usage("After clearing cache")


# --- Upload to Hugging Face Hub (using Repository clone method) ---
logger.info(f"Attempting to upload model to Hugging Face Hub: {HF_REPO_NAME}...")

try:
    # Save token to local cache (so git and hub commands can use it)
    HfFolder.save_token(HF_TOKEN)
    api = HfApi()

    # Create the repo on the Hub (if it doesn't already exist)
    try:
        api.create_repo(repo_id=HF_REPO_NAME, exist_ok=True, repo_type="model", token=HF_TOKEN)
        logger.info(f"Repository '{HF_REPO_NAME}' is ready.")
    except Exception as e:
        # Check if the error is due to the repo already existing
        if "You already have a model named" in str(e) or "RepositoryAlreadyExists" in str(e):
            logger.warning(f"Repository '{HF_REPO_NAME}' already exists. Proceeding with the upload.")
        else:
            logger.error(f"Error creating repository '{HF_REPO_NAME}': {e}", exc_info=True)
            sys.exit(1) # Exit if repository creation/check fails

    # Clone the empty repo (or pull latest changes if it exists)
    local_repo_path = os.path.join(os.getcwd(), HF_REPO_NAME.split("/")[-1])
    logger.info(f"Cloning/pulling repository to local path: '{local_repo_path}'...")
    repo = Repository(
        local_dir=local_repo_path,
        clone_from=HF_REPO_NAME,
        use_auth_token=HF_TOKEN,
        # skip_lfs_id is important if you're going to copy files and let git_add handle LFS
        # Or remove it if you have pre-existing LFS files you want to keep
        # skip_lfs_id=True 
    )
    logger.info("Repository cloned/pulled successfully.")

    # Copy model files into the local repo
    def copy_tree(src_dir, dst_dir):
        logger.info(f"Copying files from '{src_dir}' to '{dst_dir}'...")
        for root, dirs, files in os.walk(src_dir):
            # Compute path in destination
            rel_path = os.path.relpath(root, src_dir)
            target_root = os.path.join(dst_dir, rel_path)
            os.makedirs(target_root, exist_ok=True)
            for f in files:
                src_file = os.path.join(root, f)
                dst_file = os.path.join(target_root, f)
                # Overwrite any existing
                import shutil # Import shutil for file operations
                shutil.copy2(src_file, dst_file) # Use copy2 to preserve metadata
        logger.info("File copying complete.")

    logger.info(f"Copying model files from '{final_dir}' to local repo at '{local_repo_path}'...")
    copy_tree(final_dir, local_repo_path)

    # Commit & push
    logger.info("Adding files to Git and tracking LFS...")
    repo.git_add(auto_lfs_track=True)  # track large files with Git LFS
    commit_message = "Upload Qwen3-0.6B mcqa SciQ fine-tuned model"
    repo.git_commit(commit_message)
    logger.info(f"Git commit with message: '{commit_message}'")
    
    logger.info("Pushing to the Hub...")
    repo.git_push()

    logger.info(f"✅ Model successfully pushed to https://huggingface.co/{HF_REPO_NAME}")

except Exception as e:
    logger.error(f"❌ Error during Hugging Face Hub upload process: {e}", exc_info=True)
    sys.exit(1)

logger.info("Script execution complete. Your fine-tuned model is now live on Hugging Face!")
