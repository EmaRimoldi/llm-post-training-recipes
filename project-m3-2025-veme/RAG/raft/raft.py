import os
import torch
import random
import numpy as np
import logging
import wandb # Keep import, but make it optional
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling, 
)
import multiprocessing
import warnings
from huggingface_hub import HfApi 
import sys 


def require_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Set the {name} environment variable before running this script.")
    return value

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
RAFT_DATASET_PATH = "raft_dataset_generated_with_gpt_wrapper.jsonl"
BASE_MODEL_NAME = "EmaRimoldi/rag-mcqa-model"  # Base model name
OUTPUT_PATH = "./qwen_raft_finetuned_1_max_length"  # Directory where the fine-tuned model will be saved

# Hugging Face Hub details for the fine-tuned model
HF_REPO_NAME = os.getenv("HF_MODEL_REPO_ID", "EmaRimoldi/mnlp-raft-qwen-1-max-length")
HF_TOKEN = require_env_var("HF_TOKEN")

# --- Wandb Configuration (New Section) ---
# Set to True if you want to enable Wandb, False otherwise.
# You can make this configurable via command-line arguments later if needed.
USE_WANDB = True 

# --- Set random seed for reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- Check for CUDA ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info(f"\033[1m" + "-"*80 + f"\nCUDA is available! Using device: {torch.cuda.get_device_name(device)}\n" + "-"*80 + "\033[0m")
    allocated_memory = torch.cuda.memory_allocated(device) / 1024**3  # in GB
    logger.info(f"Memory Allocated: {allocated_memory:.2f} GB")
    cached_memory = torch.cuda.memory_reserved(device) / 1024**3  # in GB
    logger.info(f"Memory Cached: {cached_memory:.2f} GB")
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3  # in GB
    free_memory = total_memory - allocated_memory - cached_memory  # Available memory in GB
    logger.info(f"Available Memory: {free_memory:.2f} GB")
    cpu_cores = multiprocessing.cpu_count()
    logger.info(f"Number of CPU cores available: {cpu_cores}")
    dataloader_num_workers = min(cpu_cores // 2, 20)
    logger.info(f"Using {dataloader_num_workers} workers for data loading\n" + "-"*80 + "\033[0m")
else:
    device = torch.device("cpu")
    logger.info("\033[1m" + "-"*80 + "\nCUDA is not available. Using CPU.\n" + "-"*80 + "\033[0m")
    dataloader_num_workers = 0 # No workers for CPU

# --- Initialize Wandb (Conditional) ---
if USE_WANDB:
    try:
        wandb.init(
            project="raft-finetuning",  # Choose a name for your Wandb project
            config={  # Configuration parameters you want to track
                "learning_rate": 2e-5,
                "epochs": 10,  
                "batch_size": 2,
                "gradient_accumulation_steps": 8,
            }
        )
        logger.info("Wandb initialized successfully.")
    except Exception as e:
        logger.warning(f"Could not initialize Wandb. Continuing without it. Error: {e}")
        USE_WANDB = False # Disable Wandb if initialization fails (e.g., no API key)
else:
    logger.info("Wandb is disabled by configuration (USE_WANDB = False).")


# --- Load Tokenizer and Model ---
logger.info(f"Loading tokenizer and model: {BASE_MODEL_NAME}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 2048 
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    logger.info("Tokenizer and model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading tokenizer or model {BASE_MODEL_NAME}: {e}", exc_info=True)
    sys.exit(1)


# --- Load and Prepare Dataset ---
logger.info(f"Loading RAFT dataset from {RAFT_DATASET_PATH}...")
try:
    raw_dataset = load_dataset("json", data_files=RAFT_DATASET_PATH, split="train")
    logger.info(f"Dataset loaded successfully. Total examples: {len(raw_dataset)}")
except Exception as e:
    logger.error(f"Error loading RAFT dataset: {e}", exc_info=True)
    sys.exit(1)

# Define the prompt template for RAFT fine-tuning
def format_example(example):
    question = example['question']
    documents = "\n".join([f"##begin_document##\n{doc}\n##end_document##" for doc in example['documents']])
    answer_cot = example['answer_cot']
    prompt = f"""You are a knowledgeable and precise STEM professional with expertise in science, technology, engineering, and mathematics.
Your task is to answer the following question based ONLY on the provided documents.
If the answer is not present in the documents, state that you cannot answer.
For each piece of information you use, cite the exact text from the document by enclosing it in ##begin_quote## and ##end_quote## tags.
Provide a detailed Chain-of-Thought reasoning before giving the final answer.

Question: {question}

Documents:
{documents}

Chain-of-Thought Answer: {answer_cot}"""
    return {"text": prompt}

# Apply the formatting to the entire dataset before splitting it
logger.info("Formatting dataset examples...")
formatted_dataset = raw_dataset.map(
    format_example,
    remove_columns=raw_dataset.column_names,
    batched=False,
    num_proc=dataloader_num_workers 
)
logger.info("Dataset formatting complete.")

# Tokenization function for the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=tokenizer.model_max_length)

# Tokenize the entire dataset
logger.info("Tokenizing dataset...")
tokenized_dataset = formatted_dataset.map(
    tokenize_function,
    batched=True,
    num_proc=dataloader_num_workers,
    remove_columns=["text"]
)
logger.info("Dataset tokenization complete.")

# Split the dataset into training and evaluation
train_size = int(0.9 * len(tokenized_dataset))  
eval_size = len(tokenized_dataset) - train_size  

if train_size == 0 or eval_size == 0:
    logger.error("Not enough examples to create both train and eval splits. Please check your dataset.")
    sys.exit(1)

train_dataset, eval_dataset = torch.utils.data.random_split(tokenized_dataset, [train_size, eval_size])
logger.info(f"Dataset split into train ({len(train_dataset)} examples) and eval ({len(eval_dataset)} examples).")

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
logger.info("Data collator set up.")

# --- Set up Training Arguments ---
logger.info("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_PATH,
    overwrite_output_dir=True,
    num_train_epochs=3,  
    per_device_train_batch_size=1,  
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,  
    logging_strategy="steps",
    logging_steps=50,
    eval_steps=1500,
    save_strategy="steps",
    save_steps=1500,
    save_total_limit=1,  
    learning_rate=2e-5,  
    weight_decay=0.01,
    warmup_ratio=0.03,  
    lr_scheduler_type="cosine",
    bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
    # --- IMPORTANT Wandb Integration Change ---
    report_to=["wandb"] if USE_WANDB else "none", 
    # If USE_WANDB is True, report to wandb. Otherwise, report to "none" (disables all integrations).
    remove_unused_columns=True, 
    logging_dir=f'{OUTPUT_PATH}/logs',  
    eval_strategy="steps", 
    load_best_model_at_end=True,  
)

# --- Initialize and Run Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  
    tokenizer=tokenizer,
    data_collator=data_collator,
)

logger.info("Starting fine-tuning...")
try:
    trainer.train()
    logger.info("\nFine-tuning complete!")

    # --- Save the fine-tuned model locally ---
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
    logger.info(f"Fine-tuned model and tokenizer saved locally to {OUTPUT_PATH}")
    logger.info("You can now load this model for inference.")

except Exception as e:
    logger.error(f"An error occurred during training: {e}", exc_info=True)
    if USE_WANDB: # Only finish wandb run if it was initialized
        wandb.finish()
    sys.exit(1)


# --- Finish the Wandb run (Conditional) ---
if USE_WANDB:
    wandb.finish()  

# --- Upload to Hugging Face Hub ---
logger.info(f"Attempting to upload model to Hugging Face Hub: {HF_REPO_NAME}…")

api = HfApi()

try:
    api.create_repo(repo_id=HF_REPO_NAME, private=False, repo_type="model", token=HF_TOKEN)
    logger.info(f"Repository '{HF_REPO_NAME}' created successfully!")
except Exception as e:
    if "You already have a model named" in str(e) or "RepositoryAlreadyExists" in str(e):
        logger.warning(f"Repository '{HF_REPO_NAME}' already exists. Proceeding with the upload.")
    else:
        logger.error(f"Error creating repository '{HF_REPO_NAME}': {e}", exc_info=True)
        sys.exit(1)

try:
    trainer.push_to_hub(
        repo_id=HF_REPO_NAME,
        token=HF_TOKEN,
        commit_message="Fine-tuned Qwen model with RAFT dataset"
    )
    logger.info(f"Model & tokenizer successfully uploaded to https://huggingface.co/{HF_REPO_NAME}")
except Exception as e:
    logger.error(f"Error uploading to Hugging Face Hub: {e}", exc_info=True)
    sys.exit(1) 

logger.info("Script execution complete. Your fine-tuned model is now live on Hugging Face!")
