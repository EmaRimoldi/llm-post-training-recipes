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
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, PeftModel
import multiprocessing
import warnings
from huggingface_hub import HfApi # Import HfApi for explicit repo creation/check
import sys # Per sys.exit()


def require_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Set the {name} environment variable before running this script.")
    return value

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set up logging with date and time format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",  # Format with date, time, log level and message
    datefmt="%Y-%m-%d %H:%M:%S"  # Custom format for the timestamp
)
logger = logging.getLogger(__name__) # Use logger instead of print for consistency

# --- Configuration ---
RAFT_DATASET_PATH = "raft_dataset_generated_with_gpt_wrapper.jsonl"
BASE_MODEL_NAME = "EmaRimoldi/rag-mcqa-model"  # Base model name
OUTPUT_PATH = "./qwen_raft_finetuned_lora_adapters_2_max_length"  # Directory where the LoRA adapters will be saved
FINAL_MERGED_MODEL_PATH = "./qwen_raft_merged_model_max_length"  # Directory for the final merged model

# Hugging Face Hub details
HF_REPO_NAME = os.getenv("HF_MODEL_REPO_ID", "EmaRimoldi/mnlp-raft-qwen-2-max-length")
HF_TOKEN = require_env_var("HF_TOKEN")

# --- Wandb Configuration (New Section) ---
# Set to True if you want to enable Wandb, False otherwise.
USE_WANDB = False

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
    logger.info(f"CUDA is available! Using device: {torch.cuda.get_device_name(device)}")
    cpu_cores = multiprocessing.cpu_count()
    logger.info(f"Number of CPU cores available: {cpu_cores}")
    dataloader_num_workers = min(cpu_cores // 2, 20)
    logger.info(f"Using {dataloader_num_workers} workers for data loading")
else:
    device = torch.device("cpu")
    logger.info("CUDA is not available. Using CPU.")
    dataloader_num_workers = 0 # No workers for CPU

# --- Initialize Wandb (Conditional) ---
if USE_WANDB:
    try:
        wandb.init(
            project="raft-lora-finetuning",  # Choose a name for your Wandb project
            config={  # Configuration parameters you want to track
                "learning_rate": 2e-5,
                "epochs": 3,  
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "lora_r": 2,
                "lora_alpha": 16,
                "lora_dropout": 0.05
            }
        )
        logger.info("Wandb initialized successfully.")
    except Exception as e:
        logger.warning(f"Could not initialize Wandb. Continuing without it. Error: {e}")
        USE_WANDB = False # Disable Wandb if initialization fails (e.g., no API key)
else:
    logger.info("Wandb is disabled by configuration (USE_WANDB = False).")


# --- Load and Prepare the Model ---
logger.info(f"Loading base model: {BASE_MODEL_NAME}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    # Ensure pad_token is set for Qwen or similar models if it's None
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer pad_token set to eos_token.")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    logger.info("Base model loaded successfully.")

    tokenizer.model_max_length = 1024
    logger.info(f"Tokenizer max_length set to {tokenizer.model_max_length}.")

    # --- Prepare Model for LoRA Fine-tuning ---
    model.gradient_checkpointing_enable()
    logger.info("Gradient checkpointing enabled.")

    # LoRA configuration
    lora_config = LoraConfig(
        r=2,  # LoRA attention dimension [1]
        lora_alpha=16,  # Alpha parameter for LoRA scaling [1]
        lora_dropout=0.05,  # Dropout probability for LoRA layers
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],  # Adjust based on the base model (e.g., for Llama use "q_proj", "k_proj", "v_proj", "o_proj")
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Show how many parameters are actually being trained

except Exception as e:
    logger.error(f"Error loading model or configuring LoRA: {e}", exc_info=True)
    if USE_WANDB:
        wandb.finish()
    sys.exit(1)


# --- Load and Prepare Dataset ---
logger.info(f"Loading RAFT dataset from {RAFT_DATASET_PATH}...")
try:
    raw_dataset = load_dataset("json", data_files=RAFT_DATASET_PATH, split="train")
    logger.info(f"Dataset loaded successfully. Total examples: {len(raw_dataset)}")
except Exception as e:
    logger.error(f"Error loading RAFT dataset: {e}", exc_info=True)
    if USE_WANDB:
        wandb.finish()
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
train_size = int(0.9 * len(tokenized_dataset))  # 90% for training
eval_size = len(tokenized_dataset) - train_size  # Remaining 10% for evaluation

if train_size == 0 or eval_size == 0:
    logger.error("Not enough examples to create both train and eval splits. Please check your dataset size.")
    if USE_WANDB:
        wandb.finish()
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
    gradient_accumulation_steps=1,  
    logging_strategy="steps",
    logging_steps=50,
    eval_steps=1500, # Eval strategy "steps" requires eval_steps to be set
    save_steps=1500,
    save_strategy="steps",
    save_total_limit=1,  
    learning_rate=2e-5,  
    weight_decay=0.01,
    warmup_ratio=0.03,  
    lr_scheduler_type="cosine",
    bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8, # Use bf16 only if GPU supports it (Ampere or newer)
    # --- IMPORTANT Wandb Integration Change ---
    report_to=["wandb"] if USE_WANDB else "none", 
    remove_unused_columns=False, 
    logging_dir=f'{OUTPUT_PATH}/logs', 
    eval_strategy="steps",  
    load_best_model_at_end=True,  
    save_safetensors=False # Explicitly disable safetensors saving for shared weights issue
)

# --- Initialize and Run Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset, # Ensure eval_dataset is passed
    tokenizer=tokenizer,
    data_collator=data_collator
)

logger.info("Starting fine-tuning...")
try:
    trainer.train()
    logger.info("Fine-tuning complete!")

    # --- Save the fine-tuned LoRA adapters ---
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH) 
    logger.info(f"Fine-tuned LoRA adapters and tokenizer saved to {OUTPUT_PATH}")

except Exception as e:
    logger.error(f"An error occurred during training: {e}", exc_info=True)
    if USE_WANDB:
        wandb.finish()
    sys.exit(1)


# --- Finish the Wandb run (Conditional) ---
if USE_WANDB:
    wandb.finish()  

# --- Merge LoRA Adapters with Base Model and Save Full Model ---
logger.info("Loading base model again to merge LoRA adapters...")
try:
    base_model_for_merge = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, 
        device_map="auto" if torch.cuda.is_available() else None
    )

    # Load the LoRA adapters onto the base model
    merged_model = PeftModel.from_pretrained(base_model_for_merge, OUTPUT_PATH)

    # Merge the LoRA weights into the base model and unload the PEFT model
    logger.info("Merging LoRA adapters into the base model...")
    merged_model = merged_model.merge_and_unload()

    # Save the merged model and tokenizer
    os.makedirs(FINAL_MERGED_MODEL_PATH, exist_ok=True)
    logger.info(f"Saving merged model to {FINAL_MERGED_MODEL_PATH}...")
    merged_model.save_pretrained(FINAL_MERGED_MODEL_PATH)
    tokenizer.save_pretrained(FINAL_MERGED_MODEL_PATH) 

    logger.info(f"Full merged model and tokenizer saved to {FINAL_MERGED_MODEL_PATH}")

except Exception as e:
    logger.error(f"Error during merging or saving merged model: {e}", exc_info=True)
    sys.exit(1)


# --- Upload Merged Model to Hugging Face Hub ---
logger.info(f"Attempting to upload merged model to Hugging Face Hub: {HF_REPO_NAME}...")

api = HfApi()

# Create or check the repository
try:
    api.create_repo(repo_id=HF_REPO_NAME, private=False, repo_type="model", token=HF_TOKEN)
    logger.info(f"Repository '{HF_REPO_NAME}' created successfully!")
except Exception as e:
    if "You already have a model named" in str(e) or "RepositoryAlreadyExists" in str(e):
        logger.warning(f"Repository '{HF_REPO_NAME}' already exists. Proceeding with the upload.")
    else:
        logger.error(f"Error creating repository '{HF_REPO_NAME}': {e}", exc_info=True)
        sys.exit(1)

# Push the merged model and tokenizer to the Hub
try:
    # It's better to use push_to_hub methods on the merged model and tokenizer directly
    merged_model.push_to_hub(HF_REPO_NAME, token=HF_TOKEN)
    tokenizer.push_to_hub(HF_REPO_NAME, token=HF_TOKEN)
    
    logger.info(f"Model and tokenizer successfully uploaded to https://huggingface.co/{HF_REPO_NAME}")
except Exception as e:
    logger.error(f"Error uploading merged model to Hugging Face Hub: {e}", exc_info=True)
    sys.exit(1) 

logger.info("Script execution complete. Your fine-tuned and merged model is now live on Hugging Face!")
