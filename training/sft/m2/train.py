# Import necessary libraries
from datasets import load_dataset, concatenate_datasets
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
import logging
import os
import wandb

# before running this:
# export WANDB_PROJECT="mnlp_m2"
# export WANDB_RUN_NAME="0.6B_sft_next_token_masked_loss_run_1"

# Set up logging
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output.log') # Use abspath for robustness

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# Define model and dataset names
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
DATASET_NAMES = [
    "camel-ai/amc_aime_self_improving",
    "camel-ai/amc_aime_distilled",
    "camel-ai/gsm8k_distilled",
    #"camel-ai/math",
    "camel-ai/chemistry",
    "camel-ai/code",
]

# Define how to extract prompt and answer from each dataset
# For 'code' dataset, message_1 and message_2 are lists of dicts, e.g. [{'role':'user', 'content':'...'}], we take the first.
DATASET_CONFIGS = {
    "camel-ai/amc_aime_self_improving": {"prompt_col": "problem", "answer_col": "groud_truth_solution"},
    "camel-ai/amc_aime_distilled": {"prompt_col": "problem", "answer_col": "groud_truth_solution"},
    "camel-ai/gsm8k_distilled": {"prompt_col": "problem", "answer_col": "groud_truth_solution"},
    "camel-ai/math": {"prompt_col": "message_1", "answer_col": "message_2"},
    "camel-ai/chemistry": {"prompt_col": "message_1", "answer_col": "message_2"},
    "camel-ai/code": {"prompt_col": "message_1", "answer_col": "message_2", "is_chat_format": True, "content_key": "content"},
}

MAX_SEQ_LENGTH = 256 # Define max sequence length (let's not go above this)

# --- 1. Load Tokenizer and Model ---
print(f"Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set tokenizer.pad_token to {tokenizer.eos_token} ({tokenizer.pad_token_id})")
if tokenizer.eos_token is None:
    # Fallback if EOS is still None (should not happen for most models)
    tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'}) # Qwen uses <|endoftext|>
    print(f"Manually added eos_token: {tokenizer.eos_token} ({tokenizer.eos_token_id})")


print(f"Loading model {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
model.config.pad_token_id = tokenizer.pad_token_id
# Important for resizing if new tokens like pad_token were added and not part of original vocab
model.resize_token_embeddings(len(tokenizer))
print("Model and tokenizer loaded.")

# --- 2. Load and Preprocess Datasets ---

def preprocess_function(examples, tokenizer_ref, dataset_name_key):
    logger.info(f"Preprocessing batch for dataset: {dataset_name_key}")
    config = DATASET_CONFIGS.get(dataset_name_key)
    if config is None:
        logger.error(f'Missing config for dataset: {dataset_name_key}')
        # Return empty tokenized structure to avoid breaking map, but log error
        return {'input_ids': [], 'attention_mask': [], 'labels': []}

    prompts_raw = examples[config["prompt_col"]]
    answers_raw = examples[config["answer_col"]]

    prompts = []
    answers = []

    if config.get("is_chat_format", False):
        content_key = config["content_key"]
        # Assuming message_1/message_2 are lists of dicts, and we take the first dict's content
        # e.g., examples['message_1'] = [ [{'role':'user', 'content':'p1'}], [{'role':'user', 'content':'p2'}] ]
        for p_list in prompts_raw:
            if p_list and isinstance(p_list, list) and isinstance(p_list[0], dict) and content_key in p_list[0]:
                prompts.append(p_list[0][content_key])
            else:
                prompts.append("") # Handle empty or malformed
                logger.warning(f"Malformed prompt in {dataset_name_key}: {p_list}")
        for a_list in answers_raw:
            if a_list and isinstance(a_list, list) and isinstance(a_list[0], dict) and content_key in a_list[0]:
                answers.append(a_list[0][content_key])
            else:
                answers.append("") # Handle empty or malformed
                logger.warning(f"Malformed answer in {dataset_name_key}: {a_list}")
    else:
        prompts = prompts_raw
        answers = answers_raw

    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    for prompt, answer in zip(prompts, answers):
        if not isinstance(prompt, str) or not isinstance(answer, str):
            logger.error(f"Skipping non-string prompt/answer in {dataset_name_key}. Prompt: {type(prompt)}, Answer: {type(answer)}")
            continue

        # For Qwen, a common way to structure prompt-answer for base model fine-tuning might be:
        # "Prompt text\nAnswer: Answer text<|endoftext|>"
        # Or using chat template if available and desired. For simplicity here, direct concatenation.
        # We need the answer to end with EOS for the model to learn to generate it.
        # The prompt part should NOT have an EOS if the answer follows immediately in the same sequence.
        prompt_str = prompt # Could be "Question: " + prompt + "\nAnswer: " for more structure
        answer_str = answer + tokenizer_ref.eos_token # Ensure answer ends with EOS

        # Tokenize prompt to find its length (excluding special tokens for length calculation if tokenizer adds them by default)
        # It's safer to tokenize the prompt_str as it will be at the start of the combined text.
        # Qwen base tokenizer usually doesn't add BOS unless it's part of a chat template.
        prompt_tokenized = tokenizer_ref(prompt_str, add_special_tokens=False) # Don't add BOS/EOS for this length count
        prompt_len = len(prompt_tokenized['input_ids'])

        full_text = prompt_str + answer_str # Combined text

        tokenized_full = tokenizer_ref(
            full_text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding='max_length',  # Pad to MAX_SEQ_LENGTH
            # return_attention_mask=True # tokenizer_ref call already returns this by default
        )

        input_ids_for_batch = tokenized_full['input_ids']
        attention_mask_for_batch = tokenized_full['attention_mask']

        # Create labels:
        # Start with a copy of the (now padded) input_ids.
        labels_for_batch = list(input_ids_for_batch)

        # Iterate through the full length of the (padded) labels
        for i in range(len(labels_for_batch)): # This will be MAX_SEQ_LENGTH
            if i < prompt_len:
                labels_for_batch[i] = -100  # Mask the prompt part
            # Check if the token in input_ids is a pad_token_id. If so, mask it in labels.
            # This handles the padding added by padding='max_length'.
            elif input_ids_for_batch[i] == tokenizer_ref.pad_token_id:
                labels_for_batch[i] = -100

        # Edge case: if prompt_len itself is >= MAX_SEQ_LENGTH, all labels might become -100.
        # Your filter_empty_examples function should handle this by removing such examples.
        if prompt_len >= MAX_SEQ_LENGTH:
            logger.warning(
                f"Prompt length ({prompt_len}) is >= max_seq_length ({MAX_SEQ_LENGTH}). "
                f"All labels for this instance will be -100. Dataset: {dataset_name_key}"
            )

        batch_input_ids.append(input_ids_for_batch)
        batch_attention_mask.append(attention_mask_for_batch)
        batch_labels.append(labels_for_batch)

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels,
    }


all_tokenized_datasets = []
print("Loading and preprocessing datasets...")

for i, dataset_name_key in enumerate(DATASET_NAMES): # Renamed to dataset_name_key for clarity
    logging.info(f"\nProcessing dataset {i+1}/{len(DATASET_NAMES)}: {dataset_name_key}")
    try:
        try:
            dataset = load_dataset(dataset_name_key, split="train", trust_remote_code=True, num_proc=os.cpu_count() // 2)
        except Exception as e_train:
            logging.error(f"  Could not load 'train' split for {dataset_name_key}: {e_train}. Trying to load the first available split.")
            try:
                dataset_info = load_dataset(dataset_name_key, trust_remote_code=True, num_proc=os.cpu_count() // 2) # Load info to find splits
                first_split = list(dataset_info.keys())[0]
                dataset = load_dataset(dataset_name_key, split=first_split, trust_remote_code=True, num_proc=os.cpu_count() // 2)
                logging.info(f"  Loaded split '{first_split}' for {dataset_name_key}.")
            except Exception as e_fallback:
                logging.error(f"  ERROR: Could not load any split for dataset {dataset_name_key}: {e_fallback}")
                logging.error(f"  Skipping this dataset.")
                continue
        
        # Original columns to remove before tokenization (if any specific ones known).
        # Preprocess function will handle removing original data columns by design with remove_columns=dataset.column_names.
        logging.info(f"  Starting data preprocessing on CPU for {dataset_name_key}...")
        current_remove_columns = list(dataset.column_names) # Get columns for this specific dataset

        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=current_remove_columns, # Remove original columns from this dataset
            num_proc=os.cpu_count() // 2 or 1, # Use half CPUs or 1
            fn_kwargs={'tokenizer_ref': tokenizer, 'dataset_name_key': dataset_name_key} # Pass dataset_name_key
        )
        # Filter out examples that might have become empty or problematic after preprocessing
        # (e.g. if all labels were masked, or input_ids became empty)
        # This is important because DataCollator might fail with empty input_ids.
        def filter_empty_examples(example):
            return len(example['input_ids']) > 0 and any(label != -100 for label in example['labels'])

        original_len = len(tokenized_dataset)
        tokenized_dataset = tokenized_dataset.filter(filter_empty_examples)
        filtered_len = len(tokenized_dataset)
        if original_len > filtered_len:
            logger.warning(f"  Filtered out {original_len - filtered_len} examples from {dataset_name_key} due to empty inputs or no target labels.")


        logger.info(f"  Tokenized {dataset_name_key}. Number of examples: {len(tokenized_dataset)}")
        all_tokenized_datasets.append(tokenized_dataset)

    except Exception as e:
        logging.error(f"  ERROR processing dataset {dataset_name_key}: {e}", exc_info=True)
        logging.error(f"  Skipping this dataset.")


if not all_tokenized_datasets:
    logging.error("No datasets were successfully processed. Exiting.")
    exit()

# --- 3. Combine Datasets ---
print("\nCombining all processed datasets...")
if len(all_tokenized_datasets) > 1:
    final_dataset_train = concatenate_datasets(all_tokenized_datasets) # Assuming all are train splits
else:
    final_dataset_train = all_tokenized_datasets[0]
logging.info(f"Total number of tokenized examples in combined training dataset: {len(final_dataset_train)}")

# --- Save the complete dataset to disk ---
final_dataset_train.save_to_disk("sft_data")

DEBUG = False

if DEBUG and final_dataset_train: # Or check the specific tokenized_dataset
    print("\n--- Inspecting processed data structure ---")
    num_examples_to_check = min(10, len(final_dataset_train))
    problematic_indices = []
    for i in range(num_examples_to_check):
        example = final_dataset_train[i]
        print(f"\nExample {i}:")
        for key in ["input_ids", "labels", "attention_mask"]:
            if key in example:
                item = example[key]
                print(example[key])
                print(f"  {key} type: {type(item)}")
                if isinstance(item, list):
                    print(f"  {key} length: {len(item)}")
                    if len(item) > 0:
                        print(f"  {key}[0] type: {type(item[0])}")
                        if isinstance(item[0], list): # This is the "excessive nesting"
                            print(f"  !!!!!!!!!! NESTED LIST DETECTED IN {key} for example {i} !!!!!!!!!!!")
                            print(f"  {key} structure: {item[:2]}...") # Print a bit of the nested structure
                            if i not in problematic_indices:
                                problematic_indices.append(i)
                    else:
                        print(f"  {key} is an empty list.")
                else:
                    print(f"  {key} is not a list: {item}")
        print("-" * 20)

    if problematic_indices:
        print(f"Found problematic nested structures at indices: {problematic_indices}")
        print("Investigate the raw data for these samples and how the tokenizer processes them.")
    else:
        print("No obvious nesting detected in the first few examples.")
    print("--- End of data inspection ---\n")
else:
    logging.info("No data to inspect.")

# Create a DatasetDict for the Trainer (even if only 'train' split exists initially)
from datasets import DatasetDict
processed_datasets = DatasetDict()
processed_datasets["train"] = final_dataset_train

# Split into train and validation sets if you don't have a predefined validation set
# This is a simple split, consider more robust methods if needed.
# Check if the combined dataset (now under 'train' key) has enough samples for a split
if len(processed_datasets["train"]) > 100: # Arbitrary threshold to make split meaningful
    logging.info("Splitting combined dataset into train and validation (90/10)...")
    train_test_split = processed_datasets["train"].train_test_split(test_size=0.05, seed=42) # smaller validation
    processed_datasets["train"] = train_test_split["train"]
    processed_datasets["eval"] = train_test_split["test"]
    logging.info(f"  Train dataset size: {len(processed_datasets['train'])}")
    logging.info(f"  Eval dataset size: {len(processed_datasets['eval'])}")
else:
    logging.warning("Not enough data to create a validation split, or only one dataset was processed without inherent splits.")
    # Trainer can run without eval_dataset
    processed_datasets["eval"] = None


train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["eval"]


# --- 4. Set up Data Collator ---
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
logging.info("Data collator set up.")

print(transformers.__version__)
print(transformers.TrainingArguments.__module__)

# --- 5. Set up Training Arguments ---
OUTPUT_PATH = "./qwen3_0.6B_sft_next_token_masked_loss"
logging.info("Setting up training arguments...")
training_args = transformers.TrainingArguments(
    output_dir=OUTPUT_PATH,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2, # Adjust based on GPU memory
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8, # Effective batch size = 2 * 8 = 16
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=1, # Save last 1 checkpoints
    learning_rate=2e-5, # Common for fine-tuning
    weight_decay=0.01,
    warmup_ratio=0.03, # Warmup steps as a ratio of total training steps
    lr_scheduler_type="cosine",
    # fp16=True, # If using GPUs that benefit more from fp16 (e.g. T4, V100)
    bf16=True if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else False, # Ampere or newer for bf16
    report_to="wandb",
    remove_unused_columns=True, # Important, trainer will remove columns not expected by model.forward
    # torch_compile=True, # Requires PyTorch 2.0+, can speed up training, but can also have issues
    logging_dir=f'{OUTPUT_PATH}/logs', # This will still create local logs for TensorBoard if also specified
)

# --- 6. Initialize Trainer ---
logging.info("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset, # Can be None
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# --- 7. Start Training ---
logging.info("\n--- Starting training ---")
try:
    trainer.train()
    logging.info("--- Training finished ---")

    # --- 8. Save the final model and tokenizer ---
    logging.info("Saving final model and tokenizer...")
    final_save_path = os.path.join(OUTPUT_PATH, "final_model")
    trainer.save_model(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    logging.info(f"Model and tokenizer saved to {final_save_path}")

except Exception as e:
    logging.error(f"An error occurred during training: {e}", exc_info=True)

# Manually evaluate
eval_results = trainer.evaluate()
print(eval_results)

print("\nScript completed.")