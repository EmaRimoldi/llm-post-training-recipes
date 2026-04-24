from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
import logging
import os
import wandb
import json
import pandas as pd

# before running this:
# export WANDB_PROJECT="mnlp_m3"
# export WANDB_RUN_NAME="0.6B_mcqa_ft_0"

# Set up logging
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_chat_template.log')

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

#define output path
OUTPUT_PATH = "./qwen3_0.6B_mcqa_sft"

# Define model and dataset names
MODEL_NAME = "MoroM02/MoroM02"

# Your MCQA datasets on Hugging Face (all should have question, choices, answer format)
# UPDATE: Replace with your preprocessed MedMCQA dataset
MCQA_DATASETS = [
    "VinceEPFL/aimcqs-mcqa",  # Your uploaded AIMCQS dataset
    "VinceEPFL/medmcqa-standard",  # Your preprocessed MedMCQA dataset
    
    "ema1234/MNLP_M2_quantized_dataset",
    # "username/another-mcqa-dataset",    # Add more MCQA datasets here
    # "username/yet-another-mcqa",        # All should have same format
]

# Regular text datasets (if you want to include any)
REGULAR_DATASETS = [
    # "math-ai/StackMathQA",
    # Add more regular Q&A datasets here
]

# All datasets combined
DATASET_NAMES = MCQA_DATASETS + REGULAR_DATASETS

# Dataset configurations - now all MCQA datasets use the same format
MCQA_DATASET_CONFIGS = {
    "VinceEPFL/aimcqs-mcqa": {
        "prompt_col": "question",
        "choices_col": "choices",
        "answer_col": "answer",
        "is_mcqa": True,
    },
    "VinceEPFL/medmcqa-standard": {  # Same format as AIMCQS now!
        "prompt_col": "question",
        "choices_col": "choices",
        "answer_col": "answer",
        "is_mcqa": True,
    },
    "ema1234/MNLP_M2_quantized_dataset": {
        "prompt_col": "question",
        "choices_col": "options",  # Note: this dataset uses "options" instead of "choices"
        "answer_idx_col": "answer_idx",  # Uses index instead of letter
        "is_mcqa": True,
    },
}

REGULAR_DATASET_CONFIGS = {
    # "math-ai/StackMathQA": {"prompt_col": "Q", "answer_col": "A", "is_mcqa": False},
    # "Azure99/stackoverflow-qa-top-300k": {"prompt_col": "body", "answer_col": "answer_body", "is_mcqa": False},
    # Add more regular datasets here
}

# Combine all configs
DATASET_CONFIGS = {**MCQA_DATASET_CONFIGS, **REGULAR_DATASET_CONFIGS}

DATASET_SAMPLE_SIZES = {
    # MCQA datasets
    "VinceEPFL/aimcqs-mcqa": -1,  # Use all data from your AIMCQS
    "VinceEPFL/medmcqa-standard": 1000,  # Sample size for preprocessed MedMCQA
    "ema1234/MNLP_M2_quantized_dataset": 4500,  # Sample from MNLP dataset
    # "username/another-mcqa-dataset": 5000,  # Sample size for other MCQA datasets
    
    # Regular datasets  
    # "math-ai/StackMathQA": 1000,
    # Add more as needed
}

MAX_SEQ_LENGTH = 1024
# Remove the system prompt since we're using direct MMLU format
DEFAULT_SYSTEM_PROMPT = ""

# --- 1. Load Tokenizer and Model ---
print(f"Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set tokenizer.pad_token to {tokenizer.eos_token} ({tokenizer.pad_token_id})")

if tokenizer.eos_token is None:
    tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
    print(f"Manually added eos_token: {tokenizer.eos_token} ({tokenizer.eos_token_id})")

print(f"Tokenizer chat template: {tokenizer.chat_template}")
if tokenizer.chat_template is None:
    logger.warning(f"Tokenizer for {MODEL_NAME} does not have a `chat_template` attribute.")

print(f"Loading model {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
model.config.pad_token_id = tokenizer.pad_token_id
model.resize_token_embeddings(len(tokenizer))
print("Model and tokenizer loaded.")

PREPROCESS = True
if PREPROCESS:
    # --- 2. Load and Preprocess Datasets ---
    
    def preprocess_mcqa_function(examples, tokenizer_ref, dataset_name_key):
        """
        Preprocessing for MCQA datasets from Hugging Face.
        Expected format: question, choices (list), answer (letter)
        Training format: question + choices + "Answer:" + correct choice
        """
        logger.info(f"Preprocessing MCQA batch for {dataset_name_key} with {len(examples['question'])} examples")
        
        config = DATASET_CONFIGS.get(dataset_name_key)
        
        questions = examples['question']
        
        # Handle different choice column names
        choices_col = config.get("choices_col", "choices")
        choices_lists = examples[choices_col]
        
        # Handle answer format - either letter or index
        if "answer_idx_col" in config:
            # Convert index to letter for datasets like MNLP
            answer_indices = examples[config["answer_idx_col"]]
            answer_letters = []
            LETTER_INDICES = ["A", "B", "C", "D"]
            for idx in answer_indices:
                if idx < len(LETTER_INDICES):
                    answer_letters.append(LETTER_INDICES[idx])
                else:
                    answer_letters.append("A")  # Default fallback
        else:
            # Already in letter format
            answer_letters = examples[config["answer_col"]]
        
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        
        LETTER_INDICES = ["A", "B", "C", "D"]
        
        for question, choices, answer_letter in zip(questions, choices_lists, answer_letters):
            if not isinstance(question, str) or not choices or not answer_letter:
                continue
            if not question.strip() or not answer_letter.strip():
                continue
            
            # Format exactly like mmlu_harness evaluation
            topic = "knowledge and skills in advanced master-level STEM courses"
            prompt = f"The following are multiple choice questions (with answers) about {topic}.\n\n"
            prompt += question + "\n"
            
            # Handle choices format - could be list of strings or already formatted
            if isinstance(choices, list) and len(choices) >= 4:
                formatted_choices = []
                for key, choice in zip(LETTER_INDICES, choices[:4]):
                    choice_text = choice
                    # Remove existing letter prefix if present
                    if choice.strip().lower().startswith(f"{key.lower()}."):
                        choice_text = choice.strip()[2:].strip()
                    elif choice.strip().lower().startswith(f"{key.lower()})"):
                        choice_text = choice.strip()[2:].strip()
                    
                    formatted_choice = f"{key}. {choice_text}"
                    formatted_choices.append(formatted_choice)
                    prompt += formatted_choice + "\n"
            else:
                continue  # Skip if not enough choices
            
            prompt += "Answer:"
            
            # Find the correct choice text
            answer_letter_upper = answer_letter.upper()
            if answer_letter_upper in LETTER_INDICES:
                choice_idx = LETTER_INDICES.index(answer_letter_upper)
                if choice_idx < len(formatted_choices):
                    # Format answer like evaluation expects: " A. choice_text"
                    answer_text = f" {formatted_choices[choice_idx]}"
                else:
                    continue
            else:
                continue
            
            # Create full text: prompt + answer
            full_text = prompt + answer_text
            
            # Tokenize the full sequence
            tokenized_full = tokenizer_ref(
                full_text,
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                padding='max_length',
            )
            input_ids_item = tokenized_full['input_ids']
            attention_mask_item = tokenized_full['attention_mask']
            
            # Create labels - mask everything except the answer part
            labels_item = list(input_ids_item)
            
            # Tokenize just the prompt to find where answer starts
            tokenized_prompt = tokenizer_ref(prompt, add_special_tokens=False)
            prompt_len = len(tokenized_prompt['input_ids'])
            
            # Mask everything except the answer tokens
            for i in range(len(labels_item)):
                if i < prompt_len:
                    labels_item[i] = -100  # Mask prompt
                elif input_ids_item[i] == tokenizer_ref.pad_token_id:
                    labels_item[i] = -100  # Mask padding
                # Everything else (the answer) keeps its original token ID
            
            if prompt_len >= MAX_SEQ_LENGTH:
                logger.warning(f"Prompt too long ({prompt_len}) for {dataset_name_key}, all labels masked")
            
            batch_input_ids.append(input_ids_item)
            batch_attention_mask.append(attention_mask_item)
            batch_labels.append(labels_item)
        
        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
        }
    
    def preprocess_standard_function(examples, tokenizer_ref, dataset_name_key):
        """
        Standard preprocessing for HuggingFace datasets (same as your original).
        """
        logger.info(f"Preprocessing batch for dataset: {dataset_name_key}")
        config = DATASET_CONFIGS.get(dataset_name_key)
        if config is None:
            logger.error(f'Missing config for dataset: {dataset_name_key}')
            return {'input_ids': [], 'attention_mask': [], 'labels': []}

        prompts_raw = examples[config["prompt_col"]]
        answers_raw = examples[config["answer_col"]]

        prompts = []
        answers = []

        if config.get("is_chat_format", False):
            content_key = config.get("content_key", "content")
            for p_list in prompts_raw:
                if p_list and isinstance(p_list, list) and len(p_list) > 0 and isinstance(p_list[0], dict) and content_key in p_list[0]:
                    prompts.append(p_list[0][content_key])
                elif isinstance(p_list, str):
                     prompts.append(p_list)
                else:
                    prompts.append("")
            for a_list in answers_raw:
                if a_list and isinstance(a_list, list) and len(a_list) > 0 and isinstance(a_list[0], dict) and content_key in a_list[0]:
                    answers.append(a_list[0][content_key])
                elif isinstance(a_list, str):
                     answers.append(a_list)
                else:
                    answers.append("")
        else:
            prompts = prompts_raw
            answers = answers_raw

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for prompt_str, answer_str in zip(prompts, answers):
            if not isinstance(prompt_str, str) or not isinstance(answer_str, str):
                continue
            if not prompt_str.strip() or not answer_str.strip():
                continue

            answer_with_eos = answer_str + tokenizer_ref.eos_token

            messages = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_str},
                {"role": "assistant", "content": answer_with_eos}
            ]

            try:
                formatted_full_text = tokenizer_ref.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            except Exception as e:
                continue

            tokenized_full = tokenizer_ref(
                formatted_full_text,
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                padding='max_length',
            )
            input_ids_item = tokenized_full['input_ids']
            attention_mask_item = tokenized_full['attention_mask']
            labels_item = list(input_ids_item)

            context_messages = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_str}
            ]
            try:
                formatted_context_only_text = tokenizer_ref.apply_chat_template(
                    context_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                continue

            tokenized_context_only = tokenizer_ref(formatted_context_only_text, add_special_tokens=False)
            context_len = len(tokenized_context_only['input_ids'])

            for i in range(len(labels_item)):
                if i < context_len:
                    labels_item[i] = -100
                elif input_ids_item[i] == tokenizer_ref.pad_token_id:
                    labels_item[i] = -100

            batch_input_ids.append(input_ids_item)
            batch_attention_mask.append(attention_mask_item)
            batch_labels.append(labels_item)

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
        }
    
    all_tokenized_datasets = []
    print("Loading and preprocessing datasets...")
    
    # Process all datasets (both MCQA and regular)
    for i, dataset_name_key in enumerate(DATASET_NAMES):
        logging.info(f"\nProcessing dataset {i+1}/{len(DATASET_NAMES)}: {dataset_name_key}")
        
        config = DATASET_CONFIGS.get(dataset_name_key)
        if config is None:
            logging.error(f"No config found for {dataset_name_key}, skipping")
            continue
            
        is_mcqa = config.get("is_mcqa", False)
        
        try:
            # Load dataset from HuggingFace
            try:
                dataset = load_dataset(dataset_name_key, split="train", trust_remote_code=True, num_proc=os.cpu_count() // 2 or 1)
            except Exception as e_train:
                logging.warning(f"Could not load 'train' split for {dataset_name_key}: {e_train}")
                try:
                    dataset_info = load_dataset(dataset_name_key, trust_remote_code=True)
                    first_split = list(dataset_info.keys())[0]
                    dataset = load_dataset(dataset_name_key, split=first_split, trust_remote_code=True, num_proc=os.cpu_count() // 2 or 1)
                    logging.info(f"Loaded split '{first_split}' for {dataset_name_key}.")
                except Exception as e_fallback:
                    logging.error(f"ERROR: Could not load any split for dataset {dataset_name_key}: {e_fallback}")
                    continue

            # No special preprocessing needed for MedMCQA anymore!
            current_remove_columns = list(dataset.column_names)
            sample_size = DATASET_SAMPLE_SIZES.get(dataset_name_key, -1)

            if sample_size == -1:
                logger.info(f"Using all samples for {dataset_name_key}")
            elif sample_size < len(dataset):
                logger.info(f"Sampling {sample_size} examples from {dataset_name_key}")
                dataset = dataset.shuffle(seed=42).select(range(sample_size))
            else:
                logger.warning(f"Requested sample size {sample_size} exceeds dataset size ({len(dataset)})")

            # Choose preprocessing function based on dataset type
            if is_mcqa:
                logging.info(f"Processing {dataset_name_key} as MCQA dataset")
                tokenized_dataset = dataset.map(
                    preprocess_mcqa_function,
                    batched=True,
                    remove_columns=current_remove_columns,
                    num_proc=os.cpu_count() // 2 or 1,
                    fn_kwargs={'tokenizer_ref': tokenizer, 'dataset_name_key': dataset_name_key}
                )
            else:
                logging.info(f"Processing {dataset_name_key} as regular dataset")
                tokenized_dataset = dataset.map(
                    preprocess_standard_function,
                    batched=True,
                    remove_columns=current_remove_columns,
                    num_proc=os.cpu_count() // 2 or 1,
                    fn_kwargs={'tokenizer_ref': tokenizer, 'dataset_name_key': dataset_name_key}
                )
            
            def filter_empty_examples(example):
                has_input = len(example['input_ids']) > 0
                has_valid_labels = any(label != -100 for label in example['labels'])
                return has_input and has_valid_labels

            original_len = len(tokenized_dataset)
            tokenized_dataset = tokenized_dataset.filter(filter_empty_examples, num_proc=os.cpu_count() // 2 or 1)
            filtered_len = len(tokenized_dataset)
            
            if original_len > filtered_len:
                logger.warning(f"Filtered out {original_len - filtered_len} examples from {dataset_name_key}")

            if filtered_len > 0:
                logger.info(f"Successfully processed {dataset_name_key}. Examples: {filtered_len}")
                all_tokenized_datasets.append(tokenized_dataset)
            else:
                logger.warning(f"No valid examples remaining for {dataset_name_key}")

        except Exception as e:
            logging.error(f"ERROR processing dataset {dataset_name_key}: {e}", exc_info=True)

    if not all_tokenized_datasets:
        logging.error("No datasets were successfully processed. Exiting.")
        exit()

    # --- 3. Combine Datasets ---
    print("\nCombining all processed datasets...")
    if len(all_tokenized_datasets) > 1:
        final_dataset_train = concatenate_datasets(all_tokenized_datasets)
    elif len(all_tokenized_datasets) == 1:
        final_dataset_train = all_tokenized_datasets[0]
    else:
        logging.error("No datasets to combine. Exiting.")
        exit()

    logging.info(f"Total number of tokenized examples in combined training dataset: {len(final_dataset_train)}")
    
    # Create a DatasetDict for the Trainer
    processed_datasets = DatasetDict()
    processed_datasets["train"] = final_dataset_train
    
    # --- Save the complete dataset to disk ---
    processed_datasets.save_to_disk("sft_final_data_mcqa_from_hf")
else:
    from datasets import load_from_disk
    processed_datasets = load_from_disk("sft_final_data_mcqa_from_hf")
    logging.info("Dataset loaded from disk")

if len(processed_datasets["train"]) > 100:
    logging.info("Splitting combined dataset into train and validation (95/5)...")
    train_test_split = processed_datasets["train"].train_test_split(test_size=0.05, seed=42)
    processed_datasets["train"] = train_test_split["train"]
    processed_datasets["eval"] = train_test_split["test"]
    logging.info(f"Train dataset size: {len(processed_datasets['train'])}")
    logging.info(f"Eval dataset size: {len(processed_datasets['eval'])}")
else:
    logging.warning("Not enough data for validation split. Training without evaluation.")
    processed_datasets["eval"] = None

train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["eval"]

# --- 4. Set up Data Collator ---
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
logging.info("Data collator set up.")

# --- 5. Set up Training Arguments ---

logging.info("Setting up training arguments...")
training_args = transformers.TrainingArguments(
    output_dir=OUTPUT_PATH,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=1,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else False,
    report_to="wandb",
    remove_unused_columns=True,
    logging_dir=f'{OUTPUT_PATH}/logs',
)

# --- 6. Initialize Trainer ---
logging.info("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
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

# Evaluate the model
if eval_dataset is not None:
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)

print("\nScript completed.")