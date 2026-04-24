from datasets import load_dataset, concatenate_datasets, DatasetDict
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling # DataCollatorForLanguageModeling not explicitly used if labels are precomputed
import torch
import logging
import os
import wandb

# before running this:
# export WANDB_PROJECT="mnlp_m2"
# export WANDB_RUN_NAME="0.6B_sft_chat_template_run_1" # Changed run name

# Set up logging
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_chat_template.log') # Use abspath for robustness

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
questa è la lista aggiornata:
DATASET_NAMES = [
    "camel-ai/amc_aime_self_improving",
    "camel-ai/amc_aime_distilled",
    "camel-ai/gsm8k_distilled",
    "camel-ai/chemistry",
    "math-ai/StackMathQA",
    "Azure99/stackoverflow-qa-top-300k",
    "0xZee/dataset-CoT-Quantum-Mechanics-1224",
    "ayoubkirouane/arxiv-physics",
    "EmaRimoldi/physics-stackexchange-reduced",
    "EmaRimoldi/camel-ai-code-cleaned",
]

# Define how to extract prompt and answer from each dataset
# For 'code' dataset, message_1 and message_2 are lists of dicts, e.g. [{'role':'user', 'content':'...'}], we take the first.
# Added "is_chat_format" and "content_key" for datasets that need specific extraction.
# Note: The original script had "camel-ai/math" and "camel-ai/physics" in comments but not in DATASET_NAMES.
# If you use them, ensure their configs are correct.
DATASET_CONFIGS = {
    "camel-ai/amc_aime_self_improving": {"prompt_col": "problem", "answer_col": "groud_truth_solution"},
    "camel-ai/amc_aime_distilled": {"prompt_col": "problem", "answer_col": "groud_truth_solution"},
    "camel-ai/gsm8k_distilled": {"prompt_col": "problem", "answer_col": "groud_truth_solution"},
    "camel-ai/chemistry": {"prompt_col": "message_1", "answer_col": "message_2"},
    "math-ai/StackMathQA": {"prompt_col": "Q", "answer_col": "A"},
    "Azure99/stackoverflow-qa-top-300k": {"prompt_col": "body", "answer_col": "answer_body"},
    "0xZee/dataset-CoT-Quantum-Mechanics-1224": {"prompt_col": "question", "answer_col": "response"},
    "ayoubkirouane/arxiv-physics": {"prompt_col": "question", "answer_col": "answer"},
    "EmaRimoldi/physics-stackexchange-reduced": {"prompt_col": "question", "answer_col": "answer"},
    "EmaRimoldi/camel-ai-code-cleaned": {"prompt_col": "question", "answer_col": "answer"},
}


DATASET_SAMPLE_SIZES = {
    "camel-ai/amc_aime_self_improving": -1,
    "camel-ai/amc_aime_distilled": -1,
    "camel-ai/gsm8k_distilled": -1,
    "camel-ai/chemistry": 10000,
    "math-ai/StackMathQA": 30000,
    "Azure99/stackoverflow-qa-top-300k": 20000,
    "0xZee/dataset-CoT-Quantum-Mechanics-1224": -1,
    "ayoubkirouane/arxiv-physics": 20000,  
    "EmaRimoldi/physics-stackexchange-reduced": -1,  
    "EmaRimoldi/camel-ai-code-cleaned": 20000,
}



MAX_SEQ_LENGTH = 1024 # Define max sequence length
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant who solves mathematical and scientific problems." # You can customize this

# --- 1. Load Tokenizer and Model ---
print(f"Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set tokenizer.pad_token to {tokenizer.eos_token} ({tokenizer.pad_token_id})")
# Qwen models usually have eos_token set (e.g. <|endoftext|>)
if tokenizer.eos_token is None:
    tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
    print(f"Manually added eos_token: {tokenizer.eos_token} ({tokenizer.eos_token_id})")

# Display the chat template to verify it's loaded
print(f"Tokenizer chat template: {tokenizer.chat_template}")
if tokenizer.chat_template is None:
    logger.warning(f"Tokenizer for {MODEL_NAME} does not have a `chat_template` attribute. "
                   "Proceeding without it might lead to suboptimal results or errors if apply_chat_template is strictly needed. "
                   "Consider setting one manually based on Qwen's documentation if issues arise.")
    # Example manual set for Qwen if needed (usually not necessary with trust_remote_code=True for recent Qwen models):
    # QWEN_CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}{% elif message['role'] == 'user' %}{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}{% elif message['role'] == 'assistant' %}{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    # tokenizer.chat_template = QWEN_CHAT_TEMPLATE
    # logger.info("Manually set a chat template for Qwen.")


print(f"Loading model {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
model.config.pad_token_id = tokenizer.pad_token_id
model.resize_token_embeddings(len(tokenizer)) # Important if pad_token was added or other special tokens
print("Model and tokenizer loaded.")

PREPROCESS = True
if PREPROCESS:
    # --- 2. Load and Preprocess Datasets ---
    
    def preprocess_function(examples, tokenizer_ref, dataset_name_key):
        logger.info(f"Preprocessing batch for dataset: {dataset_name_key} using chat template.")
        config = DATASET_CONFIGS.get(dataset_name_key)
        if config is None:
            logger.error(f'Missing config for dataset: {dataset_name_key}')
            return {'input_ids': [], 'attention_mask': [], 'labels': []}
    
        prompts_raw = examples[config["prompt_col"]]
        answers_raw = examples[config["answer_col"]]
    
        prompts = []
        answers = []
    
        # This part extracts the string content from potentially nested structures
        if config.get("is_chat_format", False):
            content_key = config.get("content_key", "content") # Default to "content" if not specified
            for p_list in prompts_raw:
                if p_list and isinstance(p_list, list) and len(p_list) > 0 and isinstance(p_list[0], dict) and content_key in p_list[0]:
                    prompts.append(p_list[0][content_key])
                elif isinstance(p_list, str): # Handle case where it's already a string
                     prompts.append(p_list)
                else:
                    prompts.append("")
                    logger.warning(f"Malformed or empty prompt in {dataset_name_key}: {p_list}")
            for a_list in answers_raw:
                if a_list and isinstance(a_list, list) and len(a_list) > 0 and isinstance(a_list[0], dict) and content_key in a_list[0]:
                    answers.append(a_list[0][content_key])
                elif isinstance(a_list, str): # Handle case where it's already a string
                     answers.append(a_list)
                else:
                    answers.append("")
                    logger.warning(f"Malformed or empty answer in {dataset_name_key}: {a_list}")
        else:
            prompts = prompts_raw
            answers = answers_raw
    
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
    
        for prompt_str, answer_str in zip(prompts, answers):
            if not isinstance(prompt_str, str) or not isinstance(answer_str, str):
                logger.error(f"Skipping non-string prompt/answer in {dataset_name_key}. Prompt: {type(prompt_str)}, Answer: {type(answer_str)}")
                continue
            if not prompt_str.strip() or not answer_str.strip():
                logger.warning(f"Skipping empty prompt or answer in {dataset_name_key}. Prompt: '{prompt_str[:50]}...', Answer: '{answer_str[:50]}...'")
                continue
    
            # Ensure the answer content for the assistant ends with EOS.
            # The chat template usually handles turn separators like <|im_end|>,
            # but the model needs to learn to output its specific EOS token at the very end.
            answer_with_eos = answer_str + tokenizer_ref.eos_token
    
            messages = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_str},
                {"role": "assistant", "content": answer_with_eos}
            ]
    
            # 1. Format the full conversation string using the chat template
            #    `add_generation_prompt=False` because we are providing the full assistant response.
            try:
                formatted_full_text = tokenizer_ref.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            except Exception as e:
                logger.error(f"Error applying chat template for full text in {dataset_name_key}: {e}. Prompt: '{prompt_str[:50]}...', Answer: '{answer_str[:50]}...'. Skipping.")
                logger.info(f"Tokenizer chat template being used: {tokenizer_ref.chat_template}")
                continue
    
    
            # 2. Tokenize the formatted full string to get input_ids and attention_mask
            tokenized_full = tokenizer_ref(
                formatted_full_text,
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                padding='max_length',
            )
            input_ids_item = tokenized_full['input_ids']
            attention_mask_item = tokenized_full['attention_mask']
    
            # 3. Create labels as a copy of input_ids, then mask
            labels_item = list(input_ids_item)
    
            # 4. Determine the length of the context to mask (system + user + assistant's template prefix)
            #    Format the conversation up to the point where the assistant *would* start generating.
            context_messages = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_str}
                # No assistant message here; add_generation_prompt=True will add the assistant's turn prefix (e.g., "<|im_start|>assistant\n")
            ]
            try:
                formatted_context_only_text = tokenizer_ref.apply_chat_template(
                    context_messages,
                    tokenize=False,
                    add_generation_prompt=True # Crucial: this adds the prefix for the assistant's turn
                )
            except Exception as e:
                logger.error(f"Error applying chat template for context_only text in {dataset_name_key}: {e}. Prompt: '{prompt_str[:50]}...'. Skipping.")
                continue
    
    
            # Tokenize this context string. `add_special_tokens=False` is important here if the template
            # itself already adds BOS tokens, to avoid miscalculating context_len.
            # Qwen's template often includes {{ bos_token }}.
            tokenized_context_only = tokenizer_ref(formatted_context_only_text, add_special_tokens=False)
            context_len = len(tokenized_context_only['input_ids'])
    
            # 5. Mask labels for the context part and padding part
            for i in range(len(labels_item)):
                if i < context_len:
                    labels_item[i] = -100  # Mask context tokens
                elif input_ids_item[i] == tokenizer_ref.pad_token_id:
                    labels_item[i] = -100  # Mask padding tokens
            
            if context_len >= MAX_SEQ_LENGTH:
                logger.warning(
                    f"Context length ({context_len}) after applying chat template is >= max_seq_length ({MAX_SEQ_LENGTH}) "
                    f"for an example in {dataset_name_key}. All labels for this instance will be -100."
                )
                # The filter_empty_examples function later should catch and remove this.
    
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
    
    for i, dataset_name_key in enumerate(DATASET_NAMES):
        logging.info(f"\nProcessing dataset {i+1}/{len(DATASET_NAMES)}: {dataset_name_key}")
        try:
            # Try to load 'train' split, fallback to first available split
            try:
                dataset = load_dataset(dataset_name_key, split="train", trust_remote_code=True, num_proc=os.cpu_count() // 2 or 1)
            except Exception as e_train:
                logging.warning(f"Could not load 'train' split for {dataset_name_key}: {e_train}. Trying to load the first available split.")
                try:
                    dataset_info = load_dataset(dataset_name_key, trust_remote_code=True) # Load info to find splits
                    first_split = list(dataset_info.keys())[0]
                    dataset = load_dataset(dataset_name_key, split=first_split, trust_remote_code=True, num_proc=os.cpu_count() // 2 or 1)
                    logging.info(f"Loaded split '{first_split}' for {dataset_name_key}.")
                except Exception as e_fallback:
                    logging.error(f"ERROR: Could not load any split for dataset {dataset_name_key}: {e_fallback}")
                    logging.error(f"Skipping this dataset.")
                    continue
            
            logging.info(f"Starting data preprocessing on CPU for {dataset_name_key}...")
            current_remove_columns = list(dataset.column_names)

            sample_size = DATASET_SAMPLE_SIZES.get(dataset_name_key, -1)  # Default to -1 if not found

            if sample_size == -1:
                logger.info(f"Using all samples for {dataset_name_key}")
            elif sample_size < len(dataset):
                logger.info(f"Sampling {sample_size} examples from {dataset_name_key}")
                dataset = dataset.shuffle(seed=42).select(range(sample_size))
            else:
                logger.warning(f"Requested sample size {sample_size} exceeds dataset size ({len(dataset)}). Using full dataset.")



            tokenized_dataset = dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=current_remove_columns,
                num_proc=os.cpu_count() // 2 or 1,
                fn_kwargs={'tokenizer_ref': tokenizer, 'dataset_name_key': dataset_name_key}
            )
            
            def filter_empty_examples(example):
                has_input = len(example['input_ids']) > 0
                has_valid_labels = any(label != -100 for label in example['labels'])
                if not has_valid_labels and has_input: # Log if input exists but all labels are masked
                     logger.debug(f"Filtering example from {dataset_name_key} because all labels were masked. Input IDs start: {example['input_ids'][:10]}")
                return has_input and has_valid_labels
    
            original_len = len(tokenized_dataset)
            tokenized_dataset = tokenized_dataset.filter(filter_empty_examples, num_proc=os.cpu_count() // 2 or 1)
            filtered_len = len(tokenized_dataset)
            
            if original_len > filtered_len:
                logger.warning(f"Filtered out {original_len - filtered_len} examples from {dataset_name_key} due to empty inputs or no target labels after chat template processing.")
    
            if filtered_len > 0:
                logger.info(f"Tokenized {dataset_name_key}. Number of examples: {len(tokenized_dataset)}")
                all_tokenized_datasets.append(tokenized_dataset)
            else:
                logger.warning(f"No valid examples remaining for {dataset_name_key} after preprocessing and filtering. Skipping this dataset.")
    
        except Exception as e:
            logging.error(f"ERROR processing dataset {dataset_name_key}: {e}", exc_info=True)
            logging.error(f"Skipping this dataset.")
    
    
    if not all_tokenized_datasets:
        logging.error("No datasets were successfully processed. Exiting.")
        exit()
    
    # --- 3. Combine Datasets ---
    print("\nCombining all processed datasets...")
    if len(all_tokenized_datasets) > 1:
        final_dataset_train = concatenate_datasets(all_tokenized_datasets)
    elif len(all_tokenized_datasets) == 1:
        final_dataset_train = all_tokenized_datasets[0]
    else: # Should have exited if no datasets, but as a safeguard
        logging.error("No datasets to combine. This should not happen if pre-check passed. Exiting.")
        exit()
    
    logging.info(f"Total number of tokenized examples in combined training dataset: {len(final_dataset_train)}")
    
    DEBUG = False # Set to True to inspect data
    if DEBUG and final_dataset_train and len(final_dataset_train) > 0:
        print("\n--- Inspecting processed data structure (first 2 examples) ---")
        num_examples_to_check = min(2, len(final_dataset_train))
        for i in range(num_examples_to_check):
            example = final_dataset_train[i]
            print(f"\nExample {i}:")
            print(f"  Input IDs: {example['input_ids'][:50]}... (len: {len(example['input_ids'])})")
            # print(f"  Decoded Input: {tokenizer.decode(example['input_ids'])}") # Can be very long
            print(f"  Labels:    {example['labels'][:50]}... (len: {len(example['labels'])})")
            
            # Decode only the part where labels are not -100
            try:
                target_tokens = [tok for tok, lab in zip(example['input_ids'], example['labels']) if lab != -100]
                if target_tokens:
                     print(f"  Decoded Target Labels: {tokenizer.decode(target_tokens)}")
                else:
                     print("  No target labels to decode (all masked).")
            except Exception as e:
                print(f"  Error decoding target labels: {e}")
                
            print(f"  Attention Mask: {example['attention_mask'][:50]}... (len: {len(example['attention_mask'])})")
            print("-" * 20)
        print("--- End of data inspection ---\n")
    
    
    # Create a DatasetDict for the Trainer
    processed_datasets = DatasetDict()
    processed_datasets["train"] = final_dataset_train
    
    # --- Save the complete dataset to disk ---
    processed_datasets.save_to_disk("sft_final_data_red")
else:
    from datasets import load_from_disk
    processed_datasets = load_from_disk("sft_final_data_red")
    logging.info("Dataset loaded from disk")


if len(processed_datasets["train"]) > 100: # Min samples for a meaningful split
    logging.info("Splitting combined dataset into train and validation (95/5)...")
    train_test_split = processed_datasets["train"].train_test_split(test_size=0.05, seed=42)
    processed_datasets["train"] = train_test_split["train"]
    processed_datasets["eval"] = train_test_split["test"]
    logging.info(f"Train dataset size: {len(processed_datasets['train'])}")
    logging.info(f"Eval dataset size: {len(processed_datasets['eval'])}")
else:
    logging.warning("Not enough data to create a validation split (less than 100 samples in train). Trainer will run without evaluation.")
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
    num_train_epochs=2,
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