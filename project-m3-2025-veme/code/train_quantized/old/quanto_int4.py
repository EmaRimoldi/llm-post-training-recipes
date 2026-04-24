import logging
import json
import re
import ast
import os
from pathlib import Path

# --- Key Change: Force single GPU to avoid DataParallel issues with quanto ---
# This will make PyTorch see only GPU 0. Adjust '0' if you want to use a different specific GPU.
# If you have only one GPU, this effectively ensures DataParallel is not used.
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    QuantoConfig
)
import torch.nn.functional as F

# --- 0. Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. Configuration ---
# IMPORTANT: "Qwen/Qwen3-0.6B-Base" was not found on Hugging Face Hub at the time of the original error.
# Using "Qwen/Qwen2-0.5B" as a placeholder from your original script's comments.
# PLEASE REPLACE THIS WITH YOUR CORRECT MODEL IDENTIFIER OR LOCAL PATH.
MODEL_ID = "Qwen/Qwen3-0.6B-Base" # Or your "Qwen/Qwen3-0.6B-Base" if available/local
# Example if local: MODEL_ID = "./path_to_my_qwen3_0.6b_base_model/"

DATA_FILE_PATH = Path("../../data/mathqa/train.json") # Make sure this path is correct
OUTPUT_DIR = Path("./qwen_qat_4bit_mcqa_quanto_mathqa_single_gpu") # Changed output dir name slightly

MAX_SEQ_LENGTH = 256
TRAIN_BATCH_SIZE = 1 # Per device; remains 1
EVAL_BATCH_SIZE = 1  # Per device
NUM_TRAIN_EPOCHS = 1
LEARNING_RATE = 3e-5
CONTRASTIVE_MARGIN = 1.0

# Global tokenizer, initialized in main
tokenizer = None

# --- User's Data Loading and Transformation Logic ---
def transform_loaded_data(examples):
    new_questions = []
    new_options_lists = []
    new_answer_indices = []
    new_num_options_list = []
    new_ids = []

    num_examples_in_batch = len(examples['Problem'])

    for i in range(num_examples_in_batch):
        problem_text = examples['Problem'][i]
        options_input = examples['options'][i]
        correct_char = examples['correct'][i].lower()
        current_id_val = examples.get('id', [None]*num_examples_in_batch)[i] or f"gen_id_{i}"
        parsed_options = []

        if isinstance(options_input, str):
            is_list_repr_successfully_parsed = False
            if options_input.startswith("[") and options_input.endswith("]"):
                try:
                    evaluated_list = ast.literal_eval(options_input)
                    if isinstance(evaluated_list, list):
                        for item_str in evaluated_list:
                            item_str_cleaned = str(item_str).strip()
                            match = re.match(r"^[a-zA-Z]\s*\)\s*(.*)", item_str_cleaned, re.DOTALL)
                            if match:
                                parsed_options.append(match.group(1).strip())
                            else:
                                if item_str_cleaned: parsed_options.append(item_str_cleaned)
                        is_list_repr_successfully_parsed = True
                except (ValueError, SyntaxError, TypeError): pass

            if not is_list_repr_successfully_parsed:
                option_matches = re.findall(
                    r"[a-zA-Z]\s*\)\s*(.*?)(?=\s*[a-zA-Z]\s*\)\s*|$)",
                    options_input, re.IGNORECASE | re.DOTALL
                )
                if option_matches:
                    parsed_options = [match.strip() for match in option_matches if match.strip()]
                elif options_input.strip() and not option_matches:
                    parsed_options = [] # No options found, but input string was not empty
        elif isinstance(options_input, list):
            for item_str_or_val in options_input:
                item_str = str(item_str_or_val).strip()
                match = re.match(r"^[a-zA-Z]\s*\)\s*(.*)", item_str, re.DOTALL)
                if match: parsed_options.append(match.group(1).strip())
                else: parsed_options.append(item_str)
        else:
            parsed_options = []

        answer_idx = -1
        if correct_char and 'a' <= correct_char <= 'z':
            answer_idx = ord(correct_char) - ord('a')

        # This ensures that examples filtered out later by filter_valid_transformed_data
        # don't cause issues here if they have fewer than 5 options from the start.
        # The actual filtering for 5 options happens in filter_valid_transformed_data.
        if not (0 <= answer_idx < len(parsed_options)):
             # Add empty lists and dummy values for consistency, will be filtered out
            new_questions.append(problem_text)
            new_options_lists.append([])
            new_answer_indices.append(-1)
            new_num_options_list.append(0)
            new_ids.append(current_id_val)
            continue


        new_questions.append(problem_text)
        new_options_lists.append(parsed_options)
        new_answer_indices.append(answer_idx)
        new_num_options_list.append(len(parsed_options))
        new_ids.append(current_id_val)

    return {
        "id": new_ids, "question": new_questions, "options": new_options_lists,
        "answer_idx": new_answer_indices, "num_options": new_num_options_list,
    }

def filter_valid_transformed_data(example):
    # MathQA specific: expecting 5 options.
    if not (example["options"] and example["num_options"] == 5): return False
    if not (0 <= example["answer_idx"] < example["num_options"]): return False
    return True

def preprocess_mcqa_contrastive(examples):
    global tokenizer, MAX_SEQ_LENGTH
    if tokenizer is None:
        raise ValueError("Tokenizer is not initialized before calling preprocess_mcqa_contrastive.")

    processed_batch = {
        "positive_input_ids": [], "positive_attention_mask": [], "positive_prompt_len": [],
        "negative_input_ids": [], "negative_attention_mask": [], "negative_prompt_len": [],
    }

    for i in range(len(examples['question'])):
        question = examples['question'][i]
        options = examples['options'][i] # Should be a list of 5 strings now
        answer_idx = examples['answer_idx'][i]
        num_options = examples['num_options'][i] # Should be 5

        if not (isinstance(options, list) and len(options) == num_options and num_options == 5 and 0 <= answer_idx < num_options):
            logger.debug(f"Skipping malformed MCQA example: ID {examples['id'][i] if 'id' in examples else 'N/A'} in contrastive preprocessing due to option/answer mismatch.")
            continue

        correct_option_text = options[answer_idx]
        incorrect_options_texts = [opt for idx, opt in enumerate(options) if idx != answer_idx]

        # Ensure tokenizer.eos_token is not None
        eos_token_str = tokenizer.eos_token if tokenizer.eos_token else ""

        prompt_positive_prefix = f"Question: {question} Option: "
        full_positive_text = prompt_positive_prefix + correct_option_text + eos_token_str
        tokenized_positive_prefix = tokenizer(prompt_positive_prefix, add_special_tokens=False, truncation=False)
        positive_prompt_len = len(tokenized_positive_prefix['input_ids'])
        tokenized_positive = tokenizer(full_positive_text, truncation=True, max_length=MAX_SEQ_LENGTH, padding=False)

        processed_batch["positive_input_ids"].append(tokenized_positive['input_ids'])
        processed_batch["positive_attention_mask"].append(tokenized_positive['attention_mask'])
        processed_batch["positive_prompt_len"].append(positive_prompt_len)

        current_negatives_input_ids = []
        current_negatives_attention_mask = []
        current_negatives_prompt_len = []
        for neg_opt_text in incorrect_options_texts:
            prompt_negative_prefix = f"Question: {question} Option: "
            full_negative_text = prompt_negative_prefix + neg_opt_text + eos_token_str
            tokenized_negative_prefix = tokenizer(prompt_negative_prefix, add_special_tokens=False, truncation=False)
            negative_prompt_len = len(tokenized_negative_prefix['input_ids'])
            tokenized_negative = tokenizer(full_negative_text, truncation=True, max_length=MAX_SEQ_LENGTH, padding=False)

            current_negatives_input_ids.append(tokenized_negative['input_ids'])
            current_negatives_attention_mask.append(tokenized_negative['attention_mask'])
            current_negatives_prompt_len.append(negative_prompt_len)

        processed_batch["negative_input_ids"].append(current_negatives_input_ids)
        processed_batch["negative_attention_mask"].append(current_negatives_attention_mask)
        processed_batch["negative_prompt_len"].append(current_negatives_prompt_len)
    return processed_batch

def filter_empty_contrastive(example):
    positive_ids = example.get('positive_input_ids')
    if not positive_ids or not isinstance(positive_ids, list) or len(positive_ids) == 0: return False

    all_negative_options_token_lists = example.get('negative_input_ids')
    if not all_negative_options_token_lists or not isinstance(all_negative_options_token_lists, list): return False
    # MathQA: 1 positive, 4 negatives
    if len(all_negative_options_token_lists) != 4: return False
    for single_negative_option_tokens in all_negative_options_token_lists:
        if not isinstance(single_negative_option_tokens, list) or len(single_negative_option_tokens) == 0: return False

    if not isinstance(example.get('positive_prompt_len'), int) or example.get('positive_prompt_len') < 0: return False
    neg_prompt_lens = example.get('negative_prompt_len')
    if not isinstance(neg_prompt_lens, list) or len(neg_prompt_lens) != 4: return False
    for l in neg_prompt_lens:
        if not isinstance(l, int) or l < 0 : return False
    return True

# --- Custom Data Collator (User Provided) ---
class DataCollatorForContrastiveMCQA:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.data_collator_lm = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def __call__(self, features):
        batch = {}
        positive_examples = [{"input_ids": f["positive_input_ids"], "attention_mask": f["positive_attention_mask"]} for f in features if "positive_input_ids" in f] # Basic check
        if not positive_examples: return {} # Should not happen with filtered data

        collated_positive = self.data_collator_lm(positive_examples)
        batch["positive_input_ids"] = collated_positive["input_ids"]
        batch["positive_attention_mask"] = collated_positive["attention_mask"]
        batch["positive_labels"] = collated_positive["labels"]
        batch["positive_prompt_len"] = [f["positive_prompt_len"] for f in features]

        num_negatives_per_prompt = 0
        if features and features[0].get("negative_input_ids") and isinstance(features[0]["negative_input_ids"], list):
            num_negatives_per_prompt = len(features[0]["negative_input_ids"]) # Should be 4

        batch_negative_input_ids = []
        batch_negative_attention_mask = []
        batch_negative_labels = []
        batch_negative_prompt_len = []

        for i in range(num_negatives_per_prompt):
            current_ith_negative_examples = []
            current_ith_negative_prompt_lens = []
            for f_idx, f in enumerate(features):
                if f.get("negative_input_ids") and i < len(f["negative_input_ids"]):
                    current_ith_negative_examples.append({
                        "input_ids": f["negative_input_ids"][i],
                        "attention_mask": f["negative_attention_mask"][i]
                    })
                    current_ith_negative_prompt_lens.append(f["negative_prompt_len"][i])
                else:
                    logger.debug(f"Feature {f_idx} missing negative option at index {i} during collation.")
                    # This case should ideally be filtered out before collation.
                    # If it happens, subsequent code might fail if batch sizes don't match.
                    # For robustness, one might add dummy/padded entries, but better to ensure data integrity upstream.

            if not current_ith_negative_examples:
                # This can happen if all examples in a batch were malformed for this negative index
                # Add empty tensors to maintain structure, though loss calculation should handle this.
                # This part needs careful handling based on how loss expects misaligned batches.
                # For now, we assume filtering works well. If it does, this branch shouldn't be hit often.
                logger.warning(f"No valid negative examples for negative index {i} in this batch.")
                # To avoid downstream errors, we might need to append appropriately shaped empty tensors
                # or ensure the loss function can handle variable numbers of negative sets.
                # For simplicity, we'll assume well-formed batches post-filtering.
                continue


            collated_negative_i = self.data_collator_lm(current_ith_negative_examples)
            batch_negative_input_ids.append(collated_negative_i["input_ids"])
            batch_negative_attention_mask.append(collated_negative_i["attention_mask"])
            batch_negative_labels.append(collated_negative_i["labels"])
            batch_negative_prompt_len.append(current_ith_negative_prompt_lens)

        batch["negative_input_ids"] = batch_negative_input_ids
        batch["negative_attention_mask"] = batch_negative_attention_mask
        batch["negative_labels"] = batch_negative_labels
        batch["negative_prompt_len"] = batch_negative_prompt_len
        return batch

# --- Custom Trainer (User Provided, slight modification for tokenizer warning) ---
class ContrastiveTrainer(Trainer):
    # Removed tokenizer from __init__ params to avoid FutureWarning,
    # it's accessible via self.tokenizer (from Trainer base) or self.data_collator.tokenizer
    def __init__(self, *args, margin=1.0, **kwargs):
        # The FutureWarning for `tokenizer` in `ContrastiveTrainer.__init__`
        # was likely because the base Trainer handles tokenizer initialization.
        # If `tokenizer` is passed here AND to TrainingArguments, it can be redundant.
        # We rely on the Trainer to correctly set up `self.tokenizer`.
        if 'tokenizer' in kwargs:
            # logger.info("Tokenizer passed to ContrastiveTrainer kwargs, removing to use default Trainer handling.")
            del kwargs['tokenizer'] # Base Trainer will get it from data_collator or args
        super().__init__(*args, **kwargs)
        self.margin = margin

    def _get_option_nll(self, model, input_ids, attention_mask, labels, prompt_lengths):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        batch_option_nll = []
        logits = outputs.logits

        shifted_logits = logits[..., :-1, :].contiguous()
        # labels are already shifted by DataCollatorForLanguageModeling
        shifted_labels = labels[..., 1:].contiguous() if labels.ndim > 1 and labels.size(-1) > 0 else labels


        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

        for i in range(input_ids.size(0)):
            prompt_len = prompt_lengths[i]
            # Check if current_labels is 1D (already shifted and sliced) or 2D (batch item)
            current_labels_i = shifted_labels[i]
            current_logits_i = shifted_logits[i]

            option_token_indices_start = max(0, prompt_len -1) # prompt_len is for original text

            option_token_indices = (current_labels_i != -100) & \
                                   (torch.arange(current_labels_i.size(0), device=current_labels_i.device) >= option_token_indices_start)

            if option_token_indices.sum() == 0:
                batch_option_nll.append(torch.tensor(0.0, device=model.device))
                continue

            option_labels = current_labels_i[option_token_indices]
            option_logits = current_logits_i[option_token_indices]

            nll_per_token = loss_fct(option_logits.view(-1, option_logits.size(-1)), option_labels.view(-1))
            sum_nll_option = nll_per_token.sum()
            batch_option_nll.append(sum_nll_option)

        return torch.stack(batch_option_nll)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None): # num_items_in_batch arg added
        positive_input_ids = inputs["positive_input_ids"]
        positive_attention_mask = inputs["positive_attention_mask"]
        positive_labels = inputs["positive_labels"]
        positive_prompt_len = inputs["positive_prompt_len"]

        nll_positive = self._get_option_nll(model, positive_input_ids, positive_attention_mask, positive_labels, positive_prompt_len)

        total_contrastive_loss = torch.tensor(0.0, device=model.device, requires_grad=True if model.training else False)
        num_valid_contrastive_pairs = 0

        if inputs.get("negative_input_ids") and len(inputs["negative_input_ids"]) > 0:
            num_negatives_sets = len(inputs["negative_input_ids"])

            for i in range(num_negatives_sets):
                if i >= len(inputs["negative_input_ids"]) or inputs["negative_input_ids"][i].nelement() == 0:
                    logger.debug(f"Skipping empty or missing negative set {i} in compute_loss.")
                    continue

                negative_input_ids = inputs["negative_input_ids"][i]
                negative_attention_mask = inputs["negative_attention_mask"][i]
                negative_labels = inputs["negative_labels"][i]
                negative_prompt_len_for_set_i = inputs["negative_prompt_len"][i]

                if nll_positive.size(0) != negative_input_ids.size(0):
                    logger.warning(f"Batch size mismatch in compute_loss between positive ({nll_positive.size(0)}) and negative set {i} ({negative_input_ids.size(0)}). Skipping this negative set.")
                    continue

                nll_negative = self._get_option_nll(model, negative_input_ids, negative_attention_mask, negative_labels, negative_prompt_len_for_set_i)

                for j in range(nll_positive.size(0)): # Iterate over items in the batch
                    # Ensure device consistency for the relu operation
                    current_nll_positive = nll_positive[j].to(model.device)
                    current_nll_negative = nll_negative[j].to(model.device)

                    if torch.isinf(current_nll_positive) or torch.isinf(current_nll_negative) or current_nll_positive == 0 or current_nll_negative == 0:
                        continue

                    loss_ij = F.relu(self.margin + current_nll_positive - current_nll_negative)
                    if total_contrastive_loss.device != loss_ij.device: # Ensure accumulation on same device
                         total_contrastive_loss = total_contrastive_loss.to(loss_ij.device)

                    total_contrastive_loss = total_contrastive_loss + loss_ij # In-place might cause issues with graph
                    num_valid_contrastive_pairs += 1

        final_loss = total_contrastive_loss / num_valid_contrastive_pairs if num_valid_contrastive_pairs > 0 else torch.tensor(0.0, device=model.device, requires_grad=True if model.training else False)

        return (final_loss, {"nll_positive_avg": nll_positive.mean().item() if nll_positive.numel() > 0 else 0.0}) if return_outputs else final_loss


# --- Main Script Logic ---
def main():
    global tokenizer, MAX_SEQ_LENGTH, MODEL_ID # Make them available to preprocessing functions

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("REMINDER: This script is configured for SINGLE GPU execution by default "
                "to avoid potential DataParallel issues with `quanto`.")
    logger.info("Ensure `CUDA_HOME` is set correctly in your environment if you encounter "
                "`quanto` CUDA kernel warnings (as seen in original logs).")


    # --- Initialize Tokenizer ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load tokenizer for {MODEL_ID}: {e}. "
                     "Please ensure the MODEL_ID is correct and accessible.")
        return

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set tokenizer pad_token to eos_token: {tokenizer.pad_token}")
        else:
            new_pad_token = '[PAD]'
            tokenizer.add_special_tokens({'pad_token': new_pad_token})
            logger.warning(f"Added new [PAD] token: {new_pad_token}")
    logger.info(f"Tokenizer pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    if tokenizer.eos_token is None: # Critical for formatting text
        new_eos_token = '<|endoftext|>'
        tokenizer.add_special_tokens({'eos_token': new_eos_token})
        logger.warning(f"Added new EOS token: {new_eos_token}")
    logger.info(f"Tokenizer EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")


    # --- Load and Process Data ---
    logger.info(f"Loading data from {DATA_FILE_PATH}...")
    if not DATA_FILE_PATH.exists():
        logger.error(f"Data file not found: {DATA_FILE_PATH}. Please check the path.")
        return
    try:
        raw_dataset_loaded = load_dataset("json", data_files=str(DATA_FILE_PATH), split="train")
    except Exception as e:
        logger.error(f"Failed to load dataset from {DATA_FILE_PATH}: {e}"); return

    logger.info("Transforming base data structure...")
    cols_to_remove_initial = list(raw_dataset_loaded.column_names)
    transformed_dataset = raw_dataset_loaded.map(transform_loaded_data, batched=True, remove_columns=cols_to_remove_initial, num_proc=max(os.cpu_count() // 2, 1) if os.cpu_count() else 1)
    transformed_dataset = transformed_dataset.filter(filter_valid_transformed_data, num_proc=max(os.cpu_count() // 2, 1) if os.cpu_count() else 1)

    logger.info(f"Number of records after initial transform & filter (expecting 5 options): {len(transformed_dataset)}")
    if len(transformed_dataset) == 0: logger.error("No data after initial transform and filtering for 5 options. Check data or filter_valid_transformed_data. Exiting."); return

    logger.info("Preprocessing MCQA dataset for contrastive learning...")
    cols_to_remove_contrastive = list(transformed_dataset.column_names)
    tokenized_dataset = transformed_dataset.map(
        preprocess_mcqa_contrastive,
        batched=True,
        remove_columns=cols_to_remove_contrastive,
        num_proc=max(os.cpu_count() // 2, 1) if os.cpu_count() else 1,
    )

    logger.info(f"Number of records after contrastive preprocessing: {len(tokenized_dataset)}")
    tokenized_dataset = tokenized_dataset.filter(filter_empty_contrastive, num_proc=max(os.cpu_count() // 2, 1) if os.cpu_count() else 1)
    logger.info(f"Tokenized dataset after filtering empty/malformed contrastive examples. Number of examples: {len(tokenized_dataset)}")

    if len(tokenized_dataset) == 0:
        logger.error("No data after contrastive preprocessing and filtering. Check input data and preprocessing functions."); return

    if len(tokenized_dataset) > 20:
        train_test_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
        processed_datasets = DatasetDict({'train': train_test_split['train'], 'eval': train_test_split['test']})
    else: # Handle very small datasets for testing/debugging
        logger.warning("Dataset is very small, using the same data for train and eval.")
        processed_datasets = DatasetDict({'train': tokenized_dataset, 'eval': tokenized_dataset})

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["eval"]

    PROCESSED_DATA_SAVE_PATH = OUTPUT_DIR / 'processed_data' / 'mathqa_contrastive'
    PROCESSED_DATA_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving processed data to disk at {PROCESSED_DATA_SAVE_PATH}...")
    try:
        processed_datasets.save_to_disk(str(PROCESSED_DATA_SAVE_PATH))
        logger.info("Processed data saved.")
    except Exception as e:
        logger.error(f"Could not save processed dataset: {e}")


    # --- Load Model (CausalLM) ---
    logger.info(f"Loading base CausalLM model: {MODEL_ID}")
    quantization_config = QuantoConfig(weights="int4")
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quantization_config)
    except Exception as e:
        logger.error(f"Failed to load model {MODEL_ID}: {e}. "
                     "Ensure MODEL_ID is correct, model is accessible, and you have an internet connection if remote.")
        return

    if len(tokenizer) > model.config.vocab_size:
        logger.info(f"Resizing model token embeddings from {model.config.vocab_size} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
        model.config.vocab_size = len(tokenizer)

    # --- Data Collator ---
    data_collator = DataCollatorForContrastiveMCQA(tokenizer=tokenizer)

    # --- Training Arguments ---
    # Determine available GPUs for logging_steps calculation (will be 1 if CUDA_VISIBLE_DEVICES is set to one GPU)
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    effective_train_batch_size = TRAIN_BATCH_SIZE * num_gpus
    
    # Calculate logging_steps, ensuring it's at least 1 and not excessively frequent for small datasets
    if len(train_dataset) > 0 and effective_train_batch_size > 0:
        steps_per_epoch = len(train_dataset) // effective_train_batch_size
        logging_steps = max(1, steps_per_epoch // 10 if steps_per_epoch > 10 else 1) # Log ~10 times per epoch
    else:
        logging_steps = 10 # Default for very small or empty train_dataset

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR), num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        warmup_ratio=0.1, weight_decay=0.01,
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=logging_steps,
        learning_rate=LEARNING_RATE,
        max_grad_norm=1.0,  # A common starting value; you can tune this
        report_to="none", # "tensorboard" or "wandb"
        remove_unused_columns=False, # Important for custom collator
        gradient_checkpointing=True, # Kept from user's script
        # evaluation_strategy="epoch" if eval_dataset and len(eval_dataset) > 0 else "no", # Example
        # save_strategy="epoch", # Example
        # load_best_model_at_end=True if eval_dataset and len(eval_dataset) > 0 else False, # Example
    )

    # --- Initialize Custom Trainer ---
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Cleared CUDA cache before Trainer initialization.")

    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if eval_dataset and len(eval_dataset) > 0 else None,
        data_collator=data_collator,
        # tokenizer=tokenizer, # Pass tokenizer here if not relying on data_collator.tokenizer
        margin=CONTRASTIVE_MARGIN
    )
    # The Trainer will set self.tokenizer from data_collator if tokenizer is not passed directly

    # --- Start QAT Training ---
    logger.info("Starting Contrastive QAT Training with `quanto`...")
    try:
        trainer.train()
        logger.info("Contrastive QAT training complete.")
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True); return

    logger.info("Freezing `quanto` model...")
    try:
        quanto.freeze(model)
        logger.info("Model frozen successfully.")
    except Exception as e:
        logger.error(f"Error freezing `quanto` model: {e}")

    # --- Save the Quantized Model ---
    final_model_dir = OUTPUT_DIR / "quantized_contrastive_model_final_quanto_4bit"
    logger.info(f"Saving final 4-bit quantized model to {final_model_dir}...")
    try:
        # Ensure model is on CPU before saving if it was on GPU and DataParallel wasn't used
        # (Trainer usually handles moving model to args.device)
        # model.cpu() # Optional: move to CPU before saving if issues occur
        model.save_pretrained(str(final_model_dir))
        tokenizer.save_pretrained(str(final_model_dir))
        logger.info(f"4-bit quantized model saved. Load with `AutoModelForCausalLM.from_pretrained('{final_model_dir}', trust_remote_code=True)`")
    except Exception as e:
        logger.error(f"Error saving final `quanto` model: {e}")

if __name__ == "__main__":
    main()