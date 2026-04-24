# Import necessary libraries
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorWithPadding, BitsAndBytesConfig # Removed DataCollatorForLanguageModeling as we make a custom one
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType # Added for QLoRA
import torch
import torch.nn.functional as F
import logging
import os
from pathlib import Path
import wandb
import random
import re
import ast

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
EXTERNAL_DATA_ROOT = PROJECT_ROOT / "datasets" / "external"

# --- Script Configuration ---
# QLoRA Configuration
USE_QLORA = True  # Set to True to enable QLoRA, False for full fine-tuning
QUANTIZATION_BITS = 4  # 4 or 8. Set to None or other if USE_QLORA is False
NUM_INCORRECT_OPTIONS = 3
LORA_R = 4  # LoRA rank
LORA_ALPHA = 16  # LoRA alpha
LORA_DROPOUT = 0.05 # LoRA dropout
DEBUG = True

# Model and Training Configuration
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
MAX_SEQ_LENGTH = 512
DATASET_NAME_SCIQ = 'allenai/sciq'
DATA_FILE_PATH_MATHQA = str(EXTERNAL_DATA_ROOT / "mathqa" / "train.json")

# Update output path to reflect combined data and QLoRA status
output_suffix = "qlora" if USE_QLORA else "full"
# Changed "contrastive" to "ntp" (Next Token Prediction)
OUTPUT_PATH = str(SCRIPT_DIR / f"qwen_mcqa_ntp_mathqa_sciq_{output_suffix}")


# --- Logging and GPU Setup ---
log_file = os.path.join(OUTPUT_PATH, f'output_ntp_mathqa_sciq_{output_suffix}.log')
os.makedirs(OUTPUT_PATH, exist_ok=True) # Ensure output directory exists

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
transformers.set_seed(SEED)


logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
    logger.info(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Transformers version: {transformers.__version__}")
    # from peft import __version__ as peft_version; logger.info(f"PEFT version: {peft_version}")

# --- 1. Load Tokenizer ---
logger.info(f"Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    logger.info(f"Set tokenizer.pad_token to {tokenizer.eos_token} ({tokenizer.pad_token_id})")
if tokenizer.eos_token is None:
    tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
    logger.info(f"Manually added eos_token: {tokenizer.eos_token} ({tokenizer.eos_token_id})")

# --- 2. Load Model (Potentially with QLoRA) ---
logger.info(f"Loading model {MODEL_NAME}...")

bnb_config = None
if USE_QLORA:
    logger.info(f"Setting up QLoRA with {QUANTIZATION_BITS}-bit quantization...")
    if QUANTIZATION_BITS == 4:
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        logger.info(f"Using 4-bit NF4 quantization with compute_dtype={compute_dtype}.")
    elif QUANTIZATION_BITS == 8:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        logger.info("Using 8-bit quantization.")
    else:
        logger.warning(f"Unsupported QUANTIZATION_BITS: {QUANTIZATION_BITS} for QLoRA. QLoRA will not be applied effectively.")

model_kwargs = {"trust_remote_code": True}
if bnb_config:
    model_kwargs["quantization_config"] = bnb_config
    # model_kwargs["device_map"] = "auto" # Usually handled by Trainer
else:
    model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32

print(MODEL_NAME)
print(model_kwargs)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

# Important: Set pad_token_id in model config for correct behavior with padding
# if model.config.pad_token_id is None: #This should be done after tokenizer.pad_token is set
model.config.pad_token_id = tokenizer.pad_token_id

if len(tokenizer) > model.config.vocab_size:
    logger.info(f"Resizing token embeddings from {model.config.vocab_size} to {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))

if USE_QLORA and bnb_config:
    logger.info("Preparing model for k-bit training and applying LoRA...")
    model = prepare_model_for_kbit_training(model)
    if "Qwen2" in MODEL_NAME or "Qwen3" in MODEL_NAME: # Qwen3 is a guess, Qwen2 is known
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else: # Fallback, verify for other models
        logger.warning(f"Potentially non-optimal LoRA target modules for {MODEL_NAME}. Please verify.")
        # Common modules for GPT-like models
        target_modules = ["c_attn", "c_proj", "w1", "w2"]
        # Or more specific ones if known, e.g. for Llama:
        # target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=target_modules,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    logger.info("LoRA applied. Trainable parameters:")
    model.print_trainable_parameters()
else:
    logger.info("Proceeding with full model fine-tuning (QLoRA not enabled or configured).")

logger.info("Model (potentially with QLoRA) and tokenizer loaded.")


# --- 3. Load and Preprocess MCQA Datasets ---

def transform_loaded_data(examples):
    """
    Transforms data loaded from the MathQA JSON format to the intermediate format.
    """
    new_questions = []
    new_options_lists = []
    new_answer_indices = []
    new_num_options_list = []
    new_ids = []

    num_examples_in_batch = len(examples.get('Problem', [])) # Handle empty batch

    for i in range(num_examples_in_batch):
        problem_text = examples['Problem'][i]
        options_input = examples['options'][i]
        correct_char = examples['correct'][i].lower()
        example_id = examples.get('id', [f"mathqa_gen_id_{i}"]*num_examples_in_batch)[i] # Use existing or generate ID

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
                            elif item_str_cleaned: # If no letter prefix, take the whole item
                                parsed_options.append(item_str_cleaned)
                        is_list_repr_successfully_parsed = True
                except (ValueError, SyntaxError, TypeError):
                    pass # Fall through

            if not is_list_repr_successfully_parsed:
                # Regex for "a) option a b) option b" or "a ) option a b ) option b"
                option_matches = re.findall(
                    r"[a-zA-Z]\s*\)\s*(.*?)(?=\s*[a-zA-Z]\s*\)\s*|$)",
                    options_input, re.IGNORECASE | re.DOTALL
                )
                if option_matches:
                    parsed_options = [match.strip() for match in option_matches]
                elif options_input.strip(): # If regex fails but there's content, treat as single option or warn
                    logger.warning(f"Regex parsing found no options in flat string: '{options_input}' for MathQA id {example_id}. Treating as single option or empty.")
                    parsed_options = [options_input.strip()] if options_input.strip() else []


        elif isinstance(options_input, list): # If options are already a list
            for item_str_or_val in options_input:
                item_str = str(item_str_or_val).strip()
                match = re.match(r"^[a-zA-Z]\s*\)\s*(.*)", item_str, re.DOTALL) # Check for "a) text"
                if match:
                    parsed_options.append(match.group(1).strip())
                else: # Assume item is the option text itself
                    parsed_options.append(item_str)
        else:
            logger.warning(f"Options field is neither a string nor a list for MathQA id {example_id}: type={type(options_input)}. No options parsed.")
            parsed_options = []


        answer_idx = -1
        if correct_char and 'a' <= correct_char <= 'z':
            answer_idx = ord(correct_char) - ord('a')
        else:
            logger.error(f"Invalid correct_char: '{correct_char}' for MathQA problem: '{problem_text[:50]}...'. Skipping.")
            continue

        if not (0 <= answer_idx < len(parsed_options)):
            logger.error(f"Invalid answer_idx {answer_idx} for {len(parsed_options)} parsed options in MathQA problem: '{problem_text[:50]}...'. Correct char: '{correct_char}'. Parsed: {parsed_options}. Skipping.")
            continue

        new_questions.append(problem_text)
        new_options_lists.append(parsed_options)
        new_answer_indices.append(answer_idx)
        new_num_options_list.append(len(parsed_options))
        new_ids.append(str(example_id)) # Ensure ID is a string

    return {
        "id": new_ids,
        "question": new_questions,
        "options": new_options_lists,
        "answer_idx": new_answer_indices,
        "num_options": new_num_options_list,
    }

def transform_sciq_data_to_common_format(examples):
    """Transforms SciQ data to the common intermediate format."""
    new_ids = []
    new_questions = []
    new_options_lists = []
    new_answer_indices = []
    new_num_options_list = []

    num_examples_in_batch = len(examples['question'])

    for i in range(num_examples_in_batch):
        question = examples['question'][i]
        correct_answer = examples['correct_answer'][i]
        distractors = [examples['distractor1'][i], examples['distractor2'][i], examples['distractor3'][i]]
        
        current_options = [correct_answer] + distractors # Correct answer first for now
        answer_idx = 0 # Correct answer is at index 0 before shuffling by standardize_options_count

        new_ids.append(f"sciq_train_{examples.get('id',[i]*num_examples_in_batch)[i]}" if 'id' in examples else f"sciq_genid_{random.randint(0,1000000)}_{i}")
        new_questions.append(question)
        new_options_lists.append(current_options)
        new_answer_indices.append(answer_idx)
        new_num_options_list.append(len(current_options))

    return {
        "id": new_ids,
        "question": new_questions,
        "options": new_options_lists,
        "answer_idx": new_answer_indices,
        "num_options": new_num_options_list,
    }

def standardize_options_count(examples, target_incorrect_options):
    """
    Ensures each example has 1 correct and `target_incorrect_options`.
    Shuffles the final list of options.
    """
    updated_ids = []
    updated_questions = []
    updated_options_lists = []
    updated_answer_indices = []
    updated_num_options_list = []

    num_examples_in_batch = len(examples['question'])

    for i in range(num_examples_in_batch):
        original_id = examples['id'][i]
        question = examples['question'][i]
        current_options = examples['options'][i]
        current_answer_idx = examples['answer_idx'][i]

        if not (isinstance(current_options, list) and 0 <= current_answer_idx < len(current_options)):
            logger.warning(f"Skipping example ID {original_id} due to invalid options/answer_idx ({current_answer_idx}, {len(current_options)} opts) before standardization.")
            continue
        
        correct_option = current_options[current_answer_idx]
        incorrect_options = [opt for idx, opt in enumerate(current_options) if idx != current_answer_idx]

        if len(incorrect_options) > target_incorrect_options:
            random.shuffle(incorrect_options) 
            selected_incorrect = incorrect_options[:target_incorrect_options]
        elif len(incorrect_options) < target_incorrect_options:
            # logger.warning(f"Skipping example ID {original_id} due to insufficient incorrect options ({len(incorrect_options)} vs {target_incorrect_options} required).")
            # Instead of skipping, let's try to use it if there's at least one incorrect. Evaluation needs to handle variable number of options.
            # For simplicity with NUM_INCORRECT_OPTIONS, we'll skip for now if it doesn't meet the exact count.
            # This ensures evaluation logic is simpler as it expects a fixed number of incorrect options.
            if len(incorrect_options) == 0 and target_incorrect_options > 0: # Cannot select if no incorrect options exist but are needed
                 logger.warning(f"Skipping example ID {original_id}: No incorrect options available, but {target_incorrect_options} required.")
                 continue
            # If we allow fewer, the eval logic must adapt. For now, enforce target_incorrect_options.
            logger.warning(f"Skipping example ID {original_id} due to insufficient incorrect options ({len(incorrect_options)} vs {target_incorrect_options} required).")
            continue

        else: # Exactly target_incorrect_options
            selected_incorrect = incorrect_options

        final_options_for_sample = [correct_option] + selected_incorrect
        random.shuffle(final_options_for_sample) 
        final_answer_idx = final_options_for_sample.index(correct_option)

        updated_ids.append(original_id)
        updated_questions.append(question)
        updated_options_lists.append(final_options_for_sample)
        updated_answer_indices.append(final_answer_idx)
        # The num_options should now be 1 (correct) + target_incorrect_options
        updated_num_options_list.append(1 + target_incorrect_options)


    return {
        "id": updated_ids,
        "question": updated_questions,
        "options": updated_options_lists,
        "answer_idx": updated_answer_indices,
        "num_options": updated_num_options_list,
    }

def filter_valid_intermediate_data(example):
    is_valid = True
    if not example.get("options") or not isinstance(example["options"], list) or example.get("num_options", 0) == 0:
        is_valid = False
    elif not (0 <= example.get("answer_idx", -1) < example.get("num_options", 0)):
        is_valid = False
    # After standardize_options_count, num_options should be 1 + NUM_INCORRECT_OPTIONS
    # This filter is applied before standardize_options_count, so it checks general validity.
    # The standardize_options_count itself filters more strictly.
    return is_valid

# --- Load and Process MathQA ---
logger.info(f"Loading MathQA data from {DATA_FILE_PATH_MATHQA}...")
try:
    raw_mathqa_loaded = load_dataset("json", data_files=DATA_FILE_PATH_MATHQA, split="train")
    logger.info(f"Successfully loaded {len(raw_mathqa_loaded)} records from MathQA JSON.")
except Exception as e:
    logger.error(f"Failed to load MathQA dataset from {DATA_FILE_PATH_MATHQA}: {e}")
    exit()

logger.info("Transforming MathQA data...")
mathqa_cols_to_remove = raw_mathqa_loaded.column_names
processed_mathqa_interim = raw_mathqa_loaded.map(
    transform_loaded_data,
    batched=True,
    remove_columns=mathqa_cols_to_remove
)
original_mathqa_count = len(processed_mathqa_interim)
processed_mathqa_interim = processed_mathqa_interim.filter(filter_valid_intermediate_data)
logger.info(f"MathQA: Filtered {original_mathqa_count - len(processed_mathqa_interim)} invalid examples after initial transformation. Remaining: {len(processed_mathqa_interim)}")


# --- Load and Process SciQ ---
logger.info(f"Loading SciQ data from {DATASET_NAME_SCIQ}...")
try:
    raw_sciq_loaded = load_dataset(DATASET_NAME_SCIQ, split="train")
    logger.info(f"Successfully loaded {len(raw_sciq_loaded)} records from SciQ dataset.")
except Exception as e:
    logger.error(f"Failed to load SciQ dataset: {e}")
    exit()

logger.info("Transforming SciQ data...")
sciq_cols_to_remove = raw_sciq_loaded.column_names
processed_sciq_interim = raw_sciq_loaded.map(
    transform_sciq_data_to_common_format,
    batched=True,
    remove_columns=sciq_cols_to_remove
)
original_sciq_count = len(processed_sciq_interim)
processed_sciq_interim = processed_sciq_interim.filter(filter_valid_intermediate_data)
logger.info(f"SciQ: Filtered {original_sciq_count - len(processed_sciq_interim)} invalid examples after initial transformation. Remaining: {len(processed_sciq_interim)}")

# --- Combine and Standardize Datasets ---
logger.info("Combining MathQA and SciQ datasets...")
if len(processed_mathqa_interim) == 0 and len(processed_sciq_interim) == 0:
    logger.error("Both datasets are empty after initial processing. Exiting.")
    exit()
elif len(processed_mathqa_interim) == 0:
    logger.warning("MathQA dataset is empty. Proceeding with SciQ only.")
    combined_dataset_interim = processed_sciq_interim
elif len(processed_sciq_interim) == 0:
    logger.warning("SciQ dataset is empty. Proceeding with MathQA only.")
    combined_dataset_interim = processed_mathqa_interim
else:
    combined_dataset_interim = concatenate_datasets([processed_mathqa_interim, processed_sciq_interim])
logger.info(f"Combined dataset size before standardization: {len(combined_dataset_interim)}")


logger.info(f"Standardizing options count to 1 correct + {NUM_INCORRECT_OPTIONS} incorrect options...")
# This dataset will be used for custom evaluation (before tokenization for NTP training)
mcqa_formatted_dataset = combined_dataset_interim.map(
    lambda x: standardize_options_count(x, target_incorrect_options=NUM_INCORRECT_OPTIONS),
    batched=True
)
# standardize_options_count now filters internally by skipping, so length might change
logger.info(f"Filtered {len(combined_dataset_interim) - len(mcqa_formatted_dataset)} examples during option count standardization.")
logger.info(f"Final dataset size after standardization (for tokenization/evaluation): {len(mcqa_formatted_dataset)}")


if len(mcqa_formatted_dataset) == 0:
    logger.error("No data remaining after transformation, combination, and standardization. Please check data and logic.")
    exit()


def preprocess_mcqa_for_ntp(examples):
    """Prepares data for Next Token Prediction loss on the correct option only."""
    processed_batch = {
        "input_ids": [], "attention_mask": [], "prompt_len": [], "id": []
    }

    for i in range(len(examples['question'])):
        question = examples['question'][i]
        options = examples['options'][i] # This is the list of all options
        answer_idx = examples['answer_idx'][i] # Index of the correct option in the 'options' list
        example_id = examples['id'][i]

        # Ensure data integrity after standardization
        expected_num_options = 1 + NUM_INCORRECT_OPTIONS
        if not (isinstance(options, list) and len(options) == expected_num_options and 0 <= answer_idx < expected_num_options):
            logger.warning(f"Skipping malformed MCQA example ID {example_id} in preprocess_mcqa_for_ntp. Options: {len(options)}, answer_idx: {answer_idx}")
            continue
            
        correct_option_text = options[answer_idx]
        
        # Construct prompt and full text
        # The prompt should not include the start of the answer.
        prompt_text = f"Question: {question}\nAnswer: " # Or "Option: " as before
        full_text = prompt_text + correct_option_text + tokenizer.eos_token

        # Tokenize prompt to find its length (for masking in loss)
        # Tokenize without special tokens to get the pure length of the prompt part
        tokenized_prompt = tokenizer(prompt_text, add_special_tokens=False, truncation=False) # Don't truncate prompt itself
        prompt_length_in_tokens = len(tokenized_prompt['input_ids'])

        # Tokenize the full text (prompt + correct_option + eos)
        tokenized_full = tokenizer(
            full_text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            # padding="max_length" # Padding will be handled by the collator
        )
        
        # Ensure prompt is not longer than max_seq_length
        if prompt_length_in_tokens >= MAX_SEQ_LENGTH -1: # -1 for at least one token of the answer
            logger.warning(f"Prompt for example ID {example_id} is too long ({prompt_length_in_tokens} tokens) for MAX_SEQ_LENGTH {MAX_SEQ_LENGTH}. Skipping.")
            continue


        processed_batch["input_ids"].append(tokenized_full['input_ids'])
        processed_batch["attention_mask"].append(tokenized_full['attention_mask'])
        processed_batch["prompt_len"].append(prompt_length_in_tokens)
        processed_batch["id"].append(example_id)
    
    if not processed_batch["id"]: # If all examples in a batch were skipped
        return {key: [] for key in processed_batch.keys()}
        
    return processed_batch

logger.info("Preprocessing dataset for Next Token Prediction training...")
# `mcqa_formatted_dataset` has 'id', 'question', 'options', 'answer_idx', 'num_options'
# `preprocess_mcqa_for_ntp` uses 'question', 'options', 'answer_idx', 'id'.
# It outputs 'input_ids', 'attention_mask', 'prompt_len', 'id'.
# We should remove the original columns that are not needed by the collator or trainer.
# The standard Trainer will only use 'input_ids', 'attention_mask', 'labels'. 'id' is fine. 'prompt_len' is for collator.
# Columns to remove would be 'question', 'options', 'answer_idx', 'num_options'.
tokenized_dataset_cols_to_remove = ['question', 'options', 'answer_idx', 'num_options']



tokenized_dataset_for_training = mcqa_formatted_dataset.map(
    preprocess_mcqa_for_ntp,
    batched=True,
    num_proc=max(os.cpu_count() // 4, 1), 
    remove_columns=tokenized_dataset_cols_to_remove 
)


def filter_empty_after_tokenization(example):
    input_ids = example.get('input_ids')
    prompt_len = example.get('prompt_len')
    example_id = example.get('id', 'N/A') # For logging

    if not input_ids or len(input_ids) == 0:
        # logger.info(f"Filtering out example ID {example_id} due to empty/missing input_ids.") # Use info for visibility if needed
        return False
    
    if prompt_len is None:
        logger.warning(f"Filtering out example ID {example_id} because prompt_len is None. Input_ids length: {len(input_ids)}")
        return False
        
    if not (prompt_len < len(input_ids)):
        # Ensure there's at least one token for the answer part after the prompt
        # logger.info(f"Filtering out example ID {example_id} because prompt_len ({prompt_len}) is not less than input_ids length ({len(input_ids)}).")
        return False
        
    return True

original_tokenized_count = len(tokenized_dataset_for_training)
tokenized_dataset_for_training = tokenized_dataset_for_training.filter(filter_empty_after_tokenization)
logger.info(f"Filtered {original_tokenized_count - len(tokenized_dataset_for_training)} invalid examples after NTP tokenization/filtering.")
logger.info(f"Tokenized dataset for training. Number of examples: {len(tokenized_dataset_for_training)}")



if len(tokenized_dataset_for_training) == 0:
    logger.error("No data after preprocessing for NTP training. Check input data and preprocessing functions.")
    exit()

# Splitting dataset for training
# The `mcqa_formatted_dataset` is split first to get a raw eval set.
# Then `tokenized_dataset_for_training` (which is derived from the whole `mcqa_formatted_dataset`) is split for trainer's train/eval.

if len(tokenized_dataset_for_training) > 20: # Ensure enough for a split of the raw formatted data
    # This split is for the *raw formatted data*. The 'test' part is for custom evaluation.
    # The 'train' part is what will be tokenized for training.
    raw_train_test_split = tokenized_dataset_for_training.train_test_split(test_size=0.1, seed=SEED)
    train_data_for_tokenization = raw_train_test_split['train']
    eval_data_for_custom_eval = raw_train_test_split['test'] # This is passed to custom eval
else:
    logger.warning("Dataset too small for raw train/test split. Using all data for training and custom eval source.")
    train_data_for_tokenization = tokenized_dataset_for_training
    eval_data_for_custom_eval = tokenized_dataset_for_training # Use all for custom eval as well


logger.info(f"Final tokenized train dataset size: {len(train_data_for_tokenization)}")
logger.info(f"Final tokenized eval dataset size (for Trainer): {len(eval_data_for_custom_eval)}")


processed_datasets_for_trainer = DatasetDict({
    'train': train_data_for_tokenization,
    'eval': eval_data_for_custom_eval
})


# --- Save the data on disk ---
# Note: eval_data_for_custom_eval is not saved here, but it's derived from mcqa_formatted_dataset
# which could be saved if needed. The processed_datasets_for_trainer contains tokenized data.
processed_data_save_path = os.path.join(OUTPUT_PATH, "processed_data_for_trainer")
logging.info(f"Saving processed (tokenized) datasets for Trainer to disk at {processed_data_save_path}...")
os.makedirs(processed_data_save_path, exist_ok=True)
processed_datasets_for_trainer.save_to_disk(processed_data_save_path)


# --- 4. Custom Data Collator ---
class DataCollatorForNTPMCQA:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        initial_feature_count = len(features)
        # A feature is valid if it has non-empty input_ids and prompt_len is an actual number.
        valid_features = [
            f for f in features 
            if f.get("input_ids") and len(f.get("input_ids")) > 0 and f.get("prompt_len") is not None
        ]
        
        if not valid_features:
            # This is the problematic case leading to the error
            logger.warning(
                f"Data collator received a batch of {initial_feature_count} features, "
                f"but NONE were valid (e.g. input_ids missing/empty or prompt_len is None). "
                f"Returning empty tensor structure. This will likely cause a runtime error in model forward pass."
            )
            # Log details of the first few problematic features if possible (be careful with large outputs)
            if initial_feature_count > 0:
                for i, f_bad in enumerate(features[:min(3, initial_feature_count)]): # Log first 3 problematic
                    logger.warning(f"Problematic feature {i} ID {f_bad.get('id', 'N/A')}: input_ids len: {len(f_bad.get('input_ids', [])) if f_bad.get('input_ids') else 'None'}, prompt_len: {f_bad.get('prompt_len')}")

            return {
                "input_ids": torch.empty(0, dtype=torch.long),
                "attention_mask": torch.empty(0, dtype=torch.long),
                "labels": torch.empty(0, dtype=torch.long),
            }
        
        # If we reached here, valid_features is not empty. Proceed with these.
        features = valid_features

        batch_input_ids = [f["input_ids"] for f in features]
        batch_attention_mask = [f["attention_mask"] for f in features]
        batch_prompt_len = [f["prompt_len"] for f in features]

        max_len = max(len(ids) for ids in batch_input_ids)
        
        padded_input_ids = []
        padded_attention_mask = []

        for i in range(len(batch_input_ids)):
            ids = batch_input_ids[i]
            mask = batch_attention_mask[i]
            padding_length = max_len - len(ids)
            
            padded_input_ids.append(ids + [self.tokenizer.pad_token_id] * padding_length)
            padded_attention_mask.append(mask + [0] * padding_length)

        input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long)
        attention_mask_tensor = torch.tensor(padded_attention_mask, dtype=torch.long)

        labels_tensor = input_ids_tensor.clone()

        for i in range(len(features)):
            prompt_len = batch_prompt_len[i]
            labels_tensor[i, :prompt_len] = -100
        
        labels_tensor[labels_tensor == self.tokenizer.pad_token_id] = -100
        
        batch = {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "labels": labels_tensor,
        }
        return batch

data_collator = DataCollatorForNTPMCQA(tokenizer=tokenizer)


# --- 5. Helper for Evaluation NLL Calculation ---
# This function is similar to the original _get_option_nll,
# but it's now primarily for evaluation.
def _get_nll_for_sequence_option(model, input_ids, attention_mask, labels_for_loss, prompt_lengths_list):
    """
    Calculates the sum of Negative Log-Likelihood for the option part of sequences.
    Assumes input_ids, attention_mask, labels_for_loss are already on the correct device.
    labels_for_loss: These are the target tokens (e.g., a clone of input_ids).
                     The function will handle shifting and masking based on prompt_lengths_list.
    prompt_lengths_list: A list of prompt lengths, one for each sequence in the batch.
    """
    if input_ids.nelement() == 0:
        return torch.empty(0, device=model.device)

    # Get logits from the model
    # The model's forward pass will handle shifting if 'labels' are passed for loss calculation.
    # Here, we want to calculate a custom NLL sum, so we work with raw logits.
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits # Shape: (batch_size, seq_len, vocab_size)

    batch_option_nll_sum = []
    
    # Shift logits to align with labels for next token prediction
    # logit for token i predicts token i+1
    shifted_logits = logits[..., :-1, :].contiguous()
    # labels for token i+1 is the target for logit for token i
    shifted_labels = labels_for_loss[..., 1:].contiguous()

    loss_fct_sum = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=-100)
    loss_fct_none = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)


    for i in range(input_ids.size(0)): # Iterate over items in the batch
        prompt_len_scalar = prompt_lengths_list[i]
        
        current_shifted_labels = shifted_labels[i] # Shape: (seq_len - 1)
        current_shifted_logits = shifted_logits[i] # Shape: (seq_len - 1, vocab_size)

        # We want to calculate loss only for the tokens *after* the prompt.
        # The prompt_len_scalar is for the original, unshifted sequence.
        # So, the first token of the answer corresponds to index `prompt_len_scalar` in the original sequence.
        # In the shifted sequences, this is index `prompt_len_scalar - 1` (if prompt_len > 0).
        # Example: Prompt P1 P2, Answer A1 A2. prompt_len = 2.
        # original_labels: P1 P2 A1 A2
        # shifted_labels:  P2 A1 A2 (targets for P1, P2, A1 respectively)
        # We want loss for A1, A2. These are at index 1, 2 in shifted_labels.
        # This corresponds to original prompt_len-1 onwards.
        
        option_token_indices_start_in_shifted = max(0, prompt_len_scalar -1) # if prompt_len is 0, start from 0.
                                                                            # if prompt_len is 1 (e.g. BOS), start from 0.
                                                                            # if prompt_len is 2 (e.g. BOS Q), start from 1.
        
        # Create a mask for the option tokens in the shifted labels
        # We are interested in tokens from `option_token_indices_start_in_shifted` onwards
        # AND where the label is not -100 (already padding or intentionally masked)
        valid_option_mask = torch.zeros_like(current_shifted_labels, dtype=torch.bool)
        if option_token_indices_start_in_shifted < current_shifted_labels.size(0):
             valid_option_mask[option_token_indices_start_in_shifted:] = True
        valid_option_mask = valid_option_mask & (current_shifted_labels != -100)


        if valid_option_mask.sum() == 0:
            # logger.debug(f"No valid option tokens found for NLL calculation in eval sample {i}.")
            batch_option_nll_sum.append(torch.tensor(float('inf'), device=model.device))
            continue

        # Select only the option part for loss calculation
        option_labels = current_shifted_labels[valid_option_mask]
        option_logits = current_shifted_logits[valid_option_mask]

        if option_labels.nelement() == 0: # Should be caught by valid_option_mask.sum() == 0
             batch_option_nll_sum.append(torch.tensor(float('inf'), device=model.device))
             continue

        # Calculate sum of NLL for these option tokens
        # loss_fct_sum expects (N, C) and (N)
        nll_sum_for_option = loss_fct_sum(option_logits.view(-1, option_logits.size(-1)), option_labels.view(-1))
        batch_option_nll_sum.append(nll_sum_for_option)

    if not batch_option_nll_sum:
        return torch.empty(0, device=model.device)
    return torch.stack(batch_option_nll_sum)


# --- 6. Set up Training Arguments ---
logger.info("Setting up training arguments...")
training_args_dict = {
    "output_dir": OUTPUT_PATH,
    "overwrite_output_dir": True,
    "num_train_epochs": 1, 
    "per_device_train_batch_size": 16, 
    "per_device_eval_batch_size": 16,  
    "gradient_accumulation_steps": 8,
    "logging_strategy": "steps",
    "logging_steps": 10,
    "save_strategy": "epoch",
    "save_total_limit": 2,
    "learning_rate": 5e-5 if not USE_QLORA else 2e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "lr_scheduler_type": "cosine",
    "report_to": "wandb" if os.environ.get("WANDB_PROJECT") else "none",
    "remove_unused_columns": False, # Important: set to True if collator doesn't pass all original cols
                                  # Our collator only outputs input_ids, attention_mask, labels
    "logging_dir": f'{OUTPUT_PATH}/logs',
    "bf16": False, # Explicitly false, QLoRA handles its compute_dtype
    "fp16": False, # Explicitly false
    "gradient_checkpointing": True if USE_QLORA else False,
}

if USE_QLORA:
    # For QLoRA, the compute_dtype is handled by bnb_config.
    # bf16/fp16 in TrainingArguments can be for optimizer state and gradients if not using deepspeed.
    # It's generally recommended to align this with the QLoRA compute_dtype if possible.
    if QUANTIZATION_BITS == 4:
        if bnb_config and bnb_config.bnb_4bit_compute_dtype == torch.bfloat16:
            training_args_dict["bf16"] = True # If compute is bfloat16, use bf16 for optimizer
            logger.info("QLoRA with bf16 compute dtype. Setting TrainingArguments.bf16 = True")
        elif bnb_config and bnb_config.bnb_4bit_compute_dtype == torch.float16:
            training_args_dict["fp16"] = True # If compute is float16, use fp16 for optimizer
            logger.info("QLoRA with fp16 compute dtype. Setting TrainingArguments.fp16 = True")
    elif QUANTIZATION_BITS == 8: # For 8-bit, fp16 is often used for optimizer
        training_args_dict["fp16"] = True
        logger.info("QLoRA 8-bit. Setting TrainingArguments.fp16 = True")


training_args = TrainingArguments(**training_args_dict)


# --- 7. Initialize Trainer ---
logger.info("Initializing standard Trainer...")
trainer = Trainer( # Using standard Trainer
    model=model,
    args=training_args,
    train_dataset=processed_datasets_for_trainer["train"],
    eval_dataset=processed_datasets_for_trainer["eval"] if processed_datasets_for_trainer["eval"] and len(processed_datasets_for_trainer["eval"]) > 0 else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    # compute_metrics=None, # Can add a compute_metrics function if needed for Trainer's eval
)


if DEBUG:
    for sample in processed_datasets_for_trainer["train"]:
        if len(sample['input_ids']) == 0:
            print(sample)

# --- 8. Start Training ---
if processed_datasets_for_trainer["train"] and len(processed_datasets_for_trainer["train"]) > 0 :
    logger.info("\n--- Starting training ---")
    try:
        train_result = trainer.train()
        logger.info("--- Training finished ---")

        logger.info("Saving final model/adapters and tokenizer...")
        trainer.save_model(OUTPUT_PATH) 
        tokenizer.save_pretrained(OUTPUT_PATH)
        logger.info(f"Model/adapters and tokenizer saved to {OUTPUT_PATH}")

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info(f"Training metrics: {metrics}")

    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
else:
    logger.warning("No training data available after all processing. Skipping training.")

# --- 9. Custom Evaluation ---
# def evaluate_mcqa_likelihood_ntp(eval_trainer_instance, dataset_for_custom_eval, local_tokenizer, local_max_seq_length):
#     """
#     dataset_for_custom_eval: This is the raw formatted dataset (e.g., eval_data_for_custom_eval)
#                              It contains 'question', 'options' (list of strings), 'answer_idx'.
#     """
#     logger.info(f"Starting custom NTP-based MCQA evaluation on {len(dataset_for_custom_eval)} examples.")
#     model_to_eval = eval_trainer_instance.model
#     device = model_to_eval.device
#     model_to_eval.eval()

#     correct_predictions = 0
#     total_predictions = 0
    
#     # Store NLLs for analysis
#     all_option_nlls_per_question = [] # list of lists of nlls

#     for example_idx, example in enumerate(dataset_for_custom_eval):
#         if example_idx % 100 == 0 and example_idx > 0:
#             logger.info(f"Custom eval processing example {example_idx}/{len(dataset_for_custom_eval)}")

#         question_text = example['question']
#         options_texts = example['options'] # List of option strings
#         correct_answer_idx = example['answer_idx']
#         example_id = example.get('id', f"eval_ex_{example_idx}")

#         prompt_base = f"Question: {question_text}\nAnswer: " # Same prompt as used in training
        
#         # Tokenize the base prompt once to get its length
#         tokenized_prompt_base = local_tokenizer(prompt_base, add_special_tokens=False, truncation=False)
#         prompt_base_len_tokens = len(tokenized_prompt_base['input_ids'])

#         if prompt_base_len_tokens >= local_max_seq_length -1 : # Ensure prompt itself isn't too long
#             logger.warning(f"Base prompt for eval example ID {example_id} is too long ({prompt_base_len_tokens}). Skipping.")
#             continue

#         nlls_for_current_options = []
#         with torch.no_grad():
#             for opt_idx, option_text in enumerate(options_texts):
#                 full_text_eval = prompt_base + option_text + local_tokenizer.eos_token
                
#                 tokenized_eval_instance = local_tokenizer(
#                     full_text_eval,
#                     truncation=True,
#                     max_length=local_max_seq_length,
#                     return_tensors="pt" 
#                 )
                
#                 input_ids_eval = tokenized_eval_instance.input_ids.to(device)
#                 attention_mask_eval = tokenized_eval_instance.attention_mask.to(device)
                
#                 # Create labels for _get_nll_for_sequence_option (just a clone of input_ids)
#                 labels_for_nll_calc = input_ids_eval.clone() 

#                 # The prompt_base_len_tokens is the length of the prompt part *within* full_text_eval's tokenization.
#                 # It should be consistent if tokenization of prompt_base is stable.
                
#                 # Check if the option part got truncated away
#                 if prompt_base_len_tokens >= input_ids_eval.size(1) -1 : # -1 for EOS
#                      logger.warning(f"Option {opt_idx} for eval example ID {example_id} got truncated. Assigning inf NLL. Prompt len: {prompt_base_len_tokens}, total len: {input_ids_eval.size(1)}")
#                      nlls_for_current_options.append(float('inf'))
#                      continue

#                 nll_tensor = _get_nll_for_sequence_option(
#                     model_to_eval,
#                     input_ids_eval,
#                     attention_mask_eval,
#                     labels_for_nll_calc, # Pass the clone of input_ids
#                     [prompt_base_len_tokens]  # Pass scalar prompt length as a list for the batch of 1
#                 )

#                 if nll_tensor.nelement() == 0 or torch.isinf(nll_tensor[0]):
#                     # logger.warning(f"Inf or empty NLL for option {opt_idx} in eval (ID: {example_id}).")
#                     nlls_for_current_options.append(float('inf'))
#                 else:
#                     nlls_for_current_options.append(nll_tensor[0].item())
        
#         all_option_nlls_per_question.append(nlls_for_current_options)

#         if not nlls_for_current_options or all(nll == float('inf') for nll in nlls_for_current_options):
#             logger.warning(f"All options for example ID {example_id} resulted in Inf NLL. Skipping prediction.")
#             # total_predictions +=1 # Count as an attempt, but it will be wrong. Or skip total_predictions increment.
#                                 # Let's count it as a prediction made (incorrectly, as we can't choose).
#             continue # Or assign a random prediction / default to first option. For strictness, let's see.


#         # Handle cases where all NLLs are inf (e.g. if all options too long / bad data)
#         valid_nlls = [n for n in nlls_for_current_options if n != float('inf')]
#         if not valid_nlls:
#             predicted_best_idx = 0 # Default or skip, for now default to 0 if all inf
#             logger.warning(f"All options for example ID {example_id} had Inf NLL. Defaulting prediction or consider this an error.")
#         else:
#             # Choose option with the minimum NLL
#             min_nll = min(valid_nlls)
#             # Find first index of this min_nll in original list to handle multiple infs correctly
#             predicted_best_idx = nlls_for_current_options.index(min_nll)


#         if predicted_best_idx == correct_answer_idx:
#             correct_predictions += 1
#         total_predictions += 1

#     accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
#     avg_nll_correct_option_overall = []
#     avg_nll_incorrect_options_overall = []

#     for i, nlls_options in enumerate(all_option_nlls_per_question):
#         # Need to know which one was correct for this analysis from `eval_data_for_custom_eval`
#         # This part requires access to `correct_answer_idx` for each item in `all_option_nlls_per_question`
#         # Let's assume `dataset_for_custom_eval` is iterated in the same order.
#         original_example_correct_idx = dataset_for_custom_eval[i]['answer_idx']
#         if original_example_correct_idx < len(nlls_options) and nlls_options[original_example_correct_idx] != float('inf'):
#             avg_nll_correct_option_overall.append(nlls_options[original_example_correct_idx])
        
#         incorrect_nlls_temp = []
#         for opt_j, nll_val in enumerate(nlls_options):
#             if opt_j != original_example_correct_idx and nll_val != float('inf'):
#                 incorrect_nlls_temp.append(nll_val)
#         if incorrect_nlls_temp:
#             avg_nll_incorrect_options_overall.append(sum(incorrect_nlls_temp) / len(incorrect_nlls_temp))


#     final_avg_nll_correct = sum(avg_nll_correct_option_overall) / len(avg_nll_correct_option_overall) if avg_nll_correct_option_overall else float('nan')
#     final_avg_nll_incorrect = sum(avg_nll_incorrect_options_overall) / len(avg_nll_incorrect_options_overall) if avg_nll_incorrect_options_overall else float('nan')


#     eval_metrics = {
#         "eval_mcqa_accuracy_ntp": accuracy,
#         "eval_avg_nll_correct_option_ntp": final_avg_nll_correct,
#         "eval_avg_nll_incorrect_options_ntp": final_avg_nll_incorrect,
#         "eval_total_predictions_custom_ntp": total_predictions,
#     }
#     logger.info(f"Custom MCQA NTP Likelihood Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
#     logger.info(f"Avg NLL Correct (overall): {final_avg_nll_correct:.4f}, Avg NLL Incorrect (avg per question, then overall avg): {final_avg_nll_incorrect:.4f}")
#     return eval_metrics


# if eval_data_for_custom_eval and len(eval_data_for_custom_eval) > 0:
#     logger.info("\n--- Starting custom evaluation (NTP-based MCQA likelihood) ---")
#     try:
#         # Ensure the model used for eval is the trained one, and it's on the correct device.
#         # Trainer.model should be the up-to-date model.
#         eval_results_ntp = evaluate_mcqa_likelihood_ntp(
#             trainer, # Pass the trainer instance
#             eval_data_for_custom_eval, # This is the raw formatted data for eval
#             tokenizer, # Pass tokenizer
#             MAX_SEQ_LENGTH # Pass max_seq_length
#         ) 
#         logger.info(f"Custom NTP Evaluation Results: {eval_results_ntp}")
        
#         if os.environ.get("WANDB_PROJECT") and wandb.run:
#             wandb.log(eval_results_ntp)
        
#         # Save custom metrics
#         final_metrics_to_save = {}
#         for k, v in eval_results_ntp.items():
#             final_metrics_to_save[k] = v.item() if isinstance(v, torch.Tensor) else v
#         trainer.save_metrics("custom_eval_ntp", final_metrics_to_save)

#     except Exception as e:
#         logger.error(f"An error occurred during custom NTP evaluation: {e}", exc_info=True)
# else:
#     logger.warning("No data available for custom NTP evaluation. Skipping.")

if os.environ.get("WANDB_PROJECT") and wandb.run:
    wandb.finish()

logger.info("\nScript completed.")
