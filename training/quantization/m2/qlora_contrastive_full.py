# Import necessary libraries
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict, load_from_disk
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorWithPadding, BitsAndBytesConfig, DataCollatorForLanguageModeling
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
LORA_R = 4  # LoRA rank
LORA_ALPHA = 16  # LoRA alpha
LORA_DROPOUT = 0.05 # LoRA dropout

# Model and Training Configuration
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
MAX_SEQ_LENGTH = 512
CONTRASTIVE_MARGIN = 1.0
NUM_INCORRECT_OPTIONS = 3 # Standardize to 3 incorrect options
DATASET_NAME_SCIQ = 'allenai/sciq'
DATA_FILE_PATH_MATHQA = str(EXTERNAL_DATA_ROOT / "mathqa" / "train.json")
DATA_CODING_MCQA = str(EXTERNAL_DATA_ROOT / "tuandunghcmut_coding-mcq-reasoning")
DATA_GPQA = str(EXTERNAL_DATA_ROOT / "gpqa")


# Update output path to reflect combined data and QLoRA status
output_suffix = "qlora" if USE_QLORA else "full"
OUTPUT_PATH = str(SCRIPT_DIR / f"qwen_mcqa_contrastive_mathqa_sciq_{output_suffix}")


# --- Logging and GPU Setup ---
log_file = os.path.join(OUTPUT_PATH, f'output_contrastive_mathqa_sciq_{output_suffix}.log')
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
    # model_kwargs["device_map"] = "auto" # Usually handled by Trainer, but can be set for bitsandbytes
else:
    model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
model.config.pad_token_id = tokenizer.pad_token_id
if len(tokenizer) > model.config.vocab_size:
    logger.info(f"Resizing token embeddings from {model.config.vocab_size} to {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))

if USE_QLORA and bnb_config:
    logger.info("Preparing model for k-bit training and applying LoRA...")
    model = prepare_model_for_kbit_training(model)
    if "Qwen2" in MODEL_NAME or "Qwen3" in MODEL_NAME:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        logger.warning(f"Potentially non-optimal LoRA target modules for {MODEL_NAME}. Please verify.")
        target_modules = ["c_attn", "c_proj", "w1", "w2", "q_proj", "k_proj", "v_proj", "o_proj"]

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
                            elif item_str_cleaned:
                                parsed_options.append(item_str_cleaned)
                        is_list_repr_successfully_parsed = True
                except (ValueError, SyntaxError, TypeError):
                    pass # Fall through

            if not is_list_repr_successfully_parsed:
                option_matches = re.findall(
                    r"[a-zA-Z]\s*\)\s*(.*?)(?=\s*[a-zA-Z]\s*\)\s*|$)",
                    options_input, re.IGNORECASE | re.DOTALL
                )
                if option_matches:
                    parsed_options = [match.strip() for match in option_matches]
                elif options_input.strip():
                    logger.warning(f"Regex parsing found no options in flat string: '{options_input}' for MathQA id {example_id}. Treating as single option or empty.")
                    parsed_options = [options_input.strip()] if options_input.strip() else []

        elif isinstance(options_input, list):
            for item_str_or_val in options_input:
                item_str = str(item_str_or_val).strip()
                match = re.match(r"^[a-zA-Z]\s*\)\s*(.*)", item_str, re.DOTALL)
                if match:
                    parsed_options.append(match.group(1).strip())
                else:
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
            logger.error(f"Invalid answer_idx {answer_idx} for {len(parsed_options)} parsed options in MathQA problem: '{problem_text[:50]}...'. Correct char: '{correct_char}'. Skipping.")
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

def transform_distractor_data_to_common_format(examples):
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
        # SciQ provides distractors directly
        distractors = [examples['distractor1'][i], examples['distractor2'][i], examples['distractor3'][i]]
        
        current_options = [correct_answer] + distractors # Correct answer first, then distractors
        
        # Shuffle options to avoid positional bias if model sees unshuffled data before contrastive processing
        # However, `standardize_options_count` will shuffle again. For here, let's keep correct at 0 for now.
        # Or shuffle here and `standardize_options_count` takes them as is if count matches.
        # Let `standardize_options_count` handle the final shuffle.
        # So, fixed order for now: Correct, D1, D2, D3
        
        answer_idx = 0 # Correct answer is at index 0 in `current_options` before shuffling

        new_ids.append(f"sciq_train_{examples.get('id',[i]*num_examples_in_batch)[i]}" if 'id' in examples else f"sciq_genid_{random.randint(0,1000000)}_{i}") # Create unique ID
        new_questions.append(question)
        new_options_lists.append(current_options) # List of 4 options
        new_answer_indices.append(answer_idx)
        new_num_options_list.append(len(current_options)) # Should be 4

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

        if not (0 <= current_answer_idx < len(current_options)):
            logger.warning(f"Skipping example ID {original_id} due to invalid answer_idx ({current_answer_idx}) for {len(current_options)} options before standardization.")
            continue
        
        correct_option = current_options[current_answer_idx]
        incorrect_options = [opt for idx, opt in enumerate(current_options) if idx != current_answer_idx]

        if len(incorrect_options) > target_incorrect_options:
            random.shuffle(incorrect_options) # Shuffle to pick random ones
            selected_incorrect = incorrect_options[:target_incorrect_options]
        elif len(incorrect_options) < target_incorrect_options:
            logger.warning(f"Skipping example ID {original_id} due to insufficient incorrect options ({len(incorrect_options)} vs {target_incorrect_options} required).")
            continue
        else: # Exactly target_incorrect_options
            selected_incorrect = incorrect_options

        final_options_for_sample = [correct_option] + selected_incorrect
        random.shuffle(final_options_for_sample) # Shuffle the final list (correct + chosen incorrect)
        final_answer_idx = final_options_for_sample.index(correct_option)

        updated_ids.append(original_id)
        updated_questions.append(question)
        updated_options_lists.append(final_options_for_sample)
        updated_answer_indices.append(final_answer_idx)
        updated_num_options_list.append(len(final_options_for_sample)) # Should be 1 + target_incorrect_options

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
        # logger.debug(f"Filtering out example {example.get('id', 'N/A')} with no options or num_options=0.")
        is_valid = False
    # num_options should be len(options). The check on answer_idx uses num_options.
    # Ensure num_options reflects the actual length of options list.
    elif not (0 <= example.get("answer_idx", -1) < example.get("num_options", 0)):
        # logger.debug(f"Filtering out example {example.get('id', 'N/A')} with invalid answer_idx: {example.get('answer_idx', -1)}, num_options: {example.get('num_options',0)}")
        is_valid = False
    return is_valid

# --- Load and Process MathQA ---
logger.info(f"Loading MathQA data from {DATA_FILE_PATH_MATHQA}...")
try:
    # Ensure MathQA json has an "id" field or generate one. `transform_loaded_data` handles this.
    # If your JSON is one big list, load_dataset usually names the split "train" by default.
    raw_mathqa_loaded = load_dataset("json", data_files=DATA_FILE_PATH_MATHQA, split="train")
    logger.info(f"Successfully loaded {len(raw_mathqa_loaded)} records from MathQA JSON.")
except Exception as e:
    logger.error(f"Failed to load MathQA dataset from {DATA_FILE_PATH_MATHQA}: {e}")
    exit()

logger.info("Transforming MathQA data...")
# It's safer to ensure all original columns are present for `remove_columns`
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
    # Using 'train' split for SciQ. Consider 'validation' for evaluation later.
    raw_sciq_loaded = load_dataset(DATASET_NAME_SCIQ, split="train")
    logger.info(f"Successfully loaded {len(raw_sciq_loaded)} records from SciQ dataset.")
except Exception as e:
    logger.error(f"Failed to load SciQ dataset: {e}")
    exit()

logger.info("Transforming SciQ data...")
sciq_cols_to_remove = raw_sciq_loaded.column_names
processed_sciq_interim = raw_sciq_loaded.map(
    transform_distractor_data_to_common_format,
    batched=True,
    remove_columns=sciq_cols_to_remove
)
original_sciq_count = len(processed_sciq_interim)
processed_sciq_interim = processed_sciq_interim.filter(filter_valid_intermediate_data)
logger.info(f"SciQ: Filtered {original_sciq_count - len(processed_sciq_interim)} invalid examples after initial transformation. Remaining: {len(processed_sciq_interim)}")

# --- Load and Process Code ---
raw_code_loaded = load_from_disk(DATA_CODING_MCQA)['train']
code_cols_to_remove = raw_code_loaded.column_names
processed_code_interim = raw_code_loaded.map(
    transform_distractor_data_to_common_format,
    batched=True,
    remove_columns=code_cols_to_remove,
)
original_code_count = len(processed_code_interim)
processed_code_interim = processed_code_interim.filter(filter_valid_intermediate_data)
logger.info(f"Code: Filtered {original_code_count - len(processed_code_interim)} invalid examples after initial trasnformation. Remaining: {len(processed_code_interim)}")

# --- Load and Process GPQA ---
raw_gpqa_loaded = load_from_disk(DATA_GPQA)['train']
code_cols_to_remove = raw_gpqa_loaded.column_names
processed_gpqa_interim = raw_gpqa_loaded.map(
    transform_distractor_data_to_common_format,
    batched=True,
    remove_columns=code_cols_to_remove,
)
original_gpqa_count = len(processed_gpqa_interim)
processed_gpqa_interim = processed_gpqa_interim.filter(filter_valid_intermediate_data)
logger.info(f"Code: Filtered {original_gpqa_count - len(processed_gpqa_interim)} invalid examples after initial trasnformation. Remaining: {len(processed_gpqa_interim)}")

# --- Combine and Standardize Datasets ---
combined_dataset_interim = concatenate_datasets([processed_mathqa_interim, processed_sciq_interim, processed_code_interim, processed_gpqa_interim])
logger.info(f"Combined dataset size: {len(combined_dataset_interim)}")


logger.info(f"Standardizing options count to 1 correct + {NUM_INCORRECT_OPTIONS} incorrect options...")
# `standardize_options_count` returns the same column names, so no `remove_columns` needed if it updates them.
# If it selects a subset of columns, then `remove_columns` might be needed or ensure it keeps all necessary ones.
# The current `standardize_options_count` outputs the target schema.
raw_dataset = combined_dataset_interim.map(
    lambda x: standardize_options_count(x, target_incorrect_options=NUM_INCORRECT_OPTIONS),
    batched=True
    # No remove_columns needed if standardize_options_count outputs the same schema it received,
    # or if it outputs the target schema and the input schema was a superset.
    # Since standardize_options_count re-constructs the batch, it implicitly handles columns.
)
original_combined_count = len(raw_dataset) # Length after standardize_options_count map
# The standardize_options_count function already filters internally by skipping non-conformant examples
# So, an explicit filter afterward might not be needed unless there are other criteria.
# Let's re-check the count after standardization which might have dropped more examples.
logger.info(f"Filtered {len(combined_dataset_interim) - original_combined_count} examples during option count standardization.")
logger.info(f"Final raw dataset size for preprocessing: {len(raw_dataset)}")

# --- Save the raw data on disk ---
processed_data_save_path = str(Path(OUTPUT_PATH) / "processed" / "raw_dataset")
logging.info(f"Saving processed datasets to disk at {processed_data_save_path}...")
os.makedirs(processed_data_save_path, exist_ok=True)
raw_dataset.save_to_disk(processed_data_save_path)


if len(raw_dataset) == 0:
    logger.error("No data remaining after transformation, combination, and standardization. Please check data and logic.")
    exit()


def preprocess_mcqa_contrastive(examples):
    processed_batch = {
        "positive_input_ids": [], "positive_attention_mask": [], "positive_prompt_len": [],
        "negative_input_ids": [], "negative_attention_mask": [], "negative_prompt_len": [],
        # Keep 'id' for debugging/tracking if needed, ensure Trainer doesn't complain
        "id": []
    }

    for i in range(len(examples['question'])):
        question = examples['question'][i]
        options = examples['options'][i]
        answer_idx = examples['answer_idx'][i]
        num_options = examples['num_options'][i] # This should be 1 (correct) + NUM_INCORRECT_OPTIONS
        example_id = examples['id'][i]

        if not (isinstance(options, list) and len(options) == num_options and 0 <= answer_idx < num_options):
            logger.warning(f"Skipping malformed MCQA example ID {example_id} in preprocess_mcqa_contrastive.")
            continue
        
        # This check is crucial: ensure we have the expected number of options after standardization
        if num_options != (1 + NUM_INCORRECT_OPTIONS):
            logger.warning(f"Skipping example ID {example_id}: Expected {1+NUM_INCORRECT_OPTIONS} options, got {num_options}.")
            continue

        processed_batch["id"].append(example_id) # Keep ID

        correct_option_text = options[answer_idx]
        incorrect_options_texts = [opt for idx, opt in enumerate(options) if idx != answer_idx]

        # This should now yield NUM_INCORRECT_OPTIONS texts
        if len(incorrect_options_texts) != NUM_INCORRECT_OPTIONS:
            logger.warning(f"Skipping example ID {example_id}: Expected {NUM_INCORRECT_OPTIONS} incorrect options, found {len(incorrect_options_texts)}.")
            # Remove the ID if we skip the example
            processed_batch["id"].pop()
            continue

        prompt_positive_prefix = f"Question: {question} Option: "
        full_positive_text = prompt_positive_prefix + correct_option_text + tokenizer.eos_token
        tokenized_positive_prefix = tokenizer(prompt_positive_prefix, add_special_tokens=False)
        positive_prompt_len = len(tokenized_positive_prefix['input_ids'])
        tokenized_positive = tokenizer(full_positive_text, truncation=True, max_length=MAX_SEQ_LENGTH)
        processed_batch["positive_input_ids"].append(tokenized_positive['input_ids'])
        processed_batch["positive_attention_mask"].append(tokenized_positive['attention_mask'])
        processed_batch["positive_prompt_len"].append(positive_prompt_len)

        current_negatives_input_ids = []
        current_negatives_attention_mask = []
        current_negatives_prompt_len = []
        for neg_opt_text in incorrect_options_texts:
            prompt_negative_prefix = f"Question: {question} Option: "
            full_negative_text = prompt_negative_prefix + neg_opt_text + tokenizer.eos_token
            tokenized_negative_prefix = tokenizer(prompt_negative_prefix, add_special_tokens=False)
            negative_prompt_len = len(tokenized_negative_prefix['input_ids'])
            tokenized_negative = tokenizer(full_negative_text, truncation=True, max_length=MAX_SEQ_LENGTH)
            current_negatives_input_ids.append(tokenized_negative['input_ids'])
            current_negatives_attention_mask.append(tokenized_negative['attention_mask'])
            current_negatives_prompt_len.append(negative_prompt_len)

        processed_batch["negative_input_ids"].append(current_negatives_input_ids)
        processed_batch["negative_attention_mask"].append(current_negatives_attention_mask)
        processed_batch["negative_prompt_len"].append(current_negatives_prompt_len)
    
    # If all examples in a batch were skipped, return an empty dict with correct structure
    if not processed_batch["id"]: # Check if any example was processed
        return {key: [] for key in processed_batch.keys()}

    return processed_batch

logger.info("Preprocessing combined dataset for contrastive learning...")
# Remove original columns before tokenization if they are not needed by preprocess_mcqa_contrastive
# `raw_dataset` has 'id', 'question', 'options', 'answer_idx', 'num_options'
# `preprocess_mcqa_contrastive` uses all of them and outputs new ones.
# It's usually safer to remove columns explicitly.
tokenized_dataset_cols_to_remove = [col for col in raw_dataset.column_names if col not in ['id', 'question', 'options', 'answer_idx', 'num_options']]

tokenized_dataset = raw_dataset.map(
    preprocess_mcqa_contrastive,
    batched=True,
    num_proc=max(os.cpu_count() // 4, 1), # Adjust num_proc based on memory and CPU
    remove_columns=raw_dataset.column_names # Remove old columns, keep only what preprocess returns
)

def filter_empty_contrastive(example):
    # Check positive example
    positive_ids = example.get('positive_input_ids')
    if not positive_ids or not isinstance(positive_ids, list) or len(positive_ids) == 0:
        return False

    # Check negative examples structure
    all_negative_options_token_lists = example.get('negative_input_ids')
    if not all_negative_options_token_lists or not isinstance(all_negative_options_token_lists, list):
        return False

    # Check number of negative options (should be NUM_INCORRECT_OPTIONS)
    if len(all_negative_options_token_lists) != NUM_INCORRECT_OPTIONS:
        # logger.debug(f"Filtering out due to incorrect number of negative options ({len(all_negative_options_token_lists)} instead of {NUM_INCORRECT_OPTIONS}) for example ID: {example.get('id', 'N/A')}")
        return False

    # Check each individual negative option's token list
    for single_negative_option_tokens in all_negative_options_token_lists:
        if not isinstance(single_negative_option_tokens, list) or len(single_negative_option_tokens) == 0:
            return False
    return True

original_tokenized_count = len(tokenized_dataset)
tokenized_dataset = tokenized_dataset.filter(filter_empty_contrastive)
logger.info(f"Filtered {original_tokenized_count - len(tokenized_dataset)} invalid examples after contrastive preprocessing.")
logger.info(f"Tokenized dataset for training/evaluation. Number of examples: {len(tokenized_dataset)}")


if len(tokenized_dataset) == 0:
    logger.error("No data after preprocessing for contrastive learning. Check input data and preprocessing functions.")
    exit()

# Splitting dataset
# Ensure 'id' column is not used by train_test_split if it causes issues, though it should be fine.
if len(tokenized_dataset) > 20: # Ensure enough for a split
    train_test_split = tokenized_dataset.train_test_split(test_size=0.1, seed=SEED)
    processed_datasets = DatasetDict({'train': train_test_split['train'], 'eval': train_test_split['test']})
else:
    logger.warning("Dataset too small for train/test split. Using all data for both train and eval.")
    processed_datasets = DatasetDict({'train': tokenized_dataset, 'eval': tokenized_dataset})

train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["eval"]

logger.info(f"Train dataset size: {len(train_dataset)}, Eval dataset size: {len(eval_dataset)}")

# --- Save the tokenized data on disk ---
processed_data_save_path = str(Path(OUTPUT_PATH) / "processed" / "tokenized_dataset")
logging.info(f"Saving processed datasets to disk at {processed_data_save_path}...")
os.makedirs(processed_data_save_path, exist_ok=True)
processed_datasets.save_to_disk(processed_data_save_path)

# --- 4. Custom Data Collator ---
class DataCollatorForContrastiveMCQA:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.data_collator_lm = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def __call__(self, features):
        batch = {}
        # Filter out features that might be malformed (e.g. empty after preprocessing steps)
        valid_features = [f for f in features if f.get("positive_input_ids") and f.get("negative_input_ids")]
        if not valid_features: # If all features in this batch are invalid
            # Return an empty batch or a batch with the correct structure but empty tensors
            # This case should be rare if filtering is done properly before.
            # For now, let's assume Trainer handles or errors on truly empty batches.
            # Or, more robustly:
            logger.warning("Data collator received a batch with no valid features. Returning empty structure.")
            # Construct a dummy structure based on what Trainer expects for an empty batch
            # This depends on the model and trainer. For now, let's proceed hoping valid_features is not empty.
            # If it can be empty, one might need to return None or raise an error,
            # or return a batch of minimal size with padding that the model can ignore.
            # Given the filtering steps, this should be unlikely. If it happens, it's a data issue.
            if not features: return {} # Completely empty input
            # If features were present but all invalid:
            return { # Return minimal structure expected by the model
                "positive_input_ids": torch.empty(0, dtype=torch.long), "positive_attention_mask": torch.empty(0, dtype=torch.long),
                "positive_labels": torch.empty(0, dtype=torch.long), "positive_prompt_len": [],
                "negative_input_ids": [], "negative_attention_mask": [],
                "negative_labels": [], "negative_prompt_len": []
            }


        features = valid_features # Use only valid features for this batch

        positive_examples = [{"input_ids": f["positive_input_ids"], "attention_mask": f["positive_attention_mask"]} for f in features]
        collated_positive = self.data_collator_lm(positive_examples)
        batch["positive_input_ids"] = collated_positive["input_ids"]
        batch["positive_attention_mask"] = collated_positive["attention_mask"]
        batch["positive_labels"] = collated_positive["labels"]
        batch["positive_prompt_len"] = [f["positive_prompt_len"] for f in features]

        # num_negatives_per_prompt should be consistent (NUM_INCORRECT_OPTIONS) due to preprocessing
        num_negatives_per_prompt = 0
        if features and features[0].get("negative_input_ids"):
            num_negatives_per_prompt = len(features[0]["negative_input_ids"]) # Should be NUM_INCORRECT_OPTIONS

        batch["negative_input_ids"] = []
        batch["negative_attention_mask"] = []
        batch["negative_labels"] = []
        batch["negative_prompt_len"] = []

        for i in range(num_negatives_per_prompt):
            current_negative_examples_i = []
            current_negative_prompt_lens_i = []
            for f_idx, f in enumerate(features):
                # Ensure this feature has the i-th negative example and it's valid
                if i < len(f["negative_input_ids"]) and \
                   f["negative_input_ids"][i] and f["negative_attention_mask"][i]:
                    current_negative_examples_i.append({
                        "input_ids": f["negative_input_ids"][i],
                        "attention_mask": f["negative_attention_mask"][i]
                    })
                    current_negative_prompt_lens_i.append(f["negative_prompt_len"][i])
                else:
                    logger.warning(f"Feature {f.get('id', f_idx)} missing or has malformed negative example at index {i}. Padding or inconsistency may occur.")
                    # This indicates an issue upstream if it happens often.
                    # For robustness, one might pad here, but better to fix data pipeline.

            if not current_negative_examples_i: # if no examples were added for this negative index for the whole batch
                # This means for ALL features, the i-th negative was missing/invalid.
                # Add empty tensors to maintain structure, assuming model can handle 0-batch dim for this negative.
                # This should align with how `_get_option_nll` and `compute_loss` handle potentially empty tensors for a negative set.
                # A more robust collator might pad these to match other negatives' sequence lengths if this scenario is common and problematic.
                # For now, let DataCollatorForLanguageModeling handle empty list if passed.
                if features: # Only if there were positive examples to begin with
                    dummy_collated = self.data_collator_lm([]) # Gets empty tensors
                    batch["negative_input_ids"].append(dummy_collated["input_ids"])
                    batch["negative_attention_mask"].append(dummy_collated["attention_mask"])
                    batch["negative_labels"].append(dummy_collated["labels"])
                    batch["negative_prompt_len"].append([])

                continue

            collated_negative_i = self.data_collator_lm(current_negative_examples_i)
            batch["negative_input_ids"].append(collated_negative_i["input_ids"])
            batch["negative_attention_mask"].append(collated_negative_i["attention_mask"])
            batch["negative_labels"].append(collated_negative_i["labels"])
            batch["negative_prompt_len"].append(current_negative_prompt_lens_i)
        return batch

data_collator = DataCollatorForContrastiveMCQA(tokenizer=tokenizer)

# --- 5. Custom Trainer ---
class ContrastiveTrainer(Trainer):
    def __init__(self, *args, margin=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.margin = margin

    def _get_option_nll(self, model, input_ids, attention_mask, labels, prompt_lengths):
        if input_ids.nelement() == 0: # Handle empty input tensor (e.g. if a negative set was empty)
             return torch.empty(0, device=model.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        batch_option_nll = []
        logits = outputs.logits
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

        for i in range(input_ids.size(0)): # Iterate over items in this specific (positive or one negative) batch
            prompt_len = prompt_lengths[i]
            current_labels = shifted_labels[i]
            current_logits = shifted_logits[i]

            option_token_indices_start = max(0, prompt_len - 1)
            option_token_indices = (current_labels != -100) & \
                                   (torch.arange(current_labels.size(0), device=current_labels.device) >= option_token_indices_start)

            if option_token_indices.sum() == 0:
                # logger.warning("No option tokens found for NLL calculation in a sample.")
                batch_option_nll.append(torch.tensor(float('inf'), device=model.device))
                continue

            option_labels = current_labels[option_token_indices]
            option_logits = current_logits[option_token_indices]

            nll_per_token = loss_fct(option_logits.view(-1, option_logits.size(-1)), option_labels.view(-1))
            sum_nll_option = nll_per_token.sum()
            batch_option_nll.append(sum_nll_option)

        if not batch_option_nll: # If loop was skipped for all items
            return torch.empty(0, device=model.device)
        return torch.stack(batch_option_nll)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None): # Removed num_items_in_batch
        positive_input_ids = inputs["positive_input_ids"]
        positive_attention_mask = inputs["positive_attention_mask"]
        positive_labels = inputs["positive_labels"]
        positive_prompt_len = inputs["positive_prompt_len"]

        if positive_input_ids.nelement() == 0: # No positive examples in batch
            logger.warning("Compute_loss called with no positive examples.")
            return torch.tensor(0.0, device=model.device, requires_grad=True) if not return_outputs else (torch.tensor(0.0), None)


        nll_positive = self._get_option_nll(model, positive_input_ids, positive_attention_mask, positive_labels, positive_prompt_len)

        total_loss = 0
        num_valid_contrastive_pairs = 0

        if inputs.get("negative_input_ids") and len(inputs["negative_input_ids"]) > 0:
            num_negatives_sets = len(inputs["negative_input_ids"]) # Should be NUM_INCORRECT_OPTIONS

            for i in range(num_negatives_sets):
                if i >= len(inputs["negative_input_ids"]) or inputs["negative_input_ids"][i].nelement() == 0 :
                    # logger.debug(f"Negative set {i} is empty or missing. Skipping.")
                    continue

                negative_input_ids = inputs["negative_input_ids"][i]
                negative_attention_mask = inputs["negative_attention_mask"][i]
                negative_labels = inputs["negative_labels"][i]
                negative_prompt_len = inputs["negative_prompt_len"][i]

                # Check for consistent batch size for this negative set vs positive set
                if nll_positive.size(0) != negative_input_ids.size(0):
                    logger.warning(f"Batch size mismatch: positive ({nll_positive.size(0)}) vs negative set {i} ({negative_input_ids.size(0)}). Skipping this negative set.")
                    continue
                # Also ensure prompt lengths list matches the batch size for this negative set
                if len(negative_prompt_len) != negative_input_ids.size(0):
                    logger.warning(f"Prompt length list size mismatch for negative set {i}: {len(negative_prompt_len)} vs batch size {negative_input_ids.size(0)}. Skipping.")
                    continue


                nll_negative = self._get_option_nll(model, negative_input_ids, negative_attention_mask, negative_labels, negative_prompt_len)
                
                if nll_negative.nelement() == 0: # if _get_option_nll returned empty
                    continue

                # Ensure nll_positive and nll_negative can be compared element-wise
                # This means they should have the same number of elements (batch size for this comparison)
                if nll_positive.size(0) != nll_negative.size(0):
                    logger.warning(f"NLL tensor size mismatch after _get_option_nll for negative set {i}. Pos: {nll_positive.size(0)}, Neg: {nll_negative.size(0)}. Skipping.")
                    continue


                for j in range(nll_positive.size(0)): # Iterate over batch items
                    if j >= nll_negative.size(0): # Should not happen if sizes matched
                        break
                    if torch.isinf(nll_positive[j]) or torch.isinf(nll_negative[j]):
                        continue

                    loss = F.relu(self.margin + nll_positive[j] - nll_negative[j])
                    total_loss += loss
                    num_valid_contrastive_pairs +=1

        final_loss = total_loss / num_valid_contrastive_pairs if num_valid_contrastive_pairs > 0 else torch.tensor(0.0, device=model.device, requires_grad=True)
        # Dummy outputs if return_outputs is True, actual model outputs are complex here
        return (final_loss, None) if return_outputs else final_loss


# --- 6. Set up Training Arguments ---
logger.info("Setting up training arguments...")
# Effective batch size will be per_device_train_batch_size * gradient_accumulation_steps * num_gpus
# For contrastive loss, each "item" in batch creates 1 positive and N negative forward passes if not batched carefully.
# The current setup batches positives together, and each set of negatives together.
# One sample in `train_dataset` results in 1 positive and NUM_INCORRECT_OPTIONS negative sequences.
# The `per_device_train_batch_size` refers to the number of original samples from `train_dataset`.
training_args_dict = {
    "output_dir": OUTPUT_PATH,
    "overwrite_output_dir": True,
    "num_train_epochs": 2, # Adjust based on dataset size and convergence
    "per_device_train_batch_size": 16,  # Number of MCQA examples per batch. Each expands.
    "per_device_eval_batch_size": 16,   # Similarly for eval
    "gradient_accumulation_steps": 8, # Effective train batch size = 4 * 8 = 32
    "logging_strategy": "steps",
    "logging_steps": 10, # Log more frequently for smaller datasets
    "save_strategy": "epoch",
    "save_total_limit": 2,
    "learning_rate": 5e-5 if not USE_QLORA else 2e-4, # QLoRA might benefit from higher LR
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "lr_scheduler_type": "cosine",
    "report_to": "wandb" if os.environ.get("WANDB_PROJECT") else "none",
    "remove_unused_columns": False, # Critical for custom columns passed by collator
    "logging_dir": f'{OUTPUT_PATH}/logs',
    "bf16": torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 and not USE_QLORA, # bfloat16 for non-QLoRA Ampere+
    "fp16": torch.cuda.is_available() and not (torch.cuda.get_device_capability()[0] >= 8) and not USE_QLORA, # float16 for non-QLoRA pre-Ampere
    # QLoRA handles its own mixed precision via BitsAndBytesConfig. Trainer fp16/bf16 should be False for QLoRA.
    # However, `prepare_model_for_kbit_training` might set `is_loaded_in_8bit` or `is_loaded_in_4bit` which
    # could interact with Trainer's AMP. Let's ensure they are False if USE_QLORA.
    # Update: If using QLoRA with 4-bit, `bnb_4bit_compute_dtype` (e.g. bfloat16) is used for compute,
    # so bf16/fp16 in TrainingArguments might conflict or be redundant. It's safer to set them based on QLoRA.
    # For QLoRA, the compute_dtype is handled by bnb_config.
    # Setting bf16/fp16 in TrainingArguments is for the *optimizer* and *gradient* types if not using deepspeed.
    # If QLoRA `compute_dtype` is bf16, it's good to match.
    "gradient_checkpointing": True if USE_QLORA else False, # Good for QLoRA
    # "ddp_find_unused_parameters": False, # May be needed for DDP with PEFT
}
if USE_QLORA: # QLoRA handles its own precision for base model layers.
    training_args_dict["bf16"] = (QUANTIZATION_BITS == 4 and bnb_config.bnb_4bit_compute_dtype == torch.bfloat16)
    training_args_dict["fp16"] = (QUANTIZATION_BITS == 4 and bnb_config.bnb_4bit_compute_dtype == torch.float16) \
                             or (QUANTIZATION_BITS == 8) # For 8-bit, fp16 is common for optimizer/grads


training_args = TrainingArguments(**training_args_dict)

# --- 7. Initialize Trainer ---
logger.info("Initializing ContrastiveTrainer...")
trainer = ContrastiveTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    margin=CONTRASTIVE_MARGIN,
)

# --- 8. Start Training ---
if train_dataset and len(train_dataset) > 0 :
    logger.info("\n--- Starting training ---")
    try:
        train_result = trainer.train()
        logger.info("--- Training finished ---")

        logger.info("Saving final model/adapters and tokenizer...")
        trainer.save_model(OUTPUT_PATH) # Saves adapter if PEFT, full model otherwise
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
def evaluate_mcqa_likelihood(trainer_instance, eval_dataset_processed):
    # eval_dataset_processed is already tokenized and structured by preprocess_mcqa_contrastive
    logger.info(f"Starting custom evaluation on {len(eval_dataset_processed)} examples.")
    model_to_eval = trainer_instance.model
    # tokenizer_to_eval = trainer_instance.tokenizer # Already available globally as `tokenizer`
    device = model_to_eval.device
    model_to_eval.eval()

    correct_predictions = 0
    total_predictions = 0
    all_nll_correct = []
    all_nll_incorrect_avg = []

    for example_bundle in eval_dataset_processed: # This eval_dataset is already preprocessed
        with torch.no_grad():
            # Positive (Correct) Option
            # The example_bundle contains already tokenized positive_input_ids, positive_attention_mask, positive_prompt_len
            pos_input_ids = torch.tensor([example_bundle["positive_input_ids"]], device=device)
            pos_attn_mask = torch.tensor([example_bundle["positive_attention_mask"]], device=device)
            question_prompt_len_pos = example_bundle["positive_prompt_len"] # This is a scalar

            # Reconstruct labels for _get_option_nll, which expects labels for Causal LM loss
            pos_labels = pos_input_ids.clone()
            # Mask out prompt tokens from labels (original script masks from index 0 to prompt_len)
            # _get_option_nll shifts labels and logits, so prompt_len needs to be accurate for that shift.
            # The `positive_prompt_len` is for the unshifted input.
            # If _get_option_nll uses shifted_labels[i][max(0, prompt_len - 1):], this is fine.

            nll_correct_option_tensor = trainer_instance._get_option_nll(
                model_to_eval, pos_input_ids, pos_attn_mask, pos_labels, [question_prompt_len_pos] # Pass prompt len as a list
            )
            if nll_correct_option_tensor.nelement() == 0 or torch.isinf(nll_correct_option_tensor[0]):
                logger.warning(f"Inf or empty NLL for correct option in eval (ID: {example_bundle.get('id', 'N/A')}). Skipping example.")
                continue
            nll_correct_option = nll_correct_option_tensor[0].item()
            all_nll_correct.append(nll_correct_option)

            current_example_option_scores = [-nll_correct_option] # Score is -NLL (higher is better)

            # Negative (Incorrect) Options
            current_example_nll_incorrect = []
            # example_bundle["negative_input_ids"] is a list of lists of tokens (one list per negative option)
            for i in range(len(example_bundle["negative_input_ids"])):
                neg_input_ids_list = example_bundle["negative_input_ids"][i]
                neg_attn_mask_list = example_bundle["negative_attention_mask"][i]
                # example_bundle["negative_prompt_len"] is a list of lists of prompt_lengths.
                # We need the i-th list, and then its 0-th element since we process one item at a time here.
                # No, negative_prompt_len in example_bundle is already a list of scalar prompt_lengths for each negative option.
                question_prompt_len_neg = example_bundle["negative_prompt_len"][i] # This is a scalar for the i-th negative option


                neg_input_ids = torch.tensor([neg_input_ids_list], device=device)
                neg_attn_mask = torch.tensor([neg_attn_mask_list], device=device)
                neg_labels = neg_input_ids.clone() # Reconstruct labels

                nll_incorrect_option_tensor = trainer_instance._get_option_nll(
                    model_to_eval, neg_input_ids, neg_attn_mask, neg_labels, [question_prompt_len_neg]
                )
                if nll_incorrect_option_tensor.nelement() == 0 or torch.isinf(nll_incorrect_option_tensor[0]):
                    logger.warning(f"Inf or empty NLL for incorrect option {i} in eval (ID: {example_bundle.get('id', 'N/A')}). Assigning bad score.")
                    current_example_option_scores.append(-float('inf'))
                    current_example_nll_incorrect.append(float('inf'))
                else:
                    nll_inc_opt = nll_incorrect_option_tensor[0].item()
                    current_example_option_scores.append(-nll_inc_opt)
                    current_example_nll_incorrect.append(nll_inc_opt)
            
            if current_example_nll_incorrect: # If there were any valid incorrect options
                all_nll_incorrect_avg.append(sum(current_example_nll_incorrect) / len(current_example_nll_incorrect))

            if not current_example_option_scores: continue # Should not happen if data is valid

            # The correct option's score is at index 0 of current_example_option_scores.
            # We compare it against scores of incorrect options.
            predicted_best_idx = current_example_option_scores.index(max(current_example_option_scores))
            if predicted_best_idx == 0: # Correct option was chosen
                correct_predictions += 1
            total_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    avg_nll_correct = sum(all_nll_correct) / len(all_nll_correct) if all_nll_correct else float('nan')
    avg_nll_incorrect = sum(all_nll_incorrect_avg) / len(all_nll_incorrect_avg) if all_nll_incorrect_avg else float('nan')

    eval_metrics = {
        "eval_mcqa_accuracy": accuracy,
        "eval_avg_nll_correct_option": avg_nll_correct,
        "eval_avg_nll_incorrect_options": avg_nll_incorrect,
        "eval_total_predictions_custom": total_predictions, # Renamed to avoid conflict with Trainer's own metrics
    }
    logger.info(f"Custom MCQA Likelihood Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
    logger.info(f"Avg NLL Correct: {avg_nll_correct:.4f}, Avg NLL Incorrect (avg per question): {avg_nll_incorrect:.4f}")
    return eval_metrics


if eval_dataset and len(eval_dataset) > 0:
    logger.info("\n--- Starting custom evaluation ---")
    try:
        eval_results = evaluate_mcqa_likelihood(trainer, eval_dataset) # eval_dataset is already processed
        logger.info(f"Custom Evaluation Results: {eval_results}")
        # Log these metrics. Trainer might have already logged its own eval metrics.
        # If trainer.evaluate() was called, it would use compute_metrics.
        # This is a separate evaluation.
        if os.environ.get("WANDB_PROJECT") and wandb.run:
            wandb.log(eval_results) # Log custom eval metrics to wandb
        # Save custom metrics
        # Convert torch tensors in metrics to float if any, before saving to JSON
        for k, v in eval_results.items():
            if isinstance(v, torch.Tensor):
                eval_results[k] = v.item()
        trainer.save_metrics("custom_eval", eval_results)


    except Exception as e:
        logger.error(f"An error occurred during custom evaluation: {e}", exc_info=True)
else:
    logger.warning("No evaluation data available. Skipping custom evaluation.")

if os.environ.get("WANDB_PROJECT") and wandb.run:
    wandb.finish()

logger.info("\nScript completed.")
