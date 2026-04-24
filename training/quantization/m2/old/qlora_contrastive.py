# Import necessary libraries
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorWithPadding, BitsAndBytesConfig, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType # Added for QLoRA
import torch
import torch.nn.functional as F
import logging
import os
import wandb
import random
import re
import ast

# For the quantized model it makes sense to train to achieve higher performance rather than 
# simply relying on PTQ methods. So it would be nice to explore mixtures of PEFTs methods with
# quantization methods. Common PEFT methods are:
# Selective: This approach involves fine-tuning a carefully chosen subset of the pre-trained model’s weights.
# Reparametrization: Reparametrization methods such as Low-Rank Adaptation (LoRA), create a low-dimensional representation of a specific module (set of parameters e.g. query vector of a transformer model) in the original LLM.
# Additive: This approach involves adjusting the pre-trained model by adding new modules for fine-tuning. These modules are further trained to incorporate knowledge of the new domain into the pre-trained.

# In this script we will a well known approach that combines quantization with LoRA: QLoRA.

# --- Script Configuration ---
# QLoRA Configuration
USE_QLORA = True  # Set to True to enable QLoRA, False for full fine-tuning
QUANTIZATION_BITS = 4  # 4 or 8. Set to None or other if USE_QLORA is False
LORA_R = 16  # LoRA rank
LORA_ALPHA = 32  # LoRA alpha
LORA_DROPOUT = 0.05 # LoRA dropout

# Model and Training Configuration
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
MAX_SEQ_LENGTH = 256
CONTRASTIVE_MARGIN = 1.0
OUTPUT_PATH = "./quantization/qwen_mcqa_contrastive_qlora" if USE_QLORA else "./quantization/qwen_mcqa_contrastive_full"

# --- Logging and GPU Setup ---
log_file = os.path.join(OUTPUT_PATH, 'output_contrastive_qlora.log' if USE_QLORA else 'output_contrastive_full.log')
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

logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
    logger.info(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Transformers version: {transformers.__version__}")
    # from peft import __version__ as peft_version; logger.info(f"PEFT version: {peft_version}") # Requires peft to be imported

# --- 1. Load Tokenizer ---
logger.info(f"Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Set tokenizer.pad_token to {tokenizer.eos_token} ({tokenizer.pad_token_id})")
if tokenizer.eos_token is None: # Should not be needed for recent Qwen models
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
        logger.warning(f"Unsupported QUANTIZATION_BITS: {QUANTIZATION_BITS} for QLoRA. QLoRA will not be applied effectively without quantization bits.")
        # Fallback to no QLoRA if bits are not 4 or 8 for this setup
        # USE_QLORA = False # Or raise error

model_kwargs = {"trust_remote_code": True}
if bnb_config:
    model_kwargs["quantization_config"] = bnb_config
    # For QLoRA, device_map="auto" is often used to put layers on GPU and offload others if needed.
    # However, Trainer typically handles device placement. If you have issues with multi-GPU or specific setups,
    # you might need to experiment with device_map. For single GPU, it's usually fine without.
    # model_kwargs["device_map"] = "auto" # Add if experiencing OOMs or for multi-GPU with bitsandbytes
else: # Not using QLoRA or invalid QLoRA bits
    model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

# Common setup for the model (before PEFT if QLoRA is used)
model.config.pad_token_id = tokenizer.pad_token_id
# Resize token embeddings if tokenizer vocab was expanded (e.g., by adding pad_token)
# This should generally be done on the base model.
if len(tokenizer) > model.config.vocab_size:
    logger.info(f"Resizing token embeddings from {model.config.vocab_size} to {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))


if USE_QLORA and bnb_config: # Apply PEFT if QLoRA is truly active
    logger.info("Preparing model for k-bit training and applying LoRA...")
    # Gradient checkpointing is enabled through TrainingArguments later,
    # prepare_model_for_kbit_training will respect that.
    model = prepare_model_for_kbit_training(model)

    # Determine target modules based on model type (Qwen2 is common)
    # For Qwen2: "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    # For older Qwen1/1.5: "c_attn", "c_proj", "w1", "w2"
    # It's best to inspect your specific model architecture if unsure.
    if "Qwen2" in MODEL_NAME or "Qwen3" in MODEL_NAME: # Assuming Qwen3 is similar to Qwen2 for LoRA
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else: # Fallback for older Qwen or other models; ADAPT AS NEEDED
        logger.warning(f"Potentially non-optimal LoRA target modules for {MODEL_NAME}. Please verify.")
        target_modules = ["c_attn", "c_proj", "w1", "w2", "q_proj", "k_proj", "v_proj", "o_proj"] # A broader set

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=target_modules,
        lora_dropout=LORA_DROPOUT,
        bias="none", # Typically "none" for QLoRA
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    logger.info("LoRA applied. Trainable parameters:")
    model.print_trainable_parameters()
else:
    logger.info("Proceeding with full model fine-tuning (QLoRA not enabled or configured).")

logger.info("Model (potentially with QLoRA) and tokenizer loaded.")


# --- 3. Load and Preprocess MCQA Datasets ---

# Define the path to your JSON data file
DATA_FILE_PATH = "data/mathqa/train.json"

logger.info(f"Loading data from {DATA_FILE_PATH}...")

# Load the dataset from the JSON file
# load_dataset will infer if it's a single JSON array or JSON Lines
try:
    # If your JSON file contains a single list of records, 'train' split might be inferred or you might need to specify field.
    # If it's a .jsonl file, it's typically straightforward.
    raw_dataset_loaded = load_dataset("json", data_files=DATA_FILE_PATH, split="train")
    logger.info(f"Successfully loaded {len(raw_dataset_loaded)} records.")
except Exception as e:
    logger.error(f"Failed to load dataset from {DATA_FILE_PATH}: {e}")
    logger.error("Please ensure the DATA_FILE_PATH is correct and the file is a valid JSON or JSON Lines file.")
    logger.error("If it's a single JSON file with a specific structure (e.g., data under a key), you might need to adjust load_dataset parameters (e.g. field='your_data_field').")
    exit()

def transform_loaded_data(examples):
    """
    Transforms data loaded from the user's JSON format to the format
    expected by preprocess_mcqa_contrastive.
    """
    new_questions = []
    new_options_lists = []
    new_answer_indices = []
    new_num_options_list = []
    new_ids = []

    num_examples_in_batch = len(examples['Problem'])

    for i in range(num_examples_in_batch):
        problem_text = examples['Problem'][i]
        options_input = examples['options'][i] # This is the field to parse
        correct_char = examples['correct'][i].lower()

        parsed_options = []

        if isinstance(options_input, str):
            # Scenario 1: options_input is a string representation of a list
            # e.g., "['a) text one, with comma', 'b) text two']"
            is_list_repr_successfully_parsed = False
            if options_input.startswith("[") and options_input.endswith("]"):
                try:
                    evaluated_list = ast.literal_eval(options_input)
                    if isinstance(evaluated_list, list):
                        for item_str in evaluated_list:
                            item_str_cleaned = str(item_str).strip()
                            # Extract text after "X) " from each item
                            # Match a letter (a-z, A-Z), optional spaces, ')', optional spaces, then capture the rest.
                            match = re.match(r"^[a-zA-Z]\s*\)\s*(.*)", item_str_cleaned, re.DOTALL)
                            if match:
                                parsed_options.append(match.group(1).strip())
                            else:
                                logger.warning(f"Could not extract option from list item: '{item_str_cleaned}' in options input: '{options_input}'. Adding item as is if non-empty.")
                                if item_str_cleaned: # Avoid adding empty strings if an item was just "a)"
                                    parsed_options.append(item_str_cleaned) # Fallback for this item
                        is_list_repr_successfully_parsed = True
                except (ValueError, SyntaxError, TypeError) as e:
                    logger.debug(f"ast.literal_eval failed for options_input '{options_input}': {e}. Will treat as a flat string.")
                    # Fall through to flat string parsing if ast.literal_eval fails

            if not is_list_repr_successfully_parsed:
                # Scenario 2: options_input is a flat string
                # e.g., "a) text one, with comma b) text two" or "a) text one, b) text two"
                # Regex: find a letter, optional spaces, ')', optional spaces,
                # then capture (non-greedily) everything until the next similar pattern or end of string.
                # re.DOTALL allows '.' to match newline characters.
                option_matches = re.findall(
                    r"[a-zA-Z]\s*\)\s*(.*?)(?=\s*[a-zA-Z]\s*\)\s*|$)",
                    options_input,
                    re.IGNORECASE | re.DOTALL
                )
                if option_matches:
                    parsed_options = [match.strip() for match in option_matches]
                elif options_input.strip(): # If regex found nothing but string wasn't empty
                    logger.warning(f"Regex parsing found no options in flat string: '{options_input}'. The string will be split by commas as a last resort if applicable, or treated as a single option if no commas.")
                    # Last resort fallback (closer to original but still risky if options have commas)
                    # This part is tricky because simple comma split is the source of the original problem.
                    # If "a ) b )" structure is guaranteed, the regex above should work.
                    # If it fails, it implies the format is very different.
                    # For now, if regex fails on a non-empty string, we log a warning and `parsed_options` remains empty or has what regex found.
                    # Consider if any other fallback is safe given "a) b) c)..." constraint.
                    # If the regex above is good, this 'elif options_input.strip():' might not be needed,
                    # as an empty `option_matches` on a non-empty `options_input` means the "a) text" pattern isn't there.
                    if not option_matches: # Re-confirm, if regex really yielded nothing
                        logger.error(f"Robust regex parsing failed for flat string options: '{options_input}'. No options extracted.")
                        parsed_options = []


        elif isinstance(options_input, list):
            # Scenario 3: options_input is already a Python list
            for item_str_or_val in options_input:
                item_str = str(item_str_or_val).strip()
                # Check if items in the list are already prefixed like "a) text"
                match = re.match(r"^[a-zA-Z]\s*\)\s*(.*)", item_str, re.DOTALL)
                if match:
                    parsed_options.append(match.group(1).strip())
                else:
                    # Assume it's already a clean option text if not prefixed
                    parsed_options.append(item_str)
        else:
            logger.warning(f"Options field is neither a string nor a list: type={type(options_input)}, content='{options_input}'. No options parsed.")
            parsed_options = [] # Ensure it's an empty list

        # --- End of new parsing logic for options_input ---

        # Convert correct character ('a', 'b', ...) to 0-based index
        answer_idx = -1 # Default to invalid
        if correct_char and 'a' <= correct_char <= 'z':
            answer_idx = ord(correct_char) - ord('a')
        else:
            logger.error(f"Invalid correct_char: '{correct_char}' for problem: '{problem_text}'. Cannot determine answer_idx.")
            # Continue to append example, but answer_idx validation will likely fail it

        # Basic validation for answer_idx
        if not (0 <= answer_idx < len(parsed_options)):
            logger.error(f"Invalid or out-of-bounds answer_idx {answer_idx} for {len(parsed_options)} parsed options. Problem: '{problem_text}'. Correct char: '{correct_char}'. Options input: '{options_input}'. Parsed: '{parsed_options}'. Skipping this example by not adding it.")
            continue # Skip adding this problematic example to the batch

        new_questions.append(problem_text)
        new_options_lists.append(parsed_options)
        new_answer_indices.append(answer_idx)
        new_num_options_list.append(len(parsed_options))
        # Handle 'id' safely
        batch_ids = examples.get('id')
        current_id = f"gen_id_{i}" # Default generated ID
        if batch_ids and i < len(batch_ids):
            current_id = batch_ids[i]
        new_ids.append(current_id)

    return {
        "id": new_ids,
        "question": new_questions,
        "options": new_options_lists,
        "answer_idx": new_answer_indices,
        "num_options": new_num_options_list,
    }

logger.info("Transforming loaded data to the required format...")
# Apply the transformation. `batched=True` is generally faster.
# The `transform_loaded_data` function needs to handle receiving batches (dict of lists).
raw_dataset = raw_dataset_loaded.map(
    transform_loaded_data,
    batched=True,
    # Remove original columns that are not needed by preprocess_mcqa_contrastive
    # This depends on what columns load_dataset inferred. We only need the output of transform_loaded_data.
    remove_columns=raw_dataset_loaded.column_names
)

# Filter out any examples that might have become invalid during transformation
# (e.g., no options parsed, or answer_idx out of bounds if not handled strictly in transform)
def filter_valid_transformed_data(example):
    is_valid = True
    if not example["options"] or example["num_options"] == 0:
        # logger.warning(f"Filtering out example with no options: {example.get('id', 'N/A')}")
        is_valid = False
    if not (0 <= example["answer_idx"] < example["num_options"]):
        # logger.warning(f"Filtering out example with invalid answer_idx: {example.get('id', 'N/A')}, answer_idx: {example['answer_idx']}, num_options: {example['num_options']}")
        is_valid = False
    return is_valid

original_count = len(raw_dataset)
raw_dataset = raw_dataset.filter(filter_valid_transformed_data)
filtered_count = len(raw_dataset)
if original_count > filtered_count:
    logger.warning(f"Filtered out {original_count - filtered_count} invalid examples after transformation.")


logger.info(f"Data transformation complete. Number of processed records: {len(raw_dataset)}")
if len(raw_dataset) == 0:
    logger.error("No data remaining after transformation and filtering. Please check your JSON data and transformation logic.")
    exit()

# At this point, `raw_dataset` contains your data in the format:
# {'id': ..., 'question': ..., 'options': [...], 'answer_idx': ..., 'num_options': ...}
# and is ready for the `preprocess_mcqa_contrastive` function.


# BE EXTRA CAREFUL ABOUT THE NUMBER OF INCORRECT OPTIONS, YOU WANT TO HANDLE THAT HERE RATHER THAN IN THE COLLATOR
# for MathQA you alsways have 5 options with 1 correct and 4 not correct.
# in general if that's not the case bootstrap the sample to have a fixed number of options, otherwise you won't be able to create batches
def preprocess_mcqa_contrastive(examples):
    processed_batch = {
        "positive_input_ids": [], "positive_attention_mask": [], "positive_prompt_len": [],
        "negative_input_ids": [], "negative_attention_mask": [], "negative_prompt_len": [],
    }

    for i in range(len(examples['question'])):
        question = examples['question'][i]
        options = examples['options'][i] 
        answer_idx = examples['answer_idx'][i]
        num_options = examples['num_options'][i]

        if not (isinstance(options, list) and len(options) == num_options and 0 <= answer_idx < num_options):
            logger.warning(f"Skipping malformed MCQA example: {examples['id'][i] if 'id' in examples else 'N/A'}")
            continue

        correct_option_text = options[answer_idx]
        incorrect_options_texts = [opt for idx, opt in enumerate(options) if idx != answer_idx]
        
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
    return processed_batch

logger.info("Preprocessing MCQA dataset for contrastive learning...")
tokenized_dataset = raw_dataset.map(
    preprocess_mcqa_contrastive,
    batched=True,
    num_proc=max(os.cpu_count() // 2, 1), # Use half CPUs or at least 1
)

def filter_empty_contrastive(example):
    # 1. Check the positive example's input_ids
    # It should exist, be a list, and not be empty.
    positive_ids = example.get('positive_input_ids')
    if not positive_ids or not isinstance(positive_ids, list) or len(positive_ids) == 0:
        # logger.debug(f"Filtering out due to positive_input_ids issue for example ID: {example.get('id', 'N/A')}")
        return False

    # 2. Check the overall structure for negative examples
    # 'negative_input_ids' should be a list.
    all_negative_options_token_lists = example.get('negative_input_ids')
    if not all_negative_options_token_lists or not isinstance(all_negative_options_token_lists, list):
        # logger.debug(f"Filtering out due to missing or non-list negative_input_ids for example ID: {example.get('id', 'N/A')}")
        return False

    # 3. Check if there are exactly 4 negative options, as per your data structure
    # (since you have 5 total options, 1 correct means 4 incorrect)
    if len(all_negative_options_token_lists) != 4:
        # logger.debug(f"Filtering out due to incorrect number of negative options ({len(all_negative_options_token_lists)} instead of 4) for example ID: {example.get('id', 'N/A')}")
        return False

    # 4. Check each individual negative option's token list
    # Each item in all_negative_options_token_lists should be a list of token IDs, and it shouldn't be empty.
    for single_negative_option_tokens in all_negative_options_token_lists:
        if not isinstance(single_negative_option_tokens, list) or len(single_negative_option_tokens) == 0:
            # logger.debug(f"Filtering out due to an empty or non-list token list for a negative option for example ID: {example.get('id', 'N/A')}")
            return False
            
    # If all checks pass, the example is valid
    return True

tokenized_dataset = tokenized_dataset.filter(filter_empty_contrastive)
logger.info(f"Tokenized dataset. Number of examples: {len(tokenized_dataset)}")

if len(tokenized_dataset) == 0:
    logger.error("No data after preprocessing. Check input data and preprocess_mcqa_contrastive function.")
    exit()

if len(tokenized_dataset) > 20: # Ensure enough for a split
    train_test_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    processed_datasets = DatasetDict({'train': train_test_split['train'], 'eval': train_test_split['test']})
else:
    processed_datasets = DatasetDict({'train': tokenized_dataset, 'eval': tokenized_dataset}) # Use all for eval if too small for split
train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["eval"]

# --- Save the data on disk ---
logging.info("Saving data on disk...")
processed_datasets.save_to_disk("./quantization/data/qlora_contrastive_train_eval_mathqaonly")

# --- 4. Custom Data Collator ---
class DataCollatorForContrastiveMCQA:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.data_collator_lm = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def __call__(self, features):
        batch = {}
        positive_examples = [{"input_ids": f["positive_input_ids"], "attention_mask": f["positive_attention_mask"]} for f in features]
        collated_positive = self.data_collator_lm(positive_examples)
        batch["positive_input_ids"] = collated_positive["input_ids"]
        batch["positive_attention_mask"] = collated_positive["attention_mask"]
        batch["positive_labels"] = collated_positive["labels"]
        batch["positive_prompt_len"] = [f["positive_prompt_len"] for f in features]

        num_negatives_per_prompt = 0
        if features and features[0]["negative_input_ids"]: # Check if negative_input_ids exist and is not empty
            # Assuming negative_input_ids is a list of lists of tokens
            # Example: features[0]["negative_input_ids"] = [[neg1_tokens], [neg2_tokens], ...]
             num_negatives_per_prompt = len(features[0]["negative_input_ids"])

        # NOTE: num_negatives_per_prompt is the number of negative options in the batch (e.g. for mathQA this number is always 4)
        
        batch["negative_input_ids"] = []
        batch["negative_attention_mask"] = []
        batch["negative_labels"] = []
        batch["negative_prompt_len"] = []

        for i in range(num_negatives_per_prompt):
            # Ensure that all features have this i-th negative example
            current_negative_examples_i = []
            current_negative_prompt_lens_i = []
            for f_idx, f in enumerate(features):
                if i < len(f["negative_input_ids"]): # Check if this feature has the i-th negative
                    current_negative_examples_i.append({
                        "input_ids": f["negative_input_ids"][i],
                        "attention_mask": f["negative_attention_mask"][i]
                    })
                    current_negative_prompt_lens_i.append(f["negative_prompt_len"][i])
                else:
                    # This case should ideally not happen if preprocess_mcqa_contrastive is consistent
                    # Or handle by padding with dummy data if lengths can vary significantly (more complex)
                    logger.warning(f"Feature {f_idx} has fewer than {i+1} negative examples. Skipping this negative for this feature.")
                    # Add dummy/empty placeholder that collator can handle, or ensure consistent num_negatives
                    # For simplicity, if this happens, it might lead to batch construction issues.
                    # A robust solution is to ensure all examples have the same number of negatives (e.g., 3).
                    # For now, let's assume consistent number of negatives from preprocessing.
                    pass


            if not current_negative_examples_i: # if no examples were added for this negative index
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
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        batch_option_nll = []
        logits = outputs.logits
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

        for i in range(input_ids.size(0)):
            prompt_len = prompt_lengths[i]
            current_labels = shifted_labels[i]
            current_logits = shifted_logits[i]
            
            # Option part starts at index `prompt_len -1` in shifted_labels
            # Ensure prompt_len is at least 1 to avoid negative indexing if prompt_len is 0
            option_token_indices_start = max(0, prompt_len - 1) 
            option_token_indices = (current_labels != -100) & \
                                   (torch.arange(current_labels.size(0), device=current_labels.device) >= option_token_indices_start)
            
            if option_token_indices.sum() == 0:
                # If this happens often, check MAX_SEQ_LENGTH and typical prompt/option lengths
                # logger.warning("No option tokens found for NLL calculation in a sample during _get_option_nll.")
                batch_option_nll.append(torch.tensor(float('inf'), device=model.device))
                continue

            option_labels = current_labels[option_token_indices]
            option_logits = current_logits[option_token_indices]
            
            nll_per_token = loss_fct(option_logits.view(-1, option_logits.size(-1)), option_labels.view(-1))
            sum_nll_option = nll_per_token.sum()
            batch_option_nll.append(sum_nll_option)
            
        return torch.stack(batch_option_nll)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        positive_input_ids = inputs["positive_input_ids"]
        positive_attention_mask = inputs["positive_attention_mask"]
        positive_labels = inputs["positive_labels"]
        positive_prompt_len = inputs["positive_prompt_len"]
        
        nll_positive = self._get_option_nll(model, positive_input_ids, positive_attention_mask, positive_labels, positive_prompt_len)
        
        total_loss = 0
        num_valid_contrastive_pairs = 0

        # Check if negative samples are present and correctly batched
        if inputs["negative_input_ids"] and len(inputs["negative_input_ids"]) > 0:
            num_negatives_sets = len(inputs["negative_input_ids"]) # Should be number of negative options

            for i in range(num_negatives_sets):
                # Ensure the batch for this negative set is not empty
                if inputs["negative_input_ids"][i].nelement() == 0 : continue # Skip if this negative batch is empty
                
                negative_input_ids = inputs["negative_input_ids"][i]
                negative_attention_mask = inputs["negative_attention_mask"][i]
                negative_labels = inputs["negative_labels"][i]
                negative_prompt_len = inputs["negative_prompt_len"][i] # This is a list of prompt_lengths

                # Check if batch sizes match for positive and current negative set
                if nll_positive.size(0) != negative_input_ids.size(0):
                    # logger.warning(f"Batch size mismatch between positive ({nll_positive.size(0)}) and negative set {i} ({negative_input_ids.size(0)}). Skipping this negative set.")
                    continue

                nll_negative = self._get_option_nll(model, negative_input_ids, negative_attention_mask, negative_labels, negative_prompt_len)
                
                for j in range(nll_positive.size(0)): # Iterate over batch items
                    if torch.isinf(nll_positive[j]) or torch.isinf(nll_negative[j]):
                        continue 

                    loss = F.relu(self.margin + nll_positive[j] - nll_negative[j])
                    total_loss += loss
                    num_valid_contrastive_pairs +=1
        
        final_loss = total_loss / num_valid_contrastive_pairs if num_valid_contrastive_pairs > 0 else torch.tensor(0.0, device=model.device, requires_grad=True)
        return final_loss


# --- 6. Set up Training Arguments ---
logger.info("Setting up training arguments...")
training_args_dict = {
    "output_dir": OUTPUT_PATH,
    "overwrite_output_dir": True,
    "num_train_epochs": 1, # Adjust, QLoRA often needs more epochs than full finetune on same data amount
    "per_device_train_batch_size": 16, # Keep low for contrastive; each item expands
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 16, # Effective batch size = 1 * 16 = 16
    "logging_strategy": "steps",
    "logging_steps": 5, # Log more frequently for small datasets/debugging
    "save_strategy": "epoch",
    "save_total_limit": 2,
    "learning_rate": 5e-5 if not USE_QLORA else 2e-4, # QLoRA can sometimes use higher LR
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "report_to": "wandb" if "WANDB_PROJECT" in os.environ else "none",
    "remove_unused_columns": False, # Critical for custom columns
    "logging_dir": f'{OUTPUT_PATH}/logs',
    "bf16": torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
    "fp16": False, # Do not use with bf16
    "gradient_checkpointing": False, # Enable for QLoRA to save memory
    # "ddp_find_unused_parameters": False, # May be needed in DDP if some PEFT params are not used in forward
}
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
        
        # Save final model (adapters if QLoRA, full model otherwise)
        logger.info("Saving final model/adapters and tokenizer...")
        # For PEFT models, trainer.save_model() saves the adapter.
        # The base model is not saved by default here.
        trainer.save_model(OUTPUT_PATH) # Saves to output_dir
        # Tokenizer can be saved alongside if needed, though usually not modified by PEFT
        tokenizer.save_pretrained(OUTPUT_PATH)
        logger.info(f"Model/adapters and tokenizer saved to {OUTPUT_PATH}")

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
else:
    logger.warning("No training data available. Skipping training.")

# --- 9. Custom Evaluation ---
def evaluate_mcqa_likelihood(trainer_instance, eval_dataset_processed):
    model_to_eval = trainer_instance.model
    tokenizer_to_eval = trainer_instance.tokenizer
    device = model_to_eval.device
    model_to_eval.eval() # Set model to evaluation mode
    
    correct_predictions = 0
    total_predictions = 0
    all_nll_correct = []
    all_nll_incorrect_avg = []

    for example_bundle in eval_dataset_processed:
        with torch.no_grad(): # Ensure no gradients are computed during eval
            # Positive (Correct) Option
            question_prompt_len_pos = example_bundle["positive_prompt_len"]
            pos_input_ids = torch.tensor([example_bundle["positive_input_ids"]], device=device)
            pos_attn_mask = torch.tensor([example_bundle["positive_attention_mask"]], device=device)
            pos_labels = pos_input_ids.clone()
            pos_labels[:, :question_prompt_len_pos] = -100
            pos_labels[pos_input_ids == tokenizer_to_eval.pad_token_id] = -100
            
            nll_correct_option_tensor = trainer_instance._get_option_nll(
                model_to_eval, pos_input_ids, pos_attn_mask, pos_labels, [question_prompt_len_pos]
            )
            if torch.isinf(nll_correct_option_tensor[0]): 
                # logger.warning("Inf NLL for correct option in eval, skipping example.")
                continue
            nll_correct_option = nll_correct_option_tensor[0].item()
            all_nll_correct.append(nll_correct_option)
            
            current_example_option_scores = [-nll_correct_option] # Score is -NLL
            
            # Negative (Incorrect) Options
            current_example_nll_incorrect = []
            for i in range(len(example_bundle["negative_input_ids"])):
                neg_input_ids = torch.tensor([example_bundle["negative_input_ids"][i]], device=device)
                neg_attn_mask = torch.tensor([example_bundle["negative_attention_mask"][i]], device=device)
                question_prompt_len_neg = example_bundle["negative_prompt_len"][i]
                neg_labels = neg_input_ids.clone()
                neg_labels[:, :question_prompt_len_neg] = -100
                neg_labels[neg_input_ids == tokenizer_to_eval.pad_token_id] = -100
                
                nll_incorrect_option_tensor = trainer_instance._get_option_nll(
                    model_to_eval, neg_input_ids, neg_attn_mask, neg_labels, [question_prompt_len_neg]
                )
                if torch.isinf(nll_incorrect_option_tensor[0]):
                    # logger.warning("Inf NLL for an incorrect option in eval.")
                    # Assign a very high NLL (bad score) if inf, or skip this option.
                    # For simplicity, we'll add a high NLL, making it unlikely to be chosen.
                    current_example_option_scores.append(-float('inf')) 
                    current_example_nll_incorrect.append(float('inf'))
                else:
                    nll_inc_opt = nll_incorrect_option_tensor[0].item()
                    current_example_option_scores.append(-nll_inc_opt)
                    current_example_nll_incorrect.append(nll_inc_opt)

            if current_example_nll_incorrect: # If there were any valid incorrect options
                 all_nll_incorrect_avg.append(sum(current_example_nll_incorrect) / len(current_example_nll_incorrect))

        if not current_example_option_scores: continue # Should not happen if data is valid

        predicted_best_idx = current_example_option_scores.index(max(current_example_option_scores))
        # In this setup, the correct option's score is always at index 0 of current_example_option_scores
        if predicted_best_idx == 0:
            correct_predictions += 1
        total_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    avg_nll_correct = sum(all_nll_correct) / len(all_nll_correct) if all_nll_correct else float('nan')
    avg_nll_incorrect = sum(all_nll_incorrect_avg) / len(all_nll_incorrect_avg) if all_nll_incorrect_avg else float('nan')
    
    eval_metrics = {
        "eval_mcqa_accuracy": accuracy,
        "eval_avg_nll_correct_option": avg_nll_correct,
        "eval_avg_nll_incorrect_options": avg_nll_incorrect,
        "eval_total_predictions": total_predictions,
    }
    logger.info(f"Custom MCQA Likelihood Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
    logger.info(f"Avg NLL Correct: {avg_nll_correct:.4f}, Avg NLL Incorrect: {avg_nll_incorrect:.4f}")
    return eval_metrics

if eval_dataset and len(eval_dataset) > 0:
    logger.info("\n--- Starting custom evaluation ---")
    eval_results = evaluate_mcqa_likelihood(trainer, eval_dataset)
    logger.info(f"Custom Evaluation Results: {eval_results}")
    if trainer.args.report_to and "wandb" in trainer.args.report_to and wandb.run:
        wandb.log(eval_results) # Log metrics to wandb
else:
    logger.warning("No evaluation data. Skipping custom evaluation.")

if trainer.args.report_to and "wandb" in trainer.args.report_to and wandb.run:
    wandb.finish()

logger.info("\nScript completed.")