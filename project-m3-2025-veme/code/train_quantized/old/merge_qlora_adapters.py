import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import logging
import re # For your transform_loaded_data
import ast # For your transform_loaded_data
from datasets import load_dataset # Added for loading your JSON data

BASE_MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
ADAPTER_PATH = "./qwen_mcqa_contrastive_qlora" # Where your adapters are saved by trainer
MERGED_MODEL_SAVE_PATH = "./qwen_mcqa_merged_model_only_mathqa" # Path for the bf16/fp16 merged model
MAX_SEQ_LENGTH = 256

# Configure logger (basic configuration for demonstration)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# --- Part 1: bf16/fp16 merged model ---
logger.info(f"Loading base model {BASE_MODEL_NAME} for merging...")
compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=compute_dtype,
    trust_remote_code=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

logger.info(f"Loading PEFT model from {ADAPTER_PATH}...")
peft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

logger.info("Merging LoRA adapters into the base model...")
merged_model = peft_model.merge_and_unload()
logger.info("Merging complete.")

logger.info(f"Saving higher-precision merged model to {MERGED_MODEL_SAVE_PATH}...")
os.makedirs(MERGED_MODEL_SAVE_PATH, exist_ok=True)
merged_model.save_pretrained(MERGED_MODEL_SAVE_PATH)
tokenizer.save_pretrained(MERGED_MODEL_SAVE_PATH)
logger.info(f"Higher-precision merged model and tokenizer saved to {MERGED_MODEL_SAVE_PATH}.")

del base_model
del peft_model
del merged_model
torch.cuda.empty_cache()


# --- Part 2: Post-Training Quantization of the Merged Model using AutoGPTQ (via Optimum) ---
logger.info("--- Starting AutoGPTQ Post-Training Quantization ---")

from transformers import GPTQConfig
from optimum.gptq import GPTQQuantizer

GPTQ_QUANTIZED_MODEL_SAVE_PATH = "./qwen_mcqa_gptq_4bit_standalone_model"
CALIBRATION_DATASET_SIZE = 128 # Number of samples for calibration, adjust as needed

# --- 1. Prepare Calibration Dataset using your provided logic ---
DATA_FILE_PATH = "../data/mathqa/train.json" # Path to your JSON data file

logger.info(f"Loading data from {DATA_FILE_PATH} for GPTQ calibration...")

try:
    raw_dataset_loaded = load_dataset("json", data_files=DATA_FILE_PATH, split="train")
    logger.info(f"Successfully loaded {len(raw_dataset_loaded)} records for calibration.")
except Exception as e:
    logger.error(f"Failed to load dataset from {DATA_FILE_PATH}: {e}")
    logger.error("Please ensure the DATA_FILE_PATH is correct and the file is a valid JSON or JSON Lines file.")
    exit()

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
                                logger.debug(f"Could not extract option from list item: '{item_str_cleaned}'. Adding as is if non-empty.")
                                if item_str_cleaned:
                                    parsed_options.append(item_str_cleaned)
                        is_list_repr_successfully_parsed = True
                except (ValueError, SyntaxError, TypeError) as e:
                    logger.debug(f"ast.literal_eval failed for options_input '{options_input}': {e}. Will treat as a flat string.")
            
            if not is_list_repr_successfully_parsed:
                option_matches = re.findall(
                    r"[a-zA-Z]\s*\)\s*(.*?)(?=\s*[a-zA-Z]\s*\)\s*|$)",
                    options_input,
                    re.IGNORECASE | re.DOTALL
                )
                if option_matches:
                    parsed_options = [match.strip() for match in option_matches]
                elif options_input.strip():
                    logger.warning(f"Regex parsing found no options in flat string: '{options_input}'.")
                    if not option_matches:
                         logger.error(f"Robust regex parsing failed for flat string options: '{options_input}'. No options extracted.")
                         parsed_options = []
        
        elif isinstance(options_input, list):
            for item_str_or_val in options_input:
                item_str = str(item_str_or_val).strip()
                match = re.match(r"^[a-zA-Z]\s*\)\s*(.*)", item_str, re.DOTALL)
                if match:
                    parsed_options.append(match.group(1).strip())
                else:
                    parsed_options.append(item_str)
        else:
            logger.warning(f"Options field is neither a string nor a list: type={type(options_input)}, content='{options_input}'. No options parsed.")
            parsed_options = []

        answer_idx = -1
        if correct_char and 'a' <= correct_char <= 'z':
            answer_idx = ord(correct_char) - ord('a')
        else:
            logger.debug(f"Invalid correct_char: '{correct_char}' for problem: '{problem_text}'.")
        
        if not (0 <= answer_idx < len(parsed_options)):
            logger.debug(f"Invalid answer_idx {answer_idx} for {len(parsed_options)} options. Problem: '{problem_text}'. Skipping.")
            continue

        new_questions.append(problem_text)
        new_options_lists.append(parsed_options)
        new_answer_indices.append(answer_idx)
        new_num_options_list.append(len(parsed_options))
        batch_ids = examples.get('id')
        current_id = f"gen_id_{i}"
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

logger.info("Transforming loaded data for calibration...")
raw_dataset = raw_dataset_loaded.map(
    transform_loaded_data,
    batched=True,
    remove_columns=raw_dataset_loaded.column_names
)

def filter_valid_transformed_data(example):
    is_valid = True
    if not example["options"] or example["num_options"] == 0:
        is_valid = False
    if not (0 <= example["answer_idx"] < example["num_options"]):
        is_valid = False
    return is_valid

original_count = len(raw_dataset)
raw_dataset = raw_dataset.filter(filter_valid_transformed_data)
filtered_count = len(raw_dataset)
if original_count > filtered_count:
    logger.warning(f"Filtered out {original_count - filtered_count} invalid examples after transformation for calibration.")

if len(raw_dataset) == 0:
    logger.error("The processed calibration dataset is empty after transformation and filtering. Cannot proceed.")
    exit()

# Shuffle the dataset and select a subset for calibration
# We will use the 'question' field which corresponds to the original 'Problem' field.
num_samples_for_calibration = min(CALIBRATION_DATASET_SIZE, len(raw_dataset))
calibration_subset = raw_dataset.shuffle(seed=42).select(range(num_samples_for_calibration))

calibration_texts = [f"Question: {item['question']}, Answer: {item['options'][torch.randint(0, item["num_options"], (1,)).item()]}" for item in calibration_subset] # 'question' is from your transform_loaded_data

if not calibration_texts:
    logger.error("Calibration dataset is empty after processing. AutoGPTQ requires calibration data. Exiting.")
    exit()

logger.info(f"Using {len(calibration_texts)} samples for GPTQ calibration from '{DATA_FILE_PATH}'.")
# --- End of new calibration data preparation ---


# 2. Load Tokenizer for GPTQ
tokenizer_for_gptq = AutoTokenizer.from_pretrained(MERGED_MODEL_SAVE_PATH, trust_remote_code=True)
if tokenizer_for_gptq.pad_token is None:
    tokenizer_for_gptq.pad_token = tokenizer_for_gptq.eos_token
    logger.info(f"Set tokenizer_for_gptq.pad_token to {tokenizer_for_gptq.eos_token}")


# 3. Initialize GptqConfig and GPTQQuantizer
if MAX_SEQ_LENGTH is None: # Add a check for MAX_SEQ_LENGTH
    logger.error("MAX_SEQ_LENGTH is not defined. Please set it before running GPTQ.")
    exit()

gptq_transformers_config = GPTQConfig(
    bits=4,
    dataset=calibration_texts,    # This should be a list of strings
    tokenizer=tokenizer_for_gptq,
    desc_act=False,
    # group_size=128,           # Optional: Add if you want to customize
    # damp_percent=0.01,        # Optional: Add if you want to customize
    # use_exllama=False,        # Optional: Set based on model compatibility
)
quantizer = GPTQQuantizer(
    bits=gptq_transformers_config.bits,
    dataset=gptq_transformers_config.dataset,  # Pass the actual list of calibration strings
    tokenizer=gptq_transformers_config.tokenizer,
    model_seqlen=MAX_SEQ_LENGTH,             # This is where model_seqlen is typically used
    desc_act=gptq_transformers_config.desc_act,
    # group_size=gptq_transformers_config.group_size, # Uncomment if you defined it in GPTQConfig
    # damp_percent=gptq_transformers_config.damp_percent, # Uncomment if you defined it
    # use_exllama=getattr(gptq_transformers_config, 'use_exllama', None), # Safely get if defined
                                                                   # Or directly set: use_exllama=False
)


# 5. Load the (higher-precision) merged model that we want to quantize
logger.info(f"Loading merged model from {MERGED_MODEL_SAVE_PATH} for GPTQ quantization...")
# compute_dtype should be defined from Part 1, or defined here again if this is a separate scope
# compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
model_to_quantize = AutoModelForCausalLM.from_pretrained(
    MERGED_MODEL_SAVE_PATH,
    torch_dtype=compute_dtype, # Or the precision it was saved in
    trust_remote_code=True,
    device_map="auto"
)

# 6. Quantize the model
logger.info("Starting GPTQ quantization process... This may take some time.")
try:
    
    quantized_model = quantizer.quantize_model(
        model=model_to_quantize,  # Explicitly name the 'model' argument
        tokenizer=tokenizer_for_gptq  # Explicitly pass the tokenizer
    )
    logger.info("Model quantization complete.")

    # Save the quantized model
    logger.info(f"Saving GPTQ 4-bit quantized model to {GPTQ_QUANTIZED_MODEL_SAVE_PATH}...")
    os.makedirs(GPTQ_QUANTIZED_MODEL_SAVE_PATH, exist_ok=True) # Ensure directory exists
    quantized_model.save_pretrained(GPTQ_QUANTIZED_MODEL_SAVE_PATH)
    
    # Save the tokenizer
    tokenizer_for_gptq.save_pretrained(GPTQ_QUANTIZED_MODEL_SAVE_PATH)
    
    logger.info(f"GPTQ 4-bit quantized model and tokenizer saved to {GPTQ_QUANTIZED_MODEL_SAVE_PATH}")

except Exception as e:
    logger.error(f"Error during GPTQ quantization or saving: {e}", exc_info=True) # Log full traceback
    logger.error("Make sure you have enough GPU memory and check compatibility (e.g., consider use_exllama=False in GPTQConfig/GPTQQuantizer).")


# Example of how to upload (commented out):
# from huggingface_hub import HfApi
# api = HfApi()
# your_gptq_repo_id = "your_username/qwen3-0.6B-mathqa-gptq" # Choose a new repo ID
# api.upload_folder(
#     folder_path=GPTQ_QUANTIZED_MODEL_SAVE_PATH,
#     repo_id=your_gptq_repo_id,
#     repo_type="model",
#     commit_message="Upload GPTQ 4-bit quantized model (merged from QLoRA)"
# )
# print(f"GPTQ model uploaded to: https://huggingface.co/{your_gptq_repo_id}")