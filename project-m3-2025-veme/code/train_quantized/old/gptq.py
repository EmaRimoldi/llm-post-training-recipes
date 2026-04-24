
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import logging
from datasets import load_dataset
import re
import ast
from torch.utils.data import DataLoader
import torch
import random

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s (%(filename)s:%(lineno)d)"
)


model_id = "../sft/qwen3_0.6B_sft_next_token_masked_loss/final_model/"
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Ensure tokenizer has a padding token if it doesn't; many models require it.
# Qwen models typically use <|endoftext|> as eos and potentially for padding if pad_token is not set.
if tokenizer.pad_token is None:
    if tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Tokenizer.pad_token was None, set to eos_token: {tokenizer.eos_token}")
    else:
        # Add a generic padding token if no EOS token is available, though this is less ideal.
        # For Qwen, '<|endoftext|>' is common. If your model uses a different one, adjust.
        tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
        logger.info("Tokenizer.pad_token was None and no eos_token, added '<|endoftext|>' as pad_token.")



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
                    if not option_matches: # Re-confirm
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
            logger.debug(f"Invalid answer_idx {answer_idx} for {len(parsed_options)} options. Problem: '{problem_text}'. Skipping this example.")
            continue # Skip this example by not adding it to the batch's results

        new_questions.append(problem_text) # This is the 'Problem' field
        new_options_lists.append(parsed_options)
        new_answer_indices.append(answer_idx)
        new_num_options_list.append(len(parsed_options))
        batch_ids = examples.get('id') # Safely get 'id'
        current_id = f"gen_id_{i}" # Default generated ID
        if batch_ids and i < len(batch_ids):
            current_id = batch_ids[i]
        new_ids.append(current_id)

    return {
        "id": new_ids,
        "question": new_questions, # This will be used for calibration text
        "options": new_options_lists,
        "answer_idx": new_answer_indices,
        "num_options": new_num_options_list,
    }

def filter_valid_transformed_data(example):
    is_valid = True
    if not example["options"] or example["num_options"] == 0:
        is_valid = False
    if not (0 <= example["answer_idx"] < example["num_options"]):
        is_valid = False
    return is_valid


def prepare_mathqa_calibration(
    data_file_path: str,
    num_samples: int,
    # tokenizer_for_gptq: AutoTokenizer, # Tokenizer might be passed if needed for max_length check
    # max_seq_length: int # max_seq_length for GPTQ can be handled by tokenizer in GPTQConfig
):
    """
    Prepares a list of text strings for GPTQ calibration.
    """
    logger.info(f"Loading and preparing MathQA calibration dataset from: {data_file_path}")

    try:
        raw_dataset_loaded = load_dataset("json", data_files=data_file_path, split="train", trust_remote_code=True)
        logger.info(f"Successfully loaded {len(raw_dataset_loaded)} records for MathQA calibration.")
    except Exception as e:
        logger.error(f"Failed to load dataset from {data_file_path}: {e}", exc_info=True)
        raise

    logger.info("Transforming loaded MathQA data for calibration...")
    # Determine columns to remove carefully; if raw_dataset_loaded has 'id', it might be kept by transform_loaded_data
    # It's safer to remove columns that are definitely not in the output of transform_loaded_data
    # Or, specify `features` in `map` if you know the exact output structure.
    # For now, assuming transform_loaded_data correctly returns all needed columns and we want to keep only those.
    current_columns = raw_dataset_loaded.column_names
    processed_dataset = raw_dataset_loaded.map(
        transform_loaded_data,
        batched=True,
        remove_columns=current_columns
    )

    original_count = len(processed_dataset)
    # Filter again to be sure, especially if transform_loaded_data's skipping logic is complex
    processed_dataset = processed_dataset.filter(filter_valid_transformed_data)
    filtered_count = len(processed_dataset)
    if original_count > filtered_count:
        logger.warning(f"Filtered out {original_count - filtered_count} invalid MathQA examples after transformation/filtering.")

    if len(processed_dataset) == 0:
        raise ValueError("The processed MathQA calibration dataset is empty after transformation and filtering.")

    actual_samples_to_use = min(num_samples, len(processed_dataset))
    if actual_samples_to_use < num_samples:
        logger.warning(f"Using {actual_samples_to_use} MathQA calibration samples, less than requested {num_samples}.")
    if actual_samples_to_use == 0:
        raise ValueError("No MathQA samples available for calibration after filtering.")

    calibration_subset = processed_dataset.shuffle(seed=42).select(range(actual_samples_to_use))


    # Extract the text data for calibration. GPTQConfig usually expects a list of strings.
    # The "question" field from your `transform_loaded_data` function seems most appropriate.
    
    calibration_texts = []
    for i in range(len(calibration_subset)):
        example = calibration_subset[i] # Access example by index for HF Dataset
        question_text = example["question"]
        options_list = example["options"]
        num_options = example["num_options"] # This should be len(options_list)

        if num_options > 0:
            # Randomly select an option text
            # The user suggested something like calibration_subset[options][torch.rand(num_options)]
            # A simpler and correct way to pick a random option text:
            selected_option_text = random.choice(options_list)
            
            formatted_string = f"Question: {question_text}, answer: {selected_option_text}"
            calibration_texts.append(formatted_string)
        else:
            # This case should ideally be filtered out before this stage by filter_valid_transformed_data
            # If it still occurs, we can just use the question or skip.
            # For calibration, it's better to have well-formed examples.
            logger.warning(
                f"Skipping example for calibration string construction due to no options "
                f"(ID: {example.get('id', 'N/A')}, Question: '{question_text[:50]}...'). "
                f"This might indicate an issue with earlier filtering."
            )
            # Optionally, if you still want to include it somehow:
            # formatted_string = f"Question: {question_text}, answer: <No options available>"
            # calibration_texts.append(formatted_string)


    logger.info(f"Prepared {len(calibration_texts)} formatted text samples for GPTQ calibration.")
    # Example of the first formatted calibration text:
    if calibration_texts:
        logger.info(f"First formatted calibration sample: '{calibration_texts[0][:250]}...'")
    else:
        # This would be a critical issue if no calibration texts are generated.
        logger.error("No calibration texts were generated. GPTQ will likely fail. Check data processing and filtering.")
        # Consider raising an error here if len(calibration_texts) == 0 and actual_samples_to_use > 0
        if actual_samples_to_use > 0 : # actual_samples_to_use was defined earlier in your function
             raise ValueError("Failed to generate any calibration strings despite having samples selected. Check formatting logic.")
    
    return calibration_texts


# Configuration
TARGET_WEIGHT_BITS = 4
CUSTOM_DATA_FILE_PATH = "../data/mathqa/train.json"
NUM_CALIBRATION_SAMPLES = 128 # You can increase this for better results, e.g., 128, 256, or 512
# MAX_SEQ_LENGTH for GPTQ is often handled internally by the tokenizer passed to GPTQConfig,
# or by the model's max length. If you need to truncate texts before passing to GPTQ, you'd do it here.
# For now, we pass full question texts.

logger.info("Starting GPTQ quantization process...")
calibration_data_strings = prepare_mathqa_calibration(
    CUSTOM_DATA_FILE_PATH,
    NUM_CALIBRATION_SAMPLES
    # tokenizer, # Not passing tokenizer here anymore, GPTQConfig will use its own
    # MAX_SEQ_LENGTH
)


# 1. Define GPTQ configuration
# The `dataset` parameter in GPTQConfig expects a list of strings or an iterable yielding strings.
logger.info(f"Initializing GPTQConfig with {TARGET_WEIGHT_BITS}-bit quantization.")
gptq_config = GPTQConfig(
    bits=TARGET_WEIGHT_BITS,
    dataset=calibration_data_strings,
    tokenizer=tokenizer, # Pass the tokenizer for GPTQ to use, e.g., for sequence length
    # model_seqlen=MAX_SEQ_LENGTH, # Optionally set max sequence length for calibration
                                 # If not set, GPTQ might use tokenizer.model_max_length
    # For more advanced options, refer to Optimum/AutoGPTQ documentation:
    # group_size=128, # Example: common group size
    # damp_percent=0.1, # Example: dampening percentage
    # desc_act=False, # Example: for ActivationOrder quantization
)

# 2. Quantize the model
# This step performs the actual quantization using the calibration data.
# It can be memory and time-intensive.
logger.info(f"Loading model '{model_id}' for GPTQ quantization. This may take a while...")
quantized_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=gptq_config,
    device_map="auto", # Or "cuda:0" if you want to pin it to a specific GPU
    # torch_dtype=torch.float16, # Optional: specify dtype for loading unquantized layers if needed
    trust_remote_code=True # Add if your model requires it
)
logger.info("Model quantization complete.")

# 3. Save the quantized model
output_dir = f"./qwen_sft_gptq_mathqa_only"
logger.info(f"Saving quantized model and tokenizer to {output_dir}...")
quantized_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

logger.info(f"GPTQ quantized model saved successfully to {output_dir}")





