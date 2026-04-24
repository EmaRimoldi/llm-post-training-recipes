import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import logging
import os
import re
import ast

# NOTE: llmcompressor is a Neural Magic library.
# You'd need to install it: pip install llmcompressor
# This script is a high-level illustration and might require adjustments
# based on the exact llmcompressor API and your specific model.
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

# Corrected import for import_modules_for_quantization_stages
# If this line causes a ModuleNotFoundError, ensure 'sparseml' is installed
# (it should be a dependency of 'llmcompressor').
from sparseml.pytorch.utils import import_modules_for_quantization_stages

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s (%(filename)s:%(lineno)d)"
)

# Ensure quantization modules are imported and an enabled stage is set.
# This call is generally important for SparseML/llmcompressor to prepare for quantization.
# If you remove this and quantization fails or isn't applied, this call was likely necessary.
import_modules_for_quantization_stages(True)


# --- Hardcoded Parameters ---
MODEL_PATH = "Qwen/Qwen3-0.6B-Base"
OUTPUT_PATH = "./output_smoothquant_qwen3_0.6b_mathqa"
SMOOTHING_STRENGTH_ALPHA = 0.5
TARGET_WEIGHT_BITS = 4
# Custom Calibration Data Path
CUSTOM_DATA_FILE_PATH = "../data/mathqa/train.json" # CHANGED
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQ_LENGTH = 512 # For tokenization during calibration
QUANT_CONFIG_NAME_DISPLAY = "W4A16-SmoothQuant-MathQA-Calib"

# --- User's Data Transformation Functions (defined globally) ---
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

# --- New Calibration DataLoader Preparation ---
def prepare_mathqa_calibration_dataloader(
    tokenizer,
    data_file_path: str,
    num_samples: int,
    max_seq_length: int
):
    """
    Prepares a DataLoader for calibration using the custom MathQA data.
    """
    logger.info(f"Loading and preparing MathQA calibration dataset from: {data_file_path}")

    try:
        # Load the dataset from the JSON file
        # Using trust_remote_code=True generally, though for local JSON it might not be strictly needed
        # unless the dataset script itself relies on it.
        raw_dataset_loaded = load_dataset("json", data_files=data_file_path, split="train", trust_remote_code=True)
        logger.info(f"Successfully loaded {len(raw_dataset_loaded)} records for MathQA calibration.")
    except Exception as e:
        logger.error(f"Failed to load dataset from {data_file_path}: {e}", exc_info=True)
        raise

    logger.info("Transforming loaded MathQA data for calibration...")
    processed_dataset = raw_dataset_loaded.map(
        transform_loaded_data, # User's global function
        batched=True,
        remove_columns=raw_dataset_loaded.column_names # Remove original cols, keep only what transform_loaded_data returns
    )

    original_count = len(processed_dataset)
    processed_dataset = processed_dataset.filter(filter_valid_transformed_data) # User's global filter function
    filtered_count = len(processed_dataset)
    if original_count > filtered_count:
        logger.warning(f"Filtered out {original_count - filtered_count} invalid MathQA examples after transformation.")

    if len(processed_dataset) == 0:
        raise ValueError("The processed MathQA calibration dataset is empty after transformation and filtering.")

    actual_samples_to_use = min(num_samples, len(processed_dataset))
    if actual_samples_to_use < num_samples:
        logger.warning(f"Using {actual_samples_to_use} MathQA calibration samples, less than requested {num_samples}.")
    if actual_samples_to_use == 0:
        raise ValueError("No MathQA samples available for calibration after filtering.")

    calibration_subset = processed_dataset.shuffle(seed=42).select(range(actual_samples_to_use))

    # For calibration data for llmcompressor, we need to provide text that gets tokenized.
    # We'll use the 'question' field which corresponds to the original 'Problem' text.
    def prepare_for_tokenization(examples):
        return {"text_for_calibration": examples["question"]} # 'question' is output by transform_loaded_data

    # Select only the text data to be tokenized
    text_dataset = calibration_subset.map(prepare_for_tokenization, batched=True, remove_columns=calibration_subset.column_names)
    
    def tokenize_for_calib(examples):
        return tokenizer(
            examples["text_for_calibration"],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt" # llmcompressor DataLoader usually expects PyTorch tensors
        )

    tokenized_calib_dataset = text_dataset.map(tokenize_for_calib, batched=True, remove_columns=["text_for_calibration"])
    
    # Collate function to batch the tokenized data correctly
    def collate_fn(batch):
        input_ids = torch.stack([item['input_ids'].squeeze(0) for item in batch])
        attention_mask = torch.stack([item['attention_mask'].squeeze(0) for item in batch])
        # Add other keys if your model/llmcompressor expects them (e.g., labels, token_type_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    logger.info(f"Prepared DataLoader with {len(tokenized_calib_dataset)} tokenized MathQA samples for calibration.")
    return DataLoader(tokenized_calib_dataset, batch_size=1, collate_fn=collate_fn) # batch_size=1 is common for oneshot calib


def apply_smoothquant(
    model_name_or_path: str,
    output_dir: str,
    # Removed direct calibration dataset name/config, will use custom_data_file_path
    custom_data_file_path: str, # New parameter
    num_calibration_samples: int,
    max_seq_length: int,
    smoothing_strength: float,
    target_weight_bits: int,
    quant_config_name_display: str
):
    logger.info(f"Starting SmoothQuant for model: {model_name_or_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Calibration: {num_calibration_samples} samples from custom data path '{custom_data_file_path}', max_seq_len {max_seq_length}")
    logger.info(f"SmoothQuant strength (alpha): {smoothing_strength}")
    logger.info(f"Target quantization display name: {quant_config_name_display}")
    logger.info(f"Actual target weight bits for recipe: {target_weight_bits}")

    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading FP32 model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float32, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Error loading model or tokenizer: {e}", exc_info=True)
        return

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set tokenizer.pad_token to tokenizer.eos_token: {tokenizer.eos_token}")
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
            logger.info("Added a new pad_token [PAD] and resized model embeddings.")
    model.eval()

    logger.info("Preparing custom MathQA calibration dataloader...")
    try:
        # Use the new dataloader function with the custom data path
        calibration_dataloader = prepare_mathqa_calibration_dataloader(
            tokenizer,
            custom_data_file_path, # Pass the path here
            num_calibration_samples,
            max_seq_length
        )
        print(calibration_dataloader)
        assert 0
    except Exception as e:
        logger.error(f"Failed to prepare MathQA calibration data: {e}", exc_info=True)
        return

    recipe_str = f"""
    version: 1.2.0
    modifiers:
      - !SmoothQuantModifier
        mapping_name: "smooth_stage"
        smoothing_strength: {smoothing_strength}
        targets: ["Linear"]
        ignore: []
      - !QuantizationModifier
        mapping_name: "weight_quant_stage"
        config_groups:
          group_0:
            weights:
              num_bits: {target_weight_bits}
              symmetric: True
            input_activations: null
            output_activations: null
        targets: ["Linear"]
        ignore: ["lm_head"]
    """
    logger.info("Applying SmoothQuant and Quantization using llmcompressor.oneshot...")
    logger.info(f"Recipe used:\n{recipe_str}")
    try:
        oneshot(
            model=model,
            tokenizer=tokenizer,
            recipe=recipe_str,
            data_loader=calibration_dataloader,
        )
        logger.info("SmoothQuant and quantization applied successfully.")
    except Exception as e:
        logger.error(f"Error during llmcompressor.oneshot application: {e}", exc_info=True)
        return

    logger.info(f"Saving quantized model to {output_dir}...")
    try:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        with open(os.path.join(output_dir, "recipe_applied.yaml"), "w") as f:
            f.write(recipe_str)
        logger.info(f"Quantized model, tokenizer, and recipe saved to {output_dir}")
    except Exception as e:
        logger.error(f"Error saving quantized model: {e}", exc_info=True)

# --- Main script execution ---
if __name__ == "__main__":
    logger.info("Starting hardcoded SmoothQuant script with custom MathQA calibration data...")
    apply_smoothquant(
        model_name_or_path=MODEL_PATH,
        output_dir=OUTPUT_PATH,
        custom_data_file_path=CUSTOM_DATA_FILE_PATH, # Use the new path
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        max_seq_length=MAX_SEQ_LENGTH,
        smoothing_strength=SMOOTHING_STRENGTH_ALPHA,
        target_weight_bits=TARGET_WEIGHT_BITS,
        quant_config_name_display=QUANT_CONFIG_NAME_DISPLAY
    )
    logger.info("Hardcoded SmoothQuant script finished.")