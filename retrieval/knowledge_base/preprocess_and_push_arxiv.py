import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from huggingface_hub import HfApi
from tqdm.notebook import tqdm
import logging
import sys
import time
import os


def require_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Set the {name} environment variable before running this script.")
    return value

# --- 1. Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) # Direct logs to standard output
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting comprehensive script for ArXiv dataset preparation and upload.")

# --- 2. Configuration Variables ---
DATASET_NAME = "common-pile/arxiv_abstracts"
SPLIT_NAME = "train"
TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L12-v2"

# Filtering and Selection Parameters
MAX_TOKEN_LENGTH_THRESHOLD = 512 # Keep abstracts with token_length <= 512
TOP_K_LONGEST = 100000 # Select the top K longest abstracts after filtering
BATCH_SIZE = 1000 # Adjust for tokenization and filtering performance

# Hugging Face Hub Upload Parameters
HF_TOKEN = require_env_var("HF_TOKEN")
USERNAME = os.getenv("HF_USERNAME", "EmaRimoldi")
REPO_NAME = os.getenv("HF_DATASET_REPO_NAME", "mnlp_arxiv_100k")
# ---------------------------------

# --- 3. Load the raw dataset (retaining 'id' and 'text') ---
logger.info(f"Attempting to load original dataset '{DATASET_NAME}' split '{SPLIT_NAME}', keeping 'id' and 'text' columns...")
try:
    start_time = time.time()
    # Explicitly specify columns to keep, though 'text' and 'id' are usually default.
    dataset = load_dataset(DATASET_NAME, split=SPLIT_NAME)
    loading_time = time.time() - start_time
    logger.info(f"Dataset loaded successfully. Number of rows: {len(dataset)}. Took {loading_time:.2f} seconds.")

    # Verify essential columns are present
    if "id" not in dataset.column_names:
        logger.error("The original dataset does not contain an 'id' column. This is unexpected. Exiting.")
        sys.exit(1)
    if "text" not in dataset.column_names:
        logger.error("The original dataset does not contain a 'text' column. This is unexpected. Exiting.")
        sys.exit(1)

except Exception as e:
    logger.error(f"Error loading dataset '{DATASET_NAME}': {e}", exc_info=True)
    logger.error("Please ensure you have an active internet connection or try specifying a different split.")
    sys.exit(1)

# --- 4. Load the tokenizer ---
logger.info(f"Attempting to load tokenizer: '{TOKENIZER_NAME}'...")
try:
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    loading_time = time.time() - start_time
    logger.info(f"Tokenizer loaded successfully. Took {loading_time:.2f} seconds.")
except Exception as e:
    logger.error(f"Error loading tokenizer '{TOKENIZER_NAME}': {e}", exc_info=True)
    logger.error("Please check the tokenizer name or your internet connection.")
    sys.exit(1)

# --- 5. Define token length calculation function ---
def calculate_token_length(batch):
    """
    Calculates the token length for each 'text' entry in a batch using the pre-loaded tokenizer.
    """
    texts = batch["text"]
    try:
        tokenized_inputs = tokenizer(
            texts,
            return_length=True,
            truncation=False, # Get the actual length, not truncated length
            padding=False      # No padding needed for length calculation
        )
        batch["token_length"] = tokenized_inputs["length"]
    except Exception as e:
        logger.error(f"Error during tokenization for a batch: {e}", exc_info=True)
        # Assign 0 length if error occurs for the entire batch to prevent script crash
        batch["token_length"] = [0] * len(texts)
    return batch

# --- 6. Process the dataset to add 'token_length' ---
logger.info(f"Starting token length calculation. Processing with batch_size={BATCH_SIZE}.")
start_processing_time = time.time()

try:
    processed_dataset = dataset.map(
        calculate_token_length,
        batched=True,
        batch_size=BATCH_SIZE,
        desc="Calculating token lengths"
    )
    processing_time = time.time() - start_processing_time
    logger.info(f"Token length calculation complete. Processed {len(processed_dataset)} rows in {processing_time:.2f} seconds.")
    logger.info(f"Average processing speed: {len(processed_dataset) / processing_time:.2f} rows/second.")

except Exception as e:
    logger.error(f"An error occurred during dataset processing: {e}", exc_info=True)
    sys.exit(1)

# --- 7. Filter out rows with token_length > MAX_TOKEN_LENGTH_THRESHOLD ---
logger.info(f"Filtering out abstracts with token length > {MAX_TOKEN_LENGTH_THRESHOLD}...")
start_filter_time = time.time()

filtered_dataset_step1 = processed_dataset.filter(
    lambda example: example["token_length"] <= MAX_TOKEN_LENGTH_THRESHOLD,
    num_proc=os.cpu_count() if os.cpu_count() else 1, # Use multiple processes, default to 1 if not available
    desc="Filtering by max token length"
)

filter_time = time.time() - start_filter_time
logger.info(f"Step 1: Filtered dataset now has {len(filtered_dataset_step1)} rows (originally {len(processed_dataset)}). Took {filter_time:.2f} seconds.")
logger.info(f"{len(processed_dataset) - len(filtered_dataset_step1)} rows were removed due to exceeding {MAX_TOKEN_LENGTH_THRESHOLD} tokens.")

# --- 8. Sort by token_length (descending) and select the top K ---
logger.info(f"Sorting by token length and selecting the top {TOP_K_LONGEST} rows...")
start_sort_time = time.time()

try:
    # Convert to Pandas for easier sorting, then back to Dataset
    # This might be memory-intensive for very large intermediate datasets.
    # Select only the needed columns ('text', 'id', 'token_length') to minimize memory usage
    df = filtered_dataset_step1.select_columns(['text', 'id', 'token_length']).to_pandas()
    
    # Sort in descending order based on 'token_length'
    df_sorted = df.sort_values(by='token_length', ascending=False)
    
    # Select the top K entries
    final_df = df_sorted.head(TOP_K_LONGEST)
    
    # Convert back to Hugging Face Dataset, preserving original index to avoid re-indexing if not needed
    final_rag_dataset = Dataset.from_pandas(final_df, preserve_index=False) # preserve_index=False ensures clean integer index

except Exception as e:
    logger.error(f"Error during sorting or selecting top K: {e}", exc_info=True)
    logger.error("Consider reducing TOP_K_LONGEST or using a more memory-efficient sorting approach for extremely large datasets.")
    sys.exit(1)

sort_select_time = time.time() - start_sort_time
logger.info(f"Step 2: Final RAG dataset created with {len(final_rag_dataset)} rows. Took {sort_select_time:.2f} seconds.")


# --- 9. Filter columns to "text" and "id", then rename "id" to "source" ---
logger.info("Filtering columns to keep 'text' and 'id', then renaming 'id' to 'source'...")
try:
    # Select only the desired columns ('text' and 'id' as 'token_length' is no longer needed in the final dataset)
    # The 'id' column is now guaranteed to exist from the initial load.
    final_filtered_and_renamed_dataset = final_rag_dataset.select_columns(["text", "id"])
    
    # Rename the 'id' column to 'source'
    final_filtered_and_renamed_dataset = final_filtered_and_renamed_dataset.rename_column("id", "source")
    
    logger.info(f"Final dataset columns: {final_filtered_and_renamed_dataset.column_names}")
    logger.info(f"Sample entry from final dataset: {final_filtered_and_renamed_dataset[0]}")

except Exception as e:
    logger.error(f"Error during final column filtering or renaming: {e}", exc_info=True)
    sys.exit(1)


# --- 10. Push the dataset to Hugging Face Hub ---
api = HfApi()
full_repo_id = f"{USERNAME}/{REPO_NAME}"

logger.info(f"Attempting to create/check Hugging Face repository: {full_repo_id} (type: dataset, private: False)")
try:
    # --- CORRECTED LINE HERE ---
    api.create_repo(repo_id=full_repo_id, private=False, repo_type="dataset", token=HF_TOKEN)
    logger.info(f"Repository '{full_repo_id}' created successfully!")
except Exception as e:
    if "You already have a dataset named" in str(e) or "RepositoryAlreadyExists" in str(e): # Added generic "RepositoryAlreadyExists" check
        logger.warning(f"Repository '{full_repo_id}' already exists. Proceeding with the upload.")
    else:
        logger.error(f"Error creating repository '{full_repo_id}': {e}", exc_info=True)
        sys.exit(1)

logger.info(f"Starting upload of the final dataset to Hugging Face repo: {full_repo_id}...")
start_upload_time = time.time()
try:
    final_filtered_and_renamed_dataset.push_to_hub(
        repo_id=full_repo_id,
        token=HF_TOKEN,
        commit_message="Upload custom RAG dataset (filtered by length and top K longest, 'id' renamed to 'source')"
    )
    upload_time = time.time() - start_upload_time
    logger.info(f"✔ Dataset uploaded successfully to Hugging Face repo: {full_repo_id}. Took {upload_time:.2f} seconds.")
except Exception as e:
    logger.error(f"Error pushing dataset to Hugging Face Hub: {e}", exc_info=True)
    sys.exit(1)

logger.info("Script execution complete. Your refined RAG dataset is now live on Hugging Face!")
