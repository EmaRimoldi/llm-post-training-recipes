import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import logging
import sys
import numpy as np

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration Variables ---
DATASET_NAME = "EmaRimoldi/mnlp_arxiv_100k"
MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
MAX_TOKENS = 512 # Maximum allowed token length for a chunk

# Determine the device (primarily for informational purposes, tokenization is CPU-bound)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE} for potential future steps (tokenization itself is CPU-bound).")

# ─── 1. Load the Dataset ────────────────────────────────────────────────
logger.info(f"Loading dataset: {DATASET_NAME} from Hugging Face Hub...")
try:
    # Assuming 'train' split, adjust if your dataset uses a different split (e.g., 'text', 'documents')
    dataset = load_dataset(DATASET_NAME, split='train')
    logger.info(f"Dataset loaded successfully. Number of entries: {len(dataset)}")
    if 'text' not in dataset.column_names:
        logger.error("Error: 'text' column not found in the dataset. Please check the dataset structure.")
        sys.exit(1)
except Exception as e:
    logger.error(f"Error loading dataset '{DATASET_NAME}': {e}", exc_info=True)
    sys.exit(1)

# ─── 2. Load the Tokenizer ──────────────────────────────────────────────
logger.info(f"Loading tokenizer: {MODEL_NAME}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    logger.info("Tokenizer loaded successfully.")
except Exception as e:
    logger.error(f"Error loading tokenizer '{MODEL_NAME}': {e}", exc_info=True)
    sys.exit(1)

# ─── 3. Tokenize and Collect Lengths ────────────────────────────────────
logger.info(f"Tokenizing all chunks and collecting their lengths...")
token_lengths = []
chunks_exceeding_max_length = 0
total_chunks = len(dataset)

for i, entry in tqdm(enumerate(dataset), total=total_chunks, desc="Processing chunks"):
    text = entry['text']
    if text is None:
        logger.warning(f"Skipping empty or None text entry at index {i}.")
        continue

    tokens = tokenizer.encode(text) # Get list of token IDs
    token_count = len(tokens)
    token_lengths.append(token_count)

    if token_count > MAX_TOKENS:
        chunks_exceeding_max_length += 1
        # Optionally log individual chunks that exceed the max length
        # logger.warning(f"Chunk at index {i} exceeds {MAX_TOKENS} tokens. Actual length: {token_count}")

logger.info("Finished tokenizing all chunks.")

# ─── 4. Display Statistics ──────────────────────────────────────────────
logger.info("\n" + "=" * 50)
logger.info("Chunk Length Statistics (in tokens)")
logger.info("=" * 50)

if token_lengths:
    min_length = np.min(token_lengths)
    max_length = np.max(token_lengths)
    avg_length = np.mean(token_lengths)
    median_length = np.median(token_lengths)
    std_dev_length = np.std(token_lengths)
    
    # Calculate quantiles for more detailed distribution
    q1 = np.percentile(token_lengths, 25)
    q3 = np.percentile(token_lengths, 75)

    logger.info(f"  Total Chunks Processed: {total_chunks}")
    logger.info(f"  Chunks Exceeding {MAX_TOKENS} tokens: {chunks_exceeding_max_length}")
    logger.info(f"  Minimum Length: {min_length:.0f} tokens")
    logger.info(f"  Maximum Length: {max_length:.0f} tokens")
    logger.info(f"  Average Length: {avg_length:.2f} tokens")
    logger.info(f"  Median Length (50th percentile): {median_length:.0f} tokens")
    logger.info(f"  25th Percentile (Q1): {q1:.0f} tokens")
    logger.info(f"  75th Percentile (Q3): {q3:.0f} tokens")
    logger.info(f"  Standard Deviation: {std_dev_length:.2f} tokens")
else:
    logger.warning("No valid chunks were processed to calculate statistics.")

logger.info("=" * 50)
logger.info("Script execution complete.")