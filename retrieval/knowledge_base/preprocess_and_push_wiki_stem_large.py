import os
import json
import gc
import torch
import numpy as np
import pandas as pd
from datasets import Dataset, load_from_disk
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm.auto import tqdm
from huggingface_hub import HfApi
import logging
import sys
import time


def require_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Set the {name} environment variable before running this script.")
    return value

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration Variables ---
# KAGGLE_DATASET is no longer needed as we're loading locally
DATA_DIR = "." # Current directory for the CSV file
CSV_FILE_NAME = "wiki_stem_corpus.csv" # Name of your local CSV file
CHUNK_SIZE = 512
CHUNK_OVERLAP_PERCENTAGE = 0.25 # 25% overlap
HF_TOKEN = require_env_var("HF_TOKEN")
USERNAME = os.getenv("HF_USERNAME", "EmaRimoldi")
REPO_NAME = os.getenv("HF_DATASET_REPO_NAME", "wiki-stem-large")

# Determine the device for potential future embedding steps (not strictly needed for chunking)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

# ─── 1. Load & Prepare the STEM Wiki Corpus (from local CSV) ───────────
def load_and_prepare_corpus(csv_file_name: str, data_dir: str = "."):
    """
    Loads the CSV from the specified directory, builds a HuggingFace Dataset
    with only 'text' and 'source' columns, and saves it to disk.
    """
    csv_path = os.path.join(data_dir, csv_file_name)
    
    if not os.path.exists(csv_path):
        logger.error(f"Error: CSV file not found at '{csv_path}'. Please ensure '{csv_file_name}' is in the same directory as the script.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} documents from {csv_path}")

    def make_source(row):
        page = row["page_title"].replace(" ", "_")
        section = str(row.get("section_title", "")).strip()
        if not section:
            return f"https://en.wikipedia.org/wiki/{page}"
        return f"https://en.wikipedia.org/wiki/{page}#{section.replace(' ', '_')}"

    df["source"] = df.apply(make_source, axis=1)
    df = df[["text", "source"]]

    ds = Dataset.from_pandas(df, preserve_index=False)
    corpus_save_path = "wiki_stem_with_source"
    ds.save_to_disk(corpus_save_path)
    logger.info(f"Corpus prepared and saved to disk at '{corpus_save_path}'.")
    return corpus_save_path

# ─── 2. Load & Chunk Documents ───────────────────────────────────────────
def load_and_chunk(corpus_path: str = "wiki_stem_with_source",
                   chunk_size: int = 800, chunk_overlap_percentage: float = 0.1):
    """
    Loads the HF Dataset from disk and splits each document into overlapping chunks.
    Returns two lists: all_chunks and their corresponding sources.
    """
    ds = load_from_disk(corpus_path)
    texts, sources = ds["text"], ds["source"]

    # Calculate the overlap size based on the percentage
    chunk_overlap = int(chunk_size * chunk_overlap_percentage)
    logger.info(f"Chunking documents with chunk_size={chunk_size} and chunk_overlap={chunk_overlap}")

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", ". ", "? ", "! ", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    all_chunks, all_chunk_sources = [], []
    for doc, src in tqdm(zip(texts, sources), total=len(texts), desc="Chunking documents"):
        if doc is None: # Skip null or empty documents
            continue
        chunks = splitter.split_text(doc)
        all_chunks.extend(chunks)
        all_chunk_sources.extend([src] * len(chunks))
    
    logger.info(f"Finished chunking. Total chunks created: {len(all_chunks)}")
    return all_chunks, all_chunk_sources

# ─── Main Execution ────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Load and prepare the corpus from the local CSV
    corpus_path = load_and_prepare_corpus(CSV_FILE_NAME, DATA_DIR)

    # 2. Load and chunk the documents
    chunks, chunk_sources = load_and_chunk(corpus_path, CHUNK_SIZE, CHUNK_OVERLAP_PERCENTAGE)

    # Create a new Hugging Face dataset from the chunks
    chunked_data = {"text": chunks, "source": chunk_sources}
    final_dataset = Dataset.from_dict(chunked_data)
    logger.info(f"Created a Hugging Face Dataset with {len(final_dataset)} chunks.")

    # 3. Push the dataset to Hugging Face Hub
    api = HfApi()
    full_repo_id = f"{USERNAME}/{REPO_NAME}"

    logger.info(f"Attempting to create/check Hugging Face repository: {full_repo_id} (type: dataset, private: False)")
    try:
        api.create_repo(repo_id=full_repo_id, private=False, repo_type="dataset", token=HF_TOKEN)
        logger.info(f"Repository '{full_repo_id}' created successfully!")
    except Exception as e:
        if "You already have a dataset named" in str(e) or "RepositoryAlreadyExists" in str(e):
            logger.warning(f"Repository '{full_repo_id}' already exists. Proceeding with the upload.")
        else:
            logger.error(f"Error creating repository '{full_repo_id}': {e}", exc_info=True)
            sys.exit(1)

    logger.info(f"Starting upload of the final dataset to Hugging Face repo: {full_repo_id}...")
    start_upload_time = time.time()
    try:
        final_dataset.push_to_hub(
            repo_id=full_repo_id,
            token=HF_TOKEN,
            commit_message="Upload custom RAG dataset (chunked with specified size and overlap)"
        )
        upload_time = time.time() - start_upload_time
        logger.info(f"✔ Dataset uploaded successfully to Hugging Face repo: {full_repo_id}. Took {upload_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"Error pushing dataset to Hugging Face Hub: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Script execution complete. Your refined RAG dataset is now live on Hugging Face Hub.")
