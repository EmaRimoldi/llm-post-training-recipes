import os
from huggingface_hub import login
from datasets import load_dataset
import logging
import time

# --- Configurazione del Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Variabili di Configurazione ---
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Set HF_TOKEN before running this script.")

LOCAL_FILE_PATH = "raft_dataset_generated_with_gpt_wrapper.jsonl"
TARGET_REPO_ID = os.getenv("HF_DATASET_REPO_ID", "MNLP_M3_rag_dataset")

# --- Login a Hugging Face Hub ---
def hf_login(token):
    try:
        login(token=token)
        logger.info("Successfully logged in to Hugging Face Hub.")
    except Exception as e:
        logger.error(f"Failed to log in to Hugging Face Hub: {e}")
        exit(1) # Esci se il login fallisce, non puoi pushare.

# --- Esegui il Login ---
hf_login(HF_TOKEN)

# --- Verifica l'esistenza del file locale ---
if not os.path.exists(LOCAL_FILE_PATH):
    logger.error(f"Error: Local file '{LOCAL_FILE_PATH}' not found. Please ensure the file is in the same directory as the script, or provide its full path.")
    exit(1)

# --- Carica il file JSONL locale come Dataset ---
logger.info(f"Loading local JSONL file: {LOCAL_FILE_PATH} as a Hugging Face dataset...")
start_load_time = time.time()
try:
    # load_dataset può caricare direttamente da un file locale specificandone il tipo
    # 'json' è il builder per i file JSON, 'jsonl' viene gestito automaticamente se l'estensione è .jsonl
    dataset = load_dataset("json", data_files=LOCAL_FILE_PATH)
    load_time = time.time() - start_load_time
    logger.info(f"✔ Dataset loaded successfully from local file. Took {load_time:.2f} seconds.")
    logger.info(f"Dataset structure: {dataset}") # Stampa la struttura del dataset (dovrebbe essere un DatasetDict con uno split 'train')
    
    # Se il tuo file JSONL non ha split espliciti e load_dataset lo carica come
    # DatasetDict con un solo split 'train', potresti voler lavorare direttamente con quello split.
    # Ad esempio, dataset = dataset['train'] se sai che è sempre lo split 'train'.
    # Per il push, DatasetDict funziona bene in quanto pubblicherà lo split 'train'.
    
except Exception as e:
    logger.error(f"Error loading local JSONL file '{LOCAL_FILE_PATH}': {e}", exc_info=True)
    exit(1)

# --- Push del Dataset alla Repo di Destinazione ---
logger.info(f"Pushing dataset to target repository: {TARGET_REPO_ID}...")
start_push_time = time.time()
try:
    dataset.push_to_hub(
        repo_id=TARGET_REPO_ID,
        token=HF_TOKEN,
        commit_message=f"Upload of {LOCAL_FILE_PATH} for RAG documents"
    )
    
    push_time = time.time() - start_push_time
    logger.info(f"✔ Dataset successfully pushed to Hugging Face repo: {TARGET_REPO_ID}. Took {push_time:.2f} seconds.")

except Exception as e:
    logger.error(f"Error pushing dataset to Hugging Face Hub: {e}", exc_info=True)
    exit(1)

logger.info("Script execution complete. Local JSONL file successfully uploaded as a Hugging Face dataset.")
