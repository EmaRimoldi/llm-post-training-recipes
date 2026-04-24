# ================================================================
# train_mcqa_cot.py
#
# Fine-tuning di un modello causale (es. Qwen 0.6B) a prevedere
# l’intera catena di ragionamento (CoT) su un dataset MC-QA.
# ================================================================
import os, logging, torch, getpass
from datasets import load_dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          DataCollatorForLanguageModeling,
                          TrainingArguments, Trainer)

# ----------------- CONFIG -------------------------------------------------
WANDB_PROJECT        = "mnlp_m3"                # export WANDB_PROJECT=…
WANDB_RUN_NAME       = "0.6B_mcqa_cot_run1"     # export WANDB_RUN_NAME=…
MODEL_NAME           = "MoroM02/MoroM02"        # base model
COT_DATASET          = "VinceEPFL/mcqa-reasoned" # <— sostituisci con il tuo repo
SAMPLE_SIZE          = -1        # -1 = usa tutto, 500 = prova veloce
OUTPUT_DIR           = "./qwen3_0.6B_mcqa_cot"
MAX_SEQ_LENGTH       = 2048
NUM_EPOCHS           = 3
PER_DEVICE_BATCH     = 1          # attento alla VRAM
GRAD_ACC_STEPS       = 8

# ----------------- LOGGING ------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "train.log"), mode="w"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger()

logger.info("CUDA available: %s", torch.cuda.is_available())
if torch.cuda.is_available():
    logger.info("GPU: %s", torch.cuda.get_device_name(0))

# ----------------- TOKENIZER & MODEL --------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True,
)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

# ----------------- PREPROCESSING FN --------------------------------------
DEFAULT_SYSTEM_PROMPT = ""

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
def preprocess_mcqa_cot(examples):
    """Tok. di question+choices (prompt) + CoT (assistant) con maschera NTP."""
    q_list   = examples["question"]
    ch_list  = examples["choices"]          # dict letter->text
    cot_list = examples["reasoning_answer"] # full CoT

    batch_inp, batch_att, batch_lbl = [], [], []


    for q, choices, cot in zip(q_list, ch_list, cot_list):
        if not (q and choices and cot):
            continue
    
        # Convert list → dict if needed
        if isinstance(choices, list):
            choices = {LETTERS[i]: c for i, c in enumerate(choices)}
    
        prompt = q.strip() + "\n\nOptions:\n"
        prompt += "\n".join(f"{k}. {v}" for k, v in choices.items())
        prompt += "\n\nPlease think step by step and give the answer."

        messages = [
            {"role": "system",    "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": cot + tokenizer.eos_token},
        ]
        full_txt = tokenizer.apply_chat_template(messages, tokenize=False)

        toks = tokenizer(full_txt,
                         max_length=MAX_SEQ_LENGTH,
                         truncation=True,
                         padding="max_length")

        # maschera: -100 su system+user e su pad
        ctx_txt = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True)
        ctx_len = len(tokenizer(ctx_txt, add_special_tokens=False)["input_ids"])

        labels = toks["input_ids"].copy()
        for idx in range(len(labels)):
            if idx < ctx_len or toks["input_ids"][idx] == tokenizer.pad_token_id:
                labels[idx] = -100

        batch_inp.append(toks["input_ids"])
        batch_att.append(toks["attention_mask"])
        batch_lbl.append(labels)

    return {"input_ids": batch_inp,
            "attention_mask": batch_att,
            "labels": batch_lbl}

# ----------------- DATASET LOAD ------------------------------------------
logger.info("Caricamento dataset %s …", COT_DATASET)
split_name = "train"  # il tuo repo dovrebbe avere solo train
ds = load_dataset(COT_DATASET, split=split_name, streaming=False)

if SAMPLE_SIZE != -1 and SAMPLE_SIZE < len(ds):
    ds = ds.shuffle(seed=42).select(range(SAMPLE_SIZE))
    logger.info("Esemplari campionati: %d", SAMPLE_SIZE)

# mapping
remove_cols = list(ds.column_names)
ds_tok = ds.map(preprocess_mcqa_cot,
                batched=True,
                remove_columns=remove_cols,
                num_proc=os.cpu_count() // 2 or 1)

# filtraggio esempi vuoti
def keep_ok(e): return any(l != -100 for l in e["labels"])
ds_tok = ds_tok.filter(keep_ok, num_proc=os.cpu_count() // 2 or 1)

logger.info("Totale esempi utilizzati: %d", len(ds_tok))

# train / eval split (5 %)
if len(ds_tok) > 100:
    data = ds_tok.train_test_split(test_size=0.05, seed=42)
    train_ds, eval_ds = data["train"], data["test"]
    logger.info("Train: %d  |  Eval: %d", len(train_ds), len(eval_ds))
else:
    train_ds, eval_ds = ds_tok, None
    logger.warning("Dataset piccolo: niente valutazione in corso d’addestramento.")

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# ----------------- TRAINING ----------------------------------------------
train_args = TrainingArguments(
    output_dir            = OUTPUT_DIR,
    overwrite_output_dir  = True,
    num_train_epochs      = NUM_EPOCHS,
    per_device_train_batch_size = PER_DEVICE_BATCH,
    per_device_eval_batch_size  = PER_DEVICE_BATCH,
    gradient_accumulation_steps = GRAD_ACC_STEPS,
    learning_rate         = 2e-5,
    weight_decay          = 0.01,
    warmup_ratio          = 0.03,
    lr_scheduler_type     = "cosine",
    bf16                  = torch.cuda.is_available()
                              and torch.cuda.get_device_capability(0)[0] >= 8,
    logging_strategy      = "steps",
    logging_steps         = 50,
    save_strategy         = "steps",
    save_steps            = 200,
    save_total_limit      = 1,
    report_to             = "wandb",
    run_name              = WANDB_RUN_NAME,
    remove_unused_columns = True,
    logging_dir           = os.path.join(OUTPUT_DIR, "logs"),
)

trainer = Trainer(
    model           = model,
    args            = train_args,
    train_dataset   = train_ds,
    eval_dataset    = eval_ds,
    tokenizer       = tokenizer,
    data_collator   = data_collator,
)

logger.info("----- INIZIO TRAINING -----")
trainer.train()
logger.info("----- TRAINING TERMINATO -----")

# ----------------- SALVATAGGIO -------------------------------------------
final_dir = os.path.join(OUTPUT_DIR, "final_model")
trainer.save_model(final_dir)
tokenizer.save_pretrained(final_dir)
logger.info("Modello salvato in %s", final_dir)

# ----------------- EVALUATION (facoltative) ------------------------------
if eval_ds is not None:
    metrics = trainer.evaluate()
    logger.info("Eval metrics: %s", metrics)
