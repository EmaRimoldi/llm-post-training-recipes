#!/usr/bin/env python
"""
Fine-tuning leggero di Qwen/Qwen3-0.6B su CPU
– Usa solo 100 esempi del dataset VinceEPFL/MNLP_M2_mcqa_dataset
– Aggiorna esclusivamente l’ultimo blocco e la final_layernorm
"""

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
import torch
import os

# 1. Carica un micro-campione del dataset (100 righe)
DATASET = "VinceEPFL/MNLP_M2_mcqa_dataset"
ds = load_dataset(DATASET, split="train[:100]")

# 2. Tokenizer e modello
MODEL_NAME = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)

# 3. Congela tutti i parametri tranne l’ultimo blocco + final_layernorm
last_layer_idx = model.config.num_hidden_layers - 1  # 31 per Qwen3-0.6B
trainable_substrings = [f"model.layers.{last_layer_idx}", "final_layernorm"]

for name, param in model.named_parameters():
    param.requires_grad = any(substr in name for substr in trainable_substrings)

print(
    f"Parametri totali: {sum(p.numel() for p in model.parameters()):,}\n"
    f"Parametri addestrabili: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)

# 4. Pre-processing del dataset
def preprocess(example):
    answer_text = example["options"][example["answer_idx"]].strip()
    options_str = " | ".join(
        [f"{i}) {opt.strip()}" for i, opt in enumerate(example["options"])]
    )
    prompt = (
        f"Domanda: {example['question'].strip()} "
        f"Opzioni: {options_str} "
        f"Risposta: {answer_text}"
    )
    tokens = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=256,
    )
    tokens["labels"] = tokens["input_ids"]
    return tokens


ds_tok = ds.map(preprocess, batched=False)

# 5. Argomenti di training mini-mal (CPU)
training_args = TrainingArguments(
    output_dir="qwen_cpu_ft",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=20,
    save_total_limit=1,
    report_to="none",
    fp16=False,                  # CPU
    bf16=False,
    optim="adamw_torch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_tok,
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model("qwen_cpu_ft/final")
    tokenizer.save_pretrained("qwen_cpu_ft/final")
    print("Fine-tuning completato e modello salvato in qwen_cpu_ft/final")
