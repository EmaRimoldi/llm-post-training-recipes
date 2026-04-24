from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import torch.nn.functional as F
from tqdm import tqdm

# === CONFIG ===
model_name = "MoroM02/MNLP_M3_dpo_model"
ref_model_name = "Qwen/Qwen3-0.6B-Base"
dataset_name = "zechen-nlp/MNLP_dpo_evals"
split = "test"

# === DEVICE ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD MODELS ===
print(f"Loading DPO model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
model.eval()

print("Loading reference model...")
ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name, trust_remote_code=True).to(device)
ref_model.eval()

# === LOAD TOKENIZER (DPO tokenizer used for both models) ===
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# === LOAD DATASET ===
print(f"Loading benchmark dataset: {dataset_name}")
dataset = load_dataset(dataset_name, split=split)

# === RELATIVE LOGPROB FUNCTION ===
def compute_relative_logprob(prompt, completion):
    full_input = prompt + completion
    inputs = tokenizer(full_input, return_tensors="pt", truncation=True, padding=True).to(device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        # DPO model
        logits = model(**inputs).logits[:, :-1, :]
        labels = input_ids[:, 1:]
        model_log_probs = F.log_softmax(logits, dim=-1)
        model_token_log_probs = model_log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

        # Reference model
        ref_logits = ref_model(**inputs).logits[:, :-1, :]
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        ref_token_log_probs = ref_log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

        # Identify completion portion
        prompt_len = len(tokenizer(prompt)["input_ids"]) - 1  # -1 to exclude EOS if added

        # Compute logprob of the completion only
        model_score = model_token_log_probs[:, prompt_len:].sum(dim=1)
        ref_score = ref_token_log_probs[:, prompt_len:].sum(dim=1)

    return (model_score - ref_score).item()

# === EVALUATE REWARD ACCURACY ===
correct, total = 0, 0

for example in tqdm(dataset, desc="Evaluating reward accuracy with reference model"):
    if "prompt" not in example or "chosen" not in example or "rejected" not in example:
        continue

    prompt = example["prompt"]
    chosen = example["chosen"]
    rejected = example["rejected"]

    try:
        c_score = compute_relative_logprob(prompt, chosen)
        r_score = compute_relative_logprob(prompt, rejected)
    except Exception as e:
        print("Skipped due to error:", e)
        continue

    if c_score > r_score:
        correct += 1
    total += 1

accuracy = correct / total if total > 0 else 0.0
print(f"\n Reward accuracy (with reference model): {correct}/{total} = {accuracy:.4f}")
