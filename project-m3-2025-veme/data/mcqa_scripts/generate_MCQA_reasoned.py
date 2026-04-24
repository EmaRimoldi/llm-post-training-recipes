import os, json, getpass
from itertools import islice
from tqdm import tqdm
from datasets import load_dataset, Dataset, Features, Value, ClassLabel
from huggingface_hub import HfApi
import gpt_wrapper
from gpt_wrapper.chat import Chat

gpt_wrapper.api_base = "http://mnlp-backend-lb-1062233132.eu-central-1.elb.amazonaws.com"
gpt_wrapper.api_key = os.getenv("MNLP_GPT_WRAPPER_API_KEY")

if not gpt_wrapper.api_key:
    raise RuntimeError("Set MNLP_GPT_WRAPPER_API_KEY before running this script.")

INSTRUCTION = """
You are a knowledgeable and precise STEM professional. For every multiple-choice
question, evaluate each option (A, B, C, …) in 1-2 sentences, you will be provided the right answer. Reason your way to it, then finish with
'Answer: <letter>'. Keep the tone analytical but concise.
"""

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
OUTPUT_JSONL = "mcqa_reasoned.jsonl"
PUSH_REPO = "VinceEPFL/mcqa-reasoned"
PUSH = True

# === 1. DATASET BLUEPRINTS ===
BLUEPRINTS = {
    "VinceEPFL/medmcqa-standard": {
        "splits": ["train", "validation", "test"],
        "fields": dict(q="question", choices="choices", ans="answer", ans_type="letter"),
    },
    "VinceEPFL/aimcqs-mcqa": {
        "splits": ["test"],
        "fields": dict(q="question", choices="choices", ans="answer", ans_type="letter"),
    },
    "ema1234/MNLP_M2_quantized_dataset": {
        "splits": ["train"],
        "fields": dict(q="question", choices="options", ans="answer_idx", ans_type="index"),
    },
}

# === 2. SAMPLE COUNTS PER DATASET/SPLIT ===
SAMPLE_COUNTS = {
    ("ema1234/MNLP_M2_quantized_dataset", "train"): 4500,
    ("VinceEPFL/medmcqa-standard", "train"): 1000,
    ("VinceEPFL/aimcqs-mcqa", "test"): 400,
}

# === 3. HELPERS ===
def to_choice_dict(choices_list):
    return {LETTERS[i]: c for i, c in enumerate(choices_list)}

def get_correct_letter(row, fld, ans_type):
    return row[fld] if ans_type == "letter" else LETTERS[int(row[fld])]

def generate_reasoning(q, choices_dict, correct, uid):
    chat = Chat.create(name=f"mcqa_{uid}")
    prompt = (
        f"{q.strip()}\n\nOptions:\n" +
        "\n".join(f"{k}. {v}" for k, v in choices_dict.items()) +
        f"\n\nThe correct answer is {correct}. "
        f"Think step by step, then end with 'Answer: {correct}'."
    )
    return chat.ask(prompt, instruction=INSTRUCTION).content.strip()

def already_written_ids(jsonl_path):
    if not os.path.exists(jsonl_path):
        return set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return {json.loads(line)["entry_id"] for line in f}

def append_jsonl(path, entry):
    with open(path, "a", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False)
        f.write("\n")

# === 4. MAIN ===
def build_all_and_write():
    uid = 0
    written_ids = already_written_ids(OUTPUT_JSONL)

    for ds_name, cfg in BLUEPRINTS.items():
        for split in cfg["splits"]:
            sample_limit = SAMPLE_COUNTS.get((ds_name, split), 0)
            if sample_limit <= 0:
                print(f"⚠️ Skipping {ds_name}/{split} — no sample count given")
                continue

            print(f"📥 Loading {ds_name} [{split}] — {sample_limit} samples")
            src = load_dataset(ds_name, split=split, streaming=True)
            for raw in tqdm(islice(src, sample_limit), total=sample_limit, desc=f"{ds_name}/{split}"):
                q = raw[cfg["fields"]["q"]]
                ch_list = raw[cfg["fields"]["choices"]]
                ch_dict = to_choice_dict(ch_list)
                cor = get_correct_letter(raw, cfg["fields"]["ans"], cfg["fields"]["ans_type"])

                entry_id = f"{ds_name}::{split}::{uid}"
                if entry_id in written_ids:
                    uid += 1
                    continue

                reasoning = generate_reasoning(q, ch_dict, cor, uid)

                entry = {
                    "id": uid,
                    "entry_id": entry_id,
                    "source_dataset": ds_name,
                    "source_split": split,
                    "question": q,
                    "choices": ch_dict,
                    "correct": cor,
                    "reasoning_answer": reasoning,
                }

                append_jsonl(OUTPUT_JSONL, entry)
                written_ids.add(entry_id)
                uid += 1

    print(f"\n✅ DONE. Total new entries written: {len(written_ids)}")

# === 5. LOAD FROM JSONL + PUSH ===
def jsonl_to_hf_dataset(jsonl_path):
    print("📦 Converting JSONL to 🤗 Dataset …")
    ds = Dataset.from_json(jsonl_path)
    ds = ds.cast(Features({
        "id":               Value("int64"),
        "entry_id":         Value("string"),
        "source_dataset":   Value("string"),
        "source_split":     Value("string"),
        "question":         Value("string"),
        "choices":          {ltr: Value("string") for ltr in LETTERS[:4]},
        "correct":          ClassLabel(names=list(LETTERS[:4])),
        "reasoning_answer": Value("string"),
    }))
    return ds

def push_to_hub(ds, repo):
    print(f"🚀 Pushing to {repo}")
    api = HfApi()
    try:
        api.whoami()
    except:
        token = os.getenv("HF_TOKEN") or getpass.getpass("HF token: ")
        api.set_access_token(token)

    ds.push_to_hub(repo, private=True)
    print("✓ Pushed to Hugging Face Hub")

# === 6. ENTRY POINT ===
if __name__ == "__main__":
    build_all_and_write()

    if PUSH:
        ds = jsonl_to_hf_dataset(OUTPUT_JSONL)
        push_to_hub(ds, PUSH_REPO)
