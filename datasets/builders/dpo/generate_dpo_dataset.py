import os
import json
from tqdm import tqdm

# === Import and configure GPT wrapper ===
import gpt_wrapper
gpt_wrapper.api_base = "http://mnlp-backend-lb-1062233132.eu-central-1.elb.amazonaws.com"
gpt_wrapper.api_key = os.getenv("MNLP_GPT_WRAPPER_API_KEY")

if not gpt_wrapper.api_key:
    raise RuntimeError("Set MNLP_GPT_WRAPPER_API_KEY before running this script.")

from gpt_wrapper.chat import Chat

# === Prompt and Instruction ===
INSTRUCTION_OPEN_ENDED = """You are a knowledgeable and precise STEM professional with expertise in science, 
technology, engineering, and mathematics. Provide clear, accurate, and well-reasoned
explanations for  open-ended questions. Use appropriate terminology
and concise logic, and when helpful, include formulas, diagrams, or real-world examples. Always
aim to educate and inform, while maintaining a professional and approachable tone."""

GPT_SUGAR = """Let's think step by step. """

# === Configuration ===
INPUT_FOLDER = "../DataNLP"  # Your folder with raw ["question", "answer"] .jsonl files
OUTPUT_FILE = "dpo_dataset_combined_def.jsonl"

# === GPT wrapper call to generate a rejected answer ===
def generate_rejected_answer(prompt, chosen, question_id):
    chat = Chat.create(name=str(question_id))
    message = chat.ask(
        GPT_SUGAR + f"Provide a slightly worse or less accurate answer to the question: {prompt}\n"
                            f"Compared to this good answer: {chosen}. It has to be similar enough to be still considered an acceptable answer, not necessarily true, but acceptable.",
        instruction=INSTRUCTION_OPEN_ENDED
    )
    return message.content.strip()

# === Main DPO generation function ===
def build_dpo_from_datanlp(input_folder, output_file):
    total_written = 0
    question_id = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        for fname in os.listdir(input_folder):
            if not fname.endswith(".jsonl"):
                continue

            fpath = os.path.join(input_folder, fname)
            print(f"\n Processing: {fname}")

            with open(fpath, "r", encoding="utf-8") as in_f:
                for i, line in enumerate(tqdm(in_f, desc=fname)):
                    if i >= 750:
                        break
                    try:
                        prompt, chosen = json.loads(line)

                        # Skip empty entries
                        if not prompt.strip() or not chosen.strip():
                            continue

                        rejected = generate_rejected_answer(prompt, chosen, question_id)

                        dpo_entry = {
                            "prompt": prompt.strip(),
                            "chosen": chosen.strip(),
                            "rejected": rejected.strip()
                        }

                        json.dump(dpo_entry, out_f, ensure_ascii=False)
                        out_f.write("\n")
                        total_written += 1
                        question_id += 1

                    except Exception as e:
                        print(f" Error at line {question_id} in file {fname}: {e}")
                        question_id += 1
                        continue

    print(f"\n Done! Total DPO triples written: {total_written} to {output_file}")

# === Run script ===
if __name__ == "__main__":
    build_dpo_from_datanlp(INPUT_FOLDER, OUTPUT_FILE)
