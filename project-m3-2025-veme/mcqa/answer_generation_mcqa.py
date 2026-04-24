from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
set_seed(2)
import os
import random
import numpy as np
import torch
from transformers import set_seed as hf_set_seed

def set_all_seeds(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)         # Python built-in hash
    random.seed(seed)                                # Python
    np.random.seed(seed)                             # NumPy
    torch.manual_seed(seed)                          # PyTorch CPU
    torch.cuda.manual_seed_all(seed)                 # PyTorch GPU
    hf_set_seed(seed)                                # Hugging Face internal (e.g., Trainer)

    # Optional but helpful for determinism:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


import torch
set_all_seeds(2)

# Load your model from Hugging Face
#Qwen/Qwen3-0.6B
#VinceEPFL/mcqa_matmlmed_long
#VinceEPFL/mcqa_cot
model_id = "VinceEPFL/mcqa_cot"  # change to your model name

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda" if torch.cuda.is_available() else "cpu")

# Set up generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Your question

prompt = (
    "The following are multiple choice questions (with answers) about knowledge and skills in advanced master-level STEM courses.\n\nWhat is the domain of the function $f(x)=\frac{x+6}{\sqrt{x^2-3x-4}}$?\nA. (-4,4)\nB. [0,4]\nC. (-inf, -1) U (4, inf)\nD. (-inf, -1) U (-1, 4) U (4, inf)\nAnswer:"
    
)

# Generate answer
output = generator(prompt, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.90)[0]["generated_text"]

# Print
print("\n🧠 Model's answer:")
print(output)
