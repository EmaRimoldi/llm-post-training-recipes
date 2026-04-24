from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
import os


NUM_CALIBRATION_SAMPLES=512
MAX_SEQUENCE_LENGTH=2048


save_directory_hf = "./train_quantized/qwen_mcqa_llm_compressor_a8w8"
os.makedirs(save_directory_hf, exist_ok=True)


MODEL_ID = "VinceEPFL/MNLP_M3_mcqa_model"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


print("Loading calibration dataset from Hugging Face Hub: ema1234/MNLP_M3_quantized_dataset")
dataset = load_dataset("ema1234/MNLP_M3_quantized_dataset")["train"]

def tokenize_function(examples):
    return tokenizer(examples["text"], padding=False, truncation=True, max_length=MAX_SEQUENCE_LENGTH)

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, remove_columns=dataset.column_names)


# Configure the quantization algorithms to run.
print("Configuring recipe...")
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]

# Apply quantization.
print("Applying one-shot quantization...")
oneshot(
    model=model,
    dataset=tokenized_dataset,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save to disk compressed.
print(f"Saving compressed model to: {save_directory_hf}")
model.save_pretrained(save_directory_hf, save_compressed=True)
tokenizer.save_pretrained(save_directory_hf)

print("✅ Python script finished successfully.")