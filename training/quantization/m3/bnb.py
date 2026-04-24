from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_id = "VinceEPFL/ft_on_mlandmedicine_model"
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Configure 4-bit quantization with BitsAndBytes
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     # bnb_4bit_use_double_quant=True,       # Optional: for higher precision
#     bnb_4bit_quant_type="fp4",            # nf4 (NormalFloat4) is a common choice
#     bnb_4bit_compute_dtype=torch.bfloat16 # Or torch.float16, for computation type
# )
# Load the model with the BitsAndBytes quantization config
# quantized_model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     quantization_config=bnb_config,
#     device_map="auto"
# )
# model_repo_id = "ema1234/qwen_mcqa_bnb_fp4" 
# quantized_model.push_to_hub(model_repo_id)
# tokenizer.push_to_hub(model_repo_id)


bnb_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True
)

model_8bit_explicit = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config_8bit,
    device_map="auto" # Helps distribute the model efficiently
)

model_repo_id = "ema1234/qwen_mcqa_bnb_llm_int8" 
model_8bit_explicit.push_to_hub(model_repo_id)
tokenizer.push_to_hub(model_repo_id)


# # Now you can use the quantized_model
# # For saving, models quantized with BitsAndBytes often have better
# # direct serialization support with save_pretrained, or specific instructions.
# # Usually, you can save it directly:
# model_8bit_explicit.save_pretrained('./qwen_sft_bnb_int8')
# tokenizer.save_pretrained('./qwen_sft_bnb_int8')

# print("Model quantized with BitsAndBytes and saved.")


