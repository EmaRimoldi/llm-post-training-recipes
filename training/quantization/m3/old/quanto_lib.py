from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.quanto import quantize, freeze, qint4 # Assuming you use these directly
import torch

# 1. Load your float model
# model_id = "../sft/qwen3_0.6B_sft_next_token_masked_loss/final_model/"
model_id= "Qwen/Qwen3-0.6B-Base"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16) # Or your preferred dtype
tokenizer = AutoTokenizer.from_pretrained(model_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 2. Quantize using optimum.quanto
quantize(model, weights=qint4, activations=qint4)

# 3. Freeze
freeze(model)

# 4. Save
save_dir = "./qwen_sft_quanto_config_int4"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

# 5. Reload
# This reloaded model should be quantized and use the optimum.quanto layers
reloaded_quantized_model = AutoModelForCausalLM.from_pretrained(save_dir, device_map="auto", trust_remote_code=True)

print(reloaded_quantized_model.dtype)


