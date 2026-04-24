
# import torch
# import os
# from tqdm import tqdm
# from compressed_tensors.quantization import (
#     QuantizationConfig,
#     QuantizationStatus,
#     apply_quantization_config,
#     compress_quantized_weights
# )
# from compressed_tensors.compressors import ModelCompressor
# from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator
# from datasets import load_dataset
# from torch.utils.data import RandomSampler
# from torch.utils.data import DataLoader
# from datasets import load_dataset

# # 1. Define the model ID and device
# model_id = "VinceEPFL/mcqa_matmlmed"
    

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# os.chdir('/users/nevali/projects/mnlp/')
# save_directory_hf = "./quantization/qwen_mcqa_compressed_gptq_4bit"
# os.makedirs(save_directory_hf, exist_ok=True)

# # 2. Load the pre-trained model and tokenizer
# model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
# model.to(device)

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# tokenizer.save_pretrained(save_directory_hf)

# quantization_config_dict = {
# 	"quant_method": "gptq",
# 	"format": "pack-quantized",
# 	"global_compression_ratio": None,
# 	"config_groups": {
#         "group_1": {
#             "weights": {
#                 "num_bits": 4,
#                 "type": "int",
#                 "symmetric": False,
#                 "strategy": "tensor"
#             },
#             "targets": ["Linear"]
#         }
#     },
# 	"ignore": ["lm_head"]
# }

# config = QuantizationConfig(**quantization_config_dict)
# config.quantization_status = QuantizationStatus.CALIBRATION
# apply_quantization_config(model, config)

# dataset = load_dataset("json", data_files={"train": "./data/calibration/gpqa_only_calibration_data.jsonl"})["train"]


# tokenizer = AutoTokenizer.from_pretrained(model_id)

# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding=False, truncation=True, max_length=1024)

# tokenized_dataset = dataset.map(tokenize_function, batched=True)

# data_loader = DataLoader(
#     tokenized_dataset, batch_size=1, collate_fn=DefaultDataCollator(), sampler=RandomSampler(tokenized_dataset)
# )


# # calibrate scale and zero points for quantization using a small amount of train data
# num_calibration_samples = 512

# with torch.no_grad():
#     for idx, sample in tqdm(enumerate(data_loader), desc="Running calibration"):
#         sample = {key: value.to(device) for key,value in sample.items()}
#         _ = model(**sample)

#         if idx >= num_calibration_samples:
#             break


# compression_format = config.format
# print(f"Compression format: {compression_format}")

# compressor = ModelCompressor(quantization_config=config)
# compressed_state_dict = compressor.compress(model)
# model.save_pretrained(save_directory_hf, state_dict=compressed_state_dict)
# compressor.update_config(save_directory_hf)

# compressed_size_on_disk_mb = os.path.getsize(os.path.join(save_directory_hf, "model.safetensors")) / 1024 / 1024
# print(f"Size of the model's weights on disk using safetensors: {compressed_size_on_disk_mb:.2f} MB")


##########################################################

import torch
import os
from pathlib import Path
from tqdm import tqdm
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationStatus,
    apply_quantization_config,
    compress_quantized_weights
)
from compressed_tensors.compressors import ModelCompressor
from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator
from datasets import load_dataset
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader

# 1. Define the model ID and device
model_id = "VinceEPFL/mcqa_matmlmed"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

save_directory_hf = SCRIPT_DIR / "qwen_mcqa_compressed_smoothquant_int8"
os.makedirs(save_directory_hf, exist_ok=True)

# 2. Load the pre-trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(str(save_directory_hf))

# SmoothQuant configuration - INT8 for both weights and activations
quantization_config_dict = {
    "quant_method": "smoothquant",
    "format": "pack-quantized", 
    "global_compression_ratio": None,
    "config_groups": {
        "group_1": {
            "weights": {
                "num_bits": 8,
                "type": "int",
                "symmetric": True,
                "strategy": "group",        # Changed from "channel" to "group"
                "group_size": 128
            },
            "input_activations": {
                "num_bits": 8,
                "type": "int",
                "symmetric": True,
                "strategy": "token",
                "observer": "minmax"
            },
            "targets": ["Linear"]           # Moved targets to same level as weights/activations
        }
    },
    "ignore": ["lm_head"],
    "smoothquant_compat": True,
    "alpha": 0.5,
    "folding": True
}


config = QuantizationConfig(**quantization_config_dict)
config.quantization_status = QuantizationStatus.CALIBRATION
apply_quantization_config(model, config)

# Load calibration dataset
dataset = load_dataset(
    "json",
    data_files={"train": str(PROJECT_ROOT / "datasets" / "calibration" / "gpqa_only_calibration_data.jsonl")},
)["train"]

def tokenize_function(examples):
    return tokenizer(examples["text"], padding=False, truncation=True, max_length=1024)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

data_loader = DataLoader(
    tokenized_dataset, batch_size=1, collate_fn=DefaultDataCollator(), sampler=RandomSampler(tokenized_dataset)
)

# Calibrate for SmoothQuant - needs more samples for activation statistics
num_calibration_samples = 1024  # More samples for activation calibration

print("Running SmoothQuant calibration...")
with torch.no_grad():
    for idx, sample in tqdm(enumerate(data_loader), desc="SmoothQuant calibration"):
        sample = {key: value.to(device) for key, value in sample.items()}
        _ = model(**sample)
        
        if idx >= num_calibration_samples:
            break

# Apply SmoothQuant transformations
print("Applying SmoothQuant transformations...")
config.quantization_status = QuantizationStatus.FROZEN

# Compress the model
compression_format = config.format
print(f"Compression format: {compression_format}")

compressor = ModelCompressor(quantization_config=config)
compressed_state_dict = compressor.compress(model)
model.save_pretrained(str(save_directory_hf), state_dict=compressed_state_dict)
compressor.update_config(str(save_directory_hf))

compressed_size_on_disk_mb = os.path.getsize(save_directory_hf / "model.safetensors") / 1024 / 1024
print(f"Size of the model's weights on disk using safetensors: {compressed_size_on_disk_mb:.2f} MB")
