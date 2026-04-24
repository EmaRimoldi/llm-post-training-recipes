
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
from datasets import load_dataset

# 1. Define the model ID and device
model_id = "Qwen/Qwen3-0.6B-Base"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_directory_hf = PROJECT_ROOT / "quantization" / "qwen_compressed_8bit"
os.makedirs(save_directory_hf, exist_ok=True)

# 2. Load the pre-trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(str(save_directory_hf))

quantization_config_dict = {
	"quant_method": "sparseml",
	"format": "pack-quantized",
	"global_compression_ratio": None,
	"config_groups": {
        "group_1": {
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": False,
                "strategy": "tensor"
            },
            "targets": ["Linear"]
        }
    },
	"ignore": ["lm_head"]
}

config = QuantizationConfig(**quantization_config_dict)
config.quantization_status = QuantizationStatus.CALIBRATION
apply_quantization_config(model, config)

dataset = load_dataset(
    "json",
    data_files={"train": str(PROJECT_ROOT / "data" / "calibration" / "gpqa_only_calibration_data.jsonl")},
)["train"]


tokenizer = AutoTokenizer.from_pretrained(model_id)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding=False, truncation=True, max_length=1024)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

data_loader = DataLoader(
    tokenized_dataset, batch_size=1, collate_fn=DefaultDataCollator(), sampler=RandomSampler(tokenized_dataset)
)


# calibrate scale and zero points for quantization using a small amount of train data
num_calibration_samples = 512

with torch.no_grad():
    for idx, sample in tqdm(enumerate(data_loader), desc="Running calibration"):
        sample = {key: value.to(device) for key,value in sample.items()}
        _ = model(**sample)

        if idx >= num_calibration_samples:
            break


compression_format = config.format
print(f"Compression format: {compression_format}")

compressor = ModelCompressor(quantization_config=config)
compressed_state_dict = compressor.compress(model)
model.save_pretrained(str(save_directory_hf), state_dict=compressed_state_dict)
compressor.update_config(str(save_directory_hf))

compressed_size_on_disk_mb = os.path.getsize(save_directory_hf / "model.safetensors") / 1024 / 1024
print(f"Size of the model's weights on disk using safetensors: {compressed_size_on_disk_mb:.2f} MB")

