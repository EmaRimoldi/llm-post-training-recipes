"""
Script to check if the environment is properly configured before running QLoRA
"""

import os
import sys
import torch
import transformers
from datasets import load_dataset

def check_environment():
    print("=== Environment Check ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    
    print("\n=== Path Check ===")
    print(f"Current directory: {os.getcwd()}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    
    print("\n=== File Check ===")
    required_files = [
        'qlora_contrastive_full.py',
        '../data/mathqa/train.json',
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ Found: {file_path}")
        else:
            print(f"✗ Missing: {file_path}")
    
    print("\n=== Import Check ===")
    try:
        from peft import LoraConfig, get_peft_model
        print("✓ PEFT import successful")
    except ImportError as e:
        print(f"✗ PEFT import failed: {e}")
    
    try:
        from transformers import BitsAndBytesConfig
        print("✓ BitsAndBytes import successful")
    except ImportError as e:
        print(f"✗ BitsAndBytes import failed: {e}")

if __name__ == "__main__":
    check_environment()