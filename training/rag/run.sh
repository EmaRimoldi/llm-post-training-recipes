#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

echo "Creating virtual environment with Python 3.12.8..."

# Ensure python3.12 is available
if ! command -v python3.12 &> /dev/null
then
    echo "Error: python3.12 is not installed or not in PATH."
    exit 1
fi

# Create and activate virtual environment
python3.12 -m venv venv_mcqa
source venv_mcqa/bin/activate

echo "Virtual environment activated."

# Upgrade pip to specified version
echo "Upgrading pip to version 25.1.1..."
python -m pip install --upgrade pip==25.1.1

# Install dependencies
echo "Installing dependencies from requirements.txt..."
# Assuming requirements.txt is in the root or a known location. 
# If it's in a specific folder, you'll need to adjust this path too.
pip install -r ./train_rag/requirements.txt

# Run the MCQA training script
echo "Running MCQA model training..."
python train_rag/train_mcqa_rag.py

# Run the RAG-based fine-tuning script
echo "Running RAG-based fine-tuning (RAFT + LoRA)..."
# Assuming raft_LoRA.py is within the train_rag directory based on common practices,
# though it's not explicitly shown in your screenshot. If it's elsewhere, adjust the path.
python train_rag/raft_LoRA.py 

echo "Training process completed successfully."