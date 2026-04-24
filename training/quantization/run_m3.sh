#!/bin/bash

# This script creates a virtual environment, installs dependencies,
# and then runs the main Python script.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- 1. Set up the Virtual Environment ---

# Define the name of the virtual environment directory
VENV_DIR="venv"

# Check if the directory does NOT exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment in './$VENV_DIR'..."
    # Create the virtual environment
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment './$VENV_DIR' already exists."
fi

# Activate the virtual environment
# After this line, all 'pip' and 'python3' commands are local to the venv
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"


# --- 2. Install Dependencies into the venv ---

echo "Installing Python dependencies..."
# The 'pip' command now points to the venv's pip
pip install -r ./train_quantized/requirements.txt
echo "Dependencies installed."


# --- 3. Run the Python Script using the venv's python ---

echo "Executing Python script: quantize.py..."
# The 'python3' command now points to the venv's python
python3 ./train_quantized/llm-compressor.py
echo "----------------------------------------"
echo "✅ Bash script completed successfully."
echo "----------------------------------------"

# The virtual environment is automatically deactivated when the script finishes.