from transformers import AutoModelForCausalLM, AutoTokenizer # Or the specific model/tokenizer class you used

# --- IMPORTANT: Load your model and tokenizer first ---
# This assumes you have the model and tokenizer loaded from the directory shown in your image.
# Replace 'AutoModelForCausalLM' with the correct class for your model if it's different
# (e.g., AutoModelForSequenceClassification, AutoModelForTokenClassification, etc.)

model_directory = "qwen3_0.6B_mcqa_cot/final_model/"
model_repo_id = "VinceEPFL/mcqa_cot"  # <--- CHOOSE YOUR MODEL REPO ID (can be different from dataset repo)

try:
    print(f"Loading model from {model_directory}...")
    # Adjust AutoModelForCausalLM if your model is for a different task
    model = AutoModelForCausalLM.from_pretrained(model_directory)
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    print("Model and tokenizer loaded successfully.")

    # --- Push the model and tokenizer to the Hub ---
    print(f"\nPushing model and tokenizer to Hugging Face Hub at {model_repo_id}...")
    
    # This will create the repository if it doesn't exist.
    # You can add `private=True` if you want the model to be private.
    # You can add `commit_message="feat: Upload fine-tuned Qwen3 0.6B"`
    model.push_to_hub(model_repo_id)
    tokenizer.push_to_hub(model_repo_id)
    
    print(f"Model and tokenizer successfully pushed to {model_repo_id}")
    print(f"Access your model at: https://huggingface.co/{model_repo_id}")

except FileNotFoundError:
    print(f"ERROR: The directory {model_directory} was not found. Please check the path.")
except Exception as e:
    print(f"An error occurred: {e}")
    print("Make sure you are authenticated with `huggingface-cli login` or `notebook_login()`.")