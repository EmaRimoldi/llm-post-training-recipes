import json
import re
import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, login

def prepare_and_upload_aimcqs(input_file, dataset_name, hf_username, private=False):
    """
    Prepare and upload the AIMCQS dataset to Hugging Face Hub.
    """
    print(f"Loading data from {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} questions")
    
    prepared_data = []
    
    for i, item in enumerate(data):
        try:
            question = item.get('question', '').strip()
            choices = item.get('choices', [])
            answer = item.get('answer', '').upper().strip()
            category = item.get('category', 'unknown')
            
            # Clean question - remove numbering
            question = re.sub(r'^\d+\.\s*', '', question)
            
            if not question or not choices or not answer or len(choices) != 4:
                print(f"Skipping item {i}: invalid structure")
                continue
            
            # Clean choices - remove a./b./c./d. prefixes and format consistently
            cleaned_choices = []
            for j, choice in enumerate(choices):
                choice_text = choice.strip()
                # Remove prefixes like "a.", "b.", "a)", "b)", etc.
                choice_text = re.sub(r'^[a-d][.)\s]*', '', choice_text).strip()
                
                # Format as "A. text", "B. text", etc.
                letter = chr(65 + j)  # A, B, C, D
                cleaned_choices.append(f"{letter}. {choice_text}")
            
            # Ensure answer is valid
            if answer not in ['A', 'B', 'C', 'D']:
                print(f"Skipping item {i}: invalid answer '{answer}'")
                continue
            
            prepared_item = {
                'question': question,
                'choices': cleaned_choices,
                'answer': answer,
                'category': category
            }
            
            prepared_data.append(prepared_item)
            
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            continue
    
    print(f"Successfully prepared {len(prepared_data)} questions")
    
    # Show sample
    if prepared_data:
        print("\nSample prepared question:")
        sample = prepared_data[0]
        print(f"Question: {sample['question']}")
        print(f"Choices: {sample['choices']}")
        print(f"Answer: {sample['answer']}")
        print(f"Category: {sample['category']}")
    
    # Show statistics
    categories = {}
    for item in prepared_data:
        cat = item['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\n📊 Dataset statistics:")
    print(f"Total questions: {len(prepared_data)}")
    print("Questions by category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    
    # Convert to DataFrame and create Dataset
    print(f"\n📤 Uploading to Hugging Face...")
    df = pd.DataFrame(prepared_data)
    dataset = Dataset.from_pandas(df)
    
    # Create single test split (no train/test split)
    dataset_dict = DatasetDict({'test': dataset})
    print(f"Created test split: {len(dataset)} examples")
    
    # Upload to Hub
    repo_id = f"{hf_username}/{dataset_name}"
    print(f"Uploading dataset to {repo_id}...")
    
    try:
        dataset_dict.push_to_hub(
            repo_id=repo_id,
            private=private,
            commit_message="Upload AIMCQS multiple choice dataset"
        )
        print(f"✅ Dataset successfully uploaded!")
        print(f"🔗 View at: https://huggingface.co/datasets/{repo_id}")
        
        print(f"\n📋 To use in your training script, update:")
        print(f'MCQA_DATASETS = ["{repo_id}"]')
        
        return repo_id
        
    except Exception as e:
        print(f"❌ Error uploading dataset: {e}")
        print(f"\n🔑 Make sure you're logged in:")
        print("Run: huggingface-cli login")
        return None

def main():
    # UPDATE THESE VALUES
    INPUT_FILE = "aimcqs_mcqa.json"       # Your JSON file
    DATASET_NAME = "aimcqs-mcqa"          # Name for HF dataset
    HF_USERNAME = "VinceEPFL"         # YOUR HF username
    PRIVATE = False                       # Set True for private dataset
    
    print("🚀 Uploading AIMCQS dataset to Hugging Face...")
    
    repo_id = prepare_and_upload_aimcqs(
        input_file=INPUT_FILE,
        dataset_name=DATASET_NAME, 
        hf_username=HF_USERNAME,
        private=PRIVATE
    )
    
    if repo_id:
        print(f"\n🎉 Success! Dataset ready at: {repo_id}")
        print(f"\n📝 Next steps:")
        print(f"1. Update your training script:")
        print(f'   MCQA_DATASETS = ["{repo_id}"]')
        print(f"2. Run your training script!")
    else:
        print(f"\n❌ Upload failed. Check your credentials and try again.")

if __name__ == "__main__":
    main()