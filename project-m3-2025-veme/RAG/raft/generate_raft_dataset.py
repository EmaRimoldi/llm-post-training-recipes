
import os
import pandas as pd
from datasets import load_dataset
import random
import json
import time
from tqdm import tqdm 

# --- Configure your gpt_wrapper API ---
import gpt_wrapper
from gpt_wrapper.chat import Chat

gpt_wrapper.api_base = "http://mnlp-backend-lb-1062233132.eu-central-1.elb.amazonaws.com"
gpt_wrapper.api_key = os.getenv("MNLP_GPT_WRAPPER_API_KEY")

if not gpt_wrapper.api_key:
    raise RuntimeError("Set MNLP_GPT_WRAPPER_API_KEY before running this script.")

# --- Configuration for Dataset Generation ---
BASE_DATASET_NAME = "tommymarto/mnlp_project_camelai_subset"
OUTPUT_FILE = "raft_dataset_generated_with_gpt_wrapper.jsonl"
NUM_DISTRACTORS = 3
NUM_EXAMPLES_TO_GENERATE = 3000  # Set the number of examples to 10,000

RAFT_INSTRUCTION = """You are a knowledgeable and precise STEM professional with expertise in science, technology, engineering, and mathematics.
Your task is to answer the following question based ONLY on the provided documents.
If the answer is not present in the documents, state that you cannot answer.
For each piece of information you use, cite the exact text from the document by enclosing it in ##begin_quote## and ##end_quote## tags.
Provide a detailed Chain-of-Thought reasoning before giving the final answer.
"""

# --- Load Base Dataset ---
print(f"Loading base dataset: {BASE_DATASET_NAME}...")
try:
    base_dataset = load_dataset(BASE_DATASET_NAME, split="train")
    print(f"Dataset loaded successfully. Total examples: {len(base_dataset)}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please ensure the dataset name is correct and you have an internet connection.")
    exit()

# Convert to pandas DataFrame for easier manipulation
base_df = base_dataset.to_pandas()
base_df = base_df.dropna(subset=['input', 'target'])
print(f"Dataset after dropping NaNs: {len(base_df)} examples.")

# --- Function to Generate RAFT Example using gpt_wrapper API ---
def generate_raft_example_with_gpt_wrapper(question: str, golden_answer: str, all_possible_answers: list, num_distractors: int, chat_name: str) -> dict:
    distractor_candidates = [ans for ans in all_possible_answers if ans != golden_answer]
    
    if len(distractor_candidates) < num_distractors:
        selected_distractors = random.sample(distractor_candidates, len(distractor_candidates))
    else:
        selected_distractors = random.sample(distractor_candidates, num_distractors)

    # Prepare the documents for the prompt
    documents_for_prompt = [f"##begin_document##\n{golden_answer}\n##end_document##"] + \
                           [f"##begin_document##\n{d}\n##end_document##" for d in selected_distractors]
    random.shuffle(documents_for_prompt)  # Shuffle to mix golden and distractors

    # Construct the content for the chat.ask method
    content_prompt = f"""
    Question: {question}

    Documents:
    {"\n".join(documents_for_prompt)}

    Chain-of-Thought Answer:
    """

    try:
        # Create a new chat instance for each question as recommended by gpt_wrapper docs
        chat_session = Chat.create(name=chat_name)
        
        # Use the ask method with the instruction and content
        message = chat_session.ask(
            content=content_prompt,
            instruction=RAFT_INSTRUCTION,
            model_args={
                "temperature": 0.7,  # Default is 0.7, can be adjusted
                "max_tokens": 500,   # Adjust based on expected answer length
                "top_p": 0.9,        # Default is 0.9, can be adjusted
            }
        )
        cot_answer = message.content
    except Exception as e:
        print(f"Error during gpt_wrapper API call for question '{question[:50]}...': {e}")
        cot_answer = "Error: Could not generate answer."

    raft_example = {
        "question": question,
        "documents": [golden_answer] + selected_distractors,  # Store raw documents for clarity
        "answer_cot": cot_answer,
        "golden_document_index": 0  # Assuming the first document in 'documents' list is always the golden one
    }
    return raft_example

# --- Main Generation Loop ---
generated_raft_data = []  # Initialize the list for storing generated data
all_base_answers = base_df['target'].tolist()  # Get all answers for distractor sampling

print(f"Starting RAFT dataset generation for {NUM_EXAMPLES_TO_GENERATE} examples...")
with tqdm(total=NUM_EXAMPLES_TO_GENERATE, desc="Generating RAFT Examples", unit="example") as pbar:
    for i in range(min(NUM_EXAMPLES_TO_GENERATE, len(base_df))):
        row = base_df.iloc[i]
        question = row['input']
        golden_answer = row['target']
        
        # Create a unique chat name for each example
        chat_session_name = f"raft_gen_example_{i}"

        raft_example = generate_raft_example_with_gpt_wrapper(question, golden_answer, all_base_answers, NUM_DISTRACTORS, chat_session_name)
        generated_raft_data.append(raft_example)

        # Print every 10 questions
        if (i + 1) % 10 == 0:
            print(f"Processing example {i + 1}/{min(NUM_EXAMPLES_TO_GENERATE, len(base_df))}...")

        # Save to file every iteration
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(raft_example, ensure_ascii=False) + '\n')
        
        # Update progress bar
        pbar.update(1)

        # Add a small delay to avoid hitting API rate limits
        time.sleep(1)  # Increased delay for external API stability

print("\nGeneration complete!")

# --- Display a sample of the generated data ---
print("\n--- Sample of Generated RAFT Data ---")
if generated_raft_data:
    for j, example in enumerate(generated_raft_data[:3]):  # Display first 3 examples
        print(f"\nExample {j + 1}:")
        print(f"  Question: {example['question']}")
        print(f"  Documents (first is golden): {example['documents']}")
        print(f"  Generated CoT Answer: {example['answer_cot']}")
else:
    print("No data generated.")
