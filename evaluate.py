# evaluate.py

import csv
import os
from dotenv import load_dotenv

# --- Step 1: Import your RAGPipeline class ---
# Make sure your rag_pipeline.py file is in the same directory
from rag_pipeline import RAGPipeline 


# --- Step 2: SETUP PHASE ---
print("--- Setting up the evaluation ---")

# Use your .env file to load the API key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Define the file paths using raw strings (r"...")
golden_dataset_path = r"C:\Users\amrHa\Desktop\TransformerQA\evaluation\golden_dataset_attention.csv"
results_output_path = r"C:\\Users\\amrHa\Desktop\\TransformerQA\\evaluation\\results_attention.csv"
pdf_path = r"C:\Users\amrHa\Desktop\TransformerQA\data\NIPS-2017-attention-is-all-you-need-Paper.pdf" # <-- Make sure this path is correct

# Initialize your RAGPipeline with the API key
# --- Your code here to create an instance of RAGPipeline ---
rag_system = RAGPipeline(groq_api_key=groq_api_key)


# Index the PDF document so the pipeline can answer questions about it
print(f"Indexing document: {pdf_path}")
# --- Your code here to call the index_pdf method ---
rag_system.index_pdf(pdf_path)


# This list will hold the results of our evaluation
evaluation_results = []


# --- Step 3: EXECUTION PHASE ---
print("\n--- Starting the evaluation loop ---")

# Open and read your golden dataset CSV
with open(golden_dataset_path, mode='r', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)

    # Loop through each row
    for i, row in enumerate(reader):
        question = row['question']
        golden_answer = row['answer']

        print(f"Testing question {i+1}: '{question}'")

        # Call the correct method from your RAGPipeline to get the answer
        # --- Your code here to get the generated_answer ---
        generated_answer = rag_system.answer_question(question)
        

        # Store the results in a dictionary with clear key names
        result_entry = {
            'question': question,
            'golden_answer': golden_answer,
            'generated_answer': generated_answer
        }
        evaluation_results.append(result_entry)

print("\n--- Execution complete ---")


# --- Step 4: REPORTING PHASE ---
print(f"--- Writing results to {results_output_path} ---")

if evaluation_results:
    # Get the headers from the keys of the first dictionary
    headers = evaluation_results[0].keys()

    with open(results_output_path, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(evaluation_results)

print("--- Evaluation finished successfully! ---")