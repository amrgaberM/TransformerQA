import fitz  # PyMuPDF
from pathlib import Path
import re
import os
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv

# --- FUNCTIONS FOR DATA PREPARATION ---

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Opens a PDF and returns its cleaned text content."""
    doc = fitz.open(pdf_path)
    full_text = []
    for page in doc:
        text = page.get_text().replace('\n', ' ').strip()
        text = re.sub(r'\s+', ' ', text)
        full_text.append(text)
    doc.close()
    return " ".join(full_text)

def chunk_text_smarter(text: str, chunk_size: int = 1024, chunk_overlap: int = 128) -> list[str]:
    """Splits text into semantically aware chunks."""
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 < chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = current_chunk[-chunk_overlap:] + sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# --- FUNCTION FOR RAG PIPELINE ---

def answer_question(query, groq_client, embedding_model, db_collection):
    """Performs the Retrieve-Augment-Generate process."""
    # 1. RETRIEVE: Embed the query and find relevant context
    query_embedding = embedding_model.encode([query])
    results = db_collection.query(
        query_embeddings=query_embedding,
        n_results=2 # Retrieve top 2 most relevant chunks
    )
    retrieved_context = "\n\n---\n\n".join(results['documents'][0])

    # 2. AUGMENT & 3. GENERATE: Create a prompt and get an answer from the LLM
    prompt = f"""
    CONTEXT:
    {retrieved_context}

    QUESTION:
    {query}

    INSTRUCTIONS:
    Answer the user's QUESTION based ONLY on the CONTEXT provided above.
    Provide a detailed and comprehensive answer. If the context doesn't contain the answer,
    state that you don't have enough information from the provided document.
    """
    
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # --- SETUP ---
    load_dotenv()
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    db_client = chromadb.Client()
    collection = db_client.get_or_create_collection(name="attention_paper")

    # --- INDEXING PHASE ---
    print("Starting the indexing process...")
    pdf_path = Path("C:/Users/amrHa/Desktop/proejctA/data/NIPS-2017-attention-is-all-you-need-Paper.pdf")
    document_text = extract_text_from_pdf(pdf_path)
    text_chunks = chunk_text_smarter(document_text)

    print(f"Embedding {len(text_chunks)} chunks... (This may take a moment)")
    embeddings = embedding_model.encode(text_chunks, show_progress_bar=True)
    
    collection.add(
        embeddings=embeddings,
        documents=text_chunks,
        ids=[f"chunk_{i}" for i in range(len(text_chunks))]
    )
    print(" Indexing complete. You can now ask questions.")

    # --- QUERY PHASE ---
    while True:
        print("\n" + "="*50)
        user_query = input("Ask a question about the paper (or type 'quit' to exit): ")
        if user_query.lower() == 'quit':
            break
        
        final_answer = answer_question(user_query, groq_client, embedding_model, collection)
        print("\n--- Answer ---")
        print(final_answer)