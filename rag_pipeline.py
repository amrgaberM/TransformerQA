import fitz  # PyMuPDF
from pathlib import Path
import re
import os
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv

class RAGPipeline:
    def __init__(self, groq_api_key: str, embedding_model_name: str = "BAAI/bge-base-en-v1.5",chromadb_dir: str = "chroma_db",chromadb_collection_name: str = "documents"):
        load_dotenv()
        self.groq_client = Groq(api_key=groq_api_key)
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.db_client = chromadb.PersistentClient(path=chromadb_dir)
        self.db_collection = self.db_client.get_or_create_collection(name="documents")

 
    def _extract_text_from_pdf(self,pdf_path: Path) -> str: 
       """Opens a PDF and returns its cleaned text content."""
       doc = fitz.open(pdf_path)
       full_text = []
       for page in doc:
         text = page.get_text().replace('\n', ' ').strip()
         text = re.sub(r'\s+', ' ', text)
         full_text.append(text)
       doc.close()
       return " ".join(full_text)
    
    def _chunk_text_smarter(self,text: str, chunk_size: int = 1536, chunk_overlap: int = 256) -> list[str]:
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

    def index_pdf(self,pdf_path: Path):
       print("Extracting text from PDF...")
       text=self._extract_text_from_pdf(pdf_path)
       print("Chunking text...")
       chunks=self._chunk_text_smarter(text)
       self.db_collection.add( 
        documents=chunks,
        embeddings=self.embedding_model.encode(chunks, show_progress_bar=True).tolist(), # Added a progress bar for large docs
        ids=[f"chunk_{i}" for i in range(len(chunks))]  # Added the required IDs
        )
    

    def answer_question(self,query: str) -> str:
       """Performs the Retrieve-Augment-Generate process."""
       # 1. RETRIEVE: Embed the query and find relevant context
       query_embedding = self.embedding_model.encode([query])
       results = self.db_collection.query(
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
       respond with "The provided context does not contain the answer to the question."
       """
       response = self.groq_client.chat.completions.create(
         model="llama-3.1-8b-instant",
         messages=[
           {"role": "system", "content": "You are a helpful assistant that provides accurate information based on the given context."},
           {"role": "user", "content": prompt}
         ],
         max_tokens=512,
         temperature=0.2,
         top_p=0.95,
         frequency_penalty=0,
         presence_penalty=0
       )
       return {
          "answer": response.choices[0].message.content.strip(),
          "retrieved_context": retrieved_context
       }
        



   