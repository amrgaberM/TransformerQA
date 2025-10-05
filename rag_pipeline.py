"""
RAG Pipeline Module
===================
A professional Retrieval-Augmented Generation system for PDF document Q&A.

Features:
- PDF text extraction with page tracking
- Intelligent text chunking with overlap
- Hybrid retrieval: Detail chunks + Summary chunks
- Vector embeddings using SentenceTransformers
- ChromaDB for efficient similarity search
- Groq LLM integration for answer generation

Author: Amr Hassan
Version: 2.0 (Hybrid Retrieval)
"""

import fitz  # PyMuPDF
from pathlib import Path
import re
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Data class for text chunks with metadata."""
    text: str
    pages: List[int]
    chunk_id: Optional[str] = None


@dataclass
class RAGResponse:
    """Data class for RAG system responses."""
    answer: str
    retrieved_context: str
    pages: List[int]
    confidence_score: Optional[float] = None
    sources_count: int = 0


class RAGPipeline:
    """
    Professional RAG Pipeline for document-based question answering.
    
    This pipeline extracts text from PDFs, creates embeddings, stores them in
    a vector database, and uses them to provide context-aware answers to queries.
    
    Attributes:
        groq_client: Client for Groq LLM API
        embedding_model: SentenceTransformer model for embeddings
        db_client: ChromaDB persistent client
        detail_collection: ChromaDB collection for detailed chunks
        summary_collection: ChromaDB collection for per-page summaries
    """
    
    # Class constants
    DEFAULT_CHUNK_SIZE = 1536
    DEFAULT_CHUNK_OVERLAP = 256
    DEFAULT_TOP_K_RESULTS = 3
    DEFAULT_MODEL = "llama-3.1-8b-instant"
    SIMILARITY_THRESHOLD = 0.5
    MAX_DISTANCE = 2.0
    
    def __init__(
        self,
        groq_api_key: str,
        embedding_model_name: str = "BAAI/bge-base-en-v1.5",
        chromadb_dir: str = "chroma_db",
        chromadb_collection_name: str = "documents"
    ):
        """
        Initialize the RAG Pipeline.
        
        Args:
            groq_api_key: API key for Groq LLM service
            embedding_model_name: Name of the SentenceTransformer model
            chromadb_dir: Directory path for ChromaDB persistence
            chromadb_collection_name: Name of the ChromaDB collection
        """
        load_dotenv()
        
        try:
            logger.info("Initializing RAG Pipeline...")
            
            # Initialize Groq client
            self.groq_client = Groq(api_key=groq_api_key)
            logger.info("✓ Groq client initialized")
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(embedding_model_name)
            logger.info(f"✓ Embedding model loaded: {embedding_model_name}")
            
            # Initialize ChromaDB
            self.db_client = chromadb.PersistentClient(path=chromadb_dir)
            
            # Create the collection for detailed, small chunks
            self.detail_collection = self.db_client.get_or_create_collection(
                name=f"{chromadb_collection_name}_details"
            )
            logger.info(f"✓ Detail collection ready: {self.detail_collection.name}")
            
            # Create the collection for high-level, per-page summaries
            self.summary_collection = self.db_client.get_or_create_collection(
                name=f"{chromadb_collection_name}_summaries"
            )
            logger.info(f"✓ Summary collection ready: {self.summary_collection.name}")
            
            logger.info("RAG Pipeline initialization complete!")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG Pipeline: {e}")
            raise
    
    def _extract_text_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Extract text content from PDF with page number tracking.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing text and page numbers
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If PDF processing fails
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            logger.info(f"Opening PDF: {pdf_path.name}")
            doc = fitz.open(pdf_path)
            text_with_pages = []
            
            for page_num in tqdm(range(len(doc)), desc="Extracting pages"):
                page = doc[page_num]
                text = page.get_text()
                
                # Clean text: normalize whitespace and remove excess newlines
                text = text.replace('\n', ' ').strip()
                text = re.sub(r'\s+', ' ', text)
                
                if text:  # Only add non-empty pages
                    text_with_pages.append({
                        "text": text,
                        "page": page_num + 1
                    })
            
            doc.close()
            logger.info(f"✓ Extracted {len(text_with_pages)} pages")
            return text_with_pages
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    def _chunk_text_smarter(
        self,
        text_with_pages: List[Dict[str, Any]],
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    ) -> List[TextChunk]:
        """
        Split text into semantically-aware chunks with page tracking.
        
        This method uses sentence boundaries to create chunks that maintain
        semantic coherence while respecting size constraints.
        
        Args:
            text_with_pages: List of text segments with page numbers
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks
            
        Returns:
            List of TextChunk objects with text and page metadata
        """
        logger.info(f"Chunking text (size={chunk_size}, overlap={chunk_overlap})...")
        
        # Concatenate all text and track page boundaries
        full_text = " ".join([item["text"] for item in text_with_pages])
        page_boundaries = []
        current_pos = 0
        
        for item in text_with_pages:
            page_text = item["text"]
            page_boundaries.append({
                "page": item["page"],
                "start": current_pos,
                "end": current_pos + len(page_text)
            })
            current_pos += len(page_text) + 1  # +1 for the space separator
        
        # Split by sentence boundaries for semantic coherence
        sentences = re.split(r'(?<=[.!?])\s+', full_text)
        chunks = []
        current_chunk = ""
        current_pages = set()
        char_pos = 0
        
        for sentence in sentences:
            sentence_with_space = sentence + " "
            
            # Check if adding this sentence exceeds chunk size
            if len(current_chunk) + len(sentence_with_space) < chunk_size:
                current_chunk += sentence_with_space
                
                # Track which pages this chunk spans
                for boundary in page_boundaries:
                    if (char_pos < boundary["end"] and 
                        (char_pos + len(sentence_with_space)) > boundary["start"]):
                        current_pages.add(boundary["page"])
                
                char_pos += len(sentence_with_space)
            else:
                # Save current chunk
                if current_chunk.strip():
                    chunks.append(TextChunk(
                        text=current_chunk.strip(),
                        pages=sorted(list(current_pages))
                    ))
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else ""
                current_chunk = overlap_text + sentence_with_space
                current_pages = set()
                
                # Recalculate pages for the new chunk (including overlap)
                overlap_start = char_pos - len(overlap_text)
                for boundary in page_boundaries:
                    if (overlap_start < boundary["end"] and 
                        (char_pos + len(sentence_with_space)) > boundary["start"]):
                        current_pages.add(boundary["page"])
                
                char_pos += len(sentence_with_space) - len(overlap_text)
        
        # Add final chunk if it exists
        if current_chunk.strip():
            chunks.append(TextChunk(
                text=current_chunk.strip(),
                pages=sorted(list(current_pages))
            ))
        
        logger.info(f"✓ Created {len(chunks)} chunks")
        return chunks
    
    def index_pdf(
        self,
        pdf_path: Path,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    ) -> int:
        """
        Extract, chunk, and index a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            Number of chunks indexed
            
        Raises:
            Exception: If indexing fails
        """
        try:
            logger.info(f"Starting indexing process for: {pdf_path.name}")
            
            # Step 1: Extract text
            text_with_pages = self._extract_text_from_pdf(pdf_path)
            
            # Step 2: Chunk text for detail index
            detail_chunks = self._chunk_text_smarter(
                text_with_pages,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            # Step 3: Generate embeddings and store in the detail_collection
            logger.info("Generating embeddings for detail chunks...")
            chunk_texts = [chunk.text for chunk in detail_chunks]
            detail_embeddings = self.embedding_model.encode(
                chunk_texts,
                show_progress_bar=True,
                batch_size=32
            ).tolist()
            
            logger.info("Storing detail chunks in vector database...")
            self.detail_collection.add(
                documents=chunk_texts,
                embeddings=detail_embeddings,
                metadatas=[{"pages": ",".join(map(str, chunk.pages))} for chunk in detail_chunks],
                ids=[f"chunk_{i:04d}" for i in range(len(detail_chunks))]
            )
            
            logger.info(f"✓ Successfully indexed {len(detail_chunks)} detail chunks")
            
            # Step 4: Index the per-page summaries
            logger.info("Indexing per-page summaries...")
            page_texts = [page['text'] for page in text_with_pages]
            page_metadatas = [{"pages": str(page['page'])} for page in text_with_pages]
            
            summary_embeddings = self.embedding_model.encode(
                page_texts,
                show_progress_bar=True,
                batch_size=32
            ).tolist()
            
            self.summary_collection.add(
                documents=page_texts,
                embeddings=summary_embeddings,
                metadatas=page_metadatas,
                ids=[f"page_{i+1}" for i in range(len(page_texts))]
            )
            logger.info(f"✓ Successfully indexed {len(page_texts)} summary chunks")
            
            return len(detail_chunks) + len(page_texts)
            
        except Exception as e:
            logger.error(f"Failed to index PDF: {e}")
            raise
    
    def answer_question(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K_RESULTS,
        temperature: float = 0.2,
        max_tokens: int = 512
    ) -> Dict[str, Any]:
        """
        Answer a question using the RAG pipeline.
        
        This method:
        1. Routes query to appropriate collection (detail vs summary)
        2. Retrieves relevant context from the vector database
        3. Augments the query with retrieved context
        4. Generates an answer using the LLM
        
        Args:
            query: User's question
            top_k: Number of relevant chunks to retrieve
            temperature: LLM temperature parameter
            max_tokens: Maximum tokens in response
            
        Returns:
            Dictionary containing answer, context, and metadata
        """
        try:
            logger.info(f"Processing query: {query[:50]}...")
            
            # ROUTER: Decide which collection to use
            summary_keywords = ["summary", "summarize", "main contribution", "contribution", 
                              "overview", "in general", "main idea", "high-level", 
                              "key innovation", "main point", "purpose"]
            
            use_summary_collection = any(keyword in query.lower() for keyword in summary_keywords)
            
            if use_summary_collection:
                target_collection = self.summary_collection
                top_k_results = 3  # Retrieve more to ensure we get the best pages
                logger.info("Query type is 'Summary'. Using summary collection.")
            else:
                target_collection = self.detail_collection
                top_k_results = self.DEFAULT_TOP_K_RESULTS
                logger.info("Query type is 'Detail'. Using detail collection.")
            
            # Step 1: RETRIEVE - Find relevant context
            query_embedding = self.embedding_model.encode([query]).tolist()
            results = target_collection.query(
                query_embeddings=query_embedding,
                n_results=top_k_results
            )
            
            # Ensure deterministic ordering by sorting by distance then by ID
            # This prevents inconsistency when similarity scores are very close
            distances = results['distances'][0]
            retrieved_docs = results['documents'][0]
            metadatas = results['metadatas'][0]
            ids = results['ids'][0]
            
            # Create list of tuples for sorting
            combined = list(zip(distances, retrieved_docs, metadatas, ids))
            # Sort by distance (ascending), then by ID for deterministic ordering
            combined.sort(key=lambda x: (round(x[0], 6), x[3]))
            
            # Unpack sorted results
            distances, retrieved_docs, metadatas, ids = zip(*combined) if combined else ([], [], [], [])
            distances = list(distances)
            retrieved_docs = list(retrieved_docs)
            metadatas = list(metadatas)
            
            # Extract distances and filter based on threshold
            # distances, retrieved_docs, metadatas already unpacked above
            
            # Normalize distances to similarities (assuming L2; 0=perfect match, higher=worse)
            similarities = [max(0, 1 - (d / self.MAX_DISTANCE)) for d in distances]
            
            # For summary queries, prioritize early pages (likely intro/abstract)
            if use_summary_collection:
                # Boost scores for early pages
                boosted_results = []
                for i, (doc, sim, meta) in enumerate(zip(retrieved_docs, similarities, metadatas)):
                    if meta and 'pages' in meta:
                        page_num = int(meta['pages'])
                        # Give significant boost to first 3 pages (likely intro/abstract/conclusion)
                        if page_num <= 3:
                            sim = min(1.0, sim * 1.3)  # 30% boost
                        elif page_num <= 5:
                            sim = min(1.0, sim * 1.15)  # 15% boost
                    boosted_results.append((sim, doc, meta))
                
                # Re-sort by boosted similarity
                boosted_results.sort(key=lambda x: x[0], reverse=True)
                similarities = [x[0] for x in boosted_results]
                retrieved_docs = [x[1] for x in boosted_results]
                metadatas = [x[2] for x in boosted_results]
                
                # Take top 2 for summary
                similarities = similarities[:2]
                retrieved_docs = retrieved_docs[:2]
                metadatas = metadatas[:2]
            
            logger.info(f"Retrieved {len(retrieved_docs)} chunks before filtering")
            logger.info(f"Pages retrieved: {[meta.get('pages') for meta in metadatas if meta]}")
            
            # Filter: Keep only chunks above threshold
            filtered_docs = []
            filtered_pages = set()
            filtered_similarities = []
            for doc, sim, meta in zip(retrieved_docs, similarities, metadatas):
                if sim >= self.SIMILARITY_THRESHOLD:
                    filtered_docs.append(doc)
                    filtered_similarities.append(sim)
                    if meta and 'pages' in meta and meta['pages']:
                        pages_str = meta['pages']
                        if pages_str and isinstance(pages_str, str):
                            filtered_pages.update([int(p.strip()) for p in pages_str.split(',') if p.strip()])
            
            retrieved_context = "\n\n---\n\n".join(filtered_docs) if filtered_docs else ""
            
            logger.info(f"Retrieved {len(filtered_docs)} relevant chunks (after filtering) from pages: {sorted(filtered_pages)}")
            
            # If no good matches, short-circuit with low confidence
            if not filtered_docs:
                return {
                    "answer": "No relevant information found in the document.",
                    "retrieved_context": "",
                    "pages": [],
                    "confidence_score": 0.0,
                    "sources_count": 0
                }
            
            # Step 2: AUGMENT - Create enhanced prompt
            system_prompt = """You are an expert research assistant specializing in academic paper analysis. 
Your role is to provide accurate, detailed answers based strictly on the provided context from research papers.
Always maintain academic rigor and cite information appropriately."""
            
            user_prompt = f"""CONTEXT FROM RESEARCH PAPER:
{retrieved_context}

USER QUESTION:
{query}

INSTRUCTIONS:
- Provide a comprehensive and detailed answer based ONLY on the context above
- If the context contains the answer, explain it clearly and thoroughly
- If the context doesn't contain sufficient information, state: "The provided context does not contain enough information to answer this question."
- Maintain academic tone and precision
- Do not add information beyond what's in the context

ANSWER:"""
            
            # Step 3: GENERATE - Get LLM response
            logger.info("Generating answer with LLM...")
            response = self.groq_client.chat.completions.create(
                model=self.DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info("✓ Answer generated successfully")
            
            # Calculate improved confidence: Average similarity of filtered chunks
            confidence_score = sum(filtered_similarities) / len(filtered_similarities) if filtered_similarities else 0.0
            
            return {
                "answer": answer,
                "retrieved_context": retrieved_context,
                "pages": sorted(list(filtered_pages)),
                "confidence_score": confidence_score,
                "sources_count": len(filtered_docs)
            }
            
        except Exception as e:
            logger.error(f"Error during question answering: {e}")
            return {
                "answer": f"An error occurred while processing your question: {str(e)}",
                "retrieved_context": "",
                "pages": [],
                "confidence_score": 0.0,
                "sources_count": 0
            }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the indexed document collection.
        
        Returns:
            Dictionary containing collection statistics
        """
        try:
            detail_count = self.detail_collection.count()
            summary_count = self.summary_collection.count()
            return {
                "detail_chunks": detail_count,
                "summary_chunks": summary_count,
                "total_chunks": detail_count + summary_count,
                "detail_collection_name": self.detail_collection.name,
                "summary_collection_name": self.summary_collection.name,
                "status": "active" if (detail_count > 0 or summary_count > 0) else "empty"
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from both collections.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.warning("Clearing collections...")
            
            # Clear detail collection
            self.db_client.delete_collection(name=self.detail_collection.name)
            self.detail_collection = self.db_client.create_collection(
                name=self.detail_collection.name
            )
            
            # Clear summary collection
            self.db_client.delete_collection(name=self.summary_collection.name)
            self.summary_collection = self.db_client.create_collection(
                name=self.summary_collection.name
            )
            
            logger.info("✓ Both collections cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing collections: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    # Initialize pipeline
    pipeline = RAGPipeline(groq_api_key=api_key)

    #breakpoint()  # For debugging purposes
    
    # Example: Index a PDF
    pdf_path = Path(r"C:\Users\amrHa\Desktop\TransformerQA\data\NIPS-2017-attention-is-all-you-need-Paper.pdf")
    if pdf_path.exists():
        pipeline.index_pdf(pdf_path)
        
        # Example: Ask a question
        result = pipeline.answer_question("What is the main contribution of this paper?")
        print(f"\nAnswer: {result['answer']}")
        print(f"Pages: {result['pages']}")
        print(f"Confidence: {result['confidence_score']:.2%}")