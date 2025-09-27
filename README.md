# PaperQA: Advanced RAG Chatbot for "Attention Is All You Need"

An end-to-end Retrieval-Augmented Generation (RAG) application enabling interactive conversations with the seminal AI paper "Attention Is All You Need." This system provides accurate information retrieval and complex reasoning capabilities based solely on the paper's content.

---

## Features

- **Interactive Web Interface**: Clean, intuitive UI built with Streamlit for seamless user interaction
- **High-Quality Retrieval**: Powered by `BAAI/bge-base-en-v1.5` embedding model for accurate, semantically relevant context retrieval
- **Fast Generation**: Utilizes `llama-3.1-8b-instant` model via Groq API for rapid, high-quality response generation
- **Verifiable Sources**: All answers include exact text chunks from the paper used as context, ensuring transparency and reliability

---

## Engineering Approach

This project represents a systematic approach to building and improving a production-ready AI system through iterative, data-driven development:

### Development Process

1. **Initial Scaffolding**
   - Established proof-of-concept with basic Python implementation
   - Created initial evaluation dataset to define success metrics

2. **Systematic Evaluation**
   - Built dedicated evaluation suite for quantitative performance measurement
   - Enabled data-driven improvement cycles beyond subjective assessments

3. **Diagnosing and Solving Retrieval Failure**
   - Identified critical retrieval failure with baseline `all-MiniLM-L6-v2` model
   - Diagnosed issues through context inspection
   - Upgraded to `bge-base-en-v1.5` embedding model, dramatically improving accuracy on complex technical questions

4. **Backend Optimization**
   - Refined performance through chunking strategy experimentation
   - Optimized chunk size and overlap parameters for enhanced LLM context

5. **Professional Refactoring**
   - Migrated from linear script to clean, object-oriented `RAGPipeline` class
   - Implemented separation of concerns and improved maintainability

6. **Application Development**
   - Built user-facing application with Streamlit
   - Connected robust backend to intuitive frontend

7. **Environment Management**
   - Ensured stability and reproducibility with dedicated Python 3.11 virtual environment

---

## Installation and Setup

### Prerequisites

- Python 3.11
- Groq API Key

### Installation Steps

#### 1. Clone Repository

```bash
git clone [your-repo-link]
cd [your-repo-folder]
```

#### 2. Create Virtual Environment

```bash
# Create environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Configure API Key

Create a `.env` file in the project root:

```env
GROQ_API_KEY="your-key-here"
```

#### 5. Launch Application

```bash
streamlit run app.py
```

The application will open automatically in your web browser.

---

## Project Structure

```
TransformerQA/
├── .gitignore          # Git ignore specifications
├── README.md           # Project documentation
├── requirements.txt    # Python dependencies
├── app.py             # Streamlit frontend application
├── rag_pipeline.py    # Core RAG backend implementation
├── .env               # API key configuration (local only)
├── data/              # Source PDF documents
└── evaluation/        # Testing datasets and results
```

---

## Technical Architecture

The system implements a modular RAG pipeline with the following components:

- **Document Processing**: Efficient chunking and preprocessing of source materials
- **Embedding Generation**: Semantic vector representations using state-of-the-art models
- **Retrieval System**: Context-aware document section retrieval
- **Generation Pipeline**: LLM-powered response generation with source attribution
- **Evaluation Framework**: Quantitative performance assessment and improvement tracking