# app.py

import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

# Import your RAGPipeline class from the other file
from rag_pipeline import RAGPipeline

# --- PAGE CONFIGURATION ---
# Set the page title and a descriptive layout
st.set_page_config(
    page_title="Attention Is All You Need Q&A",
    layout="wide"
)

# --- LOAD ENVIRONMENT VARIABLES ---
# Load the API key from your .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- DEFINE FILE PATHS ---
# Define the path to the PDF document you want to query
# Using Path() makes it work on any operating system (Windows, Mac, Linux)
PDF_PATH = Path("C:/Users/amrHa/Desktop/TransformerQA/data/NIPS-2017-attention-is-all-you-need-Paper.pdf") #<-- IMPORTANT: Make sure this path is correct!

# --- INITIALIZE THE RAG PIPELINE (with Caching) ---
# This is a critical optimization. Streamlit reruns the script on every interaction.
# @st.cache_resource tells Streamlit to run this function only ONCE and then
# keep the returned object in memory. This prevents us from reloading the models
# and re-indexing the PDF every single time, which would be very slow.
@st.cache_resource
def load_rag_pipeline():
    """
    Loads the RAGPipeline and indexes the specified PDF.
    This function is cached to ensure it runs only once.
    """
    print("--- Running Initial Setup: Loading RAG Pipeline and Indexing PDF ---")
    pipeline = RAGPipeline(groq_api_key=GROQ_API_KEY)
    
    # Check if the PDF file exists before trying to index it
    if PDF_PATH.exists():
        pipeline.index_pdf(PDF_PATH)
    else:
        # If the PDF is not found, we stop the app and show an error.
        st.error(f"PDF file not found at: {PDF_PATH}")
        st.stop()
        
    print("--- Initial Setup Complete ---")
    return pipeline

# Load the pipeline. The first time the app runs, this will execute the function.
# On subsequent reruns, it will instantly return the cached object.
rag_system = load_rag_pipeline()


# --- USER INTERFACE (UI) ---

st.title("📄 Chat with 'Attention Is All You Need'")
st.markdown("""
    Welcome! This app allows you to ask questions about the research paper that introduced the Transformer architecture.
    Your questions will be answered based *only* on the content of the paper.
""")

# Add a text input box for the user to ask their question
user_question = st.text_input(
    "Ask your question:",
    placeholder="e.g., What is the main novelty of the Transformer architecture?"
)

# Add a button to submit the question
if st.button("Get Answer"):
    # Check if the user has entered a question
    if user_question:
        # Show a spinner while the model is processing the question
        with st.spinner("Finding the answer..."):
            # Call your RAG system's answer_question method
            response = rag_system.answer_question(user_question)
            
            # Display the answer
            st.success("Here is the answer:")
            st.write(response["answer"])

            # Use an expander to show the sources (retrieved context)
            # This is a great UI practice to keep the main view clean
            with st.expander("Show Sources"):
                st.info("The answer was generated based on the following context from the paper:")
                st.text(response["retrieved_context"])
    else:
        # Show a warning if the user clicks the button without entering a question
        st.warning("Please enter a question first.")