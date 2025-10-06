import streamlit as st
import time
import json
import plotly.graph_objects as go
import os
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Transformer Paper Q&A System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DARK ARTISTIC CSS DESIGN ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global Dark Theme */
    * {
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .main {
        background: #000000;
        color: #e0e0e0;
    }
    
    /* Animated Gradient Background */
    .main::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle at 20% 50%, rgba(120, 119, 198, 0.05) 0%, transparent 50%),
                    radial-gradient(circle at 80% 80%, rgba(74, 144, 226, 0.05) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }
    
    .main > div {
        position: relative;
        z-index: 1;
    }
    
    /* Header with Geometric Art */
    .header-container {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        padding: 3rem 2rem;
        border-radius: 0;
        margin: -1rem -1rem 3rem -1rem;
        border-bottom: 2px solid #7877c6;
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, 
            transparent 0%, 
            #7877c6 20%, 
            #4a90e2 40%, 
            #7877c6 60%, 
            transparent 100%);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .header-title {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        letter-spacing: -1px;
        text-transform: uppercase;
    }
    
    .header-subtitle {
        color: #a0a0a0;
        font-size: 0.95rem;
        margin: 0;
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    
    /* Glass Morphism Cards */
    .glass-card {
        background: rgba(26, 26, 26, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(120, 119, 198, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(120, 119, 198, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 12px 40px 0 rgba(120, 119, 198, 0.15);
    }
    
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #7877c6;
        margin: 0 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(120, 119, 198, 0.2);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Input Styling */
    .stTextInput > div > div > input {
        background: rgba(26, 26, 26, 0.8) !important;
        border: 1px solid rgba(120, 119, 198, 0.3) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        font-size: 0.95rem !important;
        color: #e0e0e0 !important;
        transition: all 0.3s !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #7877c6 !important;
        box-shadow: 0 0 0 2px rgba(120, 119, 198, 0.2) !important;
        background: rgba(26, 26, 26, 0.95) !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #666666 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7877c6 0%, #4a90e2 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        transition: all 0.3s !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(120, 119, 198, 0.4) !important;
    }
    
    /* Answer Box */
    .answer-box {
        background: linear-gradient(135deg, rgba(26, 26, 26, 0.95) 0%, rgba(35, 35, 35, 0.95) 100%);
        padding: 2rem;
        border-left: 3px solid #7877c6;
        border-radius: 8px;
        font-size: 1rem;
        line-height: 1.8;
        color: #d0d0d0;
        position: relative;
        overflow: hidden;
    }
    
    .answer-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 3px;
        height: 100%;
        background: linear-gradient(180deg, #7877c6 0%, #4a90e2 100%);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Metrics Cards */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #7877c6 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        color: #808080 !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    /* Confidence Indicators */
    .confidence-high {
        color: #4ade80;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(74, 222, 128, 0.5);
    }
    
    .confidence-medium {
        color: #fbbf24;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(251, 191, 36, 0.5);
    }
    
    .confidence-low {
        color: #f87171;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(248, 113, 113, 0.5);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: #0a0a0a !important;
        border-right: 1px solid rgba(120, 119, 198, 0.2) !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: #0a0a0a !important;
    }
    
    /* Info Cards in Sidebar */
    .info-card {
        background: rgba(26, 26, 26, 0.6);
        border: 1px solid rgba(120, 119, 198, 0.2);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .info-label {
        font-size: 0.7rem;
        color: #7877c6;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .info-value {
        font-size: 0.9rem;
        color: #e0e0e0;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .info-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(120, 119, 198, 0.1);
    }
    
    .info-item:last-child {
        border-bottom: none;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 700;
        background: rgba(74, 222, 128, 0.2);
        color: #4ade80;
        border: 1px solid rgba(74, 222, 128, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* History Item */
    .history-item {
        background: rgba(26, 26, 26, 0.4);
        border: 1px solid rgba(120, 119, 198, 0.15);
        border-left: 3px solid #7877c6;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        transition: all 0.3s;
    }
    
    .history-item:hover {
        background: rgba(26, 26, 26, 0.6);
        border-color: rgba(120, 119, 198, 0.3);
        transform: translateX(4px);
    }
    
    .history-number {
        display: inline-block;
        background: linear-gradient(135deg, #7877c6 0%, #4a90e2 100%);
        color: #ffffff;
        font-weight: 700;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-size: 0.75rem;
        margin-right: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .history-question {
        font-weight: 600;
        color: #e0e0e0;
        margin-bottom: 0.75rem;
        font-size: 0.95rem;
    }
    
    .history-answer {
        color: #a0a0a0;
        font-size: 0.85rem;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    
    .history-meta {
        display: flex;
        gap: 1.5rem;
        font-size: 0.75rem;
        color: #666666;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .history-meta-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(26, 26, 26, 0.6) !important;
        border: 1px solid rgba(120, 119, 198, 0.2) !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        color: #e0e0e0 !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(26, 26, 26, 0.8) !important;
        border-color: rgba(120, 119, 198, 0.3) !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #7877c6 0%, #4a90e2 100%) !important;
    }
    
    /* Stats Summary */
    .stats-summary {
        background: linear-gradient(135deg, rgba(120, 119, 198, 0.1) 0%, rgba(74, 144, 226, 0.1) 100%);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid rgba(120, 119, 198, 0.2);
        margin-bottom: 1.5rem;
        font-size: 0.9rem;
        color: #a0a0a0;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Text Area */
    .stTextArea textarea {
        background: rgba(10, 10, 10, 0.8) !important;
        border: 1px solid rgba(120, 119, 198, 0.2) !important;
        color: #a0a0a0 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: rgba(120, 119, 198, 0.2) !important;
    }
    
    .stSlider > div > div > div > div {
        background: #7877c6 !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 3rem 0 2rem 0;
        color: #4a4a4a;
        font-size: 0.8rem;
        margin-top: 4rem;
        border-top: 1px solid rgba(120, 119, 198, 0.1);
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Divider Line */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, 
            transparent 0%, 
            rgba(120, 119, 198, 0.3) 50%, 
            transparent 100%);
        margin: 1rem 0;
    }
    
    /* Loading Animation */
    .stSpinner > div {
        border-color: #7877c6 transparent transparent transparent !important;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a0a0a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #7877c6;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #4a90e2;
    }
    
    /* Section Gap */
    .section-gap {
        margin-top: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- ENVIRONMENT SETUP ---
load_dotenv()

# Try Streamlit secrets first, then fall back to .env
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    #st.write(" API Key loaded from Streamlit secrets")  # Debug line
except Exception as e:
    st.write(f"Secrets error: {e}")  # Debug line
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#st.write(f"API Key present: {GROQ_API_KEY is not None}")  # Debug line
#st.write(f"API Key length: {len(GROQ_API_KEY) if GROQ_API_KEY else 0}")  # Debug line

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please add it in Streamlit Cloud Settings → Secrets")
    st.stop()

# --- CONFIGURATION ---
PDF_PATH = Path("./data/NIPS-2017-attention-is-all-you-need-Paper.pdf")

# --- SESSION STATE INITIALIZATION ---
if 'rag_settings' not in st.session_state:
    st.session_state.rag_settings = {
        "top_k": 3,
        "temperature": 0.2,
        "max_tokens": 512
    }
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
if 'question_input' not in st.session_state:
    st.session_state.question_input = ''
if 'submit_requested' not in st.session_state:
    st.session_state.submit_requested = False
if 'last_response' not in st.session_state:
    st.session_state.last_response = None
if 'error_msg' not in st.session_state:
    st.session_state.error_msg = None
if 'success_msg' not in st.session_state:
    st.session_state.success_msg = None
if 'error_details' not in st.session_state:
    st.session_state.error_details = None

# --- CACHED RAG PIPELINE INITIALIZATION ---
@st.cache_resource
def initialize_rag_system() -> RAGPipeline:
    pipeline = RAGPipeline(groq_api_key=GROQ_API_KEY)
    
    if not PDF_PATH.exists():
        st.error(f"PDF file not found at: {PDF_PATH}")
        st.stop()
    
    chunk_count = pipeline.index_pdf(PDF_PATH)
    st.session_state.total_chunks = chunk_count
    st.session_state.system_initialized = True
    
    return pipeline

# --- UTILITY FUNCTIONS ---
def get_confidence_label(score: float) -> Tuple[str, str]:
    if score >= 0.75:
        return ("HIGH", "confidence-high")
    elif score >= 0.5:
        return ("MEDIUM", "confidence-medium")
    else:
        return ("LOW", "confidence-low")

def create_confidence_gauge(confidence: float) -> go.Figure:
    """Create a visual gauge for confidence score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence", 'font': {'size': 16, 'color': '#a0a0a0'}},
        number={'suffix': "%", 'font': {'size': 32, 'color': '#7877c6'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#7877c6"},
            'bar': {'color': "#7877c6"},
            'bgcolor': "rgba(26, 26, 26, 0.6)",
            'borderwidth': 2,
            'bordercolor': "rgba(120, 119, 198, 0.3)",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(248, 113, 113, 0.2)'},
                {'range': [50, 75], 'color': 'rgba(251, 191, 36, 0.2)'},
                {'range': [75, 100], 'color': 'rgba(74, 222, 128, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "#4a90e2", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#a0a0a0", 'family': "Space Grotesk"},
        height=200,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_history_chart() -> Optional[go.Figure]:
    """Create a line chart showing confidence trends over time."""
    if len(st.session_state.conversation_history) < 2:
        return None
    
    confidences = [item["confidence"] * 100 for item in st.session_state.conversation_history]
    queries = [f"Q{i+1}" for i in range(len(confidences))]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=queries,
        y=confidences,
        mode='lines+markers',
        name='Confidence',
        line=dict(color='#7877c6', width=3),
        marker=dict(size=8, color='#4a90e2', line=dict(color='#7877c6', width=2)),
        fill='tozeroy',
        fillcolor='rgba(120, 119, 198, 0.1)'
    ))
    
    fig.update_layout(
        title="Query Confidence Trend",
        xaxis_title="Query Number",
        yaxis_title="Confidence %",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26, 26, 26, 0.4)',
        font={'color': "#a0a0a0", 'family': "Space Grotesk"},
        height=250,
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(gridcolor='rgba(120, 119, 198, 0.1)', showgrid=True),
        yaxis=dict(gridcolor='rgba(120, 119, 198, 0.1)', showgrid=True, range=[0, 100])
    )
    
    return fig

def format_response(response: Dict[str, Any]) -> None:
    st.markdown('<p class="section-title">Generated Answer</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="answer-box">{response["answer"]}</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            pages_str = ", ".join(map(str, response["pages"])) if response.get("pages") else "N/A"
            st.metric("Pages", pages_str)
        
        with metric_col2:
            st.metric("Sources", response.get("sources_count", 0))
        
        with metric_col3:
            confidence = response.get("confidence_score", 0)
            label, css_class = get_confidence_label(confidence)
            st.metric("Score", f"{confidence:.0%}")
            st.markdown(f'<p class="{css_class}" style="text-align: center; margin-top: -0.5rem; font-size: 0.75rem;">{label}</p>', unsafe_allow_html=True)
        
        with metric_col4:
            st.metric("Time", f"{response.get('response_time', 0):.2f}s")
    
    with col2:
        gauge_fig = create_confidence_gauge(response.get("confidence_score", 0))
        st.plotly_chart(gauge_fig, use_container_width=True)
    
    with st.expander("View Retrieved Context", expanded=False):
        st.text_area(
            label="Context",
            value=response["retrieved_context"],
            height=300,
            disabled=True,
            label_visibility="collapsed"
        )

def add_to_history(question: str, response: Dict[str, Any]) -> None:
    st.session_state.conversation_history.append({
        "question": question,
        "answer": response["answer"],
        "confidence": response.get("confidence_score", 0),
        "pages": response.get("pages", []),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })

def calculate_avg_confidence() -> float:
    if not st.session_state.conversation_history:
        return 0.0
    confidences = [item["confidence"] for item in st.session_state.conversation_history]
    return sum(confidences) / len(confidences)

def export_history() -> None:
    if st.session_state.conversation_history:
        history_json = json.dumps(st.session_state.conversation_history, indent=4)
        st.download_button(
            label="Export History",
            data=history_json,
            file_name="conversation_history.json",
            mime="application/json",
            use_container_width=True
        )

# --- MAIN APPLICATION ---
def main():
    rag_system = initialize_rag_system()
    stats = rag_system.get_collection_stats()
    
    # Header
    st.markdown("""
        <div class="header-container">
            <h1 class="header-title">Transformer Paper Q&A</h1>
            <p class="header-subtitle">
                Advanced RAG-Powered Intelligence System • Attention Is All You Need (Vaswani et al., 2017)
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<p class="section-title">System Monitor</p>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<div class="info-label">Configuration</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-item"><span>Model</span><span class="info-value">llama-3.1-8b</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="info-item"><span>Embeddings</span><span class="info-value">BGE-base</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="info-item"><span>Database</span><span class="info-value">ChromaDB</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="info-item"><span>Status</span><span class="status-badge">Active</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<div class="info-label">Document Index</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-item"><span>Detail Chunks</span><span class="info-value">{stats.get("detail_chunks", 0)}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-item"><span>Summary Chunks</span><span class="info-value">{stats.get("summary_chunks", 0)}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-item"><span>Total Chunks</span><span class="info-value">{stats.get("total_chunks", 0)}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<div class="info-label">Session Analytics</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-item"><span>Queries</span><span class="info-value">{st.session_state.query_count}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-item"><span>Avg Confidence</span><span class="info-value">{calculate_avg_confidence():.0%}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-item"><span>History</span><span class="info-value">{len(st.session_state.conversation_history)}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        st.markdown('<p class="section-title">Advanced Controls</p>', unsafe_allow_html=True)
        with st.expander("RAG Parameters", expanded=False):
            top_k = st.slider(
                "Retrieval Sources", 
                1, 
                5, 
                st.session_state.rag_settings["top_k"],
                help="How many chunks of text are retrieved from the knowledge base."
            )
            temperature = st.slider(
                "Temperature", 
                0.0, 
                1.0, 
                st.session_state.rag_settings["temperature"], 
                0.1,
                help="Controls randomness in generation (lower = focused, higher = creative)."
            )
            max_tokens = st.slider(
                "Max Tokens", 
                256, 
                1024, 
                st.session_state.rag_settings["max_tokens"], 
                64,
                help="Maximum length of the model's response."
            )
            
            if st.button("Apply Parameters", use_container_width=True):
                st.session_state.rag_settings = {
                    "top_k": top_k,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        st.markdown('<p class="section-title">Session Actions</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reset History", use_container_width=True):
                st.session_state.conversation_history = []
                st.session_state.query_count = 0
                st.rerun()
        
        with col2:
            if st.button("Reset Database", use_container_width=True):
                rag_system.clear_collection()
                rag_system.index_pdf(PDF_PATH)
                st.success("Database reset")
                time.sleep(1)
                st.rerun()
        
        export_history()
    
    # Main Content Area
    col1, col2 = st.columns([2.5, 1])
    
    with col1:
        # Query Section
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Query Interface</p>', unsafe_allow_html=True)
        
        with st.form(key="query_form"):
            question = st.text_input(
                label="Enter your question",
                value=st.session_state.question_input,
                placeholder="What is the main innovation of the Transformer architecture?",
                label_visibility="collapsed"
            )
            submit_button = st.form_submit_button("Execute Query", use_container_width=True)
        
        if submit_button:
            if not question.strip():
                st.warning("Please enter a valid question")
            else:
                st.session_state.question_input = question
                st.session_state.submit_requested = True
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process Query
        if st.session_state.submit_requested:
            user_question = st.session_state.question_input
            with st.spinner("Processing query..."):
                try:
                    start_time = time.time()
                    response = rag_system.answer_question(
                        user_question,
                        top_k=st.session_state.rag_settings["top_k"],
                        temperature=st.session_state.rag_settings["temperature"],
                        max_tokens=st.session_state.rag_settings["max_tokens"]
                    )
                    response["response_time"] = time.time() - start_time
                    
                    if response.get("sources_count", 0) == 0:
                        st.session_state.error_msg = "No relevant information found. Try rephrasing."
                    else:
                        st.session_state.query_count += 1
                        add_to_history(user_question, response)
                        st.session_state.last_response = response
                        st.session_state.success_msg = "Query executed successfully"
                    
                except Exception as e:
                    st.session_state.error_msg = "Query execution failed."
                    st.session_state.error_details = traceback.format_exc()
            
            st.session_state.question_input = ''
            st.session_state.submit_requested = False
            st.rerun()
        
        # Display Response
        if st.session_state.last_response:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            format_response(st.session_state.last_response)
            st.session_state.last_response = None
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.session_state.error_msg:
            st.error(st.session_state.error_msg)
            if st.session_state.error_details:
                with st.expander("Error Details"):
                    st.code(st.session_state.error_details)
            st.session_state.error_msg = None
            st.session_state.error_details = None
        
        if st.session_state.success_msg:
            st.success(st.session_state.success_msg)
            st.session_state.success_msg = None
        
        # History with Trend Chart
        if st.session_state.conversation_history:
            st.markdown('<div class="glass-card section-gap">', unsafe_allow_html=True)
            st.markdown('<p class="section-title">Query History</p>', unsafe_allow_html=True)
            
            avg_conf = calculate_avg_confidence()
            st.markdown(f'<div class="stats-summary">SESSION: {st.session_state.query_count} queries executed | AVG CONFIDENCE: {avg_conf:.0%}</div>', unsafe_allow_html=True)
            
            # Confidence trend chart
            chart = create_history_chart()
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            
            for idx, item in enumerate(reversed(st.session_state.conversation_history[-10:]), 1):
                q_num = len(st.session_state.conversation_history) - idx + 1
                st.markdown('<div class="history-item">', unsafe_allow_html=True)
                st.markdown(f'<span class="history-number">Q{q_num}</span>', unsafe_allow_html=True)
                st.markdown(f'<div class="history-question">{item["question"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="history-answer">{item["answer"]}</div>', unsafe_allow_html=True)
                
                pages_display = ", ".join(map(str, item["pages"])) if item["pages"] else "N/A"
                conf_label, conf_class = get_confidence_label(item["confidence"])
                
                st.markdown(f'''
                <div class="history-meta">
                    <div class="history-meta-item">PAGES: {pages_display}</div>
                    <div class="history-meta-item">CONF: <span class="{conf_class}">{item["confidence"]:.0%}</span></div>
                    <div class="history-meta-item">TIME: {item["timestamp"]}</div>
                </div>
                ''', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Example Questions
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Quick Queries</p>', unsafe_allow_html=True)
        
        examples = [
            "What is the main contribution of this paper?",
            "Explain self-attention mechanism",
            "What are the key advantages?",
            "How does multi-head attention work?",
            "What datasets were used?",
            "Define positional encoding",
            "Describe encoder-decoder structure",
            "Explain computational complexity"
        ]
        
        for i, q in enumerate(examples, 1):
            if st.button(q, key=f"ex_{i}", use_container_width=True):
                st.session_state.question_input = q
                st.session_state.submit_requested = True
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Real-time Stats Visualization
        if st.session_state.query_count > 0:
            st.markdown('<div class="glass-card section-gap">', unsafe_allow_html=True)
            st.markdown('<p class="section-title">Performance Metrics</p>', unsafe_allow_html=True)
            
            if st.session_state.conversation_history:
                confidences = [item["confidence"] for item in st.session_state.conversation_history]
                high_conf = sum(1 for c in confidences if c >= 0.75)
                med_conf = sum(1 for c in confidences if 0.5 <= c < 0.75)
                low_conf = sum(1 for c in confidences if c < 0.5)
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=['High', 'Medium', 'Low'],
                        y=[high_conf, med_conf, low_conf],
                        marker=dict(
                            color=['#4ade80', '#fbbf24', '#f87171'],
                            line=dict(color='rgba(120, 119, 198, 0.3)', width=2)
                        )
                    )
                ])
                
                fig.update_layout(
                    title="Confidence Distribution",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(26, 26, 26, 0.4)',
                    font={'color': "#a0a0a0", 'family': "Space Grotesk"},
                    height=250,
                    margin=dict(l=40, r=40, t=60, b=40),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(gridcolor='rgba(120, 119, 198, 0.1)', showgrid=True, title="Count")
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
        <div class="footer">
            <p>POWERED BY RETRIEVAL-AUGMENTED GENERATION TECHNOLOGY</p>
            <p>STREAMLIT • GROQ • CHROMADB • LLAMA 3.1</p>
            <p style="margin-top: 1rem; font-size: 0.7rem;">
                RESEARCH PAPER: "ATTENTION IS ALL YOU NEED" • VASWANI ET AL. • NIPS 2017
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
