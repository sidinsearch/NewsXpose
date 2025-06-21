import streamlit as st
import warnings
import numpy as np
import re
import pickle
import requests
from datetime import datetime
import whois
from urllib.parse import urlparse
import tldextract
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from scraper import scrape_article, clean_text, is_youtube_link
import textwrap
import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yt_dlp
from bs4 import BeautifulSoup
import os
from articlefinder import find_related_articles
from llm import analyze_article as get_llm_analysis
from scraper import transcribe_youtube_video, get_youtube_thumbnail
from model_utils import safe_load_model, is_model_compatible

# Suppress warnings about incompatible dtype in node arrays
warnings.filterwarnings("ignore", message=".*node array from the pickle has an incompatible dtype.*")

# Page configuration
st.set_page_config(
    page_title="NewsXpose - AI-Powered Fake News Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Base styling */
    :root {
        --primary-color: #6c5ce7;
        --primary-dark: #5649c0;
        --secondary-color: #a29bfe;
        --text-color: #e2e2e2;
        --light-text: #b8b8b8;
        --background-color: #1e1e1e;
        --card-background: #2d2d2d;
        --border-color: #444444;
        --success-color: #00b894;
        --warning-color: #fdcb6e;
        --danger-color: #ff7675;
        --info-color: #74b9ff;
        --dark-bg: #121212;
        --dark-card: #1a1a1a;
    }
    
    /* Override Streamlit's default theme */
    .stApp {
        background-color: var(--dark-bg);
    }
    
    /* Main container styling */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Custom font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--text-color);
    }
    
    /* Streamlit default element overrides */
    .stTextInput > div > div > input {
        color: var(--text-color);
        background-color: var(--card-background);
        border-color: var(--border-color);
    }
    
    .stTextArea > div > div > textarea {
        color: var(--text-color);
        background-color: var(--card-background);
        border-color: var(--border-color);
    }
    
    .stSelectbox > div > div > div {
        color: var(--text-color);
        background-color: var(--card-background);
    }
    
    /* Radio buttons */
    .stRadio > div {
        color: var(--text-color) !important;
    }
    
    /* Make sure all text in markdown is visible */
    p, h1, h2, h3, h4, h5, h6, li, span {
        color: var(--text-color) !important;
    }
    
    a {
        color: var(--secondary-color) !important;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .main-title {
        color: white !important;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .main-subtitle {
        color: rgba(255,255,255,0.9) !important;
        font-size: 1.2rem;
        text-align: center;
        font-weight: 400;
    }
    
    /* Input section styling */
    .input-section {
        background: var(--card-background);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
        margin-bottom: 2rem;
        border: 1px solid var(--border-color);
    }
    
    .input-section h3 {
        color: var(--text-color) !important;
        margin-bottom: 1.5rem;
    }
    
    /* Results section styling */
    .results-section {
        background: var(--card-background);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
        margin-bottom: 1.5rem;
        border: 1px solid var(--border-color);
    }
    
    .results-section h2, .results-section h3 {
        color: var(--text-color) !important;
        margin-bottom: 1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #2d3436 0%, #636e72 100%);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 3px 10px rgba(0,0,0,0.2);
    }
    
    .metric-title {
        font-size: 0.9rem;
        color: var(--light-text) !important;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--text-color) !important;
    }
    
    /* Status indicators */
    .status-real {
        background: linear-gradient(135deg, #0984e3 0%, #00cec9 100%);
        color: white !important;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(9, 132, 227, 0.3);
    }
    
    .status-fake {
        background: linear-gradient(135deg, #d63031 0%, #e17055 100%);
        color: white !important;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(214, 48, 49, 0.3);
    }
    
    /* Loading spinner */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    /* Component breakdown styling */
    .component-breakdown {
        background: var(--dark-card);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid var(--primary-color);
    }
    
    .component-breakdown p {
        color: var(--text-color) !important;
        margin-bottom: 0.5rem;
    }
    
    .component-breakdown strong {
        font-weight: 600;
        color: var(--text-color) !important;
    }
    
    /* Article content styling */
    .article-content {
        background: var(--dark-card);
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid var(--border-color);
        margin: 1rem 0;
        line-height: 1.6;
        color: var(--text-color) !important;
    }
    
    .article-content p {
        color: var(--text-color) !important;
        margin-bottom: 1rem;
    }
    
    /* Domain info styling */
    .domain-info {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: var(--text-color) !important;
        border: 1px solid var(--border-color);
    }
    
    .domain-info h3 {
        color: var(--text-color) !important;
        margin-bottom: 1rem;
    }
    
    .domain-info p, .domain-info strong {
        color: var(--text-color) !important;
    }
    
    /* Trust score styling */
    .trust-high {
        color: var(--success-color) !important;
        font-weight: 600;
    }
    
    .trust-medium {
        color: var(--warning-color) !important;
        font-weight: 600;
    }
    
    .trust-low {
        color: var(--danger-color) !important;
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white !important;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(108, 92, 231, 0.4);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: var(--dark-bg);
        border-right: 1px solid var(--border-color);
    }
    
    .sidebar h3 {
        color: var(--text-color) !important;
        font-weight: 600;
    }
    
    .sidebar p, .sidebar li {
        color: var(--text-color) !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Expander styling */
    .streamlit-expanderHeader {
        color: var(--text-color) !important;
        font-weight: 600;
        background-color: var(--dark-card);
    }
    
    .streamlit-expanderContent {
        background-color: var(--dark-card);
        border-color: var(--border-color);
    }
    
    /* Link styling */
    a {
        color: var(--secondary-color) !important;
        text-decoration: none;
    }
    
    a:hover {
        text-decoration: underline;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background-color: var(--primary-color);
    }
    
    /* X highlight styling */
    .x-highlight {
        color: #00b894;
        font-weight: 800;
        text-shadow: 0 0 10px rgba(0, 184, 148, 0.7);
        font-style: italic;
        font-size: 110%;
    }
    
    /* Creator link styling */
    .creator-link {
        display: inline-block;
        color: var(--text-color) !important;
        text-decoration: none;
        transition: all 0.3s ease;
        padding: 5px 10px;
        border-radius: 5px;
        background-color: rgba(108, 92, 231, 0.1);
        border: 1px solid rgba(108, 92, 231, 0.3);
        margin-top: 10px;
    }
    
    .creator-link:hover {
        background-color: rgba(108, 92, 231, 0.2);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        text-decoration: none !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }
        .main-subtitle {
            font-size: 1rem;
        }
        .input-section, .results-section {
            padding: 1.5rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_models_and_data():
    """Load all models and data with caching for better performance."""
    try:
        # Check for model files in multiple locations
        import os
        
        # Define possible model paths
        model_paths = [
            '/app/models/ensemble_fake_news_detector.pkl',  # Docker container path
            'models/ensemble_fake_news_detector.pkl',       # Subdirectory path
            'ensemble_fake_news_detector.pkl'               # Root directory path
        ]
        
        # Find the first existing model path
        ensemble_path = None
        for path in model_paths:
            if os.path.exists(path):
                ensemble_path = path
                st.success(f"Found model at: {path}")
                break
        
        if not ensemble_path:
            st.error("Model file not found in any of the expected locations")
            st.info(f"Current working directory: {os.getcwd()}")
            st.info(f"Files in directory: {os.listdir('.')}")
            
            # Check if models directory exists
            if os.path.exists('/app/models'):
                st.info(f"Files in /app/models: {os.listdir('/app/models')}")
            elif os.path.exists('models'):
                st.info(f"Files in models: {os.listdir('models')}")
                
            return None, None, None, set()
        
        # Load ensemble model with detailed error handling
        try:
            st.info(f"Loading ensemble model from: {ensemble_path}")
            ensemble_model_data = safe_load_model(ensemble_path)
            if ensemble_model_data:
                ensemble_model, vector = ensemble_model_data
                st.success("Successfully loaded ensemble model")
            else:
                st.error(f"Failed to load ensemble model from {ensemble_path}")
                ensemble_model, vector = None, None
        except Exception as model_error:
            st.error(f"Error loading ensemble model: {str(model_error)}")
            ensemble_model, vector = None, None
        
        # Define possible image model paths
        image_model_paths = [
            '/app/models/image-model.pkl',  # Docker container path
            'models/image-model.pkl',       # Subdirectory path
            'image-model.pkl'               # Root directory path
        ]
        
        # Find the first existing image model path
        image_model_path = None
        for path in image_model_paths:
            if os.path.exists(path):
                image_model_path = path
                st.success(f"Found image model at: {path}")
                break
        
        # Load image model
        try:
            if image_model_path:
                st.info(f"Loading image model from: {image_model_path}")
                image_model = safe_load_model(image_model_path)
                if image_model:
                    st.success("Successfully loaded image model")
            else:
                st.error("Image model file not found in any of the expected locations")
                image_model = None
        except Exception as img_error:
            st.error(f"Error loading image model: {str(img_error)}")
            image_model = None
        
        # Load trustworthy domains
        trustworthy_domains = set()
        try:
            with open('domains.txt', 'r') as file:
                trustworthy_domains = {line.strip().lower() for line in file if line.strip()}
        except Exception as domain_error:
            st.warning(f"Error loading domains: {str(domain_error)}")
        
        # Download NLTK data
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
        except Exception as nltk_error:
            st.warning(f"Error downloading NLTK data: {str(nltk_error)}")
        
        return ensemble_model, vector, image_model, trustworthy_domains
    except Exception as e:
        st.error(f"Error in load_models_and_data: {str(e)}")
        return None, None, None, set()

# Initialize models
ensemble_model, vector, image_model, TRUSTWORTHY_DOMAINS = load_models_and_data()
STOPWORDS = set(stopwords.words('english'))
ps = PorterStemmer()

def create_header():
    """Create the main header section."""
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🛡️ News<span class="x-highlight">X</span>pose</h1>
        <p class="main-subtitle">AI-Powered Fake News Detection & Analysis Platform</p>
        <div style="text-align:center; margin-top:15px;">
            <span style="background:rgba(0,0,0,0.3); color:white; padding:5px 15px; border-radius:20px; font-size:0.8rem; margin:0 5px; border:1px solid rgba(255,255,255,0.2);">
                Text Analysis
            </span>
            <span style="background:rgba(0,0,0,0.3); color:white; padding:5px 15px; border-radius:20px; font-size:0.8rem; margin:0 5px; border:1px solid rgba(255,255,255,0.2);">
                Image Detection
            </span>
            <span style="background:rgba(0,0,0,0.3); color:white; padding:5px 15px; border-radius:20px; font-size:0.8rem; margin:0 5px; border:1px solid rgba(255,255,255,0.2);">
                Domain Verification
            </span>
            <span style="background:rgba(0,0,0,0.3); color:white; padding:5px 15px; border-radius:20px; font-size:0.8rem; margin:0 5px; border:1px solid rgba(255,255,255,0.2);">
                AI Analysis
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_sidebar():
    """Create an informative sidebar."""
    with st.sidebar:
        st.markdown("<h3 style='color:#e2e2e2; font-weight:600;'>📊 Analysis Components</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div style='color:#e2e2e2; background-color:#2d2d2d; padding:15px; border-radius:8px; border-left:3px solid #6c5ce7;'>
        <p><strong>News<span class="x-highlight">X</span>pose analyzes multiple factors:</strong></p>
        
        <p>🔍 <strong>Text Analysis</strong></p>
        <ul>
            <li>Content credibility</li>
            <li>Language patterns</li>
            <li>Factual consistency</li>
        </ul>
        
        <p>🖼️ <strong>Image Analysis</strong></p>
        <ul>
            <li>AI-generated detection</li>
            <li>Image authenticity</li>
        </ul>
        
        <p>🌐 <strong>Domain Trust</strong></p>
        <ul>
            <li>Domain reputation</li>
            <li>Registration history</li>
            <li>Source credibility</li>
        </ul>
        
        <p>🤖 <strong>LLM Analysis</strong></p>
        <ul>
            <li>Advanced reasoning</li>
            <li>Context understanding</li>
            <li>Fact verification</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<hr style='margin:20px 0; border-color:#444444;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:#e2e2e2; font-weight:600;'>📋 Supported Formats</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div style='color:#e2e2e2; background-color:#2d2d2d; padding:15px; border-radius:8px; border-left:3px solid #6c5ce7;'>
        <ul>
            <li>🔗 News article URLs</li>
            <li>📺 YouTube videos</li>
            <li>📝 Raw text content</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<hr style='margin:20px 0; border-color:#444444;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:#e2e2e2; font-weight:600;'>⚠️ Disclaimer</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div style='color:#e2e2e2; background-color:#2d2d2d; padding:15px; border-radius:8px; border-left:3px solid #d63031;'>
        <p>This tool provides AI-assisted analysis. 
        Always verify information from multiple 
        reliable sources before drawing conclusions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add creator credit at the bottom
        st.markdown("<div style='position:fixed; bottom:20px; left:20px; right:20px;'>", unsafe_allow_html=True)
        st.markdown("""
        <a href="https://github.com/sidinsearch" target="_blank" class="creator-link">
            <div style="display:flex; align-items:center; justify-content:center;">
                <div style="margin-right:10px;">💻</div>
                <div>Made By Siddharth Shinde</div>
            </div>
        </a>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

def is_valid_url(url):
    """Check if URL is valid."""
    try:
        result = urlparse(url)
        return all([result.scheme in ['http', 'https'], result.netloc])
    except ValueError:
        return False

def extract_domain(url):
    """Extract the main domain from a URL."""
    try:
        extracted = tldextract.extract(url)
        domain = f"{extracted.domain}.{extracted.suffix}"
        return domain
    except Exception:
        return None

def get_domain_info(url):
    """Get domain information using whois."""
    try:
        domain = extract_domain(url)
        if not domain:
            return None

        w = whois.whois(domain)
        creation_date = w.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]

        domain_age = None
        if creation_date:
            domain_age = (datetime.now() - creation_date).days

        # Calculate trust score
        trust_score = 0.1
        if domain in TRUSTWORTHY_DOMAINS:
            trust_score = 1.0
        elif domain_age:
            if domain_age > 3650:
                trust_score = 1.0
            elif domain_age > 1825:
                trust_score = 0.8
            elif domain_age > 365:
                trust_score = 0.6
            elif domain_age > 180:
                trust_score = 0.4
            elif domain_age > 90:
                trust_score = 0.2

        return {
            'domain': domain,
            'registrar': w.registrar,
            'creation_date': creation_date,
            'domain_age': domain_age,
            'trust_score': trust_score,
            'country': w.country
        }
    except Exception:
        return None

def preprocess_image(image_url):
    """Preprocess image for model prediction."""
    try:
        response = requests.get(image_url, timeout=10)
        img_array = np.array(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.resize(img, (32, 32))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception:
        return None

def predict_image(model, image):
    """Predict if image is real or AI-generated."""
    if model is None or image is None:
        return "Real"
    prediction = model.predict(image)
    return "Real" if prediction < 0.5 else "AI-generated"

def stemming(content):
    """Apply stemming to text content."""
    content = re.sub(r'[^a-zA-Z]', ' ', content)
    content = content.lower().split()
    stemmed_content = [ps.stem(word) for word in content if word not in STOPWORDS]
    return ' '.join(stemmed_content)

def prediction(input_text):
    """Make prediction on text content."""
    if ensemble_model is None or vector is None:
        return 0, [0.5, 0.5]
    input_data = vector.transform([stemming(input_text)])
    prediction = ensemble_model.predict(input_data)
    probabilities = ensemble_model.predict_proba(input_data)[0]
    return prediction[0], probabilities

def calculate_combined_prediction(text_probs, image_result, domain_trust_score, llm_verdict):
    """Calculate combined prediction from all components."""
    
    TEXT_WEIGHT = 0.50
    IMAGE_WEIGHT = 0.15
    DOMAIN_WEIGHT = 0.15
    LLM_WEIGHT = 0.20
    
    text_real_contrib = text_probs[0] * TEXT_WEIGHT * 100
    text_fake_contrib = text_probs[1] * TEXT_WEIGHT * 100
    
    image_real_contrib = (1.0 if image_result == "Real" else 0.0) * IMAGE_WEIGHT * 100
    image_fake_contrib = (1.0 if image_result == "AI-generated" else 0.0) * IMAGE_WEIGHT * 100
    
    domain_real_contrib = domain_trust_score * DOMAIN_WEIGHT * 100
    domain_fake_contrib = (1.0 - domain_trust_score) * DOMAIN_WEIGHT * 100
    
    # Handle LLM verdict - default to 50/50 if unknown
    if llm_verdict == "real":
        llm_real_contrib = 1.0 * LLM_WEIGHT * 100
        llm_fake_contrib = 0.0 * LLM_WEIGHT * 100
    elif llm_verdict == "fake":
        llm_real_contrib = 0.0 * LLM_WEIGHT * 100
        llm_fake_contrib = 1.0 * LLM_WEIGHT * 100
    else:  # unknown or any other value
        llm_real_contrib = 0.5 * LLM_WEIGHT * 100
        llm_fake_contrib = 0.5 * LLM_WEIGHT * 100
    
    final_real_prob = text_real_contrib + image_real_contrib + domain_real_contrib + llm_real_contrib
    final_fake_prob = text_fake_contrib + image_fake_contrib + domain_fake_contrib + llm_fake_contrib
    
    total = final_real_prob + final_fake_prob
    normalize_factor = 100 / total if total > 0 else 1
    
    return {
        'final_real_prob': final_real_prob * normalize_factor,
        'final_fake_prob': final_fake_prob * normalize_factor,
        'contributions': {
            'text_real': text_real_contrib * normalize_factor,
            'text_fake': text_fake_contrib * normalize_factor,
            'image_real': image_real_contrib * normalize_factor,
            'image_fake': image_fake_contrib * normalize_factor,
            'domain_real': domain_real_contrib * normalize_factor,
            'domain_fake': domain_fake_contrib * normalize_factor,
            'llm_real': llm_real_contrib * normalize_factor,
            'llm_fake': llm_fake_contrib * normalize_factor
        }
    }

def create_modern_chart(combined_results):
    """Create a modern, interactive chart."""
    contributions = combined_results['contributions']
    
    # Create gauge chart for overall confidence
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = combined_results['final_real_prob'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Credibility Score", 'font': {'color': '#e2e2e2'}},
        delta = {'reference': 50, 'increasing': {'color': '#00b894'}, 'decreasing': {'color': '#ff7675'}},
        number = {'font': {'color': '#e2e2e2'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickfont': {'color': '#b8b8b8'}},
            'bar': {'color': "#6c5ce7"},
            'bgcolor': "#2d2d2d",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 25], 'color': "#2d3436"},
                {'range': [25, 50], 'color': "#636e72"},
                {'range': [50, 75], 'color': "#0984e3"},
                {'range': [75, 100], 'color': "#00b894"}
            ],
            'threshold': {
                'line': {'color': "#d63031", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig_gauge.update_layout(
        height=300, 
        font={'size': 16, 'color': '#e2e2e2'},
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a'
    )
    
    # Create component breakdown chart
    categories = ['Text Analysis', 'Image Analysis', 'Domain Trust', 'LLM Analysis']
    real_values = [
        contributions['text_real'],
        contributions['image_real'], 
        contributions['domain_real'],
        contributions['llm_real']
    ]
    fake_values = [
        contributions['text_fake'],
        contributions['image_fake'],
        contributions['domain_fake'], 
        contributions['llm_fake']
    ]
    
    fig_breakdown = go.Figure()
    fig_breakdown.add_trace(go.Bar(
        name='Real/Credible',
        x=categories,
        y=real_values,
        marker_color='#00b894'
    ))
    fig_breakdown.add_trace(go.Bar(
        name='Fake/Suspicious',
        x=categories,
        y=fake_values,
        marker_color='#d63031'
    ))
    
    fig_breakdown.update_layout(
        barmode='stack',
        title={'text': 'Component Analysis Breakdown', 'font': {'color': '#e2e2e2'}},
        xaxis_title={'text': 'Analysis Components', 'font': {'color': '#e2e2e2'}},
        yaxis_title={'text': 'Confidence (%)', 'font': {'color': '#e2e2e2'}},
        xaxis={'tickfont': {'color': '#b8b8b8'}},
        yaxis={'tickfont': {'color': '#b8b8b8'}},
        height=400,
        font={'size': 12, 'color': '#e2e2e2'},
        showlegend=True,
        legend={'font': {'color': '#e2e2e2'}},
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a'
    )
    
    return fig_gauge, fig_breakdown

def display_results(combined_results, llm_explanation, related_articles):
    """Display analysis results in a modern format."""
    st.markdown("## 📊 Analysis Results")
    
    # Overall verdict
    is_fake = combined_results['final_fake_prob'] > combined_results['final_real_prob']
    confidence = max(combined_results['final_fake_prob'], combined_results['final_real_prob'])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if is_fake:
            st.markdown(f"""
            <div class="status-fake">
                ⚠️ LIKELY FAKE/MISLEADING
                <br>Confidence: {confidence:.1f}%
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-real">
                ✅ LIKELY CREDIBLE
                <br>Confidence: {confidence:.1f}%
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Trust level indicator
        if confidence >= 80:
            trust_level = "High"
            trust_class = "trust-high"
        elif confidence >= 60:
            trust_level = "Medium"
            trust_class = "trust-medium"
        else:
            trust_level = "Low"
            trust_class = "trust-low"
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Trust Level</div>
            <div class="metric-value {trust_class}">{trust_level}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
    st.markdown("### 📈 Detailed Analysis")
    fig_gauge, fig_breakdown = create_modern_chart(combined_results)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_gauge, use_container_width=True)
    with col2:
        st.plotly_chart(fig_breakdown, use_container_width=True)
    
    # Component breakdown
    st.markdown("### 🔍 Component Breakdown")
    contributions = combined_results['contributions']
    
    components = [
        ("Text Analysis", contributions['text_real'], contributions['text_fake'], "50%"),
        ("Image Analysis", contributions['image_real'], contributions['image_fake'], "15%"),
        ("Domain Trust", contributions['domain_real'], contributions['domain_fake'], "15%"),
        ("LLM Analysis", contributions['llm_real'], contributions['llm_fake'], "20%")
    ]
    
    for name, real_val, fake_val, weight in components:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"<p><strong>{name}</strong> ({weight})</p>", unsafe_allow_html=True)
        with col2:
            real_class = "trust-high" if real_val > fake_val else ""
            st.markdown(f"<p class='{real_class}'>✅ {real_val:.1f}%</p>", unsafe_allow_html=True)
        with col3:
            fake_class = "trust-low" if fake_val > real_val else ""
            st.markdown(f"<p class='{fake_class}'>⚠️ {fake_val:.1f}%</p>", unsafe_allow_html=True)
        with col4:
            if real_val > fake_val:
                dominant = "Credible"
                dominant_class = "trust-high"
            elif fake_val > real_val:
                dominant = "Suspicious"
                dominant_class = "trust-low"
            else:
                dominant = "Neutral"
                dominant_class = "trust-medium"
            st.markdown(f"<p class='{dominant_class}'>→ {dominant}</p>", unsafe_allow_html=True)
    
    # LLM Explanation
    if llm_explanation:
        st.markdown("### 🤖 AI Analysis")
        # Format the LLM explanation with proper paragraphs
        formatted_explanation = llm_explanation.replace("\n", "<br>")
        st.markdown(f'<div class="article-content">{formatted_explanation}</div>', unsafe_allow_html=True)
    
    # Related Articles
    if related_articles:
        st.markdown("### 📰 Related Articles")
        for i, article in enumerate(related_articles[:5], 1):
            st.markdown(f"{i}. [{article['title']}]({article['url']})")

def display_domain_info(domain_info):
    """Display domain information in a modern format."""
    if not domain_info:
        return
    
    st.markdown("### 🌐 Domain Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"<p><strong>Domain:</strong> {domain_info['domain']}</p>", unsafe_allow_html=True)
        if domain_info['country']:
            st.markdown(f"<p><strong>Country:</strong> {domain_info['country']}</p>", unsafe_allow_html=True)
    
    with col2:
        if domain_info['domain_age']:
            years = domain_info['domain_age'] // 365
            st.markdown(f"<p><strong>Age:</strong> {years} years</p>", unsafe_allow_html=True)
        if domain_info['registrar']:
            registrar = domain_info['registrar']
            # Truncate long registrar names
            if registrar and len(registrar) > 30:
                registrar = registrar[:27] + "..."
            st.markdown(f"<p><strong>Registrar:</strong> {registrar}</p>", unsafe_allow_html=True)
    
    with col3:
        trust_score = domain_info['trust_score'] * 100
        if trust_score >= 70:
            trust_class = "trust-high"
            trust_label = "High"
        elif trust_score >= 40:
            trust_class = "trust-medium"
            trust_label = "Medium"
        else:
            trust_class = "trust-low"
            trust_label = "Low"
        
        st.markdown(f'<p><strong>Trust Score:</strong> <span class="{trust_class}">{trust_score:.0f}% ({trust_label})</span></p>', 
                   unsafe_allow_html=True)
    
def main():
    """Main application function."""
    load_custom_css()
    create_header()
    create_sidebar()
    
    # Input section
    st.markdown("### 📝 Enter Content for Analysis")
    
    # Input options
    input_type = st.radio(
        "Choose input type:",
        ["URL (News Article)", "YouTube Video", "Text Content"],
        horizontal=True
    )
    
    if input_type == "URL (News Article)":
        input_text = st.text_input(
            "Enter news article URL:",
            placeholder="https://example.com/news-article",
            help="Enter a valid news article URL starting with https://"
        )
    elif input_type == "YouTube Video":
        input_text = st.text_input(
            "Enter YouTube video URL:",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Enter a valid YouTube video URL"
        )
    else:
        input_text = st.text_area(
            "Enter text content:",
            height=200,
            placeholder="Paste the news article text here...",
            help="Enter the text content you want to analyze"
        )
    
    analyze_button = st.button("🔍 Analyze Content", type="primary")
    
    if analyze_button and input_text:
        # Initialize variables
        text_for_analysis = input_text
        image_result = "Real"
        domain_trust_score = 0.1
        llm_verdict = "unknown"
        llm_explanation = ""
        related_articles = []
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            is_youtube = is_youtube_link(input_text)
            
            # Determine the type of content and process accordingly
            content_type = None
            if is_valid_url(input_text) and not is_youtube:
                content_type = "article"
            elif is_youtube:
                content_type = "youtube"
            else:
                content_type = "text"
                
            # Process based on content type
            if content_type == "article":
                # Process news article
                status_text.text("📰 Scraping article...")
                progress_bar.progress(20)
                
                article_data = scrape_article(input_text)
                if article_data:
                    st.success("✅ Article scraped successfully!")
                    
                    # Display article info
                    st.markdown("### 📄 Article Information")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Title:** {article_data['title']}")
                        st.markdown(f"**Publisher:** {article_data['publisher']}")
                    with col2:
                        if article_data['publish_date']:
                            formatted_date = article_data['publish_date'].strftime("%Y-%m-%d")
                            st.markdown(f"**Published:** {formatted_date}")
                    
                    # Image analysis
                    if article_data['image_url']:
                        status_text.text("🖼️ Analyzing image...")
                        progress_bar.progress(40)
                        
                        image = preprocess_image(article_data['image_url'])
                        if image is not None:
                            image_result = predict_image(image_model, image)
                            
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.image(article_data['image_url'], caption="Article Image", use_column_width=True)
                            with col2:
                                if image_result == "Real":
                                    st.success("✅ Image appears authentic")
                                else:
                                    st.warning("⚠️ Image may be AI-generated")
                    
                    # Display article content
                    st.markdown("### 📖 Article Content")
                    with st.expander("View full article", expanded=False):
                        # Format the article text with proper paragraphs
                        formatted_text = article_data["text"].replace("\n", "<br>")
                        st.markdown(f'<div class="article-content">{formatted_text}</div>', 
                                  unsafe_allow_html=True)
                    
                    text_for_analysis = article_data['text']
                    
                    # Domain analysis
                    status_text.text("🌐 Analyzing domain...")
                    progress_bar.progress(60)
                    
                    domain_info = get_domain_info(input_text)
                    if domain_info:
                        display_domain_info(domain_info)
                        domain_trust_score = domain_info['trust_score']
                    
                    # LLM analysis
                    status_text.text("🤖 Running AI analysis...")
                    progress_bar.progress(80)
                    
                    try:
                        llm_verdict, llm_explanation = get_llm_analysis(article_data['text'], article_data['title'])
                    except Exception as e:
                        st.warning(f"LLM analysis unavailable: {str(e)}")
                        llm_verdict = "unknown"
                        llm_explanation = "AI analysis could not be completed."
                    
                    # Find related articles
                    try:
                        related_articles = find_related_articles(article_data['title'])
                    except Exception as e:
                        st.warning(f"Related articles search unavailable: {str(e)}")
                        related_articles = []
                    
                else:
                    st.error("❌ Failed to scrape the article. Please check the URL and try again.")
                    return
                    
            elif content_type == "youtube":
                # Process YouTube video
                status_text.text("📺 Processing YouTube video...")
                progress_bar.progress(40)
                
                try:
                    yt_title, yt_transcription = transcribe_youtube_video(input_text)
                    if yt_title and yt_transcription:
                        st.success("✅ YouTube video processed successfully!")
                        
                        st.markdown(f"### 📺 Video: {yt_title}")
                        
                        # Get thumbnail
                        yt_thumbnail = get_youtube_thumbnail(input_text)
                        if yt_thumbnail:
                            st.image(yt_thumbnail, caption="Video Thumbnail", width=400)
                        
                        # Display transcription
                        with st.expander("View full transcription", expanded=False):
                            # Format the transcription with proper paragraphs
                            formatted_transcription = yt_transcription.replace("\n", "<br>")
                            st.markdown(f'<div class="article-content">{formatted_transcription}</div>', 
                                      unsafe_allow_html=True)
                        
                        text_for_analysis = yt_transcription
                        
                        # LLM analysis
                        status_text.text("🤖 Running AI analysis...")
                        progress_bar.progress(80)
                        try:
                            llm_verdict, llm_explanation = get_llm_analysis(yt_transcription, yt_title)
                        except Exception as e:
                            st.warning(f"LLM analysis unavailable: {str(e)}")
                            llm_verdict = "unknown"
                            llm_explanation = "AI analysis could not be completed."
                        
                        try:
                            related_articles = find_related_articles(yt_title)
                        except Exception as e:
                            st.warning(f"Related articles search unavailable: {str(e)}")
                            related_articles = []
                    else:
                        st.error("❌ Failed to process the YouTube video.")
                        return
                except Exception as e:
                    st.error(f"❌ Error processing YouTube video: {str(e)}")
                    return

            elif content_type == "text":
                # Raw text input
                status_text.text("🧠 Analyzing text...")
                progress_bar.progress(40)

                text_for_analysis = clean_text(input_text)

                status_text.text("🤖 Running AI analysis...")
                progress_bar.progress(80)
                try:
                    llm_verdict, llm_explanation = get_llm_analysis(text_for_analysis, "User Provided Content")
                except Exception as e:
                    st.warning(f"LLM analysis unavailable: {str(e)}")
                    llm_verdict = "unknown"
                    llm_explanation = "AI analysis could not be completed."

        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            return
            
        # Run prediction
        status_text.text("📊 Calculating final prediction...")
        progress_bar.progress(90)

        pred, probs = prediction(text_for_analysis)

        combined_results = calculate_combined_prediction(probs, image_result, domain_trust_score, llm_verdict)

        progress_bar.progress(100)
        status_text.text("✅ Analysis complete.")

        # Display final results
        display_results(combined_results, llm_explanation, related_articles)

# Run the app
if __name__ == "__main__":
    main()
