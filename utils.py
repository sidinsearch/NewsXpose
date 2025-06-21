import streamlit as st
import numpy as np
import pickle
import whois
from urllib.parse import urlparse
import tldextract
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import cv2
import pandas as pd
import plotly.express as px
import textwrap
from datetime import datetime

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
STOPWORDS = set(stopwords.words('english'))
ps = PorterStemmer()

# Load trustworthy domains
def load_trustworthy_domains(file_path):
    """Load trustworthy domains from a text file."""
    try:
        with open(file_path, 'r') as file:
            return {line.strip().lower() for line in file if line.strip()}
    except Exception as e:
        st.error(f"Error loading trustworthy domains: {str(e)}")
        return set()

TRUSTWORTHY_DOMAINS = load_trustworthy_domains('domains.txt')

def is_valid_url(url):
    """Check if URL is valid and uses HTTPS."""
    try:
        result = urlparse(url)
        return all([result.scheme == 'https', result.netloc])
    except ValueError:
        return False

def extract_domain(url):
    """Extract the main domain from a URL."""
    try:
        extracted = tldextract.extract(url)
        domain = f"{extracted.domain}.{extracted.suffix}"
        return domain
    except Exception as e:
        st.error(f"Error extracting domain: {str(e)}")
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

        expiration_date = w.expiration_date
        if isinstance(expiration_date, list):
            expiration_date = expiration_date[0]

        domain_age = None
        if creation_date:
            domain_age = (datetime.now() - creation_date).days

        # Trust score based on the domain's age and whether it's in the trustworthy list
        trust_score = 0.1  # Default low trust score
        if domain in TRUSTWORTHY_DOMAINS:
            trust_score = 1.0  # Full trust if in the trustworthy list
        elif domain_age:
            if domain_age > 3650:  # More than 10 years
                trust_score = 1.0
            elif domain_age > 1825:  # More than 5 years
                trust_score = 0.8
            elif domain_age > 365:  # More than 1 year
                trust_score = 0.6
            elif domain_age > 180:  # More than 6 months
                trust_score = 0.4
            elif domain_age > 90:  # More than 3 months
                trust_score = 0.2

        return {
            'domain': domain,
            'registrar': w.registrar,
            'creation_date': creation_date,
            'expiration_date': expiration_date,
            'domain_age': domain_age,
            'trust_score': trust_score,
            'country': w.country,
            'state': w.state
        }
    except Exception as e:
        st.error(f"Error getting domain info: {str(e)}")
        return None

def load_image_model(pickle_file):
    """Load the saved model from a .pkl file for image prediction."""

from model_utils import safe_load_model, is_model_compatible
    with open(pickle_file, 'rb') as file:
        model = pickle.load(file)
    return model

def calculate_combined_prediction(text_probs, image_result, domain_trust_score, llm_verdict):
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
    
    llm_real_contrib = (1.0 if llm_verdict == "real" else 0.0) * LLM_WEIGHT * 100
    llm_fake_contrib = (1.0 if llm_verdict == "fake" else 0.0) * LLM_WEIGHT * 100
    
    final_real_prob = text_real_contrib + image_real_contrib + domain_real_contrib + llm_real_contrib
    final_fake_prob = text_fake_contrib + image_fake_contrib + domain_fake_contrib + llm_fake_contrib
    
    total = final_real_prob + final_fake_prob
    normalize_factor = 100 / total
    
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

def preprocess_image(image_url):
    """Preprocess image for model prediction."""
    try:
        response = requests.get(image_url)
        img_array = np.array(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.resize(img, (32, 32))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def predict_image(model, image):
    """Predict if image is real or AI-generated."""
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
    input_data = vector.transform([stemming(input_text)])
    prediction = ensemble_model.predict(input_data)
    probabilities = ensemble_model.predict_proba(input_data)[0]
    return prediction[0], probabilities

def display_article_content(content):
    """Display article content with proper formatting."""
    paragraphs = content.split('\n\n')
    for para in paragraphs:
        wrapped_text = textwrap.fill(para, width=80)
        st.write(wrapped_text)
        st.write("")

def display_domain_info(domain_info):
    """Display domain information in a formatted way."""
    if domain_info:
        st.subheader("Domain Information:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Domain:** {domain_info['domain']}")
            if domain_info['registrar']:
                st.markdown(f"**Registrar:** {domain_info['registrar']}")
            if domain_info['creation_date']:
                st.markdown(f"**Creation Date:** {domain_info['creation_date'].strftime('%Y-%m-%d')}")
            if domain_info['domain_age']:
                st.markdown(f"**Domain Age:** {domain_info['domain_age']} days")
        
        with col2:
            if domain_info['country']:
                st.markdown(f"**Country:** {domain_info['country']}")
            if domain_info['state']:
                st.markdown(f"**State:** {domain_info['state']}")
            if domain_info['expiration_date']:
                st.markdown(f"**Expiration Date:** {domain_info['expiration_date'].strftime('%Y-%m-%d')}")

        trust_score_percentage = domain_info['trust_score'] * 100
        if trust_score_percentage > 50:
            st.success(f"**Trust Score:** {trust_score_percentage:.2f}%")
        else:
            st.error(f"**Trust Score:** {trust_score_percentage:.2f}%")

def create_stacked_bar_chart(combined_results):
    """Create a stacked bar chart showing component contributions."""
    contributions = combined_results['contributions']
    
    chart_data = pd.DataFrame({
        'Category': ['Real', 'Fake'],
        'Article Analysis': [contributions['text_real'], contributions['text_fake']],
        'Image Analysis': [contributions['image_real'], contributions['image_fake']],
        'Domain Trust': [contributions['domain_real'], contributions['domain_fake']],
        'LLM Analysis': [contributions['llm_real'], contributions['llm_fake']]
    })
    
    chart_data_melted = pd.melt(
        chart_data,
        id_vars=['Category'],
        value_vars=['Article Analysis', 'Image Analysis', 'Domain Trust', 'LLM Analysis']
    )
    
    fig = px.bar(
        chart_data_melted,
        x='Category',
        y='value',
        color='variable',
        title='Prediction Breakdown by Component',
        labels={'value': 'Confidence (%)', 'variable': 'Component'},
        color_discrete_map={
            'Article Analysis': '#1f77b4',  # Blue
            'Image Analysis': '#2ca02c',  # Green
            'Domain Trust': '#ff7f0e',    # Orange
            'LLM Analysis': '#d62728'     # Red
        },
        height=400
    )
    
    fig.update_layout(
        barmode='stack',
        xaxis_title="Prediction",
        yaxis_title="Confidence (%)",
        showlegend=True,
        legend_title="Components",
        yaxis_range=[0, 100]
    )
    
    return fig