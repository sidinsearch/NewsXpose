import streamlit as st
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
import yt_dlp
from bs4 import BeautifulSoup
import os
from articlefinder import find_related_articles
from llm import analyze_article as get_llm_analysis  # Import the function from llm.py
from scraper import transcribe_youtube_video, get_youtube_thumbnail



# Load pre-trained ensemble model and vectorizer for text classification
with open('ensemble_fake_news_detector.pkl', 'rb') as f:
    ensemble_model, vector = pickle.load(f)

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

# Streamlit app
def main():
    st.title('NewsXpose: Advanced News Analysis Platform')
    input_text = st.text_area('Enter news article URL, YouTube link, or content', height=200)

    if input_text:
        text_for_analysis = input_text
        image_result = "Real"  # Default if no image
        domain_trust_score = 0.1  # Default if no domain info
        article_data = None
        llm_verdict = "unknown"
        llm_explanation = ""
        related_articles = []
        yt_title = None
        yt_transcription = None
        yt_thumbnail = None
        
        is_youtube = is_youtube_link(input_text)
        
        if is_valid_url(input_text) and not is_youtube:
            article_data = scrape_article(input_text)
            if article_data:
                st.success("Article scraped successfully!")
                
                st.header(article_data['title'])
                
                st.write(f"**Publisher:** {article_data['publisher']}")
                if article_data['publish_date']:
                    formatted_date = article_data['publish_date'].strftime("%Y-%m-%d %H:%M:%S%z")
                    st.write(f"**Publish Date:** {formatted_date}")

                if article_data['image_url']:
                    image_model = load_image_model('image-model.pkl')
                    image = preprocess_image(article_data['image_url'])
                    if image is not None:
                        image_result = predict_image(image_model, image)
                        st.image(article_data['image_url'], caption="", use_column_width=True)
                        
                        if image_result == "Real":
                            st.success(f'Image Analysis: Real')
                        else:
                            st.error(f'Image Analysis: AI-generated')
                
                st.subheader("Article Content:")
                display_article_content(article_data['text'])
                text_for_analysis = article_data['text']

                domain_info = get_domain_info(input_text)
                if domain_info:
                    display_domain_info(domain_info)
                    domain_trust_score = domain_info['trust_score']
                
                # Get LLM analysis
                with st.spinner("Analyzing with LLM..."):
                    llm_verdict, llm_explanation = get_llm_analysis(article_data['text'], article_data['title'])
                
                # Find related articles
                with st.spinner("Finding related articles..."):
                    related_articles = find_related_articles(article_data['title'])
            else:
                st.error("Failed to scrape the article. Please check the URL and try again.")
                text_for_analysis = ""
        elif is_youtube:
            # Process YouTube video
            yt_title, yt_transcription = transcribe_youtube_video(input_text)
            if yt_title and yt_transcription:
                st.success("YouTube video processed successfully!")
                st.header(yt_title)
                yt_thumbnail = get_youtube_thumbnail(input_text)
                if yt_thumbnail:
                    st.image(yt_thumbnail, caption="YouTube Thumbnail", use_column_width=True)
                
                st.subheader("Transcribed Content:")
                display_article_content(yt_transcription)
                text_for_analysis = yt_transcription
                
                # Get LLM analysis for YouTube content
                with st.spinner("Analyzing with LLM..."):
                    llm_verdict, llm_explanation = get_llm_analysis(yt_transcription, yt_title)
                
                # Find related articles
                with st.spinner("Finding related articles..."):
                    related_articles = find_related_articles(yt_title)
            else:
                st.error("Failed to process YouTube video. Please check the URL and try again.")
                text_for_analysis = ""
        else:
            st.subheader("Entered Article Content:")
            display_article_content(clean_text(input_text))
            st.info("Domain information is only available for URL inputs.")
            text_for_analysis = clean_text(input_text)

        if text_for_analysis:
            # Article prediction
            pred, probs = prediction(text_for_analysis)
            
            # Calculate combined prediction
            combined_results = calculate_combined_prediction(probs, image_result, domain_trust_score, llm_verdict)
            
            # Display results
            st.header("Detection Results")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Display stacked bar chart
                fig = create_stacked_bar_chart(combined_results)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Display component breakdown
                st.write("### Component Weights")
                contributions = combined_results['contributions']
                
                st.write("**Article Analysis (50%)**")
                st.write(f"- Real: {contributions['text_real']:.1f}%")
                st.write(f"- Fake: {contributions['text_fake']:.1f}%")
                
                st.write("**Image Analysis (15%)**")
                st.write(f"- Real: {contributions['image_real']:.1f}%")
                st.write(f"- Fake: {contributions['image_fake']:.1f}%")
                
                st.write("**Domain Trust (15%)**")
                st.write(f"- Real: {contributions['domain_real']:.1f}%")
                st.write(f"- Fake: {contributions['domain_fake']:.1f}%")
                
                st.write("**LLM Analysis (20%)**")
                st.write(f"- Real: {contributions['llm_real']:.1f}%")
                st.write(f"- Fake: {contributions['llm_fake']:.1f}%")
            
            # LLM explanation
            st.subheader("LLM Analysis Explanation")
            st.write(llm_explanation)
            
            # Related articles
            if related_articles:
                st.subheader("Related Articles")
                for article in related_articles:
                    st.markdown(f"- [{article['title']}]({article['url']})")
            
            # Final verdict
            st.subheader("Final Verdict")
            is_fake = combined_results['final_fake_prob'] > combined_results['final_real_prob']
            detailed_summary = ""
            if is_fake:
                detailed_summary += "This content is likely to be fake. Here's why:\n"
                detailed_summary += "- The content may contain sensationalized or exaggerated claims.\n"
                detailed_summary += "- There might be a lack of credible sources or verifiable facts.\n"
                detailed_summary += "- The tone and language used may be emotionally charged or biased.\n"
                detailed_summary += "- The content could be trying to provoke an emotional response.\n"
                detailed_summary += "- It might lack references or links to original sources.\n"
            else:
                detailed_summary += "This content is likely to be real. Here's why:\n"
                detailed_summary += "- The content appears to be based on verifiable facts and credible sources.\n"
                detailed_summary += "- The tone is generally neutral and objective.\n"
                detailed_summary += "- The content likely presents multiple perspectives on the topic.\n"
                detailed_summary += "- It cites reputable experts or studies relevant to the topic.\n"
                detailed_summary += "- The publication has a history of reliable reporting.\n"

            st.write(detailed_summary)
            if combined_results['final_fake_prob'] > combined_results['final_real_prob']:
                st.error(f'This content is likely to be FAKE (Confidence: {combined_results["final_fake_prob"]:.1f}%)')
            else:
                st.success(f'This content is likely to be REAL (Confidence: {combined_results["final_real_prob"]:.1f}%)')
            st.write('Note: Always verify news from reliable sources, regardless of these predictions.')

def add_footer():
    """Add a footer with attribution and GitHub link."""
    st.markdown("---")
    st.markdown(
        "Made with 💻 by [Siddharth Shinde](https://github.com/sidinsearch)",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
    add_footer()