import requests
from bs4 import BeautifulSoup
import os

# Set up Gemini credentials
GEMINI_API_KEY = " Replace with your actual API key"  # Replace with your actual API key

def analyze_article(article_text, article_title, domain_info=None, image_result=None):
    """Analyze article using Google Gemini."""
    try:
        # Prepare context information
        context = f"""Context:
        - Source Domain: {domain_info['domain'] if domain_info else 'Unknown'}
        - Domain Trust Score: {domain_info['trust_score'] if domain_info else 'Unknown'}
        - Image Analysis: {image_result if image_result else 'Not available'}"""
        
        # Prepare prompt for Gemini
        prompt = f"""Analyze this news article for authenticity:

        Title: {article_title}
        
        Content: {article_text[:1500]}
        
        {context}
        
        Focus on:
        - Factual consistency
        - Source credibility
        - Writing style
        - Potential biases
        - Sensationalism
        
        Respond with:
        Verdict: real/fake
        Explanation: [brief reasoning]"""
        
        # Configure Gemini request
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        }
        
        # Add API key to URL
        url_with_key = f"{url}?key={GEMINI_API_KEY}"
        
        # Get Gemini response
        response = requests.post(url_with_key, headers=headers, json=data, timeout=30)
        
        # Parse response
        verdict = "unknown"
        explanation = "No explanation available"
        
        if response.status_code == 200:
            response_json = response.json()
            candidates = response_json.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if content:
                    lines = content.strip().split('\n')
                    for line in lines:
                        if line.lower().startswith("verdict:"):
                            verdict = line.split(":")[1].strip().lower()
                        elif line.lower().startswith("explanation:"):
                            explanation = line.split(":")[1].strip()
                    
                    # If didn't find labeled verdict/explanation, use the entire content as explanation
                    if verdict == "unknown" and explanation == "No explanation available":
                        explanation = content
                        # Try to determine verdict from content
                        if "fake" in content.lower() and "real" not in content.lower():
                            verdict = "fake"
                        elif "real" in content.lower() and "fake" not in content.lower():
                            verdict = "real"

        else:
            print(f"Gemini API Error: {response.status_code} - {response.text}")
        
        return verdict, explanation
    except requests.exceptions.Timeout:
        print("The request to the LLM API timed out. Please try again later.")
        return "unknown", "Request timed out"
    except Exception as e:
        print(f"Error getting LLM analysis: {str(e)}")
        return "unknown", "Error occurred during analysis"
