import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def scrape_google_news(query):
    """Scrape Google News for related articles."""
    google_news_url = f"https://news.google.com/search?q={query.replace(' ', '%20')}"
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(google_news_url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.select("article a")

        return [{"title": a.text.strip(), "url": "https://news.google.com" + a['href']} for a in articles if a.text.strip()][:5]
    except Exception as e:
        print(f"Google News Scraping Error: {e}")
        return []

def scrape_rss_feed(url):
    """Scrape news articles from RSS feed."""

from model_utils import safe_load_model, is_model_compatible
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'xml')
        items = soup.find_all("item")

        return [{"title": item.title.text.strip(), "url": item.link.text.strip()} for item in items if item.title and item.link][:5]
    except Exception as e:
        print(f"RSS Scraping Error: {e}")
        return []

def scrape_wikipedia_events():
    """Scrape Wikipedia Current Events page."""
    url = "https://en.wikipedia.org/wiki/Portal:Current_events"
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        events = soup.select("#mp-itn b a")
        return [{"title": e.text.strip(), "url": "https://en.wikipedia.org" + e["href"]} for e in events if e.text.strip()][:5]
    except Exception as e:
        print(f"Wikipedia Scraping Error: {e}")
        return []

def local_tfidf_similarity(query, local_articles):
    """Find related articles using TF-IDF NLP."""
    try:
        if not local_articles:
            return []
        
        vectorizer = TfidfVectorizer(stop_words='english')
        corpus = [query] + [a['title'] for a in local_articles]
        tfidf_matrix = vectorizer.fit_transform(corpus)
        cosine_similarities = np.dot(tfidf_matrix[0].toarray(), tfidf_matrix[1:].T.toarray()).flatten()
        sorted_indices = np.argsort(-cosine_similarities)[:5]
        
        return [local_articles[i] for i in sorted_indices if local_articles[i]['title'].strip()]
    except Exception as e:
        print(f"TF-IDF Error: {e}")
        return []

def find_related_articles(query):
    """Find related articles using only free sources."""
    results = []

    # Scraping news sources
    results.extend(scrape_google_news(query))
    results.extend(scrape_rss_feed("http://feeds.bbci.co.uk/news/rss.xml"))  # BBC News
    results.extend(scrape_rss_feed("http://rss.cnn.com/rss/edition.rss"))   # CNN News
    results.extend(scrape_wikipedia_events())  # Wikipedia Current Events

    # Remove empty or duplicate articles
    seen = set()
    results = [article for article in results if article['title'].strip() and article['title'] not in seen and not seen.add(article['title'])]

    # Apply local NLP similarity for better ranking
    if results:
        results = local_tfidf_similarity(query, results)

    return results

if __name__ == "__main__":
    query = input("Enter a topic to search for related news articles: ")
    articles = find_related_articles(query)

    if articles:
        print("\nRelated Articles:")
        for idx, article in enumerate(articles, 1):
            print(f"{idx}. {article['title']} - {article['url']}")
    else:
        print("\nNo related articles found.")
