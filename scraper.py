import requests
from bs4 import BeautifulSoup
# Fix for newspaper3k import
try:
    from newspaper import Article, Config
except ImportError:
    # If direct import fails, try importing from newspaper3k
    from newspaper3k import Article, Config
import nltk
from urllib.parse import urlparse, urljoin
import re
import yt_dlp

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

# Configure newspaper
config = Config()
config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

def is_news_article(url, content):
    # Check URL for common news domains
    news_domains = ['news', 'article', 'blog', 'post', 'story']
    domain = urlparse(url).netloc
    if any(word in domain for word in news_domains):
        return True

    # Check for common news article elements in the content
    soup = BeautifulSoup(content, 'html.parser')
    
    # Look for article or news related schema
    if soup.find('meta', property='og:type', content='article'):
        return True
    
    # Check for typical news article structure
    if soup.find('article') or soup.find('div', class_=re.compile(r'article|story|news')):
        return True
    
    # Look for a dateline, byline, or published date
    if soup.find(['time', 'span', 'p'], class_=re.compile(r'date|time|publish|byline')):
        return True
    
    # Check for social sharing buttons, common in news articles
    if soup.find(['div', 'span'], class_=re.compile(r'share|social')):
        return True
    
    return False

def extract_publisher_from_url(url):
    domain = urlparse(url).netloc
    parts = domain.split('.')
    if len(parts) > 2:
        return parts[-2].capitalize()
    return domain.capitalize()

def clean_text(text):
    # Remove advertisements
    ad_patterns = [
        r'Advertisement\s*',
        r'Sponsored\s*',
        r'Ads by\s*',
        r'\[advertisement\]',
        r'\(advertisement\)',
    ]
    for pattern in ad_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Split text into paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    
    # Clean each paragraph
    cleaned_paragraphs = []
    for para in paragraphs:
        # Remove extra whitespace within each paragraph
        cleaned_para = re.sub(r'\s+', ' ', para).strip()
        if cleaned_para:
            cleaned_paragraphs.append(cleaned_para)
    
    # Join paragraphs with double newlines
    return '\n\n'.join(cleaned_paragraphs)

def scrape_article_newspaper3k(url):
    try:
        article = Article(url, config=config)
        article.download()
        article.parse()
        article.nlp()
        
        # Get the top image
        top_image = article.top_image if article.top_image else None
        
        return {
            'title': article.title,
            'text': clean_text(article.text),
            'summary': article.summary,
            'keywords': article.keywords,
            'publish_date': article.publish_date,
            'image_url': top_image
        }
    except Exception as e:
        print(f"Error using newspaper3k: {e}")
        return None

def scrape_article_bs4(url):
    try:
        response = requests.get(url, headers={'User-Agent': config.browser_user_agent})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        title = soup.find('h1').text if soup.find('h1') else "Title not found"
        
        # Try to find the main content
        main_content = soup.find('article') or soup.find('main') or soup.find('div', class_='content')
        
        if main_content:
            paragraphs = main_content.find_all(['p', 'h2', 'h3', 'h4', 'h5', 'h6'])
        else:
            paragraphs = soup.find_all(['p', 'h2', 'h3', 'h4', 'h5', 'h6'])
        
        text = '\n\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        # Find the first image
        image = soup.find('meta', property='og:image') or soup.find('meta', property='twitter:image')
        if image:
            image_url = image.get('content')
        else:
            image = soup.find('img', src=True)
            image_url = image['src'] if image else None
        
        # Ensure the image URL is absolute
        if image_url and not image_url.startswith(('http://', 'https://')):
            image_url = urljoin(url, image_url)
        
        return {
            'title': title,
            'text': clean_text(text),
            'summary': None,
            'keywords': None,
            'publish_date': None,
            'image_url': image_url
        }
    except Exception as e:
        print(f"Error using BeautifulSoup: {e}")
        return None

def scrape_article(url):
    print(f"Scraping article from: {url}")
    
    try:
        response = requests.get(url, headers={'User-Agent': config.browser_user_agent})
        response.raise_for_status()
        
        if not is_news_article(url, response.text):
            print("The provided URL does not appear to be a news article.")
            return None
        
        # Try newspaper3k first
        article_data = scrape_article_newspaper3k(url)
        
        # If newspaper3k fails, fall back to BeautifulSoup
        if not article_data:
            article_data = scrape_article_bs4(url)
        
        if article_data:
            article_data['publisher'] = extract_publisher_from_url(url)
            return article_data
        
    except requests.RequestException as e:
        print(f"Error fetching the URL: {e}")
    
    return None

def is_youtube_link(url):
    """Check if the URL is a YouTube video link."""
    youtube_domains = ['youtube.com', 'youtu.be']
    parsed_url = urlparse(url)
    return any(domain in parsed_url.netloc for domain in youtube_domains)

def transcribe_youtube_video(url):
    """Transcribe YouTube video and create article text using web scraping."""
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'Untitled Video')
            
            # Get video description
            description = info.get('description', '')
            
            # Get video transcript using web scraping
            video_id = url.split('v=')[-1].split('&')[0]
            transcript_url = f"https://www.youtube.com/transcript?video_id={video_id}"
            
            try:
                transcript_response = requests.get(transcript_url)
                if transcript_response.status_code == 200:
                    soup = BeautifulSoup(transcript_response.content, 'html.parser')
                    transcript_elements = soup.find_all('div', class_='transcript-line')
                    
                    transcript_text = ""
                    for element in transcript_elements:
                        transcript_text += element.get_text() + " "
                    
                    if transcript_text.strip():
                        return title, transcript_text
                else:
                    print("Transcript not available for this video")
            except Exception as e:
                print(f"Error getting transcript: {str(e)}")
            
            # Fallback to video description if transcript not available
            if description:
                return title, description
            else:
                return title, f"Transcription of YouTube video: {title}\n\n[Transcription not available]"
    except Exception as e:
        print(f"Error transcribing YouTube video: {str(e)}")
        return None, None

def get_youtube_thumbnail(url):
    """Get YouTube video thumbnail URL."""
    try:
        video_id = url.split('v=')[-1].split('&')[0]
        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
        return thumbnail_url
    except Exception as e:
        print(f"Error getting YouTube thumbnail: {str(e)}")
        return None