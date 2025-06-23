# NewsXpose

NewsXpose is an advanced news analysis platform that helps users detect fake news and analyze content from various sources including articles and YouTube videos.

## Features

- Analyze news articles from URLs
- Process YouTube video content
- Detect fake news using multiple analysis components:
  - Text analysis
  - Image analysis
  - Domain trust evaluation
  - LLM-based content analysis
- Find related articles
- Visualize analysis results

## Deployment on Render

This project can be easily deployed on Render using Docker. Follow these steps:

1. Fork or clone this repository to your GitHub account
2. Sign up for a Render account at [render.com](https://render.com)
3. Create a new Web Service on Render
4. Select "Deploy from GitHub repo"
5. Connect your GitHub account and select this repository
6. Choose "Docker" as the Environment
7. Configure the service:
   - Name: NewsXpose (or your preferred name)
   - Environment: Docker
   - Branch: main (or your default branch)
   - Region: Choose the closest to your users
   - Instance Type: Free or paid tier based on your needs
8. Click "Create Web Service"

Render will automatically build and deploy your Docker image. Once deployed, you can access your application at the URL provided by Render.

## Local Development

To run the application locally:

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Docker

To build and run the Docker image locally:

```bash
# Build the Docker image
docker build -t newsxpose .

# Run the container
docker run -p 8501:8501 newsxpose
```

Then access the application at http://localhost:8501

## Deploying on Render

This project is configured for easy deployment on Render using Docker:

1. Push your code to GitHub
2. Sign up for a Render account at [render.com](https://render.com)
3. Create a new Web Service
4. Select "Deploy from GitHub repo"
5. Connect your GitHub account and select this repository
6. Choose "Docker" as the Environment
7. Configure the service:
   - Name: NewsXpose (or your preferred name)
   - Environment: Docker
   - Branch: main (or your default branch)
   - Region: Choose the closest to your users
   - Instance Type: Free or paid tier based on your needs
8. Click "Create Web Service"

Render will automatically build and deploy your Docker image. Once deployed, you can access your application at the URL provided by Render.

## Made with 💻 by [Siddharth Shinde](https://github.com/sidinsearch)