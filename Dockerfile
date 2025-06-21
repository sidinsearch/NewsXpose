FROM python:3.11.5-slim AS builder

WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Create a virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader stopwords punkt

# Second stage - smaller final image
FROM python:3.11.5-slim

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy NLTK data
COPY --from=builder /root/nltk_data /root/nltk_data

# Copy the application
COPY . .

# Verify model files exist
RUN ls -la *.pkl || echo "Model files not found"
RUN python check_model.py || echo "Model check failed but continuing"

# Expose the port Streamlit will run on
EXPOSE 8501

# Set environment variables to suppress specific warnings
ENV PYTHONWARNINGS="ignore::UserWarning:sklearn.base"
ENV PYTHONWARNINGS="${PYTHONWARNINGS},ignore::UserWarning:sklearn.tree._tree"

# Command to run the application
CMD streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0