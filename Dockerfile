FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source files
COPY src/ ./src/

# Environment variables
ENV PYTHONPATH=${PYTHONPATH}:/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HOST=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose Streamlit port
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "src/RAG/frontend/frontend.py"]