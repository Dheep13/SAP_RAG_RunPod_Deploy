FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY langchain_requirements.txt .
RUN pip install --no-cache-dir -r langchain_requirements.txt

# Copy the handler script
COPY langchain_runpod_rag_handler.py .

# Set environment variables for model caching
ENV TRANSFORMERS_CACHE=/cache
ENV HF_HOME=/cache

# Create cache directory
RUN mkdir -p /cache

# Command to run the handler
CMD ["python", "-u", "langchain_runpod_rag_handler.py"]