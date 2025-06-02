FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY langchain_requirements.txt .
RUN pip install --no-cache-dir -r langchain_requirements.txt

# Copy your handler code
COPY langchain_runpod_rag_handler.py .

# Set environment variables for model caching
ENV TRANSFORMERS_CACHE=/cache
ENV HF_HOME=/cache
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create cache directory
RUN mkdir -p /cache

# Command to run when the container starts
CMD ["python", "-u", "langchain_runpod_rag_handler.py"]
