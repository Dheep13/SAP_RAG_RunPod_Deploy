FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY langchain_requirements.txt .
RUN pip install --no-cache-dir -r langchain_requirements.txt

# Copy your handler code and utilities
COPY langchain_runpod_rag_handler.py .
COPY check_storage.py .

# Set environment variables for memory optimization
ENV TRANSFORMERS_CACHE=/runpod-volume
ENV HF_HOME=/runpod-volume
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
ENV CUDA_LAUNCH_BLOCKING=1
ENV TOKENIZERS_PARALLELISM=false

# Create cache directory with proper permissions (fallback if volume not mounted)
RUN mkdir -p /runpod-volume && chmod 777 /runpod-volume

# Pre-download models to reduce cold start time (optional)
# RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('codellama/CodeLlama-13b-Instruct-hf', cache_dir='/runpod-volume')"
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2', cache_folder='/runpod-volume')"

# Expose port for health checks
EXPOSE 8000

# Command to run when the container starts
CMD ["python", "-u", "langchain_runpod_rag_handler.py"]
