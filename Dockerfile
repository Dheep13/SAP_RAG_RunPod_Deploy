# Optimized Dockerfile for SAP RAG with CodeLlama 13B
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set working directory
WORKDIR /

# Environment variables for optimization
ENV PYTHONPATH="${PYTHONPATH}:/workspace:/app"
ENV PYTHONUNBUFFERED=1
ENV HF_HOME="/runpod-volume/huggingface"
ENV TRANSFORMERS_CACHE="/runpod-volume/transformers"
ENV TORCH_HOME="/runpod-volume/torch"
ENV HF_DATASETS_CACHE="/runpod-volume/datasets"
ENV TOKENIZERS_PARALLELISM=false

# Create cache directories
RUN mkdir -p /runpod-volume/huggingface /runpod-volume/transformers /runpod-volume/torch /runpod-volume/datasets

# Update system and install dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    nvtop \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch and related packages first (use exact versions for stability)
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==2.2.0 \
    accelerate==0.28.0

# Install transformers and quantization libraries
RUN pip install --no-cache-dir \
    transformers==4.40.0 \
    bitsandbytes==0.43.0 \
    flash-attn==2.5.6 \
    tokenizers==0.19.1

# Install LangChain ecosystem (compatible versions)
RUN pip install --no-cache-dir \
    langchain==0.1.20 \
    langchain-core==0.1.52 \
    langchain-community==0.0.38 \
    langchain-huggingface==0.0.3

# Install embeddings and vector stores
RUN pip install --no-cache-dir \
    sentence-transformers==2.7.0 \
    faiss-cpu==1.8.0

# Install Supabase and database clients
RUN pip install --no-cache-dir \
    supabase==2.4.2 \
    psycopg2-binary==2.9.9 \
    postgrest==0.16.6

# Install additional ML/data science packages
RUN pip install --no-cache-dir \
    numpy==1.24.4 \
    pandas==2.0.3 \
    scikit-learn==1.3.2 \
    scipy==1.11.4

# Install utility packages
RUN pip install --no-cache-dir \
    requests==2.31.0 \
    python-dotenv==1.0.1 \
    pydantic==2.6.4 \
    tqdm==4.66.2 \
    datasets==2.18.0 \
    huggingface-hub==0.22.2

# Install RunPod SDK
RUN pip install --no-cache-dir runpod==1.6.0

# Create application directory
RUN mkdir -p /app

# Copy application files
COPY . /app/

# Set working directory to app
WORKDIR /app

# Make handler executable
RUN chmod +x handler.py

# Default command to run the handler
CMD ["python", "handler.py"]