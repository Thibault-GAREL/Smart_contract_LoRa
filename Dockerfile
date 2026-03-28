# Base image: PyTorch 2.5.1 + CUDA 12.1 + Python 3.10
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements first (layer cache optimization)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download TinyLlama model into the image
# This avoids re-downloading on every run (~2.2 Go, cached in the image)
ENV HF_HOME=/workspace/hf_cache
RUN python -c "\
from transformers import AutoModelForCausalLM, AutoTokenizer; \
AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0'); \
AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')"

# Copy project files
COPY . .

# Keep container alive (required for RunPod SSH/Jupyter)
CMD ["sleep", "infinity"]
