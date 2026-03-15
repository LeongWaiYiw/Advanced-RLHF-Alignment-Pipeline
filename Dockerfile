# Use NVIDIA CUDA base image for GPU acceleration
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set non-interactive timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Singapore

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install PyTorch and core ML dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install --no-cache-dir -r requirements.txt

# Install DeepSpeed and Flash Attention for optimal performance
RUN pip3 install deepspeed flash-attn --no-build-isolation

# Copy project files
COPY . .

CMD ["/bin/bash"]
