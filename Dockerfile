# Dockerfile for Délimitation des champs agricoles
# Base image: Python 3.12-slim with CPU-only PyTorch

FROM python:3.12-slim

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies for GDAL/rasterio and other libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    gdal-bin \
    libgdal-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU via official index (as specified in README)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install Python dependencies
COPY requirements-docker.txt /app/requirements-docker.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy project source code
COPY . /app/

# Create necessary directories
RUN mkdir -p data/multispectral-images artifacts/aligned artifacts/indices \
    artifacts/boundary_maps artifacts/masks_raw artifacts/masks_filtered \
    artifacts/masks_final artifacts/reports artifacts/models artifacts/preds logs

# Set default command to bash
CMD ["/bin/bash"]