# Web Demo Dockerfile for Agricultural Boundary Detection
# Based on Python 3.12-slim with GDAL dependencies

FROM python:3.12-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies for GDAL/rasterio and web server
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
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU via official index (as specified in README)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy web demo requirements and install dependencies
COPY webdemo/requirements-webdemo.txt /app/requirements-webdemo.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements-webdemo.txt

# Copy web demo source code
COPY webdemo/ /app/webdemo/

# Set working directory for web demo
WORKDIR /app/webdemo

# Expose port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Command to run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]