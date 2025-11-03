# Web Demo - Agricultural Boundary Detection

Interactive web interface for detecting field boundaries in multispectral satellite imagery using the UNet-lite model.

## Features

- **Dual Pipeline Support**: 
  - Lightweight pipeline for PNG/JPG images (quick processing)
  - Full pipeline for GeoTIFF images (enhanced accuracy with all spectral bands)
- **Interactive Web Interface**: Drag-and-drop upload with real-time processing
- **FastAPI Backend**: High-performance async API with comprehensive error handling
- **Docker Deployment**: Ready-to-run containerized setup
- **Real-time Metrics**: Inference time, image dimensions, and boundary coverage statistics

## Architecture

```
webdemo/
├── app/
│   ├── main.py          # FastAPI application
│   └── inference.py     # Model loading and inference logic
├── static/
│   ├── index.html       # Frontend interface (Tailwind CSS)
│   └── app.js           # JavaScript functionality
├── Dockerfile           # Container configuration
├── docker-compose.yml   # Multi-service setup
├── requirements-webdemo.txt  # Python dependencies
└── README.md            # This file
```

## API Endpoints

- `GET /` - Serves the web interface
- `GET /health` - Health check endpoint
- `POST /infer` - Main inference endpoint

### POST /infer

Accepts multipart form data with a file upload.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `file` (UploadFile) - PNG, JPG, or GeoTIFF image

**Response:**
```json
{
    "success": true,
    "pipeline_type": "png_jpg_lightweight|geotiff_full",
    "inference_time_seconds": 2.156,
    "total_processing_time_seconds": 2.423,
    "image_info": {
        "filename": "field_image.tif",
        "width": 1024,
        "height": 1024,
        "bands": 4,
        "format": "GeoTIFF"
    },
    "result": {
        "mask_dimensions": {"height": 1024, "width": 1024},
        "mask_base64_png": "data:image/png;base64,iVBORw0KGgoAAAANS...",
        "mask_stats": {
            "min_value": 0.0,
            "max_value": 1.0,
            "mean_value": 0.032,
            "positive_ratio": 0.0298
        }
    }
}
```

## Local Development

### Prerequisites

- Python 3.12+
- GDAL system libraries
- Trained model file at `artifacts/models/unet_lite_best.h5`

### Setup

1. **Install dependencies:**
```bash
pip install -r requirements-webdemo.txt
```

2. **Install PyTorch CPU:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

3. **Ensure model exists:**
```bash
# The model should be available at:
artifacts/models/unet_lite_best.h5
```

4. **Run the server:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
```

5. **Access the web interface:**
Open http://localhost:7860 in your browser.

## Docker Deployment

### Quick Start with Make (Recommended)

```bash
# Build and start the web demo
make web-build
make web-up

# Access at http://localhost:7860

# View logs
make web-logs

# Stop the demo
make web-down

# Clean up resources
make web-clean
```

### Manual Docker Commands

```bash
# Build the Docker image
docker-compose -f webdemo/docker-compose.yml build

# Start the service
docker-compose -f webdemo/docker-compose.yml up -d

# View logs
docker-compose -f webdemo/docker-compose.yml logs -f

# Stop the service
docker-compose -f webdemo/docker-compose.yml down

# Clean up
docker-compose -f webdemo/docker-compose.yml down --volumes --remove-orphans
docker image rm agritech-webdemo:latest
```

### Docker Compose Configuration

The web demo uses the following volume mounts:
- `./data:/app/data` - Input multispectral images
- `./artifacts:/app/artifacts` - Model and output files
- `./logs:/app/logs` - Application logs
- `./src:/app/src:ro` - Source code (read-only)

**Port:** 7860

## Usage Guide

### Supported Image Formats

1. **PNG/JPG (Lightweight Pipeline)**
   - Quick processing for demonstration
   - Converts RGB to 4-channel format (REG/RED/NIR/GRE)
   - Best for testing and quick results

2. **GeoTIFF (Full Pipeline)**
   - Uses all available spectral bands
   - Enhanced accuracy with multispectral data
   - Requires at least 4 bands

### Upload Process

1. **Drag and Drop**: Drop image files onto the upload area
2. **File Browser**: Click to browse and select files
3. **Automatic Detection**: System detects image format automatically
4. **Processing**: Real-time progress with loading animations
5. **Results**: Display side-by-side original and boundary detection results

### Result Interpretation

- **White areas**: Detected field boundaries
- **Black areas**: Field interiors
- **Metrics displayed**:
  - Inference time: Model processing speed
  - Image dimensions: Original image size
  - Boundary coverage: Percentage of pixels classified as boundaries

## Error Handling

### Common Issues

1. **"Model not found"**
   - Ensure `artifacts/models/unet_lite_best.h5` exists
   - Run the training pipeline first: `make train`

2. **"Unsupported file type"**
   - Supported: PNG, JPG, JPEG, TIFF, GeoTIFF
   - Check file extension and MIME type

3. **"File too large"**
   - Maximum size: 50MB
   - Consider resizing large images

4. **"Insufficient bands" (GeoTIFF)**
   - Requires minimum 4 spectral bands
   - Check your multispectral image composition

### Health Check

Monitor service health:
```bash
curl http://localhost:7860/health
```

Expected response:
```json
{
    "status": "healthy",
    "timestamp": 1234567890.123,
    "model_loaded": true
}
```

## Performance Optimization

### For Production

1. **Increase Docker resources** (4GB+ RAM recommended)
2. **Use GPU acceleration** (modify Dockerfile for CUDA support)
3. **Implement caching** for frequently used models
4. **Add load balancing** for multiple instances

### For Development

1. **Use volume mounts** for hot reloading
2. **Enable debug mode** in uvicorn
3. **Monitor memory usage** during inference

## Troubleshooting

### Container Won't Start

```bash
# Check logs
make web-logs

# Verify port availability
netstat -an | grep 7860

# Check volume permissions
docker-compose -f webdemo/docker-compose.yml exec webdemo ls -la /app/artifacts/
```

### Model Loading Errors

```bash
# Verify model file exists
ls -la artifacts/models/

# Check model format
python -c "import torch; print(torch.load('artifacts/models/unet_lite_best.h5').keys())"
```

### Memory Issues

```bash
# Monitor container resources
docker stats agritech-webdemo

# Increase Docker Desktop memory limit (Settings > Resources)
```

## Development

### Frontend Customization

The interface uses Tailwind CSS via CDN. Modify `static/index.html` and `static/app.js` for:
- UI layout changes
- Additional metrics display
- Custom styling

### Backend Extensions

Add new endpoints in `app/main.py`:
- Batch processing
- Model comparison
- Advanced metrics
- Export functionality

### Model Integration

Modify `app/inference.py` to:
- Support additional model architectures
- Add preprocessing options
- Implement post-processing filters

## License

Same as main project: MIT License