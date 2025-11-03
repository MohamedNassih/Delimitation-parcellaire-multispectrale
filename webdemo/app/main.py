"""FastAPI application for agricultural boundary detection web demo."""
from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from inference import get_model_manager, ModelManager

# Create FastAPI app
app = FastAPI(
    title="Agricultural Boundary Detection API",
    description="Web demo for detecting field boundaries in multispectral images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the main web interface."""
    index_path = static_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="Frontend files not found")
    
    with open(index_path, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "model_loaded": True
    }


@app.post("/infer")
async def infer_image(request: Request, file: UploadFile = File(...)):
    """Perform inference on uploaded image.
    
    Args:
        request: FastAPI request object
        file: Uploaded image file (PNG, JPG, or GeoTIFF)
        
    Returns:
        JSON response with inference results
    """
    start_time = time.time()
    
    # Validate file type
    allowed_types = ["image/png", "image/jpeg", "image/jpg", "image/tiff", "image/x-tiff"]
    content_type = file.content_type or ""
    
    if content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {content_type}. Allowed types: PNG, JPG, GeoTIFF"
        )
    
    # Check file size (limit to 50MB)
    if hasattr(file, 'size') and file.size and file.size > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size: 50MB")
    
    try:
        # Read file content
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Get model manager
        model_manager = get_model_manager()
        
        # Determine pipeline based on file type
        is_geotiff = content_type in ["image/tiff", "image/x-tiff"] or file.filename.lower().endswith(('.tif', '.tiff'))
        
        # Perform inference
        if is_geotiff:
            # Full pipeline for GeoTIFF
            mask, inference_time = model_manager.infer_geotiff(content)
            pipeline_type = "geotiff_full"
        else:
            # Lightweight pipeline for PNG/JPG
            mask, inference_time = model_manager.infer_png_jpg(content)
            pipeline_type = "png_jpg_lightweight"
        
        # Get image information
        image_info = model_manager.get_image_info(content, is_geotiff)
        
        # Convert mask to base64 PNG
        mask_base64 = model_manager.mask_to_base64(mask)
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        # Prepare response
        response = {
            "success": True,
            "pipeline_type": pipeline_type,
            "inference_time_seconds": round(inference_time, 3),
            "total_processing_time_seconds": round(total_time, 3),
            "image_info": {
                "filename": file.filename,
                "content_type": content_type,
                **image_info
            },
            "result": {
                "mask_dimensions": {
                    "height": int(mask.shape[0]),
                    "width": int(mask.shape[1])
                },
                "mask_base64_png": mask_base64,
                "mask_stats": {
                    "min_value": float(mask.min()),
                    "max_value": float(mask.max()),
                    "mean_value": float(mask.mean()),
                    "positive_ratio": float((mask > 0.5).sum() / mask.size)
                }
            },
            "api_info": {
                "model_path": str(model_manager.model_path),
                "device": str(model_manager.device),
                "timestamp": time.time()
            }
        }
        
        return JSONResponse(content=response)
        
    except ValueError as e:
        # Specific validation errors
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # Model or inference errors
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    except Exception as e:
        # Generic errors
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    """Handle internal server errors."""
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=7860,
        reload=True
    )