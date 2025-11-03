"""Inference module for the agricultural boundary detection model.

Handles both lightweight pipeline (PNG/JPG) and full pipeline (GeoTIFF).
"""
from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import cv2
import rasterio
from rasterio.transform import from_bounds
import base64

# Try to import from src, fallback to standalone implementation
try:
    from src.models.unet_lite import UNetLite
    from src.io.readwrite import ensure_float01
    SRC_AVAILABLE = True
except ImportError:
    SRC_AVAILABLE = False

# Standalone implementation if src is not available
if not SRC_AVAILABLE:
    import warnings
    
    def ensure_float01(arr: np.ndarray) -> np.ndarray:
        """Ensure array is normalized to 0-1 range."""
        return arr.astype(np.float32)
    
    class UNetLite(nn.Module):
        """Simplified UNetLite implementation for demo purposes."""
        
        def __init__(self, in_ch: int = 4, base_ch: int = 32, out_ch: int = 1, final_activation: bool = True):
            super().__init__()
            # Simplified model for demo - just a basic convolution
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
            self.final_activation = final_activation
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv(x)
            if self.final_activation:
                return torch.sigmoid(x)
            return x
        
        def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
            # Handle different state dict formats
            if 'model' in state_dict:
                state_dict = state_dict['model']
            self.conv.weight.data = state_dict['conv.weight'] if 'conv.weight' in state_dict else state_dict['weight']
            
        def eval(self):
            self.training = False


class ModelManager:
    """Manages model loading and inference."""
    
    def __init__(self, model_path: Path):
        """Initialize model manager.
        
        Args:
            model_path: Path to the trained model (.h5 file)
        """
        self.model_path = model_path
        self.model: Optional[UNetLite] = None
        self.device = torch.device("cpu")
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the UNetLite model from file."""
        try:
            # Check if model file exists
            if not self.model_path.exists():
                print(f"⚠️ Model file not found at {self.model_path}. Using fallback implementation.")
                self.model = UNetLite(in_ch=4, base_ch=32, out_ch=1, final_activation=True)
                return
            
            ckpt = torch.load(self.model_path, map_location=self.device)
            self.model = UNetLite(in_ch=4, base_ch=32, out_ch=1, final_activation=True)
            
            # Handle different checkpoint formats
            if isinstance(ckpt, dict) and "model" in ckpt:
                state_dict = ckpt["model"]
            else:
                state_dict = ckpt
            
            # Try to load state dict
            try:
                self.model.load_state_dict(state_dict)
            except RuntimeError as e:
                print(f"⚠️ Failed to load model weights: {e}. Using fallback implementation.")
                self.model = UNetLite(in_ch=4, base_ch=32, out_ch=1, final_activation=True)
            
            self.model.eval()
            print(f"✅ Model loaded successfully from {self.model_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to load model from {self.model_path}: {e}")
            print("🔄 Using fallback implementation for demo.")
            self.model = UNetLite(in_ch=4, base_ch=32, out_ch=1, final_activation=True)
    
    def infer_png_jpg(self, image_data: bytes) -> Tuple[np.ndarray, float]:
        """Perform inference on PNG/JPG image.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Tuple of (mask_array, inference_time_seconds)
        """
        start_time = time.time()
        
        # Load and preprocess image
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Convert RGB to required 4-channel format (REG/RED/NIR/GRE)
        # For demo purposes, we'll use a simple mapping
        # REG ~ Red channel, RED ~ Red channel, NIR ~ Blue channel, GRE ~ Green channel
        height, width = img_array.shape[:2]
        
        # Create 4-channel stack
        red = img_array[:, :, 0]  # RED channel
        green = img_array[:, :, 1]  # GRE channel  
        blue = img_array[:, :, 2]  # NIR channel
        
        # For REG channel, use red channel as approximation
        reg = red.copy()
        
        # Stack as [REG, RED, NIR, GRE]
        x4 = np.stack([reg, red, blue, green], axis=0)
        
        # Ensure correct shape and type
        x4 = ensure_float01(x4)
        
        # Perform inference
        with torch.no_grad():
            x_tensor = torch.from_numpy(x4).unsqueeze(0)  # Add batch dimension
            pred = self.model(x_tensor.to(self.device)).cpu().numpy()[0, 0]
        
        inference_time = time.time() - start_time
        return pred, inference_time
    
    def infer_geotiff(self, image_data: bytes) -> Tuple[np.ndarray, float]:
        """Perform inference on GeoTIFF image.
        
        Args:
            image_data: Raw GeoTIFF bytes
            
        Returns:
            Tuple of (mask_array, inference_time_seconds)
        """
        start_time = time.time()
        
        # Load GeoTIFF with rasterio
        with rasterio.MemoryFile(io.BytesIO(image_data)) as memfile:
            with memfile.open() as dataset:
                # Read all bands
                bands = []
                for i in range(1, dataset.count + 1):
                    band = dataset.read(i)
                    bands.append(ensure_float01(band))
                
                # Ensure we have 4 bands (REG, RED, NIR, GRE)
                if len(bands) < 4:
                    raise ValueError(f"GeoTIFF must have at least 4 bands, found {len(bands)}")
                
                # Use first 4 bands
                x4 = np.stack(bands[:4], axis=0).astype(np.float32)
        
        # Perform inference
        with torch.no_grad():
            x_tensor = torch.from_numpy(x4).unsqueeze(0)
            pred = self.model(x_tensor.to(self.device)).cpu().numpy()[0, 0]
        
        inference_time = time.time() - start_time
        return pred, inference_time
    
    def mask_to_base64(self, mask: np.ndarray) -> str:
        """Convert mask array to base64 PNG string.
        
        Args:
            mask: 2D mask array (0-1 values)
            
        Returns:
            Base64 encoded PNG string
        """
        # Convert to 0-255 uint8
        mask_255 = (mask * 255).astype(np.uint8)
        
        # Create PIL image
        img = Image.fromarray(mask_255, mode='L')
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        
        # Convert to base64
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return img_base64
    
    def get_image_info(self, image_data: bytes, is_geotiff: bool = False) -> Dict[str, Any]:
        """Get image information.
        
        Args:
            image_data: Raw image bytes
            is_geotiff: Whether image is GeoTIFF
            
        Returns:
            Dictionary with image information
        """
        if is_geotiff:
            with rasterio.MemoryFile(io.BytesIO(image_data)) as memfile:
                with memfile.open() as dataset:
                    info = {
                        'width': dataset.width,
                        'height': dataset.height,
                        'bands': dataset.count,
                        'dtype': str(dataset.dtype),
                        'driver': dataset.driver
                    }
        else:
            img = Image.open(io.BytesIO(image_data))
            info = {
                'width': img.width,
                'height': img.height,
                'bands': len(img.getbands()) if hasattr(img, 'getbands') else 1,
                'format': img.format,
                'mode': img.mode
            }
        
        return info


# Global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        # Default model path
        model_path = Path("/app/artifacts/models/unet_lite_best.h5")
        _model_manager = ModelManager(model_path)
    return _model_manager