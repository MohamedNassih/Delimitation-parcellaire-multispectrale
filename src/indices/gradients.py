"""Gradient/edge operators for boundary cues.
Étape 2 — Doc 03

Implémenté:
- Sobel magnitude (ksize=3 par défaut)
- Scharr magnitude (optionnel)
- LoG (Laplacian of Gaussian)
- Structure tensor (lambda1 ou coherence)
- Morphological gradient (dilate - erode, structurant disque/ellipse)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import cv2  # type: ignore

try:
    from skimage.feature import structure_tensor, structure_tensor_eigenvalues
except ImportError:
    # compat: anciens alias possibles
    from skimage.feature import structure_tensor, structure_tensor_eigvals as structure_tensor_eigenvalues
from skimage.filters import gaussian


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)


@dataclass
class GradConfig:
    eps: float = 1e-6


def sobel_mag(x: np.ndarray, ksize: int = 3) -> np.ndarray:
    x = x.astype(np.float32)
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(gx, gy)
    m = float(mag.max()) if np.isfinite(mag.max()) else 0.0
    return (mag / (m + 1e-6)).astype(np.float32)


def scharr_mag(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    gx = cv2.Scharr(x, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(x, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    m = float(mag.max()) if np.isfinite(mag.max()) else 0.0
    return (mag / (m + 1e-6)).astype(np.float32)


def log_edge(x: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    x = x.astype(np.float32)
    k = max(3, int(2 * round(sigma * 3) + 1)) if sigma and sigma > 0 else ksize
    xb = cv2.GaussianBlur(x, (k, k), sigma) if sigma and sigma > 0 else x
    lap = cv2.Laplacian(xb, cv2.CV_32F, ksize=ksize)
    lap = np.abs(lap)
    m = float(lap.max()) if np.isfinite(lap.max()) else 0.0
    return (lap / (m + 1e-6)).astype(np.float32)


def structure_tensor_resp(x: np.ndarray, rho: int = 2, sigma: float = 1.0, eigen_response: str = "lambda1") -> np.ndarray:
    x = x.astype(np.float32)
    # 1) Tenseur des gradients (sigma = lissage des gradients)
    Axx, Axy, Ayy = structure_tensor(x, sigma=sigma)
    # 2) Intégration (rho = lissage du tenseur)
    r = max(0, int(rho))
    if r > 0:
        Axx = gaussian(Axx, sigma=r)
        Axy = gaussian(Axy, sigma=r)
        Ayy = gaussian(Ayy, sigma=r)
    # 3) Valeurs propres
    l1, l2 = structure_tensor_eigenvalues((Axx, Axy, Ayy))
    if eigen_response == "coherence":
        resp = (l1 - l2) / (l1 + l2 + 1e-6)
    else:  # lambda1
        resp = l1
    # 4) Normalisation robuste [0,1]
    vmin, vmax = float(np.percentile(resp, 1)), float(np.percentile(resp, 99))
    resp = np.clip((resp - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    return resp.astype(np.float32)



def morph_gradient(x: np.ndarray, se_radius: int = 2) -> np.ndarray:
    x = x.astype(np.float32)
    k = max(1, int(se_radius))
    # Disque approx avec ellipse (OpenCV n'a pas de disque natif)
    size = 2 * k + 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    u8 = (np.clip(x, 0, 1) * 255.0).astype(np.uint8)
    dil = cv2.dilate(u8, se)
    ero = cv2.erode(u8, se)
    mg = (dil - ero).astype(np.float32)
    return _clip01(mg / 255.0)


def compute_all_edges(x: np.ndarray, cfg: Dict) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    if cfg.get("sobel", {}).get("enabled", True):
        out["sobel"] = sobel_mag(x, ksize=int(cfg.get("sobel", {}).get("ksize", 3)))
    if cfg.get("scharr", {}).get("enabled", False):
        out["scharr"] = scharr_mag(x)
    if cfg.get("log", {}).get("enabled", True):
        p = cfg.get("log", {})
        out["log"] = log_edge(x, ksize=int(p.get("ksize", 5)), sigma=float(p.get("sigma", 1.0)))
    if cfg.get("structure_tensor", {}).get("enabled", True):
        p = cfg.get("structure_tensor", {})
        out["structure_tensor"] = structure_tensor_resp(
            x,
            rho=int(p.get("rho", 2)),
            sigma=float(p.get("sigma", 1.0)),
            eigen_response=str(p.get("eigen_response", "lambda1")),
        )
    if cfg.get("morph_gradient", {}).get("enabled", True):
        p = cfg.get("morph_gradient", {})
        out["morph_gradient"] = morph_gradient(x, se_radius=int(p.get("se_radius", 2)))
    return out


__all__ = [
    "GradConfig",
    "sobel_mag",
    "scharr_mag",
    "log_edge",
    "structure_tensor_resp",
    "morph_gradient",
    "compute_all_edges",
]
