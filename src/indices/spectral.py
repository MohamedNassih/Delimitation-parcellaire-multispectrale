"""Spectral indices for REG/RED/NIR/GRE.
Étape 2 — Doc 03

Indices implémentés (tous en float32, [0,1] après clamp):
- NDVI  = (NIR - RED) / (NIR + RED + eps)
- GNDVI = (NIR - GRE) / (NIR + GRE + eps)
- NDRE  = (NIR - REG) / (NIR + REG + eps)
- BRVI  = RED / (REG + eps)
- LR    = log1p(NIR) - log1p(RED)

Notes:
- Toutes les entrées doivent être des images normalisées [0,1] (voir prepare).
- On applique un clamp [0,1] pour la stabilité lors des fusions ultérieures.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)


@dataclass
class SpectralConfig:
    eps: float = 1e-6


def ndvi(nir: np.ndarray, red: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    num = nir.astype(np.float32) - red.astype(np.float32)
    den = nir.astype(np.float32) + red.astype(np.float32) + float(eps)
    return _clip01((num / den + 1.0) * 0.5)  # remappe [-1,1] -> [0,1]


def gndvi(nir: np.ndarray, gre: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    num = nir.astype(np.float32) - gre.astype(np.float32)
    den = nir.astype(np.float32) + gre.astype(np.float32) + float(eps)
    return _clip01((num / den + 1.0) * 0.5)


def ndre(nir: np.ndarray, reg: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    num = nir.astype(np.float32) - reg.astype(np.float32)
    den = nir.astype(np.float32) + reg.astype(np.float32) + float(eps)
    return _clip01((num / den + 1.0) * 0.5)


def brvi(red: np.ndarray, reg: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return _clip01(red.astype(np.float32) / (reg.astype(np.float32) + float(eps)))


def lr(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    return _clip01(np.log1p(nir.astype(np.float32)) - np.log1p(red.astype(np.float32)))


def compute_all(
    REG: np.ndarray,
    RED: np.ndarray,
    NIR: np.ndarray,
    GRE: np.ndarray,
    cfg: SpectralConfig,
    enabled: Dict[str, bool],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    e = float(cfg.eps)
    if enabled.get("NDVI", True):
        out["NDVI"] = ndvi(NIR, RED, e)
    if enabled.get("GNDVI", True):
        out["GNDVI"] = gndvi(NIR, GRE, e)
    if enabled.get("NDRE", True):
        out["NDRE"] = ndre(NIR, REG, e)
    if enabled.get("BRVI", True):
        out["BRVI"] = brvi(RED, REG, e)
    if enabled.get("LR", True):
        out["LR"] = lr(NIR, RED)
    return out


def local_std(img01: np.ndarray, ksize: int = 9) -> np.ndarray:
    """Écart-type local (fenêtre carrée ksize)."""
    import cv2  # lazy import
    x = img01.astype(np.float32)
    k = max(1, int(ksize))
    mean = cv2.blur(x, (k, k))
    mean2 = cv2.blur(x * x, (k, k))
    var = np.maximum(mean2 - mean * mean, 0.0)
    std = np.sqrt(var)
    # Normalise par max pour rester dans [0,1] (robuste pour cfg.numerics.standardize_each_map)
    m = float(std.max()) if np.isfinite(std.max()) else 0.0
    return (std / (m + 1e-6)).astype(np.float32)


__all__ = [
    "SpectralConfig",
    "ndvi",
    "gndvi",
    "ndre",
    "brvi",
    "lr",
    "compute_all",
    "local_std",
]
