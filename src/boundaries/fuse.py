"""Fusion pondérée des cartes d'indices/arêtes en S_fused.
Étape 2 — Doc 03
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import cv2  # type: ignore


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)


def minmax01(x: np.ndarray) -> np.ndarray:
    vmin = float(np.nanpercentile(x, 1))
    vmax = float(np.nanpercentile(x, 99))
    if vmax <= vmin:
        return np.zeros_like(x, dtype=np.float32)
    return _clip01((x - vmin) / (vmax - vmin))


def standardize_maps(maps: Dict[str, np.ndarray], enable: bool = True) -> Dict[str, np.ndarray]:
    if not enable:
        return {k: m.astype(np.float32, copy=False) for k, m in maps.items()}
    return {k: minmax01(m.astype(np.float32)) for k, m in maps.items()}


def fuse_weighted(maps: Dict[str, np.ndarray], weights: Dict[str, float], smooth_gaussian_sigma: float = 0.0, clip01: bool = True) -> np.ndarray:
    h = w = None
    acc = None
    for name, wgt in weights.items():
        if wgt == 0 or name not in maps:
            continue
        x = maps[name].astype(np.float32)
        if acc is None:
            h, w = x.shape[:2]
            acc = np.zeros((h, w), dtype=np.float32)
        acc += float(wgt) * x
    if acc is None:
        raise ValueError("No maps provided for fusion (check weights and enabled maps)")
    if smooth_gaussian_sigma and smooth_gaussian_sigma > 0:
        k = int(max(3, int(2 * round(smooth_gaussian_sigma * 3) + 1)))
        acc = cv2.GaussianBlur(acc, (k, k), smooth_gaussian_sigma)
    if clip01:
        acc = _clip01(acc)
    return acc.astype(np.float32, copy=False)


__all__ = [
    "standardize_maps",
    "fuse_weighted",
]
