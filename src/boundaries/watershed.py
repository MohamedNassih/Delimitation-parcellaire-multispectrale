"""Seeds (NV/VH) helpers from indices and S_fused (Étape 2)

Ce module génère uniquement des *seeds* NV/VH basés sur les seuils
configurés (indices.yaml). Le vrai watershed sera fait à l'Étape 3.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import cv2  # type: ignore


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)


@dataclass
class SeedThresholds:
    ndvi_max: float = 0.05
    gndvi_max: float = 0.05
    ndvi_min: float = 0.35
    texture_max: float = 0.10
    s_fused_max: float = 0.20
    area_min_px: int = 64


def _clean_small(mask: np.ndarray, area_min_px: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    if num_labels <= 1:
        return mask
    keep = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= int(area_min_px):
            keep[labels == i] = 1
    return keep.astype(np.uint8)


def make_seeds(ndvi: np.ndarray, gndvi: np.ndarray, s_fused: np.ndarray, texture: np.ndarray, cfg_seeds: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Retourne (NV, VH) en uint8 {0,1}."""
    nv_cfg = cfg_seeds.get("nv", {})
    vh_cfg = cfg_seeds.get("vh", {})

    ndvi = _clip01(ndvi)
    gndvi = _clip01(gndvi)
    s_fused = _clip01(s_fused)
    texture = _clip01(texture)

    nv = ((ndvi <= float(nv_cfg.get("ndvi_max", 0.05))) &
          (gndvi <= float(nv_cfg.get("gndvi_max", 0.05))) &
          (s_fused <= float(nv_cfg.get("s_fused_max", 0.15))) &
          (texture <= float(nv_cfg.get("texture_max", 0.08)))).astype(np.uint8)
    nv = _clean_small(nv, int(nv_cfg.get("area_min_px", 64)))

    vh = ((ndvi >= float(vh_cfg.get("ndvi_min", 0.35))) &
          (s_fused <= float(vh_cfg.get("s_fused_max", 0.20))) &
          (texture <= float(vh_cfg.get("texture_max", 0.10)))).astype(np.uint8)
    vh = _clean_small(vh, int(vh_cfg.get("area_min_px", 64)))

    # ouverture légère (open_radius=1) pour nettoyer le bruit
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    nv = cv2.morphologyEx(nv, cv2.MORPH_OPEN, se)
    vh = cv2.morphologyEx(vh, cv2.MORPH_OPEN, se)

    return nv.astype(np.uint8), vh.astype(np.uint8)


__all__ = [
    "SeedThresholds",
    "make_seeds",
]
