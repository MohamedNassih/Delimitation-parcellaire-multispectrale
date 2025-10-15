"""Watershed sur relief (1 - S_fused) -> lignes de partage (boundary=1/field=0)
Étape 3 — Doc 04
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
from pathlib import Path

import numpy as np
import cv2  # type: ignore
from skimage.segmentation import watershed, find_boundaries  # type: ignore
from skimage.measure import label  # type: ignore

from src.io.readwrite import read_raster, ensure_float01


def _conn2d(c: int) -> int:
    """Map {4,8} to skimage connectivity {1,2}. Accepts {1,2,4,8}."""
    c = int(c)
    if c <= 1:
        return 1
    if c <= 2:
        return 2
    return 1 if c <= 4 else 2


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)


@dataclass
class WSConfig:
    clean_seeds_dilate: int = 1
    enforce_border_black: bool = True
    min_basin_area_px: int = 256
    connectivity: int = 4  # {4,8} côté user → {1,2} côté skimage
    # Nouveaux paramètres
    min_seed_cov: float = 0.005      # fallback si (NV|VH) < 0.5 %
    minima_percentile: float = 35.0  # seuil des minima sur le relief
    use_full_mask: bool = True       # watershed sur tout le cadre


def watershed_boundaries(s_fused: np.ndarray, seeds_nv: np.ndarray, seeds_vh: np.ndarray, cfg: WSConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Retourne (boundary01, labels). Fallback auto-seeds si couverture NV|VH trop faible."""
    s = _clip01(s_fused)
    rel = _clip01(1.0 - s)

    nv = (seeds_nv > 0).astype(np.uint8)
    vh = (seeds_vh > 0).astype(np.uint8)

    if cfg.clean_seeds_dilate > 0:
        k = 2 * cfg.clean_seeds_dilate + 1
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        nv = cv2.dilate(nv, se)
        vh = cv2.dilate(vh, se)

    seeds_cov = float(((nv | vh) > 0).mean())

    # --- Marqueurs & masque ---
    if seeds_cov < float(cfg.min_seed_cov):
        # Fallback : minima du relief (zones sombres de 1-S_fused)
        blur = cv2.GaussianBlur(rel, (0, 0), 1.0)
        t = float(np.percentile(blur, float(cfg.minima_percentile)))
        minima = (blur <= t).astype(np.uint8)
        markers = label(minima, connectivity=_conn2d(cfg.connectivity))
        mask_ws = np.ones_like(minima, dtype=bool) if cfg.use_full_mask else minima.astype(bool)
    else:
        markers = np.zeros_like(vh, dtype=np.int32)
        if vh.any():
            vh_lbl = label(vh, connectivity=_conn2d(cfg.connectivity))
            markers[vh_lbl > 0] = vh_lbl[vh_lbl > 0]
        if cfg.enforce_border_black:
            markers[0, :] = 0; markers[-1, :] = 0; markers[:, 0] = 0; markers[:, -1] = 0
        mask_ws = np.ones_like(vh, dtype=bool) if cfg.use_full_mask else (nv | vh).astype(bool)

    # Watershed
    labels = watershed(rel, markers=markers, mask=mask_ws)

    # Filtrer les petites cuvettes
    if cfg.min_basin_area_px > 0:
        lab = labels.copy()
        uniq, counts = np.unique(lab, return_counts=True)
        for val, cnt in zip(uniq, counts):
            if val != 0 and cnt < int(cfg.min_basin_area_px):
                labels[lab == val] = 0

    bnd = find_boundaries(labels, connectivity=_conn2d(cfg.connectivity), mode="thick").astype(np.uint8)
    return bnd, labels.astype(np.int32)


def watershed_for_pair(boundary_dir: Path, pairkey: str, seeds_nv: np.ndarray, seeds_vh: np.ndarray, ws_cfg: WSConfig) -> Tuple[np.ndarray, np.ndarray]:
    s_fused, _ = read_raster(boundary_dir / f"{pairkey}_S_fused.tif")
    s_fused = ensure_float01(s_fused)
    return watershed_boundaries(s_fused, seeds_nv, seeds_vh, ws_cfg)


__all__ = [
    "WSConfig",
    "watershed_boundaries",
    "watershed_for_pair",
]
