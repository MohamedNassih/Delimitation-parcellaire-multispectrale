"""Seed masks (NV_strict, VH_strict) à partir des indices et de S_fused.
Étape 3 — Doc 04

Entrées attendues pour une scene (pairkey PK):
- NDVI: artifacts/indices/NDVI/PK_NDVI.tif
- GNDVI: artifacts/indices/GNDVI/PK_GNDVI.tif
- S_fused: artifacts/boundary_maps/PK_S_fused.tif

Sorties: deux masques uint8 {0,1} (0=non seed, 1=seed).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import cv2  # type: ignore

from src.io.readwrite import read_raster, ensure_float01
from src.indices.spectral import local_std


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)


@dataclass
class SeedRules:
    # NV (Non-Végétation) stricte
    nv_ndvi_max: float = 0.05
    nv_gndvi_max: float = 0.05
    nv_s_fused_max: float = 0.15
    nv_texture_max: float = 0.08
    nv_area_min: int = 64
    nv_open_radius: int = 1
    # VH (Végétation homogène) stricte
    vh_ndvi_min: float = 0.35
    vh_s_fused_max: float = 0.20
    vh_texture_max: float = 0.10
    vh_area_min: int = 64
    vh_open_radius: int = 1


def _clean_small(mask01: np.ndarray, area_min: int) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask01 > 0).astype(np.uint8), connectivity=4)
    if num <= 1:
        return (mask01 > 0).astype(np.uint8)
    keep = np.zeros_like(mask01, dtype=np.uint8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= int(area_min):
            keep[labels == i] = 1
    return keep


def make_nv_vh(ndvi: np.ndarray, gndvi: np.ndarray, s_fused: np.ndarray, texture: np.ndarray, rules: SeedRules) -> Tuple[np.ndarray, np.ndarray]:
    ndvi = _clip01(ndvi)
    gndvi = _clip01(gndvi)
    s_fused = _clip01(s_fused)
    texture = _clip01(texture)

    nv = ((ndvi <= rules.nv_ndvi_max) &
          (gndvi <= rules.nv_gndvi_max) &
          (s_fused <= rules.nv_s_fused_max) &
          (texture <= rules.nv_texture_max)).astype(np.uint8)
    nv = _clean_small(nv, rules.nv_area_min)
    if rules.nv_open_radius > 0:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rules.nv_open_radius + 1, 2 * rules.nv_open_radius + 1))
        nv = cv2.morphologyEx(nv, cv2.MORPH_OPEN, se)

    vh = ((ndvi >= rules.vh_ndvi_min) &
          (s_fused <= rules.vh_s_fused_max) &
          (texture <= rules.vh_texture_max)).astype(np.uint8)
    vh = _clean_small(vh, rules.vh_area_min)
    if rules.vh_open_radius > 0:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rules.vh_open_radius + 1, 2 * rules.vh_open_radius + 1))
        vh = cv2.morphologyEx(vh, cv2.MORPH_OPEN, se)

    return nv.astype(np.uint8), vh.astype(np.uint8)


def generate_seeds_for_pair(indices_dir: Path, boundary_dir: Path, pairkey: str, cfg_seeds: Dict) -> Tuple[np.ndarray, np.ndarray]:
    ndvi, _ = read_raster(indices_dir / "NDVI" / f"{pairkey}_NDVI.tif")
    gndvi, _ = read_raster(indices_dir / "GNDVI" / f"{pairkey}_GNDVI.tif")
    s_fused, _ = read_raster(boundary_dir / f"{pairkey}_S_fused.tif")

    ndvi = ensure_float01(ndvi)
    gndvi = ensure_float01(gndvi)
    s_fused = ensure_float01(s_fused)
    tex = local_std(s_fused, ksize=9)

    rules = SeedRules(
        nv_ndvi_max=float(cfg_seeds.get("NV_strict", {}).get("ndvi_max", 0.05)),
        nv_gndvi_max=float(cfg_seeds.get("NV_strict", {}).get("gndvi_max", 0.05)),
        nv_s_fused_max=float(cfg_seeds.get("NV_strict", {}).get("s_fused_max", 0.15)),
        nv_texture_max=float(cfg_seeds.get("NV_strict", {}).get("texture_max", 0.08)),
        nv_area_min=int(cfg_seeds.get("NV_strict", {}).get("min_area_px", 64)),
        nv_open_radius=int(cfg_seeds.get("NV_strict", {}).get("open_radius", 1)),
        vh_ndvi_min=float(cfg_seeds.get("VH_strict", {}).get("ndvi_min", 0.35)),
        vh_s_fused_max=float(cfg_seeds.get("VH_strict", {}).get("s_fused_max", 0.20)),
        vh_texture_max=float(cfg_seeds.get("VH_strict", {}).get("texture_max", 0.10)),
        vh_area_min=int(cfg_seeds.get("VH_strict", {}).get("min_area_px", 64)),
        vh_open_radius=int(cfg_seeds.get("VH_strict", {}).get("open_radius", 1)),
    )
    return make_nv_vh(ndvi, gndvi, s_fused, tex, rules)


__all__ = [
    "SeedRules",
    "make_nv_vh",
    "generate_seeds_for_pair",
]
