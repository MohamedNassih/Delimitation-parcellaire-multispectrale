"""Filtre par blocs 50×50 + règle stricte 4‑voisins.
Étape 3 — Doc 04

Objectif :
- Caler le ratio de pixels blancs (boundary=1) ~1–8%.
- Éviter la sur-propagation locale en réduisant les îlots denses.

Stratégie :
- Itérer par blocs (B=block_size).
- Si white_ratio > white_ratio_max : squelettiser localement (thinning)
  et ne garder que les composantes qui touchent le bord du bloc.
- Connectivité strictement 4-voisins.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import cv2  # type: ignore
from skimage.morphology import skeletonize  # type: ignore


def _to_u8(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.uint8)


@dataclass
class BlockFilterCfg:
    block_size: int = 50
    white_ratio_max: float = 0.08
    soften_margin_px: int = 2  # lissage léger aux bords des blocs


def _keep_components_touching_border(mask_u8: np.ndarray) -> np.ndarray:
    h, w = mask_u8.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=4)
    if num <= 1:
        return mask_u8
    keep = np.zeros_like(mask_u8)
    for i in range(1, num):
        ys, xs = np.where(labels == i)
        if ys.size == 0:
            continue
        if (ys.min() == 0) or (ys.max() == h - 1) or (xs.min() == 0) or (xs.max() == w - 1):
            keep[labels == i] = 1
    return keep


def block_filter(boundary01: np.ndarray, cfg: BlockFilterCfg) -> np.ndarray:
    b = int(cfg.block_size)
    h, w = boundary01.shape
    out = _to_u8(boundary01)

    for y0 in range(0, h, b):
        for x0 in range(0, w, b):
            y1 = min(y0 + b, h)
            x1 = min(x0 + b, w)
            tile = out[y0:y1, x0:x1]
            white_ratio = float(tile.mean())
            if white_ratio <= cfg.white_ratio_max:
                continue
            # squelettiser localement pour réduire la densité
            sk = skeletonize(tile.astype(bool), method="lee").astype(np.uint8)
            sk = _keep_components_touching_border(sk)
            out[y0:y1, x0:x1] = sk

    # adoucir contours de blocs
    if cfg.soften_margin_px > 0:
        k = 2 * cfg.soften_margin_px + 1
        se = cv2.getStructuringElement(cv2.MORPH_CROSS, (k, k))  # 4-voisins
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, se)
    return out.astype(np.uint8)


__all__ = [
    "BlockFilterCfg",
    "block_filter",
]
