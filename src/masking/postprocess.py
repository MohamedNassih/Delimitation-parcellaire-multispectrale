"""Post-process: skeleton + buffer (r=1), CRF optionnel (désactivé par défaut).
Étape 3 — Doc 04
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cv2  # type: ignore
from skimage.morphology import skeletonize  # type: ignore


def _to_u8(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.uint8)


@dataclass
class PostCfg:
    skeleton_enabled: bool = True
    buffer_radius: int = 1
    closing_radius: int = 0
    crf_enabled: bool = False  # pydensecrf optionnel (non utilisé par défaut)


def skeleton_and_buffer(boundary01: np.ndarray, cfg: PostCfg) -> np.ndarray:
    m = _to_u8(boundary01)
    if cfg.skeleton_enabled:
        m = skeletonize(m.astype(bool), method="lee").astype(np.uint8)
    r = int(cfg.buffer_radius)
    if r > 0:
        k = 2 * r + 1
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.dilate(m, se)
    if cfg.closing_radius and cfg.closing_radius > 0:
        k = 2 * cfg.closing_radius + 1
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, se)
    return m.astype(np.uint8)


__all__ = [
    "PostCfg",
    "skeleton_and_buffer",
]
