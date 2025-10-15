"""Garder le plus grand groupe noir (field=0) et mettre le reste en blanc (boundary=1).
Étape 3 — Doc 04
"""
from __future__ import annotations

import numpy as np
import cv2  # type: ignore


def keep_largest_black(boundary01: np.ndarray) -> np.ndarray:
    """Conserve la plus grande composante *noire* (0) et met le reste en blanc (1).
    Retourne un mask binaire uint8 {0,1}.
    """
    inv = (boundary01 == 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    if num <= 1:
        return (boundary01 > 0).astype(np.uint8)
    # trouver l'aire max (ignorer 0=background de 'inv')
    best_idx, best_area = 1, -1
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area, best_idx = area, i
    keep = (labels == best_idx).astype(np.uint8)
    # pixels non gardés -> boundary blanc
    out = np.where(keep == 1, 0, 1).astype(np.uint8)
    return out


__all__ = ["keep_largest_black"]
