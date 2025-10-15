"""BF-score (Boundary F1) avec tolérance r (+ IoU/Dice standard).
Étape 4 — Doc 05

Convention masque: 1=boundary, 0=field.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import cv2  # type: ignore

EPS = 1e-6


def _to_u8(x: np.ndarray) -> np.ndarray:
    return (x > 0.5).astype(np.uint8)


def _disk(radius: int) -> np.ndarray:
    r = int(max(0, radius))
    if r <= 0:
        return np.ones((1, 1), np.uint8)
    k = 2 * r + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


@dataclass
class BFConfig:
    radius: int = 2
    threshold: float = 0.5  # seuillage des probabilités


def bfscore(y_true01: np.ndarray, y_prob: np.ndarray, cfg: BFConfig) -> Dict[str, float]:
    """Calcule précision, rappel, F1 avec tolérance r pixels.
    y_true01: {0,1}
    y_prob: proba [0,1]
    """
    y_true = _to_u8(y_true01)
    y_pred = _to_u8(y_prob)

    se = _disk(cfg.radius)
    # P correspond au dilate(G) et G correspond au dilate(P) (tolerance matching)
    Gd = cv2.dilate(y_true, se)
    Pd = cv2.dilate(y_pred, se)

    tp = int((y_pred & Gd).sum())
    fp = int((y_pred & (1 - Gd)).sum())
    fn = int((y_true & (1 - Pd)).sum())

    prec = tp / (tp + fp + EPS)
    rec = tp / (tp + fn + EPS)
    f1 = 2 * prec * rec / (prec + rec + EPS)

    # Metrics de remplissage (sévérité sans tolérance) — optionnel
    inter = int((y_true & y_pred).sum())
    union = int((y_true | y_pred).sum())
    iou = inter / (union + EPS)
    dice = 2 * inter / (y_true.sum() + y_pred.sum() + EPS)

    return {"precision": float(prec), "recall": float(rec), "bf": float(f1), "iou": float(iou), "dice": float(dice)}


__all__ = ["BFConfig", "bfscore"]
