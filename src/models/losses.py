"""Losses pour segmentation de frontières (boundary=1, field=0).
Étape 4 — Doc 05

Implémentées:
- BCE (sur proba) + Dice (soft) combinées -> bce_dice
- Tversky (alpha, beta) -> tversky_loss
- Surface (distance-transform) -> surface_loss (optionnelle)

Notes:
- Le modèle sort des probabilités (Sigmoid). Pas de BCEWithLogits ici.
- Toutes les pertes renvoient un scalaire (moyenne batch).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as edt

EPS = 1e-6


def _flatten_probs_targets(p: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convertit en (N, HW)."""
    p = p.float().clamp(0.0, 1.0)
    y = y.float().clamp(0.0, 1.0)
    return p.view(p.size(0), -1), y.view(y.size(0), -1)


def bce_loss(p: torch.Tensor, y: torch.Tensor, weight: float | None = None) -> torch.Tensor:
    if weight is None:
        return F.binary_cross_entropy(p, y)
    return F.binary_cross_entropy(p, y, weight=torch.full_like(p, float(weight)))


def dice_loss(p: torch.Tensor, y: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    p, y = _flatten_probs_targets(p, y)
    inter = (p * y).sum(dim=1)
    denom = p.sum(dim=1) + y.sum(dim=1)
    dice = (2.0 * inter + smooth) / (denom + smooth + EPS)
    return (1.0 - dice).mean()


def bce_dice(p: torch.Tensor, y: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    return float(alpha) * bce_loss(p, y) + (1.0 - float(alpha)) * dice_loss(p, y)


def tversky_loss(p: torch.Tensor, y: torch.Tensor, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1.0) -> torch.Tensor:
    p, y = _flatten_probs_targets(p, y)
    tp = (p * y).sum(dim=1)
    fp = (p * (1.0 - y)).sum(dim=1)
    fn = ((1.0 - p) * y).sum(dim=1)
    t = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth + EPS)
    return (1.0 - t).mean()


# -------- Surface loss (distance to GT boundary) ---------
@dataclass
class SurfaceCfg:
    normalize: bool = True


def _distance_map_from_boundary_numpy(gt01: np.ndarray) -> np.ndarray:
    """Distance (en pixels) au plus proche pixel de frontière gt=1.
    gt01: (H,W) uint8/bool/float in {0,1}
    Retourne float32.
    """
    g = (gt01 > 0.5).astype(np.uint8)
    if g.max() == 0:
        # pas de frontière -> distance nulle
        return np.zeros_like(g, dtype=np.float32)
    # distance au bord (pixels non-frontière -> distance >0 ; frontière -> 0)
    dist = edt(1 - g)
    return dist.astype(np.float32)


def surface_loss(p: torch.Tensor, y: torch.Tensor, cfg: SurfaceCfg | None = None) -> torch.Tensor:
    """Pénalise les faux positifs proportionnellement à leur distance à la vraie frontière.
    Approche simple et CPU-friendly.
    """
    if cfg is None:
        cfg = SurfaceCfg()
    with torch.no_grad():
        dm_list = []
        for i in range(y.size(0)):
            dm = _distance_map_from_boundary_numpy(y[i, 0].detach().cpu().numpy())
            if cfg.normalize and dm.max() > 0:
                dm = dm / float(dm.max())
            dm_list.append(torch.from_numpy(dm))
        dm_t = torch.stack(dm_list, dim=0).to(y.device).unsqueeze(1)  # (N,1,H,W)
    # faux positifs loin coûtent plus cher
    return (p * dm_t).mean()


__all__ = [
    "bce_loss",
    "dice_loss",
    "bce_dice",
    "tversky_loss",
    "SurfaceCfg",
    "surface_loss",
]
