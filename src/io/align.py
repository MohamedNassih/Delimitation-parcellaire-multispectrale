"""Alignment utilities (ECC → fallback ORB+RANSAC), bilinear warp.
Étape 1 — Alignement (Doc 02)

- Référence : bande RED (par défaut) ; autres bandes alignées dessus
- ECC (translation/euclidienne/affine) -> warpAffine
- Fallback ORB + RANSAC -> homographie -> warpPerspective
- Métrique d'alignement : SSIM sur cartes d'arêtes (Sobel)

Toutes les images doivent être float32 en [0,1], H=W=1024.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import cv2  # type: ignore
from skimage.metrics import structural_similarity as ssim  # type: ignore


@dataclass
class ECCParams:
    enabled: bool = True
    warp_mode: str = "euclidean"  # {translation, euclidean, affine}
    n_iter: int = 100
    term_eps: float = 1e-6
    gauss_preblur: float = 1.0  # sigma (0=off)
    pyr_levels: int = 4         # accepté depuis la config (non utilisé explicitement ici)


@dataclass
class ORBParams:
    enabled: bool = True
    n_features: int = 3000
    scale_factor: float = 1.2
    n_levels: int = 8
    fastThreshold: int = 0
    ransacReprojThreshold: float = 3.0
    maxIters: int = 5000
    confidence: float = 0.999


@dataclass
class AlignConfig:
    reference_band: str = "RED"
    ecc: ECCParams = field(default_factory=ECCParams)
    orb: ORBParams = field(default_factory=ORBParams)


# ------------------------------
# Helpers
# ------------------------------

def _to_gray01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    if x.ndim == 3:
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    x = np.clip(x, 0.0, 1.0)
    return x


def _edges01(x: np.ndarray) -> np.ndarray:
    x = _to_gray01(x)
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    m = float(mag.max()) if np.isfinite(mag.max()) else 0.0
    if m <= 1e-12:
        return np.zeros_like(mag, dtype=np.float32)
    return (mag / m).astype(np.float32, copy=False)


def _ssim_edges(a01: np.ndarray, b01: np.ndarray) -> float:
    ea = _edges01(a01)
    eb = _edges01(b01)
    val = float(ssim(ea, eb, data_range=1.0))
    return val


# ------------------------------
# ECC alignment
# ------------------------------

def _warp_mode_cv(mode: str) -> int:
    mode = str(mode).lower()
    if mode == "translation":
        return cv2.MOTION_TRANSLATION
    if mode == "euclidean":
        return cv2.MOTION_EUCLIDEAN
    if mode == "affine":
        return cv2.MOTION_AFFINE
    # défaut raisonnable
    return cv2.MOTION_EUCLIDEAN


def ecc_align(ref01: np.ndarray, mov01: np.ndarray, p: ECCParams) -> Tuple[bool, np.ndarray, np.ndarray, float]:
    """Aligne mov→ref via ECC. Retourne (ok, warp_matrix, warped, score)."""
    ref = _to_gray01(ref01)
    mov = _to_gray01(mov01)

    if p.gauss_preblur and p.gauss_preblur > 0:
        k = int(max(3, int(2 * round(p.gauss_preblur * 3) + 1)))
        ref_blur = cv2.GaussianBlur(ref, (k, k), p.gauss_preblur)
        mov_blur = cv2.GaussianBlur(mov, (k, k), p.gauss_preblur)
    else:
        ref_blur, mov_blur = ref, mov

    warp_mode = _warp_mode_cv(p.warp_mode)
    if warp_mode in (cv2.MOTION_TRANSLATION, cv2.MOTION_EUCLIDEAN, cv2.MOTION_AFFINE):
        warp_mat = np.eye(2, 3, dtype=np.float32)
    else:  # non utilisé ici
        warp_mat = np.eye(3, 3, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(p.n_iter), float(p.term_eps))

    try:
        cc, warp_mat = cv2.findTransformECC(
            ref_blur,
            mov_blur,
            warp_mat,
            warp_mode,
            criteria,
            None,
            1
        )
        h, w = ref.shape
        warped = cv2.warpAffine(mov01, warp_mat, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT101)
        score = _ssim_edges(ref01, warped)
        return True, warp_mat, warped.astype(np.float32, copy=False), float(score)
    except cv2.error as e:
        logging.warning("ECC failed: %s", e)
        return False, warp_mat, mov01, 0.0


# ------------------------------
# ORB + RANSAC fallback
# ------------------------------

def orb_fallback_align(ref01: np.ndarray, mov01: np.ndarray, p: ORBParams) -> Tuple[bool, np.ndarray, np.ndarray, int, int]:
    """Aligne mov→ref via ORB+RANSAC. Retourne (ok, H(3x3), warped, inliers, total_matches)."""
    ref8 = (np.clip(ref01, 0, 1) * 255.0).astype(np.uint8)
    mov8 = (np.clip(mov01, 0, 1) * 255.0).astype(np.uint8)

    orb = cv2.ORB_create(nfeatures=p.n_features, scaleFactor=p.scale_factor, nlevels=p.n_levels, fastThreshold=p.fastThreshold)
    kpa, dea = orb.detectAndCompute(ref8, None)
    kpb, deb = orb.detectAndCompute(mov8, None)
    if dea is None or deb is None or len(kpa) < 4 or len(kpb) < 4:
        return False, np.eye(3, dtype=np.float32), mov01, 0, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(dea, deb)
    if len(matches) < 4:
        return False, np.eye(3, dtype=np.float32), mov01, 0, len(matches)

    matches = sorted(matches, key=lambda m: m.distance)[:500]
    src = np.float32([kpa[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kpb[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(dst, src, cv2.RANSAC, ransacReprojThreshold=p.ransacReprojThreshold, maxIters=p.maxIters, confidence=p.confidence)
    if H is None:
        return False, np.eye(3, dtype=np.float32), mov01, 0, len(matches)

    inliers = int(mask.sum()) if mask is not None else 0
    h, w = ref8.shape[:2]
    warped = cv2.warpPerspective(mov01, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    return True, H.astype(np.float32), warped.astype(np.float32, copy=False), inliers, len(matches)


# ------------------------------
# Orchestrateur : essaie ECC puis ORB
# ------------------------------

def align_to_ref(ref01: np.ndarray, mov01: np.ndarray, cfg: AlignConfig) -> Tuple[np.ndarray, dict]:
    """Aligne une bande mov sur ref. Retourne (warped, info)."""
    # 1) ECC (si activé)
    if cfg.ecc.enabled:
        ok, W, warped, score = ecc_align(ref01, mov01, cfg.ecc)
        info = {"method": "ECC", "ok": bool(ok), "ssim_edges": float(score)}
        if ok:
            info.update({"warp": W.tolist()})
            return warped, info
    else:
        info = {"method": "ECC", "ok": False, "reason": "disabled", "ssim_edges": float(_ssim_edges(ref01, mov01))}

    # 2) Fallback ORB+RANSAC (si activé)
    if cfg.orb.enabled:
        ok2, H, warped2, inl, tot = orb_fallback_align(ref01, mov01, cfg.orb)
        info = {
            "method": "ORB_RANSAC",
            "ok": bool(ok2),
            "inliers": int(inl),
            "matches": int(tot),
        }
        if ok2:
            info["ssim_edges"] = float(_ssim_edges(ref01, warped2))
            info["homography"] = H.tolist()
            return warped2, info
    else:
        info = {"method": "ORB_RANSAC", "ok": False, "reason": "disabled", "ssim_edges": float(_ssim_edges(ref01, mov01))}

    # 3) Échec : renvoyer l'original
    info["ssim_edges"] = float(_ssim_edges(ref01, mov01))
    return mov01, info


__all__ = [
    "ECCParams",
    "ORBParams",
    "AlignConfig",
    "align_to_ref",
    "ecc_align",
    "orb_fallback_align",
]
