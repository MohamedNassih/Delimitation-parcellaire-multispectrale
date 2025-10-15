"""I/O raster + normalisation + resize (Étape 1 — Doc 02)

Fonctions utilitaires CPU‑only pour lire/écrire des rasters (GeoTIFF),
normaliser par percentiles p2–p98, forcer float32 [0,1],
redimensionner à 1024×1024 en bilinéaire, et calculer des métriques
simples d'intégrité (saturation/blur).

Dépendances : rasterio, tifffile, numpy, opencv-python, pandas (tests).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# Backends d'I/O (tous optionnels selon l'environnement)
try:
    import rasterio  # type: ignore
    from rasterio.transform import Affine  # type: ignore
except Exception:  # pragma: no cover
    rasterio = None  # type: ignore
    Affine = None  # type: ignore

try:
    import tifffile  # type: ignore
except Exception:  # pragma: no cover
    tifffile = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


# ------------------------------
# Lecture / écriture
# ------------------------------

def read_raster(path: str | Path) -> Tuple[np.ndarray, Optional[Dict]]:
    """Lit un raster simple bande.

    Retourne (array, profile) où `profile` est un dict rasterio si dispo,
    sinon None. L'array est retourné en dtype d'origine (converti ensuite
    par les fonctions de normalisation si besoin).
    """
    p = Path(path)
    if rasterio is not None:
        try:
            with rasterio.open(p) as ds:  # type: ignore[attr-defined]
                arr = ds.read(1)
                profile = ds.profile
                return arr, profile
        except Exception as e:  # fallback vers tifffile
            logging.warning("rasterio read failed for %s: %s (fallback tifffile)", p, e)
    if tifffile is None:
        raise RuntimeError("No backend available to read TIFF (need rasterio or tifffile)")
    arr = tifffile.imread(str(p))  # type: ignore[union-attr]
    if arr.ndim == 3:  # si RGB accidentel => prendre la 1ère bande
        arr = arr[..., 0]
    return arr, None


def _default_profile_like(profile: Optional[Dict], arr: np.ndarray) -> Dict:
    prof = {
        "driver": "GTiff",
        "height": int(arr.shape[0]),
        "width": int(arr.shape[1]),
        "count": 1,
        "dtype": str(arr.dtype),
        "compress": "deflate",
        "predictor": 2,
        "tiled": False,
    }
    if profile:
        # hérite du geo-transform/CRS si fourni
        for k in ("transform", "crs"):
            if k in profile and profile[k] is not None:
                prof[k] = profile[k]
    return prof


def write_raster(path: str | Path, arr: np.ndarray, profile_like: Optional[Dict] = None) -> None:
    """Écrit un GeoTIFF mono-bande (float32 recommandé)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    prof = _default_profile_like(profile_like, arr)
    prof["dtype"] = str(arr.dtype)

    if rasterio is not None:
        with rasterio.open(p, "w", **prof) as dst:  # type: ignore[arg-type]
            dst.write(arr, 1)
        return
    if tifffile is None:
        raise RuntimeError("No backend available to write TIFF (need rasterio or tifffile)")
    tifffile.imwrite(str(p), arr, compression="zlib")  # type: ignore[union-attr]


# ------------------------------
# Normalisation p2–p98 et conversions
# ------------------------------

def normalize_p2p98(img: np.ndarray, p_low: float = 2, p_high: float = 98, eps: float = 1e-6, clip: bool = True) -> np.ndarray:
    """Min–max entre percentiles (p_low, p_high). Renvoie float32 in [0,1].
    Protège contre upper==lower via eps.
    """
    img = img.astype(np.float32, copy=False)
    lo = float(np.nanpercentile(img, p_low))
    hi = float(np.nanpercentile(img, p_high))
    denom = max(hi - lo, eps)
    out = (img - lo) / denom
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32, copy=False)


def ensure_float01(img: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Assure float32 dans [0,1] (clip)."""
    if not np.issubdtype(img.dtype, np.floating):
        img = img.astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    # stabilise les NaN/Inf si présents
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    return img.astype(np.float32, copy=False)


# ------------------------------
# Resize
# ------------------------------

def resize_to(img: np.ndarray, size_hw: Tuple[int, int], mode: str = "bilinear") -> np.ndarray:
    """Resize H×W -> size_hw en OpenCV. Retourne float32.
    mode: {bilinear, nearest, area, cubic}
    """
    if cv2 is None:
        raise RuntimeError("OpenCV is required for resize")
    h, w = img.shape[:2]
    th, tw = int(size_hw[0]), int(size_hw[1])
    if (h, w) == (th, tw):
        return img.astype(np.float32, copy=False)
    inter = {
        "bilinear": cv2.INTER_LINEAR,
        "nearest": cv2.INTER_NEAREST,
        "area": cv2.INTER_AREA,
        "cubic": cv2.INTER_CUBIC,
    }.get(mode, cv2.INTER_LINEAR)
    out = cv2.resize(img, (tw, th), interpolation=inter)
    return out.astype(np.float32, copy=False)


# ------------------------------
# Intégrité simple (post-normalisation)
# ------------------------------

def integrity_metrics(img01: np.ndarray, eps: float = 1e-6) -> dict:
    """Calcule des métriques simples sur une image normalisée [0,1].
    - saturation_low/high: parts de pixels ~0 et ~1
    - blur_var: variance du Laplacien (plus petit => plus flou)

    Notes Windows/AVX2: certaines builds d'OpenCV lèvent
    "Unsupported combination of source/destination format" si l'entrée
    est float32 et la sortie float64. On force donc ddepth=CV_32F et
    on rend l'array contigu en mémoire.
    """
    x = ensure_float01(img01, eps)
    n = x.size
    sat_low = float((x <= eps).sum()) / n
    sat_high = float((x >= 1.0 - eps).sum()) / n
    if cv2 is None:
        blur_var = float(np.var(x))  # fallback grossier
    else:
        try:
            x32 = np.ascontiguousarray(x, dtype=np.float32)
            lap = cv2.Laplacian(x32, cv2.CV_32F, ksize=3)
            blur_var = float(lap.var())
        except Exception as e:  # dernier recours
            logging.warning("Laplacian failed: %s; using np.var fallback", e)
            blur_var = float(np.var(x))
    return {"saturation_low": sat_low, "saturation_high": sat_high, "blur_var": blur_var}


# ------------------------------
# Pipeline élémentaire (lecture -> normalisation -> resize)
# ------------------------------

def read_normalize_resize(
    path: str | Path,
    target_hw: Tuple[int, int] = (1024, 1024),
    p_low: float = 2,
    p_high: float = 98,
    eps: float = 1e-6,
    resize_mode: str = "bilinear",
) -> Tuple[np.ndarray, Optional[dict], dict]:
    """Lit une bande, normalise (p2–p98), resize -> (H,W) et renvoie
    (img01, profile, metrics) où metrics = saturation/blur.
    """
    img, prof = read_raster(path)
    img01 = normalize_p2p98(img, p_low=p_low, p_high=p_high, eps=eps, clip=True)
    img01 = resize_to(img01, target_hw, mode=resize_mode)
    mets = integrity_metrics(img01, eps=eps)
    return img01.astype(np.float32, copy=False), prof, mets


# ------------------------------
# Debug helper
# ------------------------------

def save_debug_png(path: str | Path, img01: np.ndarray) -> None:
    """Sauvegarde un PNG 8‑bits pour visualisation rapide."""
    if cv2 is None:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    u8 = (np.clip(img01, 0, 1) * 255.0).astype(np.uint8)
    cv2.imwrite(str(p), u8)


__all__ = [
    "read_raster",
    "write_raster",
    "normalize_p2p98",
    "ensure_float01",
    "resize_to",
    "integrity_metrics",
    "read_normalize_resize",
    "save_debug_png",
]
