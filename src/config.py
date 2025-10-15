"""
Config loader & utilities — AgriEdge (CPU-only)
- Charge les YAML de config
- Assure la création des dossiers artifacts/logs
- Journalise les versions des libs (optionnel)
- Seed/déterminisme de base

Convention masque : 1 = boundary, 0 = field
"""
from __future__ import annotations

import os
import sys
import csv
import json
import time
import yaml
import hashlib
import pathlib
import platform
from datetime import datetime
from typing import Any, Dict, Iterable

import numpy as np

# Libs optionnelles (log des versions si présentes)
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None
try:
    import rasterio  # type: ignore
except Exception:  # pragma: no cover
    rasterio = None
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None
try:
    import skimage  # type: ignore
except Exception:  # pragma: no cover
    skimage = None

# ------------------------------
# YAML utils
# ------------------------------

def load_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(paths: Iterable[str | os.PathLike]) -> None:
    for p in paths:
        pathlib.Path(p).mkdir(parents=True, exist_ok=True)


# ------------------------------
# Seeding / deterministic knobs (best-effort CPU)
# ------------------------------

def set_seed(seed: int = 2025) -> None:
    np.random.seed(seed)
    # Si PyTorch/TensorFlow ajoutés à l'étape 4, régler leurs seeds ici.


# ------------------------------
# Version logging
# ------------------------------

def _versions_dict() -> Dict[str, str]:
    v = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": getattr(np, "__version__", "-"),
        "opencv": getattr(cv2, "__version__", "-") if cv2 is not None else "-",
        "rasterio": getattr(rasterio, "__version__", "-") if rasterio is not None else "-",
        "pandas": getattr(pd, "__version__", "-") if pd is not None else "-",
        "skimage": getattr(skimage, "__version__", "-") if skimage is not None else "-",
    }
    return v


def log_versions(logs_dir: str, write_csv: bool = True) -> Dict[str, str]:
    ensure_dirs([logs_dir])
    v = _versions_dict()
    v["timestamp"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    if write_csv:
        csv_path = pathlib.Path(logs_dir) / "runs.csv"
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(v.keys()))
            if write_header:
                w.writeheader()
            w.writerow(v)
    return v


# ------------------------------
# High-level project config loader
# ------------------------------

def load_project_config(path: str = "configs/project.yaml") -> Dict[str, Any]:
    cfg = load_yaml(path)
    # Crée les dossiers principaux
    paths = cfg.get("paths", {})
    ensure_dirs([
        paths.get("artifacts_root", "artifacts"),
        paths.get("aligned_dir", "artifacts/aligned"),
        paths.get("indices_dir", "artifacts/indices"),
        paths.get("boundary_maps_dir", "artifacts/boundary_maps"),
        paths.get("masks_raw_dir", "artifacts/masks_raw"),
        paths.get("masks_filtered_dir", "artifacts/masks_filtered"),
        paths.get("masks_final_dir", "artifacts/masks_final"),
        paths.get("patches_dir", "artifacts/patches"),
        paths.get("reports_dir", "artifacts/reports"),
        paths.get("logs_dir", "logs"),
    ])

    # Seed/déterminisme
    set_seed(cfg.get("seed", 2025))

    # Log des versions (optionnel via project.logging.write_versions)
    logging_cfg = cfg.get("logging", {})
    if logging_cfg.get("write_versions", True):
        log_versions(paths.get("logs_dir", "logs"), write_csv=True)

    return cfg


if __name__ == "__main__":
    # Petit test manuel
    pcfg = load_project_config()
    print("Project loaded. Mask convention:", pcfg["project"]["mask_convention"])