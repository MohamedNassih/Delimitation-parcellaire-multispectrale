"""CLI — Étape 2: Indices & Arêtes -> S_fused + seeds NV/VH

- lit les TIF alignés (artifacts/aligned/{REG,RED,NIR,GRE})
- calcule indices spectraux (NDVI, GNDVI, NDRE, BRVI, LR)
- calcule opérateurs d'arêtes (Sobel, LoG, tenseur de structure, morph grad)
- standardise (min-max) chaque carte si demandé
- fusion pondérée -> S_fused (écrit en GeoTIFF)
- génère des seeds NV/VH (option Étape 2) pour debug
- écrit stats CSV et quicklook PNG

Usage :
python -m src.cli.make_indices --aligned artifacts/aligned --out artifacts/indices --cfg configs/indices.yaml
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from src.io.readwrite import read_raster, write_raster, ensure_float01, save_debug_png
from src.indices.spectral import compute_all as spectral_all, SpectralConfig, local_std
from src.indices.gradients import compute_all_edges
from src.boundaries.fuse import standardize_maps, fuse_weighted
from src.boundaries.watershed import make_seeds


# ------------------------------
# Helpers
# ------------------------------

def _band_paths_for_pair(aligned_root: Path, pairkey: str, suffixes: Dict[str, str]) -> Dict[str, Path]:
    return {
        band: aligned_root / band / f"{pairkey}{suffixes[band]}"
        for band in ["REG","RED","NIR","GRE"]
    }


def _iter_pairkeys(aligned_root: Path, suffix_red: str) -> List[str]:
    red_dir = aligned_root / "RED"
    keys: List[str] = []
    for p in red_dir.glob(f"*{suffix_red}"):
        stem = p.stem
        # retire suffixe "_RED" seulement à la fin
        if stem.endswith("_RED"):
            pk = stem[: -len("_RED")]
        else:
            # fallback : retire suffixe exact du config
            pk = stem.replace(suffix_red.replace(".tif",""), "")
        keys.append(pk)
    return sorted(set(keys))


# ------------------------------
# CLI
# ------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute spectral indices, edges and fused boundary map")
    ap.add_argument("--cfg", type=Path, default=Path("configs/indices.yaml"))
    ap.add_argument("--aligned", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log-level", default="INFO")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="[%(levelname)s] %(message)s")

    cfg = yaml.safe_load(open(args.cfg, "r", encoding="utf-8"))
    io = cfg["io"]
    numerics = cfg.get("numerics", {})
    spectral_cfg = cfg.get("spectral_indices", {})
    edge_cfg = cfg.get("edge_operators", {})
    fusion_cfg = cfg.get("fusion", {})
    seeds_cfg = cfg.get("seeds", {})
    outputs_cfg = cfg.get("outputs", {})

    eps = float(numerics.get("eps", 1e-6))
    std_each = bool(numerics.get("standardize_each_map", True))

    aligned_root = Path(str(args.aligned))
    indices_dir = Path(io.get("indices_dir", "artifacts/indices"))
    boundary_dir = Path(io.get("boundary_maps_dir", "artifacts/boundary_maps"))
    reports_dir = Path(io.get("reports_dir", "artifacts/reports"))
    for d in [indices_dir, boundary_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    suffixes = {
        "REG": "_REG.tif",
        "RED": "_RED.tif",
        "NIR": "_NIR.tif",
        "GRE": "_GRE.tif",
    }
    # utilise la config prepare si dispo
    if "assert_size" in io:
        pass

    pairkeys = _iter_pairkeys(aligned_root, suffixes["RED"])
    logging.info("Found %d aligned pairkeys", len(pairkeys))

    rows_stats: List[Dict] = []

    for pk in pairkeys:
        try:
            # --- Lire bandes alignées (déjà float32 [0,1]) ---
            paths = _band_paths_for_pair(aligned_root, pk, suffixes)
            REG, _ = read_raster(paths["REG"])  # profile ignoré ici
            RED, _ = read_raster(paths["RED"])  # type: ignore
            NIR, _ = read_raster(paths["NIR"])  # type: ignore
            GRE, _ = read_raster(paths["GRE"])  # type: ignore

            REG = ensure_float01(REG)
            RED = ensure_float01(RED)
            NIR = ensure_float01(NIR)
            GRE = ensure_float01(GRE)

            # --- Indices spectraux ---
            spec = spectral_all(REG, RED, NIR, GRE, SpectralConfig(eps=eps), enabled={k: bool(v.get("enabled", True)) for k, v in spectral_cfg.items()})

            # --- Arêtes (on utilise RED comme référence d'arêtes par défaut) ---
            edges = compute_all_edges(RED, edge_cfg)

            # --- Maps -> (optionnel) standardisation min-max ---
            maps = {}
            maps.update(spec)
            maps.update(edges)
            maps_std = standardize_maps(maps, enable=std_each)

            # --- Fusion ---
            weights = {k: float(v) for k, v in fusion_cfg.get("weights", {}).items()}
            s_fused = fuse_weighted(maps_std, weights, smooth_gaussian_sigma=float(fusion_cfg.get("smooth_gaussian_sigma", 0.0)), clip01=bool(fusion_cfg.get("clip01", True)))

            # --- Texture locale pour seeds ---
            tex = local_std(s_fused, ksize=9)

            # --- Seeds NV/VH (debug de l'étape 2) ---
            nv, vh = make_seeds(spec.get("NDVI", RED), spec.get("GNDVI", RED), s_fused, tex, cfg.get("seeds", {}))

            # --- Sauvegardes ---
            fused_path = boundary_dir / f"{pk}{outputs_cfg.get('fused_suffix', '_S_fused.tif')}"
            write_raster(fused_path, s_fused, profile_like=None)

            # indices/edges individuels
            for name, arr in {**spec, **edges}.items():
                out_p = indices_dir / name
                out_p.mkdir(parents=True, exist_ok=True)
                write_raster(out_p / f"{pk}_{name}.tif", arr)

            # quicklooks
            if outputs_cfg.get("save_png_quicklook", True):
                save_debug_png(boundary_dir / f"{pk}_S_fused.png", s_fused)
                save_debug_png(boundary_dir / f"{pk}_NV.png", (nv > 0).astype(np.float32))
                save_debug_png(boundary_dir / f"{pk}_VH.png", (vh > 0).astype(np.float32))

            # stats
            if outputs_cfg.get("save_csv_stats", True):
                def _stats(x: np.ndarray) -> Dict[str, float]:
                    return {"min": float(np.min(x)), "p1": float(np.percentile(x,1)), "mean": float(np.mean(x)), "p99": float(np.percentile(x,99)), "max": float(np.max(x))}
                rows_stats.append({"pairkey": pk, "map": "S_fused", **_stats(s_fused)})
                for nm, arr in {**spec, **edges}.items():
                    rows_stats.append({"pairkey": pk, "map": nm, **_stats(arr)})

        except Exception as e:
            logging.exception("Failed %s: %s", pk, e)
            continue

    if rows_stats:
        pd.DataFrame(rows_stats).to_csv(Path(io.get("reports_dir", "artifacts/reports")) / "indices_stats.csv", index=False)

    logging.info("Done. S_fused -> %s, indices -> %s", boundary_dir, indices_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
