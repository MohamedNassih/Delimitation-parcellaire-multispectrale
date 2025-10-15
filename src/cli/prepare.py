"""CLI — Étape 1: Inventaire & Alignement (Doc 02)

Pipeline CPU-only :
- scan des bandes -> inventory.csv (re-généré)
- lecture + normalisation p2–p98 + resize 1024×1024 (float32)
- alignement sur RED (ECC -> fallback ORB)
- écritures GeoTIFF alignés par bande dans artifacts/aligned/{REG,RED,NIR,GRE}/
- logs CSV : alignment.csv, integrity.csv (+ manifest quarantine)

Règles d'erreur :
- paires incomplètes -> quarantine et on continue
- ECC échec -> ORB+RANSAC
- tailles inattendues -> forcées 1024×1024 côté I/O
- zéros/NaN -> eps & clips partout
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml
import numpy as np

from src.io.scan import ScanConfig, scan_pairs, write_inventory
from src.io.readwrite import read_normalize_resize, write_raster, integrity_metrics
from src.io.align import AlignConfig, ECCParams, ORBParams, align_to_ref


# ------------------------------
# Helpers
# ------------------------------

def _ensure_dirs(out_root: Path, bands: Dict[str, str], reports_dir: Path) -> None:
    for b in bands.keys():
        (out_root / b).mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)


def _compute_blur8(img01: np.ndarray) -> float:
    """Variance du Laplacien sur version 8-bit (0..255) pour seuils classiques."""
    import cv2  # local import
    u8 = (np.clip(img01, 0, 1) * 255.0).astype(np.uint8)
    lap = cv2.Laplacian(u8, cv2.CV_64F, ksize=3)
    return float(lap.var())


# ------------------------------
# CLI
# ------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare multispectral images: scan, normalize, align, write aligned tifs")
    ap.add_argument("--cfg", type=Path, default=Path("configs/prepare.yaml"))
    ap.add_argument("--root", type=Path, help="Input root with REG/RED/NIR/GRE subfolders")
    ap.add_argument("--out", type=Path, help="Output aligned root (will contain band subfolders)")
    ap.add_argument("--log-level", default="INFO")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="[%(levelname)s] %(message)s")

    prep = yaml.safe_load(open(args.cfg, "r", encoding="utf-8"))
    io = prep["io"]
    out_cfg = prep["output"]
    qual = prep.get("quality", {})
    norm = prep.get("normalize", {})
    resize = prep.get("resize", {})
    align = prep.get("alignment", {})

    # Defaults
    reference_band = align.get("reference_band", "RED")

    # Paths
    root = Path(str(args.root))
    out_root = Path(str(args.out))
    reports_dir = Path(out_cfg.get("reports_dir", "artifacts/reports"))
    _ensure_dirs(out_root, io["bands"], reports_dir)

    # 1) Scan -> inventory.csv
    scfg = ScanConfig(root=root, bands=io["bands"], pair_regex=io["pair_regex"], suffixes=io["band_suffixes"], out_csv=Path(out_cfg["inventory_csv"]), quarantine_dir=Path(out_cfg["quarantine_dir"]))
    df_inv = scan_pairs(scfg)
    write_inventory(df_inv, scfg.out_csv)

    # 2) Prépare align configs
    eccp = ECCParams(**align.get("ecc", {}))
    orbp = ORBParams(**align.get("orb_fallback", {}))
    acfg = AlignConfig(reference_band=reference_band, ecc=eccp, orb=orbp)

    # 3) Listes de logs
    align_rows: List[Dict] = []
    integ_rows: List[Dict] = []

    # 4) Seuils qualité
    sat_max = float(qual.get("saturation_max", 0.02))
    blur_min_u8 = float(qual.get("blur_var_min", 30))  # seuil en 8-bit
    ssim_min = float(qual.get("ssim_edges_min", 0.60))

    # 5) Boucle paires
    complete = df_inv[df_inv.status == "COMPLETE"].copy()
    n = len(complete)
    logging.info("Processing COMPLETE pairs: %d", n)

    for i, row in complete.iterrows():
        pk = row["pairkey"]
        try:
            # a) Lire & préparer la référence
            ref_path = Path(row[reference_band])
            ref01, ref_prof, ref_mets = read_normalize_resize(
                ref_path,
                target_hw=tuple(resize.get("target_size", [1024, 1024])),
                p_low=float(norm.get("p_low", 2)),
                p_high=float(norm.get("p_high", 98)),
                eps=float(norm.get("eps", 1e-6)),
                resize_mode=resize.get("mode", "bilinear"),
            )

            # Integrity + flags (réf)
            ref_blur8 = _compute_blur8(ref01)
            integ_rows.append({
                "pairkey": pk,
                "band": reference_band,
                "saturation_low": ref_mets.get("saturation_low", np.nan),
                "saturation_high": ref_mets.get("saturation_high", np.nan),
                "blur_var": ref_mets.get("blur_var", np.nan),
                "blur_var_u8": ref_blur8,
                "flag_saturation": int(ref_mets.get("saturation_low", 0) > sat_max or ref_mets.get("saturation_high", 0) > sat_max),
                "flag_blur": int(ref_blur8 < blur_min_u8),
            })

            # Écriture REF (identité)
            out_ref = out_root / reference_band / f"{pk}{io['band_suffixes'][reference_band]}"
            write_raster(out_ref, ref01.astype(np.float32), profile_like=ref_prof)
            align_rows.append({
                "pairkey": pk,
                "band": reference_band,
                "method": "IDENTITY",
                "ok": 1,
                "ssim_edges": 1.0,
                "out_path": str(out_ref),
                "extra": json.dumps({}),
                "flag_low_ssim": 0,
            })

            # b) Autres bandes -> aligner sur REF
            for band in io["bands"].keys():
                if band == reference_band:
                    continue
                bpath = Path(row[band])
                if not bpath.exists():
                    continue  # sécurité (devrait être COMPLETE)

                b01, b_prof, b_mets = read_normalize_resize(
                    bpath,
                    target_hw=tuple(resize.get("target_size", [1024, 1024])),
                    p_low=float(norm.get("p_low", 2)),
                    p_high=float(norm.get("p_high", 98)),
                    eps=float(norm.get("eps", 1e-6)),
                    resize_mode=resize.get("mode", "bilinear"),
                )

                warped, info = align_to_ref(ref01, b01, acfg)
                out_b = out_root / band / f"{pk}{io['band_suffixes'][band]}"
                write_raster(out_b, warped.astype(np.float32), profile_like=b_prof or ref_prof)

                # métriques + flags
                blur8 = _compute_blur8(warped)
                integ_rows.append({
                    "pairkey": pk,
                    "band": band,
                    "saturation_low": b_mets.get("saturation_low", np.nan),
                    "saturation_high": b_mets.get("saturation_high", np.nan),
                    "blur_var": b_mets.get("blur_var", np.nan),
                    "blur_var_u8": blur8,
                    "flag_saturation": int(b_mets.get("saturation_low", 0) > sat_max or b_mets.get("saturation_high", 0) > sat_max),
                    "flag_blur": int(blur8 < blur_min_u8),
                })

                align_rows.append({
                    "pairkey": pk,
                    "band": band,
                    "method": info.get("method"),
                    "ok": int(bool(info.get("ok", False))),
                    "ssim_edges": float(info.get("ssim_edges", 0.0)),
                    "out_path": str(out_b),
                    "extra": json.dumps({k: v for k, v in info.items() if k not in {"method", "ok", "ssim_edges"}}),
                    "flag_low_ssim": int(float(info.get("ssim_edges", 0.0)) < ssim_min),
                })

        except Exception as e:
            logging.exception("Failed pair %s: %s", pk, e)
            continue

    # 6) Écritures CSV
    align_csv = Path(out_cfg.get("alignment_csv", "artifacts/reports/alignment.csv"))
    integ_csv = Path(out_cfg.get("integrity_csv", "artifacts/reports/integrity.csv"))

    pd.DataFrame(align_rows).to_csv(align_csv, index=False)
    pd.DataFrame(integ_rows).to_csv(integ_csv, index=False)

    logging.info("Done. aligned tif in %s/{REG,RED,NIR,GRE}, logs: %s, %s", out_root, align_csv, integ_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
