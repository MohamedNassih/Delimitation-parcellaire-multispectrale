"""CLI — Étape 3: Masques (boundary=1, field=0)

Pipeline:
- lit S_fused (artifacts/boundary_maps) + NDVI/GNDVI (artifacts/indices)
- génère seeds NV/VH (seed_masks)
- watershed sur relief 1-S_fused pour lignes de partage (mask boundary brut)
- filtre par blocs 50×50 (block_filter)
- (optionnel) keep-largest-black
- post-process (skeleton + buffer r=1, closing optionnel)
- fallback si white_ratio quasi nul: seuillage percentile sur S_fused + skeleton+buffer
- écrit masks_raw/, masks_filtered/, masks_final/ (+ PNG & stats CSV)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
import cv2  # type: ignore
from skimage.morphology import skeletonize  # type: ignore

from src.io.readwrite import read_raster, write_raster, save_debug_png, ensure_float01
from src.masking.seed_masks import generate_seeds_for_pair
from src.masking.watershed import watershed_for_pair, WSConfig
from src.masking.block_filter import block_filter, BlockFilterCfg
from src.masking.keep_largest_black import keep_largest_black
from src.masking.postprocess import skeleton_and_buffer, PostCfg


# ------------------------------
# Helpers
# ------------------------------

def _iter_pairkeys(boundary_dir: Path, fused_suffix: str) -> List[str]:
    keys: List[str] = []
    for p in boundary_dir.glob(f"*{fused_suffix}"):
        stem = p.stem
        if stem.endswith(fused_suffix.replace('.tif','')):
            pk = stem[: -len(fused_suffix.replace('.tif',''))]
        else:
            pk = stem.replace(fused_suffix.replace('.tif',''), "")
        keys.append(pk)
    return sorted(set(keys))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Make boundary masks from fused maps and indices")
    ap.add_argument("--cfg", type=Path, default=Path("configs/masks.yaml"))
    ap.add_argument("--indices", type=Path, required=True)
    ap.add_argument("--boundaries", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log-level", default="INFO")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="[%(levelname)s] %(message)s")

    cfg = yaml.safe_load(open(args.cfg, "r", encoding="utf-8"))
    io = cfg["io"]
    fused_suffix = str(io.get("fused_suffix", "_S_fused.tif"))

    indices_dir = Path(str(args.indices))
    boundary_dir = Path(str(args.boundaries))
    masks_raw = Path(io.get("masks_raw_dir", "artifacts/masks_raw"))
    masks_filtered = Path(io.get("masks_filtered_dir", "artifacts/masks_filtered"))
    masks_final = Path(io.get("masks_final_dir", "artifacts/masks_final"))
    reports_dir = Path(io.get("reports_dir", "artifacts/reports"))
    for d in [masks_raw, masks_filtered, masks_final, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    ws_cfg = WSConfig(
        clean_seeds_dilate=int(cfg.get("watershed", {}).get("clean_seeds_dilate", 1)),
        enforce_border_black=bool(cfg.get("watershed", {}).get("enforce_border_black", True)),
        min_basin_area_px=int(cfg.get("watershed", {}).get("min_basin_area_px", 256)),
        connectivity=int(cfg.get("watershed", {}).get("connectivity", 4)),
        min_seed_cov=float(cfg.get("watershed", {}).get("min_seed_cov", 0.005)),
        minima_percentile=float(cfg.get("watershed", {}).get("minima_percentile", 35.0)),
        use_full_mask=bool(cfg.get("watershed", {}).get("use_full_mask", True)),
    )
    bf_cfg = BlockFilterCfg(
        block_size=int(cfg.get("block_filter", {}).get("block_size", 50)),
        white_ratio_max=float(cfg.get("block_filter", {}).get("white_ratio_max", 0.08)),
        soften_margin_px=int(cfg.get("block_filter", {}).get("soften_margin_px", 2)),
    )
    post_cfg = PostCfg(
        skeleton_enabled=bool(cfg.get("postprocess", {}).get("skeleton", {}).get("enabled", True)),
        buffer_radius=int(cfg.get("postprocess", {}).get("skeleton", {}).get("buffer_radius", 1)),
        closing_radius=int(cfg.get("postprocess", {}).get("closing_radius", 0)),
        crf_enabled=bool(cfg.get("postprocess", {}).get("crf", {}).get("enabled", False)),
    )

    expected_min = float(cfg.get("quality", {}).get("expected_white_ratio_min", 0.01))
    expected_max = float(cfg.get("quality", {}).get("expected_white_ratio_max", 0.08))
    clamp = bool(cfg.get("quality", {}).get("clamp_to_expected_range", True))
    min_white_hard = float(cfg.get("quality", {}).get("min_white_ratio_hard", 0.003))
    target_white = float(cfg.get("quality", {}).get("target_white_ratio", 0.03))

    # pairkeys par S_fused
    pairkeys = _iter_pairkeys(boundary_dir, fused_suffix)
    logging.info("Found %d fused maps", len(pairkeys))

    rows: List[Dict] = []

    for pk in pairkeys:
        try:
            # 1) Seeds NV/VH
            nv, vh = generate_seeds_for_pair(indices_dir, boundary_dir, pk, cfg.get("seeds", {}))
            if cfg.get("runtime", {}).get("save_debug", True):
                save_debug_png(masks_raw / f"{pk}_NV.png", nv.astype(np.float32))
                save_debug_png(masks_raw / f"{pk}_VH.png", vh.astype(np.float32))

            # 2) Watershed -> boundary brut
            bnd_raw, labels = watershed_for_pair(boundary_dir, pk, nv, vh, ws_cfg)
            write_raster(masks_raw / f"{pk}_boundary_raw.tif", bnd_raw.astype(np.float32))

            # 3) Filtre par blocs
            bnd_blk = block_filter(bnd_raw, bf_cfg)
            write_raster(masks_filtered / f"{pk}_boundary_block.tif", bnd_blk.astype(np.float32))

            # 4) Keep largest black (optionnel)
            klb_enabled = bool(cfg.get("postprocess", {}).get("keep_largest_black", {}).get("enabled", False))
            if klb_enabled:
                bnd_k = keep_largest_black(bnd_blk)
            else:
                bnd_k = bnd_blk

            # 5) Post-process (skeleton + buffer)
            bnd_final = skeleton_and_buffer(bnd_k, post_cfg)
            write_raster(masks_final / f"{pk}_mask.tif", bnd_final.astype(np.float32))
            if cfg.get("outputs", {}).get("save_png_quicklook", True):
                save_debug_png(masks_final / f"{pk}_mask.png", bnd_final.astype(np.float32))

            white_ratio = float((bnd_final > 0).mean())
            fb_used = False
            fb_q = -1.0

            # 6) Fallback si quasi nul -> densité cible depuis S_fused
            if white_ratio < min_white_hard:
                s_fused, _ = read_raster(boundary_dir / f"{pk}_S_fused.tif")
                s = ensure_float01(s_fused)
                qs = [99.9, 99.5, 99.0, 98.5, 98.0, 97.0, 96.0, 95.0, 93.0, 92.0, 90.0, 88.0, 85.0, 80.0]
                best = None
                for q in qs:
                    t = float(np.percentile(s, q))
                    cand = (s >= t).astype(np.uint8)
                    sk = skeletonize(cand.astype(bool)).astype(np.uint8)
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
                    cand = cv2.dilate(sk, k)
                    wr = float(cand.mean())
                    if (best is None) or (abs(wr - target_white) < abs(best[1] - target_white)):
                        best = (cand, wr, q)
                    if wr >= target_white * 0.9:
                        break
                if best is not None:
                    bnd_final = best[0].astype(np.uint8)
                    white_ratio = float(best[1])
                    fb_q = float(best[2])
                    fb_used = True
                    write_raster(masks_final / f"{pk}_mask.tif", bnd_final.astype(np.float32))
                    if cfg.get("outputs", {}).get("save_png_quicklook", True):
                        save_debug_png(masks_final / f"{pk}_mask.png", bnd_final.astype(np.float32))

            # 7) Clamp doux dans la plage visée
            if clamp and (white_ratio < expected_min or white_ratio > expected_max):
                if white_ratio > expected_max:
                    tmp = skeletonize((bnd_final > 0).astype(bool)).astype(np.uint8)
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
                    bnd_final = cv2.dilate(tmp, k)
                else:
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
                    bnd_final = cv2.dilate((bnd_final>0).astype(np.uint8), k)
                write_raster(masks_final / f"{pk}_mask.tif", bnd_final.astype(np.float32))
                if cfg.get("outputs", {}).get("save_png_quicklook", True):
                    save_debug_png(masks_final / f"{pk}_mask.png", bnd_final.astype(np.float32))
                white_ratio = float((bnd_final > 0).mean())

            rows.append({
                "pairkey": pk,
                "white_ratio": white_ratio,
                "fallback": int(fb_used),
                "fallback_percentile": fb_q,
            })
        except Exception as e:
            logging.exception("Failed %s: %s", pk, e)
            continue

    if rows:
        pd.DataFrame(rows).to_csv(Path(io.get("reports_dir", "artifacts/reports")) / "masks_stats.csv", index=False)

    logging.info("Done. masks -> %s", masks_final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
