"""CLI — Inférence par fenêtres glissantes (patch 512, overlap 64) sur aligned.

Exemple:
    python -m src.cli.infer \
      --imgs artifacts/aligned/NIR \
      --cfg configs/train_unet_lite.yaml \
      --weights artifacts/models/unet_lite_best.h5 \
      --out artifacts/preds

Notes:
- Accepte --imgs pointant soit vers artifacts/aligned, soit vers un sous-dossier (ex: .../NIR).
- Construit les entrées 4 canaux REG/RED/NIR/GRE pour chaque pairkey.
- Sauve proba (*.tif) et binaire (*.png) seuillé à 0.5.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch

from src.models.unet_lite import UNetLite, UNetLiteCfg
from src.train.dataset import list_pairkeys
from src.io.readwrite import read_raster, ensure_float01, write_raster, save_debug_png


def _suffixes():
    return {"REG": "_REG.tif", "RED": "_RED.tif", "NIR": "_NIR.tif", "GRE": "_GRE.tif"}


def _grid(h: int, w: int, ph: int, pw: int, overlap: int):
    sy = max(1, ph - overlap)
    sx = max(1, pw - overlap)
    ys = list(range(0, max(1, h - ph + 1), sy))
    xs = list(range(0, max(1, w - pw + 1), sx))
    if ys[-1] != h - ph:
        ys.append(h - ph)
    if xs[-1] != w - pw:
        xs.append(w - pw)
    return [(y, x) for y in ys for x in xs]


def parse_args():
    ap = argparse.ArgumentParser(description="Inference on aligned multispectral images")
    ap.add_argument("--imgs", type=Path, required=True, help="artifacts/aligned or a band subfolder like .../NIR")
    ap.add_argument("--cfg", type=Path, default=Path("configs/train_unet_lite.yaml"))
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--patch", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    return ap.parse_args()


def _aligned_root_from_imgs(imgs: Path) -> Path:
    p = imgs
    # si .../aligned/BAND -> remonter
    if p.name in {"REG", "RED", "NIR", "GRE"}:
        return p.parent
    return p


def _load_stack(aligned_root: Path, pk: str) -> np.ndarray:
    sfx = _suffixes()
    arrs = []
    for b in ["REG", "RED", "NIR", "GRE"]:
        a, _ = read_raster(aligned_root / b / f"{pk}{sfx[b]}")
        arrs.append(ensure_float01(a))
    x = np.stack(arrs, axis=0).astype(np.float32)
    return x


def _infer_one(model: UNetLite, x4: np.ndarray, patch: int, overlap: int) -> np.ndarray:
    """Fenêtres glissantes avec moyenne sur les recouvrements."""
    model.eval()
    C, H, W = x4.shape
    out = np.zeros((H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)
    device = torch.device("cpu")

    for (yy, xx) in _grid(H, W, patch, patch, overlap):
        tile = torch.from_numpy(x4[:, yy:yy+patch, xx:xx+patch]).unsqueeze(0)
        with torch.no_grad():
            p = model(tile.to(device)).cpu().numpy()[0, 0]
        out[yy:yy+patch, xx:xx+patch] += p
        cnt[yy:yy+patch, xx:xx+patch] += 1.0

    out = out / np.maximum(cnt, 1e-6)
    return out


def main() -> int:
    args = parse_args()
    aligned_root = _aligned_root_from_imgs(args.imgs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # charger modèle
    ckpt = torch.load(args.weights, map_location="cpu")
    model = UNetLite(in_ch=4, base_ch=32, out_ch=1, final_activation=True)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)

    keys = list_pairkeys(aligned_root)

    for pk in keys:
        x4 = _load_stack(aligned_root, pk)
        prob = _infer_one(model, x4, patch=args.patch, overlap=args.overlap)
        # sauvegarde
        write_raster(out_dir / f"{pk}_prob.tif", prob)
        save_debug_png(out_dir / f"{pk}_prob.png", prob)
        mask = (prob >= 0.5).astype(np.float32)
        save_debug_png(out_dir / f"{pk}_bin.png", mask)

    print(f"Done. preds -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
