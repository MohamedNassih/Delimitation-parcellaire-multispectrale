"""Dataset patch-based (512, overlap 64) boundary-aware.
Étape 4 — Doc 05

- Lit bandes alignées (REG/RED/NIR/GRE) + masque de frontière (1=boundary)
- Construit une grille de patches stride = patch - overlap
- Garde prioritairement les patches contenant des pixels positifs (ratio>=min_pos)
- Retourne tenseurs PyTorch (X: (C,H,W) float32, Y: (1,H,W) float32)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.io.readwrite import read_raster, ensure_float01


@dataclass
class DSConfig:
    aligned_root: str = "artifacts/aligned"
    mask_dir: str = "artifacts/masks_final"
    patch_h: int = 512
    patch_w: int = 512
    overlap: int = 64
    min_pos_ratio: float = 0.001  # 0.1% de pixels positifs
    bands: Tuple[str, ...] = ("REG", "RED", "NIR", "GRE")


def _suffixes() -> Dict[str, str]:
    return {"REG": "_REG.tif", "RED": "_RED.tif", "NIR": "_NIR.tif", "GRE": "_GRE.tif"}


def list_pairkeys(aligned_root: Path) -> List[str]:
    keys: List[str] = []
    red_dir = Path(aligned_root) / "RED"
    for p in red_dir.glob("*_RED.tif"):
        s = p.stem
        if s.endswith("_RED"):
            keys.append(s[:-4])
        else:
            keys.append(s)
    return sorted(set(keys))


def _grid(h: int, w: int, ph: int, pw: int, overlap: int) -> List[Tuple[int, int]]:
    sy = max(1, ph - overlap)
    sx = max(1, pw - overlap)
    ys = list(range(0, max(1, h - ph + 1), sy))
    xs = list(range(0, max(1, w - pw + 1), sx))
    # garantir couverture bord
    if ys[-1] != h - ph:
        ys.append(h - ph)
    if xs[-1] != w - pw:
        xs.append(w - pw)
    return [(y, x) for y in ys for x in xs]


def _load_bands_stack(aligned_root: Path, pk: str, bands: Tuple[str, ...]) -> np.ndarray:
    sfx = _suffixes()
    arrs = []
    for b in bands:
        arr, _ = read_raster(aligned_root / b / f"{pk}{sfx[b]}")
        arrs.append(ensure_float01(arr))
    x = np.stack(arrs, axis=0).astype(np.float32)  # (C,H,W)
    return x


def _load_mask(mask_dir: Path, pk: str) -> np.ndarray:
    y, _ = read_raster(mask_dir / f"{pk}_mask.tif")
    y = (ensure_float01(y) > 0.5).astype(np.float32)
    if y.ndim == 2:
        y = y[None, ...]  # (1,H,W)
    return y


class PatchDataset(Dataset):
    def __init__(self, cfg: DSConfig, split_indices: List[str]):
        super().__init__()
        self.cfg = cfg
        self.aligned_root = Path(cfg.aligned_root)
        self.mask_dir = Path(cfg.mask_dir)
        self.bands = cfg.bands

        self.samples: List[Tuple[str, int, int]] = []  # (pairkey, y, x)

        for pk in split_indices:
            y = _load_mask(self.mask_dir, pk)[0]
            h, w = y.shape
            coords = _grid(h, w, cfg.patch_h, cfg.patch_w, cfg.overlap)
            # priorité aux patches avec du positif
            pos_coords = [(yy, xx) for (yy, xx) in coords if y[yy:yy+cfg.patch_h, xx:xx+cfg.patch_w].mean() >= cfg.min_pos_ratio]
            if len(pos_coords) == 0:
                pos_coords = coords[:: max(1, int(len(coords) / 16))]  # downsample quelques patches
            for (yy, xx) in pos_coords:
                self.samples.append((pk, yy, xx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        pk, yy, xx = self.samples[idx]
        x = _load_bands_stack(Path(self.cfg.aligned_root), pk, self.bands)
        y = _load_mask(Path(self.cfg.mask_dir), pk)
        x = x[:, yy:yy+self.cfg.patch_h, xx:xx+self.cfg.patch_w]
        y = y[:, yy:yy+self.cfg.patch_h, xx:xx+self.cfg.patch_w]
        return torch.from_numpy(x), torch.from_numpy(y)


def split_train_val(keys: List[str], val_ratio: float = 0.1, seed: int = 1337) -> Tuple[List[str], List[str]]:
    rng = np.random.default_rng(int(seed))
    keys = sorted(keys)
    n = len(keys)
    n_val = max(1, int(round(val_ratio * n)))
    idx = rng.permutation(n)
    val_idx = set(idx[:n_val].tolist())
    train = [keys[i] for i in range(n) if i not in val_idx]
    val = [keys[i] for i in range(n) if i in val_idx]
    return train, val


__all__ = [
    "DSConfig",
    "PatchDataset",
    "list_pairkeys",
    "split_train_val",
]
