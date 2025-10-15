"""Boucle d'entraînement (CPU-only), early-stopping sur BF-score (val), best-on-BF.
Étape 4 — Doc 05
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.unet_lite import UNetLite, UNetLiteCfg
from src.models.losses import bce_dice, tversky_loss, surface_loss, SurfaceCfg
from src.metrics.bfscore import BFConfig, bfscore
from src.train.dataset import DSConfig, list_pairkeys, split_train_val, PatchDataset
from src.io.readwrite import write_raster, save_debug_png


@dataclass
class TrainCfg:
    # dataset
    ds: DSConfig = field(default_factory=DSConfig)
    # training
    epochs: int = 50
    batch_size: int = 1
    lr: float = 1e-3
    num_workers: int = 0
    # model
    model: UNetLiteCfg = field(default_factory=UNetLiteCfg)
    # losses/metrics
    primary_loss: str = "bce_dice"  # {bce_dice, tversky}
    tversky_alpha: float = 0.5
    tversky_beta: float = 0.5
    use_surface_loss: bool = False
    surface_weight: float = 0.2
    bf_radius: int = 2
    # early stopping
    patience: int = 8
    # checkpoint & outputs
    ckpt_outfile: str = "artifacts/models/unet_lite_best.h5"  # extension libre
    preds_dir: str = "artifacts/preds"


def _make_loss(cfg: TrainCfg):
    if cfg.primary_loss == "tversky":
        return lambda p, y: tversky_loss(p, y, alpha=cfg.tversky_alpha, beta=cfg.tversky_beta)
    return lambda p, y: bce_dice(p, y, alpha=0.5)


def _val_metrics(model: nn.Module, loader: DataLoader, bf_radius: int, device: torch.device) -> Dict[str, float]:
    model.eval()
    m_sum = {"bf": 0.0, "precision": 0.0, "recall": 0.0, "iou": 0.0, "dice": 0.0}
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = model(x)
            p_np = p.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            for i in range(p_np.shape[0]):
                stats = bfscore(y_np[i, 0], p_np[i, 0], BFConfig(radius=bf_radius, threshold=0.5))
                for k in m_sum.keys():
                    m_sum[k] += float(stats[k])
                n += 1
    if n == 0:
        return {k: 0.0 for k in m_sum.keys()}
    return {k: v / n for k, v in m_sum.items()}


def train_loop(cfg: TrainCfg) -> Dict[str, float]:
    device = torch.device("cpu")

    # --- data ---
    keys = list_pairkeys(Path(cfg.ds.aligned_root))
    tr_keys, va_keys = split_train_val(keys, val_ratio=0.1, seed=1337)

    ds_tr = PatchDataset(cfg.ds, tr_keys)
    ds_va = PatchDataset(cfg.ds, va_keys)

    loader_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    loader_va = DataLoader(ds_va, batch_size=1, shuffle=False, num_workers=0)

    # --- model ---
    model = UNetLite(in_ch=cfg.model.in_ch, base_ch=cfg.model.base_ch, out_ch=cfg.model.out_ch, final_activation=True)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    primary_loss = _make_loss(cfg)

    best_bf = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batch = 0
        t0 = time.time()
        for x, y in loader_tr:
            x = x.to(device)
            y = y.to(device)
            p = model(x)
            loss = primary_loss(p, y)
            if cfg.use_surface_loss:
                loss = loss + cfg.surface_weight * surface_loss(p, y, SurfaceCfg(normalize=True))
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach().cpu().item())
            n_batch += 1
        epoch_loss /= max(1, n_batch)

        # validation
        metrics = _val_metrics(model, loader_va, bf_radius=cfg.bf_radius, device=device)

        print(f"[E{epoch:03d}] loss={epoch_loss:.4f} val_bf={metrics['bf']:.4f} (P={metrics['precision']:.3f} R={metrics['recall']:.3f}) time={time.time()-t0:.1f}s")

        # early stopping / save best
        if metrics["bf"] > best_bf:
            best_bf = metrics["bf"]
            best_state = {"model": model.state_dict(), "epoch": epoch, "bf": best_bf}
            Path(cfg.ckpt_outfile).parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, cfg.ckpt_outfile)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f"Early stopping at epoch {epoch} (best_bf={best_bf:.4f})")
                break

    # sauvegarder dernière métrique
    return {"best_bf": float(best_bf), "best_epoch": int(best_state.get("epoch", 0) if best_state else 0)}


__all__ = ["TrainCfg", "train_loop"]
