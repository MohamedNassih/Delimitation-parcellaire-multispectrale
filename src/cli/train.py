"""CLI — Entraînement U-Net lite (Étape 4)

Usage:
    python -m src.cli.train --cfg configs/train_unet_lite.yaml
"""
from __future__ import annotations

import argparse
import yaml
from pathlib import Path

from src.train.trainer import TrainCfg, train_loop
from src.train.dataset import DSConfig
from src.models.unet_lite import UNetLiteCfg


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train UNet-lite on boundary masks")
    ap.add_argument("--cfg", type=Path, default=Path("configs/train_unet_lite.yaml"))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg_yaml = yaml.safe_load(open(args.cfg, "r", encoding="utf-8"))

    ds = cfg_yaml.get("dataset", {})
    tr = cfg_yaml.get("training", {})
    model_yaml = cfg_yaml.get("model", {})
    metrics_yaml = cfg_yaml.get("metrics", {})
    loss_yaml = cfg_yaml.get("loss", {})

    cfg = TrainCfg(
        ds=DSConfig(
            aligned_root=str(ds.get("aligned_root", "artifacts/aligned")),
            mask_dir=str(ds.get("mask_dir", "artifacts/masks_final")),
            patch_h=int(ds.get("patch_size", [512, 512])[0]),
            patch_w=int(ds.get("patch_size", [512, 512])[1]),
            overlap=int(ds.get("overlap", 64)),
            min_pos_ratio=float(ds.get("min_pos_ratio", 0.001)),
        ),
        epochs=int(tr.get("epochs", 50)),
        batch_size=int(tr.get("batch_size", 1)),
        lr=float(tr.get("lr", 1e-3)),
        num_workers=int(tr.get("num_workers", 0)),
        model=UNetLiteCfg(
            in_ch=int(model_yaml.get("in_channels", 4)),
            base_ch=int(model_yaml.get("base_channels", 32)),
            out_ch=1,
            final_activation=True,
        ),
        primary_loss=str(loss_yaml.get("primary", "bce_dice")),
        tversky_alpha=float(loss_yaml.get("tversky", {}).get("alpha", 0.5)),
        tversky_beta=float(loss_yaml.get("tversky", {}).get("beta", 0.5)),
        use_surface_loss=bool(loss_yaml.get("surface", {}).get("enabled", False)),
        surface_weight=float(loss_yaml.get("surface", {}).get("weight", 0.2)),
        bf_radius=int(metrics_yaml.get("bfscore", {}).get("radius", 2)),
        patience=int(tr.get("early_stopping", {}).get("patience", 8)),
        ckpt_outfile=str(tr.get("checkpoint", {}).get("outfile", "artifacts/models/unet_lite_best.h5")),
        preds_dir=str(tr.get("preds_dir", "artifacts/preds")),
    )

    res = train_loop(cfg)
    print("Training done:", res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
