"""UNet-lite (CPU-only, no BatchNorm) — Étape 4 (Doc 05)

- Entrées: C=in_channels (par défaut 4: REG/RED/NIR/GRE)
- Sortie: 1 canal (logit -> Sigmoid pour prob de boundary)
- Blocs: Conv2d -> GroupNorm -> SiLU (x2)
- Down: stride-2 conv (évite MaxPool)
- Up: ConvTranspose2d + concat skip + bloc conv
- Pas de BatchNorm (CPU-only). GroupNorm avec nb_group=min(8, C).

Utilisation:
    from src.models.unet_lite import UNetLite
    model = UNetLite(in_ch=4, base_ch=32)
    y = model(torch.randn(1,4,512,512))  # -> (1,1,512,512)

Remarque: L'activation finale (Sigmoid) est incluse par défaut. Les pertes
(BCE+Dice/Tversky/Surface) et métriques (BF-score, IoU/Dice) sont dans
leurs modules dédiés.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------
# Building blocks
# ------------------------------

def _gn_groups(c: int) -> int:
    # GroupNorm robuste CPU (évite BN). 8 groupes max, divisibles.
    for g in [8, 4, 2, 1]:
        if c % g == 0:
            return g
    return 1


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int | None = None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.gn = nn.GroupNorm(_gn_groups(out_ch), out_ch)
        self.act = nn.SiLU(inplace=True)

        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid = out_ch
        self.block = nn.Sequential(
            ConvGNAct(in_ch, mid, 3, 1),
            ConvGNAct(mid, out_ch, 3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # stride-2 conv (plus efficace CPU que MaxPool+conv)
        self.down = ConvGNAct(in_ch, out_ch, 3, 2)
        self.block = DoubleConv(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        nn.init.kaiming_normal_(self.up.weight, nonlinearity="relu")
        # concat([up, skip]) -> DoubleConv
        self.block = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # pad si nécessaire (robuste à tailles impaires)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


# ------------------------------
# UNet-lite
# ------------------------------

@dataclass
class UNetLiteCfg:
    in_ch: int = 4
    base_ch: int = 32
    out_ch: int = 1
    final_activation: bool = True  # Sigmoid


class UNetLite(nn.Module):
    def __init__(self, in_ch: int = 4, base_ch: int = 32, out_ch: int = 1, final_activation: bool = True):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        self.enc1 = DoubleConv(in_ch, c1)
        self.enc2 = Down(c1, c2)
        self.enc3 = Down(c2, c3)
        self.enc4 = Down(c3, c4)

        self.bottleneck = DoubleConv(c4, c4)

        self.dec3 = Up(c4, c3, c3)
        self.dec2 = Up(c3, c2, c2)
        self.dec1 = Up(c2, c1, c1)

        self.head = nn.Conv2d(c1, out_ch, kernel_size=1)
        nn.init.kaiming_normal_(self.head.weight, nonlinearity="linear")

        self.final_activation = final_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        b = self.bottleneck(s4)
        x = self.dec3(b, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        logits = self.head(x)
        if self.final_activation:
            return torch.sigmoid(logits)
        return logits

    @staticmethod
    def count_params(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_unet_lite(cfg: UNetLiteCfg) -> UNetLite:
    return UNetLite(in_ch=cfg.in_ch, base_ch=cfg.base_ch, out_ch=cfg.out_ch, final_activation=cfg.final_activation)


__all__ = [
    "UNetLite",
    "UNetLiteCfg",
    "build_unet_lite",
]
