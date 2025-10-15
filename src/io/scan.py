"""Scan bands and build inventory of REG/RED/NIR/GRE pairs.
Étape 1 — Inventaire (Doc 02)

- Parcourt data/multispectral-images/{REG,RED,NIR,GRE}
- Extrait le PAIRKEY via regex commun (ex: ^(IMG_..._####_))
- Construit inventory.csv avec statut COMPLET/INCOMPLETE et raisons de quarantaine
- NE DÉPLACE PAS les fichiers : seulement un manifest CSV + logs

Convention masque globale du projet : 1=boundary, 0=field (rappel)
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml


@dataclass
class ScanConfig:
    root: Path
    bands: Dict[str, str]
    pair_regex: str
    suffixes: Dict[str, str]
    out_csv: Path
    quarantine_dir: Path


def load_prepare_cfg(cfg_path: Path) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def iter_band_files(band_root: Path, suffix: str) -> Iterable[Path]:
    if not band_root.exists():
        logging.warning("Band folder missing: %s", band_root)
        return []
    return band_root.glob(f"*{suffix}")


def build_pairkey(path: Path, pair_re: re.Pattern[str]) -> Optional[str]:
    m = pair_re.match(path.stem)
    if not m:
        return None
    return m.group(1)


def scan_pairs(cfg: ScanConfig) -> pd.DataFrame:
    pair_re = re.compile(cfg.pair_regex)

    # Collect files per band -> map[pairkey] = filepath
    per_band: Dict[str, Dict[str, Path]] = {b: {} for b in cfg.bands.keys()}
    for band, subdir in cfg.bands.items():
        band_dir = cfg.root / subdir
        suffix = cfg.suffixes.get(band, "")
        for fp in iter_band_files(band_dir, suffix):
            pk = build_pairkey(fp, pair_re)
            if not pk:
                logging.debug("Skip (no pairkey): %s", fp)
                continue
            # Keep the first, warn on duplicates
            if pk in per_band[band]:
                logging.warning("Duplicate pairkey for band %s: %s (keeping first %s)", band, fp, per_band[band][pk])
                continue
            per_band[band][pk] = fp

    # Union of all pairkeys observed
    all_keys = set().union(*[set(d.keys()) for d in per_band.values()])

    rows: List[Dict[str, object]] = []
    for pk in sorted(all_keys):
        row: Dict[str, object] = {"pairkey": pk}
        complete = True
        reasons: List[str] = []
        for band in cfg.bands.keys():
            fp = per_band[band].get(pk)
            row[band] = str(fp) if fp else ""
            if fp is None:
                complete = False
                reasons.append(f"missing_{band}")
        row["status"] = "COMPLETE" if complete else "INCOMPLETE"
        row["quarantine"] = ("yes" if not complete else "no")
        row["quarantine_reason"] = ",".join(reasons)
        rows.append(row)

    df = pd.DataFrame(rows, columns=["pairkey", *cfg.bands.keys(), "status", "quarantine", "quarantine_reason"])  # type: ignore[arg-type]
    return df


def write_inventory(df: pd.DataFrame, out_csv: Path) -> None:
    ensure_parent(out_csv)
    df.to_csv(out_csv, index=False)


def summarize(df: pd.DataFrame) -> str:
    n = len(df)
    n_ok = int((df["status"] == "COMPLETE").sum())
    n_bad = n - n_ok
    return f"pairs_total={n} complete={n_ok} incomplete={n_bad}"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Scan multispectral bands and build inventory.csv")
    ap.add_argument("--root", type=Path, required=False, help="Root folder containing band subfolders (REG/RED/NIR/GRE)")
    ap.add_argument("--cfg", type=Path, default=Path("configs/prepare.yaml"), help="Prepare YAML path")
    ap.add_argument("--out", type=Path, default=None, help="Output CSV (defaults to cfg.output.inventory_csv)")
    ap.add_argument("--quarantine-dir", type=Path, default=None, help="Quarantine dir (defaults to cfg.output.quarantine_dir)")
    ap.add_argument("--log-level", default="INFO", help="Logging level")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="[%(levelname)s] %(message)s")

    prep = load_prepare_cfg(args.cfg)
    io = prep["io"]
    out_cfg = prep["output"]

    root = args.root or Path(io["root"])  # type: ignore[assignment]
    bands = io["bands"]
    pair_regex = io["pair_regex"]
    suffixes = io["band_suffixes"]

    out_csv = args.out or Path(out_cfg["inventory_csv"])  # type: ignore[assignment]
    quarantine_dir = args.quarantine_dir or Path(out_cfg.get("quarantine_dir", "artifacts/quarantine"))  # type: ignore[assignment]
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    scfg = ScanConfig(root=Path(root), bands=bands, pair_regex=pair_regex, suffixes=suffixes, out_csv=Path(out_csv), quarantine_dir=Path(quarantine_dir))

    logging.info("Scanning root=%s", scfg.root)
    for b, sub in scfg.bands.items():
        logging.info(" - band %-3s in %s (suffix %s)", b, scfg.root / sub, scfg.suffixes.get(b, ""))

    df = scan_pairs(scfg)
    write_inventory(df, scfg.out_csv)

    # Écrit aussi un manifest des paires incomplètes dans la quarantine
    q = df[df["quarantine"] == "yes"].copy()
    if not q.empty:
        q_path = scfg.quarantine_dir / "inventory_incomplete.csv"
        q.to_csv(q_path, index=False)
        logging.warning("Incomplete pairs: %d (manifest: %s)", len(q), q_path)

    logging.info("Done. %s | CSV: %s", summarize(df), scfg.out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
