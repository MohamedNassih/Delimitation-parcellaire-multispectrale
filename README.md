# Délimitation parcellaire multispectrale

> Pipeline complet pour détecter les **frontières de parcelles** à partir d’images multispectrales (REG/RED/NIR/GRE). Masques pseudo‑labels → entraînement U‑Net léger → inférence & évaluation.

---

## 🔧 Environnement

* **Python** : 3.12.8
* **PyTorch** : `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
* Dépendances principales : `numpy==1.26.4`, `scipy==1.11.4`, `opencv-python==4.9.0.80`, `scikit-image==0.22.0`, `rasterio==1.3.9`, `pandas==2.2.2`, `tifffile==2024.2.12`, `pyyaml==6.0.1`, `tqdm==4.66.1`, `joblib==1.3.2`

> **Masques** : **1 = boundary**, **0 = field** (cf. `configs/project.yaml`).

---

## 🗂️ Données & arborescence

```
project/
├─ configs/
│  ├─ project.yaml         # conventions & logs
│  ├─ prepare.yaml         # inventaire + alignement
│  ├─ indices.yaml         # indices spectraux & fusions
│  ├─ masks.yaml           # seeds NV/VH, watershed, filtres
│  └─ train_unet_lite.yaml # dataset/entraîneur/modèle
├─ data/multispectral-images/
│  ├─ REG/*.tif  ├─ RED/*.tif  ├─ NIR/*.tif  └─ GRE/*.tif
├─ artifacts/
│  ├─ aligned/{REG,RED,NIR,GRE}/*.tif
│  ├─ indices/* (NDVI/GNDVI/NDRE/…)
│  ├─ boundary_maps/*_S_fused.tif
│  ├─ masks_{raw,filtered,final}/*
│  ├─ models/unet_lite_best.h5
│  ├─ preds/*_prob.tif|png, *_bin.png
│  └─ reports/*.csv (inventory, alignment, integrity, indices_stats, masks_stats, preds_metrics, threshold_sweep)
└─ src/
```

---

## 🚀 Exécution rapide (VS Code / PowerShell)

### Étape 0 — Bootstrap

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
```

### Étape 1 — Inventaire & Alignement

```powershell
python -m src.cli.prepare --root data/multispectral-images --out artifacts/aligned --cfg configs/prepare.yaml --log-level INFO
```

**Attendus** : `artifacts/aligned/*/*.tif` (1024×1024, float32), `artifacts/reports/inventory.csv`, `alignment.csv`.

### Étape 2 — Indices & Arêtes

```powershell
python -m src.cli.make_indices --aligned artifacts/aligned --out artifacts/indices --cfg configs/indices.yaml --log-level INFO
```

**Attendus** : `artifacts/indices/*`, `artifacts/boundary_maps/*_S_fused.tif`.

### Étape 3 — Masques (pseudo‑labels)

```powershell
python -m src.cli.make_masks --indices artifacts/indices --boundaries artifacts/boundary_maps --out artifacts/masks_final --cfg configs/masks.yaml --log-level INFO
```

* Seeds **NV/VH** + **watershed** sur relief `1−S_fused`.
* **Fallback densité** activé si masques trop vides : seuillage percentile sur `S_fused` → `skeletonize+buffer` vers densité cible.

### Étape 4 — Entraînement & Inférence

```powershell
python -m src.cli.train --cfg configs/train_unet_lite.yaml
python -m src.cli.infer --imgs artifacts/aligned/NIR --cfg configs/train_unet_lite.yaml --weights artifacts/models/unet_lite_best.h5 --out artifacts/preds --patch 512 --overlap 64
```

---

## 🧠 Modèle

* **UNet‑lite** : Conv→GroupNorm→SiLU, Down = stride‑2 Conv, Up = ConvTranspose2d + concat. Pas de BatchNorm.
* Config par défaut : `in_channels=4` (REG/RED/NIR/GRE), `base_channels=32`, sortie `1` (Sigmoid).
* **Dataset** : patches `512×512`, overlap `64`, **boundary‑aware sampler** via `dataset.min_pos_ratio`.

---

## 📊 Résultats (chiffrés)

### 1) Couverture des masques (après fallback densité)

Source : `artifacts/reports/masks_stats.csv` sur **355 scènes**.

| Stat                   |   white_ratio |
| ---------------------- | ------------: |
| Min                    |    **0.0248** |
| Moyenne                |    **0.0298** |
| Max                    |    **0.0346** |
| Scènes dans `[1%, 8%]` |      **100%** |
| Fallback utilisé       | **355 / 355** |

> Densité cible ≈ 3 % atteinte, idéale pour frontières minces.

### 2) Entraînement (val)

Journal : `src/cli/train.py` (BF = Boundary‑F1 tolérance r=2).

|    Epoch |       loss |     val_bf |         P |         R |
| -------: | ---------: | ---------: | --------: | --------: |
|     E001 |     0.3279 |     0.7641 |     0.809 |     0.798 |
|     E002 |     0.2879 |     0.7516 |     0.709 |     0.904 |
|     E003 |     0.2809 |     0.7870 |     0.761 |     0.899 |
| **E004** | **0.2751** | **0.7978** | **0.795** | **0.878** |
|     E005 |     0.2698 |     0.7930 |     0.759 |     0.916 |

**Meilleur modèle** : `best_bf = 0.7978` (epoch 4) → `artifacts/models/unet_lite_best.h5`.

### 3) Évaluation des prédictions (test = toutes les scènes)

Source : `artifacts/reports/preds_metrics.csv` (seuil 0.5, r=2).

* **BF moyen** : **0.8238**
* **IoU moyen** : **0.4186**
* **Dice moyen** : **0.5844**

> Exemple top‑3 BF (illustratif) : 0.969 / 0.952 / 0.944.

### 4) Balayage de seuil (stabilité)

Source : `artifacts/reports/threshold_sweep.csv` sur 100 scènes, r=2.

| Threshold |       0.35 |       0.40 |       0.45 |       0.50 |       0.55 |       0.60 |
| --------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
|  BF moyen | **0.8394** | **0.8394** | **0.8394** | **0.8394** | **0.8394** | **0.8394** |

> **Seuil recommandé** : 0.5 (par défaut) — résultat stable de 0.35 à 0.60.

---

## 🧪 Visualisation

* Overlays proba/binaire : `artifacts/preds/*_prob.png`, `*_bin.png`
* Overlays **diagnostic** (TP vert / FP rouge / FN cyan) : `artifacts/preds_overlays_dbg/*`
* Overlays **frontière rouge sur GRE** : `artifacts/preds_overlays/*`

---

## ⚙️ Réglages utiles

* **masks.yaml**

  * `watershed.min_seed_cov = 0.005`, `use_full_mask = true`, `minima_percentile = 35` (fallback seeds)
  * `postprocess.skeleton.enabled = false` (peut être remis à `true` si bords trop épais)
  * `block_filter.white_ratio_max = 0.12` (tolérance densité)
  * `quality.target_white_ratio = 0.03` (densité visée par fallback)
* **train_unet_lite.yaml**

  * `dataset.min_pos_ratio = 0.0002` (garde des patches avec peu de positif)
  * `model.base_channels = 32` (16 pour runs rapides)
  * `metrics.bfscore.radius = 2` (tolérance BF)

---

## 🧰 Conseils/Troubleshooting

* **NotGeoreferencedWarning (rasterio)** : normal (tuiles 1024×1024 sans géotransform), sans impact.
* ECC non‑convergent : le pipeline gère le fallback ORB+RANSAC.
* Masques trop fins/épais : jouer `postprocess.skeleton.enabled`, buffer, ou le seuil binaire en inférence (0.35–0.60 stable).

---

## 📜 Licence

MIT (par défaut).

---

## ✍️ Réplication en 5 commandes

```powershell
pip install -r requirements.txt
python -m src.cli.prepare --root data/multispectral-images --out artifacts/aligned --cfg configs/prepare.yaml
python -m src.cli.make_indices --aligned artifacts/aligned --out artifacts/indices --cfg configs/indices.yaml
python -m src.cli.make_masks --indices artifacts/indices --boundaries artifacts/boundary_maps --out artifacts/masks_final --cfg configs/masks.yaml
python -m src.cli.train --cfg configs/train_unet_lite.yaml && ^
python -m src.cli.infer --imgs artifacts/aligned/NIR --cfg configs/train_unet_lite.yaml --weights artifacts/models/unet_lite_best.h5 --out artifacts/preds
```
