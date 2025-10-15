# Makefile — AgriEdge (CPU-only)
# Usage (bash/Git Bash): make prepare / make indices / make masks / make train / make infer
# Sous Windows PowerShell, vous pouvez exécuter directement les commandes python affichées ici.

PY := python
ROOT := data/multispectral-images
ALIGNED := artifacts/aligned
INDICES := artifacts/indices
BOUND := artifacts/boundary_maps
MASKS := artifacts/masks_final
REPORTS := artifacts/reports
MODELS := artifacts/models

.PHONY: prepare indices masks train infer clean dirs

dirs:
	@$(PY) - <<PY
import pathlib
for p in ["artifacts/aligned","artifacts/indices","artifacts/boundary_maps","artifacts/masks_raw","artifacts/masks_filtered","artifacts/masks_final","artifacts/reports","artifacts/models","artifacts/preds","logs"]:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)
print("OK: dirs ensured")
PY

prepare: dirs
	$(PY) -m src.cli.prepare --root $(ROOT) --out $(ALIGNED) --cfg configs/prepare.yaml

indices: dirs
	$(PY) -m src.cli.make_indices --aligned $(ALIGNED) --out $(INDICES) --cfg configs/indices.yaml

masks: dirs
	$(PY) -m src.cli.make_masks --indices $(INDICES) --boundaries $(BOUND) --out $(MASKS) --cfg configs/masks.yaml

train: dirs
	$(PY) -m src.cli.train --cfg configs/train_unet_lite.yaml

infer: dirs
	$(PY) -m src.cli.infer --imgs $(ALIGNED)/NIR --cfg configs/train_unet_lite.yaml --weights $(MODELS)/unet_lite_best.h5 --out artifacts/preds

clean:
	@echo Cleaning generated artifacts...
	-@$(PY) - <<PY
import shutil, pathlib
for p in ["artifacts/aligned","artifacts/indices","artifacts/boundary_maps","artifacts/masks_raw","artifacts/masks_filtered","artifacts/masks_final","artifacts/preds"]:
    shutil.rmtree(pathlib.Path(p), ignore_errors=True)
print("OK: artifacts cleaned")
PY
