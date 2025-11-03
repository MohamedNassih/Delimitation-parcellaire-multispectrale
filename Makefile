# Makefile — AgriEdge (CPU-only) with Docker + Web Demo support
# Usage (bash/Git Bash): make prepare / make indices / make masks / make train / make infer
# Docker usage: make build / make shell / make docker-prepare / make docker-indices / make docker-masks / make docker-train / make docker-infer
# Web Demo usage: make web-build / make web-up / make web-down
# Sous Windows PowerShell, vous pouvez exécuter directement les commandes python affichées ici.

PY := python
ROOT := data/multispectral-images
ALIGNED := artifacts/aligned
INDICES := artifacts/indices
BOUND := artifacts/boundary_maps
MASKS := artifacts/masks_final
REPORTS := artifacts/reports
MODELS := artifacts/models

# Docker configuration
DOCKER_IMAGE := agritech-delimitation:latest
DOCKER_COMPOSE := docker-compose

# Web demo configuration
WEBDEMO_COMPOSE := docker-compose -f webdemo/docker-compose.yml

.PHONY: prepare indices masks train infer clean dirs
.PHONY: build shell docker-prepare docker-indices docker-masks docker-train docker-infer docker-clean docker-build
.PHONY: web-build web-up web-down web-shell web-clean web-logs

# Local execution targets
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

# Docker-specific targets
build:
	@echo Building Docker image...
	$(DOCKER_COMPOSE) build

shell:
	@echo Starting interactive shell in Docker container...
	$(DOCKER_COMPOSE) run --rm agritech-pipeline /bin/bash

docker-prepare: build
	@echo Running prepare step in Docker...
	$(DOCKER_COMPOSE) run --rm agritech-pipeline python -m src.cli.prepare --root $(ROOT) --out $(ALIGNED) --cfg configs/prepare.yaml

docker-indices: build
	@echo Running make_indices step in Docker...
	$(DOCKER_COMPOSE) run --rm agritech-pipeline python -m src.cli.make_indices --aligned $(ALIGNED) --out $(INDICES) --cfg configs/indices.yaml

docker-masks: build
	@echo Running make_masks step in Docker...
	$(DOCKER_COMPOSE) run --rm agritech-pipeline python -m src.cli.make_masks --indices $(INDICES) --boundaries $(BOUND) --out $(MASKS) --cfg configs/masks.yaml

docker-train: build
	@echo Running train step in Docker...
	$(DOCKER_COMPOSE) run --rm agritech-pipeline python -m src.cli.train --cfg configs/train_unet_lite.yaml

docker-infer: build
	@echo Running infer step in Docker...
	$(DOCKER_COMPOSE) run --rm agritech-pipeline python -m src.cli.infer --imgs $(ALIGNED)/NIR --cfg configs/train_unet_lite.yaml --weights $(MODELS)/unet_lite_best.h5 --out artifacts/preds

docker-clean:
	@echo Cleaning Docker containers and images...
	-$(DOCKER_COMPOSE) down --volumes --remove-orphans
	-docker image rm $(DOCKER_IMAGE)

# Web demo targets
web-build:
	@echo Building web demo Docker image...
	$(WEBDEMO_COMPOSE) build

web-up:
	@echo Starting web demo server...
	@echo "🌐 Web demo will be available at: http://localhost:7860"
	$(WEBDEMO_COMPOSE) up -d

web-down:
	@echo Stopping web demo server...
	$(WEBDEMO_COMPOSE) down

web-shell:
	@echo Starting interactive shell in web demo container...
	$(WEBDEMO_COMPOSE) run --rm webdemo /bin/bash

web-clean:
	@echo Cleaning web demo Docker resources...
	-$(WEBDEMO_COMPOSE) down --volumes --remove-orphans
	-docker image rm agritech-webdemo:latest

web-logs:
	@echo Showing web demo logs...
	$(WEBDEMO_COMPOSE) logs -f

web-restart: web-down web-up

web-status:
	@echo Web demo status:
	@$(WEBDEMO_COMPOSE) ps || echo "Web demo is not running"

clean:
	@echo Cleaning generated artifacts...
	-@$(PY) - <<PY
import shutil, pathlib
for p in ["artifacts/aligned","artifacts/indices","artifacts/boundary_maps","artifacts/masks_raw","artifacts/masks_filtered","artifacts/masks_final","artifacts/preds"]:
    shutil.rmtree(pathlib.Path(p), ignore_errors=True)
print("OK: artifacts cleaned")
PY
