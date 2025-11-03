# Docker Deployment Guide - Agricultural Field Boundary Detection

This guide provides instructions for running the agricultural field boundary detection pipeline using Docker on Windows with Docker Desktop.

## Prerequisites

1. **Docker Desktop** installed on Windows
2. **Make** (optional, for convenience commands)
3. Git Bash or WSL2 for running make commands (recommended)

## Quick Start

### Option 1: Using Docker Compose directly

```powershell
# Build the Docker image
docker-compose build

# Run interactive shell in container
docker-compose run --rm agritech-pipeline /bin/bash
```

### Option 2: Using Make (recommended)

If you have Make installed (via Git Bash or WSL2):

```bash
# Build the Docker image
make build

# Start interactive shell
make shell

# Run pipeline steps inside container
make docker-prepare
make docker-indices
make docker-masks
make docker-train
make docker-infer
```

## Pipeline Steps (via Docker)

The Docker container replicates the 4-step pipeline from the README:

### Step 1: Prepare (Inventory & Alignment)
```bash
docker-compose run --rm agritech-pipeline python -m src.cli.prepare --root data/multispectral-images --out artifacts/aligned --cfg configs/prepare.yaml --log-level INFO
```

### Step 2: Make Indices
```bash
docker-compose run --rm agritech-pipeline python -m src.cli.make_indices --aligned artifacts/aligned --out artifacts/indices --cfg configs/indices.yaml --log-level INFO
```

### Step 3: Make Masks
```bash
docker-compose run --rm agritech-pipeline python -m src.cli.make_masks --indices artifacts/indices --boundaries artifacts/boundary_maps --out artifacts/masks_final --cfg configs/masks.yaml --log-level INFO
```

### Step 4: Train & Infer
```bash
# Train the model
docker-compose run --rm agritech-pipeline python -m src.cli.train --cfg configs/train_unet_lite.yaml

# Run inference
docker-compose run --rm agritech-pipeline python -m src.cli.infer --imgs artifacts/aligned/NIR --cfg configs/train_unet_lite.yaml --weights artifacts/models/unet_lite_best.h5 --out artifacts/preds --patch 512 --overlap 64
```

## Volume Mounts

The following directories are mounted as volumes:
- `./data` → `/app/data` (input multispectral images)
- `./artifacts` → `/app/artifacts` (output results)
- `./logs` → `/app/logs` (training logs)

## Windows-Specific Notes

### File Paths
- Use forward slashes (`/`) in docker-compose.yml for Windows compatibility
- Docker Desktop handles path translation automatically

### Performance
- Ensure Docker Desktop has sufficient resources (minimum 4GB RAM)
- Consider increasing CPU and memory limits in Docker Desktop settings

### PowerShell vs Git Bash
- PowerShell users: Run docker-compose commands directly
- Git Bash users: Use `make` commands for convenience
- CMD users: Use docker-compose commands directly

## Troubleshooting

### Container doesn't start
```powershell
docker-compose logs agritech-pipeline
```

### Clean up Docker resources
```bash
make docker-clean
# OR manually:
docker-compose down --volumes --remove-orphans
docker image rm agritech-delimitation:latest
```

### Permission issues (Windows)
If you encounter permission issues with volume mounts, ensure:
1. Docker Desktop has access to your project directory
2. Your user account has proper permissions on the project folder

## Development Workflow

1. **Build once**: `make build` (or `docker-compose build`)
2. **Iterate quickly**: Use `make shell` to enter container
3. **Run experiments**: Commands will persist in `./artifacts` directory
4. **Clean when needed**: `make docker-clean`

## Environment Details

- **Base Image**: python:3.12-slim
- **PyTorch**: CPU-only version from official index
- **GDAL/Rasterio**: System packages installed before Python dependencies
- **Working Directory**: `/app`
- **Python Path**: Set to include `/app/src`

## File Structure

After running, your local directory will contain:
```
project/
├── data/                    # Input multispectral images
├── artifacts/              # All pipeline outputs
│   ├── aligned/           # Step 1 output
│   ├── indices/           # Step 2 output
│   ├── boundary_maps/     # Step 2 output
│   ├── masks_final/       # Step 3 output
│   ├── models/            # Step 4 output
│   ├── preds/             # Step 4 output
│   └── reports/           # All step reports
├── logs/                  # Training and runtime logs
└── src/                   # Source code (read-only in container)