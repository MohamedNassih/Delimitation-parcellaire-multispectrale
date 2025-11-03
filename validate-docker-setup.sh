#!/bin/bash
# Docker Setup Validation Script
# This script validates that all Docker configuration files are correctly structured

echo "🔍 Validating Docker setup for Agricultural Field Boundary Detection..."

# Check if required files exist
echo "Checking required files..."
required_files=(
    "Dockerfile"
    "docker-compose.yml"
    ".dockerignore"
    "Makefile"
    "requirements-docker.txt"
    "README-Docker.md"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file missing"
        exit 1
    fi
done

echo ""
echo "🔍 Validating Dockerfile structure..."

# Check Dockerfile content
if grep -q "python:3.12-slim" Dockerfile; then
    echo "✅ Correct base image (python:3.12-slim)"
else
    echo "❌ Incorrect base image"
    exit 1
fi

if grep -q "gdal-bin" Dockerfile && grep -q "libgdal-dev" Dockerfile; then
    echo "✅ GDAL dependencies included"
else
    echo "❌ Missing GDAL dependencies"
    exit 1
fi

if grep -q "download.pytorch.org/whl/cpu" Dockerfile; then
    echo "✅ PyTorch CPU installation from official index"
else
    echo "❌ PyTorch not installed from official index"
    exit 1
fi

if grep -q "requirements-docker.txt" Dockerfile; then
    echo "✅ requirements-docker.txt used"
else
    echo "❌ requirements-docker.txt not used in Dockerfile"
    exit 1
fi

echo ""
echo "🔍 Validating docker-compose.yml structure..."

if grep -q "agritech-pipeline" docker-compose.yml; then
    echo "✅ Service name defined"
else
    echo "❌ Service name missing"
    exit 1
fi

if grep -q "./data:/app/data" docker-compose.yml && grep -q "./artifacts:/app/artifacts" docker-compose.yml; then
    echo "✅ Volume mounts for data and artifacts"
else
    echo "❌ Missing required volume mounts"
    exit 1
fi

echo ""
echo "🔍 Validating Makefile Docker targets..."

docker_targets=("build" "shell" "docker-prepare" "docker-indices" "docker-masks" "docker-train" "docker-infer" "docker-clean")

for target in "${docker_targets[@]}"; do
    if grep -q "$target:" Makefile; then
        echo "✅ Makefile target '$target' exists"
    else
        echo "❌ Makefile target '$target' missing"
        exit 1
    fi
done

echo ""
echo "🔍 Validating requirements-docker.txt..."

if grep -q "torch" requirements-docker.txt; then
    echo "❌ PyTorch should not be in requirements-docker.txt"
    exit 1
else
    echo "✅ PyTorch correctly excluded from requirements-docker.txt"
fi

if grep -q "rasterio" requirements-docker.txt && grep -q "opencv-python" requirements-docker.txt; then
    echo "✅ Key dependencies present"
else
    echo "❌ Missing key dependencies"
    exit 1
fi

echo ""
echo "🔍 Validating .dockerignore..."

if grep -q "data/" .dockerignore && grep -q "artifacts/" .dockerignore; then
    echo "✅ Data directories excluded from build context"
else
    echo "❌ Data directories not properly excluded"
    exit 1
fi

if grep -q ".git" .dockerignore; then
    echo "✅ Git directory excluded"
else
    echo "❌ Git directory not excluded"
    exit 1
fi

echo ""
echo "🎉 All validations passed!"
echo ""
echo "📋 Summary of created files:"
echo "  • Dockerfile (Python 3.12-slim + GDAL + PyTorch CPU)"
echo "  • docker-compose.yml (Windows-compatible volume mounts)"
echo "  • requirements-docker.txt (No PyTorch, for separate installation)"
echo "  • Makefile (Local + Docker targets)"
echo "  • .dockerignore (Minimal build context)"
echo "  • README-Docker.md (Comprehensive guide)"
echo ""
echo "🚀 Next steps:"
echo "  1. Ensure Docker Desktop is running on Windows"
echo "  2. Run 'make build' or 'docker-compose build'"
echo "  3. Run 'make shell' to enter container"
echo "  4. Follow pipeline steps in container"
echo ""
echo "📖 See README-Docker.md for detailed usage instructions"