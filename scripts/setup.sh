#!/bin/bash
# =============================================================
# SETUP SCRIPT — Descarga todo lo necesario para reproducir
# Uso: chmod +x scripts/setup.sh && ./scripts/setup.sh
# =============================================================
set -e

echo "============================================"
echo "  YOLOv8 COCO128 Detection — Setup"
echo "============================================"

# 1. Install Python dependencies
echo "[1/4] Installing dependencies..."
pip install -r requirements.txt --quiet

# 2. Download COCO128 dataset
echo "[2/4] Downloading COCO128 dataset (~7MB)..."
python scripts/download_dataset.py

# 3. Download pre-trained weights
echo "[3/4] Downloading YOLOv8n weights (~6MB)..."
python scripts/download_weights.py

# 4. Download test video
echo "[4/4] Downloading test video (~5MB)..."
python scripts/download_test_video.py

echo ""
echo "✅ Setup complete!"
echo "   Run: make train (or python src/train.py)"
echo "   Or open notebooks/training_pipeline.ipynb in Colab"
echo "============================================"
