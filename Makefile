.PHONY: setup train evaluate export demo test lint clean all report

# ============================================================
# SETUP — Run this first after git clone
# ============================================================
setup:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "📥 Downloading dataset and weights..."
	python scripts/download_dataset.py
	python scripts/download_weights.py
	python scripts/download_test_video.py
	@echo "✅ Setup complete. Run 'make train' to start."

# ============================================================
# PIPELINE — Run phases in order
# ============================================================
train:
	@echo "🏋️ Training YOLOv8n on COCO128..."
	python src/train.py
	python scripts/fase2_entrenamiento_verificar.py

evaluate:
	@echo "📊 Evaluating model..."
	python src/evaluate.py
	python scripts/fase3_evaluacion_verificar.py

inference:
	@echo "🔍 Running inference..."
	python src/inference.py
	python scripts/fase4_inferencia_verificar.py

export:
	@echo "📤 Exporting to ONNX..."
	python src/export.py
	python scripts/fase5_exportacion_verificar.py

demo:
	@echo "🎬 Generating demo video..."
	python src/demo_video.py
	python scripts/fase6_demo_verificar.py

# ============================================================
# FULL PIPELINE
# ============================================================
all: setup train evaluate inference export demo report
	@echo "🎉 Full pipeline complete!"
	python scripts/comprobador_general.py

# ============================================================
# REPORTING
# ============================================================
report:
	@echo "📈 Generating executive report..."
	python scripts/generate_executive_report.py

# ============================================================
# QUALITY
# ============================================================
test:
	pytest tests/ -v

lint:
	ruff check src/ scripts/ tests/

# ============================================================
# CLEANUP
# ============================================================
clean:
	rm -rf runs/ models/*.pt models/*.onnx data/coco128/ results/samples/*.jpg
	@echo "🧹 Cleaned generated files. Run 'make setup' to restore."
