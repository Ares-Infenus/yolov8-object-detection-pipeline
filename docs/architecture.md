# Architecture — YOLOv8 Object Detection Pipeline

## Pipeline Design

This project implements a 7-phase computer vision pipeline:

| Phase | Module | Description |
|-------|--------|-------------|
| 0 | `fase0_setup_verificar.py` | Environment validation (Python, GPU, dependencies) |
| 1 | `fase1_datos_verificar.py` | COCO128 dataset download and validation |
| 2 | `src/train.py` | Fine-tune YOLOv8n on COCO128 (50 epochs) |
| 3 | `src/evaluate.py` | Evaluate and compare base vs trained model |
| 4 | `src/inference.py` | Run inference + speed benchmark |
| 5 | `src/export.py` | Export to ONNX format |
| 6 | `src/demo_video.py` | Generate annotated demo video |

## Model Architecture

- **Model**: YOLOv8n (nano) — smallest and fastest variant
- **Backbone**: CSP-Darknet53
- **Neck**: PAN-FPN (Path Aggregation Network)
- **Head**: Anchor-free split head with Task-Aligned Assigner
- **Parameters**: ~3.2M
- **FLOPs**: ~8.7G

## Data Flow

```
COCO128 (128 images, 80 classes)
    |
    v
YOLOv8n (pre-trained on full COCO)
    |
    v
Fine-tuning (50 epochs, 640x640, batch=16)
    |
    v
Evaluation (mAP, precision, recall)
    |
    v
Export (.pt -> ONNX)
    |
    v
Demo (video inference with annotations)
```

## Reproducibility Strategy

- Heavy files (models, datasets, videos) are NOT in the repo
- Download scripts fetch everything needed from official sources
- GitHub Actions CI validates that setup scripts work
- Each phase has an automated verifier (exit code 0/1)
