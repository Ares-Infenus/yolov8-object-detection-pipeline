"""
Export trained YOLOv8 model to ONNX format.
ONNX enables cross-platform deployment (cloud, edge, mobile, browser).
"""
import sys
import shutil
from pathlib import Path


def find_best_weights():
    """Find the most recent best.pt from training runs."""
    candidates = sorted(Path("runs/detect").rglob("best.pt"), key=lambda p: p.stat().st_mtime)
    if candidates:
        return str(candidates[-1])
    return None


def export_onnx(imgsz=640):
    from ultralytics import YOLO

    weights = find_best_weights()
    if not weights:
        print("❌ No trained weights found. Run training first.")
        return False

    print(f"Exporting {weights} to ONNX...")
    model = YOLO(weights)
    onnx_path = model.export(format="onnx", imgsz=imgsz)
    print(f"  ONNX exported to: {onnx_path}")

    # Copy to models/ directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    dst = models_dir / "yolov8n_trained.onnx"
    shutil.copy(str(onnx_path), str(dst))
    print(f"  Copied to: {dst}")

    # Also copy best.pt to models/
    pt_dst = models_dir / "yolov8n_trained.pt"
    shutil.copy(weights, str(pt_dst))
    print(f"  Copied weights to: {pt_dst}")

    # Validate ONNX
    try:
        import onnx
        onnx_model = onnx.load(str(dst))
        onnx.checker.check_model(onnx_model)
        print("  ✅ ONNX model validation passed")
    except Exception as e:
        print(f"  ⚠️  ONNX validation warning: {e}")

    size_mb = Path(dst).stat().st_size / (1024 * 1024)
    print(f"✅ Export complete! ONNX size: {size_mb:.1f} MB")
    return True


if __name__ == "__main__":
    ok = export_onnx()
    sys.exit(0 if ok else 1)
