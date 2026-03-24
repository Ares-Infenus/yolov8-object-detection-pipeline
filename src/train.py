"""
Train YOLOv8n on COCO128 dataset.
Designed for Google Colab Free (Tesla T4, 16GB VRAM).
Saves checkpoints and training artifacts.
"""
import shutil
from pathlib import Path


def train(
    model_path="models/yolov8n.pt",
    data="coco128.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="train",
    exist_ok=True,
):
    from ultralytics import YOLO

    # Fallback: if weights not in models/, try default location
    if not Path(model_path).exists():
        model_path = "yolov8n.pt"

    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    print(f"Training on {data} for {epochs} epochs...")
    print(f"  Image size: {imgsz}, Batch: {batch}")

    results = model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=name,
        exist_ok=exist_ok,
        verbose=True,
    )

    # Copy training plots to docs/images/ for README
    # Find actual training directory via best.pt
    train_dir = None
    for best_pt in Path(".").rglob("best.pt"):
        train_dir = best_pt.parent.parent
        break
    if train_dir is None:
        train_dir = Path("runs/detect/train")
    docs_dir = Path("docs/images")
    docs_dir.mkdir(parents=True, exist_ok=True)

    for src_name, dst_name in [
        ("results.png", "training_curves.png"),
        ("confusion_matrix.png", "confusion_matrix.png"),
    ]:
        src = train_dir / src_name
        if src.exists():
            shutil.copy(str(src), str(docs_dir / dst_name))
            print(f"  Copied {src_name} -> docs/images/{dst_name}")

    print("✅ Training complete!")
    return results


if __name__ == "__main__":
    train()
