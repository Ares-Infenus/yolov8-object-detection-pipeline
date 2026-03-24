"""
Download YOLOv8n pre-trained weights from Ultralytics.
Weights are ~6MB and stored in models/ directory.
"""
import sys
from pathlib import Path


def download_weights():
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    print("Downloading YOLOv8n pre-trained weights...")
    try:
        from ultralytics import YOLO
        import shutil
        # This automatically downloads yolov8n.pt if not present
        YOLO('yolov8n.pt')
        # Move to models/ directory
        src = Path('yolov8n.pt')
        dst = models_dir / 'yolov8n.pt'
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
        print(f"✅ Weights saved to: {dst}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


if __name__ == "__main__":
    success = download_weights()
    sys.exit(0 if success else 1)
