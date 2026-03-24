"""
Download a public test video for inference demo.
Tries multiple sources. Falls back to generating a video from dataset images.
"""
import sys
import urllib.request
from pathlib import Path

VIDEO_URLS = [
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/decelera_portrait_min.mov",
    "https://ultralytics.com/assets/decelera.mp4",
]
VIDEO_PATH = Path("data/test_videos/sample.mp4")


def download_video():
    VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
    if VIDEO_PATH.exists():
        print(f"✅ Video already exists: {VIDEO_PATH}")
        return True

    # Try each URL
    for url in VIDEO_URLS:
        print(f"Trying: {url}...")
        try:
            urllib.request.urlretrieve(url, str(VIDEO_PATH))
            size_mb = VIDEO_PATH.stat().st_size / (1024 * 1024)
            print(f"✅ Video saved: {VIDEO_PATH} ({size_mb:.1f} MB)")
            return True
        except Exception as e:
            print(f"  ⚠️ Failed: {e}")
            if VIDEO_PATH.exists():
                VIDEO_PATH.unlink()

    # Fallback: generate a simple video from dataset images
    print("Generating test video from dataset images...")
    try:
        import cv2
        import numpy as np

        # Find dataset images
        images_dir = None
        for p in [Path("datasets/coco128/images/train2017"),
                  Path("data/coco128/images/train2017")]:
            if p.exists():
                images_dir = p
                break
        if images_dir is None:
            for p in Path(".").rglob("coco128"):
                candidate = p / "images" / "train2017"
                if candidate.exists():
                    images_dir = candidate
                    break

        if images_dir is None:
            print("❌ No dataset images found to generate video.")
            return False

        imgs = sorted(images_dir.glob("*.jpg"))[:30]
        if not imgs:
            print("❌ No jpg images found.")
            return False

        # Read first image to get dimensions
        first = cv2.imread(str(imgs[0]))
        h, w = first.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(VIDEO_PATH), fourcc, 5, (w, h))

        for img_path in imgs:
            frame = cv2.imread(str(img_path))
            if frame is not None:
                frame = cv2.resize(frame, (w, h))
                writer.write(frame)

        writer.release()
        size_mb = VIDEO_PATH.stat().st_size / (1024 * 1024)
        print(f"✅ Generated video from {len(imgs)} images: {VIDEO_PATH} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"❌ Failed to generate video: {e}")
        return False


if __name__ == "__main__":
    success = download_video()
    sys.exit(0 if success else 1)
