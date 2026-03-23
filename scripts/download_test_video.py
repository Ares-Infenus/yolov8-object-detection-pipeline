"""
Download a public test video for inference demo.
Uses Ultralytics sample video (traffic/pedestrians, ~5MB).
"""
import sys
import urllib.request
from pathlib import Path

VIDEO_URL = "https://ultralytics.com/assets/decelera.mp4"
VIDEO_PATH = Path("data/test_videos/sample.mp4")


def download_video():
    VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
    if VIDEO_PATH.exists():
        print(f"✅ Video already exists: {VIDEO_PATH}")
        return True
    print(f"Downloading test video from {VIDEO_URL}...")
    try:
        urllib.request.urlretrieve(VIDEO_URL, str(VIDEO_PATH))
        size_mb = VIDEO_PATH.stat().st_size / (1024 * 1024)
        print(f"✅ Video saved: {VIDEO_PATH} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


if __name__ == "__main__":
    success = download_video()
    sys.exit(0 if success else 1)
