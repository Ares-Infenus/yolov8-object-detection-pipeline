"""
Generate demo video with object detections overlaid.
Uses the test video downloaded by download_test_video.py.
"""
import sys
from pathlib import Path


def find_best_weights():
    """Find the most recent best.pt from training runs."""
    candidates = sorted(Path("runs/detect").rglob("best.pt"), key=lambda p: p.stat().st_mtime)
    if candidates:
        return str(candidates[-1])
    return None


def generate_demo(
    video_path="data/test_videos/sample.mp4",
    output_dir="results",
):
    from ultralytics import YOLO

    video = Path(video_path)
    if not video.exists():
        print(f"❌ Test video not found: {video_path}")
        print("   Run: python scripts/download_test_video.py")
        return False

    weights = find_best_weights()
    if not weights:
        print("❌ No trained weights found. Run training first.")
        return False

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Generating demo video...")
    print(f"  Model: {weights}")
    print(f"  Input: {video_path}")

    model = YOLO(weights)
    results = model.predict(
        source=str(video),
        save=True,
        project="runs/detect",
        name="demo",
        exist_ok=True,
    )

    # Find the output video
    demo_dir = Path("runs/detect/demo")
    output_videos = list(demo_dir.glob("*.mp4")) + list(demo_dir.glob("*.avi"))
    if output_videos:
        print(f"✅ Demo video saved to: {output_videos[0]}")
    else:
        print("⚠️  Video may have been saved in a different format.")

    return True


if __name__ == "__main__":
    ok = generate_demo()
    sys.exit(0 if ok else 1)
