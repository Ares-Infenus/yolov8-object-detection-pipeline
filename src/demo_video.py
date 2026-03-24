"""
Generate demo video with object detections overlaid.
Uses the test video downloaded by download_test_video.py.
"""
import sys
from pathlib import Path


def find_best_weights():
    """Find the most recent best.pt from training runs."""
    candidates = sorted(Path(".").rglob("best.pt"), key=lambda p: p.stat().st_mtime)
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
        # Try to generate one on the fly
        print(f"⚠️  Test video not found at {video_path}, attempting to generate...")
        import subprocess
        subprocess.run([sys.executable, "scripts/download_test_video.py"], check=False)
    if not video.exists():
        print(f"❌ Test video not found: {video_path}")
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
        name="demo",
        exist_ok=True,
    )

    # Find the output video — search broadly
    demo_dir = None
    for candidate in sorted(Path(".").rglob("demo"), key=lambda p: p.stat().st_mtime):
        if candidate.is_dir():
            vids = list(candidate.glob("*.mp4")) + list(candidate.glob("*.avi")) + list(candidate.glob("*.mov"))
            if vids:
                demo_dir = candidate
    if demo_dir is None:
        demo_dir = Path("runs/detect/demo")

    if demo_dir.exists():
        output_videos = list(demo_dir.glob("*.mp4")) + list(demo_dir.glob("*.avi")) + list(demo_dir.glob("*.mov"))
        if output_videos:
            print(f"✅ Demo video saved to: {output_videos[0]}")
        else:
            print("⚠️  No video files found in demo directory.")
    else:
        print("⚠️  Demo directory not found.")

    return True


if __name__ == "__main__":
    ok = generate_demo()
    sys.exit(0 if ok else 1)
