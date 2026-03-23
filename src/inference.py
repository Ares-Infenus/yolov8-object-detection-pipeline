"""
Run inference on images and benchmark speed.
Saves sample detections and speed metrics.
"""
import sys
import json
import time
import shutil
from pathlib import Path


def find_best_weights():
    """Find the most recent best.pt from training runs."""
    candidates = sorted(Path("runs/detect").rglob("best.pt"), key=lambda p: p.stat().st_mtime)
    if candidates:
        return str(candidates[-1])
    return None


def run_inference(
    data="coco128.yaml",
    n_samples=3,
    n_speed_iterations=100,
):
    from ultralytics import YOLO
    import numpy as np

    # Find trained weights
    weights = find_best_weights()
    if not weights:
        print("❌ No trained weights found. Run training first.")
        return False

    model = YOLO(weights)

    # Find dataset images
    possible_dirs = [
        Path("data/coco128/images/train2017"),
        Path("datasets/coco128/images/train2017"),
        Path.home() / "datasets" / "coco128" / "images" / "train2017",
    ]
    images_dir = None
    for d in possible_dirs:
        if d.exists():
            images_dir = d
            break

    if not images_dir:
        print("❌ Dataset images not found. Run download_dataset.py first.")
        return False

    images = sorted(images_dir.glob("*.jpg"))
    if not images:
        print("❌ No images found in dataset.")
        return False

    # ── Run inference on samples and save ──
    samples_dir = Path("results/samples")
    samples_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running inference on {n_samples} sample images...")
    sample_images = images[:n_samples]
    results = model.predict(
        source=[str(img) for img in sample_images],
        save=True,
        project="runs/detect",
        name="predict",
        exist_ok=True,
    )

    # Copy sample detections
    predict_dir = Path("runs/detect/predict")
    if predict_dir.exists():
        saved_imgs = sorted(predict_dir.glob("*.jpg"))[:n_samples]
        for i, img in enumerate(saved_imgs):
            dst = samples_dir / f"detection_sample_{i + 1}.jpg"
            shutil.copy(str(img), str(dst))
            print(f"  Saved: {dst}")

    # ── Speed benchmark ──
    print(f"Running speed benchmark ({n_speed_iterations} iterations)...")
    import numpy as np

    # Warm up
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    for _ in range(5):
        model.predict(dummy, verbose=False)

    # Benchmark
    latencies = []
    for _ in range(n_speed_iterations):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms

    latencies = np.array(latencies)
    speed = {
        "fps_avg": round(1000.0 / latencies.mean(), 2),
        "latency_avg_ms": round(float(latencies.mean()), 2),
        "latency_std_ms": round(float(latencies.std()), 2),
        "latency_min_ms": round(float(latencies.min()), 2),
        "latency_max_ms": round(float(latencies.max()), 2),
        "n_iterations": n_speed_iterations,
        "image_size": 640,
        "model": weights,
    }

    metrics_dir = Path("results/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    speed_path = metrics_dir / "speed_benchmark.json"
    with open(speed_path, "w") as f:
        json.dump(speed, f, indent=2)
    print(f"✅ Speed benchmark saved to: {speed_path}")
    print(f"   FPS: {speed['fps_avg']} | Latency: {speed['latency_avg_ms']}ms")

    return True


if __name__ == "__main__":
    ok = run_inference()
    sys.exit(0 if ok else 1)
