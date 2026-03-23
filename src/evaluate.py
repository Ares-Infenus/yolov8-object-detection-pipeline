"""
Evaluate trained YOLOv8 model and compare with base (pre-trained) model.
Saves comparison metrics to results/metrics/.
"""
import sys
import json
from pathlib import Path


def find_best_weights():
    """Find the most recent best.pt from training runs."""
    candidates = sorted(Path("runs/detect").rglob("best.pt"), key=lambda p: p.stat().st_mtime)
    if candidates:
        return str(candidates[-1])
    return None


def evaluate_model(model_path, data="coco128.yaml"):
    """Run validation and return metrics dict."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    results = model.val(data=data, verbose=False)
    return {
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
    }


def evaluate(
    base_model="models/yolov8n.pt",
    data="coco128.yaml",
):
    from ultralytics import YOLO

    metrics_dir = Path("results/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Find trained weights
    trained_path = find_best_weights()
    if not trained_path:
        print("❌ No trained weights found. Run training first.")
        return False

    # Fallback for base model
    if not Path(base_model).exists():
        base_model = "yolov8n.pt"

    print(f"Evaluating base model: {base_model}")
    base_metrics = evaluate_model(base_model, data)
    print(f"  Base mAP@50: {base_metrics['mAP50']:.4f}")

    print(f"Evaluating trained model: {trained_path}")
    trained_metrics = evaluate_model(trained_path, data)
    print(f"  Trained mAP@50: {trained_metrics['mAP50']:.4f}")

    # Save comparison
    comparison = {
        "base_model": base_metrics,
        "trained_model": trained_metrics,
        "base_model_path": base_model,
        "trained_model_path": trained_path,
    }
    comp_path = metrics_dir / "comparison_base_vs_trained.json"
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"✅ Comparison saved to: {comp_path}")

    # Save training summary
    summary = {
        "model": "YOLOv8n",
        "dataset": "COCO128",
        "trained_weights": trained_path,
        **trained_metrics,
    }
    summary_path = metrics_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Summary saved to: {summary_path}")

    return True


if __name__ == "__main__":
    ok = evaluate()
    sys.exit(0 if ok else 1)
