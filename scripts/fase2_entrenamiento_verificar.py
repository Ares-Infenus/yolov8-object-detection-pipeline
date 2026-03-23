"""
Fase 2 Verifier: Training
Checks that training completed successfully and produced expected outputs.
"""
import sys
import json
from pathlib import Path


def verify_training():
    results = {"fase": 2, "nombre": "entrenamiento", "checks": {}, "status": "PASS"}

    # Look for training outputs in common locations
    possible_dirs = [
        Path("runs/detect/train"),
        Path("runs/detect/train2"),
        Path("runs/detect/train3"),
    ]
    train_dir = None
    for d in possible_dirs:
        if d.exists():
            train_dir = d

    results["checks"]["train_dir_exists"] = {
        "ok": train_dir is not None,
        "path": str(train_dir) if train_dir else None,
    }

    if train_dir:
        # Check for best weights
        best_pt = train_dir / "weights" / "best.pt"
        last_pt = train_dir / "weights" / "last.pt"
        results["checks"]["best_weights"] = {"ok": best_pt.exists(), "path": str(best_pt)}
        results["checks"]["last_weights"] = {"ok": last_pt.exists(), "path": str(last_pt)}

        # Check for results file
        results_csv = train_dir / "results.csv"
        results["checks"]["results_csv"] = {"ok": results_csv.exists()}

        # Check for results plot
        results_png = train_dir / "results.png"
        results["checks"]["results_plot"] = {"ok": results_png.exists()}

        # Check for confusion matrix
        cm_png = train_dir / "confusion_matrix.png"
        results["checks"]["confusion_matrix"] = {"ok": cm_png.exists()}

        # Verify weights file size (should be > 1MB)
        if best_pt.exists():
            size_mb = best_pt.stat().st_size / (1024 * 1024)
            results["checks"]["weights_size_mb"] = {
                "value": round(size_mb, 2),
                "ok": size_mb > 1,
            }

    # Determine overall status
    failed = [k for k, v in results["checks"].items() if not v.get("ok", True)]
    if failed:
        results["status"] = "FAIL"
        results["failed_checks"] = failed

    print(json.dumps(results, indent=2))
    return len(failed) == 0


if __name__ == "__main__":
    ok = verify_training()
    sys.exit(0 if ok else 1)
