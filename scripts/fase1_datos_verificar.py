"""
Fase 1 Verifier: Data Preparation
Checks that COCO128 dataset is downloaded and valid.
"""
import sys
import json
from pathlib import Path


def verify_data():
    results = {"fase": 1, "nombre": "datos", "checks": {}, "status": "PASS"}

    # Check dataset config exists
    config_path = Path("config/coco128.yaml")
    results["checks"]["config_exists"] = {"ok": config_path.exists()}

    # Check dataset directory
    # Ultralytics may download to different locations
    possible_paths = [
        Path("data/coco128"),
        Path("datasets/coco128"),
        Path.home() / "datasets" / "coco128",
    ]
    dataset_found = False
    dataset_path = None
    for p in possible_paths:
        if p.exists():
            dataset_found = True
            dataset_path = str(p)
            break

    # Fallback: search recursively for coco128 images directory
    if not dataset_found:
        for p in Path(".").rglob("coco128"):
            if (p / "images").exists():
                dataset_found = True
                dataset_path = str(p)
                break

    results["checks"]["dataset_downloaded"] = {
        "ok": dataset_found,
        "path": dataset_path,
    }

    if dataset_found and dataset_path:
        # Count images
        images_dir = Path(dataset_path) / "images" / "train2017"
        if images_dir.exists():
            n_images = len(list(images_dir.glob("*.jpg")))
            results["checks"]["image_count"] = {
                "value": n_images,
                "ok": n_images >= 100,
            }

        # Count labels
        labels_dir = Path(dataset_path) / "labels" / "train2017"
        if labels_dir.exists():
            n_labels = len(list(labels_dir.glob("*.txt")))
            results["checks"]["label_count"] = {
                "value": n_labels,
                "ok": n_labels >= 100,
            }

    # Check config validity
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        results["checks"]["config_nc"] = {
            "value": cfg.get("nc"),
            "ok": cfg.get("nc") == 80,
        }
        results["checks"]["config_names"] = {
            "count": len(cfg.get("names", {})),
            "ok": len(cfg.get("names", {})) == 80,
        }

    # Determine overall status
    failed = [k for k, v in results["checks"].items() if not v.get("ok", True)]
    if failed:
        results["status"] = "FAIL"
        results["failed_checks"] = failed

    print(json.dumps(results, indent=2))
    return len(failed) == 0


if __name__ == "__main__":
    ok = verify_data()
    sys.exit(0 if ok else 1)
