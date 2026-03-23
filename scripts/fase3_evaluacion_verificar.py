"""
Fase 3 Verifier: Evaluation
Checks that evaluation metrics were generated and saved.
"""
import sys
import json
from pathlib import Path


def verify_evaluation():
    results = {"fase": 3, "nombre": "evaluacion", "checks": {}, "status": "PASS"}

    # Check comparison JSON
    comp_path = Path("results/metrics/comparison_base_vs_trained.json")
    results["checks"]["comparison_json"] = {"ok": comp_path.exists()}

    if comp_path.exists():
        with open(comp_path) as f:
            comp = json.load(f)
        # Verify structure
        has_base = "base_model" in comp
        has_trained = "trained_model" in comp
        results["checks"]["comparison_structure"] = {
            "ok": has_base and has_trained,
            "has_base": has_base,
            "has_trained": has_trained,
        }
        # Verify metrics exist
        if has_trained:
            trained = comp["trained_model"]
            has_map = "mAP50" in trained and "mAP50_95" in trained
            results["checks"]["metrics_present"] = {
                "ok": has_map,
                "mAP50": trained.get("mAP50"),
                "mAP50_95": trained.get("mAP50_95"),
            }

    # Check training summary
    summary_path = Path("results/metrics/training_summary.json")
    results["checks"]["training_summary"] = {"ok": summary_path.exists()}

    # Determine overall status
    failed = [k for k, v in results["checks"].items() if not v.get("ok", True)]
    if failed:
        results["status"] = "FAIL"
        results["failed_checks"] = failed

    print(json.dumps(results, indent=2))
    return len(failed) == 0


if __name__ == "__main__":
    ok = verify_evaluation()
    sys.exit(0 if ok else 1)
