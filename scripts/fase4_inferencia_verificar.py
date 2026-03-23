"""
Fase 4 Verifier: Inference + Speed Benchmark
Checks that inference ran successfully and speed metrics were saved.
"""
import sys
import json
from pathlib import Path


def verify_inference():
    results = {"fase": 4, "nombre": "inferencia", "checks": {}, "status": "PASS"}

    # Check speed benchmark
    speed_path = Path("results/metrics/speed_benchmark.json")
    results["checks"]["speed_benchmark"] = {"ok": speed_path.exists()}

    if speed_path.exists():
        with open(speed_path) as f:
            speed = json.load(f)
        results["checks"]["fps_measured"] = {
            "ok": "fps_avg" in speed,
            "value": speed.get("fps_avg"),
        }
        results["checks"]["latency_measured"] = {
            "ok": "latency_avg_ms" in speed,
            "value": speed.get("latency_avg_ms"),
        }

    # Check sample detections
    samples_dir = Path("results/samples")
    if samples_dir.exists():
        samples = list(samples_dir.glob("detection_sample_*.jpg"))
        results["checks"]["sample_images"] = {
            "count": len(samples),
            "ok": len(samples) >= 3,
        }
    else:
        results["checks"]["sample_images"] = {"count": 0, "ok": False}

    # Determine overall status
    failed = [k for k, v in results["checks"].items() if not v.get("ok", True)]
    if failed:
        results["status"] = "FAIL"
        results["failed_checks"] = failed

    print(json.dumps(results, indent=2))
    return len(failed) == 0


if __name__ == "__main__":
    ok = verify_inference()
    sys.exit(0 if ok else 1)
