"""
Fase 5 Verifier: ONNX Export
Checks that ONNX export was successful and the model is valid.
"""
import sys
import json
from pathlib import Path


def verify_export():
    results = {"fase": 5, "nombre": "exportacion", "checks": {}, "status": "PASS"}

    # Check ONNX file exists
    onnx_paths = list(Path("models").glob("*.onnx"))
    if not onnx_paths:
        onnx_paths = list(Path("runs").rglob("*.onnx"))

    results["checks"]["onnx_exists"] = {
        "ok": len(onnx_paths) > 0,
        "paths": [str(p) for p in onnx_paths],
    }

    if onnx_paths:
        onnx_path = onnx_paths[0]
        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        results["checks"]["onnx_size_mb"] = {
            "value": round(size_mb, 2),
            "ok": size_mb > 1,
        }

        # Validate ONNX model
        try:
            import onnx
            model = onnx.load(str(onnx_path))
            onnx.checker.check_model(model)
            results["checks"]["onnx_valid"] = {"ok": True}
        except Exception as e:
            results["checks"]["onnx_valid"] = {"ok": False, "error": str(e)}

    # Determine overall status
    failed = [k for k, v in results["checks"].items() if not v.get("ok", True)]
    if failed:
        results["status"] = "FAIL"
        results["failed_checks"] = failed

    print(json.dumps(results, indent=2))
    return len(failed) == 0


if __name__ == "__main__":
    ok = verify_export()
    sys.exit(0 if ok else 1)
