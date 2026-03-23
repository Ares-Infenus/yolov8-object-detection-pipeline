"""
Fase 0 Verifier: Environment Setup
Checks that all dependencies, GPU, and git are available.
"""
import sys
import json


def verify_setup():
    results = {"fase": 0, "nombre": "setup", "checks": {}, "status": "PASS"}

    # Check Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    results["checks"]["python_version"] = {
        "value": py_version,
        "ok": sys.version_info >= (3, 8),
    }

    # Check core imports
    core_packages = {
        "ultralytics": "ultralytics",
        "cv2": "opencv-python-headless",
        "torch": "torch",
        "onnx": "onnx",
        "matplotlib": "matplotlib",
        "yaml": "pyyaml",
    }
    for module, pkg in core_packages.items():
        try:
            __import__(module)
            results["checks"][f"import_{module}"] = {"ok": True}
        except ImportError:
            results["checks"][f"import_{module}"] = {"ok": False, "error": f"pip install {pkg}"}

    # Check GPU
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else "N/A"
        results["checks"]["gpu"] = {
            "available": gpu_available,
            "name": gpu_name,
            "ok": True,  # GPU is optional
        }
    except Exception as e:
        results["checks"]["gpu"] = {"available": False, "ok": True, "note": str(e)}

    # Check git
    import shutil
    git_path = shutil.which("git")
    results["checks"]["git"] = {"available": git_path is not None, "ok": git_path is not None}

    # Determine overall status
    failed = [k for k, v in results["checks"].items() if not v.get("ok", True)]
    if failed:
        results["status"] = "FAIL"
        results["failed_checks"] = failed

    print(json.dumps(results, indent=2))
    return len(failed) == 0


if __name__ == "__main__":
    ok = verify_setup()
    sys.exit(0 if ok else 1)
