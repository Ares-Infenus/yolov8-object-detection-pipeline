"""
Fase 6 Verifier: Demo Video
Checks that demo video was generated successfully.
"""
import sys
import json
from pathlib import Path


def verify_demo():
    results = {"fase": 6, "nombre": "demo", "checks": {}, "status": "PASS"}

    # Check for output video
    video_patterns = ["results/*.mp4", "runs/**/*.mp4", "output*.mp4"]
    video_found = False
    video_path = None

    for pattern in video_patterns:
        matches = list(Path(".").glob(pattern))
        if matches:
            video_found = True
            video_path = str(matches[0])
            break

    results["checks"]["demo_video_exists"] = {
        "ok": video_found,
        "path": video_path,
    }

    if video_found and video_path:
        size_mb = Path(video_path).stat().st_size / (1024 * 1024)
        results["checks"]["video_size_mb"] = {
            "value": round(size_mb, 2),
            "ok": size_mb > 0.1,
        }

    # Check test video input exists
    test_video = Path("data/test_videos/sample.mp4")
    results["checks"]["test_video_input"] = {"ok": test_video.exists()}

    # Determine overall status
    failed = [k for k, v in results["checks"].items() if not v.get("ok", True)]
    if failed:
        results["status"] = "FAIL"
        results["failed_checks"] = failed

    print(json.dumps(results, indent=2))
    return len(failed) == 0


if __name__ == "__main__":
    ok = verify_demo()
    sys.exit(0 if ok else 1)
