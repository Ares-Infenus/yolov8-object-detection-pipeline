"""
Test that the repository structure is complete and valid.
Runs in CI without GPU — only checks file presence and format.
"""
import pytest
from pathlib import Path

REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    "Makefile",
    ".gitignore",
    "LICENSE",
    "config/coco128.yaml",
    "scripts/setup.sh",
    "scripts/download_dataset.py",
    "scripts/download_weights.py",
    "scripts/download_test_video.py",
    "scripts/comprobador_general.py",
    "src/__init__.py",
    "src/train.py",
    "src/evaluate.py",
    "src/inference.py",
    "src/export.py",
    "src/demo_video.py",
    "notebooks/training_pipeline.ipynb",
]

REQUIRED_DIRS = [
    ".github/workflows",
    "config",
    "scripts",
    "src",
    "tests",
    "notebooks",
    "docs",
    "results",
    "models",
    "data",
]

PHASE_VERIFIERS = [
    f"scripts/fase{i}_{name}_verificar.py"
    for i, name in enumerate([
        "setup", "datos", "entrenamiento",
        "evaluacion", "inferencia", "exportacion", "demo",
    ])
]


class TestRepoStructure:
    @pytest.mark.parametrize("filepath", REQUIRED_FILES)
    def test_required_file_exists(self, filepath):
        assert Path(filepath).exists(), f"Missing required file: {filepath}"

    @pytest.mark.parametrize("dirpath", REQUIRED_DIRS)
    def test_required_dir_exists(self, dirpath):
        assert Path(dirpath).exists(), f"Missing required directory: {dirpath}"

    @pytest.mark.parametrize("verifier", PHASE_VERIFIERS)
    def test_phase_verifier_exists(self, verifier):
        assert Path(verifier).exists(), f"Missing phase verifier: {verifier}"

    def test_gitignore_excludes_heavy_files(self):
        gitignore = Path(".gitignore").read_text()
        assert "*.pt" in gitignore, ".gitignore must exclude .pt files"
        assert "*.onnx" in gitignore, ".gitignore must exclude .onnx files"
        assert "runs/" in gitignore, ".gitignore must exclude runs/"
        assert "*.mp4" in gitignore, ".gitignore must exclude .mp4 files"

    def test_requirements_has_core_deps(self):
        reqs = Path("requirements.txt").read_text().lower()
        for dep in ["ultralytics", "opencv", "onnx", "torch", "matplotlib"]:
            assert dep in reqs, f"requirements.txt missing: {dep}"
