"""
Test that configuration files are valid.
"""
import yaml
from pathlib import Path


class TestConfig:
    def test_coco128_yaml_exists(self):
        assert Path("config/coco128.yaml").exists()

    def test_coco128_yaml_valid(self):
        with open("config/coco128.yaml") as f:
            cfg = yaml.safe_load(f)
        assert cfg is not None
        assert "nc" in cfg, "YAML must have 'nc' (number of classes)"
        assert "names" in cfg, "YAML must have 'names' (class names)"

    def test_coco128_has_80_classes(self):
        with open("config/coco128.yaml") as f:
            cfg = yaml.safe_load(f)
        assert cfg["nc"] == 80, f"Expected 80 classes, got {cfg['nc']}"
        assert len(cfg["names"]) == 80, f"Expected 80 names, got {len(cfg['names'])}"

    def test_class_names_are_strings(self):
        with open("config/coco128.yaml") as f:
            cfg = yaml.safe_load(f)
        for idx, name in cfg["names"].items():
            assert isinstance(name, str), f"Class {idx} name must be string, got {type(name)}"
