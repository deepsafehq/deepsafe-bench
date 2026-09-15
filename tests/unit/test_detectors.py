"""Tests for loading third-party detectors."""

import pathlib

import pytest

from deepsafe.detectors import DetectorLoadError, load_detector


def _write(tmp_path: pathlib.Path, name: str, body: str) -> pathlib.Path:
    path = tmp_path / name
    path.write_text(body)
    return path


class TestLoadDetector:
    def test_loads_module_level_function(self, tmp_path):
        f = _write(tmp_path, "d.py", "def predict(path):\n    return 0.42\n")
        assert load_detector(f)(pathlib.Path("x.jpg")) == 0.42

    def test_loads_detector_class(self, tmp_path):
        f = _write(tmp_path, "d.py",
                   "class Detector:\n"
                   "    def predict(self, path):\n"
                   "        return 0.7\n")
        assert load_detector(f)(pathlib.Path("x.jpg")) == 0.7

    def test_function_wins_over_class(self, tmp_path):
        f = _write(tmp_path, "d.py",
                   "def predict(path):\n    return 1.0\n"
                   "class Detector:\n"
                   "    def predict(self, path):\n        return 0.0\n")
        assert load_detector(f)(pathlib.Path("x.jpg")) == 1.0

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(DetectorLoadError, match="no such detector"):
            load_detector(tmp_path / "nope.py")

    def test_import_error_is_wrapped(self, tmp_path):
        f = _write(tmp_path, "bad.py", "raise RuntimeError('boom')\n")
        with pytest.raises(DetectorLoadError, match="raised while importing"):
            load_detector(f)

    def test_missing_predict_explains_the_contract(self, tmp_path):
        f = _write(tmp_path, "empty.py", "x = 1\n")
        with pytest.raises(DetectorLoadError, match="exposes no `predict`"):
            load_detector(f)
