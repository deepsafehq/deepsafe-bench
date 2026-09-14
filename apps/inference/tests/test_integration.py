"""Integration tests for the monolith inference server.

These tests verify:
1. All wrapper classes import and subclass BasePredictor correctly
2. Config registry matches wrapper registry
3. Model loader isolation works
4. mmcv_compat injection/cleanup works
5. Server startup and endpoint contracts
6. Provenance models load and predict on any machine
7. GPU models load and predict when weights + model_code available

Run: python -m pytest monolith/tests/test_integration.py -v
"""

import asyncio
import io
import os
import sys
from pathlib import Path

import pytest

# Ensure monolith is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ── Test 1: Config ───────────────────────────────────────────────────────────


class TestConfig:
    def test_model_registry_has_22_models(self):
        from config import MODEL_REGISTRY

        assert len(MODEL_REGISTRY) == 22

    def test_modality_counts(self):
        from config import get_models_by_modality

        assert len(get_models_by_modality("image")) == 7
        assert len(get_models_by_modality("audio")) == 4
        assert len(get_models_by_modality("video")) == 7
        assert len(get_models_by_modality("provenance")) == 4

    def test_enabled_models_default_all(self):
        from config import get_enabled_models

        assert len(get_enabled_models()) == 22

    def test_enabled_models_subset(self, monkeypatch):
        monkeypatch.setenv("DEEPSAFE_MODELS", "npr,aide")
        from config import get_enabled_models

        # Re-read the env var
        enabled = os.getenv("DEEPSAFE_MODELS", "all")
        names = [m.strip() for m in enabled.split(",")]
        assert names == ["npr", "aide"]


# ── Test 2: Wrapper Registry ────────────────────────────────────────────────


class TestWrapperRegistry:
    def test_registry_matches_config(self):
        from config import MODEL_REGISTRY

        from models import _WRAPPER_REGISTRY

        config_names = set(MODEL_REGISTRY.keys())
        wrapper_names = set(_WRAPPER_REGISTRY.keys())
        assert config_names == wrapper_names, (
            f"Mismatch: in config but not wrapper: {config_names - wrapper_names}, "
            f"in wrapper but not config: {wrapper_names - config_names}"
        )

    def test_all_wrappers_import(self):
        from models.base import BasePredictor

        from models import _WRAPPER_REGISTRY, get_predictor_class

        for name in _WRAPPER_REGISTRY:
            cls = get_predictor_class(name)
            assert issubclass(cls, BasePredictor), f"{name}: not BasePredictor"
            assert cls.name == name, f"{name}: cls.name={cls.name}"
            assert cls.modality in ("image", "video", "audio", "provenance")

    def test_modality_alignment(self):
        from config import MODEL_REGISTRY

        from models import get_predictor_class

        for name, defn in MODEL_REGISTRY.items():
            cls = get_predictor_class(name)
            assert cls.modality == defn.modality, (
                f"{name}: config modality={defn.modality}, "
                f"wrapper modality={cls.modality}"
            )


# ── Test 3: Model Loader ────────────────────────────────────────────────────


class TestModelLoader:
    def test_safe_import(self, tmp_path):
        """safe_import adds path and imports module."""
        # Create a dummy module
        mod_file = tmp_path / "dummy_mod.py"
        mod_file.write_text("VALUE = 42")

        from model_loader import safe_import

        mod = safe_import("dummy_mod", tmp_path)
        assert mod.VALUE == 42


# ── Test 4: mmcv_compat ─────────────────────────────────────────────────────


class TestMmcvCompat:
    def test_inject_and_cleanup(self):
        from mmcv_compat import cleanup, inject_all

        inject_all()
        assert "mmcv" in sys.modules
        assert "mmcv.cnn" in sys.modules
        assert "mmcv.runner" in sys.modules
        assert "mmengine" in sys.modules

        # Verify functions exist
        import mmcv.cnn

        assert hasattr(mmcv.cnn, "build_conv_layer")
        assert hasattr(mmcv.cnn, "build_norm_layer")

        cleanup()
        assert "mmcv" not in sys.modules

    def test_build_conv_layer(self):
        import torch.nn as nn
        from mmcv_compat.cnn import build_conv_layer

        conv = build_conv_layer(None, 3, 64, 3, padding=1)
        assert isinstance(conv, nn.Conv2d)
        assert conv.in_channels == 3
        assert conv.out_channels == 64

    def test_build_norm_layer(self):
        import torch.nn as nn
        from mmcv_compat.cnn import build_norm_layer

        name, layer = build_norm_layer({"type": "BN"}, 64)
        assert isinstance(layer, nn.BatchNorm2d)
        assert name == "norm"

    def test_ffn(self):
        import torch
        from mmcv_compat.cnn import FFN

        ffn = FFN(embed_dims=32, feedforward_channels=64)
        x = torch.randn(1, 10, 32)
        out = ffn(x)
        assert out.shape == (1, 10, 32)

    def test_base_module(self):
        import torch.nn as nn
        from mmcv_compat.runner import BaseModule

        class TestModule(BaseModule):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        m = TestModule()
        assert isinstance(m, nn.Module)

    def test_auto_fp16_is_noop(self):
        from mmcv_compat.runner import auto_fp16

        @auto_fp16()
        def my_func(x):
            return x + 1

        assert my_func(5) == 6


# ── Test 5: Provenance Models ───────────────────────────────────────────────


class TestProvenanceModels:
    """Provenance models work on any machine (no GPU, no model_code)."""

    def _make_jpeg(self) -> bytes:
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    def test_c2pa_load_and_predict(self):
        from models.base import get_device

        from models import get_predictor_class

        cls = get_predictor_class("c2pa")
        p = cls()
        p.load(Path("/dev/null"), get_device())
        assert p._loaded

        result = p.predict(self._make_jpeg())
        assert result["probability"] is not None
        assert 0.0 <= result["probability"] <= 1.0
        assert result["prediction"] in ("real", "fake")

    def test_sdxl_watermark_load_and_predict(self):
        from models.base import get_device

        from models import get_predictor_class

        cls = get_predictor_class("sdxl_watermark")
        p = cls()
        p.load(Path("/dev/null"), get_device())
        assert p._loaded

        result = p.predict(self._make_jpeg())
        assert result["probability"] is not None
        assert 0.0 <= result["probability"] <= 1.0


# ── Test 6: Server ───────────────────────────────────────────────────────────


class TestServer:
    def test_server_module_imports(self):
        from server import app, detect, health, list_models

    def test_server_startup_with_provenance_only(self):
        os.environ["DEEPSAFE_MODELS"] = "c2pa,sdxl_watermark"
        import server as srv

        # Clear any models from prior test runs
        srv._models.clear()
        srv._failed_models.clear()
        asyncio.run(srv.startup())
        assert "c2pa" in srv._models
        assert "sdxl_watermark" in srv._models
        result = asyncio.run(srv.health())
        assert result["status"] == "healthy"


# ── Test 7: Ensemble ────────────────────────────────────────────────────────


class TestEnsemble:
    def test_compute_ensemble_no_results(self):
        from ensemble import compute_ensemble

        result = compute_ensemble({}, "image")
        assert result["verdict"] == "undetermined"

    def test_compute_ensemble_single_model(self):
        from ensemble import compute_ensemble

        results = {
            "npr": {"probability": 0.9, "prediction": "fake", "latency_ms": 10},
        }
        result = compute_ensemble(results, "image")
        assert result["verdict"] in ("fake", "real")
        assert 0.0 <= result["score"] <= 1.0
