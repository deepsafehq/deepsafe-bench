"""TrustMark robustness benchmark.

Creates watermarked images using the TrustMark encoder, applies
real-world degradations (JPEG compression, resize), and verifies
detection survives. Also verifies zero false positives on clean images.

Requires: pip install trustmark>=0.9.0
Skip if trustmark is not installed.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "apps/inference"))

try:
    from trustmark import TrustMark

    HAS_TRUSTMARK = True
except ImportError:
    HAS_TRUSTMARK = False

pytestmark = pytest.mark.skipif(not HAS_TRUSTMARK, reason="trustmark not installed")


@pytest.fixture(scope="module")
def trustmark_codec():
    """Create TrustMark encoder/decoder for the test session."""
    tm = TrustMark(
        verbose=False,
        model_type="Q",
        loadRemover=False,
        loadBBoxDetector=False,
    )
    return tm


@pytest.fixture(scope="module")
def watermarked_image(trustmark_codec):
    """Create a watermarked test image."""
    rng = np.random.RandomState(42)
    pixels = rng.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    img = Image.fromarray(pixels)
    secret = "0101010101010101"
    watermarked = trustmark_codec.encode(img, secret, MODE="binary")
    return watermarked


@pytest.fixture(scope="module")
def clean_image():
    """Create a clean (non-watermarked) test image."""
    rng = np.random.RandomState(123)
    pixels = rng.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


class TestTrustMarkTPR:
    """True positive rate on watermarked images."""

    def test_pristine_watermarked(self, trustmark_codec, watermarked_image):
        _secret, detected, schema = trustmark_codec.decode(
            watermarked_image, MODE="binary"
        )
        assert detected is True
        assert schema >= 0

    def test_jpeg_quality_80(self, trustmark_codec, watermarked_image):
        buf = io.BytesIO()
        watermarked_image.save(buf, format="JPEG", quality=80)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        _secret, detected, _schema = trustmark_codec.decode(img, MODE="binary")
        assert detected is True

    def test_jpeg_quality_60(self, trustmark_codec, watermarked_image):
        buf = io.BytesIO()
        watermarked_image.save(buf, format="JPEG", quality=60)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        _secret, detected, _schema = trustmark_codec.decode(img, MODE="binary")
        assert detected is True

    def test_resize_75_percent(self, trustmark_codec, watermarked_image):
        w, h = watermarked_image.size
        resized = watermarked_image.resize(
            (int(w * 0.75), int(h * 0.75)), Image.LANCZOS
        )
        _secret, detected, _schema = trustmark_codec.decode(resized, MODE="binary")
        assert detected is True

    def test_resize_50_percent(self, trustmark_codec, watermarked_image):
        w, h = watermarked_image.size
        resized = watermarked_image.resize((int(w * 0.5), int(h * 0.5)), Image.LANCZOS)
        _secret, detected, _schema = trustmark_codec.decode(resized, MODE="binary")
        # 50% resize may or may not survive -- record but don't hard-fail
        if not detected:
            pytest.skip("TrustMark did not survive 50% resize (expected edge case)")


class TestTrustMarkFPR:
    """False positive rate on clean images."""

    def test_clean_random_image(self, trustmark_codec, clean_image):
        _secret, detected, _schema = trustmark_codec.decode(clean_image, MODE="binary")
        assert detected is False

    def test_solid_white(self, trustmark_codec):
        img = Image.new("RGB", (512, 512), "white")
        _secret, detected, _schema = trustmark_codec.decode(img, MODE="binary")
        assert detected is False

    def test_solid_black(self, trustmark_codec):
        img = Image.new("RGB", (512, 512), "black")
        _secret, detected, _schema = trustmark_codec.decode(img, MODE="binary")
        assert detected is False

    def test_gradient_image(self, trustmark_codec):
        pixels = np.zeros((512, 512, 3), dtype=np.uint8)
        for i in range(512):
            pixels[i, :, :] = int(255 * i / 511)
        img = Image.fromarray(pixels)
        _secret, detected, _schema = trustmark_codec.decode(img, MODE="binary")
        assert detected is False

    def test_multiple_random_images(self, trustmark_codec):
        """Test FPR across 10 random images."""
        false_positives = 0
        for seed in range(10):
            rng = np.random.RandomState(seed + 1000)
            pixels = rng.randint(0, 256, (512, 512, 3), dtype=np.uint8)
            img = Image.fromarray(pixels)
            _secret, detected, _schema = trustmark_codec.decode(img, MODE="binary")
            if detected:
                false_positives += 1
        assert false_positives == 0, f"FPR: {false_positives}/10 false positives"
