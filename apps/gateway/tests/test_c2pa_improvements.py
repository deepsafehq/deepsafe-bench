"""Tests for C2PA provenance detector improvements.

Covers: file extension detection from magic bytes, composite content
handling, expanded generator string list, IPTC metadata extraction.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# c2pa.py lives in apps/inference, add to path for direct import in tests
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "apps/inference"))

from models.c2pa import (
    _AI_GENERATORS,
    _check_c2pa,
    _detect_extension,
    _extract_provenance_metadata,
)


class TestDetectExtension:
    def test_jpeg_magic_bytes(self):
        assert _detect_extension(b"\xff\xd8\xff\xe0" + b"\x00" * 100) == ".jpg"

    def test_png_magic_bytes(self):
        assert _detect_extension(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100) == ".png"

    def test_webp_magic_bytes(self):
        data = b"RIFF" + b"\x00" * 4 + b"WEBP" + b"\x00" * 100
        assert _detect_extension(data) == ".webp"

    def test_mp4_magic_bytes(self):
        data = b"\x00\x00\x00\x1c" + b"ftyp" + b"\x00" * 100
        assert _detect_extension(data) == ".mp4"

    def test_wav_magic_bytes(self):
        data = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 100
        assert _detect_extension(data) == ".wav"

    def test_flac_magic_bytes(self):
        assert _detect_extension(b"fLaC" + b"\x00" * 100) == ".flac"

    def test_mp3_id3_magic_bytes(self):
        assert _detect_extension(b"ID3" + b"\x00" * 100) == ".mp3"

    def test_mp3_sync_magic_bytes(self):
        assert _detect_extension(b"\xff\xfb" + b"\x00" * 100) == ".mp3"

    def test_unknown_format(self):
        assert _detect_extension(b"\x00\x01\x02\x03" * 25) == ".bin"

    def test_empty_bytes(self):
        assert _detect_extension(b"") == ".bin"

    def test_short_bytes(self):
        assert _detect_extension(b"\xff\xd8") == ".bin"


class TestCompositeC2PA:
    """Test compositeWithTrainedAlgorithmicMedia handling."""

    def _setup_c2pa_mock(self, manifest_json, is_valid=True):
        """Install a mock c2pa module in sys.modules."""
        reader = MagicMock()
        reader.json.return_value = manifest_json
        reader.is_valid = is_valid
        mock_c2pa = MagicMock()
        mock_c2pa.Reader.try_create.return_value = reader
        sys.modules["c2pa"] = mock_c2pa
        return mock_c2pa

    def teardown_method(self):
        sys.modules.pop("c2pa", None)

    def test_composite_returns_070_with_valid_sig(self):
        manifest = (
            '{"assertions": [{"data": {"digitalSourceType": '
            '"http://cv.iptc.org/newscodes/digitalsourcetype/'
            'compositeWithTrainedAlgorithmicMedia"}}]}'
        )
        self._setup_c2pa_mock(manifest)
        score, valid = _check_c2pa("/tmp/test.jpg")
        assert score == 0.70
        assert valid is True

    def test_composite_returns_055_with_invalid_sig(self):
        manifest = (
            '{"assertions": [{"data": {"digitalSourceType": '
            '"compositeWithTrainedAlgorithmicMedia"}}], '
            '"validation_status": [{"code": "signature.mismatch"}]}'
        )
        self._setup_c2pa_mock(manifest, is_valid=False)
        score, valid = _check_c2pa("/tmp/test.jpg")
        assert score == 0.55
        assert valid is False

    def test_pure_ai_still_returns_095(self):
        manifest = (
            '{"assertions": [{"data": {"digitalSourceType": '
            '"trainedAlgorithmicMedia"}}], '
            '"generator": "Adobe Firefly"}'
        )
        self._setup_c2pa_mock(manifest)
        score, valid = _check_c2pa("/tmp/test.jpg")
        assert score == 0.95
        assert valid is True

    def test_composite_checked_before_generator_loop(self):
        """compositeWithTrainedAlgorithmicMedia should return 0.70 even
        if the manifest also contains a known generator string."""
        manifest = (
            '{"generator": "Adobe Firefly", "assertions": [{"data": '
            '{"digitalSourceType": '
            '"compositeWithTrainedAlgorithmicMedia"}}]}'
        )
        self._setup_c2pa_mock(manifest)
        score, valid = _check_c2pa("/tmp/test.jpg")
        assert score == 0.70


class TestExpandedGenerators:
    """Verify all expected generator strings are in _AI_GENERATORS."""

    EXPECTED_NEW = [
        "recraft",
        "luma",
        "pika",
        "minimax",
        "hailuo",
        "jimeng",
        "doubao",
        "nightcafe",
        "playground ai",
        "canva ai",
    ]

    def test_new_generators_present(self):
        for gen in self.EXPECTED_NEW:
            assert gen in _AI_GENERATORS, f"Missing generator: {gen}"

    def test_original_generators_still_present(self):
        originals = [
            "adobe firefly",
            "dall-e",
            "openai",
            "midjourney",
            "stable diffusion",
            "google gemini",
            "runway",
            "flux",
            "veo",
        ]
        for gen in originals:
            assert gen in _AI_GENERATORS, f"Missing original: {gen}"

    def test_total_count_at_least_31(self):
        assert len(_AI_GENERATORS) >= 31


class TestIPTCMetadata:
    """Test IPTC 2025.1 metadata extraction."""

    def _setup_exiftool_mock(self, metadata_dict):
        """Create exiftool mock returning the given metadata."""
        mock_et = MagicMock()
        mock_et.get_metadata.return_value = [metadata_dict]
        mock_module = MagicMock()
        mock_module.ExifToolHelper.return_value.__enter__ = lambda s: mock_et
        mock_module.ExifToolHelper.return_value.__exit__ = MagicMock(return_value=False)
        sys.modules["exiftool"] = mock_module
        return mock_module

    def teardown_method(self):
        sys.modules.pop("exiftool", None)

    def test_extracts_digital_source_type(self):
        self._setup_exiftool_mock({"IPTC:DigitalSourceType": "trainedAlgorithmicMedia"})
        meta = _extract_provenance_metadata("/tmp/test.jpg")
        assert meta["digital_source_type"] == "trainedAlgorithmicMedia"

    def test_extracts_ai_system_used(self):
        self._setup_exiftool_mock({"XMP:AISystemUsed": "Midjourney v6.1"})
        meta = _extract_provenance_metadata("/tmp/test.jpg")
        assert meta["ai_system_used"] == "Midjourney v6.1"

    def test_returns_empty_dict_when_no_metadata(self):
        self._setup_exiftool_mock({})
        meta = _extract_provenance_metadata("/tmp/test.jpg")
        assert meta == {}

    def test_returns_empty_dict_on_exception(self):
        mock_module = MagicMock()
        mock_module.ExifToolHelper.side_effect = Exception("not found")
        sys.modules["exiftool"] = mock_module
        meta = _extract_provenance_metadata("/tmp/test.jpg")
        assert meta == {}

    def test_extracts_description(self):
        self._setup_exiftool_mock(
            {"XMP:Description": "A cat wearing a hat, digital painting style"}
        )
        meta = _extract_provenance_metadata("/tmp/test.jpg")
        assert "description" in meta
        assert "cat wearing a hat" in meta["description"]
