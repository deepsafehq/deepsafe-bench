"""C2PA Content Credentials provenance detector wrapper.

Checks media files for C2PA manifests and EXIF/IPTC metadata indicating
AI generation. CPU only -- no PyTorch model involved.

"""

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Set

import torch
from models.base import BasePredictor

logger = logging.getLogger("models.c2pa")


def _detect_extension(raw_bytes: bytes) -> str:
    """Detect file format from magic bytes and return extension.

    Used to write temp files with the correct extension so the
    c2pa library selects the right parser (JUMBF for JPEG, XMP
    for PNG, ISO BMFF for MP4, etc.).

    Args:
        raw_bytes: First bytes of the file.

    Returns:
        File extension string including the dot (e.g., '.jpg').
    """
    if len(raw_bytes) < 12:
        return ".bin"
    if raw_bytes[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if raw_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WEBP":
        return ".webp"
    if raw_bytes[4:8] == b"ftyp":
        return ".mp4"
    if raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WAVE":
        return ".wav"
    if raw_bytes[:4] == b"fLaC":
        return ".flac"
    if raw_bytes[:3] == b"ID3" or raw_bytes[:2] == b"\xff\xfb":
        return ".mp3"
    return ".bin"


# Known AI generator strings (mirrors the service exactly).
_AI_GENERATORS: Set[str] = {
    # Major generators (original set)
    "adobe firefly",
    "dall-e",
    "dall\u00b7e",
    "openai",
    "midjourney",
    "stable diffusion",
    "stability ai",
    "google gemini",
    "imagen",
    "microsoft designer",
    "copilot",
    "meta ai",
    "leonardo ai",
    "runway",
    "sora",
    "kling",
    "flux",
    "ideogram",
    "veo",
    "google generative ai",
    "trainedalgorithmicmedia",
    # Expanded coverage (2026-04-12)
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
}


def _check_c2pa(file_path: str) -> tuple:
    """Check file for C2PA manifest with AI generator assertions.

    Returns:
        Tuple of (score_or_none, has_valid_c2pa).
    """
    try:
        import c2pa

        reader = c2pa.Reader.try_create(file_path)
        if reader is None:
            return None, False
        manifest_json = reader.json()

        sig_valid = True
        try:
            if hasattr(reader, "is_valid"):
                sig_valid = reader.is_valid
            mdata = json.loads(manifest_json)
            vstatus = mdata.get("validation_status", [])
            if vstatus:
                sig_codes = [v.get("code", "") for v in vstatus]
                has_sig_failure = any(
                    "signature" in c and "untrusted" not in c for c in sig_codes
                )
                if has_sig_failure:
                    sig_valid = False
        except Exception:
            pass

        manifest_lower = manifest_json.lower()

        # Check composite content first -- the substring
        # 'trainedAlgorithmicMedia' appears inside
        # 'compositeWithTrainedAlgorithmicMedia', so check
        # composite before the generator loop to give it a
        # lower score (mixed real + AI content).
        if "compositewithtrainedalgorithmicmedia" in manifest_lower:
            if sig_valid:
                logger.info("C2PA (verified): composite AI content detected.")
                return 0.70, True
            else:
                logger.info(
                    "C2PA (UNVERIFIED): composite AI content -- "
                    "signature invalid, downweighting."
                )
                return 0.55, False

        for generator in _AI_GENERATORS:
            if generator in manifest_lower:
                if sig_valid:
                    logger.info(
                        "C2PA (verified): AI generator '%s' detected.",
                        generator,
                    )
                    return 0.95, True
                else:
                    logger.info(
                        "C2PA (UNVERIFIED): AI generator '%s' -- "
                        "signature invalid, downweighting.",
                        generator,
                    )
                    return 0.60, False

        logger.info("C2PA manifest found but no AI generator detected.")
        return None, sig_valid

    except Exception:
        return None, False


def _check_exif_iptc(
    file_path: str,
    has_valid_c2pa: bool = False,
) -> Optional[float]:
    """Check file EXIF/IPTC metadata for AI generation indicators.

    Returns:
        0.90 if C2PA-backed + DigitalSourceType AI tag,
        0.85 if C2PA-backed + EXIF AI tool signature,
        None otherwise.
    """
    try:
        import exiftool

        with exiftool.ExifToolHelper() as et:
            metadata_list = et.get_metadata(file_path)
            if not metadata_list:
                return None

            metadata = metadata_list[0]

            for key, value in metadata.items():
                key_lower = key.lower()
                if "digitalsourcetype" in key_lower:
                    value_str = str(value).lower()
                    if "trainedalgorithmicmedia" in value_str:
                        if has_valid_c2pa:
                            return 0.90
                        else:
                            return None

            text_fields = []
            for key, value in metadata.items():
                key_lower = key.lower()
                if any(
                    field in key_lower
                    for field in ("software", "description", "usercomment")
                ):
                    text_fields.append(str(value).lower())

            combined_text = " ".join(text_fields)
            for generator in _AI_GENERATORS:
                if generator in combined_text:
                    if has_valid_c2pa:
                        return 0.85
                    else:
                        return None

            return None

    except Exception:
        return None


# IPTC 2025.1 AI-related metadata field mappings.
_IPTC_AI_FIELDS = {
    "digitalsourcetype": "digital_source_type",
    "aisystemused": "ai_system_used",
    "aisystemversionused": "ai_system_version",
    "aipromptinformation": "ai_prompt",
}


def _extract_provenance_metadata(file_path: str) -> dict:
    """Extract IPTC 2025.1 AI metadata fields for display.

    These fields are unsigned and trivially editable. They are
    returned for user-facing display only, never used for scoring.

    Args:
        file_path: Path to the media file.

    Returns:
        Dict of field_name -> value. Empty dict if nothing found.
    """
    try:
        import exiftool

        with exiftool.ExifToolHelper() as et:
            metadata_list = et.get_metadata(file_path)
            if not metadata_list:
                return {}

            metadata = metadata_list[0]
            result = {}

            for key, value in metadata.items():
                key_lower = key.lower().split(":")[-1]
                for iptc_key, out_key in _IPTC_AI_FIELDS.items():
                    if iptc_key in key_lower:
                        result[out_key] = str(value)
                        break

            # Also extract description which may contain prompt info
            for key, value in metadata.items():
                key_lower = key.lower()
                if "description" in key_lower and "ai_prompt" not in result:
                    val_str = str(value)
                    if len(val_str) > 10:
                        result.setdefault("description", val_str[:500])

            return result

    except Exception:
        return {}


def _detect_provenance(file_bytes: bytes) -> float:
    """Run all provenance checks on raw file bytes.

    Writes bytes to a temporary file with .jpg extension (the most
    common modality for C2PA), runs C2PA and EXIF/IPTC checks, and
    returns the maximum signal found.

    Args:
        file_bytes: Raw bytes of the media file.

    Returns:
        Float probability in [0, 1]. 0.5 means neutral (no signal).
    """
    if not file_bytes:
        return 0.5

    tmp_path = None
    try:
        ext = _detect_extension(file_bytes)
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        signals = []

        c2pa_score, has_valid_c2pa = _check_c2pa(tmp_path)
        if c2pa_score is not None:
            signals.append(c2pa_score)

        exif_result = _check_exif_iptc(tmp_path, has_valid_c2pa)
        if exif_result is not None:
            signals.append(exif_result)

        if signals:
            return max(signals)
        return 0.5

    except Exception as exc:
        logger.warning("Provenance detection error: %s", exc)
        return 0.5

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


class C2PAPredictor(BasePredictor):
    """C2PA + EXIF/IPTC provenance checker wrapper.

    CPU only. No model weights to load -- the c2pa and exiftool
    libraries are used directly.
    """

    name = "c2pa"
    modality = "provenance"

    def __init__(self):
        pass

    def load(self, weights_dir: Path, device: torch.device) -> None:
        """Store device reference and mark as loaded.

        No model weights to load; c2pa and exiftool are used at
        predict time.
        """
        self._device = device
        self._loaded = True
        logger.info("C2PA checker ready (CPU only)")

    def predict(self, raw_bytes: bytes) -> dict:
        """Check media for C2PA/EXIF/IPTC AI provenance signals."""
        return self._timed_predict(self._run, raw_bytes)

    def _run(self, raw_bytes: bytes) -> dict:
        """Run provenance detection on raw file bytes."""
        probability = _detect_provenance(raw_bytes)
        result = {
            "probability": float(probability),
            "prediction": "fake" if probability > 0.5 else "real",
        }

        # Extract IPTC metadata for display (best-effort, never
        # affects scoring). Only attempt when a signal was found.
        if probability > 0.5 and raw_bytes:
            try:
                ext = _detect_extension(raw_bytes)
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(raw_bytes)
                    tmp_path = tmp.name
                try:
                    meta = _extract_provenance_metadata(tmp_path)
                    if meta:
                        result["provenance_metadata"] = meta
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
            except Exception:
                pass

        return result
