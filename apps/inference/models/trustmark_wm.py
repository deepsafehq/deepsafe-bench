"""TrustMark invisible watermark provenance detector wrapper.

Detects Adobe/CAI TrustMark watermarks embedded in images using
the trustmark library. TrustMark watermarks survive JPEG
compression, resizing, cropping, and social media re-encoding,
making them the only provenance signal that persists through
content re-sharing.

Detection is binary: the BCH error-correcting code either
validates a watermark payload (detected=True) or fails
(detected=False). False positive rate is extremely low because
random bits virtually never pass BCH syndrome checks.

TrustMark presence does NOT confirm AI generation -- real photos
from Photoshop, Samsung Galaxy, and Google Pixel also embed
TrustMark. The score (0.65) reflects this ambiguity.
"""

import io
import logging
from pathlib import Path

import torch
from models.base import BasePredictor
from PIL import Image

logger = logging.getLogger("models.trustmark")

# Lazy import to avoid loading TrustMark at module level
TrustMark = None


def _map_detection_to_probability(detected: bool) -> float:
    """Map TrustMark detection result to probability score.

    Args:
        detected: True if BCH ECC validated a watermark payload.

    Returns:
        0.65 if detected (moderate provenance signal),
        0.50 if not detected (neutral).
    """
    if detected:
        return 0.65
    return 0.50


class TrustMarkPredictor(BasePredictor):
    """TrustMark watermark detector wrapper.

    Loads the TrustMark Q-variant decoder (~45 MB) and checks
    images for embedded watermarks. CPU by default.
    """

    name = "trustmark"
    modality = "provenance"
    _use_amp = False  # CPU model, AMP not needed

    def __init__(self):
        self._tm = None

    def load(self, weights_dir: Path, device: torch.device) -> None:
        """Load TrustMark decoder model."""
        self._device = device

        global TrustMark
        if TrustMark is None:
            from trustmark import TrustMark as _TM

            TrustMark = _TM

        self._tm = TrustMark(
            verbose=False,
            model_type="Q",
            loadRemover=False,
            loadBBoxDetector=False,
        )
        self._loaded = True
        logger.info("TrustMark decoder loaded (Q variant, CPU)")

    def predict(self, raw_bytes: bytes) -> dict:
        """Detect TrustMark watermark in raw image bytes."""
        return self._timed_predict(self._run, raw_bytes)

    def _run(self, raw_bytes: bytes) -> dict:
        """Run TrustMark detection on raw image bytes."""
        if not raw_bytes:
            return {
                "probability": 0.50,
                "prediction": "real",
            }

        try:
            image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        except Exception as exc:
            logger.debug("TrustMark: failed to open image: %s", exc)
            return {
                "probability": 0.50,
                "prediction": "real",
            }

        try:
            _secret, detected, _schema = self._tm.decode(image, MODE="binary")
        except Exception as exc:
            logger.debug("TrustMark: decode failed: %s", exc)
            return {
                "probability": 0.50,
                "prediction": "real",
            }

        probability = _map_detection_to_probability(detected)

        if detected:
            logger.info("TrustMark watermark DETECTED (schema=%d)", _schema)

        return {
            "probability": float(probability),
            "prediction": "fake" if probability > 0.5 else "real",
        }
