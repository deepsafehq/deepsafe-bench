"""SDXL invisible watermark provenance detector wrapper.

Detects Stable Diffusion XL invisible watermarks embedded in images
using the invisible-watermark library. CPU only -- no PyTorch model.

"""

import io
import logging
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from models.base import BasePredictor
from PIL import Image

logger = logging.getLogger("models.sdxl_watermark")

# SDXL embeds 136 bits (17 bytes). The first 4 bytes are "SDV2".
_WATERMARK_BITS = 136
_SDV2_PREFIX = b"SDV2"
MAX_IMAGE_DIMENSION = 4096


def _hamming_distance(a: bytes, b: bytes) -> int:
    """Count the number of differing bits between two byte sequences.

    Args:
        a: First byte sequence.
        b: Second byte sequence (same length as a).

    Returns:
        Number of differing bits.
    """
    dist = 0
    for x, y in zip(a, b):
        dist += bin(x ^ y).count("1")
    return dist


def _decode_watermark(rgb_array: np.ndarray) -> float:
    """Attempt to decode SDXL watermark from an RGB numpy array.

    Tries two decoding methods (dwtDct, dwtDctSvd) and scores the
    result based on pattern matching.

    Args:
        rgb_array: RGB image as numpy array.

    Returns:
        Probability score: 0.70 for exact SDV2 match, 0.65 for
        fuzzy SDV2 match, 0.5 for no signal.
    """
    from imwatermark import WatermarkDecoder

    methods = ["dwtDct", "dwtDctSvd"]

    for method in methods:
        try:
            decoder = WatermarkDecoder("bytes", _WATERMARK_BITS)
            watermark_bytes = decoder.decode(rgb_array, method)

            if watermark_bytes is None or len(watermark_bytes) == 0:
                continue

            # Exact SDV2 prefix match
            if watermark_bytes[:4] == _SDV2_PREFIX:
                logger.info(
                    "SDXL watermark detected via %s: SDV2 prefix matched",
                    method,
                )
                return 0.70

            # Fuzzy SDV2 match: tolerate up to 2 bit errors in the
            # first 4 bytes (32 bits).
            prefix_hamming = _hamming_distance(watermark_bytes[:4], _SDV2_PREFIX)
            if prefix_hamming <= 2:
                logger.info(
                    "SDXL watermark detected via %s: fuzzy SDV2 match " "(hamming=%d)",
                    method,
                    prefix_hamming,
                )
                return 0.65

        except Exception as exc:
            logger.debug("Watermark decode failed with %s: %s", method, exc)
            continue

    return 0.5


def _detect_watermark(image_bytes: bytes) -> float:
    """Run watermark detection on raw image bytes.

    Args:
        image_bytes: Raw bytes of the image file.

    Returns:
        Float probability in [0, 1]. 0.5 means neutral (no signal).
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.width > MAX_IMAGE_DIMENSION or image.height > MAX_IMAGE_DIMENSION:
            logger.warning(
                "Image too large: %dx%d, max %d",
                image.width,
                image.height,
                MAX_IMAGE_DIMENSION,
            )
            return 0.5
        image = image.convert("RGB")
        rgb_array = np.array(image)
        return _decode_watermark(rgb_array)

    except Exception as exc:
        logger.warning("Watermark detection error: %s", exc)
        return 0.5


class SDXLWatermarkPredictor(BasePredictor):
    """SDXL invisible watermark detector wrapper.

    CPU only. Uses the imwatermark library to decode DWT-DCT
    steganographic watermarks from images.
    """

    name = "sdxl_watermark"
    modality = "provenance"

    def __init__(self):
        pass

    def load(self, weights_dir: Path, device: torch.device) -> None:
        """Store device reference and mark as loaded.

        No model weights to load; imwatermark is used at predict time.
        """
        self._device = device
        self._loaded = True
        logger.info("SDXL watermark detector ready (CPU only)")

    def predict(self, raw_bytes: bytes) -> dict:
        """Detect SDXL invisible watermark in raw image bytes."""
        return self._timed_predict(self._run, raw_bytes)

    def _run(self, raw_bytes: bytes) -> dict:
        """Run watermark detection on raw image bytes."""
        probability = _detect_watermark(raw_bytes)
        return {
            "probability": float(probability),
            "prediction": "fake" if probability > 0.5 else "real",
        }
