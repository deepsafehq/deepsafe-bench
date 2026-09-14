"""AudioSeal watermark provenance detector wrapper.

Detects Meta AudioSeal watermarks embedded in audio files using the
audioseal_detector_16bits model. GPU-capable.

"""

import logging
import os
import tempfile
from pathlib import Path

import torch
import torchaudio
from config import MODELS_ROOT
from models.base import BasePredictor

logger = logging.getLogger("models.audioseal")

DETECTION_THRESHOLD = 0.3
TARGET_SAMPLE_RATE = 16000


def _map_confidence_to_probability(confidence: float) -> float:
    """Map AudioSeal detection confidence to a probability score.

    Confidence values below the detection threshold are treated as
    noise and mapped to 0.5 (neutral). Values at or above the
    threshold are linearly scaled into [0.5, 1.0].

    Args:
        confidence: Raw detection confidence from AudioSeal in [0, 1].

    Returns:
        Probability in [0.5, 1.0]. 0.5 means neutral (no watermark).
    """
    if confidence < DETECTION_THRESHOLD:
        return 0.5
    return 0.5 + confidence * 0.5


class AudioSealPredictor(BasePredictor):
    """AudioSeal watermark detector wrapper.

    Loads Meta's audioseal_detector_16bits model and runs watermark
    detection on audio files.
    """

    name = "audioseal"
    modality = "provenance"

    def __init__(self):
        self._detector = None

    def load(self, weights_dir: Path, device: torch.device) -> None:
        """Load AudioSeal detector model onto the given device."""
        self._device = device

        # Set AUDIOSEAL_CACHE_DIR so audioseal's loader reads from
        # .ext_cache/audioseal/ instead of ~/.cache/audioseal/ at runtime.
        # The audioseal library appends "/audioseal/" to this value, so we
        # point to the parent dir (.ext_cache/).
        os.environ["AUDIOSEAL_CACHE_DIR"] = str(MODELS_ROOT / ".ext_cache")

        from audioseal import AudioSeal

        self._detector = AudioSeal.load_detector("audioseal_detector_16bits")
        self._detector = self._detector.to(device)
        self._detector.train(mode=False)
        self._loaded = True
        logger.info("AudioSeal detector loaded on %s", device)

    def predict(self, raw_bytes: bytes) -> dict:
        """Detect AudioSeal watermark in raw audio bytes."""
        return self._timed_predict(self._run, raw_bytes)

    @torch.inference_mode()
    def _run(self, raw_bytes: bytes) -> dict:
        """Run AudioSeal detection on raw audio bytes.

        Writes bytes to a temp file, loads with torchaudio (falling
        back to soundfile), resamples to 16 kHz mono, and runs
        detection.
        """
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(raw_bytes)
                tmp_path = tmp.name

            # Try soundfile first (most reliable across formats),
            # fall back to torchaudio if soundfile fails.
            try:
                import soundfile as sf

                data, sample_rate = sf.read(tmp_path, dtype="float32")
                waveform = torch.from_numpy(data).T
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
            except Exception:
                waveform, sample_rate = torchaudio.load(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        # Stereo to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to target sample rate
        if sample_rate != TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=TARGET_SAMPLE_RATE,
            )
            waveform = resampler(waveform)

        # Add batch dimension: [1, channels, samples]
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        waveform = waveform.to(self._device)

        result, message = self._detector.detect_watermark(
            waveform, sample_rate=TARGET_SAMPLE_RATE
        )

        confidence = float(result)
        probability = _map_confidence_to_probability(confidence)

        return {
            "probability": float(probability),
            "prediction": "fake" if probability > 0.5 else "real",
        }
