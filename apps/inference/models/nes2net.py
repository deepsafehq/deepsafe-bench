"""Nes2Net (IEEE T-IFS 2025) audio deepfake detector wrapper.

XLS-R 300M frontend + Nested Res2Net-TDNN backend for cross-language
synthetic speech detection.  Outputs 2 logits [spoof, bonafide];
softmax index 0 is the fake probability.

Like ShiftySpeech, requires the omegaconf monkey-patch for fairseq.

"""

import argparse
import io
import logging
import warnings
from pathlib import Path

import fairseq_compat  # noqa: F401 — must patch before fairseq import
import librosa
import numpy as np
import torch
from config import get_model_code_path, get_model_dir, get_weights_path
from model_loader import namespaced_import
from models.base import BasePredictor

logger = logging.getLogger("models.nes2net")

SAMPLE_RATE = 16000
TARGET_SAMPLES = 64600  # ~4.04 s at 16 kHz


class Nes2NetPredictor(BasePredictor):
    """Nes2Net (XLSR + Nested Res2Net TDNN) wrapper."""

    name = "nes2net"
    modality = "audio"

    def __init__(self):
        self._model = None

    def load(self, weights_dir: Path, device: torch.device) -> None:
        """Load Nes2Net model with fine-tuned weights."""
        self._device = device

        # Suppress fairseq/omegaconf deprecation noise
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Monkey-patch omegaconf before fairseq is imported
        import omegaconf._utils as _omegaconf_utils

        if not hasattr(_omegaconf_utils, "is_primitive_type"):
            _omegaconf_utils.is_primitive_type = lambda t: t in (
                int,
                float,
                bool,
                str,
                bytes,
            )

        model_path = get_model_dir("nes2net")
        code_path = get_model_code_path("nes2net")
        model_mod = namespaced_import(
            "wav2vec2_Nes2Net_X",
            code_path,
            namespace="_ds_nes2net",
        )
        Nes2NetModel = model_mod.wav2vec2_Nes2Net_no_Res_w_allT

        args = argparse.Namespace(
            n_output_logits=2,
            dilation=2,
            pool_func="mean",
            SE_ratio=[1],
            Nes_ratio=[8, 8],
        )
        self._model = Nes2NetModel(args, str(device))

        weights_file = get_weights_path("nes2net") / "nes2net_itw_valaug.pt"
        state_dict = torch.load(
            weights_file,
            map_location=device,
            weights_only=False,
        )
        self._model.load_state_dict(state_dict)
        self._model.to(device)
        self._model.train(mode=False)
        self._loaded = True
        logger.info("Nes2Net model loaded on %s", device)

    def predict(self, raw_bytes: bytes) -> dict:
        """Run inference on raw audio bytes."""
        return self._timed_predict(self._run, raw_bytes)

    @torch.inference_mode()
    def _run(self, raw_bytes: bytes) -> dict:
        audio, _ = librosa.load(
            io.BytesIO(raw_bytes),
            sr=SAMPLE_RATE,
            mono=True,
        )

        # Pad/trim via tiling (matches training preprocessing)
        if len(audio) >= TARGET_SAMPLES:
            audio = audio[:TARGET_SAMPLES]
        else:
            num_repeats = TARGET_SAMPLES // len(audio) + 1
            audio = np.tile(audio, num_repeats)[:TARGET_SAMPLES]

        tensor = torch.FloatTensor(audio).unsqueeze(0).to(self._device)

        output = self._model(tensor)
        # output: [batch, 2] -- index 0 = spoof, index 1 = bonafide
        probs = torch.softmax(output, dim=1)
        probability = probs[0, 0].item()

        return {
            "probability": float(probability),
            "prediction": "fake" if probability >= 0.5 else "real",
        }
