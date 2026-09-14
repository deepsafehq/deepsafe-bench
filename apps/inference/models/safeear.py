"""SafeEar (CCS 2024) audio deepfake detector wrapper.

Two-stage pipeline:
  1. SpeechTokenizer (neural audio codec) decouples acoustic features
  2. SafeEar1s (transformer classifier) detects spoofing from tokens

Uses Monte Carlo averaging (5 passes) for stable predictions and
temperature-scaled softmax (T=5.0) on the averaged logits.

"""

import io
import logging
from pathlib import Path

import librosa
import numpy as np
import torch
from config import get_model_dir, get_weights_path
from model_loader import safe_import
from models.base import BasePredictor

logger = logging.getLogger("models.safeear")

SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 64600  # ~4 s at 16 kHz (ASVspoof standard)
SOFTMAX_TEMPERATURE = 5.0
NUM_INFERENCE_PASSES = 5


class SafeEarPredictor(BasePredictor):
    """SafeEar deepfake detector wrapper."""

    name = "safeear"
    modality = "audio"

    def __init__(self):
        self._decouple_model = None
        self._detect_model = None

    def load(self, weights_dir: Path, device: torch.device) -> None:
        """Load SpeechTokenizer and SafeEar1s from weights."""
        self._device = device

        model_path = get_model_dir("safeear")
        repo_path = model_path / "code"

        # Import from the SafeEar code package
        safeear_decouple = safe_import(
            "safeear.models.decouple",
            repo_path,
        )
        safeear_model = safe_import(
            "safeear.models.safeear",
            repo_path,
        )
        SpeechTokenizer = safeear_decouple.SpeechTokenizer
        SafeEar1s = safeear_model.SafeEar1s
        SE_Rawformer_front = safeear_model.SE_Rawformer_front

        weights_path = get_weights_path("safeear")

        # --- SpeechTokenizer (decouple model) ---
        self._decouple_model = SpeechTokenizer(
            n_filters=64,
            strides=[8, 5, 4, 2],
            dimension=1024,
            semantic_dimension=768,
            bidirectional=True,
            dilation_base=2,
            residual_kernel_size=3,
            n_residual_layers=1,
            lstm_layers=2,
            activation="ELU",
            codebook_size=1024,
            n_q=8,
            sample_rate=16000,
        )
        st_state = torch.load(
            weights_path / "SpeechTokenizer.pt",
            map_location="cpu",
        )
        self._decouple_model.load_state_dict(st_state)
        self._decouple_model.to(device)
        self._decouple_model.train(mode=False)

        # --- SafeEar1s (detect model) from Lightning checkpoint ---
        self._detect_model = SafeEar1s(
            front=SE_Rawformer_front(),
            embedding_dim=1024,
            dropout_rate=0.1,
            attention_dropout=0.1,
            stochastic_depth=0.1,
            num_layers=2,
            num_heads=8,
            num_classes=2,
            positional_embedding="sine",
            mlp_ratio=1.0,
        )

        ckpt = torch.load(
            weights_path / "model.ckpt",
            map_location="cpu",
        )
        state_dict = ckpt.get("state_dict", ckpt)

        # Lightning prefixes keys with "detect_model."
        detect_state = {}
        for k, v in state_dict.items():
            if k.startswith("detect_model."):
                detect_state[k.replace("detect_model.", "", 1)] = v

        self._detect_model.load_state_dict(detect_state)
        self._detect_model.to(device)
        self._detect_model.train(mode=False)

        self._loaded = True
        logger.info("SafeEar models loaded on %s", device)

    def predict(self, raw_bytes: bytes) -> dict:
        """Run inference on raw audio bytes."""
        return self._timed_predict(self._run, raw_bytes)

    @torch.inference_mode()
    def _run(self, raw_bytes: bytes) -> dict:
        waveform, _ = librosa.load(
            io.BytesIO(raw_bytes),
            sr=SAMPLE_RATE,
            mono=True,
        )

        # Zero-pad or truncate
        if len(waveform) < MAX_AUDIO_LENGTH:
            waveform = np.pad(
                waveform,
                (0, MAX_AUDIO_LENGTH - len(waveform)),
            )
        else:
            waveform = waveform[:MAX_AUDIO_LENGTH]

        # Shape: (1, 1, samples) -- batch, channels, time
        x_wav = torch.FloatTensor(waveform).unsqueeze(0).unsqueeze(0).to(self._device)

        # Step 1: Acoustic tokens via SpeechTokenizer
        _, _, _, acoustic_tokens = self._decouple_model(
            x_wav,
            layers=[0, 1, 2, 3, 4, 5, 6, 7],
        )

        # Step 2: Monte Carlo averaging for stable predictions
        logit_sum = torch.zeros(1, 2, device=self._device)
        for _ in range(NUM_INFERENCE_PASSES):
            raw_logits, _ = self._detect_model(acoustic_tokens)
            logit_sum += raw_logits
        avg_logits = logit_sum / NUM_INFERENCE_PASSES

        # Step 3: Temperature-scaled softmax
        probs = torch.softmax(
            avg_logits / SOFTMAX_TEMPERATURE,
            dim=-1,
        )
        probability = probs[0, 1].item()

        return {
            "probability": float(probability),
            "prediction": "fake" if probability >= 0.5 else "real",
        }
