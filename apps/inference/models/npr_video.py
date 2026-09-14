"""NPR Video — frame-by-frame NPR for video deepfake detection.

Applies the NPR (Neural Processing Residual) image detector to
uniformly-sampled video frames and aggregates scores.  Works on
full frames without face detection, making it effective for
AI-generated videos (Sora, Kling, Veo) where face-dependent
models output ~0.5.

Reuses the same NPR ResNet50 weights as the image model.
"""

import io
import logging
import os
import tempfile
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from config import get_model_code_path, get_weights_path
from model_loader import namespaced_import
from models.base import BasePredictor
from PIL import Image

logger = logging.getLogger("models.npr_video")

_NUM_FRAMES = 16
_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


class NPRVideoPredictor(BasePredictor):
    """NPR applied frame-by-frame to video content."""

    name = "npr_video"
    modality = "video"
    _use_amp = False

    def __init__(self):
        self._model = None

    def load(self, weights_dir: Path, device: torch.device) -> None:
        """Load ResNet50 with NPR weights (same as image NPR)."""
        self._device = device

        # code/ symlinks to image/npr/code (same ResNet50 architecture)
        code_path = get_model_code_path("npr_video")
        resnet_mod = namespaced_import(
            "networks.resnet",
            code_path,
            namespace="_ds_npr_vid",
        )
        self._model = resnet_mod.resnet50(num_classes=1)

        weights_file = get_weights_path("npr_video") / "NPR.pth"
        state_dict = torch.load(
            weights_file,
            map_location=device,
            weights_only=True,
        )
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        self._model.load_state_dict(state_dict)
        self._model.to(device)
        self._model.train(mode=False)
        self._loaded = True
        logger.info("NPR-Video model loaded on %s", device)

    def predict(self, raw_bytes: bytes) -> dict:
        """Run inference on raw video bytes."""
        return self._timed_predict(self._run, raw_bytes)

    def predict_preprocessed(self, video_data) -> dict:
        """Run on shared preprocessed video frames."""
        return self._timed_predict(
            lambda _: self._run_on_frames(video_data.frames_8),
            b"",
        )

    @torch.inference_mode()
    def _run(self, raw_bytes: bytes) -> dict:
        with tempfile.NamedTemporaryFile(
            suffix=".mp4",
            delete=False,
        ) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        try:
            frames = _extract_frames(tmp_path, _NUM_FRAMES)
            return self._run_on_frames(frames)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @torch.inference_mode()
    def _run_on_frames(self, frames: List[np.ndarray]) -> dict:
        """Run NPR on each frame and aggregate."""
        if not frames:
            return {"probability": 0.5, "prediction": "real"}

        scores = []
        # Batch frames for efficient GPU utilization
        batch_tensors = []
        for frame in frames:
            pil_img = Image.fromarray(frame)
            batch_tensors.append(_TRANSFORM(pil_img))

        batch = torch.stack(batch_tensors).to(self._device)
        logits = self._model(batch)
        probs = torch.sigmoid(logits).flatten().cpu().tolist()

        probability = float(np.mean(probs))
        return {
            "probability": probability,
            "prediction": "fake" if probability >= 0.5 else "real",
        }


def _extract_frames(
    video_path: str,
    num_frames: int = _NUM_FRAMES,
) -> List[np.ndarray]:
    """Uniformly sample RGB frames from a video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(
        0,
        total - 1,
        num_frames,
        endpoint=True,
        dtype=int,
    )
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames
