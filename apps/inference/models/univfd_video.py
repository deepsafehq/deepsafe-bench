"""UnivFD Video — Universal Fake Detector for video content.

Same architecture as the 'universal' image model (CLIP ViT-L/14 +
fc_weights.pth linear probe), applied frame-by-frame to video with
temporal score aggregation.

Full-frame analysis — no face detection needed, making it effective
for AI-generated videos (Sora, Kling, Veo) where face-dependent
models fail.

Reuses the same fc_weights.pth and CLIP backbone as the image model.

Transform fix (2026-04-11): removed erroneous Resize(224) before
CenterCrop(224).  Original UnivFD trains with CenterCrop-only at
native resolution to capture high-frequency pixel artifacts.
Resize destroyed that signal (AUC=0.469 -> should improve).
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from config import MODELS_ROOT, get_model_code_path, get_weights_path
from model_loader import namespaced_import
from models.base import BasePredictor
from PIL import Image

logger = logging.getLogger("models.univfd_video")

_NUM_FRAMES = 16

_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

_TRANSFORM = transforms.Compose(
    [
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
    ]
)

_EXT_CLIP_PT = MODELS_ROOT / ".ext_cache" / "clip-vit-l14-openclip" / "ViT-L-14.pt"


def _ensure_clip_cache_symlink():
    """Ensure CLIP model is available in ~/.cache/clip/."""
    if not _EXT_CLIP_PT.exists():
        return
    default_cache = Path.home() / ".cache" / "clip"
    default_cache.mkdir(parents=True, exist_ok=True)
    target = default_cache / "ViT-L-14.pt"
    if target.exists() or target.is_symlink():
        return
    try:
        target.symlink_to(_EXT_CLIP_PT)
    except OSError:
        import shutil

        shutil.copy2(str(_EXT_CLIP_PT), str(target))


class UnivFDVideoPredictor(BasePredictor):
    """Universal Fake Detector applied frame-by-frame to video."""

    name = "univfd_video"
    modality = "video"

    def __init__(self):
        self._model = None

    def load(self, weights_dir: Path, device: torch.device) -> None:
        """Load CLIP ViT-L/14 + fc_weights.pth linear probe."""
        self._device = device
        _ensure_clip_cache_symlink()

        # code/ symlinks to image/universal/code (same CLIP+fc arch)
        code_path = get_model_code_path("univfd_video")
        models_mod = namespaced_import(
            "models",
            code_path,
            namespace="_ds_univfd_vid",
        )
        get_model = models_mod.get_model
        self._model = get_model("CLIP:ViT-L/14")

        # Load the pre-trained fc layer weights
        fc_weights = get_weights_path("univfd_video") / "fc_weights.pth"
        if not fc_weights.exists():
            raise FileNotFoundError(f"UnivFD fc_weights.pth not found at {fc_weights}")

        state_dict = torch.load(
            fc_weights,
            map_location="cpu",
            weights_only=False,
        )
        self._model.fc.load_state_dict(state_dict, strict=True)
        self._model.to(device)
        self._model.train(mode=False)
        self._model = self._optimize_for_inference(
            self._model,
            use_compile=False,
        )
        self._loaded = True
        logger.info("UnivFD-Video model loaded on %s", device)

    def predict(self, raw_bytes: bytes) -> dict:
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
        """Run UnivFD on each frame and aggregate."""
        if not frames:
            return {"probability": 0.5, "prediction": "real"}

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
