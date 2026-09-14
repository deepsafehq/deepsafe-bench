"""FakeSTormer (ICCV 2025) video deepfake detector wrapper.

Swin Transformer backbone with temporal vulnerability modelling.
Extracts frames from a video, resizes them, and runs the temporal
classification model to produce a fake probability via sigmoid.

CRITICAL: Must inject mmcv_compat before importing model_code.

Requires isolation: model_code uses generic top-level module names
(``models``, ``configs``, ``package_utils``) that conflict.

FP16/AMP disabled: model code has hardcoded ``dtype=torch.float``
tensors in CrossAttention.forward() that crash with FP16 weights.
Must run in FP32 only.

"""

import logging
import os
import tempfile
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from config import get_model_code_path, get_model_dir
from model_loader import namespaced_import
from models.base import BasePredictor
from PIL import Image

logger = logging.getLogger("models.fakestormer")

# Default number of frames to sample from video
_NUM_FRAMES = 4


class FakeSTormerPredictor(BasePredictor):
    """FakeSTormer video deepfake detector (needs_isolation=True)."""

    name = "fakestormer"
    modality = "video"
    _use_amp = False  # model code has hardcoded FP32 tensors

    def __init__(self):
        self._model = None
        self._cfg = None
        self._transforms = None
        self._code_path = None  # saved for runtime sys.path fix

    def load(self, weights_dir: Path, device: torch.device) -> None:
        """Load FakeSTormer model with mmcv_compat injection."""
        self._device = device
        code_path = get_model_code_path("fakestormer")
        model_path = get_model_dir("fakestormer")

        # CRITICAL: inject mmcv/mmengine stubs before importing model_code
        from mmcv_compat import inject_all

        inject_all()

        # Import model_code modules with namespace isolation
        config_mod = namespaced_import(
            "configs.get_config", code_path, namespace="_ds_fakestormer"
        )
        models_mod = namespaced_import("models", code_path, namespace="_ds_fakestormer")
        transform_mod = namespaced_import(
            "package_utils.transform",
            code_path,
            namespace="_ds_fakestormer",
        )

        # Config path relative to service directory
        config_path = str(
            code_path / "configs" / "temporal" / "FakeSFormer_base_c23.yaml"
        )
        self._cfg = config_mod.load_config(config_path)

        # Build model
        self._model = models_mod.build_model(
            self._cfg.MODEL,
            models_mod.MODELS,
        ).to(torch.float)

        # Locate weights — check both weights/ and code/weights/
        from config import get_weights_path as _gwp

        _ckpt_name = (
            "TopDownDetector_C23_ViTBase224_ST_hm100_"
            "tempLOC0.2_4SBI_SAM_mp0.01_temp2_0vidAug_"
            "0.35_temp_normlms_normREAL_model_best.pth"
        )
        weights_file = _gwp("fakestormer") / _ckpt_name
        if not weights_file.exists():
            weights_file = code_path / "weights" / _ckpt_name
        if not weights_file.exists():
            raise FileNotFoundError(
                f"FakeSTormer weights not found in weights/ or code/weights/"
            )

        self._model = models_mod.load_pretrained(
            self._model,
            str(weights_file),
        )
        self._model = self._model.to(device)
        self._model.train(mode=False)
        # Skip _optimize_for_inference (no FP16, no torch.compile):
        # model code has hardcoded ``dtype=torch.float`` and ``.cuda()``
        # in CrossAttention that crash with FP16 weights or compiled
        # graphs.  AMP is also disabled via _use_amp = False above.
        logger.info("fakestormer: skipping FP16/compile (incompatible model code)")

        # Setup transforms
        self._transforms = transform_mod.final_transform(
            self._cfg.DATASET,
        )

        self._code_path = str(code_path)
        self._loaded = True
        logger.info("FakeSTormer model loaded on %s", device)

    def predict(self, raw_bytes: bytes) -> dict:
        """Run inference on raw video bytes."""
        return self._timed_predict(self._run, raw_bytes)

    @torch.inference_mode()
    def _run(self, raw_bytes: bytes) -> dict:
        # Ensure FakeSTormer's code_path is on sys.path so the
        # model's internal imports (models.networks) resolve
        # correctly even when other models pollute sys.modules.
        import sys

        if self._code_path and self._code_path not in sys.path:
            sys.path.insert(0, self._code_path)

        with tempfile.NamedTemporaryFile(
            suffix=".mp4",
            delete=False,
        ) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        try:
            num_frames = (
                self._cfg.DATASET.DATA.SAMPLES_PER_VIDEO.NUM_FRAMES or _NUM_FRAMES
            )
            frames = _extract_frames(tmp_path, num_frames)
            if not frames:
                return {
                    "probability": 0.5,
                    "prediction": "real",
                }

            image_size = self._cfg.DATASET.IMAGE_SIZE  # [224, 224]
            transformed = []
            for frame in frames:
                resized = frame.resize(
                    (int(image_size[0]), int(image_size[1])),
                )
                arr = np.array(resized) / 255.0
                tensor = self._transforms(arr).to(torch.float)
                transformed.append(tensor.unsqueeze(0))

            # Stack: [T, C, H, W] -> [1, T, C, H, W] -> [1, C, T, H, W]
            input_tensor = torch.cat(transformed, 0)
            input_tensor = input_tensor.to(self._device)
            input_tensor = input_tensor.unsqueeze(0)
            input_tensor = input_tensor.transpose(1, 2)

            outputs = self._model(input_tensor)
            if isinstance(outputs, list):
                outputs = outputs[0]

            probability = outputs["cls"].sigmoid().cpu().item()

            return {
                "probability": float(probability),
                "prediction": "fake" if probability >= 0.5 else "real",
            }
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


def _extract_frames(
    video_path: str,
    num_frames: int = _NUM_FRAMES,
) -> List[Image.Image]:
    """Extract uniformly-sampled frames from a video file.

    Crops 15 pixels from each side, matching the service's test.py.

    Args:
        video_path: Path to the video on disk.
        num_frames: Number of frames to extract.

    Returns:
        List of PIL Image objects.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            if h > 30 and w > 30:
                frame = frame[15 : h - 15, 15 : w - 15]
            frames.append(Image.fromarray(frame))

    cap.release()

    # Pad if not enough frames
    while len(frames) < num_frames and len(frames) > 0:
        frames.append(frames[-1])

    return frames
