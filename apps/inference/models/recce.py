"""RECCE (CVPR 2022) video deepfake detector wrapper.

Reconstruction-classification face forgery detection using an
Xception encoder with guided attention and graph reasoning.
Trained via DeepfakeBench on FaceForensics++ (c40) with 2-class
(real/fake) output.

Uses MTCNN for face detection, crops and classifies each face at
299x299 per frame, and averages per-frame maximum probabilities.

Requires isolation: model_code uses generic ``model`` module name.

Class convention fix (2026-04-11): DeepfakeBench checkpoint has
class 0 = fake, class 1 = real (opposite of code convention).
Verified empirically on 15.5K-sample eval: softmax[:,1] gives
AUC=0.332 (anti-correlated), softmax[:,0] gives AUC=0.668.

"""

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from config import get_model_code_path, get_weights_path
from facenet_pytorch import MTCNN
from model_loader import namespaced_import
from models.base import BasePredictor
from PIL import Image

logger = logging.getLogger("models.recce")

# RECCE uses 299x299 face crops
_IMAGE_SIZE = (299, 299)
# Number of frames to sample
_NUM_FRAMES = 8  # Reduced from 32 for faster inference
# Face crop margin factor
_MARGIN_FACTOR = 0.5


class RECCEPredictor(BasePredictor):
    """RECCE video deepfake detector (needs_isolation=True)."""

    name = "recce"
    modality = "video"

    def __init__(self):
        self._model = None
        self._face_detector = None

    def load(self, weights_dir: Path, device: torch.device) -> None:
        """Load RECCE model and MTCNN face detector."""
        self._device = device
        code_path = get_model_code_path("recce")

        # MTCNN face detector
        self._face_detector = MTCNN(
            keep_all=True,
            device=device,
            post_process=False,
        )

        # Patch timm's xception to skip pretrained-weight download.
        # We load our own checkpoint; downloading ImageNet weights
        # wastes bandwidth and fails in air-gapped environments.
        import timm.models

        _original_xception = timm.models.xception

        def _xception_no_pretrained(**kwargs):
            kwargs["pretrained"] = False
            return _original_xception(**kwargs)

        timm.models.xception = _xception_no_pretrained

        # Import Recce with namespace isolation (generic ``model`` module)
        network_mod = namespaced_import(
            "model.network", code_path, namespace="_ds_recce"
        )
        Recce = network_mod.Recce

        # Restore timm after import
        timm.models.xception = _original_xception

        # DeepfakeBench trains with 2-class output
        net = Recce(num_classes=2)

        weights_file = get_weights_path("recce") / "recce_checkpoint.pth"
        if not weights_file.exists():
            raise FileNotFoundError(f"RECCE weights not found at {weights_file}")

        checkpoint = torch.load(
            weights_file,
            map_location="cpu",
            weights_only=False,
        )

        # Strip DeepfakeBench "model." prefix if present
        if any(k.startswith("model.") for k in checkpoint.keys()):
            state_dict = {
                k[len("model.") :]: v
                for k, v in checkpoint.items()
                if k.startswith("model.")
            }
        else:
            state_dict = checkpoint

        net.load_state_dict(state_dict)
        net = net.to(device)
        net.train(mode=False)

        self._model = net
        self._model = self._optimize_for_inference(
            self._model,
            use_channels_last=True,
            use_compile=False,
        )
        self._loaded = True
        logger.info("RECCE model loaded on %s", device)

    def predict(self, raw_bytes: bytes) -> dict:
        """Run inference on raw video bytes."""
        return self._timed_predict(self._run, raw_bytes)

    def predict_preprocessed(self, video_data) -> dict:
        """Run inference using shared preprocessed video data."""
        return self._timed_predict(
            lambda _: self._run_on_frames(
                video_data.frames_8,
                video_data.face_boxes_8,
            ),
            b"",
        )

    @torch.inference_mode()
    def _run_on_frames(
        self,
        frames: list,
        face_boxes: list,
    ) -> dict:
        """Core inference on pre-extracted frames and face boxes."""
        if not frames:
            return {"probability": 0.5, "prediction": "real"}

        all_crops: List[np.ndarray] = []
        crop_frame_idx: List[int] = []
        for i, (frame, boxes) in enumerate(zip(frames, face_boxes)):
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                x0, y0, x1, y1 = box.tolist()
                face = _crop_face(frame, (x0, y0, x1, y1))
                if face.size == 0:
                    continue
                all_crops.append(cv2.resize(face, _IMAGE_SIZE))
                crop_frame_idx.append(i)

        if not all_crops:
            return {"probability": 0.5, "prediction": "real"}

        tensors = [_preprocess_face(c) for c in all_crops]
        batch_tensor = torch.stack(tensors).to(self._device)

        logits = self._model(batch_tensor)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        # Class 0 = fake in this DeepfakeBench checkpoint (verified
        # empirically: [:, 1] gives AUC=0.332, [:, 0] gives 0.668).
        probs = F.softmax(logits, dim=1)[:, 0].cpu().tolist()

        frame_max: dict = {}
        for idx, p in zip(crop_frame_idx, probs):
            if idx not in frame_max or p > frame_max[idx]:
                frame_max[idx] = p

        probability = float(np.mean(list(frame_max.values()))) if frame_max else 0.5
        return {
            "probability": probability,
            "prediction": "fake" if probability >= 0.5 else "real",
        }

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
            if not frames:
                return {"probability": 0.5, "prediction": "real"}

            # Batch face detection across all frames at once
            pil_frames = [Image.fromarray(f) for f in frames]
            all_boxes, _ = self._face_detector.detect(pil_frames)

            # Collect all face crops with frame tracking
            all_crops: List[np.ndarray] = []
            crop_frame_idx: List[int] = []
            for i, (frame, boxes) in enumerate(
                zip(frames, all_boxes),
            ):
                if boxes is None or len(boxes) == 0:
                    continue
                for box in boxes:
                    x0, y0, x1, y1 = box.tolist()
                    face = _crop_face(frame, (x0, y0, x1, y1))
                    if face.size == 0:
                        continue
                    all_crops.append(cv2.resize(face, _IMAGE_SIZE))
                    crop_frame_idx.append(i)

            if not all_crops:
                return {"probability": 0.5, "prediction": "real"}

            # Single batched model inference on all crops
            tensors = [_preprocess_face(c) for c in all_crops]
            batch_tensor = torch.stack(tensors).to(self._device)

            logits = self._model(batch_tensor)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            # Class 0 = fake in this DeepfakeBench checkpoint
            probs = F.softmax(logits, dim=1)[:, 0].cpu().tolist()

            # Group by frame and take max per frame
            frame_max: dict = {}
            for idx, p in zip(crop_frame_idx, probs):
                if idx not in frame_max or p > frame_max[idx]:
                    frame_max[idx] = p

            probability = float(np.mean(list(frame_max.values()))) if frame_max else 0.5

            return {
                "probability": probability,
                "prediction": ("fake" if probability >= 0.5 else "real"),
            }
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


def _extract_frames(
    video_path: str,
    num_frames: int = _NUM_FRAMES,
) -> List[np.ndarray]:
    """Uniformly sample RGB frames from a video file.

    Args:
        video_path: Path to the video on disk.
        num_frames: Number of frames to extract.

    Returns:
        List of RGB uint8 numpy arrays (H, W, 3).
    """
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
    frames: List[np.ndarray] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames


def _crop_face(
    img: np.ndarray,
    bbox: Tuple[float, float, float, float],
    margin: float = _MARGIN_FACTOR,
) -> np.ndarray:
    """Crop a face region with a relative margin."""
    h_img, w_img = img.shape[:2]
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0

    x0_new = max(0, int(x0 - w * margin / 2))
    x1_new = min(w_img, int(x1 + w * margin / 2) + 1)
    y0_new = max(0, int(y0 - h * margin / 2))
    y1_new = min(h_img, int(y1 + h * margin / 2) + 1)

    return img[y0_new:y1_new, x0_new:x1_new]


def _preprocess_face(face_crop: np.ndarray) -> torch.Tensor:
    """Apply RECCE-specific preprocessing to a face crop.

    RECCE uses Normalize(mean=[0.5]*3, std=[0.5]*3) which maps
    [0, 255] uint8 to [-1, 1] float32, matching the albumentations
    pipeline in the original inference.py.

    Args:
        face_crop: RGB uint8 array of shape (299, 299, 3).

    Returns:
        Tensor of shape (3, 299, 299) in range [-1, 1].
    """
    tensor = torch.tensor(face_crop).permute(2, 0, 1).float().div(255.0)
    tensor = (tensor - 0.5) / 0.5
    return tensor
