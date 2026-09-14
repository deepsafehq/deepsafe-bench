"""SBI (Self-Blended Images, CVPR 2022) video deepfake detector wrapper.

EfficientNet-B4 binary classifier trained on self-blended face
augmentation for robust cross-dataset face-swap detection.  Uses
MTCNN for face detection, crops and classifies each face per frame,
and averages the per-frame maximum fake probabilities.

"""

import logging
import os
import tempfile
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from config import MODELS_ROOT, get_weights_path
from efficientnet_pytorch import EfficientNet
from facenet_pytorch import MTCNN
from models.base import BasePredictor
from PIL import Image
from torch import nn

logger = logging.getLogger("models.sbi")

# SBI uses 380x380 face crops (from configs/sbi/base.json)
_IMAGE_SIZE = (380, 380)
# Number of frames to uniformly sample
_NUM_FRAMES = 8  # Reduced from 32 for faster inference (marginal accuracy loss)
# Face crop margin factor
_MARGIN_FACTOR = 0.5


class _Detector(nn.Module):
    """EfficientNet-B4 binary classifier (real vs fake).

    Mirrors the inference-time Detector from the SBI repository.
    """

    def __init__(self):
        super().__init__()
        # Point TORCH_HOME to .ext_cache/efficientnet-b4/ so
        # model_zoo.load_url finds the pre-downloaded advprop weights
        # without hitting the network at runtime.
        _prev_torch_home = os.environ.get("TORCH_HOME")
        os.environ["TORCH_HOME"] = str(MODELS_ROOT / ".ext_cache" / "efficientnet-b4")
        self.net = EfficientNet.from_pretrained(
            "efficientnet-b4",
            advprop=True,
            num_classes=2,
        )
        # Restore TORCH_HOME to avoid side-effects on other models
        if _prev_torch_home is not None:
            os.environ["TORCH_HOME"] = _prev_torch_home
        else:
            os.environ.pop("TORCH_HOME", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning 2-class logits."""
        return self.net(x)


class SBIPredictor(BasePredictor):
    """SBI video deepfake detector wrapper."""

    name = "sbi"
    modality = "video"

    def __init__(self):
        self._model = None
        self._face_detector = None

    def load(self, weights_dir: Path, device: torch.device) -> None:
        """Load EfficientNet-B4 classifier and MTCNN face detector."""
        self._device = device

        # MTCNN face detector
        self._face_detector = MTCNN(
            keep_all=True,
            device=device,
            post_process=False,
        )

        # SBI classifier
        detector = _Detector()
        weights_file = get_weights_path("sbi") / "FFc23.tar"
        if not weights_file.exists():
            raise FileNotFoundError(f"SBI weights not found at {weights_file}")

        checkpoint = torch.load(
            weights_file,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = checkpoint.get("model", checkpoint)
        detector.load_state_dict(state_dict)
        detector = detector.to(device)
        detector.train(mode=False)

        self._model = detector
        self._model = self._optimize_for_inference(
            self._model,
            use_channels_last=True,
            use_compile=False,
        )
        self._loaded = True
        logger.info("SBI model loaded on %s", device)

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

        batch = np.stack(all_crops, axis=0)
        batch_tensor = (
            torch.from_numpy(batch)
            .permute(0, 3, 1, 2)
            .float()
            .div_(255.0)
            .to(self._device)
        )

        logits = self._model(batch_tensor)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().tolist()

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
                return {
                    "probability": 0.5,
                    "prediction": "real",
                }

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
                return {
                    "probability": 0.5,
                    "prediction": "real",
                }

            # Single batched model inference on all crops
            batch = np.stack(all_crops, axis=0)
            batch_tensor = (
                torch.from_numpy(batch)
                .permute(0, 3, 1, 2)
                .float()
                .div_(255.0)
                .to(self._device)
            )

            logits = self._model(batch_tensor)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().tolist()

            # Group by frame and take max per frame
            frame_max: dict = {}
            for idx, p in zip(crop_frame_idx, probs):
                if idx not in frame_max or p > frame_max[idx]:
                    frame_max[idx] = p

            probability = float(np.mean(list(frame_max.values()))) if frame_max else 0.5

            return {
                "probability": probability,
                "prediction": "fake" if probability >= 0.5 else "real",
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
    """Crop a face region from an image with a relative margin.

    Args:
        img: RGB image array (H, W, 3).
        bbox: (x0, y0, x1, y1) face bounding box.
        margin: Fraction of bbox dimension to add as padding.

    Returns:
        Cropped face region as a numpy array.
    """
    h_img, w_img = img.shape[:2]
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0

    x0_new = max(0, int(x0 - w * margin / 2))
    x1_new = min(w_img, int(x1 + w * margin / 2) + 1)
    y0_new = max(0, int(y0 - h * margin / 2))
    y1_new = min(h_img, int(y1 + h * margin / 2) + 1)

    return img[y0_new:y1_new, x0_new:x1_new]
