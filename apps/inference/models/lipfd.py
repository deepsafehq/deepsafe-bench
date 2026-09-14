"""LipFD (NeurIPS 2024) video deepfake detector wrapper.

CLIP ViT-L/14 as a global feature extractor and a ResNet-50-based
region-aware classifier operating on multi-scale crops of detected
faces.  Analyses lip region inconsistencies for forgery detection.

Uses MTCNN for face detection, creates multi-scale crops of the
lower-face region, and runs the region-aware classifier.

Requires isolation: model_code uses generic ``models`` module name.

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
import torchvision.transforms as transforms
from config import MODELS_ROOT, get_model_code_path, get_weights_path
from facenet_pytorch import MTCNN
from model_loader import namespaced_import
from models.base import BasePredictor
from PIL import Image

logger = logging.getLogger("models.lipfd")

# Number of frames to uniformly sample
_NUM_FRAMES = 8  # Reduced from 32 for faster inference
# Face bounding-box margin factor
_MARGIN_FACTOR = 0.5
# CLIP normalization constants
_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
# Multi-scale crop indices (from LipFD datasets.py)
_CROP_IDX = [(28, 196), (61, 163)]
# Number of spatial sub-crops per face
_NUM_SUBCROP_POSITIONS = 5

# Transforms used in preprocessing
_resize_224 = transforms.Resize((224, 224))
_resize_1120 = transforms.Resize((1120, 1120))
_clip_normalize = transforms.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD)

# Path to the pre-downloaded CLIP ViT-L-14.pt in .ext_cache
_EXT_CLIP_PT = MODELS_ROOT / ".ext_cache" / "clip-vit-l14-openclip" / "ViT-L-14.pt"


def _ensure_clip_cache_symlink():
    """Ensure ~/.cache/clip/ViT-L-14.pt points to our .ext_cache copy.

    The vendored clip.py defaults to download_root=~/.cache/clip/.
    By placing a symlink there, clip.load() finds the file locally
    and skips the network download entirely.
    """
    if not _EXT_CLIP_PT.exists():
        return  # ext_cache not populated yet
    default_cache = Path.home() / ".cache" / "clip"
    default_cache.mkdir(parents=True, exist_ok=True)
    target = default_cache / "ViT-L-14.pt"
    if target.exists() or target.is_symlink():
        return  # already present
    try:
        target.symlink_to(_EXT_CLIP_PT)
    except OSError:
        import shutil

        shutil.copy2(str(_EXT_CLIP_PT), str(target))


class LipFDPredictor(BasePredictor):
    """LipFD video deepfake detector (needs_isolation=True)."""

    name = "lipfd"
    modality = "video"

    def __init__(self):
        self._model = None
        self._face_detector = None

    def load(self, weights_dir: Path, device: torch.device) -> None:
        """Load LipFD model and MTCNN face detector."""
        self._device = device
        code_path = get_model_code_path("lipfd")

        # Ensure vendored clip.load() finds the pre-downloaded ViT-L-14.pt
        # in ~/.cache/clip/ via symlink to .ext_cache -- no network at runtime.
        _ensure_clip_cache_symlink()

        # MTCNN face detector
        self._face_detector = MTCNN(
            keep_all=True,
            device=device,
            post_process=False,
        )

        # Import build_model with namespace isolation (generic ``models`` name)
        models_mod = namespaced_import("models", code_path, namespace="_ds_lipfd")
        build_model_fn = models_mod.build_model

        model = build_model_fn("CLIP:ViT-L/14")

        weights_file = get_weights_path("lipfd") / "lipfd_checkpoint.pth"
        if not weights_file.exists():
            raise FileNotFoundError(f"LipFD weights not found at {weights_file}")

        checkpoint = torch.load(
            weights_file,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])
        model = model.to(device)
        model.train(mode=False)

        self._model = model
        self._model = self._optimize_for_inference(
            self._model,
            use_compile=False,
        )
        self._loaded = True
        logger.info("LipFD model loaded on %s", device)

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

        per_frame_max: List[float] = []
        for i, (frame, boxes) in enumerate(zip(frames, face_boxes)):
            if boxes is None or len(boxes) == 0:
                continue
            frame_scores: List[float] = []
            for box in boxes:
                x0, y0, x1, y1 = box.tolist()
                face_crop = _crop_face(frame, (x0, y0, x1, y1))
                if face_crop.size == 0:
                    continue
                img_1120, crops = _prepare_lipfd_inputs(face_crop)
                img_batch = img_1120.unsqueeze(0).to(self._device)
                crops_batch = [
                    [c.unsqueeze(0).to(self._device) for c in sc] for sc in crops
                ]
                features = self._model.get_features(
                    img_batch,
                ).to(self._device)
                pred_score, _, _ = self._model(
                    crops_batch,
                    features,
                )
                frame_scores.append(pred_score.sigmoid().item())
            if frame_scores:
                per_frame_max.append(max(frame_scores))

        probability = float(np.mean(per_frame_max)) if per_frame_max else 0.5
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

            per_frame_max: List[float] = []

            # Batch face detection across all frames at once
            pil_frames = [Image.fromarray(f) for f in frames]
            all_boxes, _ = self._face_detector.detect(pil_frames)

            for i, (frame, boxes) in enumerate(
                zip(frames, all_boxes),
            ):
                if boxes is None or len(boxes) == 0:
                    continue

                frame_scores: List[float] = []
                for box in boxes:
                    x0, y0, x1, y1 = box.tolist()
                    face_crop = _crop_face(frame, (x0, y0, x1, y1))
                    if face_crop.size == 0:
                        continue

                    img_1120, crops = _prepare_lipfd_inputs(face_crop)

                    img_batch = img_1120.unsqueeze(0).to(self._device)
                    crops_batch = [
                        [c.unsqueeze(0).to(self._device) for c in sc] for sc in crops
                    ]

                    features = self._model.get_features(
                        img_batch,
                    ).to(self._device)
                    pred_score, _, _ = self._model(
                        crops_batch,
                        features,
                    )
                    prob = pred_score.sigmoid().item()
                    frame_scores.append(prob)

                if frame_scores:
                    per_frame_max.append(max(frame_scores))

            if per_frame_max:
                probability = float(np.mean(per_frame_max))
            else:
                probability = 0.5

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


def _prepare_lipfd_inputs(
    face_crop: np.ndarray,
) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
    """Convert a face crop into LipFD model inputs.

    Mirrors the preprocessing in LipFD's AVLip dataset:
      1. Convert face to float32 tensor (C, H, W) in BGR.
      2. CLIP-normalize, then create 5 horizontal sub-crops of the
         lower-face region, each at 3 zoom scales.
      3. Resize the raw image to 1120x1120 for the CLIP encoder.

    Args:
        face_crop: RGB uint8 numpy array (H, W, 3).

    Returns:
        img_1120: Tensor (3, 1120, 1120) for CLIP feature extraction.
        crops: List of 3 scale-lists, each containing 5 tensors
            of shape (3, 224, 224).
    """
    # Match training: BGR input (original uses cv2.imread directly)
    face_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
    img_tensor = torch.tensor(
        face_bgr,
        dtype=torch.float32,
    ).permute(2, 0, 1)

    # CLIP-normalize for the crop pathway
    img_norm = _clip_normalize(img_tensor)

    # Build multi-scale crops from lower-face region
    _, h, w = img_norm.shape
    half_h = h // 2
    lower_face = img_norm[:, half_h:, :]

    lf_h, lf_w = lower_face.shape[1], lower_face.shape[2]
    crop_size = min(lf_h, lf_w)
    if crop_size < 2:
        crop_size = max(lf_h, lf_w, 2)

    if lf_w > crop_size:
        positions = np.linspace(
            0,
            lf_w - crop_size,
            _NUM_SUBCROP_POSITIONS,
            dtype=int,
        )
    else:
        positions = [0] * _NUM_SUBCROP_POSITIONS

    crops: List[List[torch.Tensor]] = [[], [], []]
    for pos in positions:
        patch = lower_face[:, :crop_size, pos : pos + crop_size]
        crop_224 = _resize_224(patch)
        crops[0].append(crop_224)

        # Scale 1 (0.65x inner crop)
        crops[1].append(
            _resize_224(
                crop_224[
                    :,
                    _CROP_IDX[0][0] : _CROP_IDX[0][1],
                    _CROP_IDX[0][0] : _CROP_IDX[0][1],
                ]
            )
        )
        # Scale 2 (0.45x inner crop)
        crops[2].append(
            _resize_224(
                crop_224[
                    :,
                    _CROP_IDX[1][0] : _CROP_IDX[1][1],
                    _CROP_IDX[1][0] : _CROP_IDX[1][1],
                ]
            )
        )

    # Full image at 1120x1120 for CLIP global features
    img_1120 = _resize_1120(img_tensor)

    return img_1120, crops
