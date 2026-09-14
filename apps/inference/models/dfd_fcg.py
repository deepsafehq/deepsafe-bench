"""DFD-FCG (CVPR 2025) video deepfake detector wrapper.

CLIP ViT-L/14 with Synoptic Video Learner and Facial Component
Guidance.  Analyses temporal and spatial inconsistencies across
facial components (lips, skin, eyes, nose) using learned synoptic
attention over multi-frame CLIP embeddings.

Uses MTCNN for face detection, builds overlapping clips of face
crops, runs each clip through the model, and averages per-clip
fake probabilities.

"""

import logging
import math
import os
import sys
import tempfile
import types
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from config import MODELS_ROOT, get_model_code_path, get_weights_path
from facenet_pytorch import MTCNN
from model_loader import namespaced_import
from models.base import BasePredictor
from PIL import Image

logger = logging.getLogger("models.dfd_fcg")

# CLIP ViT-L/14 standard resolution
_IMAGE_SIZE = 224
# Number of frames per clip (from config num_frames=10)
_NUM_FRAMES = 10
# Frame sampling stride in seconds (from demo.py)
_FRAME_STRIDE = 0.333
# Face crop margin factor
_MARGIN_FACTOR = 0.5
# Max frames to extract (avoids processing all 300+ frames)
_MAX_DENSE_FRAMES = 90
# Stride for clip start positions (wider stride = fewer clips;
# 6 halves clip count vs 3 while still covering temporal span).
_CLIP_START_STRIDE = 6

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


class DFDFCGPredictor(BasePredictor):
    """DFD-FCG video deepfake detector wrapper."""

    name = "dfd_fcg"
    modality = "video"

    def __init__(self):
        self._model = None
        self._face_detector = None
        self._transform = None

    def load(self, weights_dir: Path, device: torch.device) -> None:
        """Load DFD-FCG model and MTCNN face detector."""
        self._device = device
        code_path = get_model_code_path("dfd_fcg")

        # Ensure vendored clip.load() finds the pre-downloaded ViT-L-14.pt
        # in ~/.cache/clip/ via symlink to .ext_cache -- no network at runtime.
        _ensure_clip_cache_symlink()

        # Stub wandb (imported at top of svl.py) with proper __spec__
        import importlib as _importlib

        _wandb_stub = types.ModuleType("wandb")
        _wandb_stub.__spec__ = _importlib.machinery.ModuleSpec("wandb", None)
        _wandb_stub.init = lambda *a, **kw: None
        _wandb_stub.log = lambda *a, **kw: None
        _wandb_stub.watch = lambda *a, **kw: None
        _wandb_stub.Table = type("Table", (), {"__init__": lambda *a, **kw: None})
        _wandb_stub.Image = type("Image", (), {"__init__": lambda *a, **kw: None})
        sys.modules["wandb"] = _wandb_stub

        # Face detector: SCRFD (5-10x faster) with MTCNN fallback
        try:
            from insightface.app import FaceAnalysis

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if device.type == "cuda"
                else ["CPUExecutionProvider"]
            )
            app = FaceAnalysis(
                name="buffalo_sc",
                allowed_modules=["detection"],
                providers=providers,
            )
            ctx_id = 0 if device.type == "cuda" else -1
            app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            self._face_detector = ("scrfd", app)
        except Exception:
            logger.warning("SCRFD unavailable, using MTCNN fallback")
            self._face_detector = (
                "mtcnn",
                MTCNN(
                    keep_all=True,
                    device=device,
                    post_process=False,
                ),
            )

        # Build and load model
        weights_file = get_weights_path("dfd_fcg") / "dfd_fcg_checkpoint.pth"
        if not weights_file.exists():
            raise FileNotFoundError(f"DFD-FCG weights not found at {weights_file}")

        # Face semantic features file required by FFG module
        face_features_path = str(
            code_path / "misc" / "L14_real_semantic_patches_v4_2000.pickle"
        )

        # Import with namespace isolation (generic 'src' module name)
        svl_mod = namespaced_import(
            "src.model.clip.svl",
            code_path,
            namespace="_ds_dfd_fcg",
        )
        FFGSynoVideoLearner = svl_mod.FFGSynoVideoLearner

        model = FFGSynoVideoLearner(
            face_feature_path=face_features_path,
            face_parts=["lips", "skin", "eyes", "nose"],
            architecture="ViT-L/14",
            num_frames=_NUM_FRAMES,
            ksize_s=5,
            ksize_t=5,
            s_k_attr="k",
            s_v_attr="emb",
            t_attrs=["q", "k", "v"],
            text_embed=False,
            op_mode=["S", "T"],
        )
        model_cls = model.__class__

        # Checkpoint stores face_feature_path as relative path;
        # chdir to model_code so load_from_checkpoint resolves it.
        original_cwd = os.getcwd()
        os.chdir(str(code_path))
        try:
            try:
                model = model_cls.load_from_checkpoint(str(weights_file))
            except Exception:
                logger.info(
                    "Strict checkpoint load failed, retrying non-strict.",
                )
                model = model_cls.load_from_checkpoint(
                    str(weights_file),
                    strict=False,
                )
        finally:
            os.chdir(original_cwd)

        model = model.to(device)
        model.requires_grad_(False)
        model.train(mode=False)

        self._transform = model.transform
        self._model = model
        self._model = self._optimize_for_inference(
            self._model,
            use_compile=False,
        )
        self._loaded = True
        logger.info("DFD-FCG model loaded on %s", device)

    def predict(self, raw_bytes: bytes) -> dict:
        """Run inference on raw video bytes."""
        return self._timed_predict(self._run, raw_bytes)

    def predict_preprocessed(self, video_data) -> dict:
        """Run inference using shared preprocessed video data.

        Reuses pre-extracted frames_all to skip video decoding.
        Still runs its own SCRFD face detection (model-specific
        detection quality matters for accuracy).
        """
        return self._timed_predict(
            lambda _: self._run_on_frames(
                video_data.frames_all,
                video_data.fps,
            ),
            b"",
        )

    @torch.inference_mode()
    def _run_on_frames(
        self,
        frames: List[np.ndarray],
        fps: float,
    ) -> dict:
        """Core inference on pre-extracted frames."""
        if not frames:
            return {"probability": 0.5, "prediction": "real"}

        # Face detection on dense frames
        face_crops: List[Optional[np.ndarray]] = []
        faces_detected = 0
        det_kind, detector = self._face_detector

        for frame in frames:
            if det_kind == "scrfd":
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                det_faces = detector.get(bgr)
                if not det_faces:
                    face_crops.append(None)
                    continue
                areas = [
                    (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) for f in det_faces
                ]
                best = det_faces[int(np.argmax(areas))]
                face = _crop_face(frame, best.bbox.tolist())
            else:
                pil_img = Image.fromarray(frame)
                boxes, _ = detector.detect(pil_img)
                if boxes is None or len(boxes) == 0:
                    face_crops.append(None)
                    continue
                areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
                best_idx = int(np.argmax(areas))
                face = _crop_face(
                    frame,
                    boxes[best_idx].tolist(),
                )

            if face.size == 0:
                face_crops.append(None)
                continue
            face_crops.append(face)
            faces_detected += 1

        if faces_detected == 0:
            return {"probability": 0.5, "prediction": "real"}

        # Build clips and run inference
        stride_frames = max(1, int(math.floor(_FRAME_STRIDE * fps)))
        clip_indices = [i * stride_frames for i in range(_NUM_FRAMES)]
        max_start = len(frames) - clip_indices[-1] - 1

        probs: List[float] = []
        batch_size = 8

        if max_start <= 0:
            # Short video: uniform sample
            sample_idx = np.linspace(
                0,
                len(frames) - 1,
                _NUM_FRAMES,
                endpoint=True,
                dtype=int,
            )
            clip_crops = []
            for idx in sample_idx:
                c = face_crops[idx]
                if c is None:
                    c = _find_nearest_crop(face_crops, idx)
                if c is not None:
                    clip_crops.append(c)

            if len(clip_crops) == _NUM_FRAMES:
                tensor = self._prepare_clip_tensor(clip_crops)
                if tensor is not None:
                    tensor = tensor.to(self._device)
                    result = self._model.evaluate(tensor)
                    p = result["logits"].softmax(dim=-1)[:, 1].cpu().item()
                    probs.append(p)
        else:
            clip_starts = list(
                range(0, max_start + 1, _CLIP_START_STRIDE),
            )
            for batch_start in range(
                0,
                len(clip_starts),
                batch_size,
            ):
                batch_clips = clip_starts[batch_start : batch_start + batch_size]
                tensors = []
                for start in batch_clips:
                    clip_crops = []
                    for offset in clip_indices:
                        idx = start + offset
                        c = face_crops[idx]
                        if c is None:
                            c = _find_nearest_crop(
                                face_crops,
                                idx,
                            )
                        if c is not None:
                            clip_crops.append(c)

                    if len(clip_crops) == _NUM_FRAMES:
                        t = self._prepare_clip_tensor(clip_crops)
                        if t is not None:
                            tensors.append(t)

                if tensors:
                    batch_tensor = torch.cat(
                        tensors,
                        dim=0,
                    ).to(self._device)
                    result = self._model.evaluate(batch_tensor)
                    batch_probs = (
                        result["logits"].softmax(dim=-1)[:, 1].flatten().cpu().tolist()
                    )
                    probs.extend(batch_probs)

        if probs:
            probability = float(np.mean(probs))
        else:
            probability = 0.5

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
            frames, fps = _extract_dense_frames(tmp_path)
            return self._run_on_frames(frames, fps)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _prepare_clip_tensor(
        self,
        face_crops: List[np.ndarray],
    ) -> Optional[torch.Tensor]:
        """Apply CLIP transform to face crops -> (1, T, 3, 224, 224)."""
        if not face_crops or self._transform is None:
            return None

        transformed = []
        for crop in face_crops:
            t = torch.from_numpy(crop).permute(2, 0, 1)
            t = self._transform(t)
            transformed.append(t)

        clip_tensor = torch.stack(transformed, dim=0)
        return clip_tensor.unsqueeze(0)  # (1, T, C, H, W)


def _extract_dense_frames(
    video_path: str,
    max_frames: int = _MAX_DENSE_FRAMES,
) -> Tuple[List[np.ndarray], float]:
    """Extract frames from a video, uniformly subsampled to max_frames.

    When the video has more frames than max_frames, uniformly samples
    to cap at max_frames while preserving temporal spread.  Returns
    an effective fps that reflects the subsampling so downstream clip
    building adapts automatically.

    Args:
        video_path: Path to the video on disk.
        max_frames: Maximum frames to extract.

    Returns:
        Tuple of (list of RGB uint8 arrays, effective_fps).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        return [], fps

    if total <= max_frames:
        # Short video — read all frames
        frames: List[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames, fps

    # Uniformly subsample to max_frames
    indices = np.linspace(
        0,
        total - 1,
        max_frames,
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
    effective_fps = fps * len(frames) / total
    return frames, effective_fps


def _crop_face(
    img: np.ndarray,
    bbox: Tuple[float, float, float, float],
    margin: float = _MARGIN_FACTOR,
) -> np.ndarray:
    """Crop a face region from an image with a relative margin."""
    h_img, w_img = img.shape[:2]
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0

    x0_new = max(0, int(x0 - w * margin / 2))
    x1_new = min(w_img, int(x1 + w * margin / 2) + 1)
    y0_new = max(0, int(y0 - h * margin / 2))
    y1_new = min(h_img, int(y1 + h * margin / 2) + 1)

    return img[y0_new:y1_new, x0_new:x1_new]


def _find_nearest_crop(
    crops: List[Optional[np.ndarray]],
    idx: int,
) -> Optional[np.ndarray]:
    """Find the nearest non-None face crop to a given index."""
    n = len(crops)
    for offset in range(n):
        for candidate in (idx - offset, idx + offset):
            if 0 <= candidate < n and crops[candidate] is not None:
                return crops[candidate]
    return None
