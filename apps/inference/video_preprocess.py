"""Shared video preprocessing: decode once, detect faces once.

Instead of 5+ models each independently decoding the same video
and running MTCNN, this module extracts frames and face bounding
boxes once and shares them.

Usage (in server.py):
    prep = preprocess_video(raw_bytes, device)
    # prep.frames_8   → 8 uniformly-sampled RGB frames
    # prep.frames_all → up to max_frames frames
    # prep.face_boxes_8 → MTCNN boxes per sampled frame
    # prep.fps, prep.width, prep.height
"""

import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("video_preprocess")


@dataclass
class VideoData:
    """Shared preprocessed video data for all models."""

    raw_bytes: bytes
    fps: float = 30.0
    width: int = 0
    height: int = 0
    total_frames: int = 0

    # 8 uniformly-sampled frames (for SBI, RECCE, LipFD)
    frames_8: List[np.ndarray] = field(default_factory=list)
    # MTCNN bounding boxes for frames_8: list of (N, 4) arrays or None
    face_boxes_8: List[Optional[np.ndarray]] = field(default_factory=list)

    # All frames at original fps, capped (for MINTIME, DFD-FCG, PwTF-DVD)
    frames_all: List[np.ndarray] = field(default_factory=list)
    # Face bounding boxes for frames_all (shared across dense models)
    face_boxes_all: List[Optional[np.ndarray]] = field(default_factory=list)

    # Temporary file path (caller manages cleanup)
    tmp_path: str = ""


def preprocess_video(
    raw_bytes: bytes,
    device: torch.device,
    num_sampled: int = 8,
    max_dense: int = 90,
) -> VideoData:
    """Decode video once and run shared face detection.

    Extracts two frame sets:
      - frames_8: uniformly-sampled frames for per-frame models
      - frames_all: all frames up to max_dense for dense models

    Runs MTCNN batch detection on frames_8 (the set most models use).

    Args:
        raw_bytes: Raw video file bytes.
        device: Torch device for MTCNN.
        num_sampled: Number of uniformly-sampled frames.
        max_dense: Maximum frames for dense extraction.

    Returns:
        VideoData with frames and face boxes populated.
    """
    # Write to temp file (models that need a path can use tmp_path)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(raw_bytes)
    tmp.close()
    tmp_path = tmp.name

    vd = VideoData(raw_bytes=raw_bytes, tmp_path=tmp_path)

    try:
        cap = cv2.VideoCapture(tmp_path)
        vd.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        vd.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vd.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vd.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if vd.total_frames <= 0:
            cap.release()
            return vd

        # ── Extract ALL frames (capped at max_dense) ───────────────
        if vd.total_frames <= max_dense:
            all_frames: List[np.ndarray] = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            indices = np.linspace(
                0,
                vd.total_frames - 1,
                max_dense,
                endpoint=True,
                dtype=int,
            )
            all_frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    all_frames.append(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    )

        vd.frames_all = all_frames
        cap.release()

        # ── Uniformly sample num_sampled frames from frames_all ────
        n = len(all_frames)
        if n <= num_sampled:
            vd.frames_8 = list(all_frames)
        else:
            sample_idx = np.linspace(
                0,
                n - 1,
                num_sampled,
                endpoint=True,
                dtype=int,
            )
            vd.frames_8 = [all_frames[i] for i in sample_idx]

        # ── Shared face detection on frames_8 (light models) ─────
        if vd.frames_8:
            vd.face_boxes_8 = _batch_detect_faces(
                vd.frames_8,
                device,
            )

    except Exception as e:
        logger.error("Video preprocessing failed: %s", e)

    return vd


def cleanup_video_data(vd: VideoData) -> None:
    """Remove the temporary video file."""
    if vd.tmp_path and os.path.exists(vd.tmp_path):
        os.remove(vd.tmp_path)


# ── Shared face detector singleton (SCRFD via insightface) ──────────────────

_shared_detector = None


def _get_shared_detector(device: torch.device):
    """Lazy-init a shared SCRFD face detector (5-10x faster than MTCNN)."""
    global _shared_detector
    if _shared_detector is not None:
        return _shared_detector

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
        _shared_detector = ("scrfd", app)
        logger.info("Shared face detector: SCRFD (insightface)")
    except Exception as e:
        logger.warning("SCRFD unavailable (%s), falling back to MTCNN", e)
        from facenet_pytorch import MTCNN

        mtcnn = MTCNN(
            keep_all=True,
            device=device,
            post_process=False,
        )
        _shared_detector = ("mtcnn", mtcnn)
        logger.info("Shared face detector: MTCNN (fallback)")

    return _shared_detector


def _batch_detect_faces(
    frames: List[np.ndarray],
    device: torch.device,
) -> List[Optional[np.ndarray]]:
    """Run face detection on a list of RGB frames.

    Uses SCRFD (insightface) if available, falls back to MTCNN.
    Returns list of numpy arrays of shape (N, 4) per frame,
    or None for frames with no detections.
    """
    kind, detector = _get_shared_detector(device)

    if kind == "scrfd":
        all_boxes: List[Optional[np.ndarray]] = []
        for frame in frames:
            # SCRFD expects BGR input
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            faces = detector.get(bgr)
            if faces:
                boxes = np.array([f.bbox for f in faces])
                all_boxes.append(boxes)
            else:
                all_boxes.append(None)
        return all_boxes

    # MTCNN fallback (batch mode)
    pil_frames = [Image.fromarray(f) for f in frames]
    with torch.inference_mode():
        boxes, _ = detector.detect(pil_frames)
    return list(boxes)
