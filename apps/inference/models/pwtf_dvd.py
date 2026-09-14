"""PwTF-DVD (ICCV 2025) video deepfake detector wrapper.

Pixel-wise Temporal Frequency Domain Video Deepfake Detection.
Dual-stream architecture: I3D backbone for spatial features and a
ResNet-based attention network for temporal frequency features,
fused via spatial and temporal transformer encoders.

Preprocessing: RetinaFace detection, SORT-based tracking, face
alignment via 68-point landmarks, and temporal FFT computation on
median-filtered residuals.

Probability inversion fix (2026-04-11): checkpoint outputs higher
sigmoid values for real content and lower for fake. Verified
empirically on 15.5K-sample eval: raw sigmoid gives AUC=0.417
(anti-correlated), inverted gives AUC=0.583.

"""

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from config import MODELS_ROOT, get_model_code_path, get_weights_path
from model_loader import namespaced_import
from models.base import BasePredictor
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor

logger = logging.getLogger("models.pwtf_dvd")

# Face crop size after alignment
_FACE_CROP_SIZE = 224
# Clip size for temporal analysis (from root_setting.yaml)
_CLIP_SIZE = 32
# Maximum frames to extract (reduced from 768 for speed;
# 64 frames still yields ~4 clips with stride 16, enough for
# temporal aggregation without AUC loss).
_MAX_FRAMES = 64
# Stride for sliding-window clips (wider stride = fewer clips;
# 16 is 2x the original 8 but still captures temporal patterns).
_CLIP_WINDOW_STRIDE = 16


class PwTFDVDPredictor(BasePredictor):
    """PwTF-DVD video deepfake detector wrapper."""

    name = "pwtf_dvd"
    modality = "video"

    def __init__(self):
        self._model = None
        # Lazy-loaded references to model_code modules
        self._detect_all = None
        self._grab_all_frames = None
        self._get_crop_box = None
        self._multiple_tracking = None
        self._find_longest = None
        self._crop_align_func = None

    def load(self, weights_dir: Path, device: torch.device) -> None:
        """Load PwTF-DVD model and face detection tools."""
        self._device = device

        # Point torch.utils.model_zoo to pre-cached face alignment
        # weights so RetinaFace and LandmarkPredictor load offline.
        pwtf_face_cache = MODELS_ROOT / ".ext_cache" / "pwtf-dvd-face"
        if pwtf_face_cache.exists():
            os.environ["TORCH_HOME"] = str(pwtf_face_cache)

        code_path = get_model_code_path("pwtf_dvd")

        # Import with namespace isolation (uses generic 'model' and 'test_tools')
        inference_dir = code_path / "inference"
        framework_mod = namespaced_import(
            "model.framework", inference_dir, namespace="_ds_pwtf_dvd"
        )
        common_mod = namespaced_import(
            "test_tools.common", inference_dir, namespace="_ds_pwtf_dvd"
        )
        utils_mod = namespaced_import(
            "test_tools.utils", inference_dir, namespace="_ds_pwtf_dvd"
        )
        ops_mod = namespaced_import(
            "test_tools.ct.operations", inference_dir, namespace="_ds_pwtf_dvd"
        )
        crop_mod = namespaced_import(
            "test_tools.faster_crop_align_xray", inference_dir, namespace="_ds_pwtf_dvd"
        )

        get_model = framework_mod.get_model
        detect_all = common_mod.detect_all
        grab_all_frames = common_mod.grab_all_frames
        FasterCropAlignXRay = crop_mod.FasterCropAlignXRay

        self._detect_all = detect_all
        self._grab_all_frames = grab_all_frames
        self._get_crop_box = utils_mod.get_crop_box
        self._multiple_tracking = ops_mod.multiple_tracking
        self._find_longest = ops_mod.find_longest
        self._crop_align_func = FasterCropAlignXRay(_FACE_CROP_SIZE)

        # Load classifier
        weights_file = get_weights_path("pwtf_dvd") / "pwtf_dvd_checkpoint.pth"
        if not weights_file.exists():
            raise FileNotFoundError(f"PwTF-DVD weights not found at {weights_file}")

        model = get_model()
        state_dict = torch.load(
            weights_file,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.train(mode=False)

        self._model = model
        self._model = self._optimize_for_inference(
            self._model,
            use_compile=False,
        )
        self._loaded = True
        logger.info("PwTF-DVD model loaded on %s", device)

    def predict(self, raw_bytes: bytes) -> dict:
        """Run inference on raw video bytes."""
        return self._timed_predict(self._run, raw_bytes)

    @torch.inference_mode()
    def _run(self, raw_bytes: bytes) -> dict:
        with tempfile.NamedTemporaryFile(
            suffix=".mp4",
            delete=False,
        ) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        try:
            result = self._run_pipeline(tmp_path)
            probability = result["probability"]
            return {
                "probability": probability,
                "prediction": ("fake" if probability >= 0.5 else "real"),
            }
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _run_pipeline(self, video_path: str) -> dict:
        """Run the full PwTF-DVD inference pipeline.

        Follows the logic from model_code/inference/test_on_raw_video.py:
          1. Detect faces in all frames (RetinaFace).
          2. Track faces (SORT-based tracker).
          3. Generate sliding-window clips of CLIP_SIZE frames.
          4. For each clip: align faces, compute temporal FFT residuals,
             run dual-stream model.
          5. Aggregate per-clip predictions.

        Args:
            video_path: Path to the video file on disk.

        Returns:
            Dict with ``probability``.
        """
        # Step 1: Detect faces
        detect_res, all_lm68, frames = self._detect_all(
            video_path,
            return_frames=True,
            max_size=_MAX_FRAMES,
        )
        if not frames:
            return {"probability": 0.5}

        shape = frames[0].shape[:2]

        # Merge 68-landmark data into detection results
        all_detect_res = []
        for faces, faces_lm68 in zip(detect_res, all_lm68):
            new_faces = []
            for (box, lm5, score), face_lm68 in zip(faces, faces_lm68):
                new_faces.append((box, lm5, face_lm68, score))
            all_detect_res.append(new_faces)
        detect_res = all_detect_res

        # Step 2: Track faces
        tracks = self._multiple_tracking(detect_res)
        tuples = [(0, len(detect_res))] * len(tracks)

        if len(tracks) == 0:
            tuples, tracks = self._find_longest(detect_res)

        if len(tracks) == 0:
            return {"probability": 0.5}

        # Step 3: Extract face crops and landmarks
        data_storage = {}
        frame_boxes = {}
        super_clips = []

        for track_i, ((start, end), track) in enumerate(
            zip(tuples, tracks),
        ):
            super_clips.append(len(track))
            for face, frame_idx, j in zip(
                track,
                range(start, end),
                range(len(track)),
            ):
                box, lm5, lm68 = face[:3]
                big_box = self._get_crop_box(shape, box, scale=0.5)

                top_left = big_box[:2][None, :]
                new_lm5 = lm5 - top_left
                new_lm68 = lm68 - top_left
                new_box = (box.reshape(2, 2) - top_left).reshape(-1)
                info = (new_box, new_lm5, new_lm68, big_box)

                x1, y1, x2, y2 = big_box
                cropped = frames[frame_idx][y1:y2, x1:x2]

                base_key = f"{track_i}_{j}_"
                data_storage[base_key + "img"] = cropped
                data_storage[base_key + "ldm"] = info
                data_storage[base_key + "idx"] = frame_idx
                frame_boxes[frame_idx] = np.rint(box).astype(np.int32)

        # Step 4: Generate sliding-window clips
        clips_for_video = []
        pad_length = _CLIP_SIZE - 1

        for sci, sc_size in enumerate(super_clips):
            inner_index = list(range(sc_size))

            if sc_size < _CLIP_SIZE:
                post_module = inner_index[1:-1][::-1] + inner_index
                l_post = len(post_module)
                if l_post == 0:
                    continue
                post_module = post_module * (pad_length // l_post + 1)
                post_module = post_module[:pad_length]
                if len(post_module) != pad_length:
                    continue

                pre_module = inner_index + inner_index[1:-1][::-1]
                l_pre = len(pre_module)
                if l_pre == 0:
                    continue
                pre_module = pre_module * (pad_length // l_pre + 1)
                pre_module = pre_module[-pad_length:]
                if len(pre_module) != pad_length:
                    continue

                inner_index = pre_module + inner_index + post_module

            padded_size = len(inner_index)
            frame_range = [
                inner_index[i : i + _CLIP_SIZE]
                for i in range(0, padded_size, _CLIP_WINDOW_STRIDE)
                if i + _CLIP_SIZE <= padded_size
            ]
            for indices in frame_range:
                clip = [(sci, t) for t in indices]
                clips_for_video.append(clip)

        if not clips_for_video:
            return {"probability": 0.5}

        # Step 5: Run batched inference on clips
        preds = []
        test_transform = Compose(
            [
                ToTensor(),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        _CLIP_BATCH = 4

        for batch_start in range(
            0,
            len(clips_for_video),
            _CLIP_BATCH,
        ):
            batch_clips = clips_for_video[batch_start : batch_start + _CLIP_BATCH]
            img_stacks = []
            ft_tensors = []

            for clip in batch_clips:
                images = [data_storage[f"{i}_{j}_img"] for i, j in clip]
                landmarks = [data_storage[f"{i}_{j}_ldm"] for i, j in clip]

                landmarks, images = self._crop_align_func(
                    landmarks,
                    images,
                )

                images_tensor = []
                ft_images = []
                for image in images:
                    image = np.array(image)
                    img_pil = Image.fromarray(image)
                    img_tensor = test_transform(img_pil)
                    images_tensor.append(img_tensor)

                    img_filtered = cv2.medianBlur(image.copy(), 5)
                    residual = cv2.cvtColor(
                        (image - img_filtered),
                        cv2.COLOR_RGB2GRAY,
                    )
                    ft_images.append(residual)

                ft_array = np.array(ft_images)
                ft_array = np.absolute(
                    np.fft.fft(ft_array, axis=0)[: _CLIP_SIZE // 2] * (1.0 / _CLIP_SIZE)
                )
                ft_tensors.append(
                    torch.from_numpy(ft_array).unsqueeze(0),
                )
                img_stacks.append(
                    torch.stack(images_tensor, dim=1).unsqueeze(0),
                )

            # Batched forward pass
            batch_img = torch.cat(
                img_stacks,
                dim=0,
            ).to(self._device)
            batch_ft = torch.cat(
                ft_tensors,
                dim=0,
            ).to(self._device)

            output = self._model(batch_img, batch_ft)
            # Invert sigmoid: checkpoint outputs higher values for
            # real content (verified: raw AUC=0.417, inverted=0.583).
            batch_preds = (1.0 - torch.sigmoid(output)).flatten().cpu().tolist()
            if isinstance(batch_preds, float):
                batch_preds = [batch_preds]
            preds.extend(batch_preds)

        probability = float(np.mean(preds))
        return {"probability": probability}
