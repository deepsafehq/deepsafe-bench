"""MINTIME (IEEE T-IFS 2024) video deepfake detector wrapper.

Multi-Identity Size-Invariant TimeSformer.  Uses Xception as a
feature extractor and a Size-Invariant TimeSformer for temporal
classification with identity-aware attention masks.

Pipeline: extract frames, detect faces (MTCNN), crop and cluster
by identity (InceptionResnetV1 embeddings), build identity-ordered
sequences with size embeddings and masks, extract Xception features,
run TimeSformer, return sigmoid score.

Requires isolation: model_code uses generic ``models``, ``transforms``
module names that conflict.

"""

import logging
import os
import sys
import tempfile
import types as std_types
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

import cv2
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from config import MODELS_ROOT, get_model_code_path, get_weights_path
from einops import rearrange
from facenet_pytorch import (
    MTCNN,
    InceptionResnetV1,
    fixed_image_standardization,
)
from model_loader import namespaced_import_multi
from models.base import BasePredictor
from PIL import Image
from torch import nn
from torchvision.transforms import Resize

logger = logging.getLogger("models.mintime")

# Model hyperparameters from size_invariant_timesformer.yaml
_NUM_FRAMES = 16
_IMAGE_SIZE = 224
_NUM_PATCHES = 49  # 7x7 spatial from Xception feature map
_MAX_IDENTITIES = 2
_RANGE_SIZE = 5
_SIZE_EMB_DICT = [
    (1 + i * _RANGE_SIZE, (i + 1) * _RANGE_SIZE) if i != 0 else (0, _RANGE_SIZE)
    for i in range(20)
]

# Model config matching size_invariant_timesformer.yaml
_MODEL_CONFIG = {
    "model": {
        "image-size": _IMAGE_SIZE,
        "patch-size": 1,
        "num-classes": 1,
        "num-patches": _NUM_PATCHES,
        "num-frames": _NUM_FRAMES,
        "max-identities": _MAX_IDENTITIES,
        "dim": 512,
        "depth": 9,
        "dim-head": 64,
        "channels": 2048,
        "heads": 8,
        "attn-dropout": 0.0,
        "ff-dropout": 0.0,
        "shift-tokens": False,
        "enable-size-emb": True,
        "enable-pos-emb": True,
        "enable-identity-attention": True,
    }
}


class MINTIMEPredictor(BasePredictor):
    """MINTIME video deepfake detector (needs_isolation=True)."""

    name = "mintime"
    modality = "video"

    def __init__(self):
        self._extractor = None
        self._model = None
        self._embedding_model = None
        self._IsotropicResize = None
        self._mtcnn = None  # Created once in load(), not per predict()

    def load(self, weights_dir: Path, device: torch.device) -> None:
        """Load Xception extractor, SizeInvariantTimeSformer, and
        InceptionResnetV1 for identity clustering."""
        self._device = device
        code_path = get_model_code_path("mintime")

        # Patch albumentations.augmentations.functional for compat
        _compat = std_types.ModuleType(
            "albumentations.augmentations.functional",
        )
        _compat.crop = None
        sys.modules.setdefault(
            "albumentations.augmentations.functional",
            _compat,
        )

        # Remove ALL paths that have conflicting models/ or transforms/
        # directories, then add only ours.
        import importlib as _il
        import os as _os

        code_str = str(code_path)

        for prefix in ("models", "transforms"):
            stale = [
                k for k in sys.modules if k == prefix or k.startswith(prefix + ".")
            ]
            for k in stale:
                del sys.modules[k]

        # Temporarily remove other paths with conflicting dirs
        conflicting = []
        for p in list(sys.path):
            if p == code_str:
                continue
            if _os.path.isdir(_os.path.join(p, "models")) or _os.path.isdir(
                _os.path.join(p, "transforms")
            ):
                conflicting.append(p)
                sys.path.remove(p)

        if code_str in sys.path:
            sys.path.remove(code_str)
        sys.path.insert(0, code_str)
        _il.invalidate_caches()

        imported = namespaced_import_multi(
            code_path=code_path,
            namespace="_ds_mintime",
            import_specs=[
                {
                    "module": "models.size_invariant_timesformer",
                    "attr": "SizeInvariantTimeSformer",
                },
                {
                    "module": "models.xception",
                    "attr": "xception",
                },
                {
                    "module": "transforms.albu",
                    "attr": "IsotropicResize",
                },
            ],
        )

        SizeInvariantTimeSformer = imported["SizeInvariantTimeSformer"]
        xception_fn = imported["xception"]
        self._IsotropicResize = imported["IsotropicResize"]

        # Restore removed paths
        sys.path.extend(conflicting)

        # Xception feature extractor
        extractor_path = (
            get_weights_path("mintime")
            / "MINTIME"
            / "MINTIME_XC_Extractor_checkpoint30"
        )
        if not extractor_path.exists():
            raise FileNotFoundError(
                f"MINTIME extractor weights not found: {extractor_path}"
            )

        feat_ext = xception_fn(num_classes=1, pretrain_path=None)
        ext_sd = torch.load(
            extractor_path,
            map_location="cpu",
            weights_only=False,
        )
        feat_ext.load_state_dict(_strip_module_prefix(ext_sd))
        feat_ext = feat_ext.to(device)
        feat_ext.train(mode=False)
        self._extractor = self._optimize_for_inference(
            feat_ext,
            use_channels_last=True,
            use_compile=False,
        )

        # SizeInvariantTimeSformer
        model_path = (
            get_weights_path("mintime") / "MINTIME" / "MINTIME_XC_Model_checkpoint30"
        )
        if not model_path.exists():
            raise FileNotFoundError(f"MINTIME model weights not found: {model_path}")

        sit = SizeInvariantTimeSformer(
            config=_MODEL_CONFIG,
            require_attention=False,
        )
        model_sd = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False,
        )
        sit.load_state_dict(_strip_module_prefix(model_sd))
        sit = sit.to(device)
        sit.train(mode=False)
        self._model = self._optimize_for_inference(
            sit,
            use_compile=False,
        )

        # InceptionResnetV1 for identity clustering.
        # Set TORCH_HOME to .ext_cache/inception-resnet-v1/ so
        # facenet-pytorch finds the pre-downloaded vggface2 weights
        # without hitting the network at runtime.
        _prev_torch_home = os.environ.get("TORCH_HOME")
        os.environ["TORCH_HOME"] = str(
            MODELS_ROOT / ".ext_cache" / "inception-resnet-v1"
        )
        emb = InceptionResnetV1(pretrained="vggface2").to(device)
        # Restore TORCH_HOME to avoid side-effects on other models
        if _prev_torch_home is not None:
            os.environ["TORCH_HOME"] = _prev_torch_home
        else:
            os.environ.pop("TORCH_HOME", None)
        emb.train(mode=False)
        self._embedding_model = self._optimize_for_inference(
            emb,
            use_channels_last=True,
            use_compile=False,
        )

        self._loaded = True
        # Shared MTCNN for face detection (created once, reused per predict)
        self._mtcnn = MTCNN(
            device=device,
            thresholds=[0.85, 0.95, 0.95],
            margin=0,
        )

        logger.info("MINTIME models loaded on %s", device)

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
            # Extract all frames
            frames, fps, vid_w, vid_h = _extract_all_frames(tmp_path)
            if not frames:
                return {"probability": 0.5, "prediction": "real"}

            # Detect faces (1 per second)
            bboxes_dict = self._detect_faces_mtcnn(frames, fps)
            has_faces = any(v is not None and len(v) > 0 for v in bboxes_dict.values())
            if not has_faces:
                return {"probability": 0.5, "prediction": "real"}

            return self._run_core(
                frames,
                fps,
                vid_w,
                vid_h,
                bboxes_dict,
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @torch.inference_mode()
    def _run_core(self, frames, fps, vid_w, vid_h, bboxes_dict):
        """Shared inference core used by both _run and _run_on_shared."""
        # Extract crops
        crops = _extract_crops(frames, bboxes_dict, fps)
        if not crops:
            return {"probability": 0.5, "prediction": "real"}

        # Cluster by identity
        clustered = self._cluster_faces(crops)

        # Build identity sequence
        sorted_ids = _get_sorted_identities(clustered)

        (
            videos_tensor,
            size_embeddings,
            mask_tensor,
            identities_mask,
            positions,
            _tokens,
        ) = _generate_masks(
            vid_w,
            vid_h,
            sorted_ids,
            self._IsotropicResize,
        )

        # Run inference
        b, f, h, w, c = videos_tensor.shape
        videos_tensor = videos_tensor.to(self._device)
        identities_mask = identities_mask.to(self._device)
        mask_tensor = mask_tensor.to(self._device)
        positions = positions.to(self._device)

        # Feature extraction: (B*F, C, H, W)
        video_input = rearrange(
            videos_tensor,
            "b f h w c -> (b f) c h w",
        )
        features = self._extractor(video_input)
        features = rearrange(
            features,
            "(b f) c h w -> b f c h w",
            b=b,
            f=f,
        )

        # TimeSformer classification
        pred = self._model(
            features,
            mask=mask_tensor,
            size_embedding=size_embeddings,
            identities_mask=identities_mask,
            positions=positions,
        )

        probability = float(torch.sigmoid(pred[0]).item())

        return {
            "probability": probability,
            "prediction": ("fake" if probability >= 0.5 else "real"),
        }

    def _detect_faces_mtcnn(
        self,
        frames: List[np.ndarray],
        fps: int,
    ) -> Dict[str, Any]:
        """Detect faces in sampled frames using MTCNN.

        Samples one frame per second, runs detection on
        half-resolution images matching original preprocessing.
        """
        mtcnn = self._mtcnn  # Use pre-loaded MTCNN from load()

        bboxes_dict: Dict[str, Any] = {}
        indices = list(range(0, len(frames), max(fps, 1)))
        if not indices:
            indices = [0] if frames else []

        for idx in indices:
            frame = frames[idx]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img = pil_img.resize([s // 2 for s in pil_img.size])

            boxes, _ = mtcnn.detect(pil_img)
            if boxes is not None:
                bboxes_dict[str(idx)] = boxes.tolist()
            else:
                bboxes_dict[str(idx)] = None

        return bboxes_dict

    def _cluster_faces(
        self,
        crops: List[Tuple[int, Image.Image, list]],
        similarity_threshold: float = 0.45,
    ) -> Dict[int, List]:
        """Cluster face crops by identity using embeddings."""
        if not crops:
            return {}

        crops_images = [row[1] for row in crops]
        faces = [np.asarray(Resize([128, 128])(face)) for face in crops_images]
        faces = np.stack([np.uint8(f) for f in faces])
        faces_tensor = torch.as_tensor(faces).permute(0, 3, 1, 2).float()
        faces_tensor = fixed_image_standardization(faces_tensor)
        faces_tensor = faces_tensor.to(self._device)

        embeddings = self._embedding_model(faces_tensor).cpu().numpy()
        similarities = np.dot(embeddings, embeddings.T)

        components = _generate_connected_components(
            similarities,
            similarity_threshold=similarity_threshold,
        )

        clustered: Dict[int, List] = {}
        for identity_index, component in enumerate(components):
            clustered[identity_index] = [crops[fi] for fi in component]

        return clustered


def _strip_module_prefix(state_dict: dict) -> dict:
    """Remove 'module.' prefix from DataParallel state dicts."""
    new_sd = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "", 1) if k.startswith("module.") else k
        new_sd[new_key] = v
    return new_sd


def _extract_all_frames(
    video_path: str,
) -> Tuple[List[np.ndarray], int, int, int]:
    """Extract all frames from a video file.

    Returns:
        Tuple of (all_frames, fps, width, height).
    """
    cap = cv2.VideoCapture(video_path)
    fps = max(int(cap.get(cv2.CAP_PROP_FPS)), 1)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps, width, height


def _extract_crops(
    frames: List[np.ndarray],
    bboxes_dict: Dict[str, Any],
    fps: int,
) -> List[Tuple[int, Image.Image, list]]:
    """Extract face crops from video frames using detected bboxes.

    Follows the original extract_crops logic: iterate per-second
    windows, find nearest frame with bboxes, crop with padding,
    make square.

    Returns:
        List of (frame_index, pil_crop, bbox).
    """
    frames_num = len(frames)
    crops = []

    for i in range(0, frames_num, fps):
        idx = i
        limit = min(i + fps - 1, frames_num - 1)

        while str(idx) not in bboxes_dict or bboxes_dict.get(str(idx)) is None:
            if idx >= limit:
                break
            idx += 1

        if str(idx) not in bboxes_dict or bboxes_dict.get(str(idx)) is None:
            continue

        bboxes = bboxes_dict[str(idx)]
        frame = frames[i] if i < frames_num else frames[-1]

        for bbox in bboxes:
            xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox]
            w = xmax - xmin
            h = ymax - ymin

            if w <= 0 or h <= 0:
                continue

            p_h = h // 3
            p_w = w // 3
            crop_h = (ymax + p_h) - max(ymin - p_h, 0)
            crop_w = (xmax + p_w) - max(xmin - p_w, 0)

            if crop_h > crop_w:
                p_h -= int((crop_h - crop_w) / 2)
            else:
                p_w -= int((crop_w - crop_h) / 2)

            crop = frame[
                max(ymin - p_h, 0) : ymax + p_h,
                max(xmin - p_w, 0) : xmax + p_w,
            ]

            h_c, w_c = crop.shape[:2]
            if h_c <= 0 or w_c <= 0:
                continue

            if h_c > w_c:
                diff = (h_c - w_c) // 2
                if diff > 0:
                    crop = crop[diff:-diff, :]
                else:
                    crop = crop[1:, :]
            elif h_c < w_c:
                diff = (w_c - h_c) // 2
                if diff > 0:
                    crop = crop[:, diff:-diff]
                else:
                    crop = crop[:, :-1]

            if crop.size == 0:
                continue

            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append((i, Image.fromarray(rgb_crop), bbox))

    return crops


def _generate_connected_components(
    similarities: np.ndarray,
    similarity_threshold: float = 0.80,
) -> List[List[int]]:
    """Build a similarity graph and return connected components."""
    graph = nx.Graph()
    n = len(similarities)
    for i in range(n):
        for j in range(i + 1, n):
            if similarities[i, j] > similarity_threshold:
                graph.add_edge(i, j)

    components = [sorted(c) for c in nx.connected_components(graph)]
    all_in_components = set()
    for c in components:
        all_in_components.update(c)
    for i in range(n):
        if i not in all_in_components:
            components.append([i])

    return components


def _get_sorted_identities(
    identities: Dict[int, List],
    num_frames: int = _NUM_FRAMES,
    max_identities: int = _MAX_IDENTITIES,
) -> List[list]:
    """Sort identities by face size and allocate frame slots.

    Returns list of [identity_id, mean_side, num_faces, faces_list].
    """
    sorted_ids = []
    for identity in identities:
        faces = identities[identity]
        mean_side = mean([row[1].size[0] for row in faces])
        sorted_ids.append([identity, mean_side, len(faces), faces])

    sorted_ids.sort(key=lambda x: x[1], reverse=True)

    if len(sorted_ids) > max_identities:
        sorted_ids = sorted_ids[:max_identities]

    identities_number = len(sorted_ids)
    available_additional = []

    if identities_number > 1:
        max_faces_map = {
            1: [num_frames],
            2: [num_frames // 2, num_frames // 2],
            3: [num_frames // 3, num_frames // 3, num_frames // 4],
            4: [
                num_frames // 3,
                num_frames // 3,
                num_frames // 8,
                num_frames // 8,
            ],
        }
        alloc = max_faces_map[identities_number]

        for i in range(identities_number):
            if sorted_ids[i][2] < alloc[i] and i < identities_number - 1:
                sorted_ids[i + 1][2] += alloc[i] - sorted_ids[i][2]
                available_additional.append(0)
            elif sorted_ids[i][2] > alloc[i]:
                available_additional.append(
                    sorted_ids[i][2] - alloc[i],
                )
                sorted_ids[i][2] = alloc[i]
            else:
                available_additional.append(0)
    else:
        sorted_ids[0][2] = num_frames
        available_additional.append(0)

    input_len = sum(r[2] for r in sorted_ids)
    if input_len < num_frames:
        for i in range(identities_number):
            needed = num_frames - input_len
            if available_additional[i] > 0:
                added = min(available_additional[i], needed)
                sorted_ids[i][2] += added
                input_len += added
                if input_len == num_frames:
                    break
        if input_len < num_frames:
            sorted_ids[-1][2] += num_frames - input_len

    return sorted_ids


def _generate_masks(
    video_width: int,
    video_height: int,
    identities: List[list],
    IsotropicResize,
    num_frames: int = _NUM_FRAMES,
    image_size: int = _IMAGE_SIZE,
    num_patches: int = _NUM_PATCHES,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list,
]:
    """Build input tensors, masks, size embeddings, and positions.

    Mirrors generate_masks from MINTIME predict.py.

    Args:
        video_width: Original video width.
        video_height: Original video height.
        identities: Sorted identity list from _get_sorted_identities.
        IsotropicResize: The IsotropicResize transform class.
        num_frames: Number of frames in the sequence.
        image_size: Target image size.
        num_patches: Number of spatial patches.

    Returns:
        Tuple of (videos_tensor, size_embeddings, mask_tensor,
        identities_mask, positions, tokens_per_identity).
    """
    from albumentations import Compose, PadIfNeeded
    from albumentations import Resize as AlbuResize

    mask = []
    sequence = []
    size_embeddings = []
    images_frames = []
    video_area = video_width * video_height / 2

    for identity in identities:
        max_faces = identity[2]
        identity_images = identity[3]

        if len(identity_images) > max_faces:
            idx = np.round(
                np.linspace(
                    0,
                    len(identity_images) - 2,
                    max_faces,
                ),
            ).astype(int)
            identity_images = list(
                np.asarray(identity_images, dtype=object)[idx],
            )

        images_frames.extend(img[0] for img in identity_images)
        pil_images = [img[1] for img in identity_images]

        # Size embeddings based on face-frame area ratio
        identity_size_embs = []
        for img in pil_images:
            face_area = img.size[0] * img.size[1]
            ratio = int(face_area * 100 / max(video_area, 1))
            side_ranges = list(
                map(
                    lambda a_: ratio in range(a_[0], a_[1] + 1),
                    _SIZE_EMB_DICT,
                ),
            )
            matches = np.where(side_ranges)[0]
            identity_size_embs.append(
                int(matches[0] + 1) if len(matches) > 0 else 1,
            )

        # Pad with empty frames if needed
        if len(pil_images) < max_faces:
            diff = max_faces - len(identity_size_embs)
            identity_size_embs = list(identity_size_embs) + [0] * diff
            pil_images.extend(
                [
                    np.zeros(
                        (image_size, image_size, 3),
                        dtype=np.uint8,
                    )
                    for _ in range(diff)
                ],
            )
            mask.extend(
                [1 if i < max_faces - diff else 0 for i in range(max_faces)],
            )
            images_frames.extend([max(images_frames)] * diff)
        else:
            mask.extend([1] * max_faces)

        size_embeddings.extend(identity_size_embs)
        sequence.extend(pil_images)

    # Convert PIL images to numpy arrays
    sequence = [np.asarray(img) for img in sequence]

    # Apply albumentations transform
    additional_targets_keys = [f"image{i}" for i in range(num_frames)]
    additional_targets_values = ["image"] * num_frames
    additional_targets = dict(
        zip(additional_targets_keys, additional_targets_values),
    )

    transform = Compose(
        [
            IsotropicResize(
                max_side=image_size,
                interpolation_down=cv2.INTER_AREA,
                interpolation_up=cv2.INTER_CUBIC,
            ),
            PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
            ),
            AlbuResize(height=image_size, width=image_size),
        ],
        additional_targets=additional_targets,
        is_check_shapes=False,  # faces may be non-square before resize
    )

    transform_kwargs = {"image": sequence[0]}
    for i in range(1, len(sequence)):
        transform_kwargs[f"image{i}"] = sequence[i]

    transformed = transform(**transform_kwargs)
    sequence = [transformed[k] for k in transformed]

    # Build identities_mask
    identities_mask = []
    last_range_end = 0
    for identity in identities:
        n_faces = identity[2]
        identity_mask = [
            last_range_end <= i < last_range_end + n_faces for i in range(num_frames)
        ]
        for _ in range(n_faces):
            identities_mask.append(identity_mask)
        last_range_end += n_faces

    # Temporal-positional embedding
    images_frames_positions = {
        k: v + 1 for v, k in enumerate(sorted(set(images_frames)))
    }
    frame_positions = [images_frames_positions[f] for f in images_frames]

    if num_patches is not None:
        positions = []
        for fp in frame_positions:
            positions.extend(
                [
                    i + 1
                    for i in range(
                        (fp - 1) * num_patches,
                        num_patches * fp,
                    )
                ],
            )
        positions.insert(0, 0)  # CLS token position
    else:
        positions = []

    tokens_per_identity = []
    for i, ident in enumerate(identities):
        if i > 0:
            tokens_per_identity.append(
                (
                    ident[0],
                    ident[2] * num_patches + identities[i - 1][2] * num_patches,
                ),
            )
        else:
            tokens_per_identity.append(
                (ident[0], ident[2] * num_patches),
            )

    return (
        torch.tensor([sequence]).float(),
        torch.tensor([size_embeddings]).int(),
        torch.tensor([mask]).bool(),
        torch.tensor([identities_mask]).bool(),
        torch.tensor([positions]),
        tokens_per_identity,
    )
