"""Effort (ICML 2025) image deepfake detector wrapper.

SVD orthogonal subspace decomposition on CLIP ViT-L/14.  The detector
is loaded from the cloned repo's ``detectors.effort_detector`` module.
Inference returns ``preds["prob"]`` which is already in [0, 1].

"""

import io
import logging
import os
from pathlib import Path

import torch
import torchvision.transforms as transforms
from config import MODELS_ROOT, get_model_code_path, get_weights_path
from model_loader import namespaced_import
from models.base import BasePredictor
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger("models.effort")

# CLIP normalisation (from the Effort config / OpenAI CLIP)
_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
_INPUT_RESOLUTION = 224

_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(
            (_INPUT_RESOLUTION, _INPUT_RESOLUTION),
            Image.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
    ]
)


def _build_effort_config() -> dict:
    """Build the minimal config dict for EffortDetector.__init__."""
    return {
        "model_name": "effort",
        "backbone_name": "vit",
        "pretrained": None,
        "backbone_config": {
            "mode": "original",
            "num_classes": 2,
            "inc": 3,
            "dropout": False,
        },
        "resolution": _INPUT_RESOLUTION,
        "mean": _CLIP_MEAN,
        "std": _CLIP_STD,
        "loss_func": "cross_entropy",
    }


class EffortPredictor(BasePredictor):
    """Effort (SVD + CLIP) wrapper."""

    name = "effort"
    modality = "image"

    def __init__(self):
        self._model = None

    def load(self, weights_dir: Path, device: torch.device) -> None:
        """Load EffortDetector from the cloned repo code."""
        self._device = device
        code_path = get_model_code_path("effort")

        # Load CLIP from prefetched .ext_cache instead of downloading
        # at runtime. The snapshot lives at .ext_cache/clip-vit-l14-transformers/
        _ext_clip_dir = str(MODELS_ROOT / ".ext_cache" / "clip-vit-l14-transformers")

        # Patch CLIPModel.from_pretrained to use the local snapshot
        try:
            from transformers import CLIPModel

            _original_from_pretrained = CLIPModel.from_pretrained.__func__

            @classmethod  # type: ignore[misc]
            def _local_from_pretrained(cls, pretrained, *args, **kwargs):
                # Force local-only load from .ext_cache
                kwargs["local_files_only"] = True
                return _original_from_pretrained(
                    cls,
                    _ext_clip_dir,
                    *args,
                    **kwargs,
                )

            CLIPModel.from_pretrained = _local_from_pretrained
        except ImportError:
            pass  # transformers not installed yet

        # The effort code lives under DeepfakeBench/training/ in the repo.
        dfb = code_path / "DeepfakeBench"
        training_path = dfb / "training"

        effort_mod = namespaced_import(
            "detectors.effort_detector",
            training_path,
            namespace="_ds_effort",
        )
        EffortDetector = effort_mod.EffortDetector

        cfg = _build_effort_config()
        effort_model = EffortDetector(config=cfg)
        effort_model.to(device)

        # Load checkpoint
        ckpt_path = get_weights_path("effort") / "genimage_effort.pth"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            if isinstance(ckpt, dict):
                state_dict = ckpt.get("model") or ckpt.get("state_dict") or ckpt
            else:
                state_dict = ckpt
            cleaned = {k.removeprefix("module."): v for k, v in state_dict.items()}
            missing, unexpected = effort_model.load_state_dict(
                cleaned,
                strict=False,
            )
            logger.info(
                "Effort checkpoint loaded. Missing: %d, Unexpected: %d",
                len(missing),
                len(unexpected),
            )
        else:
            logger.warning("No Effort checkpoint at %s", ckpt_path)

        effort_model.train(mode=False)
        self._model = effort_model
        self._loaded = True
        logger.info("Effort model loaded on %s", device)

    def predict(self, raw_bytes: bytes) -> dict:
        """Run inference on raw image bytes."""
        return self._timed_predict(self._run, raw_bytes)

    @torch.inference_mode()
    def _run(self, raw_bytes: bytes) -> dict:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        tensor = _TRANSFORM(image).unsqueeze(0).to(self._device)

        data_dict = {
            "image": tensor,
            "label": torch.tensor([0]).to(self._device),
        }
        preds = self._model(data_dict, inference=True)
        probability = preds["prob"].squeeze().cpu().item()

        return {
            "probability": float(probability),
            "prediction": "fake" if probability >= 0.5 else "real",
        }
