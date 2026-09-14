"""UniversalFakeDetect (CLIP ViT-L/14) wrapper.

Loads the CLIP ViT-L/14 backbone via the model_code get_model() helper,
then loads fc_weights.pth into model.fc.  Inference: sigmoid on the raw
output gives fake probability.

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
logger = logging.getLogger("models.universal")

# CLIP normalisation constants (from the repo's validate.py)
_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

_TRANSFORM = transforms.Compose(
    [
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
    ]
)

# Path to the pre-downloaded CLIP ViT-L-14.pt in .ext_cache
_EXT_CLIP_PT = MODELS_ROOT / ".ext_cache" / "clip-vit-l14-openclip" / "ViT-L-14.pt"


def _ensure_clip_cache_symlink():
    """Ensure the vendored clip.load() finds ViT-L-14.pt locally.

    The vendored clip.py defaults to download_root=~/.cache/clip/ and
    looks for <download_root>/ViT-L-14.pt.  We symlink the ext_cache
    directory itself so the file is found at the expected path without
    any network download or SHA256 re-verification.
    """
    if not _EXT_CLIP_PT.exists():
        return  # ext_cache not populated yet; will download at runtime
    ext_clip_dir = _EXT_CLIP_PT.parent
    default_cache = Path.home() / ".cache" / "clip"
    # If ~/.cache/clip already exists as a real directory (not symlink),
    # ensure ViT-L-14.pt is present inside it.
    if default_cache.is_dir() and not default_cache.is_symlink():
        target = default_cache / "ViT-L-14.pt"
        if target.exists() or target.is_symlink():
            return
        try:
            target.symlink_to(_EXT_CLIP_PT)
        except OSError:
            import shutil

            shutil.copy2(str(_EXT_CLIP_PT), str(target))
        return
    # Otherwise, symlink ~/.cache/clip -> ext_cache directory so
    # clip.load() resolves <download_root>/ViT-L-14.pt directly.
    default_cache.parent.mkdir(parents=True, exist_ok=True)
    if default_cache.is_symlink():
        default_cache.unlink()
    try:
        default_cache.symlink_to(ext_clip_dir)
    except OSError:
        # Fallback: create directory and file symlink
        default_cache.mkdir(parents=True, exist_ok=True)
        target = default_cache / "ViT-L-14.pt"
        if not target.exists():
            try:
                target.symlink_to(_EXT_CLIP_PT)
            except OSError:
                import shutil

                shutil.copy2(str(_EXT_CLIP_PT), str(target))


class UniversalPredictor(BasePredictor):
    """UniversalFakeDetect wrapper."""

    name = "universal"
    modality = "image"

    def __init__(self):
        self._model = None

    def load(self, weights_dir: Path, device: torch.device) -> None:
        """Load CLIP:ViT-L/14 with fc_weights.pth."""
        self._device = device

        # Ensure vendored clip.load() finds the pre-downloaded ViT-L-14.pt
        # in ~/.cache/clip/ (its default download_root) via a symlink to
        # our .ext_cache, so NO network call happens at runtime.
        _ensure_clip_cache_symlink()

        code_path = get_model_code_path("universal")
        models_mod = namespaced_import(
            "models",
            code_path,
            namespace="_ds_universal",
        )
        get_model = models_mod.get_model

        self._model = get_model("CLIP:ViT-L/14")

        fc_weights = get_weights_path("universal") / "fc_weights.pth"
        state_dict = torch.load(fc_weights, map_location="cpu")
        self._model.fc.load_state_dict(state_dict)

        self._model.to(device)
        self._model.train(mode=False)
        self._loaded = True
        logger.info("UniversalFakeDetect model loaded on %s", device)

    def predict(self, raw_bytes: bytes) -> dict:
        """Run inference on raw image bytes."""
        return self._timed_predict(self._run, raw_bytes)

    @torch.inference_mode()
    def _run(self, raw_bytes: bytes) -> dict:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        tensor = _TRANSFORM(image).unsqueeze(0).to(self._device)

        output = self._model(tensor).sigmoid().flatten().item()

        return {
            "probability": float(output),
            "prediction": "fake" if output >= 0.5 else "real",
        }
