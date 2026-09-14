"""CO-SPY (CVPR 2025, Sony Research) image deepfake detector wrapper.

Dual-branch fusion of SigLIP semantic features and SD VAE reconstruction
artifact features.  Preprocessing: Resize(384) -> CenterCrop(384) ->
ToTensor().  Inference: sigmoid on the raw fusion logit gives the fake
probability.

"""

import importlib
import io
import logging
import os
from pathlib import Path

import torch
import torchvision.transforms as transforms
from config import MODELS_ROOT, get_model_code_path, get_weights_path
from models.base import BasePredictor
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger("models.cospy")

# Fusion preprocessing constants (matches the service's test_transform)
_LOAD_SIZE = 384
_CROP_SIZE = 384

# HuggingFace repo for weight download
_HF_WEIGHTS_REPO = "ruojiruoli/Co-Spy-Pretrained-Weights"
_HF_WEIGHTS_SUBDIR = "sd-v1_4"
_SEMANTIC_CKPT = "semantic_weights.pth"
_ARTIFACT_CKPT = "artifact_weights.pth"
_FUSION_CKPT = "fusion_weights.pth"

_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(_LOAD_SIZE),
        transforms.CenterCrop(_CROP_SIZE),
        transforms.ToTensor(),
    ]
)


def _ensure_weights_downloaded(weights_dir: Path) -> Path:
    """Download CO-SPY pretrained weights from HuggingFace if missing.

    Returns:
        Path to the directory containing the three weight files.
    """
    weights_subdir = weights_dir / _HF_WEIGHTS_SUBDIR
    semantic = weights_subdir / _SEMANTIC_CKPT
    artifact = weights_subdir / _ARTIFACT_CKPT
    fusion = weights_subdir / _FUSION_CKPT

    if semantic.exists() and artifact.exists() and fusion.exists():
        logger.info("All CO-SPY weight files found in %s", weights_subdir)
        return weights_subdir

    logger.info(
        "Downloading CO-SPY weights from HuggingFace (%s)...",
        _HF_WEIGHTS_REPO,
    )
    weights_subdir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download

    for fname in (_SEMANTIC_CKPT, _ARTIFACT_CKPT, _FUSION_CKPT):
        remote_path = f"{_HF_WEIGHTS_SUBDIR}/{fname}"
        dest = weights_subdir / fname
        if not dest.exists():
            logger.info("  Downloading %s ...", remote_path)
            hf_hub_download(
                repo_id=_HF_WEIGHTS_REPO,
                filename=remote_path,
                local_dir=str(weights_dir),
            )

    return weights_subdir


class COSPYPredictor(BasePredictor):
    """CO-SPY dual-branch fusion detector wrapper."""

    name = "cospy"
    modality = "image"

    def __init__(self):
        self._model = None

    def load(self, weights_dir: Path, device: torch.device) -> None:
        """Load the CO-SPY fusion detector from model_code."""
        self._device = device

        code_path = get_model_code_path("cospy")
        cospy_weights = get_weights_path("cospy")

        # ── Redirect SigLIP and SD VAE to .ext_cache (no network) ──
        _ext_cache = MODELS_ROOT / ".ext_cache"
        _siglip_cache = str(_ext_cache / "siglip-so400m")
        _sd_vae_dir = str(_ext_cache / "sd-vae-v1-4")

        # Patch open_clip to use .ext_cache/siglip-so400m/ as cache_dir
        import open_clip as _oc

        _orig_create = _oc.create_model_and_transforms

        def _patched_create(*args, **kwargs):
            # Inject cache_dir so SigLIP weights are loaded from ext_cache
            kwargs.setdefault("cache_dir", _siglip_cache)
            return _orig_create(*args, **kwargs)

        _oc.create_model_and_transforms = _patched_create

        # Patch StableDiffusionPipeline.from_pretrained to load SD VAE
        # directly from .ext_cache/sd-vae-v1-4/ (flat directory, not HF cache)
        try:
            import types

            from diffusers import AutoencoderKL, StableDiffusionPipeline

            def _patched_sd_from(cls, pretrained_name, *a, **kw):
                if "CompVis/stable-diffusion-v1-4" in str(pretrained_name):
                    # Load VAE directly from local flat directory
                    vae = AutoencoderKL.from_pretrained(
                        _sd_vae_dir,
                        local_files_only=True,
                    )
                    # Return a minimal object with .vae attribute
                    result = types.SimpleNamespace(vae=vae)
                    return result
                from diffusers import StableDiffusionPipeline as _SDP

                return type.__call__(
                    _SDP.from_pretrained,
                    pretrained_name,
                    *a,
                    **kw,
                )

            StableDiffusionPipeline.from_pretrained = classmethod(
                _patched_sd_from,
            )
        except ImportError:
            pass

        # Ensure weights are present (download from HF if needed)
        weights_subdir = _ensure_weights_downloaded(cospy_weights)

        semantic_path = str(weights_subdir / _SEMANTIC_CKPT)
        artifact_path = str(weights_subdir / _ARTIFACT_CKPT)
        fusion_path = str(weights_subdir / _FUSION_CKPT)

        # Load the detector using spec_from_file_location under a unique
        # module name to avoid collisions with other models' 'detectors'.
        import importlib.util
        import sys as _sys

        from model_loader import _clean_conflicting_modules

        # The sd-v1_4 package has __init__.py that imports from siblings
        det_pkg_path = code_path / "detectors" / "sd-v1_4"
        init_file = det_pkg_path / "__init__.py"

        # Clean generic module names so CO-SPY's 'utils' resolves correctly
        _clean_conflicting_modules()

        code_str = str(code_path)
        saved_path = _sys.path.copy()
        if code_str not in _sys.path:
            _sys.path.insert(0, code_str)

        # Use importlib to handle the hyphenated module name
        spec = importlib.util.spec_from_file_location(
            "_ds_cospy_detector",
            str(init_file),
            submodule_search_locations=[str(det_pkg_path)],
        )
        detector_module = importlib.util.module_from_spec(spec)
        _sys.modules["_ds_cospy_detector"] = detector_module
        spec.loader.exec_module(detector_module)
        CoSpyFusionDetector = detector_module.CoSpyFusionDetector

        _sys.path[:] = saved_path

        cospy_model = CoSpyFusionDetector(
            semantic_weights_path=semantic_path,
            artifact_weights_path=artifact_path,
        )
        cospy_model.load_weights(fusion_path)
        cospy_model.to(device)
        cospy_model.train(mode=False)

        self._model = cospy_model
        self._loaded = True
        logger.info("CO-SPY model loaded on %s", device)

    def predict(self, raw_bytes: bytes) -> dict:
        """Run inference on raw image bytes."""
        return self._timed_predict(self._run, raw_bytes)

    @torch.inference_mode()
    def _run(self, raw_bytes: bytes) -> dict:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        tensor = _TRANSFORM(image).unsqueeze(0).to(self._device)

        output = self._model(tensor)  # raw logit [B, 1]
        probability = output.sigmoid().item()

        return {
            "probability": float(probability),
            "prediction": "fake" if probability >= 0.5 else "real",
        }
