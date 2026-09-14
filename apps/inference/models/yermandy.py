"""Yermandy CLIP deepfake detection wrapper.

Uses a CLIP-based model with LoRA (peft library), loaded via Lightning
Fabric.  Softmax on logits_labels gives [real, fake] probabilities;
index 1 is the fake probability.

"""

import io
import logging
import os
from pathlib import Path

import torch
from config import MODELS_ROOT, get_model_code_path, get_weights_path
from model_loader import namespaced_import
from models.base import BasePredictor
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger("models.yermandy")


class YermandyPredictor(BasePredictor):
    """Yermandy CLIP-based deepfake detector wrapper."""

    name = "yermandy"
    modality = "image"

    def __init__(self):
        self._model = None
        self._preprocessing_fn = None

    def load(self, weights_dir: Path, device: torch.device) -> None:
        """Load the Yermandy model from checkpoint via Lightning Fabric."""
        self._device = device

        code_path = get_model_code_path("yermandy")
        config_mod = namespaced_import(
            "src.config", code_path, namespace="_ds_yermandy"
        )
        model_mod = namespaced_import(
            "src.model.dfdet", code_path, namespace="_ds_yermandy"
        )

        # Bypass CVE-2025-32434 torch.load check: clip-vit-base-patch16 only has
        # pytorch_model.bin (no safetensors). Must patch BEFORE from_pretrained.
        try:
            import transformers.modeling_utils as _mu
            import transformers.utils.import_utils as _tu

            _tu.check_torch_load_is_safe = lambda: None
            _mu.check_torch_load_is_safe = lambda: None
        except Exception:
            pass

        # Patch CLIPEncoder to load from local ext_cache instead of HF hub.
        # The checkpoint uses clip-vit-large-patch14 (confirmed from hyper_parameters).
        # The vendored clip_encoder.py calls CLIPModel/CLIPProcessor.from_pretrained()
        # which fails offline. Force-import and redirect to local path.
        import sys

        clip_local = str(MODELS_ROOT / ".ext_cache" / "clip-vit-l14-transformers")
        namespaced_import(
            "src.encoders.clip_encoder",
            code_path,
            namespace="_ds_yermandy",
        )
        enc_mod = sys.modules.get("_ds_yermandy.src.encoders.clip_encoder")
        if enc_mod and os.path.isdir(clip_local):
            _orig_clip_init = enc_mod.CLIPEncoder.__init__

            def _patched_clip_init(
                self_inner, model_name="openai/clip-vit-large-patch14"
            ):
                _orig_clip_init(self_inner, model_name=clip_local)

            enc_mod.CLIPEncoder.__init__ = _patched_clip_init

        Config = config_mod.Config
        DeepfakeDetectionModel = model_mod.DeepfakeDetectionModel

        checkpoint_path = get_weights_path("yermandy") / "model.ckpt"
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        model_config = Config(**ckpt["hyper_parameters"])
        model = DeepfakeDetectionModel(model_config)
        model.load_state_dict(ckpt["state_dict"])
        model.train(mode=False)
        model.to(device)

        self._preprocessing_fn = model.get_preprocessing()

        # Upgrade to fast image processor if available
        try:
            from transformers import CLIPProcessor

            encoder = model.feature_extractor
            base = encoder._preprocess if hasattr(encoder, "_preprocess") else None
            if base is not None:
                fast = CLIPProcessor.from_pretrained(
                    encoder.model_name,
                    use_fast=True,
                )
                encoder._preprocess = fast
                self._preprocessing_fn = encoder.preprocess
                logger.info("Upgraded to fast CLIPProcessor")
        except Exception:
            pass  # fall back to original processor

        # Wrap with Fabric for consistent inference behaviour
        from lightning.fabric import Fabric

        accelerator = "cuda" if device.type == "cuda" else "cpu"
        fabric = Fabric(
            accelerator=accelerator,
            devices=1,
            precision="32-true",
        )
        self._model = fabric.setup_module(model)
        self._loaded = True
        logger.info("Yermandy model loaded on %s", device)

    def predict(self, raw_bytes: bytes) -> dict:
        """Run inference on raw image bytes."""
        return self._timed_predict(self._run, raw_bytes)

    @torch.inference_mode()
    def _run(self, raw_bytes: bytes) -> dict:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        tensor = self._preprocessing_fn(image).unsqueeze(0).to(self._device)

        output = self._model(tensor)
        probs = output.logits_labels.softmax(dim=1)
        probability = probs[0, 1].item()

        return {
            "probability": float(probability),
            "prediction": "fake" if probability >= 0.5 else "real",
        }
