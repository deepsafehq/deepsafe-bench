"""NPR (Neural Pattern Residual) image deepfake detector wrapper.

Loads a resnet50 (1-output) from the NPR-DeepfakeDetection model code,
applies sigmoid to the raw logit to produce a fake probability.

"""

import io
import logging
from pathlib import Path

import torch
import torchvision.transforms as transforms
from config import get_model_code_path, get_weights_path
from model_loader import namespaced_import
from models.base import BasePredictor
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger("models.npr")

# ImageNet normalisation (matches the service)
_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


class NPRPredictor(BasePredictor):
    """NPR-DeepfakeDetection wrapper."""

    name = "npr"
    modality = "image"
    _use_amp = False  # Small ResNet50, no benefit from AMP

    def __init__(self):
        self._model = None

    def load(self, weights_dir: Path, device: torch.device) -> None:
        """Load resnet50(num_classes=1) with NPR.pth weights."""
        self._device = device

        code_path = get_model_code_path("npr")
        resnet_mod = namespaced_import(
            "networks.resnet", code_path, namespace="_ds_npr"
        )
        resnet50 = resnet_mod.resnet50

        self._model = resnet50(num_classes=1)

        weights_file = get_weights_path("npr") / "NPR.pth"
        state_dict = torch.load(
            weights_file,
            map_location=device,
            weights_only=True,
        )
        # Strip DataParallel "module." prefix if present
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        self._model.load_state_dict(state_dict)
        self._model.to(device)
        self._model.train(mode=False)
        self._loaded = True
        logger.info("NPR model loaded on %s", device)

    def predict(self, raw_bytes: bytes) -> dict:
        """Run inference on raw image bytes."""
        return self._timed_predict(self._run, raw_bytes)

    @torch.inference_mode()
    def _run(self, raw_bytes: bytes) -> dict:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        tensor = _TRANSFORM(image).unsqueeze(0).to(self._device)

        logit = self._model(tensor)
        probability = torch.sigmoid(logit).item()

        return {
            "probability": float(probability),
            "prediction": "fake" if probability >= 0.5 else "real",
        }
