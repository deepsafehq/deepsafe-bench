"""AIDE (ICLR 2025) image deepfake detector wrapper.

DCT frequency decomposition + ConvNeXt-xxlarge.  Produces a 5-view input
tensor [1, 5, 3, 256, 256].  Softmax on logits gives [real, fake]
probabilities; index 1 is the fake probability.

Requires isolation: the model_code directory uses generic module names
(``models``, ``data``) that conflict with other packages.

"""

import io
import logging
import sys
import types
from pathlib import Path

import torch
import torchvision.transforms as transforms
from config import MODEL_REGISTRY, get_model_code_path, get_weights_path
from model_loader import namespaced_import_multi
from models.base import BasePredictor
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger("models.aide")


class AIDEPredictor(BasePredictor):
    """AIDE deepfake detector wrapper (needs_isolation=True)."""

    name = "aide"
    modality = "image"

    def __init__(self):
        self._model = None
        self._dct_module_cls = None

    def load(self, weights_dir: Path, device: torch.device) -> None:
        """Load AIDE model with isolated imports."""
        self._device = device
        code_path = get_model_code_path("aide")

        # Inject clip stub before importing AIDE (it imports clip at
        # module level but only uses open_clip at inference time).
        if "clip" not in sys.modules:
            sys.modules["clip"] = types.ModuleType("clip")

        # The model code needs both the root model_code dir and its
        # subdirectories on sys.path.
        data_path = code_path / "data"
        models_path = code_path / "models"
        extra_paths = [p for p in [data_path, models_path] if p.exists()]

        imported = namespaced_import_multi(
            code_path=code_path,
            namespace="_ds_aide",
            import_specs=[
                {"module": "models.AIDE", "attr": "AIDE"},
                {"module": "data.dct", "attr": "DCT_base_Rec_Module"},
            ],
            extra_paths=extra_paths,
        )

        AIDE_cls = imported["AIDE"]
        self._dct_module_cls = imported["DCT_base_Rec_Module"]

        aide_model = AIDE_cls(resnet_path=None, convnext_path=None)
        aide_model.to(device)

        # Load checkpoint
        checkpoint_name = MODEL_REGISTRY["aide"].extra.get(
            "checkpoint",
            "GenImage_train.pth",
        )
        checkpoint_path = get_weights_path("aide") / checkpoint_name
        if checkpoint_path.exists():
            ckpt = torch.load(checkpoint_path, map_location=device)
            if isinstance(ckpt, dict):
                state_dict = ckpt.get("model") or ckpt.get("state_dict") or ckpt
            else:
                state_dict = ckpt
            # Strip DataParallel "module." prefix
            cleaned = {k.removeprefix("module."): v for k, v in state_dict.items()}
            missing, unexpected = aide_model.load_state_dict(
                cleaned,
                strict=False,
            )
            logger.info(
                "AIDE checkpoint loaded. Missing: %d, Unexpected: %d",
                len(missing),
                len(unexpected),
            )
        else:
            logger.warning("No AIDE checkpoint at %s", checkpoint_path)

        aide_model.train(mode=False)
        self._model = aide_model
        self._loaded = True
        logger.info("AIDE model loaded on %s", device)

    def predict(self, raw_bytes: bytes) -> dict:
        """Run inference on raw image bytes."""
        return self._timed_predict(self._run, raw_bytes)

    @torch.inference_mode()
    def _run(self, raw_bytes: bytes) -> dict:
        tensor = self._preprocess(raw_bytes)
        logits = self._model(tensor)  # [1, 2]
        probs = torch.softmax(logits, dim=-1)
        probability = probs[0, 1].item()

        return {
            "probability": float(probability),
            "prediction": "fake" if probability >= 0.5 else "real",
        }

    def _preprocess(self, raw_bytes: bytes) -> torch.Tensor:
        """Build the 5-view DCT tensor [1, 5, 3, 256, 256]."""
        pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

        # Ensure minimum 256x256 for DCT unfold
        w, h = pil_image.size
        if w < 256 or h < 256:
            pil_image = pil_image.resize((256, 256), Image.BICUBIC)

        to_tensor = transforms.ToTensor()
        image_tensor = to_tensor(pil_image)  # [3, H, W]

        dct_module = self._dct_module_cls()
        x_minmin, x_maxmax, x_minmin1, x_maxmax1 = dct_module(
            image_tensor,
        )

        transform = transforms.Compose(
            [
                transforms.Resize([256, 256]),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        x_0 = transform(image_tensor)
        x_minmin = transform(x_minmin)
        x_maxmax = transform(x_maxmax)
        x_minmin1 = transform(x_minmin1)
        x_maxmax1 = transform(x_maxmax1)

        stacked = torch.stack(
            [x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0],
            dim=0,
        )
        return stacked.unsqueeze(0).to(self._device)
