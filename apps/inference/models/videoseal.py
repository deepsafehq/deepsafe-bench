"""VideoSeal/PixelSeal watermark provenance detector wrapper.

Detects Meta VideoSeal watermarks embedded in images and videos using
the videoseal_base model. GPU-capable.

"""

import io
import logging
import os
import pathlib
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from config import MODELS_ROOT
from models.base import BasePredictor
from PIL import Image

logger = logging.getLogger("models.videoseal")

MAX_IMAGE_DIMENSION = 4096
DETECTION_THRESHOLD = 0.6
MAX_FRAMES = 16


def _map_confidence_to_probability(confidence: float) -> float:
    """Map VideoSeal detection confidence to a probability score.

    Confidence values below the detection threshold are treated as
    noise and mapped to 0.5 (neutral). Values at or above the
    threshold are linearly scaled into [0.5, 1.0].

    Args:
        confidence: Raw detection confidence from VideoSeal in [0, 1].

    Returns:
        Probability in [0.5, 1.0]. 0.5 means neutral (no watermark).
    """
    if confidence < DETECTION_THRESHOLD:
        return 0.5
    return 0.5 + confidence * 0.5


class VideoSealPredictor(BasePredictor):
    """VideoSeal/PixelSeal watermark detector wrapper.

    Loads Meta's videoseal_base model and runs watermark detection
    on images and videos.
    """

    name = "videoseal"
    modality = "provenance"

    def __init__(self):
        self._model = None

    def load(self, weights_dir: Path, device: torch.device) -> None:
        """Load VideoSeal detector model onto the given device."""
        self._device = device

        import omegaconf
        import videoseal

        cards_dir = pathlib.Path(videoseal.__file__).parent / "cards"
        card_path = cards_dir / "videoseal_1.0.yaml"

        # Patch the card in-place to disable attenuation AND redirect
        # checkpoint_path to .ext_cache/videoseal/y_256b_img.pth so
        # videoseal loads locally instead of downloading from Meta CDN.
        card = omegaconf.OmegaConf.load(card_path)
        card.args.attenuation = "none"
        local_ckpt = MODELS_ROOT / ".ext_cache" / "videoseal" / "y_256b_img.pth"
        if local_ckpt.exists():
            card.checkpoint_path = str(local_ckpt)
        omegaconf.OmegaConf.save(card, card_path)

        self._model = videoseal.load(card_path)
        self._model = self._model.to(device)
        self._model.train(mode=False)
        self._loaded = True
        logger.info("VideoSeal detector loaded on %s", device)

    def predict(self, raw_bytes: bytes) -> dict:
        """Detect VideoSeal watermark in raw media bytes.

        Attempts image detection first. If the bytes cannot be decoded
        as an image, falls back to video detection.
        """
        return self._timed_predict(self._run, raw_bytes)

    @torch.inference_mode()
    def _run(self, raw_bytes: bytes) -> dict:
        """Run VideoSeal detection on raw media bytes."""
        # Try image first, fall back to video
        confidence = self._try_detect_image(raw_bytes)
        if confidence is None:
            confidence = self._detect_video(raw_bytes)

        probability = _map_confidence_to_probability(confidence)

        return {
            "probability": float(probability),
            "prediction": "fake" if probability > 0.5 else "real",
        }

    def _try_detect_image(self, image_bytes: bytes) -> float:
        """Attempt to detect VideoSeal watermark in image bytes.

        Args:
            image_bytes: Raw bytes of a potential image file.

        Returns:
            Confidence float in [0, 1], or None if not a valid image.
        """
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            if (
                pil_image.width > MAX_IMAGE_DIMENSION
                or pil_image.height > MAX_IMAGE_DIMENSION
            ):
                logger.warning(
                    "Image too large: %dx%d, max %d",
                    pil_image.width,
                    pil_image.height,
                    MAX_IMAGE_DIMENSION,
                )
                return 0.0
            pil_image = pil_image.convert("RGB")
        except Exception:
            return None

        tensor = TF.to_tensor(pil_image).unsqueeze(0).to(self._device)

        outputs = self._model.detect(tensor)

        preds = outputs["preds"]
        bit_confs = torch.sigmoid(preds[0, 1:])
        confident_bits = ((bit_confs > 0.8) | (bit_confs < 0.2)).float().mean()
        return float(confident_bits)

    def _detect_video(self, video_bytes: bytes) -> float:
        """Detect VideoSeal watermark in video bytes.

        Writes bytes to a temporary file, extracts up to MAX_FRAMES
        evenly sampled frames via OpenCV, runs detection on each,
        and returns the maximum confidence.

        Args:
            video_bytes: Raw bytes of a video file.

        Returns:
            Maximum confidence float across frames in [0, 1].
        """
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(video_bytes)
                tmp_path = tmp.name

            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                logger.warning("Failed to open video file.")
                return 0.0

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                return 0.0

            if total_frames <= MAX_FRAMES:
                indices = list(range(total_frames))
            else:
                indices = np.linspace(
                    0, total_frames - 1, MAX_FRAMES, dtype=int
                ).tolist()

            max_confidence = 0.0

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb_frame)
                tensor = TF.to_tensor(pil_frame).unsqueeze(0).to(self._device)

                outputs = self._model.detect(tensor)

                preds = outputs["preds"]
                bit_confs = torch.sigmoid(preds[0, 1:])
                confidence = float(
                    ((bit_confs > 0.8) | (bit_confs < 0.2)).float().mean()
                )
                if confidence > max_confidence:
                    max_confidence = confidence

            cap.release()
            return max_confidence

        except Exception as exc:
            logger.warning("Video detection error: %s", exc)
            return 0.0

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
