"""FSD (Forensic Self-Descriptions, CVPR 2025) wrapper.

FSD uses a custom FSDDetector loaded from the ``fsd`` package in the
model_code directory.  The detector's .score() method returns a z-score
which is converted to [0, 1] fake probability via a sigmoid centred at
the paper's default threshold z=-2.0.

Performance note: FSD is inherently slow (~5s) because it solves a
constrained least-squares system in float64 over ~1.4M patches.  The
FRE convolution runs in float32 and is upcast to float64 before the
solver (saves ~800ms with identical accuracy).

"""

import io
import json
import logging
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from config import get_model_code_path, get_weights_path
from model_loader import namespaced_import
from models.base import BasePredictor
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger("models.fsd")

# Z-score conversion constants (matches the service)
_THRESHOLD_Z: float = -2.0
_SCALE: float = 1.5


def _z_score_to_probability(z_score: float) -> float:
    """Map an FSD z-score to a [0, 1] fake probability via sigmoid.

    At z=_THRESHOLD_Z the output is 0.5.
    """
    exponent = -(_THRESHOLD_Z - z_score) * _SCALE
    exponent = max(-500.0, min(500.0, exponent))
    return 1.0 / (1.0 + math.exp(exponent))


class FSDPredictor(BasePredictor):
    """Forensic Self-Descriptions wrapper with optimised FRE step."""

    name = "fsd"
    modality = "image"
    _use_amp = False  # FSD uses custom z-score ops that are slower under AMP

    def __init__(self):
        self._fre = None
        self._gmm = None
        self._projections = None
        self._fsd_cfg = None
        self._scoring_cfg = None
        self._proj_mod = None
        # Pre-computed masks (set during load)
        self._mask = None
        self._K = 0
        self._n_features = 0
        self._total_features = 0
        self._center = 0
        self._B = 0

    def load(self, weights_dir: Path, device: torch.device) -> None:
        """Load FSD components (FRE, GMM, projections) directly."""
        self._device = device

        code_path = get_model_code_path("fsd")
        fre_mod = namespaced_import(
            "fsd.fre",
            code_path,
            namespace="_ds_fsd",
        )
        gmm_mod = namespaced_import(
            "fsd.gmm",
            code_path,
            namespace="_ds_fsd",
        )
        proj_mod = namespaced_import(
            "fsd.projection",
            code_path,
            namespace="_ds_fsd",
        )
        self._proj_mod = proj_mod

        fsd_weights = get_weights_path("fsd")
        with open(fsd_weights / "config.json") as f:
            config = json.load(f)

        self._fsd_cfg = config["fsd"]
        self._scoring_cfg = config["scoring"]

        device_str = str(device)
        self._fre = fre_mod.FRE.from_pretrained(
            fsd_weights / config["fre"]["weights_file"],
            device=device_str,
        )
        self._gmm = gmm_mod.load_gmm(
            fsd_weights / config["gmm"]["weights_file"],
            device=device_str,
        )
        self._projections = proj_mod.load_transforms(
            fsd_weights / config["transforms"]["weights_file"],
            device=device_str,
        )

        # Pre-compute constants
        self._K = self._fre.conv.out_channels
        self._B = self._fsd_cfg["kernel_size"]
        self._center = self._B // 2
        self._mask = torch.ones(
            self._B,
            self._B,
            dtype=torch.bool,
            device=device,
        )
        self._mask[self._center, self._center] = False
        self._n_features = self._mask.sum().item()
        self._total_features = self._K * self._n_features

        # Pre-build the constraint matrix A (constant across calls)
        self._A = torch.zeros(
            self._K,
            self._total_features,
            device=device,
            dtype=torch.float64,
        )
        for k in range(self._K):
            self._A[k, k * self._n_features : (k + 1) * self._n_features] = 1
        self._A_bottom = torch.cat(
            [
                self._A,
                torch.zeros(
                    self._K,
                    self._K,
                    device=device,
                    dtype=torch.float64,
                ),
            ],
            dim=1,
        )
        self._b = torch.ones(self._K, device=device, dtype=torch.float64)
        self._eye = torch.eye(
            self._total_features,
            device=device,
            dtype=torch.float64,
        )

        # Note: TF32 must be disabled during FSD inference (not globally).
        # This is handled per-call in _run() to avoid hurting other models.

        self._loaded = True
        logger.info("FSD detector loaded on %s", device)

    def predict(self, raw_bytes: bytes) -> dict:
        """Run inference on raw image bytes."""
        return self._timed_predict(self._run, raw_bytes)

    @torch.inference_mode()
    def _run(self, raw_bytes: bytes) -> dict:
        # Disable TF32 for this call (FSD needs float64 precision).
        # Scoped here rather than globally to avoid slowing other models.
        prev_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        prev_tf32_cudnn = torch.backends.cudnn.allow_tf32
        if self._device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        try:
            return self._run_inner(raw_bytes)
        finally:
            if self._device.type == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = prev_tf32_matmul
                torch.backends.cudnn.allow_tf32 = prev_tf32_cudnn

    def _run_inner(self, raw_bytes: bytes) -> dict:
        image = Image.open(io.BytesIO(raw_bytes)).convert("L")
        # FRE in float32 (11× faster than float64, identical accuracy)
        image_t = (
            torch.from_numpy(np.array(image)).float().unsqueeze(0).to(self._device)
        )

        K = self._K
        B = self._B
        center = self._center
        mask = self._mask
        total_features = self._total_features
        num_scales = self._fsd_cfg["num_scales"]
        max_size = self._fsd_cfg["max_size"]
        device = self._device

        # 1. FRE in float32 then upcast to float64
        residuals = self._fre(image_t)
        border = self._fre.conv.kernel_size // 2
        residuals = residuals[:, border:-border, border:-border].double()

        # 2. Resize and crop
        h, w = residuals.shape[-2:]
        sf = max_size / min(h, w)
        nh, nw = round(h * sf), round(w * sf)
        r = F.interpolate(
            residuals[None],
            size=(nh, nw),
            mode="bilinear",
            antialias=False,
            align_corners=False,
        )[0]
        ch = min(max_size, r.shape[-2])
        cw = min(max_size, r.shape[-1])
        sh = (r.shape[-2] - ch) // 2
        sw = (r.shape[-1] - cw) // 2
        r = r[:, sh : sh + ch, sw : sw + cw]

        # 3. Multi-scale patch extraction + XTX accumulation (fused)
        XTX = torch.zeros(
            total_features,
            total_features,
            dtype=torch.float64,
            device=device,
        )
        XTy = torch.zeros(total_features, dtype=torch.float64, device=device)

        for l in range(num_scales):
            sc = F.interpolate(
                r[None],
                scale_factor=1 / 2**l,
                mode="bilinear",
                antialias=False,
                align_corners=False,
            )
            sc = F.pad(sc, (B // 2, B // 2, B // 2, B // 2), mode="reflect")
            uf = (
                sc[0]
                .unfold(1, B, 1)
                .unfold(2, B, 1)
                .reshape(K, -1, B, B)
                .permute(1, 0, 2, 3)
            )
            x = uf[:, :, mask].reshape(-1, total_features).to(torch.float64)
            y = uf[:, :, center, center].sum(dim=1).to(torch.float64)
            XTX += x.T @ x
            XTy += x.T @ y

        # 4. KKT solve (reuse pre-built constraint matrix)
        XTX += 1e-5 * self._eye
        LHS = torch.cat(
            [
                torch.cat([XTX, self._A.T], dim=1),
                self._A_bottom,
            ],
            dim=0,
        )
        RHS = torch.cat([XTy, self._b], dim=0)
        solution = torch.linalg.solve(LHS, RHS)

        # 5. Projection + GMM scoring
        fsd_vec = solution[:total_features].unsqueeze(0)
        fsd_proj = self._proj_mod.apply_projections(
            fsd_vec,
            self._projections,
        )
        raw_score = self._gmm.score_samples(fsd_proj).item()
        z_score = (raw_score - self._scoring_cfg["train_mean"]) / self._scoring_cfg[
            "train_std"
        ]

        probability = _z_score_to_probability(z_score)

        return {
            "probability": float(probability),
            "prediction": "fake" if probability >= 0.5 else "real",
        }
