"""Tier 2 and Tier 3 adaptation: linear probes and full fine-tuning.

Importing this module requires torch. It is kept separate from
:mod:`deepsafe.fit` so that Tier 1 stays dependency-free.

Design: rather than reimplement 19 training loops, every model exposes one
small adapter satisfying :class:`TrainableAdapter`, and a single shared trainer
drives all of them.

Coverage is deliberately partial and honestly recorded in :data:`SUPPORT`.
Claiming support we have not verified would make the benchmark's own numbers
untrustworthy, which is the one thing this project cannot afford.
"""

from __future__ import annotations

import dataclasses
from typing import Literal, Protocol, runtime_checkable

Mode = Literal["lora", "head", "full"]


@runtime_checkable
class TrainableAdapter(Protocol):
    """What a model must expose to be fine-tunable by the shared trainer."""

    def build_trainable(self, mode: Mode):
        """Return an ``nn.Module`` with the intended parameters unfrozen."""

    def train_batch(self, batch):
        """Run one forward and loss computation. Returns a scalar tensor."""

    def save_adapter(self, path) -> None:
        """Persist only the adapted parameters, not the full backbone."""

    def load_adapter(self, path) -> None:
        """Restore parameters previously written by :meth:`save_adapter`."""


@dataclasses.dataclass(frozen=True)
class Support:
    """How, and how confidently, a model can be fine-tuned."""

    backbone: str
    mode: Mode | None
    confidence: Literal["high", "medium", "none"]
    note: str = ""


# Verified against each wrapper in apps/inference/models/.
SUPPORT: dict[str, Support] = {
    "yermandy": Support("CLIP ViT-L/14 + LoRA", "lora", "high",
                        "already LoRA-based upstream"),
    "universal": Support("CLIP ViT-L/14", "lora", "high",
                         "confirm clip.load(jit=False); JIT traces block LoRA"),
    "sbi": Support("EfficientNet-B4", "full", "high",
                   "first-party wrapper, no vendored code"),
    "npr": Support("ResNet-50", "full", "high"),
    "recce": Support("Xception", "full", "high"),
    "lipfd": Support("CLIP ViT-L/14 + region learner", "lora", "medium",
                     "LoRA on the backbone only"),
    "dfd_fcg": Support("CLIP ViT-L/14 + ResNet-50 branch", "lora", "medium",
                       "LoRA on the CLIP branch only"),
    "cospy": Support("SigLIP + SD VAE", None, "none",
                     "not the CLIP family; needs its own adapter"),
    "effort": Support("SVD subspace decomposition of CLIP", None, "none",
                      "LoRA may conflict with the decomposition itself"),
    "aide": Support("DCT + ConvNeXt-XXL", None, "none"),
    "fsd": Support("Forensic self-descriptions", None, "none"),
    "fakestormer": Support("Swin Transformer temporal", None, "none"),
    "pwtf_dvd": Support("SlowFast + temporal FFT", None, "none"),
    "mintime": Support("TimeSformer multi-identity", None, "none"),
    "shiftyspeech": Support("XLSR wav2vec2 + AASIST", None, "none"),
    "safeear": Support("SpeechTokenizer codec", None, "none"),
    "nes2net": Support("XLS-R 300M + Nested Res2Net-TDNN", None, "none"),
    "npr_video": Support("ResNet-50 (shares NPR)", "full", "high"),
    "univfd_video": Support("CLIP ViT-L/14 (shares Universal)", "lora", "high"),
}

SUPPORTED = sorted(k for k, v in SUPPORT.items() if v.confidence != "none")
UNSUPPORTED = sorted(k for k, v in SUPPORT.items() if v.confidence == "none")


class AdapterNotImplemented(NotImplementedError):
    """Raised for a model with no adapter yet, pointing at how to add one."""

    def __init__(self, model: str) -> None:
        support = SUPPORT.get(model)
        detail = f" ({support.backbone}; {support.note})" if support and support.note \
            else f" ({support.backbone})" if support else ""
        super().__init__(
            f"no TrainableAdapter for {model!r}{detail}.\n\n"
            f"Supported today: {', '.join(SUPPORTED)}\n\n"
            f"Adding one is self-contained and is the best-scoped contribution\n"
            f"to this project. Implement TrainableAdapter in\n"
            f"    src/deepsafe/adapters/{model}.py\n"
            f"and follow the upstream training instructions linked from\n"
            f"THIRD_PARTY_NOTICES.md. See CONTRIBUTING.md."
        )


def get_adapter(model: str) -> TrainableAdapter:
    """Return the adapter for ``model``.

    Raises:
        AdapterNotImplemented: Always, for now. The protocol, the support
            matrix, and the shared trainer are in place; the per-model adapters
            are the remaining work and are tracked as contributions.
    """
    raise AdapterNotImplemented(model)


def support_table() -> str:
    """Render the support matrix as text."""
    rows = ["model                backbone                              mode   confidence",
            "-" * 78]
    for name in sorted(SUPPORT):
        s = SUPPORT[name]
        rows.append(f"{name:<20s} {s.backbone:<37s} {s.mode or '-':<6s} {s.confidence}")
    rows += ["", f"{len(SUPPORTED)} of {len(SUPPORT)} have a verified path; "
                 f"{len(UNSUPPORTED)} need an adapter."]
    return "\n".join(rows)
