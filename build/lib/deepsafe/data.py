"""Access to the DeepSafe evaluation dataset and stored prediction matrices."""

from __future__ import annotations

import csv
import gzip
import json
import pathlib
from typing import Iterator, Optional

HF_DATASET = "deepsafe/evaluation-dataset"

TIERS = {
    "small": {"dir": "master_eval_small", "samples": 198, "size": "1.7 GB"},
    "medium": {"dir": "master_eval", "samples": 15454, "size": "10 GB"},
    "full": {"dir": "master_eval_full", "samples": 45954, "size": "25 GB"},
}

# Bundled with the package so the published benchmark can be re-derived from a
# plain `pip install`, without downloading 10 GB of media. Falls back to the
# repository copy when running from a source checkout.
_BUNDLED = pathlib.Path(__file__).resolve().parent / "assets" / "predictions_medium.csv.gz"
_IN_REPO = (
    pathlib.Path(__file__).resolve().parents[2]
    / "eval" / "results" / "predictions_medium.csv.gz"
)
PREDICTIONS = _BUNDLED if _BUNDLED.exists() else _IN_REPO


class DatasetNotFound(RuntimeError):
    """Raised when a tier is not present locally and cannot be fetched."""


def tier_dir(tier: str, root: pathlib.Path) -> pathlib.Path:
    """Return the directory for ``tier`` under ``root``.

    Raises:
        KeyError: If ``tier`` is not one of small, medium, or full.
    """
    if tier not in TIERS:
        raise KeyError(f"unknown tier {tier!r}; expected one of {sorted(TIERS)}")
    return root / TIERS[tier]["dir"]


def download_tier(tier: str, root: pathlib.Path) -> pathlib.Path:
    """Fetch a dataset tier from HuggingFace into ``root``.

    The medium and full tiers are gated. A user who has not accepted the terms
    gets a clear instruction rather than an opaque 403.

    Raises:
        DatasetNotFound: If huggingface_hub is missing or access is denied.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DatasetNotFound(
            "huggingface_hub is required to download the dataset:\n"
            "    uv pip install huggingface_hub"
        ) from exc

    pattern = f"{TIERS[tier]['dir']}/**"
    try:
        snapshot_download(
            HF_DATASET,
            repo_type="dataset",
            allow_patterns=pattern,
            local_dir=str(root),
        )
    except Exception as exc:  # pragma: no cover - network dependent
        raise DatasetNotFound(
            f"could not download the {tier} tier.\n"
            f"The medium and full tiers are gated. Accept the terms at\n"
            f"    https://huggingface.co/datasets/{HF_DATASET}\n"
            f"then run `hf auth login`.\n"
            f"Underlying error: {exc}"
        ) from exc
    return tier_dir(tier, root)


def load_manifest(tier: str, root: pathlib.Path) -> list[dict]:
    """Load a tier's ``metadata.json``.

    Raises:
        DatasetNotFound: If the manifest is absent.
    """
    manifest = tier_dir(tier, root) / "metadata.json"
    if not manifest.exists():
        raise DatasetNotFound(
            f"no manifest at {manifest}.\n"
            f"Fetch it with: deepsafe eval --download --tier {tier}"
        )
    with manifest.open() as handle:
        return json.load(handle)


def load_predictions(path: Optional[pathlib.Path] = None) -> list[dict]:
    """Load the stored prediction matrix shipped with the repository.

    Raises:
        DatasetNotFound: If the file is missing.
    """
    path = path or PREDICTIONS
    if not path.exists():
        raise DatasetNotFound(f"no prediction matrix at {path}")
    with gzip.open(path, "rt") as handle:
        return list(csv.DictReader(handle))


def iter_samples(tier: str, root: pathlib.Path) -> Iterator[tuple[pathlib.Path, dict]]:
    """Yield ``(path, record)`` for every sample in a tier.

    Samples whose media file is missing are skipped rather than raising, so a
    partial download still produces a usable, clearly-sized evaluation.
    """
    base = tier_dir(tier, root)
    for record in load_manifest(tier, root):
        media = base / record["path"]
        if media.exists():
            yield media, record


def as_float(row: dict, column: str) -> Optional[float]:
    """Return ``row[column]`` as a float, or None when absent or unparsable."""
    try:
        return float(row.get(column, ""))
    except (TypeError, ValueError):
        return None
