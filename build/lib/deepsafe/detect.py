"""The ``deepsafe detect`` verb: run the model ensemble over a file.

Detection needs the models loaded, which means ~20 GB of VRAM and the full
inference stack. Rather than pretend otherwise, this module talks to a running
inference server over HTTP and gives a precise error when one is not reachable.

Start a server with:

    cd apps/inference && PYTHONPATH=.:../../packages/shared python server.py
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import urllib.error
import urllib.request
from typing import Optional

DEFAULT_SERVER = "http://localhost:8000"
TIMEOUT_SECONDS = 600

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class InferenceUnavailable(RuntimeError):
    """Raised when the inference server cannot be reached."""


@dataclasses.dataclass
class Verdict:
    """The ensemble's answer for one file."""

    path: pathlib.Path
    modality: str
    score: float
    verdict: str
    per_model: dict[str, float]

    def render(self) -> str:
        """Render the verdict, with the per-model breakdown and a caveat."""
        lines = [
            f"{self.path.name}",
            f"  modality   {self.modality}",
            f"  verdict    {self.verdict.upper()}  (score {self.score:.4f})",
        ]
        if self.per_model:
            lines.append("")
            lines.append("  per-model:")
            for name, value in sorted(
                self.per_model.items(), key=lambda kv: -kv[1]
            ):
                lines.append(f"    {name:<22s} {value:.4f}")
        lines += [
            "",
            "  A score is an estimate, not evidence. Detection degrades sharply",
            "  on generators absent from training; see BENCHMARK.md.",
        ]
        return "\n".join(lines)


def infer_modality(path: pathlib.Path) -> str:
    """Infer modality from a file suffix.

    Raises:
        ValueError: If the suffix is not recognised.
    """
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "images"
    if suffix in AUDIO_SUFFIXES:
        return "audio"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    raise ValueError(
        f"unrecognised media type {suffix!r}. Supported: "
        f"{', '.join(sorted(IMAGE_SUFFIXES | AUDIO_SUFFIXES | VIDEO_SUFFIXES))}"
    )


def server_health(server: str = DEFAULT_SERVER) -> dict:
    """Query the inference server's health endpoint.

    Raises:
        InferenceUnavailable: If the server is unreachable.
    """
    try:
        with urllib.request.urlopen(f"{server}/health", timeout=10) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise InferenceUnavailable(
            f"no inference server at {server}.\n\n"
            f"Start one with:\n"
            f"    cd apps/inference && PYTHONPATH=.:../../packages/shared "
            f"python server.py\n\n"
            f"The full lineup needs roughly 20 GB of VRAM. Run a subset with\n"
            f"    DEEPSAFE_MODELS=npr,aide python server.py\n\n"
            f"Underlying error: {exc}"
        ) from exc


def detect(
    path: pathlib.Path,
    *,
    server: str = DEFAULT_SERVER,
    modality: Optional[str] = None,
) -> Verdict:
    """Run detection on one file via the inference server.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        InferenceUnavailable: If the server is unreachable or errors.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")

    modality = modality or infer_modality(path)
    server_health(server)  # fail fast with a useful message

    boundary = "----deepsafe-boundary"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; '
        f'filename="{path.name}"\r\n'.encode(),
        b"Content-Type: application/octet-stream\r\n\r\n",
        path.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    request = urllib.request.Request(
        f"{server}/predict/{modality}",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise InferenceUnavailable(f"inference request failed: {exc}") from exc

    score = float(payload.get("ensemble_score", payload.get("score", 0.0)))
    return Verdict(
        path=path,
        modality=modality,
        score=score,
        verdict=payload.get("verdict") or ("fake" if score >= 0.5 else "real"),
        per_model={
            k: float(v)
            for k, v in (payload.get("per_model") or {}).items()
            if isinstance(v, (int, float))
        },
    )
