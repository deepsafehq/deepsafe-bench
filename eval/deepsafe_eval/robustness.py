"""Robustness degradation attacks for deepfake detection evaluation.

Provides functions that apply realistic degradation to media files and
return the modified bytes. Used by ``run_robustness.py`` to measure how
model detection rates change under adverse conditions.

Image attacks: JPEG recompression, social-media simulation, Gaussian
noise, Gaussian blur.

Audio attacks: MP3 re-encoding via ffmpeg.

Video attacks: CRF-based re-encoding via ffmpeg.
"""

import io
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# --- Social Media Compression Profiles ---
# Each platform applies JPEG recompression + resize to simulate how
# platforms process uploaded images.  Quality levels and max dimensions
# are approximations of real-world behaviour.
SOCIAL_MEDIA_PROFILES: dict[str, dict] = {
    "instagram": {
        "quality": 70,
        "max_dimension": 1080,
        "description": "Instagram feed (1080px, Q70 JPEG)",
    },
    "twitter": {
        "quality": 85,
        "max_dimension": 4096,
        "description": "Twitter/X (4096px, Q85 JPEG)",
    },
    "whatsapp": {
        "quality": 50,
        "max_dimension": 1600,
        "description": "WhatsApp (1600px, Q50 JPEG, heavy compression)",
    },
    "facebook": {
        "quality": 75,
        "max_dimension": 2048,
        "description": "Facebook (2048px, Q75 JPEG)",
    },
    "telegram": {
        "quality": 80,
        "max_dimension": 2560,
        "description": "Telegram (2560px, Q80 JPEG)",
    },
}


def _load_image(image_path: str | Path) -> Image.Image:
    """Load an image from disk, converting to RGB.

    Args:
        image_path: Filesystem path to the source image.

    Returns:
        PIL Image in RGB mode.

    Raises:
        FileNotFoundError: If the image does not exist.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def jpeg_recompress(image_path: str | Path, quality: int = 50) -> bytes:
    """Re-encode an image as JPEG at the given quality.

    Args:
        image_path: Path to the source image.
        quality: JPEG quality (1-100).  Lower = more degradation.

    Returns:
        JPEG-encoded bytes.
    """
    img = _load_image(image_path)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def social_media_simulate(
    image_path: str | Path,
    platform: str = "instagram",
) -> bytes:
    """Simulate social-media upload compression.

    Resizes the image to the platform's max dimension (preserving aspect
    ratio) and re-encodes as JPEG at the platform's quality level.

    Args:
        image_path: Path to the source image.
        platform: One of 'instagram', 'twitter', 'whatsapp', 'facebook',
            'telegram'.

    Returns:
        JPEG-encoded bytes.

    Raises:
        ValueError: If the platform is unknown.
    """
    profile = SOCIAL_MEDIA_PROFILES.get(platform)
    if profile is None:
        valid = ", ".join(sorted(SOCIAL_MEDIA_PROFILES.keys()))
        raise ValueError(f"Unknown platform '{platform}'. Valid: {valid}")

    img = _load_image(image_path)
    max_dim = profile["max_dimension"]
    quality = profile["quality"]

    # Resize if larger than max_dimension (longest edge).
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def add_gaussian_noise(
    image_path: str | Path,
    sigma: float = 10.0,
) -> bytes:
    """Add Gaussian noise to an image.

    Args:
        image_path: Path to the source image.
        sigma: Standard deviation of the Gaussian noise (pixel range 0-255).

    Returns:
        JPEG-encoded bytes of the noisy image (quality 95, minimal
        additional compression artefacts).
    """
    img = _load_image(image_path)
    arr = np.array(img, dtype=np.float32)
    rng = np.random.RandomState(42)
    noise = rng.normal(0, sigma, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    noisy_img = Image.fromarray(noisy)
    buf = io.BytesIO()
    noisy_img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def gaussian_blur(
    image_path: str | Path,
    kernel_size: int = 5,
) -> bytes:
    """Apply Gaussian blur to an image.

    Args:
        image_path: Path to the source image.
        kernel_size: Size of the Gaussian kernel (must be odd).

    Returns:
        JPEG-encoded bytes of the blurred image (quality 95).
    """
    img = _load_image(image_path)
    arr = np.array(img)
    # Ensure kernel_size is odd.
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred = cv2.GaussianBlur(arr, (kernel_size, kernel_size), 0)
    blurred_img = Image.fromarray(blurred)
    buf = io.BytesIO()
    blurred_img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _run_ffmpeg(args: list[str], input_bytes: bytes | None = None) -> bytes:
    """Run an ffmpeg command and return stdout bytes.

    Args:
        args: Full command list starting with 'ffmpeg'.
        input_bytes: Optional bytes piped to stdin.

    Returns:
        Raw output bytes from ffmpeg stdout.

    Raises:
        RuntimeError: If ffmpeg exits with a non-zero code.
    """
    proc = subprocess.run(
        args,
        input=input_bytes,
        capture_output=True,
        timeout=120,
    )
    if proc.returncode != 0:
        stderr_text = proc.stderr.decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}: {stderr_text}")
    return proc.stdout


def mp3_reencode(
    audio_path: str | Path,
    bitrate: str = "128k",
) -> bytes:
    """Re-encode an audio file as MP3 at the given bitrate.

    Uses ffmpeg subprocess.  The source can be any format ffmpeg reads
    (WAV, FLAC, MP3, etc.).

    Args:
        audio_path: Path to the source audio file.
        bitrate: Target MP3 bitrate (e.g. '64k', '128k', '192k').

    Returns:
        MP3-encoded bytes.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        RuntimeError: If ffmpeg fails.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # Write to a temp file so ffmpeg can seek; pipe output to stdout.
    with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(path),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            bitrate,
            "-f",
            "mp3",
            tmp.name,
        ]
        _run_ffmpeg(cmd)
        return Path(tmp.name).read_bytes()


def video_reencode(
    video_path: str | Path,
    crf: int = 23,
) -> bytes:
    """Re-encode a video at the given CRF (Constant Rate Factor).

    Higher CRF = more compression / lower quality.  Typical range:
    18 (visually lossless) to 28 (noticeable degradation).

    Uses libx264 with ffmpeg subprocess.

    Args:
        video_path: Path to the source video file.
        crf: Constant Rate Factor (0-51, lower = better quality).

    Returns:
        H.264/MP4-encoded bytes.

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If ffmpeg fails.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(path),
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            "fast",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-f",
            "mp4",
            "-movflags",
            "+faststart",
            tmp.name,
        ]
        _run_ffmpeg(cmd)
        return Path(tmp.name).read_bytes()
