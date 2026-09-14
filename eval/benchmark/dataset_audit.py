#!/usr/bin/env python3
"""Audit the DeepSafe evaluation dataset structure and provenance.

Walks the ``{root}/{images,audio,video}/{real,fake}/{generator}/`` tree
and produces ``eval/results/dataset_audit.json`` with per-modality
counts, per-generator metadata (era, license), and flagged low-N
generators.

Usage:
    python eval/benchmark/dataset_audit.py

Set ``DEEPSAFE_DATASET`` to override the default dataset root
(``<project_root>/dataset/master_eval_full``).
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DEFAULT_DATASET_ROOT = str(_PROJECT_ROOT / "dataset" / "master_eval_full")
DATASET_ROOT = Path(
    os.getenv("DEEPSAFE_DATASET", _DEFAULT_DATASET_ROOT)
)
RESULTS_DIR = _PROJECT_ROOT / "eval" / "results"

LOW_N_THRESHOLD = 100

# ---------------------------------------------------------------------------
# File extension whitelist per modality
# ---------------------------------------------------------------------------

_IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
_AUDIO_EXTS: Set[str] = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
_VIDEO_EXTS: Set[str] = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

_MODALITY_EXTS = {
    "images": _IMAGE_EXTS,
    "audio": _AUDIO_EXTS,
    "video": _VIDEO_EXTS,
}

# ---------------------------------------------------------------------------
# Generator era classification
# ---------------------------------------------------------------------------

GENERATOR_ERA: Dict[str, str] = {
    # pre-2024: GANs, early diffusion, classic vocoders / face-swap
    "stylegan2": "pre-2024",
    "stylegan3": "pre-2024",
    "styleganxl": "pre-2024",
    "biggan": "pre-2024",
    "vqgan": "pre-2024",
    "stargan": "pre-2024",
    "starganv2": "pre-2024",
    "sd_1.5": "pre-2024",
    "sd_2.1": "pre-2024",
    "stable_diffusion_1_3": "pre-2024",
    "stable_diffusion_1_4": "pre-2024",
    "stable_diffusion_2": "pre-2024",
    "dalle_2": "pre-2024",
    "midjourney_v5": "pre-2024",
    "glide": "pre-2024",
    "melgan": "pre-2024",
    "melgan_large": "pre-2024",
    "multi_band_melgan": "pre-2024",
    "full_band_melgan": "pre-2024",
    "waveglow": "pre-2024",
    "hifigan": "pre-2024",
    "parallel_wavegan": "pre-2024",
    "griffin_lim": "pre-2024",
    "tacotron": "pre-2024",
    "wavenet": "pre-2024",
    "conformer_fastspeech2_pwg": "pre-2024",
    "neural_codec": "pre-2024",
    "faceswap": "pre-2024",
    "deepfacelab": "pre-2024",
    "face2face": "pre-2024",
    "fsgan": "pre-2024",
    "simswap": "pre-2024",
    "facedancer": "pre-2024",
    "wav2lip": "pre-2024",
    "sadtalker": "pre-2024",
    "face_swap": "pre-2024",
    "face_editing": "pre-2024",
    "face_reenactment": "pre-2024",
    "e4e": "pre-2024",
    "styleclip": "pre-2024",
    "whichfaceisreal": "pre-2024",
    "lavie": "pre-2024",
    "modelscope_gen": "pre-2024",
    "show1": "pre-2024",
    # 2024: SDXL/3, DALL-E 3, FLUX 1, modern TTS, first video gen
    "stable_diffusion_xl": "2024",
    "stable_diffusion_3": "2024",
    "dalle_3": "2024",
    "flux_1": "2024",
    "flux_schnell": "2024",
    "midjourney_6": "2024",
    "firefly": "2024",
    "imagen_3": "2024",
    "sora": "2024",
    "kling": "2024",
    "gen2": "2024",
    "veo": "2024",
    "bark": "2024",
    "elevenlabs": "2024",
    "valle": "2024",
    "xtts": "2024",
    "yourtts": "2024",
    "rvc": "2024",
    "suno": "2024",
    "udio": "2024",
    "musicgen": "2024",
    "stable_audio": "2024",
    "riffusion": "2024",
    "crafter": "2024",
    "hotshot": "2024",
    "moonvalley": "2024",
    "hunyuan": "2024",
    "morphstudio": "2024",
    "chroma": "2024",
    # 2025+: latest frontier models
    "midjourney_7": "2025+",
    "gpt_image_1": "2025+",
    "imagen_4.0": "2025+",
    "ideogram_3.0": "2025+",
    "recraft_v3": "2025+",
    "grok_2_image_1212": "2025+",
    "gemini": "2025+",
    "hidream_i1_full": "2025+",
    "mystic": "2025+",
}

# ---------------------------------------------------------------------------
# Generator license classification
# ---------------------------------------------------------------------------

GENERATOR_LICENSE: Dict[str, str] = {
    # open: weights publicly released
    "stylegan2": "open",
    "stylegan3": "open",
    "styleganxl": "open",
    "biggan": "open",
    "vqgan": "open",
    "stargan": "open",
    "starganv2": "open",
    "sd_1.5": "open",
    "sd_2.1": "open",
    "stable_diffusion_1_3": "open",
    "stable_diffusion_1_4": "open",
    "stable_diffusion_2": "open",
    "stable_diffusion_xl": "open",
    "stable_diffusion_3": "open",
    "flux_1": "open",
    "flux_schnell": "open",
    "glide": "open",
    "chroma": "open",
    "hidream_i1_full": "open",
    "mystic": "open",
    "melgan": "open",
    "melgan_large": "open",
    "multi_band_melgan": "open",
    "full_band_melgan": "open",
    "waveglow": "open",
    "hifigan": "open",
    "parallel_wavegan": "open",
    "griffin_lim": "open",
    "tacotron": "open",
    "wavenet": "open",
    "conformer_fastspeech2_pwg": "open",
    "neural_codec": "open",
    "bark": "open",
    "musicgen": "open",
    "stable_audio": "open",
    "riffusion": "open",
    "valle": "open",
    "xtts": "open",
    "yourtts": "open",
    "rvc": "open",
    "faceswap": "open",
    "deepfacelab": "open",
    "face2face": "open",
    "fsgan": "open",
    "simswap": "open",
    "facedancer": "open",
    "wav2lip": "open",
    "sadtalker": "open",
    "face_swap": "open",
    "face_editing": "open",
    "face_reenactment": "open",
    "e4e": "open",
    "styleclip": "open",
    "whichfaceisreal": "open",
    "lavie": "open",
    "modelscope_gen": "open",
    "show1": "open",
    "hunyuan": "open",
    # closed: API-only or proprietary weights
    "dalle_2": "closed",
    "dalle_3": "closed",
    "midjourney_v5": "closed",
    "midjourney_6": "closed",
    "midjourney_7": "closed",
    "firefly": "closed",
    "imagen_3": "closed",
    "imagen_4.0": "closed",
    "gpt_image_1": "closed",
    "grok_2_image_1212": "closed",
    "ideogram_3.0": "closed",
    "recraft_v3": "closed",
    "gemini": "closed",
    "sora": "closed",
    "kling": "closed",
    "gen2": "closed",
    "veo": "closed",
    "crafter": "closed",
    "hotshot": "closed",
    "moonvalley": "closed",
    "morphstudio": "closed",
    "elevenlabs": "closed",
    "suno": "closed",
    "udio": "closed",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_files(directory: Path, valid_exts: Set[str]) -> int:
    """Count files with matching extensions in a directory tree.

    Args:
        directory: Root directory to walk.
        valid_exts: Set of lowercase file extensions including the dot.

    Returns:
        Number of matching files.
    """
    count = 0
    if not directory.is_dir():
        return count
    for item in directory.rglob("*"):
        if item.is_file() and item.suffix.lower() in valid_exts:
            count += 1
    return count


def _classify_generator(name: str) -> Dict[str, str]:
    """Return era and license tags for a generator name.

    Falls back to "unknown" for generators not in the hardcoded maps.
    MLAAD-prefixed generators are matched after stripping the prefix
    and attempting a fuzzy lookup against known families.

    Args:
        name: Generator directory name.

    Returns:
        Dict with "era" and "license" keys.
    """
    era = GENERATOR_ERA.get(name, "unknown")
    lic = GENERATOR_LICENSE.get(name, "unknown")

    # Handle MLAAD generators: try matching the TTS family.
    if era == "unknown" and name.startswith("mlaad_"):
        lower = name.lower()
        if "bark" in lower:
            era, lic = "2024", "open"
        elif "xtts" in lower:
            era, lic = "2024", "open"
        elif "edge_tts" in lower:
            era, lic = "2024", "closed"
        elif "vits" in lower:
            era, lic = "pre-2024", "open"
        elif "kokoro" in lower:
            era, lic = "2025+", "open"
        elif "indicf5" in lower:
            era, lic = "2025+", "open"
        elif "llasa" in lower:
            era, lic = "2025+", "open"
        elif "facebook_mms" in lower:
            era, lic = "2024", "open"
        elif "deepgram" in lower:
            era, lic = "2024", "closed"
        elif "minimax" in lower:
            era, lic = "2025+", "closed"
        elif "elevenlabs" in lower:
            era, lic = "2024", "closed"
        elif "higgs_audio" in lower:
            era, lic = "2025+", "closed"
        elif "griffin_lim" in lower:
            era, lic = "pre-2024", "open"
        elif "outetts" in lower:
            era, lic = "2025+", "open"
        elif "fireredtts" in lower:
            era, lic = "2025+", "open"
        elif "vibevoice" in lower:
            era, lic = "2025+", "closed"
        elif "index_tts" in lower:
            era, lic = "2025+", "open"
        elif "megatts" in lower:
            era, lic = "2025+", "closed"
        elif "spark_tts" in lower:
            era, lic = "2025+", "open"
        elif "orpheus" in lower:
            era, lic = "2025+", "open"
        elif "qwen" in lower:
            era, lic = "2025+", "open"
        elif "cartesia" in lower:
            era, lic = "2025+", "closed"
        elif "neutts" in lower:
            era, lic = "2025+", "closed"

    # ASVspoof attack codes
    if era == "unknown" and name.startswith("asvspoof_"):
        era, lic = "pre-2024", "open"

    # in_the_wild
    if name == "in_the_wild":
        era, lic = "pre-2024", "open"

    return {"era": era, "license": lic}


# ---------------------------------------------------------------------------
# Main audit logic
# ---------------------------------------------------------------------------


def audit_modality(
    modality_dir: Path,
    valid_exts: Set[str],
) -> Dict[str, Any]:
    """Audit a single modality (images/audio/video).

    Args:
        modality_dir: Path like ``{root}/images``.
        valid_exts: Set of valid file extensions for this modality.

    Returns:
        Dict with keys: total_real, total_fake, total, fake_ratio,
        generators, low_n_generators, era_distribution,
        license_distribution.
    """
    real_dir = modality_dir / "real"
    fake_dir = modality_dir / "fake"

    total_real = _count_files(real_dir, valid_exts)

    generators: Dict[str, Dict[str, Any]] = {}
    total_fake = 0

    if fake_dir.is_dir():
        for gen_path in sorted(fake_dir.iterdir()):
            if not gen_path.is_dir():
                continue
            gen_name = gen_path.name
            count = _count_files(gen_path, valid_exts)
            tags = _classify_generator(gen_name)
            generators[gen_name] = {
                "count": count,
                "era": tags["era"],
                "license": tags["license"],
            }
            total_fake += count

    total = total_real + total_fake
    fake_ratio = round(total_fake / total, 4) if total > 0 else 0.0

    # Aggregate distributions.
    era_dist: Dict[str, int] = defaultdict(int)
    lic_dist: Dict[str, int] = defaultdict(int)
    low_n: List[str] = []

    for gen_name, info in generators.items():
        era_dist[info["era"]] += info["count"]
        lic_dist[info["license"]] += info["count"]
        if info["count"] < LOW_N_THRESHOLD:
            low_n.append(gen_name)

    return {
        "total_real": total_real,
        "total_fake": total_fake,
        "total": total,
        "fake_ratio": fake_ratio,
        "generators": generators,
        "low_n_generators": sorted(low_n),
        "era_distribution": dict(era_dist),
        "license_distribution": dict(lic_dist),
    }


def run_audit() -> Dict[str, Any]:
    """Run the full dataset audit across all modalities.

    Returns:
        Complete audit dict ready for JSON serialization.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    result: Dict[str, Any] = {
        "metadata": {
            "timestamp": timestamp,
            "dataset_root": str(DATASET_ROOT),
            "low_n_threshold": LOW_N_THRESHOLD,
        },
        "modalities": {},
        "grand_totals": {
            "total_real": 0,
            "total_fake": 0,
            "total": 0,
            "total_generators": 0,
            "total_low_n_generators": 0,
        },
    }

    for modality_name, valid_exts in _MODALITY_EXTS.items():
        modality_dir = DATASET_ROOT / modality_name
        if not modality_dir.is_dir():
            print(f"  WARN: modality directory not found: {modality_dir}")
            result["modalities"][modality_name] = {
                "total_real": 0,
                "total_fake": 0,
                "total": 0,
                "fake_ratio": 0.0,
                "generators": {},
                "low_n_generators": [],
                "era_distribution": {},
                "license_distribution": {},
            }
            continue

        print(f"  Scanning {modality_name}...", flush=True)
        audit = audit_modality(modality_dir, valid_exts)
        result["modalities"][modality_name] = audit

        result["grand_totals"]["total_real"] += audit["total_real"]
        result["grand_totals"]["total_fake"] += audit["total_fake"]
        result["grand_totals"]["total"] += audit["total"]
        result["grand_totals"]["total_generators"] += len(
            audit["generators"]
        )
        result["grand_totals"]["total_low_n_generators"] += len(
            audit["low_n_generators"]
        )

    return result


def _print_summary(audit: Dict[str, Any]) -> None:
    """Print a human-readable summary to stdout.

    Args:
        audit: The full audit dict from ``run_audit()``.
    """
    gt = audit["grand_totals"]
    print("\n" + "=" * 60)
    print("  Dataset Audit Summary")
    print("=" * 60)
    print(f"  Root:        {audit['metadata']['dataset_root']}")
    print(f"  Timestamp:   {audit['metadata']['timestamp']}")
    print(f"  Total files: {gt['total']:,}")
    print(f"    Real:      {gt['total_real']:,}")
    print(f"    Fake:      {gt['total_fake']:,}")
    print(
        f"  Generators:  {gt['total_generators']} "
        f"({gt['total_low_n_generators']} with N < {LOW_N_THRESHOLD})"
    )

    for mod_name, mod_data in audit["modalities"].items():
        print(f"\n  -- {mod_name.upper()} --")
        print(f"     Real: {mod_data['total_real']:,}")
        print(f"     Fake: {mod_data['total_fake']:,}")
        print(f"     Total: {mod_data['total']:,}")
        print(f"     Fake ratio: {mod_data['fake_ratio']:.2%}")
        print(f"     Generators: {len(mod_data['generators'])}")

        if mod_data["era_distribution"]:
            print("     Era distribution:")
            for era, count in sorted(
                mod_data["era_distribution"].items()
            ):
                print(f"       {era}: {count:,}")

        if mod_data["license_distribution"]:
            print("     License distribution:")
            for lic, count in sorted(
                mod_data["license_distribution"].items()
            ):
                print(f"       {lic}: {count:,}")

        if mod_data["low_n_generators"]:
            print(
                f"     Low-N generators (< {LOW_N_THRESHOLD}): "
                f"{', '.join(mod_data['low_n_generators'])}"
            )

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the dataset audit and save results."""
    if not DATASET_ROOT.is_dir():
        print(
            f"ERROR: Dataset root not found: {DATASET_ROOT}\n"
            "Set DEEPSAFE_DATASET to override.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Dataset audit: {DATASET_ROOT}")
    audit = run_audit()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "dataset_audit.json"
    with open(out_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"\n  Saved -> {out_path}")

    _print_summary(audit)


if __name__ == "__main__":
    main()
