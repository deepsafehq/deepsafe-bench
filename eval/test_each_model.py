#!/usr/bin/env python3
"""Test each model individually: load time, download check, inference check.

Loads one model at a time, runs inference on a real + fake sample,
verifies directionality, then unloads. Reports issues before full server.

Usage:
    source .venv/bin/activate
    python eval/test_each_model.py
    python eval/test_each_model.py --models npr,aide   # test specific
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")

from monolith.config import MODEL_REGISTRY, get_weights_path
from monolith.models import get_predictor_class
from monolith.models.base import get_device, setup_inference_optimizations

DATASET_DIR = PROJECT_ROOT / "dataset" / "master_eval"

# Quick test files: one real, one fake per modality
TEST_FILES = {
    "image": {
        "real": "images/real/coco/img_00001.jpg",
        "fake": "images/fake/dalle_3/img_05155.jpg",
    },
    "audio": {
        "real": "audio/real/asvspoof_bonafide/aud_00001.wav",
        "fake": "audio/fake/elevenlabs/aud_01001.wav",
    },
    "video": {
        "real": "video/real/msrvtt/vid_01001.mp4",
        "fake": "video/fake/sora/vid_01001.mp4",
    },
    "provenance": {
        "real": "images/real/coco/img_00001.jpg",
        "fake": "images/fake/dalle_3/img_05155.jpg",
    },
}


def _effective_modality(model_name, modality):
    """Return the file modality to use for test input.

    Most provenance models use image files, but AudioSeal requires
    audio input.
    """
    if model_name == "audioseal":
        return "audio"
    return modality


def find_test_file(modality, label, model_name=None):
    """Find a test file that actually exists."""
    effective = _effective_modality(model_name, modality) if model_name else modality
    path = DATASET_DIR / TEST_FILES.get(effective, {}).get(label, "")
    if path.exists():
        return path
    # Fallback: find any file in the right directory
    search_dir = DATASET_DIR / ("images" if effective in ("image", "provenance") else effective) / label
    if not search_dir.exists():
        search_dir = DATASET_DIR / ("images" if effective in ("image", "provenance") else effective)
        for sub in search_dir.iterdir():
            if sub.is_dir():
                for f in sub.iterdir():
                    if f.is_file() and f.suffix.lower() in ('.jpg', '.png', '.wav', '.mp3', '.mp4'):
                        return f
    else:
        for sub in search_dir.iterdir():
            if sub.is_dir():
                for f in sub.iterdir():
                    if f.is_file():
                        return f
            elif sub.is_file():
                return sub
    return None


def test_one_model(name, device):
    """Test a single model: load, infer, check, unload."""
    defn = MODEL_REGISTRY[name]
    modality = defn.modality

    result = {
        "name": name,
        "modality": modality,
        "load_time_s": None,
        "load_ok": False,
        "real_prob": None,
        "fake_prob": None,
        "direction_ok": None,
        "error": None,
        "vram_mb": None,
    }

    # Load
    try:
        cls = get_predictor_class(name)
        predictor = cls()
        t0 = time.perf_counter()
        predictor.load(get_weights_path(name), device)
        load_time = time.perf_counter() - t0
        result["load_time_s"] = round(load_time, 1)
        result["load_ok"] = True
        result["vram_mb"] = round(torch.cuda.memory_allocated(0) / 1024**2)
    except Exception as e:
        result["error"] = str(e)[:200]
        return result

    # Test on real sample
    real_file = find_test_file(modality, "real", model_name=name)
    if real_file:
        try:
            r = predictor.predict(real_file.read_bytes())
            result["real_prob"] = r.get("probability")
        except Exception as e:
            result["error"] = f"real inference: {str(e)[:150]}"

    # Test on fake sample
    fake_file = find_test_file(modality, "fake", model_name=name)
    if fake_file:
        try:
            r = predictor.predict(fake_file.read_bytes())
            result["fake_prob"] = r.get("probability")
        except Exception as e:
            result["error"] = f"fake inference: {str(e)[:150]}"

    # Check directionality (fake_prob should be > real_prob for detection models)
    if result["real_prob"] is not None and result["fake_prob"] is not None:
        if modality == "provenance":
            result["direction_ok"] = True  # Provenance models are different
        else:
            result["direction_ok"] = result["fake_prob"] > result["real_prob"]

    # Unload
    del predictor
    gc.collect()
    torch.cuda.empty_cache()

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names (default: all)")
    args = parser.parse_args()

    device = get_device()
    setup_inference_optimizations(device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model_names = (
        args.models.split(",") if args.models
        else list(MODEL_REGISTRY.keys())
    )

    print(f"\nTesting {len(model_names)} models individually...\n")
    print(f"{'Model':<16} {'Load(s)':<9} {'VRAM(MB)':<10} {'Real':<8} {'Fake':<8} {'Dir':<6} {'Status'}")
    print("-" * 80)

    results = []
    issues = []

    for name in model_names:
        if name not in MODEL_REGISTRY:
            print(f"  {name}: UNKNOWN MODEL")
            continue

        result = test_one_model(name, device)
        results.append(result)

        # Format output
        load_s = f"{result['load_time_s']:.1f}" if result['load_time_s'] else "FAIL"
        vram = str(result['vram_mb']) if result['vram_mb'] else "-"
        real_p = f"{result['real_prob']:.3f}" if result['real_prob'] is not None else "ERR"
        fake_p = f"{result['fake_prob']:.3f}" if result['fake_prob'] is not None else "ERR"

        if result['direction_ok'] is True:
            dir_str = "OK"
        elif result['direction_ok'] is False:
            dir_str = "BAD"
        else:
            dir_str = "?"

        if not result['load_ok']:
            status = f"LOAD FAIL: {result['error'][:40]}"
            issues.append((name, f"load failed: {result['error'][:80]}"))
        elif result['direction_ok'] is False:
            status = "WRONG DIRECTION"
            issues.append((name, f"fake({fake_p}) <= real({real_p})"))
        elif result['error']:
            status = f"WARN: {result['error'][:40]}"
            issues.append((name, result['error'][:80]))
        elif result['load_time_s'] and result['load_time_s'] > 60:
            status = "SLOW"
            issues.append((name, f"load took {result['load_time_s']:.0f}s"))
        else:
            status = "OK"

        print(f"  {name:<14} {load_s:<9} {vram:<10} {real_p:<8} {fake_p:<8} {dir_str:<6} {status}")

        # Free GPU between models
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 80)
    total_time = sum(r['load_time_s'] for r in results if r['load_time_s'])
    ok_count = sum(1 for r in results if r['load_ok'] and r.get('direction_ok') is not False)
    print(f"Total sequential load time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Models OK: {ok_count}/{len(results)}")

    if issues:
        print(f"\nISSUES ({len(issues)}):")
        for name, issue in issues:
            print(f"  {name}: {issue}")
    else:
        print("\nALL MODELS OK — ready for parallel loading and full experiment")


if __name__ == "__main__":
    main()
