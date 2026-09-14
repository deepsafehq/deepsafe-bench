#!/usr/bin/env python3
"""Pre-download ALL external model weights from our private HuggingFace repo.

All external weights (CLIP, SigLIP, EfficientNet, etc.) are mirrored in
deepsafe/deepsafe-services under .ext_cache/.  This script downloads them
into models/.ext_cache/ so model wrappers load locally with zero network
calls at server startup.

If .ext_cache/ already exists and has the right files, this is a no-op.

Usage:
    python monolith/prefetch_externals.py          # download all
    python monolith/prefetch_externals.py --check   # verify cache only
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "models" / ".ext_cache"

# Expected files with minimum sizes (bytes) for integrity check
EXPECTED_FILES = {
    "clip-vit-l14-transformers/model.safetensors": 1_500_000_000,
    "clip-vit-l14-openclip/ViT-L-14.pt": 800_000_000,
    "clip-vit-base-patch16/pytorch_model.bin": 500_000_000,
    "siglip-so400m/open_clip_model.safetensors": 3_000_000_000,
    "sd-vae-v1-4/diffusion_pytorch_model.safetensors": 300_000_000,
    "efficientnet-b4/hub/checkpoints/adv-efficientnet-b4-44fb3a87.pth": 70_000_000,
    "inception-resnet-v1/checkpoints/20180402-114759-vggface2.pt": 100_000_000,
    "audioseal/detector_base.pth": 30_000_000,
    "videoseal/y_256b_img.pth": 200_000_000,
    "pwtf-dvd-face/hub/checkpoints/mobilenet0.25_Final.pth": 1_500_000,
    "pwtf-dvd-face/hub/checkpoints/mobilenet_224_model_best_gdconv_external.pth": 3_000_000,
}


def _file_ok(path: Path, min_size: int) -> bool:
    """Check if a file exists and meets minimum size (follows symlinks)."""
    try:
        resolved = path.resolve()
        return resolved.exists() and resolved.stat().st_size >= min_size
    except (OSError, ValueError):
        return False


# Alternate paths for files that may exist in HF cache layout
_ALT_PATHS = {
    "siglip-so400m/open_clip_model.safetensors": [
        "siglip-so400m/models--timm--ViT-SO400M-14-SigLIP-384/snapshots/*/open_clip_model.safetensors",
    ],
    "sd-vae-v1-4/diffusion_pytorch_model.safetensors": [
        "sd-vae-v1-4/models--CompVis--stable-diffusion-v1-4/snapshots/*/vae/diffusion_pytorch_model.safetensors",
    ],
    "audioseal/detector_base.pth": [
        "audioseal/94c8df0b1d5ea8e45af4c884",
    ],
}


def check_cache() -> tuple:
    """Check which files are already cached.

    Checks the primary path first, then alternate HF cache paths.

    Returns:
        (ok_count, missing_list)
    """
    import glob

    ok = 0
    missing = []
    for rel_path, min_size in EXPECTED_FILES.items():
        full_path = CACHE_DIR / rel_path
        if _file_ok(full_path, min_size):
            ok += 1
            continue
        # Check alternate paths (HF cache layout, hash-named files)
        found = False
        for alt_pattern in _ALT_PATHS.get(rel_path, []):
            matches = glob.glob(str(CACHE_DIR / alt_pattern))
            for m in matches:
                if _file_ok(Path(m), min_size):
                    found = True
                    break
            if found:
                break
        if found:
            ok += 1
        else:
            missing.append(rel_path)
    return ok, missing


def download_from_hf():
    """Download .ext_cache from deepsafe/deepsafe-services on HuggingFace."""
    from huggingface_hub import snapshot_download

    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: Set HUGGINGFACE_TOKEN or HF_TOKEN")
        return False

    print("  Downloading from deepsafe/deepsafe-services (.ext_cache/)...")
    try:
        snapshot_download(
            "deepsafe/deepsafe-services",
            repo_type="model",
            allow_patterns=".ext_cache/**",
            local_dir=str(REPO_ROOT / "models"),
            token=token,
            max_workers=4,
        )
        print("  Download complete!")
        return True
    except Exception as e:
        print(f"  Download FAILED: {e}")
        return False


def fixup_audioseal():
    """AudioSeal expects a specific hash-named file. Create symlink."""
    src = CACHE_DIR / "audioseal" / "detector_base.pth"
    dst = CACHE_DIR / "audioseal" / "94c8df0b1d5ea8e45af4c884"
    if src.exists() and not dst.exists():
        os.symlink(src.name, str(dst))
        print("  Symlinked audioseal detector_base.pth -> hash name")


def fixup_siglip():
    """Rebuild HF cache structure for SigLIP if needed.

    open_clip expects HF hub cache layout (models--org--name/snapshots/...).
    If we only have the flat file, create the expected structure.
    """
    flat_file = CACHE_DIR / "siglip-so400m" / "open_clip_model.safetensors"
    hub_dir = CACHE_DIR / "siglip-so400m" / "models--timm--ViT-SO400M-14-SigLIP-384"
    if flat_file.exists() and not hub_dir.exists():
        import shutil

        blobs = hub_dir / "blobs"
        snap = hub_dir / "snapshots" / "main"
        blobs.mkdir(parents=True, exist_ok=True)
        snap.mkdir(parents=True, exist_ok=True)

        blob_path = blobs / "model.safetensors"
        shutil.move(str(flat_file), str(blob_path))
        os.symlink(
            os.path.relpath(str(blob_path), str(snap)),
            str(snap / "open_clip_model.safetensors"),
        )
        config_src = CACHE_DIR / "siglip-so400m" / "open_clip_config.json"
        if config_src.exists():
            shutil.copy2(str(config_src), str(snap / "open_clip_config.json"))

        refs = hub_dir / "refs"
        refs.mkdir(exist_ok=True)
        (refs / "main").write_text("main")
        print("  Rebuilt SigLIP HF cache structure")


def fixup_sd_vae():
    """Rebuild HF cache structure for SD VAE if needed."""
    flat_file = CACHE_DIR / "sd-vae-v1-4" / "diffusion_pytorch_model.safetensors"
    hub_dir = CACHE_DIR / "sd-vae-v1-4" / "models--CompVis--stable-diffusion-v1-4"
    if flat_file.exists() and not hub_dir.exists():
        import shutil

        blobs = hub_dir / "blobs"
        snap = hub_dir / "snapshots" / "main" / "vae"
        blobs.mkdir(parents=True, exist_ok=True)
        snap.mkdir(parents=True, exist_ok=True)

        blob_path = blobs / "vae_model.safetensors"
        shutil.move(str(flat_file), str(blob_path))
        os.symlink(
            os.path.relpath(str(blob_path), str(snap)),
            str(snap / "diffusion_pytorch_model.safetensors"),
        )
        config_src = CACHE_DIR / "sd-vae-v1-4" / "config.json"
        if config_src.exists():
            shutil.copy2(str(config_src), str(snap / "config.json"))

        mi_src = CACHE_DIR / "sd-vae-v1-4" / "model_index.json"
        mi_snap = hub_dir / "snapshots" / "main"
        if mi_src.exists():
            shutil.copy2(str(mi_src), str(mi_snap / "model_index.json"))

        refs = hub_dir / "refs"
        refs.mkdir(exist_ok=True)
        (refs / "main").write_text("main")
        print("  Rebuilt SD VAE HF cache structure")


def fixup_clip_base16():
    """Rebuild HF cache structure for CLIP ViT-Base-patch16 if needed.

    The transformers library expects HF hub cache layout
    (models--org--name/snapshots/...).  If we only have the flat files,
    create the expected structure so CLIPModel.from_pretrained() finds
    them locally.
    """
    flat_dir = CACHE_DIR / "clip-vit-base-patch16"
    hub_dir = flat_dir / "models--openai--clip-vit-base-patch16"
    if flat_dir.exists() and not hub_dir.exists():
        import shutil

        snap = hub_dir / "snapshots" / "main"
        snap.mkdir(parents=True, exist_ok=True)

        # Move/copy all files into the snapshot directory
        for f in flat_dir.iterdir():
            if f.is_file():
                shutil.copy2(str(f), str(snap / f.name))

        refs = hub_dir / "refs"
        refs.mkdir(exist_ok=True)
        (refs / "main").write_text("main")
        print("  Rebuilt CLIP ViT-Base-patch16 HF cache structure")


def download_pwtf_face_alignment():
    """Download PwTF-DVD face alignment weights if not already cached.

    These small files are normally downloaded at runtime by
    torch.utils.model_zoo from github.com/yinglinzheng, which fails
    when TRANSFORMERS_OFFLINE=1 or network is unavailable.
    """
    import urllib.request

    checkpoints_dir = CACHE_DIR / "pwtf-dvd-face" / "hub" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "mobilenet0.25_Final.pth": (
            "https://github.com/yinglinzheng/face_weights/releases"
            "/download/v1/mobilenet0.25_Final.pth"
        ),
        "mobilenet_224_model_best_gdconv_external.pth": (
            "https://github.com/yinglinzheng/face_weights/releases"
            "/download/v1/mobilenet_224_model_best_gdconv_external.pth"
        ),
    }

    for filename, url in files.items():
        target = checkpoints_dir / filename
        if target.exists() and target.stat().st_size > 100_000:
            continue
        print(f"  Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, str(target))
            print(f"  Downloaded {filename}")
        except Exception as e:
            print(f"  WARNING: Could not download {filename}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check only, no download",
    )
    args = parser.parse_args()

    print(f"External model cache: {CACHE_DIR}")

    ok, missing = check_cache()
    total = len(EXPECTED_FILES)

    if not missing:
        print(f"All {total} external models cached. Nothing to download.")
        return

    print(f"Cached: {ok}/{total}. Missing: {len(missing)}")
    for m in missing:
        print(f"  MISSING: {m}")

    if args.check:
        print("\nRun without --check to download missing files.")
        sys.exit(1 if missing else 0)

    # Download from our private HuggingFace repo
    if not download_from_hf():
        print("ERROR: Download failed. Set HUGGINGFACE_TOKEN and try again.")
        sys.exit(1)

    # Apply fixups for libraries that expect specific cache layouts
    fixup_audioseal()
    fixup_siglip()
    fixup_sd_vae()
    fixup_clip_base16()
    download_pwtf_face_alignment()

    # Verify
    ok, missing = check_cache()
    if missing:
        print(f"\nWARNING: {len(missing)} files still missing after download:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    print(f"\nAll {total} external models cached successfully.")


if __name__ == "__main__":
    main()
