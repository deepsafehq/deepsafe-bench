#!/bin/bash
# Pull model weights from HuggingFace into models/ directories.
# Usage: bash apps/inference/pull_weights.sh
#
# No token required: deepsafe/deepsafe-services is public.
# HUGGINGFACE_TOKEN is honoured if set (useful behind a proxy or for rate limits).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Token is optional. The repos are public.
HF_TOKEN="${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}"

echo "Pulling weights from HuggingFace (deepsafe/deepsafe-services)..."
echo "This downloads ~40 GB (model weights + external caches)."
echo "Using hf_transfer for fast downloads (if available)."

# Enable hf_transfer for 10-100x faster downloads
export HF_XET_HIGH_PERFORMANCE=1

python3 -c "
from huggingface_hub import snapshot_download
import os
token = os.environ.get('HUGGINGFACE_TOKEN') or os.environ.get('HF_TOKEN') or None
print('Downloading weights (deepsafe/deepsafe-services)...')
# Skip what the current lineup does not use:
#   aasist3 / sonics were removed from the registry (1.4 GB)
#   Dockerfile / app.py / requirements.txt are leftovers from the pre-2026-04
#   microservice architecture and are not read by the monolith server.
snapshot_download(
    'deepsafe/deepsafe-services',
    repo_type='model',
    local_dir='./services_weights',
    max_workers=4,
    token=token,
    ignore_patterns=[
        'audio/aasist3/**',
        'audio/sonics/**',
        'ensemble-core/**',
        '**/Dockerfile',
        '**/app.py',
        '**/requirements.txt',
        '**/.dockerignore',
    ],
)
print('Downloading model code (deepsafe/model-code)...')
snapshot_download(
    'deepsafe/model-code',
    repo_type='model',
    local_dir='./services_code',
    allow_patterns='clean/**',
    max_workers=4,
    token=token,
)
print('Download complete!')
"

echo "Copying weights into models/..."
rsync -a services_weights/ models/
rm -rf services_weights

echo "Installing model code into models/<modality>/<model>/code/ ..."
# clean/<modality>/<model>/  ->  models/<modality>/<model>/code/
for d in services_code/clean/*/*; do
    [ -d "$d" ] || continue
    target="models/${d#services_code/clean/}/code"
    mkdir -p "$target"
    rsync -a "$d/" "$target/"
done
rm -rf services_code

# Rename directories to match model registry names
cd models
[ -d "video/fake-stormer" ] && mv "video/fake-stormer" "video/fakestormer" || true
[ -d "video/dfd-fcg" ] && mv "video/dfd-fcg" "video/dfd_fcg" || true
[ -d "video/pwtf-dvd" ] && mv "video/pwtf-dvd" "video/pwtf_dvd" || true
[ -d "provenance/c2pa-checker" ] && mv "provenance/c2pa-checker" "provenance/c2pa" || true
[ -d "provenance/invisible-watermark" ] && mv "provenance/invisible-watermark" "provenance/sdxl_watermark" || true

# Fix nested weight directories from HF download structure
# (HF preserves repo paths like video/fake-stormer/weights/ inside video/fakestormer/)
for model_dir in video/fakestormer video/dfd_fcg video/pwtf_dvd; do
    if [ -d "$model_dir" ] && [ ! -d "$model_dir/weights" ]; then
        # Find nested weights dir and move it up
        nested=$(find "$model_dir" -maxdepth 3 -type d -name "weights" 2>/dev/null | head -1)
        if [ -n "$nested" ]; then
            mv "$nested" "$model_dir/weights"
            echo "  Fixed: $model_dir/weights (was nested)"
        fi
    fi
done
cd "$REPO_ROOT"

echo "Done! Weights are in models/<modality>/<model>/weights/"
