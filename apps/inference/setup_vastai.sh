#!/bin/bash
# DeepSafe Monolith - One-command vast.ai deployment
#
# Prerequisites:
#   - Vast.ai instance with PyTorch 2.5.1 + CUDA 12.1 template
#   - A100 80GB recommended (42GB VRAM needed for all 22 models)
#   - 64GB RAM minimum
#   - 200GB+ storage
#
# Usage:
#   chmod +x setup_vastai.sh && ./setup_vastai.sh
#
# After setup:
#   cd /workspace/DeepSafe
#   python -m monolith.server
#   # Or: python -m monolith.smoke_test

set -euo pipefail

echo "============================================================"
echo "DeepSafe Monolith — vast.ai Setup"
echo "============================================================"

REPO_DIR="/workspace/DeepSafe"
WEIGHTS_REPO="deepsafe/deepsafe-services"

# ── Step 1: Clone repo ──────────────────────────────────────────────────────
if [ ! -d "$REPO_DIR" ]; then
    echo "[1/8] Cloning DeepSafe repository..."
    cd /workspace
    git clone https://github.com/deepsafehq/deepsafe.git DeepSafe
    cd "$REPO_DIR"
    git checkout monolith
else
    echo "[1/8] Repository exists, pulling latest..."
    cd "$REPO_DIR"
    git checkout monolith
    git pull origin monolith
fi

# ── Step 2: Install Python dependencies ─────────────────────────────────────
echo "[2/8] Installing Python dependencies..."
pip install --no-cache-dir -r monolith/requirements.lock 2>&1 | tail -5

# Install videoseal separately (--no-deps to avoid conflicts)
pip install --no-cache-dir videoseal==1.0.1 --no-deps 2>/dev/null || \
    echo "  Warning: videoseal install failed (optional)"

echo "  PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# ── Step 3: Build & patch fairseq ──────────────────────────────────────────
echo "[3/8] Building fairseq from source..."
if python3 -c "import fairseq" 2>/dev/null; then
    echo "  fairseq already installed"
else
    FAIRSEQ_TMP="/tmp/fairseq_src_$$"
    git clone --depth 1 https://github.com/facebookresearch/fairseq.git "$FAIRSEQ_TMP"
    READTHEDOCS=1 pip install --no-deps --no-build-isolation "$FAIRSEQ_TMP"
    rm -rf "$FAIRSEQ_TMP"
fi
pip install --no-cache-dir cython bitarray portalocker sacrebleu colorama lxml tabulate
python3 monolith/patch_fairseq.py || echo "  WARNING: fairseq patch failed"

# ── Step 4: Download model weights from HuggingFace ─────────────────────────
echo "[4/8] Pulling model weights from HuggingFace..."
bash monolith/pull_weights.sh

# ── Step 5: Clone model code repos ─────────────────────────────────────────
echo "[5/8] Cloning model code repositories..."
bash monolith/clone_repos.sh

# ── Step 6: Pre-download SCRFD face detection model ─────────────────────────
echo "[6/8] Pre-downloading SCRFD face detection model..."
python -c "
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_sc', allowed_modules=['detection'],
                   providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))
print('  SCRFD cached at ~/.insightface/models/buffalo_sc/')
" 2>/dev/null || echo "  SCRFD download failed (will auto-download on first use)"

# ── Step 7: Pre-download external model weights ───────────────────────────
# CLIP, Wav2Vec2, SigLIP, EfficientNet etc. download from HuggingFace on
# first use. Without this step, server startup takes 30+ minutes.
echo "[7/8] Pre-downloading external model weights (CLIP, Wav2Vec2, etc.)..."
cd "$REPO_DIR"
python monolith/prefetch_externals.py || echo "  WARNING: Some downloads failed"

# ── Step 8: Validate ────────────────────────────────────────────────────────
echo "[8/8] Validating installation..."
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print()

from monolith.config import MODEL_REGISTRY, get_enabled_models
print(f'Model registry: {len(MODEL_REGISTRY)} models')
print(f'Enabled: {len(get_enabled_models())} models')
"

echo "============================================================"
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  cd $REPO_DIR"
echo "  python -m monolith.smoke_test          # validate all models"
echo "  python -m monolith.server              # start server on :8000"
echo "============================================================"
