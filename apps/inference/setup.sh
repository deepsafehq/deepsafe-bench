#!/bin/bash
# DeepSafe Monolith — One-command setup for fresh GPU systems.
#
# Usage:
#   git clone https://github.com/deepsafehq/deepsafe-bench.git
#   cd deepsafe
#   bash apps/inference/setup.sh
#
# Tested on: V100-32GB, A100-40GB/80GB, A6000-48GB (Python 3.11, CUDA 12.x)
# Requirements: Python 3.10-3.12, NVIDIA GPU with CUDA 12.x, git
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

echo "============================================================"
echo "DeepSafe Monolith Setup"
echo "============================================================"
echo "Repo root: $REPO_ROOT"
echo "Python:    $(python3 --version 2>&1)"
echo ""

# ── Step 1: Install uv if missing ──────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "[1/8] Installing uv..."
    pip install uv
else
    echo "[1/8] uv already installed: $(uv --version)"
fi

# ── Step 2: Create venv ────────────────────────────────────────────────────
VENV="$REPO_ROOT/.venv"
if [ ! -d "$VENV" ]; then
    echo "[2/8] Creating virtual environment..."
    uv venv "$VENV"
else
    echo "[2/8] Virtual environment exists: $VENV"
fi

# Activate
source "$VENV/bin/activate"
echo "  Python: $(which python3)"

# ── Step 3: Install PyTorch ────────────────────────────────────────────────
# CUDA wheels only exist for Linux/Windows x86_64. On macOS the cu121 index has
# no candidate and the install fails outright, so pick the build by platform.
if [ "$(uname -s)" = "Darwin" ]; then
    echo "[3/9] Installing PyTorch 2.5.1 (macOS, MPS/CPU)..."
    uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
else
    echo "[3/9] Installing PyTorch 2.5.1 + CUDA 12.1..."
    uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu121
fi

# Report the accelerator instead of asserting CUDA. The server auto-detects
# CUDA > MPS > CPU, so a machine without CUDA is slow, not broken.
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'  Accelerator: CUDA ({torch.cuda.get_device_name(0)})')
elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
    print('  Accelerator: MPS (Apple Silicon). Expect slow video inference.')
else:
    print('  Accelerator: CPU only. The full lineup will be very slow.')
"

# ── Step 4: Install all dependencies ───────────────────────────────────────
echo "[4/8] Installing dependencies..."
uv pip install -r apps/inference/requirements.lock \
    --index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.org/simple

# Install videoseal separately (conflicts with some deps)
uv pip install videoseal==1.0.1 --no-deps 2>/dev/null || true

# ── Step 5: Build & patch fairseq ──────────────────────────────────────────
echo "[5/8] Building fairseq from source + applying Python 3.12 patch..."
if python3 -c "import fairseq" 2>/dev/null; then
    echo "  fairseq already installed"
else
    # Build from source (pip version fails on Python 3.11+/CUDA mismatch)
    FAIRSEQ_TMP="/tmp/fairseq_src_$$"
    git clone --depth 1 https://github.com/facebookresearch/fairseq.git "$FAIRSEQ_TMP"
    # Build without CUDA extensions (avoids CUDA version mismatch)
    READTHEDOCS=1 uv pip install --no-deps --no-build-isolation "$FAIRSEQ_TMP"
    rm -rf "$FAIRSEQ_TMP"
fi

# Install fairseq runtime deps that --no-deps skipped
uv pip install cython bitarray portalocker sacrebleu colorama lxml tabulate

# Patch fairseq for Python 3.11+/3.12+ (fixes hydra_init MISSING defaults)
python3 apps/inference/patch_fairseq.py || echo "  WARNING: fairseq patch verification failed (may still work with fairseq_compat.py)"

# ── Step 6: Pre-download SCRFD face detection model ──────────────────────
echo "[6/9] Pre-downloading SCRFD face detection model (insightface)..."
python3 -c "
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_sc', allowed_modules=['detection'],
                   providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))
print('  SCRFD model downloaded and cached at ~/.insightface/models/buffalo_sc/')
" 2>/dev/null || echo "  WARNING: SCRFD download failed (will auto-download on first use)"

# ── Step 7: Install hf_transfer for fast HuggingFace downloads ───────────
echo "[7/9] Installing hf_transfer (Rust-based fast downloader)..."
uv pip install hf_transfer 2>/dev/null || true
export HF_HUB_ENABLE_HF_TRANSFER=1

# ── Step 8: Pull model weights + ext_cache from HuggingFace ──────────────
echo "[8/9] Pulling model weights from HuggingFace..."
# deepsafe/deepsafe-services and deepsafe/model-code are public: no token
# needed. HUGGINGFACE_TOKEN is still honoured for rate limits or a proxy.
if [ -z "${SKIP_WEIGHTS:-}" ]; then
    bash apps/inference/pull_weights.sh
else
    echo "  Skipped (SKIP_WEIGHTS=1)"
fi

# ── Step 9: Setup paths, symlinks, patches, ext_cache ─────────────────────
echo "[9/9] Setting up model paths and external model cache..."

# 9a: Model code paths, symlinks, patches (vendored code, no cloning)
bash apps/inference/clone_repos.sh

# 9b: xlsr2_300m.pt symlinks (must be AFTER weights download)
XLSR_PT="$REPO_ROOT/models/audio/shiftyspeech/weights/xlsr_53_56k.pt"
if [ -f "$XLSR_PT" ]; then
    mkdir -p /app/weights 2>/dev/null || true
    ln -sf "$XLSR_PT" models/xlsr2_300m.pt 2>/dev/null || true
    ln -sf "$XLSR_PT" apps/inference/models/xlsr2_300m.pt 2>/dev/null || true
    ln -sf "$XLSR_PT" /app/weights/xlsr2_300m.pt 2>/dev/null || true
    echo "  Symlinked xlsr2_300m.pt"
fi

# 9c: External model cache (CLIP, SigLIP, SD VAE, EfficientNet, etc.)
# Downloaded from our private deepsafe/deepsafe-services repo so
# server startup needs ZERO network calls.
python3 apps/inference/prefetch_externals.py || echo "  WARNING: Some ext_cache downloads failed"

# ── Done ───────────────────────────────────────────────────────────────────
# ── Optional: Download evaluation dataset ─────────────────────────────────
if [ -z "${SKIP_DATASET:-}" ]; then
    echo ""
    echo "[Optional] Downloading small evaluation dataset..."
    bash scripts/download_dataset.sh small || echo "  WARNING: Dataset download failed (run manually: bash scripts/download_dataset.sh)"
fi

echo ""
echo "============================================================"
echo "Setup complete!"
echo "============================================================"
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To start the server:"
echo "  bash infrastructure/scripts/start_inference.sh"
echo ""
echo "To test models:"
echo "  python eval/test_each_model.py"
echo ""
echo "To download more datasets:"
echo "  bash scripts/download_dataset.sh medium   # 15 GB"
echo "  bash scripts/download_dataset.sh full     # 40 GB"
echo ""
