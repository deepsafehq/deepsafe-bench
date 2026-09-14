#!/bin/bash
# Post-checkout setup: symlinks, patches, directories.
# Model code repos are now vendored in git — no cloning needed.
#
# Usage: bash monolith/clone_repos.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "Setting up model code paths..."

# Create required directories and symlinks
mkdir -p logs
mkdir -p /app/weights 2>/dev/null || true

# ShiftySpeech/Nes2Net need xlsr2_300m.pt (hardcoded paths in model code)
XLSR_PT="models/audio/shiftyspeech/weights/xlsr_53_56k.pt"
if [ -f "$XLSR_PT" ]; then
    ln -sf "$REPO_ROOT/$XLSR_PT" models/xlsr2_300m.pt 2>/dev/null || true
    ln -sf "$REPO_ROOT/$XLSR_PT" /app/weights/xlsr2_300m.pt 2>/dev/null || true
    echo "  Symlinked xlsr2_300m.pt for ShiftySpeech/Nes2Net"
fi

# Fix missing __init__.py in Effort utils
touch models/image/effort/code/DeepfakeBench/training/utils/__init__.py 2>/dev/null || true

# NPR_VIDEO and UNIVFD_VIDEO reuse image model weights
if [ -f "models/image/npr/weights/NPR.pth" ]; then
    mkdir -p models/video/npr_video/weights
    ln -sf "$REPO_ROOT/models/image/npr/weights/NPR.pth" models/video/npr_video/weights/NPR.pth 2>/dev/null || true
    echo "  Symlinked npr_video/weights/NPR.pth -> image/npr"
fi
if [ -f "models/image/universal/weights/fc_weights.pth" ]; then
    mkdir -p models/video/univfd_video/weights
    ln -sf "$REPO_ROOT/models/image/universal/weights/fc_weights.pth" models/video/univfd_video/weights/fc_weights.pth 2>/dev/null || true
    echo "  Symlinked univfd_video/weights/fc_weights.pth -> image/universal"
fi

# FakeSTormer: HF downloads code to fake-stormer/model_code/, create symlink
if [ -d "models/video/fakestormer/fake-stormer/model_code" ] && [ ! -e "models/video/fakestormer/code" ]; then
    ln -sf fake-stormer/model_code models/video/fakestormer/code
    echo "  Symlinked fakestormer/code -> fake-stormer/model_code"
fi

# Video model weight paths: HF repo uses hyphens (dfd-fcg, fake-stormer, pwtf-dvd)
# but config.py uses underscores (dfd_fcg, fakestormer, pwtf_dvd). Symlink weights/.
for pair in "fakestormer:fake-stormer" "dfd_fcg:dfd-fcg" "pwtf_dvd:pwtf-dvd"; do
    cfg_name="${pair%%:*}"
    hf_name="${pair##*:}"
    if [ -d "models/video/$hf_name/weights" ] && [ ! -e "models/video/$cfg_name/weights" ]; then
        ln -sfn "../$hf_name/weights" "models/video/$cfg_name/weights"
        echo "  Symlinked $cfg_name/weights -> $hf_name/weights"
    fi
done

# Effort: patch SVD to use GPU (CPU SVD takes 30+ min, GPU takes <2 min)
EFFORT_DET="models/image/effort/code/DeepfakeBench/training/detectors/effort_detector.py"
if [ -f "$EFFORT_DET" ]; then
    if ! grep -q "_svd_device" "$EFFORT_DET"; then
        sed -i 's|U, S, Vh = torch.linalg.svd(module.weight.data, full_matrices=False)|_svd_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")\n        U, S, Vh = torch.linalg.svd(module.weight.data.to(_svd_device), full_matrices=False)\n        U, S, Vh = U.cpu(), S.cpu(), Vh.cpu()|' "$EFFORT_DET"
        echo "  Patched Effort SVD to use GPU"
    else
        echo "  Effort SVD patch already applied"
    fi
fi

echo "Done! Model code setup complete."
