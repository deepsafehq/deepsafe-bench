#!/bin/bash
# Download a DeepSafe evaluation dataset tier from HuggingFace.
#
# Usage:
#   bash scripts/download_dataset.sh            # small tier (default, ~1.7 GB)
#   bash scripts/download_dataset.sh small
#   bash scripts/download_dataset.sh medium     # ~10 GB, the standard benchmark
#   bash scripts/download_dataset.sh full       # ~25 GB
#   bash scripts/download_dataset.sh all
#
# The dataset is GATED because it aggregates sources with differing terms
# (ASVspoof, DF40, LAV-DF and others). Accepting the gate is one click:
#   https://huggingface.co/datasets/deepsafe/evaluation-dataset
# then `hf auth login`.
#
# Model weights are NOT gated. You do not need this dataset to run detection,
# only to reproduce or extend the benchmark.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ -d "$REPO_ROOT/.venv" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv/bin/activate"
fi

TIER="${1:-small}"
DATASET_DIR="${DEEPSAFE_DATASET_DIR:-$REPO_ROOT/dataset}"
HF_REPO="deepsafe/evaluation-dataset"

export HF_XET_HIGH_PERFORMANCE=1

case "$TIER" in
    small)  HF_DIR="master_eval_small"; SIZE="~1.7 GB (198 samples)"   ;;
    medium) HF_DIR="master_eval";       SIZE="~10 GB (15,454 samples)" ;;
    full)   HF_DIR="master_eval_full";  SIZE="~25 GB (45,954 samples)" ;;
    all)
        for t in small medium full; do bash "$0" "$t"; done
        exit 0
        ;;
    *)
        echo "Unknown tier: $TIER" >&2
        echo "Usage: $0 [small|medium|full|all]" >&2
        exit 1
        ;;
esac

echo "Downloading '$TIER' tier $SIZE from $HF_REPO..."

python3 - "$HF_REPO" "$HF_DIR" "$DATASET_DIR" <<'PY'
import os
import sys

repo, subdir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]

try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit(
        "huggingface_hub is not installed.\n"
        "    uv pip install huggingface_hub"
    )

token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN") or None

try:
    snapshot_download(
        repo,
        repo_type="dataset",
        allow_patterns=f"{subdir}/**",
        local_dir=out_dir,
        max_workers=4,
        token=token,
    )
except Exception as exc:  # noqa: BLE001 - the message matters more than the type
    sys.exit(
        f"Download failed: {exc}\n\n"
        f"This dataset is gated. To get access:\n"
        f"  1. Open https://huggingface.co/datasets/{repo}\n"
        f"  2. Accept the terms (one click)\n"
        f"  3. Run: hf auth login\n\n"
        f"You do NOT need this to run detection. Model weights are ungated;\n"
        f"the dataset is only required to reproduce or extend the benchmark."
    )
print("Download complete.")
PY

echo "Dataset at: $DATASET_DIR/$HF_DIR"
