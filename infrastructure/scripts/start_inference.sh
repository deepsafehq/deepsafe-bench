#!/bin/bash
# DeepSafe Inference Server — Production startup script.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Check GPU availability
if command -v nvidia-smi &>/dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. Running on CPU."
fi

# Defaults
export DEEPSAFE_MODELS="${DEEPSAFE_MODELS:-all}"
export DEEPSAFE_MODELS_ROOT="${DEEPSAFE_MODELS_ROOT:-$REPO_ROOT/models}"
export DEEPSAFE_PORT="${DEEPSAFE_PORT:-8001}"
export DEEPSAFE_MAX_WORKERS="${DEEPSAFE_MAX_WORKERS:-3}"
export PYTHONPATH="${REPO_ROOT}/apps/inference:${REPO_ROOT}/packages/shared:${PYTHONPATH:-}"

echo "Starting DeepSafe Inference Server..."
echo "  Models root: $DEEPSAFE_MODELS_ROOT"
echo "  Models: $DEEPSAFE_MODELS"
echo "  Port: $DEEPSAFE_PORT"

cd "$REPO_ROOT/apps/inference"

# Optional smoke test before serving
if [ "${1:-}" = "--smoke-test" ]; then
    echo "Running smoke test..."
    python3 smoke_test.py
    echo "Smoke test passed."
fi

exec python3 server.py
