#!/bin/bash
# DeepSafe — Full platform setup for a fresh VM.
#
# This single script sets up everything: inference server (GPU models),
# API gateway, frontend, and configuration.
#
# Usage:
#   git clone https://github.com/deepsafehq/deepsafe.git
#   cd deepsafe
#   bash setup.sh
#
# Requirements:
#   - Ubuntu 22.04+ / Debian 12+
#   - Python 3.11 or 3.12
#   - NVIDIA GPU with CUDA 12.x drivers (for inference)
#   - Node.js 18+ and pnpm 9+ (for frontend)
#   - Redis server (for rate limiting and Celery)
#   - A .env file with required credentials (see infrastructure/env/.env.example)
#
# What this script does:
#   1. Sets up the Python environment (uv, venv, PyTorch, dependencies)
#   2. Installs the shared package (packages/shared)
#   3. Downloads model weights from HuggingFace (~34 GB)
#   4. Sets up the gateway (API server) dependencies
#   5. Sets up the frontend (pnpm install + build)
#   6. Verifies everything works
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

echo "============================================================"
echo "DeepSafe Platform Setup"
echo "============================================================"
echo "Repo root:  $REPO_ROOT"
echo "Python:     $(python3 --version 2>&1)"
echo "Node:       $(node --version 2>&1 || echo 'not installed')"
echo ""

# ── Pre-flight checks ────────────────────────────────────────────────────────

if [ ! -f ".env" ] && [ ! -f "infrastructure/env/.env.example" ]; then
    echo "ERROR: No .env file found. Copy the template first:"
    echo "  cp infrastructure/env/.env.example .env"
    echo "  # Then edit .env with your credentials"
    exit 1
fi

if [ ! -f ".env" ]; then
    echo "WARNING: No .env file found. Copying template..."
    cp infrastructure/env/.env.example .env
    echo "  Created .env from template. Edit it with your credentials before starting services."
fi

# ── Step 1: Inference server setup (GPU models + PyTorch) ────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Step 1/5: Setting up Inference Server (ML models)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
bash apps/inference/setup.sh

# ── Step 2: Install shared package ───────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Step 2/5: Installing shared package"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
source .venv/bin/activate
uv pip install -e packages/shared
echo "  Shared package installed."

# ── Step 3: Gateway dependencies ─────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Step 3/5: Installing Gateway dependencies"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
uv pip install -r apps/gateway/requirements.txt
uv pip install sentry-sdk[fastapi]
echo "  Gateway dependencies installed."

# ── Step 4: Frontend build ───────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Step 4/5: Building Frontend"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
if command -v pnpm &>/dev/null; then
    pnpm install --frozen-lockfile
    echo "  Frontend dependencies installed."
else
    echo "  WARNING: pnpm not found. Install with: npm install -g pnpm@9"
    echo "  Then run: pnpm install"
fi

# ── Step 5: Verify ───────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Step 5/5: Verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Verify Python imports
PYTHONPATH="$REPO_ROOT/apps/inference:$REPO_ROOT/apps/gateway:$REPO_ROOT/packages/shared" \
python3 -c "
from deepsafe_shared.ensemble import calculate_ensemble_verdict_api
from deepsafe_shared.device import get_device
print('  Shared package imports: OK')
" 2>/dev/null && echo "" || echo "  WARNING: Shared package import check failed"

# Verify inference smoke test (if models are downloaded)
if [ -d "models/image/npr/weights" ]; then
    echo "  Model weights: found"
else
    echo "  Model weights: not yet downloaded (run with HUGGINGFACE_TOKEN set)"
fi

echo ""
echo "============================================================"
echo "Setup complete!"
echo "============================================================"
echo ""
echo "To start the services:"
echo ""
echo "  # 1. Source environment"
echo "  source .venv/bin/activate"
echo "  source .env"
echo ""
echo "  # 2. Start inference server (port 8001)"
echo "  bash infrastructure/scripts/start_inference.sh"
echo ""
echo "  # 3. Start API gateway (port 8000)"
echo "  bash infrastructure/scripts/start_gateway.sh"
echo ""
echo "  # 4. Verify health"
echo "  bash infrastructure/scripts/health_check.sh"
echo ""
echo "  # 5. (Optional) Start frontend dev server"
echo "  cd apps/web && pnpm dev"
echo ""
