#!/bin/bash
# DeepSafe VM - Start all services
# Usage: bash monolith/start_all.sh
#
# Starts: Redis, Monolith (port 8001), Gateway (port 8000), Cloudflared tunnel
# Logs:   /tmp/monolith.log, /tmp/gateway.log, /tmp/cloudflared.log

set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

echo "============================================================"
echo "DeepSafe — Starting all services"
echo "============================================================"

# ── 1. Redis ───────────────────────────────────────────────────────────────
echo "[1/4] Starting Redis..."
if redis-cli ping > /dev/null 2>&1; then
    echo "  Redis already running"
else
    redis-server --daemonize yes
    echo "  Redis started"
fi

# ── 2. Monolith (port 8001) ───────────────────────────────────────────────
echo "[2/4] Starting Monolith inference server on port 8001..."
pkill -f 'python3 -m monolith.server' 2>/dev/null || true
sleep 1
cd "$REPO_DIR"
DEEPSAFE_PORT=8001 nohup python3 -m monolith.server > /tmp/monolith.log 2>&1 &
echo "  Monolith PID: $! (loading ~7 min, logs: /tmp/monolith.log)"

# ── 3. Gateway (port 8000) ────────────────────────────────────────────────
echo "[3/4] Starting API Gateway on port 8000..."
pkill -f 'python3 main.py' 2>/dev/null || true
sleep 1

# Load env from .env file or use defaults
if [ -f "$REPO_DIR/apps/gateway/.env.vm" ]; then
    set -a
    source "$REPO_DIR/apps/gateway/.env.vm"
    set +a
fi

export DEEPSAFE_CONFIG_FILE_PATH="${DEEPSAFE_CONFIG_FILE_PATH:-$REPO_DIR/deepsafe_config.json}"
export DEEPSAFE_ENV="${DEEPSAFE_ENV:-production}"
export CELERY_BROKER_URL="${CELERY_BROKER_URL:-redis://localhost:6379/0}"
export CELERY_RESULT_BACKEND="${CELERY_RESULT_BACKEND:-redis://localhost:6379/0}"
export META_MODEL_ARTIFACTS_DIR="${META_MODEL_ARTIFACTS_DIR:-$REPO_DIR/models/ensemble/artifacts}"
export PORT="${PORT:-8000}"
export WORKERS="${WORKERS:-1}"
export PYTHONPATH="$REPO_DIR/apps/gateway"

cd "$REPO_DIR/apps/gateway"
nohup python3 main.py > /tmp/gateway.log 2>&1 &
echo "  Gateway PID: $! (logs: /tmp/gateway.log)"

# ── 4. Cloudflared tunnel ─────────────────────────────────────────────────
echo "[4/4] Starting Cloudflared tunnel..."
if pgrep -f 'cloudflared.*tunnel.*run' > /dev/null 2>&1; then
    echo "  Cloudflared already running"
else
    nohup cloudflared tunnel --config /etc/cloudflared/config.yml run > /tmp/cloudflared.log 2>&1 &
    echo "  Cloudflared PID: $! (logs: /tmp/cloudflared.log)"
fi

echo ""
echo "============================================================"
echo "All services started!"
echo ""
echo "  Monolith:    http://localhost:8001 (internal)"
echo "  Gateway:     http://localhost:8000 (external)"
echo "  Cloudflared: localhost:8000 -> localhost:8000"
echo ""
echo "Wait ~7 minutes for all 24 models to load, then test:"
echo "  curl http://localhost:8000/health"
echo "============================================================"
