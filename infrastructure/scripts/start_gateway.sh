#!/bin/bash
# DeepSafe Gateway — Production startup script.
# Reads ALL configuration from environment variables.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Validate required environment variables
required_vars=(DATABASE_URL SUPABASE_URL SUPABASE_JWT_SECRET)
for var in "${required_vars[@]}"; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: Required environment variable $var is not set." >&2
        exit 1
    fi
done

# Defaults
export DEEPSAFE_CONFIG_FILE_PATH="${DEEPSAFE_CONFIG_FILE_PATH:-$REPO_ROOT/deepsafe_config.json}"
export META_MODEL_ARTIFACTS_DIR="${META_MODEL_ARTIFACTS_DIR:-$REPO_ROOT/models/ensemble/artifacts}"
export PORT="${PORT:-8000}"
export WORKERS="${WORKERS:-$(nproc)}"
export PYTHONPATH="${REPO_ROOT}/apps/gateway:${REPO_ROOT}/packages/shared:${PYTHONPATH:-}"

echo "Starting DeepSafe Gateway..."
echo "  Config: $DEEPSAFE_CONFIG_FILE_PATH"
echo "  Port: $PORT"
echo "  Workers: $WORKERS"

cd "$REPO_ROOT/apps/gateway"
exec python3 -m uvicorn main:app --host 0.0.0.0 --port "$PORT" --workers "$WORKERS"
