#!/bin/bash
# DeepSafe Health Check — Verifies both services are running.
set -euo pipefail

GATEWAY_URL="${GATEWAY_URL:-http://localhost:8000}"
INFERENCE_URL="${INFERENCE_URL:-http://localhost:8001}"

echo "Checking DeepSafe services..."

# Check gateway
if curl -sf "$GATEWAY_URL/health" > /dev/null 2>&1; then
    echo "  Gateway: OK ($GATEWAY_URL)"
else
    echo "  Gateway: FAIL ($GATEWAY_URL)" >&2
    exit 1
fi

# Check inference
if curl -sf "$INFERENCE_URL/health" > /dev/null 2>&1; then
    echo "  Inference: OK ($INFERENCE_URL)"
else
    echo "  Inference: FAIL ($INFERENCE_URL)" >&2
    exit 1
fi

echo "All services healthy."
