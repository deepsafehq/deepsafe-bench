# DeepSafe - Full VM Deployment Guide

Complete guide to deploying the DeepSafe backend (Gateway + Inference Server + Cloudflare Tunnel) on a fresh GPU VM. No macOS dependencies.

## Architecture

```
Browser -> localhost:3000 (Cloudflare Pages, static)
              |
              v
         localhost:8000 (Cloudflare Tunnel)
              |
              v
         Gateway (port 8000) -- auth, billing, rate limiting, API keys, dashboard
              |
              v
         Inference Server (port 8001) -- 24 ML models, ensemble inference
```

**Services on the VM:**
| Service | Port | Purpose |
|---------|------|---------|
| Gateway | 8000 | External API (auth, billing, detection orchestration) |
| Inference Server | 8001 | Internal ML inference (24 models) |
| Redis | 6379 | Rate limiting, job state |
| Cloudflared | -- | Tunnel to Cloudflare edge |

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 40 GB | 48+ GB |
| CPU RAM | 32 GB | 64 GB |
| Disk | 100 GB | 200 GB |
| Python | 3.11 or 3.12 | 3.12 |
| CUDA Driver | 535+ | Latest |
| OS | Ubuntu 22.04+ | Ubuntu 24.04 |

### Tested GPUs

| GPU | VRAM | Works? | Notes |
|-----|------|--------|-------|
| RTX A6000 | 48 GB | Yes | Recommended, all 24 models fit |
| RTX 5880 Ada | 48 GB | Yes | Tested, 17.5 GB used |
| A100 80 GB | 80 GB | Yes | Plenty of headroom |
| A100 40 GB | 40 GB | Yes (tight) | May OOM on parallel video inference |
| H100 | 80 GB | Yes | Fastest |
| V100 32 GB | 32 GB | Partial | ~11 models fit |
| RTX 5090 / Blackwell | -- | **NO** | sm_120 needs PyTorch 2.6+, breaks fairseq |

## Quick Deploy (vast.ai or any GPU VM)

### Step 1: Clone the repo

```bash
# If the VM has GitHub access:
git clone --depth 1 https://github.com/deepsafehq/deepsafe.git /workspace/DeepSafe
cd /workspace/DeepSafe

# Or use a PAT for private repo:
git clone --depth 1 https://x-access-token:YOUR_GH_PAT@github.com/deepsafehq/deepsafe.git /workspace/DeepSafe
```

### Step 2: Install PyTorch + dependencies

```bash
cd /workspace/DeepSafe

# PyTorch 2.5.1 + CUDA 12.1
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

# All inference server dependencies
pip install -r apps/inference/requirements.lock \
  --index-url https://download.pytorch.org/whl/cu121 \
  --extra-index-url https://pypi.org/simple

# Videoseal (installed separately due to timm conflict)
pip install videoseal==1.0.1 --no-deps

# Gateway dependencies
pip install -r apps/gateway/requirements.txt
```

### Step 3: Build fairseq from source

```bash
# fairseq must be built from source (pip install fairseq breaks on Python 3.11+)
FAIRSEQ_TMP="/tmp/fairseq_src"
git clone --depth 1 https://github.com/facebookresearch/fairseq.git "$FAIRSEQ_TMP"
READTHEDOCS=1 pip install --no-deps --no-build-isolation "$FAIRSEQ_TMP"
rm -rf "$FAIRSEQ_TMP"

# Additional fairseq runtime deps
pip install cython bitarray portalocker sacrebleu colorama lxml tabulate

# Patch fairseq for Python 3.11+/3.12+ compatibility
python3 apps/inference/patch_fairseq.py
```

### Step 4: Download model weights (~40 GB)

```bash
export HUGGINGFACE_TOKEN=hf_your_token_here
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
bash apps/inference/pull_weights.sh
```

This downloads from `deepsafe/deepsafe-services` on HuggingFace into `models/`.

### Step 5: Setup model code paths + symlinks

```bash
bash apps/inference/clone_repos.sh
```

This creates weight symlinks (fakestormer, dfd_fcg, pwtf_dvd), patches Effort SVD, and links XLS-R weights for audio models.

### Step 6: Pre-download face detection + external models

```bash
# SCRFD face detection (used by video models)
python3 -c "
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_sc', allowed_modules=['detection'],
                   providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))
print('SCRFD cached')
"

# External model weights (CLIP, SigLIP, etc.) -- should already exist from HF download
python3 apps/inference/prefetch_externals.py --check
```

### Step 7: Install Redis + Cloudflared

```bash
# Redis (needed for gateway rate limiting)
apt-get update && apt-get install -y redis-server
redis-server --daemonize yes

# Cloudflared
curl -fsSL -o /usr/local/bin/cloudflared \
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x /usr/local/bin/cloudflared
```

### Step 8: Configure Cloudflare Tunnel

```bash
mkdir -p /etc/cloudflared

# Write tunnel credentials (get from existing deployment or Cloudflare dashboard)
cat > /etc/cloudflared/TUNNEL_ID.json << 'EOF'
{"AccountTag":"YOUR_ACCOUNT_TAG","TunnelSecret":"YOUR_SECRET","TunnelID":"YOUR_TUNNEL_ID"}
EOF

# Write tunnel config
cat > /etc/cloudflared/config.yml << 'EOF'
tunnel: YOUR_TUNNEL_ID
credentials-file: /etc/cloudflared/YOUR_TUNNEL_ID.json

ingress:
  - hostname: localhost:8000
    service: http://localhost:8000
  - service: http_status:404
EOF
```

### Step 9: Generate gateway config

The gateway needs a `deepsafe_config.json` mapping model names to inference server endpoints:

```bash
cd /workspace/DeepSafe/apps/inference
python3 -c "
from config import MODEL_REGISTRY
import json

config = {'media_types': {}, 'default_api_timeout_seconds': 600, 'default_max_retries': 1}
for name, info in MODEL_REGISTRY.items():
    m = info.modality
    if m not in config['media_types']:
        config['media_types'][m] = {'model_endpoints': {}, 'health_endpoints': {}}
    config['media_types'][m]['model_endpoints'][name] = f'http://localhost:8001/models/{name}/predict'
    config['media_types'][m]['health_endpoints'][name] = 'http://localhost:8001/health'

with open('../../deepsafe_config.json', 'w') as f:
    json.dump(config, f, indent=2)
print(f'Config: {sum(len(v[\"model_endpoints\"]) for v in config[\"media_types\"].values())} model endpoints')
"
```

### Step 10: Create gateway environment file

```bash
cat > /workspace/DeepSafe/apps/gateway/.env.vm << 'EOF'
DEEPSAFE_CONFIG_FILE_PATH=/workspace/DeepSafe/deepsafe_config.json
DEEPSAFE_ENV=production
DATABASE_URL=postgresql://YOUR_SUPABASE_CONNECTION_STRING
SUPABASE_URL=https://YOUR_PROJECT.supabase.co
SUPABASE_JWT_SECRET=YOUR_JWT_SECRET
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
META_MODEL_ARTIFACTS_DIR=/workspace/DeepSafe/models/ensemble/artifacts
PORT=8000
WORKERS=1
EOF
```

### Step 11: Start everything

```bash
bash apps/inference/start_all.sh
```

Or manually:

```bash
# 1. Redis
redis-server --daemonize yes

# 2. Inference Server (port 8001, ~7 min to load 24 models)
cd /workspace/DeepSafe/apps/inference
DEEPSAFE_PORT=8001 nohup python3 server.py > /tmp/inference.log 2>&1 &

# 3. Gateway (port 8000)
set -a && source apps/gateway/.env.vm && set +a
export PYTHONPATH=/workspace/DeepSafe/apps/gateway
cd /workspace/DeepSafe/apps/gateway
nohup python3 main.py > /tmp/gateway.log 2>&1 &

# 4. Cloudflared
nohup cloudflared tunnel --config /etc/cloudflared/config.yml run > /tmp/cloudflared.log 2>&1 &
```

### Step 12: Verify

```bash
# Wait ~7 minutes for models to load, then:
curl -s http://localhost:8001/health | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Inference Server: {d[\"status\"]}, {d[\"models_loaded\"]} models')"
curl -s http://localhost:8000/health | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Gateway: {d[\"status\"]}')"
curl -s http://localhost:8000/health | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Public: {d[\"status\"]}')"
```

## Disk Space Breakdown

| Component | Size |
|-----------|------|
| Model weights (HuggingFace) | ~40 GB |
| External caches (CLIP, SigLIP, etc.) | ~6 GB |
| Python packages | ~8 GB |
| Repo code + model code | ~1 GB |
| **Total** | **~55 GB** |

## Logs

| Service | Log file |
|---------|----------|
| Inference Server | `/tmp/inference.log` |
| Gateway | `/tmp/gateway.log` |
| Cloudflared | `/tmp/cloudflared.log` |

## Troubleshooting

### Models fail to load
- Check `/tmp/inference.log` for specific model errors
- Weight symlink issues: re-run `bash apps/inference/clone_repos.sh`
- Missing weights: re-run `bash apps/inference/pull_weights.sh`

### Gateway returns 503
- Inference server not running or still loading: check `curl http://localhost:8001/health`
- Wrong `deepsafe_config.json` endpoints: regenerate with the Python script above

### "Failed to store uploaded file"
- The frontend is sending `async=true` but MinIO is not running
- Fix: ensure the frontend does NOT send `async=true` (current main branch is fixed)

### Dashboard shows "Not Found"
- Gateway is not running on port 8000 (inference server alone does not have dashboard endpoints)
- Fix: start the gateway process

### Cloudflare tunnel not connecting
- Verify credentials file matches your tunnel ID
- Check `cloudflared tunnel list` from a machine with the cert.pem

### RTX 5090 / Blackwell GPU
- PyTorch 2.5.1 does not support sm_120 (Blackwell architecture)
- Need PyTorch 2.6+ which breaks fairseq
- Use A100/H100/A6000/RTX 5880 Ada instead

## Services Not Required

| Service | Why not needed |
|---------|---------------|
| MinIO | Only for async detection (Celery). Sync mode reads files in-process. |
| Celery | Only for async detection. The web UI uses sync mode. |
| PostgreSQL (local) | Supabase provides hosted PostgreSQL. |
| Nginx | Cloudflared handles reverse proxy. |
| Docker | Inference server runs natively. No containers. |

## HuggingFace Repos

- **Weights**: [deepsafe/deepsafe-services](https://huggingface.co/deepsafe/deepsafe-services) (~40 GB)
- **Dataset**: [deepsafe/evaluation-dataset](https://huggingface.co/datasets/deepsafe/evaluation-dataset) (3-tier eval)
