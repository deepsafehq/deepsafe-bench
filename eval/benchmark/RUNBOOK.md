# GPU Benchmarking Runbook


> **Historical note.** Sections of this runbook describe the Docker
> microservice architecture used before the 2026-04-08 migration to a
> single-process monolith. References to per-model containers are kept
> for the GPU troubleshooting value, not because that setup still exists.

Reproducible guide for running DeepSafe benchmarking experiments on any cloud GPU VM.

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 48 GB (per-modality loading) | 80+ GB (all models loaded) |
| CPU RAM | 128 GB | 200+ GB |
| Disk | 120 GB | 200+ GB |
| OS | Ubuntu 22.04 | Ubuntu 22.04 |
| NVIDIA Driver | 535+ | 535+ |
| Docker | 28.x | 28.x |
| Docker Compose | v2.x | v2.x |

## Disk Budget

| Component | Size |
|-----------|------|
| OS + Docker engine | ~9 GB |
| Repo + service weights (from HuggingFace) | ~36 GB |
| Dataset master_eval_full | ~25 GB |
| Docker images (shared CUDA bases) | ~15 GB |
| Docker images (per-service layers) | ~40 GB |
| Build cache + results | ~10 GB |
| **Total** | **~135 GB** |

## CRITICAL: HuggingFace vs GitHub Code Sync Issue

**The #1 source of bugs.** When deploying, we rsync the HuggingFace `deepsafe-services` repo over the GitHub code. HuggingFace has old `app.py` files with bugs (e.g., `total_mem` instead of `total_memory`). GitHub has the fixes.

**THE CORRECT DEPLOY SEQUENCE:**
```bash
# 1. Clone from GitHub (has all fixes)
git clone https://github.com/deepsafehq/deepsafe.git

# 2. Download ONLY weights/model_code from HuggingFace
huggingface-cli download deepsafe/deepsafe-services --local-dir services_hf

# 3. Rsync HuggingFace over GitHub
rsync -a services_hf/ services/

# 4. CRITICAL: Restore GitHub code (undoes HF's outdated app.py files)
git checkout -- services/

# 5. Cleanup
rm -rf services_hf
```

Step 4 (`git checkout -- services/`) is the key step that most people forget. Without it, the HF `app.py` files overwrite our fixes.

**Why this happens:** The HF services repo is a snapshot from a previous `push_services_to_hf.py` run. If we fix bugs in app.py after the last HF push, those fixes are only in GitHub. The HF push script should be run after every fix to keep them in sync.

## Quick Start (New VM)

```bash
# 1. Clone repo
git clone https://github.com/deepsafehq/deepsafe.git
cd deepsafe

# 2. Pull service weights from HuggingFace
pip3 install huggingface_hub
python3 -c "
from huggingface_hub import login, snapshot_download
login(token='YOUR_HF_TOKEN')
snapshot_download('deepsafe/deepsafe-services', repo_type='model', local_dir='./services_hf', max_workers=8)
"
rsync -a ./services_hf/ ./services/
rm -rf ./services_hf

# 3. Pull dataset
python3 -c "
from huggingface_hub import login, snapshot_download
login(token='YOUR_HF_TOKEN')
snapshot_download('deepsafe/evaluation-dataset', repo_type='dataset', local_dir='./dataset_hf', allow_patterns='master_eval_full/**', max_workers=2)
"
ln -sfn ./dataset_hf ./dataset

# 4. Setup VM infrastructure
bash eval/benchmark/setup_vm.sh

# 5. Copy .env file (get from team)
cp /path/to/.env .env

# 6. Run experiments
python3 eval/benchmark/profile_wave.py --modality image
python3 eval/benchmark/accuracy_runner.py --modality image
python3 eval/benchmark/compute_metrics.py --modality image
python3 eval/benchmark/ensemble_lab.py --modality image
# Repeat for audio, video
python3 eval/benchmark/gpu_tier_model.py
python3 eval/benchmark/generate_report.py
```

## Known Issues & Fixes

### Issue Log

Each issue is tagged with the system where it was encountered.

---

#### [RTX PRO 6000] HuggingFace rate limiting on dataset download (2026-04-04)

**Problem:** `snapshot_download` with `max_workers=8` triggers HTTP 429 rate limits from HuggingFace after ~1000 files. Download stalls and eventually crashes.

**Fix:** Use `max_workers=2` instead of 8. The download is resumable (HF client tracks completed files), so just restart if it dies:
```python
snapshot_download(..., max_workers=2)
```

**Prevention:** Always use `max_workers=2` for datasets with 10K+ files.

---

#### [RTX PRO 6000] Docker GPU access requires nvidia-container-toolkit (2026-04-04)

**Problem:** Fresh Ubuntu 22.04 VMs with NVIDIA drivers don't have the container toolkit pre-installed. `docker run --gpus all` fails.

**Fix:**
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update && apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
```

**Prevention:** Run `eval/benchmark/setup_vm.sh` which handles this automatically.

---

#### [RTX PRO 6000] GPG key import fails without --batch flag (2026-04-04)

**Problem:** `gpg --dearmor` fails with "no such device or address" when run non-interactively (e.g., over SSH).

**Fix:** Always use `gpg --batch --yes --dearmor` for non-interactive installs.

---

#### [RTX PRO 6000] Volume mounts reference macOS paths (2026-04-04)

**Problem:** `docker-compose.yml` has `nes2net_detection` and `aasist3_detection` mounting from `/Volumes/16TB_Sid/...` which is the local Mac external drive path.

**Fix:** Replace with relative paths:
```bash
sed -i 's|/Volumes/16TB_Sid/DeepSafe/deepsafe_model_weights/audio/nes2net:/app/weights:ro|./services/audio/nes2net/weights:/app/weights:ro|' docker-compose.yml
sed -i 's|/Volumes/16TB_Sid/DeepSafe/deepsafe_model_weights/audio/aasist3:/app/weights:ro|./services/audio/aasist3/weights:/app/weights:ro|' docker-compose.yml
```

**Prevention:** Never use absolute host paths in docker-compose. Use `./services/` relative paths.

---

#### [A100 80GB] FakeSTormer mmcv build failure (2026-04-01)

**Problem:** FakeSTormer uses `mmcv==1.6.1` which requires PyTorch 1.8.0 and a specific CUDA 11.1 base image. Builds fail on newer CUDA.

**Fix:** FakeSTormer has its own Dockerfile with `pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel` base. Make sure to use the prod compose overlay which has the correct GPU deploy config.

---

#### [A100 80GB] ShiftySpeech fairseq dependency (2026-04-01)

**Problem:** ShiftySpeech depends on a custom `fairseq_ours` fork from the SafeEar repo. Build fails if the fork is not included in the service directory.

**Fix:** Ensure `services/audio/shiftyspeech/` contains the full model code from HuggingFace (not just Dockerfile + app.py).

---

#### [A100 80GB] Nes2Net weight loading (2026-04-01)

**Problem:** Nes2Net expects weights at `/app/weights/` but docker-compose mounts from external drive path.

**Fix:** Already fixed in docker-compose.yml to use `./services/audio/nes2net/weights:/app/weights:ro`.

---

#### [RTX PRO 6000] get_docker_services returns ModelDef not strings (2026-04-04)

**Problem:** `models.get_docker_services()` returns `List[ModelDef]` but `profile_wave.py` functions (`build_services`, `start_services`, `stop_services`) expect `list[str]` (docker-compose service names).

**Fix:** Extract `.docker_service` at call site:
```python
service_defs = get_docker_services(modality)
services = [s.docker_service for s in service_defs]
```

**Prevention:** When consuming `get_docker_services()`, always extract the string field you need.

---

#### [RTX PRO 6000 Blackwell] HuggingFace weight download timeout causes infinite restart loop (2026-04-04)

**Problem:** AIDE, CO-SPY, and Effort download foundation model weights (CLIP, SigLIP, ConvNeXt, SD VAE) from HuggingFace at container startup. This takes 10-20 minutes on first run. Docker's health check (interval=30s, start_period=120s, retries=3) marks them unhealthy and restarts them before weights finish downloading. Each restart re-downloads from scratch = infinite loop.

**Models affected and their weight downloads:**
- AIDE: ConvNeXt XXL (~1.5GB) from HuggingFace
- CO-SPY: SigLIP ViT-SO400M-14 + Stable Diffusion v1.4 VAE (~5GB total)
- Effort: CLIP ViT-L/14 (~1.5GB) from HuggingFace
- Universal: CLIP ViT-L/14 via OpenAI TorchScript (also needs first-run download, but crashes on Blackwell due to CUDA 11.3)

**Fix -- warm up containers before profiling:**
```bash
# 1. Start slow containers without port mapping, let them download
docker compose run -d --no-deps --name aide_warmup aide_detection
docker compose run -d --no-deps --name cospy_warmup cospy_detection
docker compose run -d --no-deps --name effort_warmup effort_detection

# 2. Wait 15-20 minutes for weight downloads to complete
sleep 1200

# 3. Stop warmup containers (weights are cached in Docker layers)
docker stop aide_warmup cospy_warmup effort_warmup
docker rm aide_warmup cospy_warmup effort_warmup

# 4. Now start normally -- weights are cached, startup will be fast
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

**UPDATE (2026-04-04): ROOT CAUSE WAS MEMORY LIMIT, NOT DOWNLOAD TIMEOUT.** The 4GB Docker memory limit (`memory: 4g` in docker-compose.prod.yml) was too small. Yermandy needed 4.47GB, AIDE's ConvNeXt XXL needed >4GB during loading. Containers were OOM-killed (exit code 137), not timing out.

**Actual fix:** Increase `memory` limit in docker-compose.prod.yml from 4g to 8g for all detection models. With 197GB RAM, this is not a constraint.

**Prevention:** Always check `docker stats` for memory pressure when containers keep restarting. Exit code 137 = OOM kill.

---

#### [RTX PRO 6000 Blackwell] Universal model incompatible with Blackwell (CUDA 11.3 tier) (2026-04-04)

**Problem:** UniversalFakeDetect uses Tier 2 Docker image (`pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime`) with PyTorch 1.11.0. This is too old for Blackwell GPUs (sm_120). Container crashes on startup.

**Fix:** None for Blackwell. Universal must be skipped on Blackwell GPUs. On A100/H100 (sm_80/sm_90) it works fine.

**Long-term fix:** Migrate Universal to Tier 3 (CUDA 12.1, PyTorch 2.5.1) -- requires updating the OpenAI CLIP TorchScript loading code.

---

#### [RTX PRO 6000] FSD Dockerfile has impossible scipy>=1.17.0 requirement (2026-04-04)

**Problem:** `services/image/fsd/Dockerfile` pinned `scipy>=1.17.0` and `pillow>=12.1.0`, but scipy 1.17 doesn't exist (latest is 1.15.x) and pillow 12.1 requires Python 3.11+.

**Fix:** Relaxed to `scipy>=1.10.0` and `pillow>=10.0.0`.

---

#### [RTX PRO 6000 Blackwell] total_mem AttributeError crashes ALL containers (2026-04-04)

**Problem:** ALL 21 service containers crash on startup on Blackwell GPUs (sm_120). PyTorch `torch.cuda.get_device_properties(0).total_mem` was renamed to `total_memory` in newer versions. Affected 20 files (17 app.py + 4 api.py for audio services). The CUDA warning about sm_120 compatibility is non-fatal, but the AttributeError is.

**Fix:** Global find-replace across all services:
```bash
find services -name "app.py" -o -name "api.py" | xargs sed -i 's/.total_mem /.total_memory /g'
```

**Prevention:** When using PyTorch device properties, use `total_memory` (not `total_mem`). Test on target GPU before deploying.

**Note:** The PyTorch warning `NVIDIA RTX PRO 6000 Blackwell with CUDA capability sm_120 is not compatible` is a WARNING only -- inference still works. Blackwell support is best-effort in PyTorch 2.5.1+cu121.

---

#### [RTX PRO 6000] wait_for_health API mismatch between profiler and gpu_utils (2026-04-04)

**Problem:** `profile_wave.py` called `wait_for_health(url_string, timeout_s=N)` but `gpu_utils.py` signature is `wait_for_health(port: int, timeout=N)` returning `tuple[bool, float]`.

**Fix:** Changed all call sites to pass `port` int and `timeout` kwarg, and unpack the tuple:
```python
healthy, wait_secs = wait_for_health(model.port, timeout=300)
```

**Prevention:** When writing scripts that import from sibling modules, verify function signatures match.

---

#### [RTX PRO 6000] Disk full from parallel Docker builds (2026-04-04)

**Problem:** Building image + audio containers simultaneously filled the 146GB disk. Each CUDA+PyTorch image is 9-16GB. 8 images = ~91GB on top of 60GB (repo+dataset) = disk full.

**Fix:** Build one modality wave at a time. After profiling a wave, remove those Docker images before building the next wave:
```bash
# After image wave, remove image containers
docker rmi $(docker images --filter "reference=deepsafe-*" -q) 2>/dev/null
docker system prune -f
docker builder prune -af
```

**Prevention:** On 150GB disks, never build more than one wave at a time. Budget: ~50GB for Docker images per wave, ~60GB for repo+dataset, ~40GB buffer.

---

#### [General] USE_GPU=false in base docker-compose (all systems)

**Problem:** Base `docker-compose.yml` has `USE_GPU=false` for some services. On GPU VMs this wastes the GPU.

**Fix:** Either use the prod overlay (`docker-compose.prod.yml` sets `USE_GPU=true`) or manually set:
```bash
sed -i 's/USE_GPU=false/USE_GPU=true/g' docker-compose.yml
```

---

## Experiment Execution Order

1. **Phase 0:** VM setup + dataset download + dataset audit
2. **Phase 1:** Image wave (build, profile, accuracy, metrics, ensemble)
3. **Phase 2:** Audio wave (same)
4. **Phase 3:** Video wave (same, expect build issues with 5 new models)
5. **Phase 4:** Full stack (all containers, combined VRAM, E2E latency)
6. **Phase 5:** Ensemble R&D (uses predictions from phases 1-3)
7. **Phase 6:** GPU tier cost modeling
8. **Phase 7:** Report generation

## Results Archive

After experiments, download all results:
```bash
scp -r root@VM_IP:/root/deepsafe/eval/results/ eval/results/$(hostname -s)/
scp root@VM_IP:/root/deepsafe/profiling/*_REPORT.md profiling/
scp root@VM_IP:/root/deepsafe/profiling/*_EXECUTIVE_SUMMARY.md profiling/
git add eval/results/ profiling/
git commit -m "docs(profiling): benchmark results from $(hostname -s)"
```

## CRITICAL: Blackwell (sm_120) GPU Incompatibility with PyTorch

**As of April 2026, NO PyTorch build supports NVIDIA Blackwell GPUs (sm_120).**

Tested:
- PyTorch 2.5.1+cu121 (stable): FAILS
- PyTorch 2.12.0.dev (nightly)+cu126: FAILS
- PyTorch nightly+cu128: No wheel available, falls back to cu126
- PyTorch nightly+cu130: No wheel available

All fail with: `CUDA error: no kernel image is available for execution on the device`

The GPU is detected correctly (model loading, VRAM allocation works), but actual tensor computation on GPU triggers the error because PyTorch's compiled CUDA kernels only include sm_50 through sm_90.

**Affected GPUs:** RTX PRO 6000, RTX 5090, RTX 5080, and all other Blackwell architecture cards.

**Workaround:** Use CPU mode (`USE_GPU=false`) for accuracy testing only. Latency/throughput data will not be representative.

**Recommended GPUs for DeepSafe benchmarking:**
- A100 80GB (sm_80) -- best value, fully tested
- H100 80GB (sm_90) -- premium, fully supported
- L40S 48GB (sm_89) -- budget option if VRAM fits
- RTX 4090 24GB (sm_89) -- cheapest, may not fit all models

#### [Lambda A100] Scripts hardcode /root/deepsafe but Lambda uses ubuntu user (2026-04-05)

**Problem:** All benchmark scripts had `/root/deepsafe` hardcoded for RESULTS_DIR, COMPOSE_FILE, etc. Lambda Stack VMs run as `ubuntu` user with home at `/home/ubuntu`. Permission denied on `/root/`.

**Fix:** Changed all paths to `/home/ubuntu/deepsafe`. Long-term fix: use relative path detection: `Path(__file__).resolve().parents[2]`.

**Prevention:** Never hardcode absolute paths in benchmark scripts. Use relative paths or environment variables.

---

## GPU-Specific Notes

### RTX PRO 6000 Blackwell (96 GB)
- Driver: 580.95.05, CUDA: 13.0
- Backward compatible with CUDA 11.1, 11.3, 12.1 containers
- 600W TDP - monitor power draw during sustained loads
- MPS works but may not be necessary with 96GB headroom

### A100 80GB (from prior run 2026-04-01)
- 38 GB VRAM for 15 models, 46% utilization
- Throughput plateau at 15.7 rps (I/O bound, not GPU bound)
- GPU util never exceeded 31%
- See `profiling/REPORT.md` for full data

### A100 40GB SXM4 (Lambda Labs, 2026-04-05)
- 17 GB VRAM for 8 image models (NPR, Yermandy, Universal, AIDE, Effort, CO-SPY + provenance)
- 9.5 GB VRAM for 5 audio models + provenance
- All models load and run correctly on sm_80 (unlike Blackwell sm_120)
- Video inference is extremely slow: ~10-30 min per video file per model (CPU-heavy face detection)
- Lambda Stack 22.04 has pre-installed Docker + NVIDIA toolkit (no setup needed)
- Runs as `ubuntu` user (not `root`) -- paths must account for `/home/ubuntu/`

---

## Lessons Learned (April 2026 Benchmarking Campaign)

### 1. GPU Architecture Compatibility
- **NEVER use Blackwell GPUs (RTX 5090, RTX PRO 6000, sm_120)** -- PyTorch has no CUDA kernel support as of April 2026
- **Safe GPU choices**: A100 (sm_80), H100 (sm_90), L40S (sm_89), RTX 4090 (sm_89)
- The GPU warning "not compatible" is sometimes non-fatal (model loading works), but actual tensor computation WILL fail

### 2. Docker Memory Limits
- Default `memory: 4g` in docker-compose.prod.yml is TOO LOW for large models
- Yermandy needs 4.5 GB, AIDE's ConvNeXt XXL needs 5+ GB during loading
- Set `memory: 8g` for all detection models, `memory: 12g` for heavy video models
- Exit code 137 = OOM kill -- always check `docker stats` for memory pressure

### 3. Health Check Timing
- Some models download HuggingFace weights on first startup (AIDE, CO-SPY, Effort, Universal)
- First startup can take 10-20 min for large model downloads (CLIP, SigLIP, ConvNeXt, SD VAE)
- Set `start_period: 1200s` (20 min) in docker-compose health checks
- Subsequent startups are fast (weights cached in Docker layer)

### 4. HuggingFace Code Overwrites
- The #1 recurring bug: `rsync` from HF overwrites GitHub `app.py` fixes
- ALWAYS run `git checkout -- services/` after rsync to restore GitHub code
- Run `scripts/push_services_to_hf.py` after every service code fix to keep HF in sync

### 5. Video Inference Speed
- Video is 100x slower than image/audio (face detection + frame extraction + per-frame inference)
- Each video file takes 10-30 min through 4-7 models
- Pre-flight health check in accuracy_runner.py is CRITICAL to skip dead models
- Without it, each unhealthy model adds 600s timeout per video file
- Consider: reducing video dataset size for quick evals, or running video separately with longer budget

### 6. Disk Space Management
- Each CUDA+PyTorch Docker image is 9-20 GB
- 21 containers = ~200+ GB of Docker images
- On 150 GB disks: build one modality wave at a time, prune between waves
- On 300+ GB disks: can keep all waves cached simultaneously
- Always run `docker builder prune -f` after builds to free cache

### 7. Python Version Conflicts
- FSD model_code requires Python >= 3.12 (pyproject.toml constraint)
- Ubuntu 22.04 ships Python 3.10 -- use deadsnakes PPA to install 3.12
- FakeSTormer requires mmcv==1.6.1 which needs PyTorch <= 1.13
- Never upgrade FakeSTormer beyond PyTorch 1.13 (mmcv v1 incompatible with PyTorch 2.x)

### 8. Benchmark Script Portability
- NEVER hardcode absolute paths in scripts (e.g., `/root/deepsafe/`, `/home/ubuntu/deepsafe/`)
- Use `_PROJECT_ROOT = Path(__file__).resolve().parents[2]` to compute paths relative to script location
- Lambda VMs run as `ubuntu`, TensorDock as `root` -- hardcoded paths break across providers

### 9. SSH Reliability for Long Commands
- SSH connections drop after ~2-5 min on cloud VMs
- Use `nohup ... &` for any command that takes more than 1 minute
- Use `ServerAliveInterval=30` in SSH config
- Check results by tailing log files, not by waiting for SSH commands to return

### 10. Dataset Symlink Gets Overwritten by git checkout
- The repo has a `dataset` symlink pointing to `/Volumes/16TB_Sid/DeepSafe/dataset` (the macOS external drive)
- This symlink is tracked in git, so `git checkout -- services/` also restores it
- On cloud VMs, the dataset lives at `dataset_hf/`, so we create `dataset -> dataset_hf`
- But any `git checkout` or `git pull` will RESTORE the old Mac symlink, breaking the dataset path
- **Fix after every git pull:** `rm dataset && ln -sfn dataset_hf dataset`
- **Long-term fix:** Add `dataset` to `.gitignore` so the symlink isn't tracked

### 11. How We Fixed Each Broken Model (Reference for Future Deployments)

**Universal (Image, port 5003)**
- Problem: requirements.txt pinned torch==1.11.0, overwriting Dockerfile's torch==2.5.1
- Fix: Removed torch/torchvision pins from requirements.txt, added `torch>=2.5.1` as floor safety net
- Also: Upgraded Dockerfile from pytorch:1.11.0-cuda11.3 to nvidia/cuda:12.1.1+torch 2.5.1

**FSD (Image, port 5005)**
- Problem: model_code pyproject.toml requires Python >=3.12, but container has 3.10
- Fix: Reverted to Python 3.10 (standard Ubuntu), used fallback chain for `pip install --no-deps --no-build-isolation -e ./model_code`
- Note: deadsnakes PPA for Python 3.12 failed on Lambda VMs (apt GPG issues)

**FakeSTormer (Video, port 7001)**
- Problem: Base image `pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel` has expired GPG keys (apt-get fails)
- Fix: Upgraded to `pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel` (valid keys, mmcv==1.6.1 compatible)
- Also: Added `ENV DEBIAN_FRONTEND=noninteractive` to prevent interactive timezone prompt
- Also: Added `hasattr(torch, "set_float32_matmul_precision")` guard (torch 1.13 may not have it)

**DFD-FCG (Video, port 7003)**
- Problem: `total_mem` AttributeError (HuggingFace code overwrote GitHub fix during rsync)
- Fix: Changed `total_mem` to `total_memory` in app.py (2 occurrences)
- Also: First startup takes 10+ min to download CLIP ViT-L/14 weights from HuggingFace
- Prevention: Always run `git checkout -- services/` after rsync from HuggingFace

**PwTF-DVD (Video, port 7005)**
- Problem: Missing `simplejson` module (required by vendored SlowFast codebase)
- Fix: Added `simplejson` to requirements.txt

**AIDE, CO-SPY, Yermandy (Image)**
- Problem: Docker containers OOM-killed (exit code 137) with 4GB memory limit
- Fix: Increased `memory` in docker-compose.prod.yml from 4g to 8g
- Yermandy needed 4.47GB, AIDE's ConvNeXt XXL needed 5+ GB during loading

### 12. Dataset Download Rate Limiting
- HuggingFace rate-limits dataset downloads (HTTP 429) with high worker counts
- Use `max_workers=2` for datasets with 10K+ files (not 8)
- Downloads are resumable -- just restart on failure, already-downloaded files are skipped
- Audio downloads first (alphabetically), then images, then video
