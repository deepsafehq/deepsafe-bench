# Technology Stack (SaaS-Ready)

## Core Technologies
- **Languages**: Python 3.11/3.12 (Backend/ML), TypeScript 5.x (Frontend).
- **Backend Framework**: **FastAPI** - High-performance asynchronous API development with automatic OpenAPI documentation. The gateway (`apps/gateway/`) handles auth, billing, and routing. The inference server (`apps/inference/server.py`) serves all 24 models in a single FastAPI process.
- **Frontend Framework**: **Next.js 14 (App Router, static export)** & **React 18** - Static export deployed to Cloudflare Pages. Client-side rendering only (no SSR). **(Converted to static export 2026-03-26)**
- **Styling**: **Tailwind CSS** & **Framer Motion** - Utility-first CSS and fluid motion for a modern, responsive interface.
- **Database**: **Supabase PostgreSQL** (hosted, via **SQLAlchemy**) -- Managed Postgres via Supabase, shared project with Auth. Session-mode pooler for the app, direct connection for Alembic migrations. **(Migrated from local Docker Postgres 2026-03-31)**
- **Object Storage**: **S3-Compatible Storage (e.g., MinIO)** - Securely storing uploaded media to support model retraining and dataset creation.
- **Schema Validation**: **Pydantic V2** - Strict data validation and serialization for API requests and responses.

## Machine Learning & Data Processing
- **Deep Learning Framework**: **PyTorch 2.5.1 + CUDA 12.1** - The primary framework for model inference, pinned for reproducibility. All 24 models run in one Python process via the monolith inference server. **(Migrated from Docker microservices to monolith 2026-04-08)**
- **Model Serialization**: **Joblib** - For loading meta-learners, scalers, and imputers. Ensemble artifacts stored at `models/ensemble/artifacts/`.
- **Model Libraries**: **Transformers**, **MMCV-full** (vendored compatibility layer), **Lightning Fabric** - Specialized libraries for advanced detection models. **Fairseq** is built from source with Python 3.11+/3.12+ patches (`apps/inference/fairseq_compat.py` runtime monkey-patch, `apps/inference/patch_fairseq.py` post-install) required by SafeEar's SpeechTokenizer neural audio codec. **open-clip-torch** used by DFD-FCG and LipFD for CLIP ViT-L/14. PwTF-DVD vendors Facebook SlowFast codebase. MINTIME uses **facenet-pytorch** (MTCNN). Nes2Net uses **XLS-R 300M**.
- **Namespaced Imports**: Multiple model repos use generic top-level module names (`models`, `utils`, `data`, `src`, `code`). The `apps/inference/model_loader.py` provides `namespaced_import()` which imports each model's code under a unique prefix (e.g., `_ds_aide.models.AIDE`) to prevent `sys.modules` collisions when all 24 models are hot-loaded in one process.
- **Data Manipulation**: **Pandas**, **NumPy**, **OpenCV**, **PIL** - Efficient processing of model outputs, image transformations, and feature extraction.
- **Video Preprocessing**: Shared pipeline in `apps/inference/video_preprocess.py` using **SCRFD** face detection (insightface) with a single video decode pass. Replaces per-model MTCNN face detection from the Docker era, giving consistent face crops to all 7 video models.
- **Ensemble Logic**: Modality-specific strategies trained on 15,499-sample medium eval (2026-04-12). Image: LightGBM meta-learner (CV AUC=0.9705, 7 models: AIDE, CO-SPY, Effort, FSD, NPR, Universal, Yermandy) with Platt calibration and AUC-weighted-average fallback. Audio: Random Forest meta-learner (CV AUC=0.8748, 3 models: ShiftySpeech, SafeEar, Nes2Net) with Platt calibration and AUC-weighted-average fallback. Video: XGBoost meta-learner (CV AUC=0.8898, 9 models: FakeSTormer, SBI, DFD-FCG, PwTF-DVD, LipFD, RECCE, MINTIME, NPR-Video, UnivFD-Video) with Platt calibration and AUC-weighted-average fallback. **(Updated 2026-04-12)**

### Model Lineup (24 models)

**Image (7 models):**
| Model | Architecture | Paper |
|-------|-------------|-------|
| NPR | ResNet50 neural patterns | - |
| Yermandy | CLIP + LoRA | - |
| Universal | CLIP ViT-L/14 TorchScript | - |
| AIDE | DCT + ConvNeXt-XXL | ICLR 2025 |
| FSD | Forensic Self-Descriptions | CVPR 2025 |
| Effort | SVD + CLIP | ICML 2025 |
| CO-SPY | SigLIP + SD VAE | CVPR 2025 |

**Audio (3 models):**
| Model | Architecture | Paper |
|-------|-------------|-------|
| ShiftySpeech | XLSR wav2vec2 + AASIST (SSL) | - |
| SafeEar | SpeechTokenizer codec | 2024 |
| Nes2Net | XLS-R 300M + Nested Res2Net-TDNN | IEEE T-IFS 2025 |

**Video (7 models):**
| Model | Architecture | Paper |
|-------|-------------|-------|
| FakeSTormer | Swin Transformer temporal | ICCV 2025 |
| SBI | EfficientNet-B4 | CVPR 2022 |
| DFD-FCG | CLIP ViT-L/14 + facial guidance | CVPR 2025 |
| PwTF-DVD | SlowFast + temporal FFT | ICCV 2025 |
| LipFD | CLIP multi-scale region | NeurIPS 2024 |
| RECCE | Xception reconstruction anomaly | CVPR 2022 |
| MINTIME | TimeSformer multi-identity | IEEE T-IFS 2024 |

**Provenance (5 services):**
| Service | What It Detects | GPU |
|---------|----------------|-----|
| C2PA | Content Credentials manifests (31 AI generators, composite handling) | No |
| SDXL Watermark | Invisible watermarks from Stable Diffusion XL | No |
| TrustMark | Adobe/CAI invisible watermarks (survives social media re-encoding) | No |
| AudioSeal | Meta AudioSeal watermarks in audio | Yes |
| VideoSeal | Meta VideoSeal/PixelSeal watermarks in video | Yes |

- **Provenance Detection (2026-03-28, updated 2026-04-12):** C2PA Content Credentials (`c2pa-python`, 31 generator strings, file extension detection from magic bytes, `compositeWithTrainedAlgorithmicMedia` at 0.70, IPTC 2025.1 metadata extraction). SDXL invisible watermarks (`invisible-watermark`). TrustMark invisible watermarks (Adobe/CAI, MIT, `trustmark` package, BCH-coded, survives JPEG/resize/social media). AudioSeal audio watermarks (Meta, MIT). VideoSeal/PixelSeal video watermarks (Meta, MIT). SynthID disabled (reverse-engineered extractor has 100% FPR). Results feed into ensemble as Stage 2 provenance boost with tiered weights (0.20 for moderate signals >=0.65, 0.05 for weak signals, OVERRIDE_THRESHOLD=0.80). **(Updated 2026-04-12)**
- **Removed/Disabled Models:** AASIST3 removed -- KAN/GAT model construction hangs indefinitely (size=200 param). Individual AUC=0.199 (anti-correlated). Can re-add if initialization is fixed. SONICS removed - designed for AI-generated music (Suno/Udio), not speech deepfakes; AUC=0.367 on speech test set (worse than random). MM-Det disabled (LLaVA weight loading issue). B-Free (CVPR 2025, DINOv2) not deployed. SynthID disabled (100% FPR). LipForensics removed (reduces video AUC from 0.672 to 0.661, errors on 35% of non-face videos). TFCU removed (weights deleted from Baidu Cloud). **(Updated 2026-04-09)**
- **Multilingual Audio Models (2026-03-31):** Nes2Net (IEEE T-IFS 2025, XLS-R 300M + Nested Res2Net-TDNN). Added to address English bias: ShiftySpeech aggregate multilingual AUC=0.634, Nes2Net=0.876. **(2026-04-01):** Audio meta-learner retrained on 3346 samples (52 languages, 291 MLAAD generators + English archive). **(2026-04-09):** AASIST3 removed (KAN/GAT model construction hangs indefinitely).

## GPU Inference (Monolith) (2026-04-08)
- **Single-process GPU strategy:** All 24 models run in one Python process with a single CUDA context. Auto-detection: CUDA > MPS > CPU. No Docker containers, no NVIDIA MPS needed. **(Migrated from three-tier Docker CUDA strategy 2026-04-08)**
- **FP16 AMP inference:** `torch.amp.autocast` enabled by default on CUDA for 1.3-2x speedup on CLIP/ViT backbones. Disabled per-model where it causes issues (FSD: slower, NPR: no benefit). Auto-fallback to FP32 if AMP fails.
- **TF32 optimization:** `torch.backends.cudnn.benchmark`, `torch.backends.cudnn.allow_tf32`, and `torch.set_float32_matmul_precision('high')` enabled on Ampere+ GPUs for faster matmuls. Silently ignored on older GPUs.
- **GPU health reporting:** The `/health` endpoint reports `gpu_name`, `vram_used_mb`, `vram_total_mb` when CUDA is active. Startup logs warn if the server falls back to CPU.
- **VRAM footprint:** ~19.5 GB for all 24 models loaded simultaneously with FP16 weights.
- **Tested GPUs:** V100 32GB (partial, ~11 models), A100 40GB (works, tight), A100 80GB (works), RTX A6000 48GB (recommended), H100 80GB (works). RTX 5090/Blackwell (sm_120) **not supported** - needs PyTorch 2.6+ which breaks fairseq.
- **Host requirements:** NVIDIA driver 535+, CUDA 12.1+, Python 3.11 or 3.12.

## Monolith Architecture (2026-04-08)
- **Migration:** The project migrated from Docker-per-model microservices (21 containers via docker-compose) to a single-process monolith inference server (`apps/inference/server.py`). This eliminates container overhead, NVIDIA MPS complexity, and multi-CUDA-version management. **(Replaced Docker microservices architecture 2026-04-08)**
- **Server:** Single FastAPI process (`apps/inference/server.py`) loads all 24 models at startup via direct Python imports. Models are called as in-process function calls, not HTTP/base64.
- **Model Registry:** `apps/inference/config.py` defines all 24 models with metadata (modality, directory, GPU flag, isolation flag). Env var `DEEPSAFE_MODELS` controls which models to load ("all" or comma-separated names).
- **Code Layout:** Model code lives at `models/<modality>/<model>/code/` (cloned from original repos). Weights at `models/<modality>/<model>/weights/` (downloaded from HuggingFace). Each model has a predictor wrapper in `apps/inference/models/<name>.py` inheriting from `BasePredictor`.
- **Setup:** `apps/inference/setup.sh` handles everything in 8 steps: installs uv, creates venv, installs PyTorch 2.5.1+cu121, installs pinned deps from `apps/inference/requirements.lock`, builds fairseq from source with patches, clones 13 model code repos via `apps/inference/clone_repos.sh`, pre-downloads SCRFD face detection model, downloads ~34 GB weights from HuggingFace (`deepsafe/deepsafe-services`).
- **Concurrent inference:** `ThreadPoolExecutor` (configurable via `DEEPSAFE_MAX_WORKERS`, default 4) runs models in parallel within the single process.

## Infrastructure & DevOps
- **Model Serving**: **Inference server** (`apps/inference/server.py`) - Single FastAPI process serving all 24 models. Setup via `apps/inference/setup.sh`. No Docker containers for model serving. **(Migrated from Docker/Docker Compose 2026-04-08)**
- **Developer Docs**: **Unmint/Fumadocs** (Next.js + MDX) - Modern API documentation at `deepsafehq.github.io/deepsafe-bench/docs`. Static export to Cloudflare Pages. **(Added 2026-03-26)**
- **Reverse Proxy**: **Nginx** - Serving the frontend and proxying requests to the API gateway (local dev only, not used in production - Cloudflare handles routing).
- **Orchestration**: **pnpm workspaces** - Managing the monorepo for frontend packages.

## SaaS Features & Scaling
- **Authentication & Database**: **Supabase** -- Auth (JWT with ES256/HS256 verification, OAuth social login) and hosted PostgreSQL. Single Supabase project for auth and application data. **Public API:** SHA-256 hashed API keys (`ds_live_` prefix) stored in PostgreSQL, managed from the dashboard.
- **Public API**: Versioned REST API at `/v1/` with API key auth, tiered rate limiting (Redis), and scan quotas. Free (200 lifetime), Starter ($29/mo, 2K/mo), Pro ($99/mo, 10K/mo). Web demo and API developers use the same `/v1/detect` endpoint. **(Added 2026-03-26, unified billing 2026-03-27)**
- **Rate Limiting**: **Redis** - Per-minute and daily rate limits enforced via Redis pipelines with `X-RateLimit-*` response headers.
- **Analytics Engine**: Custom analytics layer built on PostgreSQL to track user behavior and model performance.
- **State Management & Caching**: **Redis** - Job state management, rate limiting, and Celery task broker.
- **Package Managers**: **uv** (Python), **pnpm** (Node.js).

## Analytics & Observability
- **Product Analytics**: **PostgreSQL** (`analytics_events` table) - Self-hosted event tracking via `analytics.track()` in the gateway.
- **Operational Analytics**: **PostgreSQL** - Per-model performance, cost tracking, usage summaries via materialized views.
- **Dashboard**: Minimal scan usage + detection history dashboard. Simple CSS progress bars and HTML tables for scan usage and detection history. **(Simplified 2026-03-27)**
- **Schema Migrations**: **Alembic** - SQLAlchemy-based database migration management.

## Tooling & Quality Assurance
- **Package Managers**: **uv** (Python), **pnpm** (Node.js).
- **Component Library**: **Shadcn UI** - Reusable, accessible UI components.
- **Icons**: **Lucide React** - A comprehensive set of clean, consistent icons.
- **Testing**: **Pytest** (Backend), **Next.js Lint** (Frontend).
