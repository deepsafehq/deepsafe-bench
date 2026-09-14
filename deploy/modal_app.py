"""DeepSafe Modal Deployment.

Deploys the full backend (inference + gateway) on Modal serverless.
- Inference: GPU class that loads all 22 models, scale-to-zero
- Gateway: ASGI FastAPI app on CPU, custom domain localhost:8000
- Weights: Persistent Volume (~46 GB from HuggingFace)

Usage:
    # First time: download weights into Volume (~40 min)
    modal run deploy/modal_app.py::download_weights

    # Deploy everything
    modal deploy deploy/modal_app.py

    # Test inference locally
    modal run deploy/modal_app.py::test_inference
"""

import json
import os
import pathlib
import shutil

import modal

# ── Paths ───────────────────────────────────────────────────────────────────

REPO_ROOT = pathlib.Path(__file__).parent.parent

# ── Modal App ───────────────────────────────────────────────────────────────

app = modal.App("deepsafe")

# ── Volume for model weights ────────────────────────────────────────────────

weights_vol = modal.Volume.from_name("deepsafe-weights", create_if_missing=True)

WEIGHTS_MOUNT = "/weights"  # Volume mount path inside containers

# ── Secrets ─────────────────────────────────────────────────────────────────

deepsafe_secrets = modal.Secret.from_name("deepsafe-secrets")

# ── Inference Image ─────────────────────────────────────────────────────────
# Complex build: PyTorch + CUDA, 50+ packages, fairseq from source, model code

# System packages needed by OpenCV, audio libs, video processing
_SYSTEM_DEPS = [
    "git",
    "ffmpeg",
    "libgl1-mesa-glx",
    "libglib2.0-0",
    "libsm6",
    "libxext6",
    "libsndfile1",
    "libsox-dev",
    "exiftool",
]

# Core ML packages (installed AFTER PyTorch)
_ML_PACKAGES = [
    "numpy==1.26.4",
    "transformers==4.57.6",
    "timm==1.0.26",
    "open-clip-torch==3.3.0",
    "einops==0.8.2",
    "lightning==2.6.1",
    "pytorch-lightning==2.6.1",
    "librosa==0.11.0",
    "scipy==1.17.1",
    "scikit-learn==1.8.0",
    "albumentations==2.0.8",
    "accelerate==1.13.0",
]

# Model-specific packages
_MODEL_PACKAGES = [
    "efficientnet-pytorch==0.7.1",
    "facenet-pytorch==2.5.3",
    "peft==0.18.1",
    "loralib==0.1.2",
    "diffusers==0.37.1",
    "hydra-core==1.3.2",
    "omegaconf==2.3.0",
    "safetensors==0.7.0",
    "kornia==0.8.2",
]

# Provenance packages
_PROVENANCE_PACKAGES = [
    "c2pa-python==0.32.0",
    "invisible-watermark==0.2.0",
    "audioseal==0.1.4",
    "trustmark>=0.9.0",
    "av==17.0.0",
    "pycocotools==2.0.11",
]

# I/O packages
_IO_PACKAGES = [
    "opencv-python-headless==4.11.0.86",
    "Pillow==10.2.0",
    "pillow-heif==0.22.0",
    "ffmpeg-python==0.2.0",
    "soundfile==0.13.1",
    "pyexiftool==0.5.6",
]

# Video model dependencies
_VIDEO_PACKAGES = [
    "networkx==3.6.1",
    "filterpy==1.4.5",
    "simplejson==3.20.2",
    "fvcore==0.1.5.post20221221",
    "yacs==0.1.8",
]

# Inference optimization
_INFERENCE_PACKAGES = [
    "insightface==0.7.3",
    "onnxruntime-gpu==1.24.4",
    "onnx==1.21.0",
]

# Ensemble + misc
_MISC_PACKAGES = [
    "lightgbm==4.6.0",
    "xgboost==3.2.0",
    "joblib==1.5.1",
    "pandas==3.0.2",
    "python-box==7.4.1",
    "setuptools==69.5.1",
    "rich==14.3.3",
    "Cython==3.2.4",
    "redis==6.4.0",
    # Required by model code repos (missing causes model load failures)
    "wandb==0.25.1",
    "tensorboard==2.19.0",
    "plotly==6.6.0",
    "sacrebleu",
    "seaborn==0.13.2",
]

# Server packages
_SERVER_PACKAGES = [
    "fastapi==0.135.3",
    "uvicorn[standard]==0.44.0",
    "pydantic==2.12.5",
    "python-multipart==0.0.24",
]

_PYTORCH_INDEX = "https://download.pytorch.org/whl/cu121"

inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(_SYSTEM_DEPS)
    # PyTorch + CUDA 12.1 (must be first, other packages depend on it)
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        index_url=_PYTORCH_INDEX,
    )
    # All other dependencies (split to avoid giant single layer)
    .pip_install(
        *_ML_PACKAGES,
        *_MODEL_PACKAGES,
        *_SERVER_PACKAGES,
        extra_index_url=_PYTORCH_INDEX,
    )
    .pip_install(
        *_PROVENANCE_PACKAGES,
        *_IO_PACKAGES,
        *_VIDEO_PACKAGES,
        *_INFERENCE_PACKAGES,
        *_MISC_PACKAGES,
    )
    # videoseal: --no-deps to avoid timm version conflict
    .run_commands("pip install videoseal==1.0.1 --no-deps")
    # HuggingFace transfer acceleration
    .pip_install("hf_transfer")
    # Add application code
    .add_local_dir(str(REPO_ROOT / "apps" / "inference"), "/app/inference", copy=True)
    .add_local_dir(str(REPO_ROOT / "packages" / "shared"), "/app/shared", copy=True)
    .add_local_dir(str(REPO_ROOT / "models"), "/app/models", copy=True)
    # Build fairseq from source (required by ShiftySpeech/SafeEar audio models)
    .run_commands(
        "git clone --depth 1 https://github.com/facebookresearch/fairseq.git /tmp/fairseq",
        "cd /tmp/fairseq && READTHEDOCS=1 pip install --no-deps --no-build-isolation .",
        "pip install bitarray",
        # patch_fairseq.py verification step fails because the runtime
        # monkey-patch (fairseq_compat.py) is needed at import time.
        # The actual patch IS applied; verification is non-critical.
        "PYTHONPATH=/app/inference:/app/shared python /app/inference/patch_fairseq.py || true",
        "rm -rf /tmp/fairseq",
    )
    # Pre-download SCRFD face detection model (used by video pipeline)
    .run_commands(
        'python -c "'
        "from insightface.app import FaceAnalysis; "
        "FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])"
        '"'
    )
    .env(
        {
            "PYTHONPATH": "/app/inference:/app/shared",
            "DEEPSAFE_MODELS_ROOT": "/app/models",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # Not setting TRANSFORMERS_OFFLINE — models use .ext_cache for
            # local weights but some (cospy/SigLIP) need HF hub to resolve
            # local cache paths. Network calls are avoided by having the
            # cache pre-populated.
        }
    )
)

# ── Gateway Image ───────────────────────────────────────────────────────────

_GATEWAY_PACKAGES = [
    "fastapi==0.135.1",
    "uvicorn==0.27.0",
    "pydantic==2.12.5",
    "aiohttp==3.13.3",
    "python-multipart==0.0.22",
    "pillow==12.1.1",
    "requests==2.32.5",
    "rich==14.3.3",
    "sqlalchemy==2.0.48",
    "alembic==1.18.4",
    "psycopg2-binary",
    "redis==6.4.0",
    "PyJWT==2.12.1",
    "cryptography==46.0.5",
    "scikit-learn>=1.5.0,<2.0.0",
    "lightgbm==4.6.0",
    "xgboost==3.2.0",
    "numpy>=2.0.0,<3.0.0",
    "prometheus-client==0.24.1",
    "sentry-sdk[fastapi]>=2.0.0,<3.0.0",
    "joblib==1.5.1",
    "celery[redis]==5.6.2",
    "minio==7.2.20",
]

gateway_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*_GATEWAY_PACKAGES)
    .add_local_dir(str(REPO_ROOT / "apps" / "gateway"), "/app/gateway", copy=True)
    .add_local_dir(str(REPO_ROOT / "packages" / "shared"), "/app/shared", copy=True)
    .add_local_dir(str(REPO_ROOT / "models" / "ensemble"), "/app/ensemble", copy=True)
    # Include deepsafe_config.json (gateway reads model endpoint URLs from it,
    # not strictly needed since we patch detection, but keeps imports happy)
    .add_local_file(
        str(REPO_ROOT / "deepsafe_config.json"),
        "/app/deepsafe_config.json",
        copy=True,
    )
    .env(
        {
            "PYTHONPATH": "/app/gateway:/app/shared",
            "DEEPSAFE_CONFIG_FILE_PATH": "/app/deepsafe_config.json",
            "_CACHE_BUST": "20260413b",
        }
    )
)


# ── Weight Download (one-time) ──────────────────────────────────────────────


@app.function(
    image=inference_image,
    volumes={WEIGHTS_MOUNT: weights_vol},
    secrets=[deepsafe_secrets],
    timeout=86400,  # 24h — large download, don't rush it
    memory=8192,
    # Override TRANSFORMERS_OFFLINE from the image — we need network access
    env={
        "TRANSFORMERS_OFFLINE": "0",
        "HF_HUB_OFFLINE": "0",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    },
)
def download_weights():
    """Download model weights from HuggingFace into the Volume.

    Run once: modal run deploy/modal_app.py::download_weights
    Takes ~30-40 minutes for ~46 GB download.
    """
    token = os.environ.get("HUGGINGFACE_TOKEN", "")
    if not token:
        raise RuntimeError("HUGGINGFACE_TOKEN not set in deepsafe-secrets")

    print("Downloading weights from deepsafe/deepsafe-services...")
    print("This will download ~46 GB (model weights + external caches).")

    # Use huggingface_hub to download
    from huggingface_hub import snapshot_download

    snapshot_download(
        "deepsafe/deepsafe-services",
        repo_type="model",
        local_dir=f"{WEIGHTS_MOUNT}/hf_download",
        max_workers=4,
        token=token,
    )

    # Copy into proper structure (no rsync in container, use shutil)
    src = f"{WEIGHTS_MOUNT}/hf_download"
    dst = WEIGHTS_MOUNT

    import glob

    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            if os.path.exists(d):
                # Merge directories
                for sub in glob.glob(f"{s}/**", recursive=True):
                    rel = os.path.relpath(sub, s)
                    target = os.path.join(d, rel)
                    if os.path.isdir(sub):
                        os.makedirs(target, exist_ok=True)
                    else:
                        os.makedirs(os.path.dirname(target), exist_ok=True)
                        shutil.copy2(sub, target)
            else:
                shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)
    shutil.rmtree(src, ignore_errors=True)

    # Rename HF directories to match model registry names
    renames = {
        "video/fake-stormer": "video/fakestormer",
        "video/dfd-fcg": "video/dfd_fcg",
        "video/pwtf-dvd": "video/pwtf_dvd",
        "provenance/c2pa-checker": "provenance/c2pa",
        "provenance/invisible-watermark": "provenance/sdxl_watermark",
    }
    for old_name, new_name in renames.items():
        old_path = f"{dst}/{old_name}"
        new_path = f"{dst}/{new_name}"
        if os.path.exists(old_path) and not os.path.exists(new_path):
            os.rename(old_path, new_path)
            print(f"  Renamed: {old_name} -> {new_name}")

    # Fix nested weight directories from HF download
    for model_dir in ["video/fakestormer", "video/dfd_fcg", "video/pwtf_dvd"]:
        full = f"{dst}/{model_dir}"
        weights_dir = f"{full}/weights"
        if os.path.isdir(full) and not os.path.isdir(weights_dir):
            # Find nested weights dir
            for root, dirs, _ in os.walk(full):
                if "weights" in dirs:
                    nested = os.path.join(root, "weights")
                    if nested != weights_dir:
                        shutil.move(nested, weights_dir)
                        print(f"  Fixed nested: {model_dir}/weights")
                    break

    # Verify key weight files exist
    checks = [
        "image/npr/weights/NPR.pth",
        "image/aide/weights/GenImage_train.pth",
        "audio/shiftyspeech/weights/xlsr_53_56k.pt",
        "ensemble/artifacts/image_meta_learner.pkl",
    ]
    for check in checks:
        path = f"{dst}/{check}"
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"  OK: {check} ({size_mb:.1f} MB)")
        else:
            print(f"  MISSING: {check}")

    weights_vol.commit()
    print("Done! Weights committed to Volume.")


# ── Inference Server ────────────────────────────────────────────────────────


@app.cls(
    image=inference_image,
    gpu="A10G",
    volumes={WEIGHTS_MOUNT: weights_vol},
    secrets=[deepsafe_secrets],
    scaledown_window=300,  # 5 min idle before shutdown
    timeout=600,  # 10 min max per request
    startup_timeout=600,  # 10 min for model loading
)
class InferenceServer:
    """Loads all 22 detection models and runs ensemble inference."""

    @modal.enter()
    def load_models(self):
        """Load all models into GPU memory (runs once per container)."""
        import sys
        import time

        import torch

        # Ensure paths are set
        for p in ["/app/inference", "/app/shared"]:
            if p not in sys.path:
                sys.path.insert(0, p)

        # Set up weight symlinks: /app/models/*/weights/ -> /weights/*/weights/
        self._setup_weight_symlinks()

        # Set up model code symlinks (clone_repos.sh logic)
        self._setup_code_symlinks()

        # fairseq compatibility patch (must be before any fairseq import)
        import fairseq_compat  # noqa: F401
        from config import MAX_WORKERS, get_enabled_models, get_weights_path
        from deepsafe_shared.device import (
            get_device,
            setup_inference_optimizations,
        )

        from models import get_predictor_class

        self.device = get_device()
        setup_inference_optimizations(self.device)

        print(f"Loading models on {self.device}...")

        model_names = get_enabled_models()
        self.models = {}
        self.failed_models = []

        from concurrent.futures import ThreadPoolExecutor

        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

        for name in model_names:
            try:
                cls = get_predictor_class(name)
                predictor = cls()
                weights_dir = get_weights_path(name)
                start = time.perf_counter()
                predictor.load(weights_dir, self.device)
                elapsed = (time.perf_counter() - start) * 1000
                predictor._load_time_ms = elapsed
                self.models[name] = predictor
                print(f"  {name}: LOADED ({elapsed:.0f}ms)")
            except Exception as e:
                print(f"  {name}: FAILED - {e}")
                self.failed_models.append(name)

        print(f"Startup complete: {len(self.models)}/{len(model_names)} models")

        if self.device.type == "cuda":
            vram = torch.cuda.memory_allocated(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"VRAM: {vram:.1f} / {total:.1f} GB")

    def _setup_weight_symlinks(self):
        """Symlink weight directories from Volume into model code tree."""
        modalities = ["image", "audio", "video", "provenance"]
        for modality in modalities:
            vol_mod = f"{WEIGHTS_MOUNT}/{modality}"
            app_mod = f"/app/models/{modality}"
            if not os.path.isdir(vol_mod):
                continue
            for model_name in os.listdir(vol_mod):
                weights_src = f"{vol_mod}/{model_name}/weights"
                weights_dst = f"{app_mod}/{model_name}/weights"
                if os.path.isdir(weights_src) and not os.path.exists(weights_dst):
                    os.makedirs(os.path.dirname(weights_dst), exist_ok=True)
                    os.symlink(weights_src, weights_dst)

        # Ensemble artifacts
        vol_ensemble = f"{WEIGHTS_MOUNT}/ensemble/artifacts"
        app_ensemble = "/app/models/ensemble/artifacts"
        if os.path.isdir(vol_ensemble) and not os.path.exists(app_ensemble):
            os.makedirs(os.path.dirname(app_ensemble), exist_ok=True)
            os.symlink(vol_ensemble, app_ensemble)

        # External cache (.ext_cache)
        vol_cache = f"{WEIGHTS_MOUNT}/.ext_cache"
        app_cache = "/app/models/.ext_cache"
        if os.path.isdir(vol_cache) and not os.path.exists(app_cache):
            os.symlink(vol_cache, app_cache)

    def _setup_code_symlinks(self):
        """Replicate clone_repos.sh logic for code/weight path setup."""
        models_root = "/app/models"

        # Create required directories
        os.makedirs("/app/weights", exist_ok=True)
        os.makedirs("logs", exist_ok=True)  # FakeSTormer needs this

        # XLS-R symlinks for ShiftySpeech/Nes2Net (many hardcoded paths)
        xlsr = f"{models_root}/audio/shiftyspeech/weights/xlsr_53_56k.pt"
        if os.path.exists(xlsr):
            for target in [
                f"{models_root}/xlsr2_300m.pt",
                "/app/weights/xlsr2_300m.pt",
                "/app/inference/models/xlsr2_300m.pt",
                # ShiftySpeech uses relative path "models/xlsr2_300m.pt"
                "models/xlsr2_300m.pt",
            ]:
                if not os.path.exists(target):
                    try:
                        os.makedirs(os.path.dirname(target), exist_ok=True)
                        os.symlink(xlsr, target)
                    except OSError:
                        pass

        # NPR_VIDEO reuses image/npr weights
        npr_weights = f"{models_root}/image/npr/weights/NPR.pth"
        if os.path.exists(npr_weights):
            dst_dir = f"{models_root}/video/npr_video/weights"
            os.makedirs(dst_dir, exist_ok=True)
            dst = f"{dst_dir}/NPR.pth"
            if not os.path.exists(dst):
                os.symlink(npr_weights, dst)

        # UNIVFD_VIDEO reuses image/universal weights
        univ_weights = f"{models_root}/image/universal/weights/fc_weights.pth"
        if os.path.exists(univ_weights):
            dst_dir = f"{models_root}/video/univfd_video/weights"
            os.makedirs(dst_dir, exist_ok=True)
            dst = f"{dst_dir}/fc_weights.pth"
            if not os.path.exists(dst):
                os.symlink(univ_weights, dst)

        # FakeSTormer: code symlink
        fs_code = f"{models_root}/video/fakestormer/fake-stormer/model_code"
        fs_link = f"{models_root}/video/fakestormer/code"
        if os.path.isdir(fs_code) and not os.path.exists(fs_link):
            os.symlink("fake-stormer/model_code", fs_link)

        # Effort: __init__.py fix
        effort_utils = f"{models_root}/image/effort/code/DeepfakeBench/training/utils"
        if os.path.isdir(effort_utils):
            init_file = f"{effort_utils}/__init__.py"
            if not os.path.exists(init_file):
                open(init_file, "w").close()

        # NPR_VIDEO: code symlink to image/npr/code
        npr_vid_code = f"{models_root}/video/npr_video/code"
        npr_img_code = f"{models_root}/image/npr/code"
        if os.path.isdir(npr_img_code) and not os.path.exists(npr_vid_code):
            os.makedirs(os.path.dirname(npr_vid_code), exist_ok=True)
            os.symlink(npr_img_code, npr_vid_code)

        # UNIVFD_VIDEO: code symlink to image/universal/code
        univ_vid_code = f"{models_root}/video/univfd_video/code"
        univ_img_code = f"{models_root}/image/universal/code"
        if os.path.isdir(univ_img_code) and not os.path.exists(univ_vid_code):
            os.makedirs(os.path.dirname(univ_vid_code), exist_ok=True)
            os.symlink(univ_img_code, univ_vid_code)

        # Effort: GPU SVD patch
        effort_det = (
            f"{models_root}/image/effort/code/DeepfakeBench/"
            "training/detectors/effort_detector.py"
        )
        if os.path.exists(effort_det):
            with open(effort_det, "r") as f:
                content = f.read()
            if "_svd_device" not in content:
                content = content.replace(
                    "U, S, Vh = torch.linalg.svd(module.weight.data, "
                    "full_matrices=False)",
                    '_svd_device = torch.device("cuda") '
                    "if torch.cuda.is_available() "
                    'else torch.device("cpu")\n'
                    "        U, S, Vh = torch.linalg.svd("
                    "module.weight.data.to(_svd_device), "
                    "full_matrices=False)\n"
                    "        U, S, Vh = U.cpu(), S.cpu(), Vh.cpu()",
                )
                with open(effort_det, "w") as f:
                    f.write(content)

    @modal.method()
    def detect(
        self,
        file_bytes: bytes,
        media_type: str,
        threshold: float = 0.5,
    ) -> dict:
        """Run full ensemble detection pipeline.

        Args:
            file_bytes: Raw media file bytes.
            media_type: "image", "audio", or "video".
            threshold: Classification threshold (default 0.5).

        Returns:
            Dict with verdict, confidence, score, method, model_results, etc.
        """
        import time
        import uuid
        from concurrent.futures import as_completed

        from config import get_models_by_modality

        request_id = uuid.uuid4().hex[:8]
        start = time.perf_counter()

        # Get models for this media type + provenance
        detection_names = get_models_by_modality(media_type)
        provenance_names = get_models_by_modality("provenance")
        all_names = detection_names + provenance_names
        active = [n for n in all_names if n in self.models]

        if not active:
            return {
                "error": f"No models loaded for {media_type}",
                "models_loaded": list(self.models.keys()),
            }

        # Video: shared preprocessing
        video_data = None
        if media_type == "video":
            try:
                from video_preprocess import preprocess_video

                video_data = preprocess_video(file_bytes, self.device)
            except Exception as e:
                print(f"Video preprocess failed, falling back: {e}")

        # Run models
        model_results = {}
        futures = {}
        _DENSE = {"dfd_fcg", "pwtf_dvd", "mintime"}

        if video_data is not None and media_type == "video":
            # Dense models in thread pool
            for name in active:
                if name in _DENSE and name in self.models:
                    future = self.executor.submit(
                        self.models[name].predict_preprocessed, video_data
                    )
                    futures[future] = name
            # Light models sequentially on GPU
            for name in active:
                if name not in _DENSE and name in self.models:
                    try:
                        model = self.models[name]
                        if hasattr(model, "predict_preprocessed"):
                            model_results[name] = model.predict_preprocessed(video_data)
                        else:
                            model_results[name] = model.predict(file_bytes)
                    except Exception as e:
                        model_results[name] = {
                            "probability": None,
                            "error": str(e),
                            "latency_ms": 0,
                        }
            # Collect dense results
            for future in as_completed(futures, timeout=600):
                name = futures[future]
                try:
                    model_results[name] = future.result()
                except Exception as e:
                    model_results[name] = {
                        "probability": None,
                        "error": str(e),
                        "latency_ms": 0,
                    }
        else:
            # Non-video: all in parallel
            for name in active:
                future = self.executor.submit(self.models[name].predict, file_bytes)
                futures[future] = name
            for future in as_completed(futures, timeout=600):
                name = futures[future]
                try:
                    model_results[name] = future.result()
                except Exception as e:
                    model_results[name] = {
                        "probability": None,
                        "error": str(e),
                        "latency_ms": 0,
                    }

        # Cleanup video
        if video_data is not None:
            try:
                from video_preprocess import cleanup_video_data

                cleanup_video_data(video_data)
            except Exception:
                pass

        # Ensemble
        from ensemble import compute_ensemble

        ensemble_result = compute_ensemble(
            model_results, media_type, threshold, request_id
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "request_id": request_id,
            "media_type": media_type,
            "verdict": ensemble_result["verdict"],
            "confidence": ensemble_result["confidence"],
            "score": ensemble_result["score"],
            "method": ensemble_result["method"],
            "threshold": threshold,
            "models_used": ensemble_result["models_used"],
            "model_results": {
                name: {
                    "probability": r.get("probability"),
                    "prediction": r.get("prediction"),
                    "latency_ms": r.get("latency_ms"),
                    "error": r.get("error"),
                }
                for name, r in model_results.items()
            },
            "total_latency_ms": round(elapsed_ms, 1),
        }

    @modal.method()
    def detect_and_store(
        self,
        file_bytes: bytes,
        media_type: str,
        threshold: float,
        detection_id: str,
        user_id: str,
    ):
        """Run inference and store results in Redis (for async gateway path).

        Called via .spawn() from the gateway. Writes PROCESSING -> COMPLETE/FAILED
        to Upstash Redis so the frontend can poll /v1/results/{detection_id}.
        """
        from datetime import datetime, timezone

        import redis as redis_lib

        redis_url = os.environ.get("REDIS_URL", "")
        if not redis_url:
            print("WARNING: REDIS_URL not set, cannot store async results")
            return

        r = redis_lib.from_url(redis_url, decode_responses=True, ssl_cert_reqs=None)
        redis_key = f"job:{detection_id}"

        try:
            # Mark PROCESSING
            r.setex(
                redis_key,
                7200,
                json.dumps(
                    {
                        "job_id": detection_id,
                        "user_id": user_id,
                        "status": "PROCESSING",
                        "media_type": media_type,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "models": {},
                        "result": None,
                        "error": None,
                    }
                ),
            )

            # Run detection
            result = self.detect.local(file_bytes, media_type, threshold)

            # Anonymize model names for API response
            sorted_names = sorted(result.get("model_results", {}).keys())
            anon_results = {}
            for i, name in enumerate(sorted_names, 1):
                anon_results[f"module_{i}"] = result["model_results"][name]

            is_fake = result.get("verdict") == "fake"
            result_payload = {
                "request_id": detection_id,
                "is_likely_deepfake": is_fake,
                "deepfake_probability": result.get("score", 0.5),
                "model_count": result.get("models_used", 0),
                "fake_votes": sum(
                    1
                    for r in result.get("model_results", {}).values()
                    if r.get("probability") is not None
                    and r["probability"] >= threshold
                ),
                "real_votes": sum(
                    1
                    for r in result.get("model_results", {}).values()
                    if r.get("probability") is not None and r["probability"] < threshold
                ),
                "response_time": result.get("total_latency_ms", 0),
                "ensemble_method_used": result.get("method", "unknown"),
                "model_results": anon_results,
                "media_type_processed": media_type,
            }

            # Mark COMPLETE
            r.setex(
                redis_key,
                3600,
                json.dumps(
                    {
                        "job_id": detection_id,
                        "user_id": user_id,
                        "status": "COMPLETE",
                        "media_type": media_type,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "models": {},
                        "result": result_payload,
                        "error": None,
                    }
                ),
            )
            print(f"Job {detection_id}: COMPLETE ({result.get('verdict')})")

        except Exception as e:
            # Mark FAILED
            r.setex(
                redis_key,
                3600,
                json.dumps(
                    {
                        "job_id": detection_id,
                        "user_id": user_id,
                        "status": "FAILED",
                        "media_type": media_type,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "error": "Analysis failed. Please try again or contact support.",
                    }
                ),
            )
            print(f"Job {detection_id}: FAILED - {e}")

    @modal.method()
    def health(self) -> dict:
        """Return server health status."""
        import torch

        gpu_info = {}
        if self.device.type == "cuda":
            gpu_info = {
                "gpu_name": torch.cuda.get_device_name(0),
                "vram_used_mb": round(torch.cuda.memory_allocated(0) / 1024**2),
                "vram_total_mb": round(
                    torch.cuda.get_device_properties(0).total_memory / 1024**2
                ),
            }
        return {
            "status": "healthy" if self.models else "degraded",
            "device": str(self.device),
            "models_loaded": len(self.models),
            "models_failed": self.failed_models,
            "model_names": list(self.models.keys()),
            **gpu_info,
        }


# ── Gateway ─────────────────────────────────────────────────────────────────


@app.function(
    image=gateway_image,
    secrets=[deepsafe_secrets],
    scaledown_window=120,
    timeout=300,
)
# TODO: Add custom_domains=["localhost:8000"] after registering in Modal dashboard
@modal.asgi_app()
def gateway():
    """Deploy the FastAPI gateway as a Modal ASGI app.

    Patches the existing gateway code to:
    1. Use Upstash Redis instead of local Redis
    2. Replace Celery async with Modal .spawn()
    3. Call Modal inference instead of localhost:8001
    """
    import sys

    # Add gateway and shared to path
    for p in ["/app/gateway", "/app/shared"]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # Set environment for gateway
    redis_url = os.environ.get("REDIS_URL", "")
    os.environ["CELERY_BROKER_URL"] = redis_url
    os.environ["CELERY_RESULT_BACKEND"] = redis_url
    os.environ.setdefault("DEEPSAFE_ENV", "production")
    os.environ.setdefault("META_MODEL_ARTIFACTS_DIR", "/app/ensemble/artifacts")

    # Import gateway app (this triggers Celery init, which is fine -
    # it connects to Upstash Redis but we never start a worker)
    from main import app as gw_app

    # Monkey-patch sync + async detection to use Modal inference
    _patch_detection(gw_app)

    return gw_app


def _patch_detection(gw_app):
    """Replace HTTP+Celery+MinIO detection with direct Modal function calls.

    Patches both sync and async detection paths in the gateway so that:
    - Sync: calls InferenceServer.detect.remote() directly
    - Async: calls InferenceServer.detect_and_store.spawn() (no Celery/MinIO)
    """
    import importlib
    import sys

    # Get the detection module (already imported by main.py)
    if "services.detection" in sys.modules:
        detection_mod = sys.modules["services.detection"]
    else:
        detection_mod = importlib.import_module("services.detection")

    # ── Patch 1: Async detection (replaces Celery + MinIO) ──

    def _modal_enqueue_async(
        file_bytes: bytes,
        content_type: str,
        media_type: str,
        user_id,
        detection_id: str,
    ) -> None:
        """Replacement: store QUEUED in Redis, dispatch to Modal."""
        from datetime import datetime, timezone

        rc = detection_mod.redis_client

        redis_key = f"job:{detection_id}"
        rc.set(
            redis_key,
            json.dumps(
                {
                    "job_id": detection_id,
                    "user_id": user_id,
                    "status": "QUEUED",
                    "media_type": media_type,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "models": {},
                    "result": None,
                    "error": None,
                }
            ),
            ex=7200,
        )

        try:
            InferenceServer().detect_and_store.spawn(
                file_bytes,
                media_type,
                0.5,  # threshold
                detection_id,
                user_id,
            )
        except Exception as e:
            rc.set(
                redis_key,
                json.dumps(
                    {
                        "job_id": detection_id,
                        "user_id": user_id,
                        "status": "FAILED",
                        "error": f"Failed to dispatch inference: {e}",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                ),
                ex=3600,
            )

    detection_mod._enqueue_async_detection = _modal_enqueue_async

    # ── Patch 2: Sync detection (replaces HTTP fan-out) ──

    def _modal_sync_detection(
        file_bytes: bytes,
        content_type: str,
        media_type: str,
        user_id,
        detection_id: str,
        db,
        threshold_override=None,
    ) -> dict:
        """Replacement: call Modal InferenceServer directly."""
        from fastapi import HTTPException, status

        threshold = threshold_override if threshold_override is not None else 0.5

        try:
            result = InferenceServer().detect.remote(file_bytes, media_type, threshold)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": "inference_failed",
                    "message": f"Inference server error: {e}",
                },
            ) from e

        if "error" in result and "verdict" not in result:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": "no_models",
                    "message": result.get("error", "Unknown inference error"),
                },
            )

        # Persist to DB (best-effort, don't fail the request if DB is down)
        try:
            if hasattr(detection_mod, "_persist_analysis"):
                detection_mod._persist_analysis(
                    user_id,
                    detection_id,
                    media_type,
                    result["verdict"],
                    result["confidence"],
                    result.get("method", "modal"),
                    result.get("score", 0.5),
                    result.get("model_results", {}),
                    db=db,
                )
        except Exception as persist_err:
            print(f"DB persistence failed (non-fatal): {persist_err}")

        return {
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "media_type": media_type,
        }

    detection_mod._run_sync_detection = _modal_sync_detection

    # ── Patch 3: Neutralize Celery task (safety net) ──

    if "main" in sys.modules:
        main_mod = sys.modules["main"]
        if hasattr(main_mod, "run_detection") and hasattr(
            main_mod.run_detection, "delay"
        ):
            main_mod.run_detection.delay = lambda *a, **kw: None

    print("Patched gateway detection: sync + async now use Modal inference")


# ── Test entrypoint ─────────────────────────────────────────────────────────


@app.local_entrypoint()
def test_inference():
    """Quick test: check if inference server is healthy."""
    server = InferenceServer()
    health = server.health.remote()
    print(json.dumps(health, indent=2))
