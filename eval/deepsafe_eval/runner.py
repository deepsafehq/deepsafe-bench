"""Eval runner: fresh inference from Docker model services.

Reads master_eval/metadata.json for the file manifest and
deepsafe_config.json for model endpoints.  For each sample, fans out
HTTP calls to all relevant services in parallel, computes ensemble
via the gateway's ensemble.py, and writes predictions incrementally.
"""

import base64
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

# Import ensemble logic from the gateway.
# The gateway is not a pip package, so add its parent to sys.path.
_GATEWAY_DIR = Path(__file__).parent.parent.parent / "apps" / "gateway"
if str(_GATEWAY_DIR) not in sys.path:
    sys.path.insert(0, str(_GATEWAY_DIR))

from ensemble import (  # noqa: E402
    _MODEL_NAME_MAP,
    PROVENANCE_SERVICES,
    calculate_ensemble_verdict_api,
)

# Metadata label values to numeric labels.
_LABEL_MAP = {"real": 0, "fake": 1}

# Modality name in metadata.json -> payload key for model requests.
_PAYLOAD_KEYS = {
    "images": "image_data",
    "image": "image_data",
    "audio": "audio_data",
    "video": "video_data",
}

# Modality name in metadata.json -> media_type for deepsafe_config.json.
_MODALITY_TO_MEDIA_TYPE = {
    "images": "image",
    "audio": "audio",
    "video": "video",
}

# Timeout per modality (seconds).
_TIMEOUTS = {
    "images": 120,
    "image": 120,
    "audio": 120,
    "video": 300,
}


def load_endpoints(config_path: Path) -> dict[str, dict[str, str]]:
    """Load model endpoints from deepsafe_config.json.

    Parses Docker-internal URLs and converts them to localhost:{port}/predict.

    Args:
        config_path: Path to deepsafe_config.json.

    Returns:
        Dict mapping media_type -> {service_name: localhost_url}.

    Raises:
        FileNotFoundError: If config_path does not exist.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    endpoints: dict[str, dict[str, str]] = {}
    for media_type, media_conf in config.get("media_types", {}).items():
        model_endpoints = media_conf.get("model_endpoints", {})
        local_endpoints: dict[str, str] = {}
        for service_name, docker_url in model_endpoints.items():
            parsed = urlparse(docker_url)
            port = parsed.port
            path = parsed.path or "/predict"
            local_endpoints[service_name] = f"http://localhost:{port}{path}"
        endpoints[media_type] = local_endpoints

    return endpoints


def call_model(
    url: str,
    b64_data: str,
    payload_key: str,
    timeout: int,
) -> dict:
    """POST base64 payload to a model service with one retry.

    Always returns a wrapper dict with the response and analytics
    metadata (latency, status, error info, attempt count).

    Args:
        url: Model endpoint URL (e.g., http://localhost:5001/predict).
        b64_data: Base64-encoded file content.
        payload_key: JSON key for payload (image_data/audio_data/video_data).
        timeout: Request timeout in seconds.

    Returns:
        Dict with keys: response, latency_ms, status, http_status_code,
        error_message, attempts.
    """
    payload = {payload_key: b64_data}
    t0 = time.monotonic()
    last_status = "error"
    last_error = None
    last_http_code = None
    attempts = 0

    for attempt in range(2):  # original + 1 retry
        attempts = attempt + 1
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            last_http_code = resp.status_code
            resp.raise_for_status()
            latency_ms = int((time.monotonic() - t0) * 1000)
            return {
                "response": resp.json(),
                "latency_ms": latency_ms,
                "status": "success",
                "http_status_code": resp.status_code,
                "error_message": None,
                "attempts": attempts,
            }
        except requests.exceptions.Timeout as exc:
            last_status = "timeout"
            last_error = str(exc)[:200]
        except requests.exceptions.ConnectionError as exc:
            last_status = "connection_error"
            last_error = str(exc)[:200]
        except requests.exceptions.HTTPError as exc:
            last_status = "http_error"
            last_error = str(exc)[:200]
            last_http_code = (
                exc.response.status_code if exc.response is not None else None
            )
        except Exception as exc:
            last_status = "error"
            last_error = str(exc)[:200]

        if attempt == 0:
            logger.warning(
                "Model call failed (attempt 1/2): %s — %s. " "Retrying in 2s...",
                url,
                last_error,
            )
            time.sleep(2)
        else:
            logger.warning(
                "Model call failed (attempt 2/2): %s — %s. " "Giving up.",
                url,
                last_error,
            )

    latency_ms = int((time.monotonic() - t0) * 1000)
    return {
        "response": None,
        "latency_ms": latency_ms,
        "status": last_status,
        "http_status_code": last_http_code,
        "error_message": last_error,
        "attempts": attempts,
    }


def fan_out_sample(
    b64_data: str,
    endpoints: dict[str, str],
    payload_key: str,
    timeout: int,
) -> dict[str, dict | None]:
    """Call all model endpoints for a single sample in parallel.

    Args:
        b64_data: Base64-encoded file content.
        endpoints: Dict of service_name -> localhost URL.
        payload_key: JSON key for payload.
        timeout: Per-model request timeout in seconds.

    Returns:
        Dict of service_name -> response dict (or None on failure).
    """
    results: dict[str, dict | None] = {}

    with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
        future_to_name = {
            pool.submit(call_model, url, b64_data, payload_key, timeout): name
            for name, url in endpoints.items()
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            results[name] = future.result()

    return results


def build_prediction(
    sample_id: str,
    label: int,
    generator: str,
    raw_results: dict[str, dict],
    media_type: str,
    file_size_bytes: int = 0,
) -> dict:
    """Build a prediction dict from raw model results.

    Separates detection and provenance results, maps gateway names to
    short names for detection models, computes ensemble via the
    gateway's ensemble logic, and collects per-model analytics.

    Args:
        sample_id: Sample identifier (e.g., "img_00001").
        label: Ground truth label (0=real, 1=fake).
        generator: Generator name (e.g., "dalle_3").
        raw_results: Dict of service_name -> call_model wrapper dict.
        media_type: One of "image", "video", "audio".
        file_size_bytes: Size of the input file in bytes.

    Returns:
        Prediction dict with id, label, generator, ensemble_prob,
        model_probs, provenance, and analytics.
    """
    model_probs: dict[str, float | None] = {}
    provenance: dict[str, dict] = {}
    per_model_analytics: dict[str, dict] = {}
    n_succeeded = 0
    n_failed = 0

    for service_name, wrapper in raw_results.items():
        response = wrapper.get("response")

        # Collect per-model analytics
        per_model_analytics[service_name] = {
            "latency_ms": wrapper.get("latency_ms", 0),
            "status": wrapper.get("status", "unknown"),
            "http_status_code": wrapper.get("http_status_code"),
            "error_message": wrapper.get("error_message"),
            "attempts": wrapper.get("attempts", 1),
        }

        if response is not None:
            n_succeeded += 1
        else:
            n_failed += 1

        if service_name in PROVENANCE_SERVICES:
            if response is not None:
                provenance[service_name] = response
        else:
            short_name = _MODEL_NAME_MAP.get(service_name, service_name)
            if response is None:
                model_probs[short_name] = None
            else:
                prob = response.get("probability")
                if prob is None:
                    prob = response.get("fake_probability")
                if prob is None:
                    prob = response.get("score")
                model_probs[short_name] = float(prob) if prob is not None else None

    # Compute ensemble only if at least one detection model succeeded.
    has_any_detection = any(v is not None for v in model_probs.values())
    if has_any_detection:
        # Build results dict in the format ensemble.py expects
        # (gateway names with full response dicts).
        ensemble_input = {
            name: wrapper["response"]
            for name, wrapper in raw_results.items()
            if wrapper.get("response") is not None
        }
        _, _, _, _, ensemble_prob, _ = calculate_ensemble_verdict_api(
            results=ensemble_input,
            threshold=0.5,
            method="auto",
            media_type=media_type,
            request_id=sample_id,
        )
    else:
        ensemble_prob = None

    return {
        "id": sample_id,
        "label": label,
        "generator": generator,
        "modality": media_type,
        "ensemble_prob": ensemble_prob,
        "model_probs": model_probs,
        "provenance": provenance,
        "analytics": {
            "file_size_bytes": file_size_bytes,
            "n_models_succeeded": n_succeeded,
            "n_models_failed": n_failed,
            "per_model": per_model_analytics,
        },
    }


class PartialFile:
    """Incremental predictions file with resume support.

    Stores a header with the model list for config mismatch detection,
    and appends predictions one at a time for crash safety.

    Args:
        path: Path to the partial predictions file.
        models_list: Sorted list of model endpoint names (for config
            mismatch detection on resume).
    """

    def __init__(self, path: Path, models_list: list[str]) -> None:
        self.path = path
        self.models_list = sorted(models_list)
        self.predictions: list[dict] = []
        self.completed_ids: set[str] = set()

        if path.exists():
            self._load_existing()

    def _load_existing(self) -> None:
        """Load existing partial file and validate config."""
        try:
            with open(self.path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning(
                "Corrupted partial file %s — starting fresh.",
                self.path,
            )
            return

        saved_models = sorted(data.get("models", []))
        if saved_models != self.models_list:
            raise ValueError(
                f"Partial file config mismatch: "
                f"saved={saved_models}, current={self.models_list}. "
                f"Use --force-restart to discard."
            )

        self.predictions = data.get("predictions", [])
        self.completed_ids = {p["id"] for p in self.predictions}
        logger.info(
            "Resumed from partial file: %d completed.",
            len(self.completed_ids),
        )

    def append(self, prediction: dict) -> None:
        """Append a prediction and flush to disk.

        Args:
            prediction: Prediction dict with at least an 'id' key.
        """
        self.predictions.append(prediction)
        self.completed_ids.add(prediction["id"])
        self._flush()

    def _flush(self) -> None:
        """Write current state to disk."""
        data = {
            "models": self.models_list,
            "predictions": self.predictions,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def close(self) -> None:
        """Final flush."""
        self._flush()


def run_inference(
    config_path: Path,
    dataset_root: Path,
    output_dir: Path,
    force_restart: bool = False,
) -> list[dict]:
    """Run fresh inference on master_eval dataset.

    Loads the manifest, calls all model services per sample,
    computes ensemble, writes predictions incrementally.

    Args:
        config_path: Path to deepsafe_config.json.
        dataset_root: Dataset root (parent of master_eval/).
        output_dir: Directory for output files.
        force_restart: If True, ignore existing partial file.

    Returns:
        List of prediction dicts.
    """
    # Load manifest
    master_dir = dataset_root / "master_eval"
    metadata_path = master_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)
    logger.info("Loaded %d samples from %s", len(metadata), metadata_path)

    # Load endpoints
    endpoints = load_endpoints(config_path)

    # Collect all model names across all modalities for config check
    all_models = sorted(
        name for media_endpoints in endpoints.values() for name in media_endpoints
    )

    # Set up partial file
    partial_path = output_dir / ".partial_predictions.json"
    if force_restart and partial_path.exists():
        partial_path.unlink()

    pf = PartialFile(partial_path, all_models)
    total = len(metadata)
    skipped_files = 0
    model_failures = 0
    start_time = time.time()

    for i, entry in enumerate(metadata):
        sample_id = entry["id"]

        # Skip if already completed (resume)
        if sample_id in pf.completed_ids:
            continue

        modality = entry["modality"]
        media_type = _MODALITY_TO_MEDIA_TYPE.get(modality, modality)
        payload_key = _PAYLOAD_KEYS.get(modality, "image_data")
        timeout = _TIMEOUTS.get(modality, 120)

        # Get endpoints for this modality
        modality_endpoints = endpoints.get(media_type, {})
        if not modality_endpoints:
            logger.warning(
                "[%d/%d] %s — no endpoints for modality '%s', skipping.",
                i + 1,
                total,
                sample_id,
                modality,
            )
            continue

        # Read and encode file
        file_path = master_dir / entry["path"]
        try:
            raw_bytes = file_path.read_bytes()
            file_size_bytes = len(raw_bytes)
            b64_data = base64.b64encode(raw_bytes).decode()
        except (OSError, IOError) as exc:
            logger.error(
                "[%d/%d] %s — file read error: %s",
                i + 1,
                total,
                sample_id,
                exc,
            )
            skipped_files += 1
            continue

        # Fan out to all models
        raw_results = fan_out_sample(
            b64_data=b64_data,
            endpoints=modality_endpoints,
            payload_key=payload_key,
            timeout=timeout,
        )

        # Count failures
        n_failed = sum(1 for v in raw_results.values() if v.get("response") is None)
        model_failures += n_failed

        # Build prediction
        label = _LABEL_MAP.get(entry["label"], entry["label"])
        if isinstance(label, str):
            label = int(label)

        pred = build_prediction(
            sample_id=sample_id,
            label=label,
            generator=entry["generator"],
            raw_results=raw_results,
            media_type=media_type,
            file_size_bytes=file_size_bytes,
        )

        pf.append(pred)

        # Progress log
        if pred["ensemble_prob"] is None:
            verdict = "N/A"
        elif pred["ensemble_prob"] >= 0.5:
            verdict = "FAKE"
        else:
            verdict = "REAL"
        n_detection = sum(1 for k in raw_results if k not in PROVENANCE_SERVICES)
        n_provenance = sum(1 for k in raw_results if k in PROVENANCE_SERVICES)
        logger.info(
            "[%d/%d] %s — %.4f %s (%d models, %d provenance)",
            i + 1,
            total,
            sample_id,
            pred["ensemble_prob"] if pred["ensemble_prob"] is not None else 0.0,
            verdict,
            n_detection,
            n_provenance,
        )

    # Finalize
    elapsed = time.time() - start_time
    predictions = pf.predictions
    pf.close()

    # Rename partial to final
    timestamp = time.strftime("%Y-%m-%d-%H-%M")
    final_path = output_dir / f"{timestamp}-predictions.json"
    final_data = {"predictions": predictions}
    with open(final_path, "w") as f:
        json.dump(final_data, f, indent=2)
    if partial_path.exists():
        partial_path.unlink()

    logger.info(
        "Inference complete: %d samples, %d skipped, "
        "%d model failures, %.1fs elapsed. Saved to %s",
        len(predictions),
        skipped_files,
        model_failures,
        elapsed,
        final_path,
    )

    return predictions
