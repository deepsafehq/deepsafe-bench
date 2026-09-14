"""Ensemble logic for DeepSafe detection results.

Modality-specific ensemble strategies trained on 15.5K-sample eval
(2026-04-12):

IMAGE: LightGBM meta-learner on 7-model probabilities.
       CV AUC=0.9705, ECE=0.0096. Platt-calibrated.
       Fallback: AUC-weighted averaging.

AUDIO: Random Forest (500 trees, depth 8) on 3-model probabilities.
       CV AUC=0.8748, ECE=0.0132. Platt-calibrated.
       Fallback: AUC-weighted averaging.

VIDEO: XGBoost (300 trees, depth 3, reg_lambda=10) on 9-model
       probabilities. CV AUC=0.8898, ECE=0.0219. Platt-calibrated.
       Fallback: AUC-weighted averaging.
"""

import json
import logging
import os
import pickle  # nosec B403 — loading trusted local model artifacts only
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger("deepsafe_gateway")

# --- Meta-learner artifacts ---
_ARTIFACTS_DIR = Path(
    os.environ.get(
        "META_MODEL_ARTIFACTS_DIR",
        str(Path(__file__).resolve().parents[3] / "models" / "ensemble" / "artifacts"),
    )
)


def _safe_unpickle(path: Path):
    """Load a pickle file only if it resides within the trusted artifacts dir.

    Args:
        path: Path to the pickle file.

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If the path resolves outside _ARTIFACTS_DIR.
    """
    resolved = path.resolve()
    if not str(resolved).startswith(str(_ARTIFACTS_DIR.resolve())):
        raise ValueError(
            f"Refusing to load pickle from outside artifacts dir: {resolved}"
        )
    with open(resolved, "rb") as f:
        return pickle.load(f)  # nosec B301 — path validated above


_image_meta_learner = None
_image_scaler = None
_image_model_order = None
_image_calibrator = None

_video_meta_learner = None
_video_scaler = None
_video_model_order = None
_video_calibrator = None

_audio_meta_learner = None
_audio_scaler = None
_audio_model_order = None
_audio_calibrator = None

# Model name mapping: gateway config names -> standard names used in training
_MODEL_NAME_MAP = {
    "npr_deepfakedetection": "npr",
    "universalfakedetect": "universal",
    "yermandy_clip_detection": "yermandy",
    "aide_detection": "aide",
    "fsd_detection": "fsd",
    "effort_detection": "effort",
    "cospy_detection": "cospy",
    "shiftyspeech_detection": "shiftyspeech",
    "safeear_detection": "safeear",
    "sonics_detection": "sonics",
    "fakestormer": "fakestormer",
    "sbi_detection": "sbi",
    "dfd_fcg_detection": "dfd_fcg",
    "pwtf_dvd_detection": "pwtf_dvd",
    "lipfd_detection": "lipfd",
    "recce_detection": "recce",
    "mintime_detection": "mintime",
    "nes2net_detection": "nes2net",
    "npr_video": "npr_video",
    "univfd_video": "univfd_video",
}

_AUDIO_MODEL_AUC = {
    "shiftyspeech": 0.7347,
    "safeear": 0.6035,
    "nes2net": 0.8407,
}

# Video model names (for meta-learner)
_VIDEO_MODELS = {
    "fakestormer",
    "sbi",
    "dfd_fcg",
    "pwtf_dvd",
    "lipfd",
    "recce",
    "mintime",
    "npr_video",
    "univfd_video",
}

# Provenance service names -- filtered from Stage 1 model ensemble
PROVENANCE_SERVICES = {
    "c2pa_checker",
    "sdxl_watermark_detector",
    "audioseal_detector",
    "videoseal_detector",
    "trustmark_detector",
}
PROVENANCE_OVERRIDE_THRESHOLD = (
    0.80  # Lowered from 0.95: C2PA fires at 0.95, override at 0.80
)


def _get_provenance_weight(max_provenance: float) -> float:
    """Return tiered provenance weight based on signal strength.

    Only called for signals in [0.50, 0.80) since >= 0.80 triggers
    the override path and bypasses boost entirely.

    Moderate signals (SDXL watermark at 0.70, TrustMark at 0.65)
    get 4x the weight of weak/marginal signals.
    """
    if max_provenance >= 0.65:
        return 0.20
    return 0.05


def apply_provenance_boost(
    model_score: float,
    all_results: Dict[str, Dict],
) -> Tuple[float, bool]:
    """Apply provenance boost to model ensemble score (Stage 2).

    Uses tiered weights: moderate signals (>=0.65) get weight 0.20,
    weak signals get weight 0.05. Signals >=0.80 never reach this
    function (handled by the override path).

    Args:
        model_score: Score from Stage 1 model ensemble [0, 1].
        all_results: Full dict of all model results including provenance.

    Returns:
        Tuple of (final_score, was_boosted).
    """
    provenance_signals = []
    for name, result in all_results.items():
        if name not in PROVENANCE_SERVICES:
            continue
        if not isinstance(result, dict) or "error" in result:
            continue
        prob = result.get("probability", 0.5)
        if prob > 0.5:
            provenance_signals.append(prob)

    if not provenance_signals:
        return model_score, False

    max_provenance = max(provenance_signals)
    weight = _get_provenance_weight(max_provenance)
    boost = (max_provenance - 0.5) * weight
    final_score = model_score + boost * (1.0 - model_score)
    return min(final_score, 1.0), True


def _load_image_meta_learner():
    """Load the trained image meta-learner, scaler, and calibrator."""
    global _image_meta_learner, _image_scaler, _image_model_order
    global _image_calibrator

    config_path = _ARTIFACTS_DIR / "image_config.json"
    model_path = _ARTIFACTS_DIR / "image_meta_learner.pkl"
    scaler_path = _ARTIFACTS_DIR / "image_scaler.pkl"
    calibrator_path = _ARTIFACTS_DIR / "image_calibrator.pkl"

    if not all(p.exists() for p in [config_path, model_path, scaler_path]):
        logger.warning(
            "Image meta-learner artifacts not found at %s. "
            "Falling back to weighted averaging.",
            _ARTIFACTS_DIR,
        )
        return False

    try:
        with open(config_path) as f:
            config = json.load(f)
        _image_model_order = config["model_order"]

        _image_meta_learner = _safe_unpickle(model_path)

        _image_scaler = _safe_unpickle(scaler_path)

        # Platt calibrator is optional
        if calibrator_path.exists():
            _image_calibrator = _safe_unpickle(calibrator_path)
            logger.info("Image calibrator loaded.")
        else:
            _image_calibrator = None

        logger.info(
            "Image meta-learner loaded: %s, CV AUC=%.4f, " "models=%s, calibrated=%s",
            config["algorithm"],
            config["cv_auc"],
            _image_model_order,
            _image_calibrator is not None,
        )
        return True
    except Exception as exc:
        logger.error("Failed to load image meta-learner: %s", exc)
        _image_meta_learner = None
        return False


def _load_video_meta_learner():
    """Load the trained video meta-learner, scaler, and calibrator."""
    global _video_meta_learner, _video_scaler, _video_model_order
    global _video_calibrator

    config_path = _ARTIFACTS_DIR / "video_config.json"
    model_path = _ARTIFACTS_DIR / "video_meta_learner.pkl"
    scaler_path = _ARTIFACTS_DIR / "video_scaler.pkl"
    calibrator_path = _ARTIFACTS_DIR / "video_calibrator.pkl"

    if not all(p.exists() for p in [config_path, model_path, scaler_path]):
        logger.warning(
            "Video meta-learner artifacts not found at %s. "
            "Falling back to weighted averaging.",
            _ARTIFACTS_DIR,
        )
        return False

    try:
        with open(config_path) as f:
            config = json.load(f)
        _video_model_order = config["model_order"]

        # Update AUC weights with measured values
        measured = config.get("measured_model_aucs", {})
        if measured:
            _VIDEO_MODEL_AUC.update(
                {k: v for k, v in measured.items() if k in _VIDEO_MODEL_AUC},
            )

        _video_meta_learner = _safe_unpickle(model_path)

        _video_scaler = _safe_unpickle(scaler_path)

        if calibrator_path.exists():
            _video_calibrator = _safe_unpickle(calibrator_path)
            logger.info("Video calibrator loaded.")
        else:
            _video_calibrator = None

        logger.info(
            "Video meta-learner loaded: %s, CV AUC=%.4f, " "models=%s, calibrated=%s",
            config["algorithm"],
            config["cv_auc"],
            _video_model_order,
            _video_calibrator is not None,
        )
        return True
    except Exception as exc:
        logger.error("Failed to load video meta-learner: %s", exc)
        _video_meta_learner = None
        return False


def _load_audio_meta_learner():
    """Load the trained audio meta-learner, scaler, and calibrator."""
    global _audio_meta_learner, _audio_scaler, _audio_model_order
    global _audio_calibrator

    config_path = _ARTIFACTS_DIR / "audio_config.json"
    model_path = _ARTIFACTS_DIR / "audio_meta_learner.pkl"
    scaler_path = _ARTIFACTS_DIR / "audio_scaler.pkl"
    calibrator_path = _ARTIFACTS_DIR / "audio_calibrator.pkl"

    if not all(p.exists() for p in [config_path, model_path, scaler_path]):
        logger.warning(
            "Audio meta-learner artifacts not found at %s. "
            "Falling back to weighted averaging.",
            _ARTIFACTS_DIR,
        )
        return False

    try:
        with open(config_path) as f:
            config = json.load(f)
        _audio_model_order = config["model_order"]

        # Update AUC weights with measured values
        measured = config.get("measured_model_aucs", {})
        if measured:
            _AUDIO_MODEL_AUC.update(
                {k: v for k, v in measured.items() if k in _AUDIO_MODEL_AUC},
            )

        _audio_meta_learner = _safe_unpickle(model_path)

        _audio_scaler = _safe_unpickle(scaler_path)

        # Platt calibrator is optional
        if calibrator_path.exists():
            _audio_calibrator = _safe_unpickle(calibrator_path)
            logger.info("Audio calibrator loaded.")
        else:
            _audio_calibrator = None

        logger.info(
            "Audio meta-learner loaded: %s, CV AUC=%.4f, " "models=%s, calibrated=%s",
            config["algorithm"],
            config["cv_auc"],
            _audio_model_order,
            _audio_calibrator is not None,
        )
        return True
    except Exception as exc:
        logger.error("Failed to load audio meta-learner: %s", exc)
        _audio_meta_learner = None
        return False


def _normalize_model_name(gateway_name: str) -> str:
    """Map gateway config model names to standard short names."""
    return _MODEL_NAME_MAP.get(gateway_name, gateway_name)


def _extract_probabilities(
    results: Dict[str, Dict],
) -> Dict[str, float]:
    """Extract valid probabilities from model results.

    Returns:
        Dict of standard_model_name -> probability.
    """
    probs = {}
    for gw_name, result in results.items():
        if gw_name in PROVENANCE_SERVICES:
            continue
        if not isinstance(result, dict):
            continue
        if "error" in result:
            continue
        prob = result.get("probability")
        if prob is None:
            continue
        std_name = _normalize_model_name(gw_name)
        probs[std_name] = float(prob)
    return probs


def _ensemble_image(
    probs: Dict[str, float],
    request_id: str,
) -> Tuple[float, bool]:
    """Image ensemble: meta-learner with weighted average fallback.

    Uses a RandomForest meta-learner when all training models are
    present in ``probs``.  Falls back to signal-filtered AUC-weighted
    averaging otherwise.  Platt calibration is applied post-hoc when
    the calibrator artifact is available.

    Args:
        probs: Dict of standard_model_name -> probability.
        request_id: For logging.

    Returns:
        Tuple of (ensemble_prob, used_meta_learner).
    """
    # Try meta-learner first (requires all models in model_order)
    used_ml = False
    if (
        _image_meta_learner is not None
        and _image_scaler is not None
        and _image_model_order is not None
        and all(m in probs for m in _image_model_order)
    ):
        feature_vec = np.array(
            [[probs[m] for m in _image_model_order]],
        )
        scaled = _image_scaler.transform(feature_vec)
        ensemble_prob = float(
            _image_meta_learner.predict_proba(scaled)[0, 1],
        )

        # Apply Platt calibration if available
        if _image_calibrator is not None:
            ensemble_prob = float(
                _image_calibrator.predict_proba(
                    np.array([[ensemble_prob]]),
                )[0, 1],
            )

        used_ml = True
        logger.info(
            "Request %s (image): Meta-learner score=%.4f " "(%d models, calibrated=%s)",
            request_id,
            ensemble_prob,
            len(_image_model_order),
            _image_calibrator is not None,
        )
        return ensemble_prob, True

    # Fallback: AUC-weighted average with signal filtering.
    # Models scoring below SIGNAL_THRESHOLD are treated as "no signal"
    # and excluded to prevent confidently-wrong old models from
    # dragging the score down on modern generators.
    model_auc = {
        "fsd": 0.8961,
        "cospy": 0.8679,
        "effort": 0.8353,
        "aide": 0.7478,
        "universal": 0.6711,
        "npr": 0.6373,
        "yermandy": 0.5912,
    }

    _SIGNAL_THRESHOLD = 0.02  # Optimized via threshold sweep (AUC=0.911)

    # First pass: use only models with signal
    signal_models = {m: p for m, p in probs.items() if p >= _SIGNAL_THRESHOLD}

    if signal_models:
        total_weight = 0.0
        weighted_sum = 0.0
        for model_name, prob in signal_models.items():
            weight = model_auc.get(model_name, 0.5)
            weighted_sum += prob * weight
            total_weight += weight
        ensemble_prob = weighted_sum / total_weight if total_weight > 0 else 0.5
    else:
        # All models below threshold — use full weighted average
        total_weight = 0.0
        weighted_sum = 0.0
        for model_name, prob in probs.items():
            weight = model_auc.get(model_name, 0.5)
            weighted_sum += prob * weight
            total_weight += weight
        ensemble_prob = weighted_sum / total_weight if total_weight > 0 else 0.5

    logger.info(
        "Request %s (image): Weighted average score=%.4f (%d models)",
        request_id,
        ensemble_prob,
        len(probs),
    )
    return ensemble_prob, False


def _ensemble_audio(
    probs: Dict[str, float],
    request_id: str,
) -> Tuple[float, bool]:
    """Audio ensemble: meta-learner with weighted average fallback.

    Uses a trained meta-learner when all training models are present
    in ``probs``.  Falls back to AUC-weighted averaging otherwise.
    Platt calibration is applied post-hoc when the calibrator artifact
    is available.

    Args:
        probs: Dict of standard_model_name -> probability.
        request_id: For logging.

    Returns:
        Tuple of (ensemble_prob, used_meta_learner).
    """
    if not probs:
        return 0.5, False

    # Try meta-learner — impute missing models with 0.5 (neutral)
    # so the RF runs even when removed models (e.g., SONICS) are
    # absent. 0.5 is the no-signal default; the RF learned to give
    # near-random models low weight, so imputing 0.5 is safe.
    # All 3 audio models (ShiftySpeech, SafeEar, Nes2Net) should
    # always produce results in production.
    if (
        _audio_meta_learner is not None
        and _audio_model_order is not None
        and _audio_scaler is not None
    ):
        try:
            feature_vec = np.array(
                [[probs.get(m, 0.5) for m in _audio_model_order]],
            )
            scaled = _audio_scaler.transform(feature_vec)
            score = float(
                _audio_meta_learner.predict_proba(scaled)[0, 1],
            )

            # Apply Platt calibration if available
            if _audio_calibrator is not None:
                score = float(
                    _audio_calibrator.predict_proba(
                        np.array([[score]]),
                    )[0, 1],
                )

            logger.info(
                "Request %s (audio): Meta-learner score=%.4f "
                "(models=%s, calibrated=%s)",
                request_id,
                score,
                list(_audio_model_order),
                _audio_calibrator is not None,
            )
            return score, True
        except Exception as exc:
            logger.warning(
                "Request %s (audio): Meta-learner failed (%s), "
                "falling back to weighted average.",
                request_id,
                exc,
            )

    # Fallback: AUC-weighted average
    if len(probs) == 1:
        model_name = next(iter(probs))
        score = probs[model_name]
        logger.info(
            "Request %s (audio): Single model %s, score=%.4f",
            request_id,
            model_name,
            score,
        )
        return score, False

    total_weight = 0.0
    weighted_sum = 0.0
    for model_name, prob in probs.items():
        weight = _AUDIO_MODEL_AUC.get(model_name, 0.5)
        weighted_sum += prob * weight
        total_weight += weight

    score = weighted_sum / total_weight if total_weight > 0 else 0.5
    logger.info(
        "Request %s (audio): Weighted average score=%.4f (%d models)",
        request_id,
        score,
        len(probs),
    )
    return score, False


_VIDEO_MODEL_AUC = {
    "fakestormer": 0.6618,
    "sbi": 0.6781,
    "dfd_fcg": 0.5531,
    "pwtf_dvd": 0.5850,
    "lipfd": 0.6312,
    "recce": 0.6625,
    "mintime": 0.5813,
    "npr_video": 0.8022,
    "univfd_video": 0.6798,
}


def _ensemble_video(
    probs: Dict[str, float],
    request_id: str,
) -> Tuple[float, bool]:
    """Video ensemble: meta-learner with weighted average fallback.

    Uses a trained 9-model XGBoost meta-learner (CV AUC=0.8898).
    Imputes missing models with 0.5 (neutral) so the model runs even
    when some models fail on specific samples (e.g., no face detected).

    Returns:
        Tuple of (score, used_meta_learner).
    """
    if not probs:
        return 0.5, False

    # Try meta-learner — impute missing models with 0.5
    if (
        _video_meta_learner is not None
        and _video_model_order is not None
        and _video_scaler is not None
    ):
        try:
            feature_vec = np.array(
                [[probs.get(m, 0.5) for m in _video_model_order]],
            )
            scaled = _video_scaler.transform(feature_vec)
            score = float(
                _video_meta_learner.predict_proba(scaled)[0, 1],
            )

            # Apply Platt calibration if available
            if _video_calibrator is not None:
                score = float(
                    _video_calibrator.predict_proba(
                        np.array([[score]]),
                    )[0, 1],
                )

            logger.info(
                "Request %s (video): Meta-learner score=%.4f " "(models=%s)",
                request_id,
                score,
                list(_video_model_order),
            )
            return score, True
        except Exception as exc:
            logger.warning(
                "Request %s (video): Meta-learner failed (%s), "
                "falling back to weighted average.",
                request_id,
                exc,
            )

    # Fallback: AUC-weighted average
    if len(probs) == 1:
        model_name = next(iter(probs))
        score = probs[model_name]
        logger.info(
            "Request %s (video): Single model %s, score=%.4f",
            request_id,
            model_name,
            score,
        )
        return score, False

    total_weight = 0.0
    weighted_sum = 0.0
    for model_name, prob in probs.items():
        weight = _VIDEO_MODEL_AUC.get(model_name, 0.5)
        weighted_sum += prob * weight
        total_weight += weight

    score = weighted_sum / total_weight if total_weight > 0 else 0.5
    logger.info(
        "Request %s (video): Weighted average score=%.4f (%d models)",
        request_id,
        score,
        len(probs),
    )
    return score, False


def calculate_ensemble_verdict_api(
    results: Dict[str, Dict],
    threshold: float,
    method: str,
    media_type: str,
    request_id: str,
) -> Tuple[str, float, int, int, float, str]:
    """Calculate ensemble verdict using modality-specific strategies.

    Args:
        results: Dict of model_name -> result dict with
            probability/prediction.
        threshold: Classification threshold (default 0.5).
        method: Accepted for backwards compatibility but ignored.
            The ensemble method is selected per-modality.
        media_type: The media type (image/video/audio).
        request_id: Request ID for logging.

    Returns:
        Tuple of (verdict, confidence, fake_votes, real_votes,
        score, method_used).
    """
    # Extract valid probabilities with normalized model names
    probs = _extract_probabilities(results)

    if not probs:
        logger.warning(
            "Request %s (%s): No valid model results for ensemble.",
            request_id,
            media_type,
        )
        return "undetermined", 0.0, 0, 0, 0.5, "none"

    # Count votes from raw results (for backward compat in response)
    valid_results = {
        k: v
        for k, v in results.items()
        if isinstance(v, dict) and "error" not in v and v.get("prediction") is not None
    }
    fake_votes = sum(1 for r in valid_results.values() if r.get("prediction") == 1)
    real_votes = sum(1 for r in valid_results.values() if r.get("prediction") == 0)

    # Modality-specific ensemble
    if media_type == "image":
        # Lazy-load meta-learner artifacts on first request
        if _image_meta_learner is None:
            _load_image_meta_learner()
        ensemble_score, used_ml = _ensemble_image(probs, request_id)
        method_used = "meta_learner" if used_ml else "weighted_avg"
    elif media_type == "audio":
        if _audio_meta_learner is None:
            _load_audio_meta_learner()
        ensemble_score, used_ml = _ensemble_audio(probs, request_id)
        method_used = "meta_learner" if used_ml else "weighted_avg"
    elif media_type == "video":
        if _video_meta_learner is None:
            _load_video_meta_learner()
        ensemble_score, used_ml = _ensemble_video(probs, request_id)
        method_used = "meta_learner" if used_ml else "weighted_avg"
    else:
        # Unknown media type: fall back to simple average
        ensemble_score = sum(probs.values()) / len(probs)
        method_used = "average"
        logger.warning(
            "Request %s: Unknown media_type '%s', " "using simple average.",
            request_id,
            media_type,
        )

    # Stage 2: Provenance boost
    ensemble_score, provenance_boosted = apply_provenance_boost(ensemble_score, results)
    if provenance_boosted:
        method_used = f"{method_used}+provenance_boost"

    verdict = "fake" if ensemble_score >= threshold else "real"
    confidence = float(ensemble_score if verdict == "fake" else (1.0 - ensemble_score))

    logger.info(
        "Request %s (%s): Ensemble '%s' — score=%.4f, "
        "verdict=%s, confidence=%.4f, models=%d/%d",
        request_id,
        media_type,
        method_used,
        ensemble_score,
        verdict,
        confidence,
        len(probs),
        len(results),
    )

    return (
        verdict,
        confidence,
        fake_votes,
        real_votes,
        ensemble_score,
        method_used,
    )
