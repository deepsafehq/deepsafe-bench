"""Ensemble adapter for inference server.

Imports ensemble logic from the shared package and adapts the
interface for inference model names.
"""

import logging
from typing import Dict

from deepsafe_shared.constants import PROVENANCE_NAME_MAP
from deepsafe_shared.ensemble import (
    _ensemble_audio,
    _ensemble_image,
    _ensemble_video,
    _load_audio_meta_learner,
    _load_image_meta_learner,
    _load_video_meta_learner,
    apply_provenance_boost,
)

logger = logging.getLogger("inference.ensemble")


def compute_ensemble(
    model_results: Dict[str, dict],
    media_type: str,
    threshold: float = 0.5,
    request_id: str = "inference",
) -> dict:
    """Compute ensemble verdict from inference model results.

    Args:
        model_results: Dict of model_name -> predict() result.
            Names are standard names (npr, aide, etc.).
        media_type: image | video | audio.
        threshold: Classification threshold.
        request_id: For logging.

    Returns:
        Dict with verdict, confidence, score, method, model_results.
    """
    detection_probs = {}
    provenance_results = {}

    for name, result in model_results.items():
        if not isinstance(result, dict):
            continue
        if result.get("probability") is None:
            continue

        if name in PROVENANCE_NAME_MAP:
            gw_name = PROVENANCE_NAME_MAP[name]
            provenance_results[gw_name] = result
        else:
            detection_probs[name] = float(result["probability"])

    if not detection_probs:
        return {
            "verdict": "undetermined",
            "confidence": 0.0,
            "score": 0.5,
            "method": "none",
            "models_used": 0,
        }

    if media_type == "image":
        _load_image_meta_learner()
        score, used_ml = _ensemble_image(detection_probs, request_id)
        method = "meta_learner" if used_ml else "weighted_avg"
    elif media_type == "audio":
        _load_audio_meta_learner()
        score, used_ml = _ensemble_audio(detection_probs, request_id)
        method = "meta_learner" if used_ml else "weighted_avg"
    elif media_type == "video":
        _load_video_meta_learner()
        score, used_ml = _ensemble_video(detection_probs, request_id)
        method = "meta_learner" if used_ml else "weighted_avg"
    else:
        score = sum(detection_probs.values()) / len(detection_probs)
        method = "average"

    if provenance_results:
        score, boosted = apply_provenance_boost(score, provenance_results)
        if boosted:
            method = f"{method}+provenance_boost"

    verdict = "fake" if score >= threshold else "real"
    confidence = score if verdict == "fake" else (1.0 - score)

    return {
        "verdict": verdict,
        "confidence": round(float(confidence), 4),
        "score": round(float(score), 4),
        "method": method,
        "models_used": len(detection_probs),
    }
