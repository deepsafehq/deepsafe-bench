"""Shared constants for DeepSafe services.

Provenance name mappings, media type enums, and model identifiers
used by both gateway and inference.
"""

# Monolith provenance model names -> gateway provenance names.
# Used by the inference ensemble adapter to translate results before
# passing them to the shared ensemble functions.
PROVENANCE_NAME_MAP: dict[str, str] = {
    "c2pa": "c2pa_checker",
    "sdxl_watermark": "sdxl_watermark_detector",
    "audioseal": "audioseal_detector",
    "videoseal": "videoseal_detector",
    "trustmark": "trustmark_detector",
}

# Media modality types supported by the ensemble.
MEDIA_TYPES = ("image", "video", "audio", "provenance")
