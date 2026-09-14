"""Model wrappers for all 21 DeepSafe detection models.

Each wrapper implements BasePredictor: load(), predict(), health().
Import MODEL_CLASSES to get the name->class mapping.
"""

from models.base import BasePredictor

# Lazy registry: maps config name -> (module_path, class_name)
# Avoids importing all models at module load time.
_WRAPPER_REGISTRY = {
    # Image (7)
    "npr": ("models.npr", "NPRPredictor"),
    "yermandy": ("models.yermandy", "YermandyPredictor"),
    "universal": ("models.universal", "UniversalPredictor"),
    "aide": ("models.aide", "AIDEPredictor"),
    "fsd": ("models.fsd", "FSDPredictor"),
    "effort": ("models.effort", "EffortPredictor"),
    "cospy": ("models.cospy", "COSPYPredictor"),
    # Video (7)
    "fakestormer": ("models.fakestormer", "FakeSTormerPredictor"),
    "sbi": ("models.sbi", "SBIPredictor"),
    "dfd_fcg": ("models.dfd_fcg", "DFDFCGPredictor"),
    "pwtf_dvd": ("models.pwtf_dvd", "PwTFDVDPredictor"),
    "lipfd": ("models.lipfd", "LipFDPredictor"),
    "recce": ("models.recce", "RECCEPredictor"),
    "mintime": ("models.mintime", "MINTIMEPredictor"),
    # Video: Full-Frame Detectors (no face detection)
    "npr_video": ("models.npr_video", "NPRVideoPredictor"),
    "univfd_video": ("models.univfd_video", "UnivFDVideoPredictor"),
    # Audio (3) — SONICS removed (fake song detector), AASIST3 removed (hangs on load)
    "shiftyspeech": ("models.shiftyspeech", "ShiftySpeechPredictor"),
    "safeear": ("models.safeear", "SafeEarPredictor"),
    "nes2net": ("models.nes2net", "Nes2NetPredictor"),
    # Provenance (5)
    "c2pa": ("models.c2pa", "C2PAPredictor"),
    "sdxl_watermark": ("models.sdxl_watermark", "SDXLWatermarkPredictor"),
    "audioseal": ("models.audioseal", "AudioSealPredictor"),
    "videoseal": ("models.videoseal", "VideoSealPredictor"),
    "trustmark": ("models.trustmark_wm", "TrustMarkPredictor"),
}


def get_predictor_class(name: str) -> type:
    """Import and return the predictor class for a model name.

    Before importing, clean any vendored generic modules (e.g., "models",
    "src", "utils") from sys.modules that were left by a previous model's
    namespaced_import/load.  This ensures ``importlib.import_module``
    resolves ``models.aide`` to *our* ``apps/inference/models/aide.py``
    rather than a stale vendored ``models`` package.
    """
    import importlib
    import sys

    from model_loader import _CONFLICTING_PREFIXES

    if name not in _WRAPPER_REGISTRY:
        raise KeyError(f"Unknown model: {name}. Available: {list(_WRAPPER_REGISTRY)}")

    # Clean vendored modules that shadow our own package
    _our_dir = str(__path__[0])  # e.g. .../apps/inference/models
    _our_parent = str(__path__[0].rsplit("/", 1)[0]) if "/" in str(__path__[0]) else ""
    for prefix in _CONFLICTING_PREFIXES:
        if prefix in sys.modules:
            cur_file = getattr(sys.modules[prefix], "__file__", "") or ""
            if _our_parent and _our_parent not in cur_file:
                to_remove = [
                    k for k in sys.modules if k == prefix or k.startswith(prefix + ".")
                ]
                for k in to_remove:
                    del sys.modules[k]

    module_path, class_name = _WRAPPER_REGISTRY[name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


__all__ = ["BasePredictor", "get_predictor_class", "_WRAPPER_REGISTRY"]
