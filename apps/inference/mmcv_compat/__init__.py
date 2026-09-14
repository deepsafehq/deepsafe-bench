"""mmcv/mmengine compatibility layer for FakeSTormer.

Provides pure PyTorch replacements for mmcv v1 and mmengine functions
so FakeSTormer runs on torch 2.5.1 without installing the real packages.

Usage:
    from mmcv_compat import inject_all
    inject_all()  # Must be called BEFORE importing FakeSTormer code
"""

import logging
import sys
import types
from pathlib import Path

logger = logging.getLogger("mmcv_compat")


def _make_module(name: str, attrs: dict) -> types.ModuleType:
    """Create a fake module with the given attributes."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def inject_all() -> None:
    """Inject all mmcv, mmengine, and ptflops stubs into sys.modules.

    Call this ONCE before importing any FakeSTormer model_code.
    Safe to call multiple times (idempotent).
    """
    if "mmcv" in sys.modules and hasattr(sys.modules["mmcv"], "_is_compat"):
        return  # Already injected

    from mmcv_compat.cnn import (
        FFN,
        build_conv_layer,
        build_dropout,
        build_norm_layer,
        build_upsample_layer,
        constant_init,
        normal_init,
        trunc_normal_,
        trunc_normal_init,
    )
    from mmcv_compat.parallel import is_module_wrapper
    from mmcv_compat.runner import BaseModule, _load_checkpoint, auto_fp16
    from mmcv_compat.utils import get_logger, to_2tuple

    # ── mmcv top-level ───────────────────────────────────────────────
    mmcv = _make_module("mmcv", {"_is_compat": True})
    sys.modules["mmcv"] = mmcv

    # ── mmcv.cnn ─────────────────────────────────────────────────────
    mmcv_cnn = _make_module(
        "mmcv.cnn",
        {
            "build_conv_layer": build_conv_layer,
            "build_norm_layer": build_norm_layer,
            "build_upsample_layer": build_upsample_layer,
            "constant_init": constant_init,
            "normal_init": normal_init,
            "trunc_normal_init": trunc_normal_init,
        },
    )
    sys.modules["mmcv.cnn"] = mmcv_cnn
    mmcv.cnn = mmcv_cnn

    # mmcv.cnn.bricks.transformer
    mmcv_cnn_bricks = _make_module("mmcv.cnn.bricks", {})
    mmcv_cnn_bricks_transformer = _make_module(
        "mmcv.cnn.bricks.transformer",
        {
            "FFN": FFN,
            "build_dropout": build_dropout,
        },
    )
    sys.modules["mmcv.cnn.bricks"] = mmcv_cnn_bricks
    sys.modules["mmcv.cnn.bricks.transformer"] = mmcv_cnn_bricks_transformer
    mmcv_cnn.bricks = mmcv_cnn_bricks
    mmcv_cnn_bricks.transformer = mmcv_cnn_bricks_transformer

    # mmcv.cnn.utils.weight_init
    mmcv_cnn_utils = _make_module("mmcv.cnn.utils", {})
    mmcv_cnn_utils_weight_init = _make_module(
        "mmcv.cnn.utils.weight_init",
        {
            "trunc_normal_": trunc_normal_,
        },
    )
    sys.modules["mmcv.cnn.utils"] = mmcv_cnn_utils
    sys.modules["mmcv.cnn.utils.weight_init"] = mmcv_cnn_utils_weight_init
    mmcv_cnn_utils.weight_init = mmcv_cnn_utils_weight_init

    # ── mmcv.runner ──────────────────────────────────────────────────
    mmcv_runner = _make_module(
        "mmcv.runner",
        {
            "auto_fp16": auto_fp16,
            "_load_checkpoint": _load_checkpoint,
            "BaseModule": BaseModule,
        },
    )
    mmcv_runner_base_module = _make_module(
        "mmcv.runner.base_module",
        {
            "BaseModule": BaseModule,
        },
    )
    sys.modules["mmcv.runner"] = mmcv_runner
    sys.modules["mmcv.runner.base_module"] = mmcv_runner_base_module
    mmcv.runner = mmcv_runner
    mmcv_runner.base_module = mmcv_runner_base_module

    # ── mmcv.utils ───────────────────────────────────────────────────
    mmcv_utils = _make_module(
        "mmcv.utils",
        {
            "to_2tuple": to_2tuple,
            "get_logger": get_logger,
        },
    )
    sys.modules["mmcv.utils"] = mmcv_utils
    mmcv.utils = mmcv_utils

    # ── mmcv.parallel ────────────────────────────────────────────────
    mmcv_parallel = _make_module(
        "mmcv.parallel",
        {
            "is_module_wrapper": is_module_wrapper,
        },
    )
    sys.modules["mmcv.parallel"] = mmcv_parallel
    mmcv.parallel = mmcv_parallel

    # ── mmengine stubs ───────────────────────────────────────────────
    _inject_mmengine()

    # ── ptflops stub ─────────────────────────────────────────────────
    _inject_ptflops()

    logger.info("mmcv/mmengine/ptflops compatibility layer injected")


def _inject_mmengine() -> None:
    """Inject mmengine stubs for FakeSTormer's utility imports."""
    import os

    import torch

    # mmengine top-level
    mmengine = _make_module("mmengine", {})
    sys.modules["mmengine"] = mmengine

    # mmengine.config
    class ConfigDict(dict):
        """Minimal ConfigDict stub."""

        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError:
                raise AttributeError(key)

        def __setattr__(self, key, value):
            self[key] = value

    class Config(ConfigDict):
        """Minimal Config stub."""

        @staticmethod
        def fromfile(filename):
            return Config()

    mmengine_config = _make_module(
        "mmengine.config",
        {
            "Config": Config,
            "ConfigDict": ConfigDict,
        },
    )
    sys.modules["mmengine.config"] = mmengine_config
    mmengine.config = mmengine_config

    # mmengine.dist
    def get_dist_info():
        return 0, 1  # rank=0, world_size=1

    def master_only(func):
        return func  # No-op decorator in single-process mode

    mmengine_dist = _make_module(
        "mmengine.dist",
        {
            "get_dist_info": get_dist_info,
            "master_only": master_only,
        },
    )
    sys.modules["mmengine.dist"] = mmengine_dist
    mmengine.dist = mmengine_dist

    # mmengine.fileio
    class FileClient:
        """Minimal stub -- only used in training code paths."""

        def __init__(self, *args, **kwargs):
            pass

    def _load_file(file, *args, **kwargs):
        return {}

    mmengine_fileio = _make_module(
        "mmengine.fileio",
        {
            "FileClient": FileClient,
            "load": _load_file,
        },
    )
    sys.modules["mmengine.fileio"] = mmengine_fileio
    mmengine.fileio = mmengine_fileio

    # mmengine.utils
    def mkdir_or_exist(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    mmengine_utils = _make_module(
        "mmengine.utils",
        {
            "mkdir_or_exist": mkdir_or_exist,
        },
    )
    sys.modules["mmengine.utils"] = mmengine_utils
    mmengine.utils = mmengine_utils

    # mmengine.structures
    class InstanceData:
        """Minimal stub."""

        pass

    class PixelData:
        """Minimal stub."""

        pass

    mmengine_structures = _make_module(
        "mmengine.structures",
        {
            "InstanceData": InstanceData,
            "PixelData": PixelData,
        },
    )
    sys.modules["mmengine.structures"] = mmengine_structures
    mmengine.structures = mmengine_structures

    # mmengine.logging
    class MMLogger:
        """Minimal stub."""

        @staticmethod
        def get_current_instance():
            return logging.getLogger("mmengine")

    mmengine_logging = _make_module(
        "mmengine.logging",
        {
            "MMLogger": MMLogger,
        },
    )
    sys.modules["mmengine.logging"] = mmengine_logging
    mmengine.logging = mmengine_logging

    # mmengine.model
    def xavier_init(module, gain=1, bias=0, distribution="uniform"):
        if hasattr(module, "weight") and module.weight is not None:
            if distribution == "uniform":
                torch.nn.init.xavier_uniform_(module.weight, gain=gain)
            else:
                torch.nn.init.xavier_normal_(module.weight, gain=gain)
        if hasattr(module, "bias") and module.bias is not None:
            torch.nn.init.constant_(module.bias, bias)

    mmengine_model = _make_module(
        "mmengine.model",
        {
            "xavier_init": xavier_init,
        },
    )
    sys.modules["mmengine.model"] = mmengine_model
    mmengine.model = mmengine_model


def _inject_ptflops() -> None:
    """Inject ptflops stub (only used for training/profiling)."""

    def get_model_complexity_info(*args, **kwargs):
        return "N/A", "N/A"

    ptflops = _make_module(
        "ptflops",
        {
            "get_model_complexity_info": get_model_complexity_info,
        },
    )
    sys.modules["ptflops"] = ptflops


def cleanup() -> None:
    """Remove injected modules from sys.modules.

    Call after FakeSTormer model is fully loaded and ready for inference.
    """
    prefixes = ("mmcv", "mmengine", "ptflops")
    to_remove = [
        key
        for key in sys.modules
        if any(key == p or key.startswith(p + ".") for p in prefixes)
    ]
    for key in to_remove:
        del sys.modules[key]
    logger.debug("Cleaned up %d injected modules", len(to_remove))
