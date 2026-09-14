"""Pure PyTorch replacements for mmcv.runner functions.

Covers auto_fp16, _load_checkpoint, and BaseModule.
"""

import functools
import logging
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger("mmcv_compat")


def auto_fp16(apply_to=None, out_fp32=False):
    """No-op decorator replacing mmcv.runner.auto_fp16.

    In inference mode with torch.inference_mode(), mixed precision is
    handled by PyTorch's autocast. This decorator does nothing.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _load_checkpoint(
    filename: Union[str, Path],
    map_location: Optional[str] = None,
    logger=None,
) -> dict:
    """Load a checkpoint file.

    Replacement for mmcv.runner._load_checkpoint that handles common
    checkpoint formats (state_dict, model key, etc.).

    Args:
        filename: Path to checkpoint file.
        map_location: Device mapping for torch.load.
        logger: Optional logger (ignored, uses module logger).

    Returns:
        Loaded checkpoint dict.
    """
    checkpoint = torch.load(
        str(filename),
        map_location=map_location,
        weights_only=False,
    )

    # Handle nested checkpoint formats
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            return checkpoint
        if "model" in checkpoint:
            checkpoint["state_dict"] = checkpoint["model"]
            return checkpoint
    return checkpoint


class BaseModule(nn.Module):
    """Drop-in replacement for mmcv.runner.BaseModule.

    mmcv's BaseModule adds init_weights() and init_cfg handling on top
    of nn.Module. For inference, we only need nn.Module behavior.
    """

    def __init__(self, init_cfg=None):
        super().__init__()
        self.init_cfg = init_cfg

    def init_weights(self):
        """No-op: weights are loaded from checkpoint, not initialized."""
        pass
