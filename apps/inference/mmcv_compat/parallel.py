"""Pure PyTorch replacements for mmcv.parallel functions."""

import torch.nn as nn


def is_module_wrapper(module: nn.Module) -> bool:
    """Check if a module is a DataParallel wrapper.

    Replacement for mmcv.parallel.is_module_wrapper.
    """
    return isinstance(module, (nn.DataParallel, nn.parallel.DistributedDataParallel))
