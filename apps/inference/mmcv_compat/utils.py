"""Pure PyTorch replacements for mmcv.utils functions."""

import logging
from typing import Union


def to_2tuple(x):
    """Convert a value to a 2-tuple.

    If x is already a tuple/list, return as tuple.
    If x is a scalar, return (x, x).
    """
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x, x)


def get_logger(
    name: str,
    log_file: str = None,
    log_level: int = logging.INFO,
) -> logging.Logger:
    """Get a logger by name, optionally with a file handler.

    Replacement for mmcv.utils.get_logger.
    """
    _logger = logging.getLogger(name)
    if not _logger.handlers:
        _logger.setLevel(log_level)
        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        _logger.addHandler(handler)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        _logger.addHandler(file_handler)
    return _logger
