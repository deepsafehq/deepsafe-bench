"""Loading third-party detectors for evaluation.

A detector is any Python file exposing a callable ``predict(path) -> float``
returning the probability that the media is synthetic. That is the entire
contract; nothing about the model's internals is assumed.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class Detector(Protocol):
    """The interface a submitted detector must satisfy."""

    def predict(self, path: pathlib.Path) -> float:
        """Return P(synthetic) in [0, 1] for the media at ``path``."""


class DetectorLoadError(RuntimeError):
    """Raised when a detector file cannot be loaded or lacks ``predict``."""


def load_detector(path: pathlib.Path) -> Callable[[pathlib.Path], float]:
    """Load a ``predict`` callable from a Python file.

    Accepts either a module-level ``predict`` function or a ``Detector`` class
    with a ``predict`` method, since both conventions are common.

    Args:
        path: Path to a ``.py`` file.

    Returns:
        A callable taking a path and returning P(synthetic).

    Raises:
        DetectorLoadError: If the file is missing, fails to import, or exposes
            no usable ``predict``.
    """
    path = pathlib.Path(path).resolve()
    if not path.exists():
        raise DetectorLoadError(f"no such detector file: {path}")

    spec = importlib.util.spec_from_file_location(f"_ds_detector_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise DetectorLoadError(f"could not load {path} as a Python module")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise DetectorLoadError(f"{path.name} raised while importing: {exc}") from exc

    predict = getattr(module, "predict", None)
    if callable(predict):
        return predict

    detector_cls = getattr(module, "Detector", None)
    if detector_cls is not None:
        instance = detector_cls()
        if callable(getattr(instance, "predict", None)):
            return instance.predict

    raise DetectorLoadError(
        f"{path.name} exposes no `predict`.\n"
        f"Define either:\n"
        f"    def predict(path) -> float: ...\n"
        f"or a class `Detector` with a `predict` method."
    )
