"""A minimal detector, to show the interface and to calibrate expectations.

Score any detector against the benchmark with:

    deepsafe eval --model examples/random_detector.py --tier small

This one returns a deterministic pseudo-random score derived from the file
bytes. It exists as a template and as a floor: a real detector that cannot
beat this is not detecting anything.
"""

import hashlib
import pathlib


def predict(path: pathlib.Path) -> float:
    """Return P(synthetic) in [0, 1].

    The only contract DeepSafe requires. Anything implementing it is scoreable.

    Args:
        path: Media file to score.

    Returns:
        Probability the media is synthetic.
    """
    digest = hashlib.sha256(pathlib.Path(path).read_bytes()).digest()
    return int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
