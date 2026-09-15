"""A deliberately naive detector that exploits a dataset artifact.

    deepsafe eval --model examples/filesize_detector.py --tier small

Real and synthetic media in a benchmark often differ in file size, because they
came from different pipelines rather than because one is fake. A detector that
scores well here and nowhere else has learned the collection process, not
manipulation.

Included as a worked example of why `deepsafe eval` reports per-generator
results: this detector's aggregate number can look respectable while its
per-generator spread gives the game away.
"""

import pathlib


def predict(path: pathlib.Path) -> float:
    """Score by file size alone, which is not evidence of anything."""
    size = pathlib.Path(path).stat().st_size
    return min(1.0, size / 2_000_000)
