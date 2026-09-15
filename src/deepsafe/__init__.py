"""DeepSafe: a benchmark, model zoo, and adaptation toolkit for deepfake detection.

Three entry points:

* ``deepsafe detect`` runs the model ensemble over a file.
* ``deepsafe eval`` scores any detector against the benchmark.
* ``deepsafe fit`` adapts the stack to your own labeled data.

The package deliberately keeps ``eval`` and ``fit`` Tier 1 free of heavy
dependencies so they run on a laptop with no GPU.
"""

__version__ = "1.0.0"

CITATION = """\
@software{deepsafe2026,
  title  = {DeepSafe: A Multi-Modal Deepfake Detection Benchmark and Model Zoo},
  author = {Sah, Siddharth},
  year   = {2026},
  url    = {https://github.com/deepsafehq/deepsafe-bench},
  note   = {Free for non-commercial use under PolyForm Noncommercial 1.0.0}
}"""


def citation() -> str:
    """Return the BibTeX entry for citing DeepSafe."""
    return CITATION


__all__ = ["__version__", "CITATION", "citation"]
