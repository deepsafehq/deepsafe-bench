"""Pre-eval smoke test: validate every loaded model produces valid output.

Lesson learned from A100 40GB: FSD and Universal passed health checks
but failed every single predict call (24,999 wasted inferences).
This test catches that BEFORE running full experiments.

Usage:
    python -m monolith.smoke_test              # test all loaded models
    python -m monolith.smoke_test --models npr,aide  # test specific models
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger("smoke_test")


# Minimal test fixtures (embedded so no external files needed)
_FIXTURES_DIR = Path(__file__).parent / "tests" / "fixtures"


def _get_test_files(modality: str) -> list[Path]:
    """Get test fixture files for a modality.

    Falls back to listing whatever is in the fixtures dir.
    """
    fixtures = _FIXTURES_DIR / modality
    if not fixtures.exists():
        logger.warning("No fixtures dir for %s at %s", modality, fixtures)
        return []
    files = sorted(fixtures.iterdir())
    return files[:3]  # Max 3 per model


def smoke_test_model(name: str, model, fixtures_dir: Path = _FIXTURES_DIR) -> dict:
    """Test a single model with fixture files.

    Returns:
        {"name": str, "passed": bool, "results": list, "error": str|None}
    """
    test_files = _get_test_files(model.modality)
    if not test_files:
        return {
            "name": name,
            "passed": False,
            "results": [],
            "error": f"No test fixtures for modality={model.modality}",
        }

    results = []
    passed = True
    for f in test_files:
        try:
            result = model.predict(f.read_bytes())
            prob = result.get("probability")
            if prob is None:
                logger.error(
                    "  SMOKE FAIL: %s on %s -> error: %s",
                    name,
                    f.name,
                    result.get("error"),
                )
                passed = False
            elif not (0.0 <= prob <= 1.0):
                logger.error(
                    "  SMOKE FAIL: %s on %s -> probability=%s (out of range)",
                    name,
                    f.name,
                    prob,
                )
                passed = False
            else:
                logger.info(
                    "  SMOKE OK: %s on %s -> prob=%.4f, latency=%.1fms",
                    name,
                    f.name,
                    prob,
                    result.get("latency_ms", 0),
                )
            results.append(result)
        except Exception as e:
            logger.error("  SMOKE FAIL: %s on %s -> exception: %s", name, f.name, e)
            passed = False
            results.append({"probability": None, "error": str(e)})

    return {"name": name, "passed": passed, "results": results, "error": None}


def smoke_test_all(
    models: dict,
    fixtures_dir: Path = _FIXTURES_DIR,
) -> dict[str, bool]:
    """Run smoke test on all loaded models.

    Args:
        models: Dict of name -> loaded predictor instance.
        fixtures_dir: Path to test fixtures.

    Returns:
        Dict of model_name -> passed (bool).
    """
    logger.info("=" * 60)
    logger.info("SMOKE TEST: %d models", len(models))
    logger.info("=" * 60)

    results = {}
    start = time.perf_counter()

    for name, model in models.items():
        result = smoke_test_model(name, model, fixtures_dir)
        results[name] = result["passed"]
        status = "PASS" if result["passed"] else "FAIL"
        logger.info("  %s: %s", name, status)

    elapsed = time.perf_counter() - start
    passed = sum(1 for v in results.values() if v)
    failed = len(results) - passed

    logger.info("-" * 60)
    logger.info(
        "SMOKE TEST COMPLETE: %d/%d passed, %d failed (%.1fs)",
        passed,
        len(results),
        failed,
        elapsed,
    )
    if failed:
        failed_names = [n for n, v in results.items() if not v]
        logger.warning("FAILED MODELS: %s", ", ".join(failed_names))
    logger.info("=" * 60)

    return results


def main():
    """CLI entry point for standalone smoke testing."""
    parser = argparse.ArgumentParser(description="Smoke test monolith models")
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated model names or 'all'",
    )
    parser.add_argument(
        "--fixtures",
        type=str,
        default=str(_FIXTURES_DIR),
        help="Path to test fixtures directory",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Import here to avoid circular imports
    from config import (
        get_enabled_models,
        get_weights_path,
    )
    from models.base import get_device, setup_inference_optimizations

    from models import get_predictor_class

    device = get_device()
    setup_inference_optimizations(device)

    # Determine which models to test
    if args.models == "all":
        model_names = get_enabled_models()
    else:
        model_names = [m.strip() for m in args.models.split(",")]

    # Load models
    loaded = {}
    for name in model_names:
        try:
            cls = get_predictor_class(name)
            predictor = cls()
            weights_dir = get_weights_path(name)
            predictor.load(weights_dir, device)
            loaded[name] = predictor
            logger.info("Loaded %s", name)
        except Exception as e:
            logger.error("Failed to load %s: %s", name, e)

    if not loaded:
        logger.error("No models loaded. Exiting.")
        sys.exit(1)

    # Run smoke tests
    results = smoke_test_all(loaded, Path(args.fixtures))

    # Exit code: 0 if all pass, 1 if any fail
    if all(results.values()):
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
