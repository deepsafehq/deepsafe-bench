"""Namespaced model loading for the monolith server.

Each model's code lives under ``models/<modality>/<model>/code/`` (e.g. ``models/image/npr/code/``).
These directories often use generic top-level module names like ``models``, ``data``, ``utils``
that would collide if two models were imported in the same process.

This module provides ``namespaced_import`` which imports model code under a unique
prefix (e.g. ``_ds_universal.models`` instead of bare ``models``), preventing collisions.
All models can be hot-loaded together in one process.
"""

import importlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional  # noqa: F401 — safe_import uses Any

logger = logging.getLogger("loader")


def namespaced_import(
    module_name: str,
    code_path: Path,
    namespace: str,
    extra_paths: Optional[List[Path]] = None,
) -> Any:
    """Import a module under a unique namespace to avoid collisions.

    Instead of importing ``models.AIDE`` (which collides with Universal's ``models``),
    this imports it as ``_ds_aide.models.AIDE`` — invisible to the caller but
    unique in sys.modules.

    Args:
        module_name: The module to import as if code_path were on sys.path
                     (e.g. ``models.AIDE``, ``networks.resnet``, ``fsd``).
        code_path:   Root directory containing the module.
        namespace:   Unique prefix for this model (e.g. ``"_ds_aide"``).
        extra_paths: Additional directories to add to sys.path temporarily.

    Returns:
        The imported module object.
    """
    # Build the namespaced module name
    ns_module_name = f"{namespace}.{module_name}"

    # Return cached if already imported under this namespace
    if ns_module_name in sys.modules:
        return sys.modules[ns_module_name]

    # Clean conflicting generic modules (e.g. "models" from another model)
    # so the new import resolves from code_path, not a stale cache.
    top_level = module_name.split(".")[0]
    if top_level in _CONFLICTING_PREFIXES:
        _clean_conflicting_modules_for(top_level)

    # Temporarily add code_path (and extras) to sys.path
    paths_to_add = [str(code_path)]
    if extra_paths:
        paths_to_add.extend(str(p) for p in extra_paths)

    saved_path = sys.path.copy()
    for p in reversed(paths_to_add):
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        # Import the actual module under its original name
        mod = importlib.import_module(module_name)

        # Register it under the namespaced name too (for our tracking)
        sys.modules[ns_module_name] = mod

        logger.debug(
            "Imported %s as %s from %s", module_name, ns_module_name, code_path
        )
        return mod
    finally:
        sys.path[:] = saved_path


def namespaced_import_multi(
    code_path: Path,
    namespace: str,
    import_specs: List[Dict],
    extra_paths: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    """Import multiple modules from a model_code directory under one namespace.

    Cleans conflicting generic module names (models, utils, data, etc.) from
    sys.modules BEFORE importing, so the new imports resolve from code_path.
    After import, the loaded modules stay in sys.modules under both their
    original and namespaced names.

    Args:
        code_path:    Root of the model_code directory.
        namespace:    Unique prefix (e.g. ``"_ds_aide"``).
        import_specs: List of dicts with keys:
            - ``module``: module name to import (e.g. ``models.AIDE``)
            - ``attr``:   optional attribute to extract (e.g. ``AIDE``)
        extra_paths:  Additional sys.path entries.

    Returns:
        Dict mapping spec ``attr`` (or ``module`` if no attr) to the imported object.
    """
    # Clean conflicting modules so imports resolve from code_path
    _clean_conflicting_modules()

    paths_to_add = [str(code_path)]
    if extra_paths:
        paths_to_add.extend(str(p) for p in extra_paths)

    saved_path = sys.path.copy()
    for p in reversed(paths_to_add):
        if p not in sys.path:
            sys.path.insert(0, p)

    results = {}
    try:
        for spec in import_specs:
            mod_name = spec["module"]
            attr_name = spec.get("attr")
            mod = importlib.import_module(mod_name)

            # Register under namespace
            ns_name = f"{namespace}.{mod_name}"
            sys.modules[ns_name] = mod

            if attr_name:
                results[attr_name] = getattr(mod, attr_name)
            else:
                results[mod_name] = mod
        return results
    finally:
        sys.path[:] = saved_path


# ── Conflicting module cleanup ──────────────────────────────────────────────

_CONFLICTING_PREFIXES = frozenset(
    [
        "models",
        "model",
        "data",
        "dataset",
        "datasets",
        "utils",
        "configs",
        "config",
        "networks",
        "network",
        "modules",
        "layers",
        "transforms",
        "package_utils",
        "detectors",
        "src",
        "test_tools",
        "code",
    ]
)


def _clean_conflicting_modules() -> None:
    """Remove ALL generic module names from sys.modules."""
    to_remove = [
        key for key in sys.modules if key.split(".")[0] in _CONFLICTING_PREFIXES
    ]
    for key in to_remove:
        del sys.modules[key]


def _clean_conflicting_modules_for(prefix: str) -> None:
    """Remove a specific generic module prefix from sys.modules.

    E.g. _clean_conflicting_modules_for("models") removes "models",
    "models.xception", "models.efficientnet", etc.
    """
    to_remove = [
        key for key in sys.modules if key == prefix or key.startswith(prefix + ".")
    ]
    for key in to_remove:
        del sys.modules[key]


# ── Legacy API (kept for backward compat) ───────────────────────────────────


def safe_import(module_name: str, code_path: Path) -> Any:
    """Non-isolated import: add code_path to sys.path and import.

    DEPRECATED: Use namespaced_import() for new code.
    """
    path_str = str(code_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    return importlib.import_module(module_name)
