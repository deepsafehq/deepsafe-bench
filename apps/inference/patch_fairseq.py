#!/usr/bin/env python3
"""Patch installed fairseq for Python 3.12+ compatibility.

Run ONCE after installing fairseq:
    pip install fairseq  # or build from source
    python monolith/patch_fairseq.py

Patches:
1. dataclass/initialize.py — hydra_init() passes MISSING to OmegaConf
2. Clears .pyc cache so patches take effect immediately
"""

import importlib
import os
import shutil
import sys
from pathlib import Path


def find_fairseq_path() -> Path:
    """Find the installed fairseq package path."""
    # Search sys.path directly (works even when fairseq can't import)
    for p in sys.path:
        candidate = Path(p) / "fairseq"
        if (candidate / "__init__.py").exists():
            return candidate
    # Try importlib as fallback
    try:
        spec = importlib.util.find_spec("fairseq")
        if spec and spec.origin:
            return Path(spec.origin).parent
    except Exception:
        pass
    raise FileNotFoundError("fairseq not found. Install it first.")


INITIALIZE_PATCH = '''import dataclasses
import logging
from hydra.core.config_store import ConfigStore
from fairseq.dataclass.configs import FairseqConfig
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)


def hydra_init(cfg_name="config") -> None:

    cs = ConfigStore.instance()
    cs.store(name=f"{cfg_name}", node=FairseqConfig)

    for k in FairseqConfig.__dataclass_fields__:
        field_obj = FairseqConfig.__dataclass_fields__[k]
        v = field_obj.default
        if v is dataclasses.MISSING:
            if field_obj.default_factory is not dataclasses.MISSING:
                v = field_obj.default_factory()
            else:
                continue  # skip truly missing fields
        try:
            cs.store(name=k, node=v)
        except BaseException:
            logger.debug(f"Could not store {k} in ConfigStore (non-critical)")


def add_defaults(cfg: DictConfig) -> None:
    """This function adds default values that are stored in dataclasses that hydra doesn't know about"""

    from fairseq.registry import REGISTRIES
    from fairseq.tasks import TASK_DATACLASS_REGISTRY
    from fairseq.models import ARCH_MODEL_NAME_REGISTRY, MODEL_DATACLASS_REGISTRY
    from fairseq.dataclass.utils import merge_with_parent
    from typing import Any

    OmegaConf.set_struct(cfg, False)

    for k, v in FairseqConfig.__dataclass_fields__.items():
        field_cfg = cfg.get(k)
        if field_cfg is not None and v.type == Any:
            dc = None

            if isinstance(field_cfg, str):
                field_cfg = DictConfig({"_name": field_cfg})
                field_cfg.__dict__["_parent"] = field_cfg.__dict__["_parent"]

            name = getattr(field_cfg, "_name", None)

            if k == "task":
                dc = TASK_DATACLASS_REGISTRY.get(name)
            elif k == "model":
                name = ARCH_MODEL_NAME_REGISTRY.get(name, name)
                dc = MODEL_DATACLASS_REGISTRY.get(name)
            elif k in REGISTRIES:
                dc = REGISTRIES[k]["dataclass_registry"].get(name)

            if dc is not None:
                cfg[k] = merge_with_parent(dc, field_cfg)
'''


def patch():
    fairseq_path = find_fairseq_path()
    print(f"Patching fairseq at: {fairseq_path}")

    # Patch initialize.py
    init_file = fairseq_path / "dataclass" / "initialize.py"
    if init_file.exists():
        backup = init_file.with_suffix(".py.bak")
        if not backup.exists():
            shutil.copy2(init_file, backup)
            print(f"  Backed up: {backup}")
        init_file.write_text(INITIALIZE_PATCH)
        print(f"  Patched:   {init_file}")
    else:
        print(f"  WARNING: {init_file} not found")

    # Clear __pycache__
    for cache_dir in fairseq_path.rglob("__pycache__"):
        shutil.rmtree(cache_dir, ignore_errors=True)
    print("  Cleared __pycache__")

    # Verify
    print("\nVerifying patch...")
    # Force reimport
    for key in list(sys.modules.keys()):
        if key.startswith("fairseq"):
            del sys.modules[key]

    try:
        # Add repo root so monolith package is importable
        repo_root = str(Path(__file__).resolve().parents[1])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        # Need the dataclass patch too
        import fairseq
        import fairseq_compat  # noqa

        print("SUCCESS: fairseq imports cleanly on Python", sys.version.split()[0])
    except Exception as e:
        print(f"FAILED: {e}")
        print("You may need to also run: import fairseq_compat before import fairseq")
        return False

    return True


if __name__ == "__main__":
    success = patch()
    sys.exit(0 if success else 1)
