"""Fairseq Python 3.12+ compatibility — runtime patch.

Fixes mutable dataclass defaults that Python 3.12+ rejects.
Import this module BEFORE `import fairseq`.

Note: This handles the dataclass issue (Patch 1). The hydra_init issue
(Patch 2) is fixed by monolith/patch_fairseq.py which modifies the
installed fairseq source files. Run it once after `pip install fairseq`.
"""

import copy
import dataclasses

_original_get_field = dataclasses._get_field


def _lenient_get_field(cls, a_name, a_type, kw_only):
    """Auto-convert mutable defaults to default_factory on ValueError."""
    try:
        return _original_get_field(cls, a_name, a_type, kw_only)
    except ValueError as e:
        if "mutable default" not in str(e):
            raise
        val = cls.__dict__.get(a_name, dataclasses.MISSING)
        if isinstance(val, dataclasses.Field):
            if val.default is not dataclasses.MISSING:
                captured = val.default
                new_field = dataclasses.field(
                    default_factory=lambda d=captured: copy.deepcopy(d),
                    init=val.init,
                    repr=val.repr,
                    hash=val.hash,
                    compare=val.compare,
                    metadata=val.metadata,
                    kw_only=val.kw_only,
                )
                setattr(cls, a_name, new_field)
                return _original_get_field(cls, a_name, a_type, kw_only)
        elif val is not dataclasses.MISSING:
            captured = val
            setattr(
                cls,
                a_name,
                dataclasses.field(
                    default_factory=lambda d=captured: copy.deepcopy(d),
                ),
            )
            return _original_get_field(cls, a_name, a_type, kw_only)
        raise


dataclasses._get_field = _lenient_get_field
