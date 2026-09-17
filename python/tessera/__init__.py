# Copyright 2026 The Tessera Authors.
# SPDX-License-Identifier: Apache-2.0
"""tessera: a from-scratch PyTorch accelerator backend.

The public surface is intentionally small at this stage. See PLAN.md for the architecture and
the phase plan; the device API, the ``torch.compile`` backend and the debug tooling arrive in
later phases.
"""

from __future__ import annotations

from tessera import _C

__all__ = [
    "__version__",
    "get_log_level",
    "set_log_level",
]

#: The package version. Sourced from the native module so that the two cannot drift apart;
#: ``tests/python/test_version.py`` asserts they agree.
__version__: str = _C.version()


def set_log_level(level: str) -> None:
    """Set the native log level.

    Args:
        level: One of ``"error"``, ``"warn"``, ``"info"``, ``"debug"``, ``"trace"``.

    Raises:
        ValueError: If ``level`` is not one of the recognised names.
    """
    _C.set_log_level(level)


def get_log_level() -> str:
    """Return the current native log level as a lowercase string."""
    return _C.get_log_level()
