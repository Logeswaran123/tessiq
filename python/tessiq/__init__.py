# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""tessiq: a from-scratch PyTorch accelerator backend.

The public surface is intentionally small at this stage. See PHASES.md for
the phase plan; the device API, the ``torch.compile`` backend and the debug tooling arrive in
later phases.
"""

from __future__ import annotations

# Imported before the extension module: _C links libtorch, and importing torch first puts those
# libraries in the process regardless of how this package was installed. The extension also
# carries an RPATH, but that only covers layouts where torch sits next to us.
import torch as _torch  # noqa: F401

from tessiq import _C
from tessiq.cache import CacheStats, cache_stats, clear_cache
from tessiq.device import (
    DEVICE_NAME,
    MemoryStats,
    backend_name,
    cpu,
    device_count,
    empty_cache,
    gpu,
    gpu_at,
    gpu_count,
    memory_allocated,
    memory_reserved,
    memory_stats,
)
from tessiq.executor import execute
from tessiq.fx import emit_cuda, import_graph, partition, pass_names, run_passes
from tessiq.recipe import Recipe, build_recipe

__all__ = [
    "DEVICE_NAME",
    "CacheStats",
    "MemoryStats",
    "Recipe",
    "__version__",
    "backend_name",
    "build_recipe",
    "cache_stats",
    "clear_cache",
    "cpu",
    "device_count",
    "emit_cuda",
    "empty_cache",
    "execute",
    "get_log_level",
    "gpu",
    "gpu_at",
    "gpu_count",
    "import_graph",
    "memory_allocated",
    "memory_reserved",
    "memory_stats",
    "partition",
    "pass_names",
    "run_passes",
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
