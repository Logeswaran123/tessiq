# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""tessiq: a from-scratch PyTorch accelerator backend.

Importing it registers the ``tessiq`` device and the custom operations. Compile a model with
``torch.compile(model, backend="tessiq")``; the functions here cover devices, memory, graph
inspection and profiling.
"""

from __future__ import annotations

# Before the extension module, which links libtorch. Its RPATH only covers layouts where torch
# is installed next to tessiq.
import torch as _torch  # noqa: F401

from tessiq import _C, library
from tessiq.cache import CacheStats, cache_stats, clear_cache
from tessiq.costs import KernelCost, kernel_costs
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
from tessiq.explain import explain, explain_graph
from tessiq.fx import (
    capture,
    default_pipeline,
    emit_cuda,
    import_graph,
    partition,
    pass_names,
    run_passes,
)
from tessiq.profiler import Profile, profile
from tessiq.recipe import Recipe, build_recipe

__all__ = [
    "DEVICE_NAME",
    "CacheStats",
    "KernelCost",
    "MemoryStats",
    "Profile",
    "Recipe",
    "__version__",
    "backend_name",
    "build_recipe",
    "cache_stats",
    "capture",
    "clear_cache",
    "cpu",
    "default_pipeline",
    "device_count",
    "emit_cuda",
    "empty_cache",
    "execute",
    "explain",
    "explain_graph",
    "get_log_level",
    "gpu",
    "gpu_at",
    "gpu_count",
    "import_graph",
    "kernel_costs",
    "library",
    "memory_allocated",
    "memory_reserved",
    "memory_stats",
    "partition",
    "pass_names",
    "profile",
    "run_passes",
    "set_log_level",
]

#: The package version, read from the native module.
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
