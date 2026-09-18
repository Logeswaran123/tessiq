# Copyright 2026 The Tessera Authors.
# SPDX-License-Identifier: Apache-2.0
"""Device registration and the memory API.

Importing this module registers ``tessera`` with PyTorch as a device. After that,
``torch.randn(4, device="tessera")`` and ``tensor.to("tessera")`` work like any other backend.
"""

from __future__ import annotations

import types
from dataclasses import dataclass

import torch

from tessera import _C

#: The device name PyTorch knows us by. PyTorch's reserved identifier for out-of-tree backends is
#: renamed to this once, here; nothing else in the project refers to it.
DEVICE_NAME = "tessera"

_registered = False


def _is_available() -> bool:
    return device_count() > 0


def _current_device() -> int:
    return 0


def _set_device(_device: object) -> None:
    """Accept and ignore: tessera exposes a single device, so there is nothing to switch to."""


def _synchronize(_device: object = None) -> None:
    """Accept and ignore: the CPU backend runs everything inline."""


def _is_in_bad_fork() -> bool:
    return False


def _manual_seed(_seed: int) -> None:
    """Accept and ignore.

    Random ops reach the host through the CPU fallback and draw from PyTorch's default generator,
    so there is no separate device RNG state to seed. ``torch.manual_seed`` has already seeded the
    generator these ops use, and it calls every device module's ``manual_seed_all``, so delegating
    back to it recurses until the stack runs out.
    """


def _get_rng_state(_device: object = None) -> torch.Tensor:
    return torch.default_generator.get_state()


def _set_rng_state(state: torch.Tensor, _device: object = None) -> None:
    torch.default_generator.set_state(state)


def _initial_seed() -> int:
    return torch.default_generator.initial_seed()


def _build_torch_device_module() -> types.ModuleType:
    """Build the module PyTorch expects to find at ``torch.<DEVICE_NAME>``.

    Resolving ``device="tessera"`` makes PyTorch look up a module of that name on ``torch`` and
    call into it for device counts, the current device and RNG state. Without it, every call that
    names the device fails with a ModuleNotFoundError that says nothing about the real cause.
    """
    module = types.ModuleType(f"torch.{DEVICE_NAME}")
    members = {
        "is_available": _is_available,
        "device_count": device_count,
        "current_device": _current_device,
        "set_device": _set_device,
        "synchronize": _synchronize,
        "empty_cache": empty_cache,
        "_is_in_bad_fork": _is_in_bad_fork,
        "manual_seed": _manual_seed,
        "manual_seed_all": _manual_seed,
        "get_rng_state": _get_rng_state,
        "set_rng_state": _set_rng_state,
        "initial_seed": _initial_seed,
    }
    for name, value in members.items():
        setattr(module, name, value)
    return module


def _register() -> None:
    """Register the backend with PyTorch. Idempotent."""
    global _registered
    if _registered:
        return

    # These three are public API but are not re-exported in torch's type stubs, so mypy cannot
    # see them. The calls are checked by tests/python/test_device.py instead.
    torch.utils.rename_privateuse1_backend(DEVICE_NAME)  # type: ignore[attr-defined]
    torch._register_device_module(DEVICE_NAME, _build_torch_device_module())  # type: ignore[no-untyped-call]
    torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)  # type: ignore[attr-defined]
    _C._init_backend()
    _registered = True


@dataclass(frozen=True)
class MemoryStats:
    """Allocator counters at a point in time.

    ``allocated`` is what callers hold. ``reserved`` includes blocks the allocator is caching for
    reuse, so it is always at least ``allocated`` and only shrinks on :func:`empty_cache`.
    """

    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    allocation_count: int
    deallocation_count: int
    cache_hits: int
    cache_misses: int

    @property
    def live_blocks(self) -> int:
        """Blocks handed out and not yet returned. Non-zero at exit means a leak."""
        return self.allocation_count - self.deallocation_count


def device_count() -> int:
    """Number of tessera devices. Zero means the active backend found no hardware."""
    return int(_C.device_count())


def backend_name() -> str:
    """Backend currently executing tessera tensors, such as ``"cpu"`` or ``"cuda"``."""
    return str(_C.backend_name())


def memory_stats() -> MemoryStats:
    """Return the allocator counters."""
    raw = _C.memory_stats()
    return MemoryStats(
        allocated_bytes=raw.allocated_bytes,
        reserved_bytes=raw.reserved_bytes,
        peak_allocated_bytes=raw.peak_allocated_bytes,
        peak_reserved_bytes=raw.peak_reserved_bytes,
        allocation_count=raw.allocation_count,
        deallocation_count=raw.deallocation_count,
        cache_hits=raw.cache_hits,
        cache_misses=raw.cache_misses,
    )


def memory_allocated() -> int:
    """Bytes currently held by live tensors."""
    return memory_stats().allocated_bytes


def memory_reserved() -> int:
    """Bytes held by the allocator, including cached blocks."""
    return memory_stats().reserved_bytes


def empty_cache() -> int:
    """Release cached blocks back to the device. Returns the bytes released.

    Live tensors are unaffected. Useful before handing the machine to another process.
    """
    return int(_C.empty_cache())
