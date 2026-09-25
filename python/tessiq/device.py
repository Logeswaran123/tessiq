# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Device registration and the memory API.

Importing this module registers ``tessiq`` with PyTorch as a device. After that,
``torch.randn(4, device="tessiq")`` and ``tensor.to("tessiq")`` work like any other backend.
"""

from __future__ import annotations

import types
from dataclasses import dataclass

import torch

from tessiq import _C

#: The name PrivateUse1 is renamed to.
DEVICE_NAME = "tessiq"

#: The CPU reference backend, always present. An unqualified ``device="tessiq"`` lands here
#: unless :func:`set_device` names another.
#:
#: Assigned by :func:`_register`, because ``torch.device("tessiq", 0)`` only exists once the
#: backend has been renamed.
cpu: torch.device

#: The first GPU-backed device. Valid when :func:`gpu_count` is non-zero; using it on a machine
#: without a GPU fails at allocation with a message saying so.
gpu: torch.device

_registered = False


def _is_available() -> bool:
    return device_count() > 0


def _is_in_bad_fork() -> bool:
    return False


def _manual_seed(_seed: int) -> None:
    """Accept and ignore.

    Random ops run on the host through the fallback and draw from PyTorch's default generator,
    which ``torch.manual_seed`` has already seeded. It also calls every device module's
    ``manual_seed_all``, so delegating back to it would recurse.
    """


def _get_rng_state(_device: object = None) -> torch.Tensor:
    return torch.default_generator.get_state()


def _set_rng_state(state: torch.Tensor, _device: object = None) -> None:
    torch.default_generator.set_state(state)


def _initial_seed() -> int:
    return torch.default_generator.initial_seed()


def _build_torch_device_module() -> types.ModuleType:
    """Build the module PyTorch expects to find at ``torch.<DEVICE_NAME>``.

    PyTorch looks up ``torch.tessiq`` for device counts, the current device and RNG state whenever
    a call names the device.
    """
    module = types.ModuleType(f"torch.{DEVICE_NAME}")
    members = {
        "is_available": _is_available,
        "device_count": device_count,
        "current_device": current_device,
        "set_device": set_device,
        "synchronize": synchronize,
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

    # Public API missing from torch's type stubs; tests/python/test_device.py covers the calls.
    torch.utils.rename_privateuse1_backend(DEVICE_NAME)  # type: ignore[attr-defined]
    torch._register_device_module(DEVICE_NAME, _build_torch_device_module())  # type: ignore[no-untyped-call]
    torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)  # type: ignore[attr-defined]
    _C._init_backend()

    global cpu, gpu
    cpu = torch.device(DEVICE_NAME, 0)
    gpu = torch.device(DEVICE_NAME, 1)
    _registered = True


def gpu_at(ordinal: int) -> torch.device:
    """The GPU-backed device at `ordinal`, counting from zero.

    PyTorch exposes one device *type* to out-of-tree backends, so CPU and GPU are told apart by
    index: 0 is the reference, 1.. are GPUs. ``tessiq.gpu`` is ``gpu_at(0)``.
    """
    if ordinal < 0:
        raise ValueError(f"gpu ordinal must not be negative, got {ordinal}")
    return torch.device(DEVICE_NAME, ordinal + 1)


@dataclass(frozen=True)
class MemoryStats:
    """Allocator counters for one device, at a point in time.

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
    """Number of tessiq devices: the CPU reference, plus every usable GPU."""
    return int(_C.device_count())


def gpu_count() -> int:
    """Number of GPU-backed tessiq devices. Zero on a machine without one."""
    return device_count() - 1


def _resolve(device: torch.device | str | int | None) -> int:
    """Turn anything that names a tessiq device into its index. No index means the current one."""
    if device is None:
        return current_device()
    if isinstance(device, int):
        return device
    resolved = torch.device(device)
    if resolved.type != DEVICE_NAME:
        raise ValueError(f"expected a {DEVICE_NAME} device, got {resolved}")
    return current_device() if resolved.index is None else resolved.index


def current_device() -> int:
    """The index of the device this thread uses when none is named.

    0, the CPU reference, until :func:`set_device` or ``torch.accelerator.set_device_index``
    changes it. Each thread has its own, as with CUDA.
    """
    return int(_C.current_device())


def set_device(device: torch.device | str | int) -> None:
    """Make `device` the one this thread uses when none is named.

    ``torch.empty(4, device="tessiq")`` then allocates there, and :func:`synchronize` and the
    memory functions default to it.

    Raises:
        ValueError: If `device` names no tessiq device.
    """
    _C.set_current_device(_resolve(device))


def synchronize(device: torch.device | str | int | None = None) -> None:
    """Wait until all work queued on `device` has finished. Defaults to the current device.

    A launch returns as soon as it is queued, so this is what to call before stopping a clock.

    Raises:
        ValueError: If `device` names no tessiq device.
    """
    _C.synchronize(_resolve(device))


def backend_name(device: torch.device | str | int | None = None) -> str:
    """Backend behind a device: ``"cpu"`` for the reference, ``"cuda"`` for a GPU."""
    return str(_C.backend_name(_resolve(device)))


def cuda_ordinal(device: torch.device | str | int | None = None) -> int:
    """Which CUDA device is behind this one, or -1 when none is.

    Index 0 is the host reference, so only 1 and above map onto a GPU. -1 rather than an error, so
    callers can use one code path for both.
    """
    if not isinstance(device, torch.device):
        device = torch.device(device) if device is not None else torch.device(DEVICE_NAME)
    if device.type != DEVICE_NAME:
        return -1
    index = _resolve(device)
    if index < 1:
        return -1
    return index - 1 if backend_name(index) == "cuda" else -1


def memory_stats(device: torch.device | str | int | None = None) -> MemoryStats:
    """Allocator counters for one device. Each device has its own; they are never pooled."""
    raw = _C.memory_stats(_resolve(device))
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


def memory_allocated(device: torch.device | str | int | None = None) -> int:
    """Bytes currently held by live tensors on `device`."""
    return memory_stats(device).allocated_bytes


def memory_reserved(device: torch.device | str | int | None = None) -> int:
    """Bytes held by `device`'s allocator, including cached blocks."""
    return memory_stats(device).reserved_bytes


def empty_cache(device: torch.device | str | int | None = None) -> int:
    """Release cached blocks on `device`. Returns the bytes released.

    Live tensors are unaffected. Useful before handing the machine to another process.
    """
    return int(_C.empty_cache(_resolve(device)))


# Registering on import is what makes `cpu` and `gpu` above constructible.
_register()
