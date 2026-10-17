# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Kernel execution time from CUPTI's records, and sizes that stay clear of the L2 cache.

A host clock around a launch and a synchronise measures launch and synchronisation latency as well
as the kernel, and that overhead differs between a raw launch and a framework call. CUPTI records
each kernel's start and end on the device, so both sides of a comparison are timed the same way.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

# With a PrivateUse1 backend imported, the profiler labels CUDA kernels with that device type.
_DEVICE_TYPES = (DeviceType.CUDA, DeviceType.PrivateUse1)


def _kernel_records(work: Callable[[], object], runs: int) -> list[tuple[int, float]]:
    """(start, microseconds) of every kernel `runs` calls of `work` launch, in launch order."""
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as recorded:
        for _ in range(runs):
            work()
        torch.cuda.synchronize()
    return sorted(
        (event.time_range.start, event.time_range.elapsed_us())
        for event in recorded.events()
        if event.device_type in _DEVICE_TYPES and not event.name.startswith(("Memcpy", "Memset"))
    )


def kernel_microseconds(work: Callable[[], object], runs: int = 100, warmup: int = 20) -> float:
    """Median over `runs` calls of the summed execution time of the kernels one call launches."""
    for _ in range(warmup):
        work()
    torch.cuda.synchronize()

    # The profiler now and then returns a session without its device records; another one settles it.
    for _ in range(3):
        kernels = _kernel_records(work, runs)
        if kernels and len(kernels) % runs == 0:
            break
    else:
        raise RuntimeError(f"expected the same kernels on every call, recorded {len(kernels)}")

    per_call = len(kernels) // runs
    totals = [
        sum(duration for _, duration in kernels[i : i + per_call])
        for i in range(0, len(kernels), per_call)
    ]
    return statistics.median(totals)


def cache_and_dram_bytes() -> tuple[int, int]:
    """Working-set sizes for this GPU: half its L2 cache, and sixteen times it.

    The first is read from the cache, the second from DRAM. A working set close to the cache's own
    size is neither: how much of it stays cached depends on where its buffers land, and the same
    kernel can take twice as long from one run to the next.
    """
    l2 = torch.cuda.get_device_properties(torch.cuda.current_device()).L2_cache_size
    return l2 // 2, 16 * l2


def gigabytes_per_second(bytes_moved: int, microseconds: float) -> float:
    """Bandwidth in 10^9 bytes per second, the unit GPU specifications use."""
    return bytes_moved / (microseconds * 1e-6) / 1e9
