# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""The tessiq device: allocation, transfers, and the op fallback."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

import tessiq

DEVICE = tessiq.DEVICE_NAME

DTYPES = [torch.float32, torch.float64, torch.int32, torch.int64]


def test_backend_is_registered_under_our_name() -> None:
    assert torch.empty(1, device=DEVICE).device.type == DEVICE


def test_device_is_reported() -> None:
    assert tessiq.device_count() >= 1
    assert tessiq.backend_name() in {"cpu", "cuda"}


def test_empty_allocates_the_requested_shape() -> None:
    x = torch.empty(3, 4, device=DEVICE)
    assert tuple(x.shape) == (3, 4)
    assert x.dtype == torch.float32


@pytest.mark.parametrize("dtype", DTYPES)
def test_round_trip_preserves_values(dtype: torch.dtype) -> None:
    source = torch.arange(16, dtype=dtype)
    assert torch.equal(source.to(DEVICE).cpu(), source)


def test_round_trip_preserves_shape() -> None:
    source = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    assert torch.equal(source.to(DEVICE).cpu(), source)


def test_empty_tensor_round_trips() -> None:
    # A zero-element tensor still has an identity, and the allocator has to give it an address.
    source = torch.empty(0, dtype=torch.float32)
    assert tuple(source.to(DEVICE).cpu().shape) == (0,)


def test_dtype_conversion_on_device() -> None:
    source = torch.arange(8, dtype=torch.float32)
    converted = source.to(DEVICE).to(torch.int32)
    assert converted.dtype == torch.int32
    assert torch.equal(converted.cpu(), source.to(torch.int32))


def test_device_to_device_copy() -> None:
    source = torch.arange(8, dtype=torch.float32).to(DEVICE)
    destination = torch.empty(8, dtype=torch.float32, device=DEVICE)
    destination.copy_(source)
    assert torch.equal(destination.cpu(), torch.arange(8, dtype=torch.float32))


def test_non_contiguous_source_round_trips() -> None:
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    assert torch.equal(source.to(DEVICE).cpu(), source)


def test_copy_rejects_mismatched_shapes() -> None:
    source = torch.arange(8, dtype=torch.float32).to(DEVICE)
    destination = torch.empty(4, dtype=torch.float32, device=DEVICE)
    with pytest.raises(RuntimeError, match="matching shapes"):
        torch.ops.aten._copy_from(source, destination, False)


# --- ops, which reach the host through the fallback until the kernel library exists ------------


@pytest.mark.parametrize(
    ("name", "op"),
    [
        ("add", lambda a, b: a + b),
        ("mul", lambda a, b: a * b),
        ("sub", lambda a, b: a - b),
        ("matmul", lambda a, b: a @ b),
        ("maximum", torch.maximum),
    ],
)
def test_binary_op_matches_cpu(name: str, op: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(0)
    a = torch.randn(8, 8)
    b = torch.randn(8, 8)
    expected = op(a, b)
    actual = op(a.to(DEVICE), b.to(DEVICE)).cpu()
    assert torch.allclose(actual, expected, atol=1e-6), name


@pytest.mark.parametrize("op", [torch.relu, torch.sigmoid, torch.exp, torch.neg])
def test_unary_op_matches_cpu(op: Callable[[torch.Tensor], torch.Tensor]) -> None:
    torch.manual_seed(0)
    x = torch.randn(64)
    assert torch.allclose(op(x.to(DEVICE)).cpu(), op(x), atol=1e-6)


def test_reduction_matches_cpu() -> None:
    torch.manual_seed(0)
    x = torch.randn(128)
    assert torch.allclose(x.to(DEVICE).sum().cpu(), x.sum(), atol=1e-4)


def test_in_place_op_writes_through() -> None:
    x = torch.empty(4, device=DEVICE).fill_(2.0)
    assert torch.equal(x.cpu(), torch.full((4,), 2.0))


def test_resize_grows_and_preserves_prefix() -> None:
    x = torch.arange(4, dtype=torch.float32).to(DEVICE)
    x.resize_(8)
    assert tuple(x.shape) == (8,)
    assert torch.equal(x.cpu()[:4], torch.arange(4, dtype=torch.float32))


def test_random_generation_is_seeded_by_torch_manual_seed() -> None:
    torch.manual_seed(7)
    first = torch.randn(16, device=DEVICE).cpu()
    torch.manual_seed(7)
    second = torch.randn(16, device=DEVICE).cpu()
    assert torch.equal(first, second)


# --- memory ------------------------------------------------------------------------------------


def test_memory_stats_track_a_live_tensor() -> None:
    before = tessiq.memory_allocated()
    x = torch.empty(4096, dtype=torch.float32, device=DEVICE)
    during = tessiq.memory_allocated()
    assert during > before

    del x
    assert tessiq.memory_allocated() == before


def test_reserved_exceeds_allocated_once_a_block_is_cached() -> None:
    x = torch.empty(4096, dtype=torch.float32, device=DEVICE)
    del x
    stats = tessiq.memory_stats()
    assert stats.reserved_bytes >= stats.allocated_bytes


def test_repeated_allocation_hits_the_cache() -> None:
    # Warm the cache, then allocate the same size again and check it was reused rather than
    # fetched from the device.
    shape = (777,)
    del_me = torch.empty(shape, dtype=torch.float32, device=DEVICE)
    del del_me

    before = tessiq.memory_stats().cache_hits
    again = torch.empty(shape, dtype=torch.float32, device=DEVICE)
    assert tessiq.memory_stats().cache_hits > before
    del again


def test_empty_cache_releases_reserved_memory() -> None:
    x = torch.empty(1 << 16, dtype=torch.float32, device=DEVICE)
    del x
    released = tessiq.empty_cache()
    assert released > 0
    assert tessiq.memory_stats().reserved_bytes == tessiq.memory_stats().allocated_bytes


def test_live_blocks_balance_out() -> None:
    before = tessiq.memory_stats().live_blocks
    tensors = [torch.empty(128, device=DEVICE) for _ in range(8)]
    assert tessiq.memory_stats().live_blocks == before + 8
    del tensors
    assert tessiq.memory_stats().live_blocks == before
