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

#: Every backend on this machine, so a test written once runs against all of them. The CPU
#: reference is always present; GPUs appear only where there is one.
BACKENDS = [
    pytest.param(tessiq.cpu, id="cpu"),
    *(
        pytest.param(tessiq.gpu_at(i), id=f"gpu{i}", marks=pytest.mark.cuda)
        for i in range(tessiq.gpu_count())
    ),
]


@pytest.fixture(params=BACKENDS)
def device(request: pytest.FixtureRequest) -> torch.device:
    """Each test using this runs once per backend."""
    device: torch.device = request.param
    return device


def test_backend_is_registered_under_our_name() -> None:
    assert torch.empty(1, device=DEVICE).device.type == DEVICE


def test_index_zero_is_the_cpu_reference() -> None:
    # An unqualified device="tessiq" must land on the backend that always exists.
    assert torch.empty(1, device=DEVICE).device.index == 0
    assert tessiq.backend_name(tessiq.cpu) == "cpu"


def test_device_count_covers_every_backend() -> None:
    assert tessiq.device_count() == 1 + tessiq.gpu_count()


def test_gpu_devices_report_the_cuda_backend() -> None:
    for i in range(tessiq.gpu_count()):
        assert tessiq.backend_name(tessiq.gpu_at(i)) == "cuda"


def test_gpu_at_rejects_a_negative_ordinal() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        tessiq.gpu_at(-1)


def test_unknown_index_is_rejected_with_a_helpful_message() -> None:
    with pytest.raises(ValueError, match="cpu reference"):
        tessiq.backend_name(tessiq.device_count())


def test_empty_allocates_the_requested_shape(device: torch.device) -> None:
    x = torch.empty(3, 4, device=device)
    assert tuple(x.shape) == (3, 4)
    assert x.dtype == torch.float32
    assert x.device == device, "a factory op must honour the device it was given"


@pytest.mark.parametrize("dtype", DTYPES)
def test_round_trip_preserves_values(device: torch.device, dtype: torch.dtype) -> None:
    source = torch.arange(16, dtype=dtype)
    assert torch.equal(source.to(device).cpu(), source)


def test_round_trip_preserves_shape(device: torch.device) -> None:
    source = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    assert torch.equal(source.to(device).cpu(), source)


def test_empty_tensor_round_trips(device: torch.device) -> None:
    # A zero-element tensor still has an identity, and the allocator has to give it an address.
    source = torch.empty(0, dtype=torch.float32)
    assert tuple(source.to(device).cpu().shape) == (0,)


def test_dtype_conversion_on_device(device: torch.device) -> None:
    source = torch.arange(8, dtype=torch.float32)
    converted = source.to(device).to(torch.int32)
    assert converted.dtype == torch.int32
    assert torch.equal(converted.cpu(), source.to(torch.int32))


def test_device_to_device_copy(device: torch.device) -> None:
    source = torch.arange(8, dtype=torch.float32).to(device)
    destination = torch.empty(8, dtype=torch.float32, device=device)
    destination.copy_(source)
    assert torch.equal(destination.cpu(), torch.arange(8, dtype=torch.float32))


def test_non_contiguous_source_round_trips(device: torch.device) -> None:
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    assert torch.equal(source.to(device).cpu(), source)


def test_copy_rejects_mismatched_shapes(device: torch.device) -> None:
    source = torch.arange(8, dtype=torch.float32).to(device)
    destination = torch.empty(4, dtype=torch.float32, device=device)
    with pytest.raises(RuntimeError, match="matching shapes"):
        torch.ops.aten._copy_from(source, destination, False)


@pytest.mark.cuda
def test_copy_between_backends_stages_through_the_host() -> None:
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")

    source = torch.arange(32, dtype=torch.float32).to(tessiq.cpu)
    moved = source.to(tessiq.gpu)
    assert moved.device == tessiq.gpu
    assert torch.equal(moved.to(tessiq.cpu).cpu(), source.cpu())


@pytest.mark.cuda
def test_the_two_backends_agree() -> None:
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")

    torch.manual_seed(0)
    host = torch.randn(256)
    on_cpu = host.to(tessiq.cpu)
    on_gpu = host.to(tessiq.gpu)
    assert torch.allclose(on_cpu.relu().sum().cpu(), on_gpu.relu().sum().cpu(), atol=1e-4)


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
