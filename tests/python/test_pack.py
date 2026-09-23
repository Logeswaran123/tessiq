# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Packing a view on the device instead of on the host.

`clone` is the one operation whose fallback is pure waste. Everything else reaches the host
because there is no kernel for the arithmetic; a clone has no arithmetic. It reads a view and
writes it packed, which the device can do to itself, and sending it across costs the whole tensor
each way to produce a rearrangement of bytes that never had to leave.

`.contiguous()` goes through the same operation, so these cover both.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch

import tessiq

pytestmark = pytest.mark.cuda


@pytest.fixture(autouse=True)
def _needs_a_gpu() -> None:
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")


@pytest.fixture
def _debug_logs() -> Iterator[None]:
    saved = tessiq.get_log_level()
    tessiq.set_log_level("debug")
    yield
    tessiq.set_log_level(saved)


def views(device: str) -> list[tuple[str, torch.Tensor]]:
    """One of each way a tensor ends up not packed."""
    torch.manual_seed(0)
    base = torch.randn(4, 6, 8, device=device)
    return [
        ("already packed", base),
        ("permuted", base.permute(2, 0, 1)),
        ("transposed", base.transpose(0, 1)),
        ("sliced", base[:, 1:5, ::2]),
        ("starts partway in", base[2:]),
        ("expanded", base[:, :1, :].expand(4, 6, 8)),
        ("rank one", base.reshape(-1)[3::5]),
        ("a scalar", base[1, 2, 3]),
    ]


@pytest.mark.parametrize("name", [name for name, _ in views("cpu")])
def test_contiguous_matches_the_host(name: str) -> None:
    device = str(tessiq.gpu)
    ours = dict(views(device))[name]
    theirs = dict(views("cpu"))[name]

    packed = ours.contiguous()
    assert packed.is_contiguous()
    assert packed.device.type == tessiq.DEVICE_NAME, "packing should not move it off the device"
    torch.testing.assert_close(packed.cpu(), theirs.contiguous())


def test_clone_matches_the_host() -> None:
    device = str(tessiq.gpu)
    torch.manual_seed(0)
    base = torch.randn(3, 5, 7, device=device)
    view = base.permute(1, 2, 0)

    cloned = torch.clone(view, memory_format=torch.contiguous_format)
    assert cloned.is_contiguous()
    torch.testing.assert_close(cloned.cpu(), view.cpu().contiguous())

    # A clone is a copy, not a view. Writing through one must not reach the other.
    cloned.fill_(0.0)
    assert view.cpu().abs().sum() > 0


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int64, torch.bool]
)
def test_every_width_packs(dtype: torch.dtype) -> None:
    # The kernel is written for an element width rather than an element type, because a copy does
    # no arithmetic. That is what covers the dtypes the expression emitter has no spelling for.
    device = str(tessiq.gpu)
    source = torch.ones(3, 4, 5, device=device, dtype=torch.float32)
    base = source.to(dtype) if dtype != torch.bool else source.to(torch.bool)
    view = base.permute(2, 1, 0)

    packed = view.contiguous()
    assert packed.dtype == dtype
    assert packed.device.type == tessiq.DEVICE_NAME
    torch.testing.assert_close(packed.cpu(), view.cpu().contiguous())


def test_an_empty_tensor_has_nothing_to_pack() -> None:
    device = str(tessiq.gpu)
    empty = torch.randn(0, 4, device=device).transpose(0, 1)
    packed = empty.contiguous()
    assert packed.shape == (4, 0)
    assert packed.numel() == 0


@pytest.mark.usefixtures("_debug_logs")
def test_the_packing_stays_on_the_device(capfd: pytest.CaptureFixture[str]) -> None:
    # The whole point, and the only way to check it that does not measure a clock. The fallback
    # announces itself, so its silence is the evidence.
    device = str(tessiq.gpu)
    view = torch.randn(8, 16, 32, device=device).permute(1, 0, 2)
    capfd.readouterr()

    view.contiguous()

    logged = capfd.readouterr()
    assert "cpu fallback for aten::clone" not in logged.err + logged.out


@pytest.mark.usefixtures("_debug_logs")
def test_a_shape_too_deep_falls_back(capfd: pytest.CaptureFixture[str]) -> None:
    # Kernels exist up to eight axes. A tensor deeper than that has to still work, by the route
    # everything took before this existed.
    device = str(tessiq.gpu)
    deep = torch.randn(2, 1, 1, 1, 1, 1, 1, 1, 3, device=device).transpose(0, 8)
    capfd.readouterr()

    packed = deep.contiguous()

    assert packed.is_contiguous()
    torch.testing.assert_close(packed.cpu(), deep.cpu().contiguous())
    logged = capfd.readouterr()
    assert "cpu fallback for aten::clone" in logged.err + logged.out


def test_the_host_reference_device_still_works() -> None:
    # tessiq.cpu has no CUDA behind it, so there is no kernel to launch and the fallback is the
    # only route. It has to stay correct.
    view = torch.randn(3, 4, device=str(tessiq.cpu)).transpose(0, 1)
    packed = view.contiguous()
    assert packed.is_contiguous()
    torch.testing.assert_close(packed.cpu(), view.cpu().contiguous())
