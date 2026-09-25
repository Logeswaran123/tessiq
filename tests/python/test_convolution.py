# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Convolution through cuDNN, and the paths that have to stay correct without it.

cuDNN reads every tensor the way its descriptors say, so a misread stride or padding is a wrong
answer rather than an error. Each case is compared against eager on the host.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import tessiq
from tessiq import _C
from test_models import ResidualBlock

# Tight, because the plan uses fp32 math rather than TF32 so the answer agrees with the host.
RTOL = 1e-5
ATOL = 1e-5


@pytest.fixture(autouse=True)
def _fresh_dynamo() -> None:
    torch._dynamo.reset()


def _gpu() -> None:
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")


CASES: list[tuple[str, dict[str, object]]] = [
    (
        "a resnet convolution",
        {"in_channels": 16, "out_channels": 16, "kernel_size": 3, "padding": 1},
    ),
    (
        "stride two",
        {"in_channels": 8, "out_channels": 16, "kernel_size": 3, "stride": 2, "padding": 1},
    ),
    (
        "dilated",
        {"in_channels": 8, "out_channels": 8, "kernel_size": 3, "dilation": 2, "padding": 2},
    ),
    (
        "depthwise",
        {"in_channels": 8, "out_channels": 8, "kernel_size": 3, "padding": 1, "groups": 8},
    ),
    ("one by one", {"in_channels": 16, "out_channels": 32, "kernel_size": 1}),
    (
        "uneven axes",
        {"in_channels": 4, "out_channels": 8, "kernel_size": (3, 5), "stride": (2, 1)},
    ),
]


@pytest.mark.cuda
@pytest.mark.parametrize("bias", [False, True], ids=["no bias", "bias"])
@pytest.mark.parametrize("config", [config for _, config in CASES], ids=[name for name, _ in CASES])
def test_a_convolution_matches_eager(config: dict[str, object], bias: bool) -> None:
    _gpu()
    torch.manual_seed(0)
    layer = nn.Conv2d(bias=bias, **config).eval()  # type: ignore[arg-type]
    example = torch.randn(2, int(config["in_channels"]), 16, 12)  # type: ignore[call-overload]

    compiled = torch.compile(layer, backend="tessiq", options={"device": str(tessiq.gpu)})
    with torch.no_grad():
        torch.testing.assert_close(compiled(example), layer(example), rtol=RTOL, atol=ATOL)


@pytest.mark.cuda
def test_a_float64_convolution_runs_on_the_host_and_matches_eager() -> None:
    # cuDNN is float32 only here, so a float64 convolution is left on the host.
    _gpu()
    torch.manual_seed(0)
    layer = nn.Conv2d(4, 8, kernel_size=3, padding=1).double().eval()
    example = torch.randn(2, 4, 8, 8, dtype=torch.float64)
    compiled = torch.compile(layer, backend="tessiq", options={"device": str(tessiq.gpu)})
    with torch.no_grad():
        torch.testing.assert_close(compiled(example), layer(example))


@pytest.mark.cuda
def test_a_convolution_is_planned_rather_than_interpreted() -> None:
    # Matching eager proves nothing on its own, since the fallback matches too. This checks that
    # cuDNN ran it.
    _gpu()
    torch.manual_seed(0)
    graph = tessiq.capture(ResidualBlock().eval(), torch.randn(1, 16, 8, 8))
    recipe = tessiq.build_recipe(graph, tessiq.gpu, use_cache=False)
    convs = [k for k in recipe.kernels.values() if isinstance(k, _C.ConvKernel)]
    assert len(convs) == 2, "a residual block has two convolutions"
    assert all(k.algorithm >= 0 for k in convs)


@pytest.mark.cuda
def test_the_residual_block_runs_on_the_device() -> None:
    # Everything but the four float32-to-float32 casts the batch norm decomposition inserts on its
    # running statistics.
    _gpu()
    torch.manual_seed(0)
    block = ResidualBlock().eval()
    graph = tessiq.capture(block, torch.randn(1, 16, 8, 8))
    recipe = tessiq.build_recipe(graph, tessiq.gpu, use_cache=False)

    on_host = [
        str(node.target)
        for node in graph.graph.nodes
        if node.op == "call_function"
        and recipe.devices.get(node.name, torch.device("cpu")).type != tessiq.DEVICE_NAME
        and "getitem" not in str(node.target)
    ]
    assert all("_to_copy" in op for op in on_host), f"still on the host: {on_host}"


def test_a_convolution_on_the_reference_device_is_still_correct() -> None:
    # The reference device has no cuDNN. PyTorch routes a convolution on an unfamiliar device to
    # convolution_overrideable, which the generic fallback cannot run, so tessiq registers its own.
    torch.manual_seed(0)
    block = ResidualBlock().eval()
    example = torch.randn(1, 16, 8, 8)
    compiled = torch.compile(block, backend="tessiq", options={"device": str(tessiq.cpu)})
    with torch.no_grad():
        torch.testing.assert_close(compiled(example), block(example), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_a_convolution_on_a_tessiq_tensor_works_eagerly(device: str) -> None:
    # Eager, outside the compiler, through the same registration.
    if device == "gpu":
        _gpu()
    target = str(tessiq.cpu if device == "cpu" else tessiq.gpu)
    torch.manual_seed(0)
    image = torch.randn(1, 3, 8, 8)
    weight = torch.randn(4, 3, 3, 3)

    ours = torch.nn.functional.conv2d(image.to(target), weight.to(target), padding=1)
    assert ours.device.type == tessiq.DEVICE_NAME
    torch.testing.assert_close(
        ours.cpu(), torch.nn.functional.conv2d(image, weight, padding=1), rtol=RTOL, atol=ATOL
    )
