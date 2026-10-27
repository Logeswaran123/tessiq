# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Blocks from real architectures, checked against eager.

`test_end_to_end.py` uses small models. These use a transformer encoder block and a ResNet
residual block, and check coverage as well as answers. Both run on the reference backend and, where
there is a GPU, on that too.
"""

from __future__ import annotations

import collections
import copy

import pytest
import torch
from torch import nn

import tessiq
from conftest import load_operations
from tessiq import _C


class ResidualBlock(nn.Module):
    """A ResNet-18 residual block: two 3x3 convolutions with batch norm, and a skip connection."""

    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.norm1(self.conv1(x)))
        return torch.relu(self.norm2(self.conv2(y)) + x)


class EncoderBlock(nn.Module):
    """A transformer encoder block: self-attention, a feed-forward layer, and two layer norms.

    Written out rather than taken from `nn.TransformerEncoderLayer`, which in eval mode collapses
    into a single `aten._transformer_encoder_layer_fwd` and gives a compiler nothing to look at.
    """

    def __init__(self, width: int = 32, heads: int = 4, hidden: int = 64) -> None:
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.out = nn.Linear(width, width)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.up = nn.Linear(width, hidden)
        self.down = nn.Linear(hidden, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, width = x.shape

        def split(y: torch.Tensor) -> torch.Tensor:
            return y.view(batch, steps, self.heads, self.head_width).transpose(1, 2)

        scores = split(self.query(x)) @ split(self.key(x)).transpose(-2, -1)
        weights = torch.softmax(scores / self.head_width**0.5, dim=-1)
        attended = (weights @ split(self.value(x))).transpose(1, 2).reshape(batch, steps, width)
        x = self.norm1(x + self.out(attended))
        # Annotated because nn.Module.__call__ is untyped, so mypy reads the result as Any.
        out: torch.Tensor = self.norm2(x + self.down(torch.relu(self.up(x))))
        return out


class RMSNorm(nn.Module):
    """RMSNorm the way models write it, with no custom operation in sight."""

    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.randn(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = (
            x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        )
        return out


class RMSNormBlock(nn.Module):
    """A norm, a linear layer and the sigmoid approximation of GELU, all in plain PyTorch.

    Nothing in it names a custom operation. Compiled with `rewrites=["rms_norm", "fast_gelu"]`,
    the norm and the GELU each become one of tessiq's custom operations.
    """

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.norm = RMSNorm(width)
        self.linear = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear(self.norm(x))
        out: torch.Tensor = h * torch.sigmoid(1.702 * h)
        return out


def block_cases() -> list[object]:
    """Each block with an input, named so a failure says which one."""
    torch.manual_seed(0)
    return [
        pytest.param(ResidualBlock().eval(), torch.randn(1, 16, 8, 8), id="residual"),
        pytest.param(EncoderBlock().eval(), torch.randn(2, 6, 32), id="encoder"),
    ]


def capture(model: nn.Module, example: torch.Tensor) -> torch.fx.GraphModule:
    # The package's own, so these tests compile what the backend compiles.
    return tessiq.capture(model, example)


@pytest.fixture(autouse=True)
def _fresh_dynamo() -> None:
    torch._dynamo.reset()


@pytest.mark.parametrize(("model", "example"), block_cases())
def test_a_block_imports(model: nn.Module, example: torch.Tensor) -> None:
    # A convolution with no bias carries a None, and both blocks have a normalisation that returns
    # three tensors.
    text = tessiq.import_graph(capture(model, example))
    assert text.startswith("graph @forward("), text
    # A getitem is not an operation here. Its name survives as the name of the result it selected,
    # which is what lets the executor find the tensor, but no node is added for it.
    assert "function getitem" not in text, text


@pytest.mark.parametrize(("model", "example"), block_cases())
def test_a_block_matches_eager_on_the_reference(model: nn.Module, example: torch.Tensor) -> None:
    with torch.no_grad():
        expected = model(example)
        compiled = torch.compile(model, backend="tessiq", dynamic=False)
        torch.testing.assert_close(compiled(example), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.cuda
@pytest.mark.parametrize(("model", "example"), block_cases())
def test_a_block_matches_eager_on_a_gpu(model: nn.Module, example: torch.Tensor) -> None:
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")

    with torch.no_grad():
        expected = model(example)
        compiled = torch.compile(
            model, backend="tessiq", dynamic=False, options={"device": str(tessiq.gpu)}
        )
        torch.testing.assert_close(compiled(example), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(("model", "example"), block_cases())
def test_what_a_block_claims_matches_the_table(model: nn.Module, example: torch.Tensor) -> None:
    # What the backend claims here has to be what ops.toml says has a kernel, or a custom
    # operation, so a block cannot claim something the emitter would then decline.
    with_kernels = {
        str(spelling)
        for op in load_operations()
        if op["kernel"]
        for spelling in (op["name"], *op["aten"])
    }
    with_kernels |= {name.replace("::", ".") + ".default" for name in tessiq.library.registered()}

    text, _ = tessiq.partition(capture(model, example))
    for line in text.splitlines():
        if "tessiq.supported" not in line:
            continue
        operation = line.split("=", 1)[1].strip().split("(")[0]
        claimed = "tessiq.supported = true" in line
        assert claimed == (operation in with_kernels), f"{operation}: claimed={claimed}\n{line}"


def claimed_and_declined(model: nn.Module, example: torch.Tensor) -> tuple[int, int]:
    """How many operations the backend claims in this block, and how many it leaves."""
    text, _ = tessiq.partition(capture(model, example))
    counts = collections.Counter(
        "tessiq.supported = true" in line
        for line in text.splitlines()
        if "tessiq.supported" in line
    )
    return counts[True], counts[False]


@pytest.mark.parametrize(("model", "example"), block_cases())
def test_a_block_has_something_to_run(model: nn.Module, example: torch.Tensor) -> None:
    _, clusters = tessiq.partition(capture(model, example))
    claimed, _ = claimed_and_declined(model, example)
    assert clusters, "the backend claims nothing in this block"
    assert claimed, "the backend contributes nothing to this block"


def test_the_encoder_block_has_nothing_left_to_decline() -> None:
    # Every operation in the block runs on the device.
    _, declined = claimed_and_declined(EncoderBlock().eval(), torch.randn(2, 6, 32))
    assert declined == 0


def test_the_residual_block_still_exercises_the_fallback() -> None:
    # Keeps the fallback covered: the block still has an operation the backend declines
    # (`aten._to_copy`).
    _, declined = claimed_and_declined(ResidualBlock().eval(), torch.randn(1, 16, 8, 8))
    assert declined > 0


def test_layer_norm_is_broken_up_and_softmax_is_kept_for_its_kernel() -> None:
    # core ATen leaves layer norm whole; tessiq decomposes it into operations it runs.
    text = tessiq.import_graph(capture(EncoderBlock().eval(), torch.randn(2, 6, 32)))
    assert "aten.native_layer_norm" not in text, text
    assert "prims." not in text, "decomposing further than ATen is a different operation set"
    assert "aten.rsqrt.default" in text, text
    # Softmax is kept whole for float32, so csrc/custom_ops/softmax can replace it, and decomposed
    # otherwise.
    assert "aten._softmax.default" in text, text

    double = EncoderBlock().double().eval()
    text = tessiq.import_graph(capture(double, torch.randn(2, 6, 32, dtype=torch.float64)))
    assert "aten._softmax" not in text, text
    assert "aten.amax.default" in text, text


@pytest.mark.cuda
@pytest.mark.parametrize(("model", "example"), block_cases())
def test_a_block_runs_eagerly_on_a_gpu(model: nn.Module, example: torch.Tensor) -> None:
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")
    on_gpu = copy.deepcopy(model).to(tessiq.gpu)
    with torch.no_grad():
        expected = model(example)
        before = _C.cpu_fallback_counts()
        got = on_gpu(example.to(tessiq.gpu))
        after = _C.cpu_fallback_counts()

    torch.testing.assert_close(got.cpu(), expected, rtol=1e-5, atol=1e-5)
    fell_back = sorted(name for name, count in after.items() if count != before.get(name, 0))
    assert not fell_back, f"went through the host: {fell_back}"
