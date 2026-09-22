# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""The two blocks phase 7 is measured against.

`test_end_to_end.py` takes small models built from operations the backend already claims. These
take blocks out of real architectures, most of whose operations it does not claim, and check that
the answer is still the one eager gives. That is the claim a graph compiler has to make first:
whatever it can run itself, the parts it cannot must still be right.

Both blocks run on the reference backend and, where there is a GPU, on that too.
"""

from __future__ import annotations

import collections

import pytest
import torch
from torch import nn

import tessiq
from conftest import load_operations


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


def block_cases() -> list[object]:
    """Each block with an input, named so a failure says which one."""
    torch.manual_seed(0)
    return [
        pytest.param(ResidualBlock().eval(), torch.randn(1, 16, 8, 8), id="residual"),
        pytest.param(EncoderBlock().eval(), torch.randn(2, 6, 32), id="encoder"),
    ]


def capture(model: nn.Module, example: torch.Tensor) -> torch.fx.GraphModule:
    # The package's own, so that what these tests compile is what the backend compiles. A second
    # copy here would drift, and the way it drifts is a test passing on a graph nobody produces.
    return tessiq.capture(model, example)


@pytest.fixture(autouse=True)
def _fresh_dynamo() -> None:
    torch._dynamo.reset()


@pytest.mark.parametrize(("model", "example"), block_cases())
def test_a_block_imports(model: nn.Module, example: torch.Tensor) -> None:
    # Importing is the first thing that has to work, and until recently neither block could. A
    # convolution with no bias carries a None, and both blocks contain a normalisation that
    # returns three tensors rather than one.
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
    # Coverage checked on a real graph rather than on the table. What the backend claims here has
    # to be exactly what ops.toml says it has a kernel for, so a block cannot quietly claim
    # something the emitter would then decline.
    with_kernels = {
        str(spelling)
        for op in load_operations()
        if op["kernel"]
        for spelling in (op["name"], *op["aten"])
    }

    text, _ = tessiq.partition(capture(model, example))
    for line in text.splitlines():
        if "tessiq.supported" not in line:
            continue
        operation = line.split("=", 1)[1].strip().split("(")[0]
        claimed = "tessiq.supported = true" in line
        assert claimed == (operation in with_kernels), f"{operation}: claimed={claimed}\n{line}"


@pytest.mark.parametrize(("model", "example"), block_cases())
def test_a_block_has_something_to_run_and_something_it_cannot(
    model: nn.Module, example: torch.Tensor
) -> None:
    # Both halves matter. Nothing claimed would mean the backend contributes nothing; nothing
    # declined would mean these blocks have stopped being the hard case they were chosen to be.
    text, clusters = tessiq.partition(capture(model, example))
    counts = collections.Counter(
        "tessiq.supported = true" in line
        for line in text.splitlines()
        if "tessiq.supported" in line
    )
    assert clusters, "the backend claims nothing in this block"
    assert counts[True], text
    assert counts[False], "this block no longer exercises the fallback"


def test_softmax_and_layer_norm_become_operations_the_backend_runs() -> None:
    # Neither has a kernel here and neither needs one. core ATen leaves them whole; tessiq breaks
    # them up, and every piece they break into is something it already runs.
    text = tessiq.import_graph(capture(EncoderBlock().eval(), torch.randn(2, 6, 32)))
    assert "aten._softmax" not in text, text
    assert "aten.native_layer_norm" not in text, text
    assert "prims." not in text, "decomposing further than ATen is a different operation set"
    assert "aten.amax.default" in text, text
    assert "aten.rsqrt.default" in text, text
