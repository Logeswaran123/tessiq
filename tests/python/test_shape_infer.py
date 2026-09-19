# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Shape inference against graphs PyTorch actually produces.

The golden files pin the pass against IR written by hand, which proves the rules are
self-consistent. They cannot prove the rules agree with torch. These tests can: the importer
records the exact type torch computed for every node, so running inference over a real graph and
finding no disagreement is a direct check of the rule table against the real semantics.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch._decomp import core_aten_decompositions
from torch._dynamo.backends.common import aot_autograd

import tessiq


def imported(fn: Callable[..., Any], *inputs: torch.Tensor) -> str:
    captured: list[torch.fx.GraphModule] = []

    def backend(graph_module: torch.fx.GraphModule, _inputs: object) -> Callable[..., object]:
        captured.append(graph_module)
        return graph_module.forward

    torch.compile(
        fn,
        backend=aot_autograd(fw_compiler=backend, decompositions=core_aten_decompositions()),
        dynamic=False,
    )(*inputs)
    return tessiq.import_graph(captured[0])


MODELS: list[tuple[str, Callable[..., Any], tuple[torch.Tensor, ...]]] = [
    ("elementwise chain", lambda a, b: torch.relu(a + b) * b, (torch.randn(8), torch.randn(8))),
    ("broadcast a row", lambda a, b: a + b, (torch.randn(4, 8), torch.randn(8))),
    ("broadcast a column", lambda a, b: a * b, (torch.randn(4, 8), torch.randn(4, 1))),
    ("scalar argument", lambda a: torch.relu(a * 2.0), (torch.randn(8),)),
    ("matmul", lambda a, b: torch.tanh(a @ b), (torch.randn(4, 16), torch.randn(16, 2))),
    ("rank mixing", lambda a, b: torch.sigmoid(a + b), (torch.randn(2, 3, 8), torch.randn(8))),
]


@pytest.mark.parametrize(("name", "fn", "inputs"), MODELS, ids=[m[0] for m in MODELS])
def test_inference_agrees_with_torch(
    name: str, fn: Callable[..., Any], inputs: tuple[torch.Tensor, ...]
) -> None:
    # The importer already recorded torch's own answer on every node. If our rules disagreed, the
    # pass would reject the graph; if they were vaguer, it would leave a dimension dynamic. Neither
    # happens, so the text is unchanged.
    text = imported(fn, *inputs)
    assert tessiq.run_passes(text, "shape-infer") == text, name


def test_inference_runs_before_partitioning_without_disturbing_it() -> None:
    text = imported(lambda a, b: torch.relu(a + b) * b, torch.randn(8), torch.randn(8))
    assert tessiq.run_passes(text, "shape-infer,mark-supported,cluster") == tessiq.run_passes(
        text, "mark-supported,cluster"
    )


def test_a_contradiction_is_reported() -> None:
    text = (
        "graph @g(%a: f32[4,8], %b: f32[4,8]) -> f32[4,9] {\n"
        "  %s = add(%a, %b) : f32[4,9]\n"
        "  return %s\n"
        "}\n"
    )
    with pytest.raises(ValueError, match="recorded as f32"):
        tessiq.run_passes(text, "shape-infer")
