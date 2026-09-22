# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Partitioning a real imported model.

The golden tests in tests/cpp/golden cover the passes against hand-written IR. These check the
same passes against graphs PyTorch actually produces, which is where the operation names and the
attributes the importer writes have to line up.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import pytest
import torch
from torch._dynamo.backends.common import aot_autograd

import tessiq

PARTITION = "mark-supported,cluster"


def partitioned(fn: Callable[..., object], *inputs: torch.Tensor) -> str:
    """Compile `fn`, import the graph it produces, and partition it."""
    captured: list[torch.fx.GraphModule] = []

    def backend(graph_module: torch.fx.GraphModule, _inputs: object) -> Callable[..., object]:
        captured.append(graph_module)
        return graph_module.forward

    torch.compile(fn, backend=aot_autograd(fw_compiler=backend), dynamic=False)(*inputs)
    return tessiq.run_passes(tessiq.import_graph(captured[0]), PARTITION)


def clusters(text: str) -> dict[str, int]:
    """Map each node's result name to the cluster it landed in."""
    found = {}
    for line in text.splitlines():
        match = re.match(r"\s*%(\w+) = .*tessiq\.cluster = (\d+)", line)
        if match:
            found[match.group(1)] = int(match.group(2))
    return found


def test_the_registered_passes_are_visible() -> None:
    names = tessiq.pass_names()
    assert "mark-supported" in names
    assert "cluster" in names


def test_an_all_supported_model_becomes_one_cluster() -> None:
    text = partitioned(lambda a, b: torch.relu(a + b), torch.randn(4), torch.randn(4))
    assigned = clusters(text)
    assert len(assigned) == 2, text
    assert set(assigned.values()) == {0}, "nothing here needs the host, so it is all one cluster"


def test_an_unsupported_op_splits_the_graph() -> None:
    def model(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.tanh(torch.nn.functional.conv1d(torch.relu(a + b), b))

    text = partitioned(model, torch.randn(1, 4, 8), torch.randn(4, 4, 1))
    assigned = clusters(text)

    assert "tessiq.supported = false" in text, "nothing runs a convolution, so it is marked"
    assert len(set(assigned.values())) == 2, f"expected two clusters either side of it:\n{text}"


def test_unsupported_nodes_get_no_cluster() -> None:
    def model(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv1d(a, b)

    text = partitioned(model, torch.randn(1, 4, 8), torch.randn(4, 4, 1))
    assert "tessiq.cluster" not in text, text
    assert "tessiq.supported = false" in text, text


def test_partitioning_is_stable_across_runs() -> None:
    # Cluster numbers follow the order nodes first appear, so the same graph gives the same
    # answer every time. Golden files depend on that.
    def model(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.tanh(torch.relu(a + b) @ b)

    first = partitioned(model, torch.randn(4, 4), torch.randn(4, 4))
    second = partitioned(model, torch.randn(4, 4), torch.randn(4, 4))
    assert first == second


def test_clustering_without_marking_is_refused() -> None:
    ir = tessiq.import_graph(
        _capture(lambda a, b: a + b, torch.randn(4), torch.randn(4)),
    )
    with pytest.raises(ValueError, match="mark-supported"):
        tessiq.run_passes(ir, "cluster")


# --- the partition entry point ------------------------------------------------------------------


def test_partition_reports_which_cluster_each_node_landed_in() -> None:
    def model(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.tanh(torch.nn.functional.conv1d(torch.relu(a + b), b))

    _, clusters = tessiq.partition(_capture(model, torch.randn(1, 4, 8), torch.randn(4, 4, 1)))

    assert clusters["add"] == clusters["relu"], "both are supported and adjacent"
    assert "convolution" not in clusters, "nothing runs a convolution, so it is on the host"
    assert clusters["tanh"] != clusters["add"], "the host node splits the graph"


def test_partition_keys_match_the_fx_node_names() -> None:
    # The executor looks nodes up by the name fx gave them, so the round trip through the IR has
    # to preserve those names exactly.
    graph_module = _capture(lambda a, b: torch.relu(a + b), torch.randn(4), torch.randn(4))
    _, clusters = tessiq.partition(graph_module)

    fx_names = {node.name for node in graph_module.graph.nodes}
    assert set(clusters).issubset(fx_names), set(clusters) - fx_names


def test_partition_returns_the_ir_too() -> None:
    ir, _ = tessiq.partition(_capture(lambda a: torch.relu(a), torch.randn(4)))
    assert ir.startswith("graph @forward(")
    assert "tessiq.cluster" in ir


def test_a_fully_unsupported_model_clusters_nothing() -> None:
    _, clusters = tessiq.partition(
        _capture(
            lambda a, b: torch.nn.functional.conv1d(a, b),
            torch.randn(1, 4, 8),
            torch.randn(4, 4, 1),
        )
    )
    assert clusters == {}


def test_an_unknown_pass_names_what_exists() -> None:
    with pytest.raises(KeyError, match="cluster"):
        tessiq.run_passes("graph @g() -> () {\n  return\n}\n", "no-such-pass")


def _capture(fn: Callable[..., object], *inputs: torch.Tensor) -> torch.fx.GraphModule:
    captured: list[torch.fx.GraphModule] = []

    def backend(graph_module: torch.fx.GraphModule, _inputs: object) -> Callable[..., object]:
        captured.append(graph_module)
        return graph_module.forward

    torch.compile(fn, backend=aot_autograd(fw_compiler=backend), dynamic=False)(*inputs)
    return captured[0]
