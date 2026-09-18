# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Converting FX graphs into tessiq IR.

Each test compiles a real function, captures what a ``torch.compile`` backend would receive, and
compares the imported IR against the text it should produce. Comparing text rather than poking at
a structure means a failure is a readable diff.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch._dynamo.backends.common import aot_autograd

import tessiq


def capture(fn: Callable[..., object], *inputs: torch.Tensor) -> torch.fx.GraphModule:
    """Return the FX graph a backend receives for `fn`, after AOTAutograd."""
    captured: list[torch.fx.GraphModule] = []

    def backend(
        graph_module: torch.fx.GraphModule, _example_inputs: object
    ) -> Callable[..., object]:
        captured.append(graph_module)
        return graph_module.forward

    torch.compile(fn, backend=aot_autograd(fw_compiler=backend), dynamic=False)(*inputs)
    assert captured, "nothing was captured; the function may have been traced away"
    return captured[0]


def imported(fn: Callable[..., object], *inputs: torch.Tensor) -> str:
    return tessiq.import_graph(capture(fn, *inputs))


def test_a_single_operation() -> None:
    text = imported(lambda a, b: a + b, torch.randn(4), torch.randn(4))
    assert text == (
        "graph @forward(%arg0_1: f32[4], %arg1_1: f32[4]) -> f32[4] {\n"
        "  %add = aten.add.Tensor(%arg0_1, %arg1_1) : f32[4]\n"
        "  return %add\n"
        "}\n"
    )


def test_operations_keep_their_order_and_operands() -> None:
    text = imported(lambda a, b: torch.relu(a + b), torch.randn(2, 3), torch.randn(2, 3))
    assert text == (
        "graph @forward(%arg0_1: f32[2,3], %arg1_1: f32[2,3]) -> f32[2,3] {\n"
        "  %add = aten.add.Tensor(%arg0_1, %arg1_1) : f32[2,3]\n"
        "  %relu = aten.relu.default(%add) : f32[2,3]\n"
        "  return %relu\n"
        "}\n"
    )


def test_a_scalar_argument_becomes_a_positional_attribute() -> None:
    # 2.0 is not a tensor, so it cannot be an operand. Keying the attribute by position keeps
    # which argument it filled.
    text = imported(lambda a: a * 2.0, torch.randn(3))
    assert "{arg1 = 2.0}" in text, text
    assert "aten.mul.Tensor(%arg0_1)" in text, text


def test_the_graph_name_is_configurable() -> None:
    graph_module = capture(lambda a: torch.relu(a), torch.randn(2))
    assert tessiq.import_graph(graph_module, "block0").startswith("graph @block0(")


def test_several_results() -> None:
    text = imported(lambda a: (a + 1, a - 1), torch.randn(4))
    assert "-> (f32[4], f32[4])" in text, text
    assert text.rstrip().splitlines()[-2].startswith("  return %"), text


def test_a_parameter_used_twice_becomes_one_value() -> None:
    text = imported(lambda a: a * a, torch.randn(4))
    assert "aten.mul.Tensor(%arg0_1, %arg0_1)" in text, text


@pytest.mark.parametrize(
    ("dtype", "spelling"),
    [
        (torch.float32, "f32"),
        (torch.float64, "f64"),
        (torch.float16, "f16"),
        (torch.bfloat16, "bf16"),
        (torch.int32, "i32"),
        (torch.int64, "i64"),
    ],
)
def test_dtypes_map_onto_the_ir_spelling(dtype: torch.dtype, spelling: str) -> None:
    text = imported(lambda a: a + a, torch.ones(4, dtype=dtype))
    assert f"{spelling}[4]" in text, text


def test_shapes_carry_every_dimension() -> None:
    text = imported(lambda a: a + a, torch.randn(2, 3, 4))
    assert "f32[2,3,4]" in text, text


def test_a_scalar_tensor_prints_as_an_empty_shape() -> None:
    text = imported(lambda a: a.sum(), torch.randn(4))
    assert "f32[]" in text, text


def test_an_unsupported_dtype_says_which_one() -> None:
    with pytest.raises(TypeError, match="ComplexFloat"):
        imported(lambda a: a + a, torch.randn(4, dtype=torch.complex64))


def test_a_node_without_shape_information_is_refused() -> None:
    # Importing a graph that never went through shape propagation would put guessed types on
    # values the passes then trust.
    graph_module = capture(lambda a: a + a, torch.randn(4))
    for node in graph_module.graph.nodes:
        node.meta.pop("val", None)

    with pytest.raises(ValueError, match="shape propagation"):
        tessiq.import_graph(graph_module)


def test_a_none_in_the_output_is_not_a_result() -> None:
    # A backward graph returns None at every position whose input takes no gradient. Nothing is
    # computed there, so the IR carries no value for it, and the remaining results keep their
    # order.
    graph_module = capture(lambda a, b: a * b, torch.randn(4), torch.randn(4))
    output = next(node for node in graph_module.graph.nodes if node.op == "output")
    output.args = ((None, *output.args[0], None),)

    text = tessiq.import_graph(graph_module)
    assert text.rstrip().endswith("return %mul\n}")


def test_a_constant_in_the_output_says_what_it_was() -> None:
    graph_module = capture(lambda a: a + a, torch.randn(4))
    output = next(node for node in graph_module.graph.nodes if node.op == "output")
    output.args = ((7, *output.args[0]),)

    with pytest.raises(TypeError, match="constant '7'"):
        tessiq.import_graph(graph_module)


def test_the_result_is_valid_ir() -> None:
    # import_fx verifies before returning, so reaching here means the graph satisfied every
    # structural invariant: operands defined before use, no duplicate names, intact use lists.
    text = imported(lambda a, b: torch.relu(a * b + a), torch.randn(8), torch.randn(8))
    assert text.startswith("graph @forward(")
    assert text.rstrip().endswith("}")
