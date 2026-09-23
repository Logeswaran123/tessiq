# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""What each kernel costs the device, measured by replay.

A per-kernel reading taken inside a real call carries a mark's worth of bias, and on this card a
mark costs about what a small kernel costs. Replaying a kernel many times under one pair of marks
divides that away. These check the replay is measuring the kernels the recipe holds, rather than
something adjacent to them.
"""

from __future__ import annotations

import pytest
import torch

import tessiq
from tessiq.costs import KernelCost, kernel_costs
from test_models import EncoderBlock

pytestmark = pytest.mark.cuda


@pytest.fixture(autouse=True)
def _needs_a_gpu() -> None:
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")


@pytest.fixture
def encoder_recipe() -> tessiq.Recipe:
    torch.manual_seed(0)
    block = EncoderBlock().eval()
    graph = tessiq.capture(block, torch.randn(4, 16, 32))
    return tessiq.build_recipe(graph, tessiq.gpu, use_cache=False)


def test_every_kernel_is_measured(encoder_recipe: tessiq.Recipe) -> None:
    # A kernel left out would make the total read low, and the total is the number this exists to
    # produce. Silence about a kernel is worse than a wrong figure for it.
    costs = kernel_costs(encoder_recipe, repeats=20)
    assert {cost.name for cost in costs} == set(encoder_recipe.kernels)


def test_each_cost_is_positive_and_named(encoder_recipe: tessiq.Recipe) -> None:
    costs = kernel_costs(encoder_recipe, repeats=20)
    assert costs
    for cost in costs:
        assert isinstance(cost, KernelCost)
        assert cost.microseconds > 0.0
        assert cost.kind in {"fused", "reduction", "gemm"}
        assert cost.repeats == 20


def test_the_slowest_comes_first(encoder_recipe: tessiq.Recipe) -> None:
    costs = kernel_costs(encoder_recipe, repeats=20)
    assert costs == sorted(costs, key=lambda cost: -cost.microseconds)


def test_replaying_more_does_not_change_the_answer(encoder_recipe: tessiq.Recipe) -> None:
    # The whole point of replaying. If the cost of marking were still in the number, measuring
    # over twenty launches and over four hundred would disagree by roughly the mark.
    few = {cost.name: cost.microseconds for cost in kernel_costs(encoder_recipe, repeats=20)}
    many = {cost.name: cost.microseconds for cost in kernel_costs(encoder_recipe, repeats=400)}

    slowest = max(many, key=lambda name: many[name])
    assert many[slowest] == pytest.approx(few[slowest], rel=0.8), (
        f"{slowest}: {few[slowest]:.2f} us over 20, {many[slowest]:.2f} us over 400"
    )


def test_a_matmul_keeps_the_layout_it_was_compiled_for(encoder_recipe: tessiq.Recipe) -> None:
    # A matmul reads the layout it is given rather than requiring a packed one, so replaying it
    # against packed operands would be measuring a different call. The transposed weights in a
    # Linear are the case: every one of them arrives here transposed.
    from tessiq import _C

    gemms = {cost.name for cost in kernel_costs(encoder_recipe, repeats=20) if cost.kind == "gemm"}
    assert gemms, "the encoder block has matmuls"
    transposed = [
        name
        for name, kernel in encoder_recipe.kernels.items()
        if name in gemms and isinstance(kernel, _C.GemmKernel) and any(kernel.transposed)
    ]
    assert transposed, "a Linear stores its weight the other way round"


def test_a_host_recipe_is_refused() -> None:
    # The reference device has no kernels and no clock. Saying so beats returning an empty list,
    # which reads as a model with nothing to measure.
    torch.manual_seed(0)
    block = EncoderBlock().eval()
    graph = tessiq.capture(block, torch.randn(4, 16, 32))
    recipe = tessiq.build_recipe(graph, tessiq.cpu, use_cache=False)

    with pytest.raises(ValueError, match="compiled for a gpu"):
        kernel_costs(recipe)
