# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Rewrites: RMSNorm and GELU written out in plain PyTorch, run as tessiq's custom operations.

`rms_norm` and `fast_gelu` each have a pattern in csrc/custom_ops, and a pass that replaces what the
pattern describes with one call. The golden files in golden/ run each pass over graphs it must
rewrite and graphs it must leave alone, through the passes the extension registered. The rest
compile models that never name a custom operation and check what the rewrites made of them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import tessiq
from tessiq import _C
from tessiq.recipe import build_recipe
from test_models import RMSNormBlock

GOLDEN = Path(__file__).parent / "golden"


BOTH = ["rms_norm", "fast_gelu"]


def compiled(model: nn.Module, rewrites: list[str], *, dynamic: bool = False) -> Any:
    options: dict[str, Any] = {"device": str(tessiq.gpu), "rewrites": rewrites}
    return torch.compile(model, backend="tessiq", options=options, dynamic=dynamic)


@pytest.fixture
def needs_gpu() -> None:
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")


@pytest.fixture(autouse=True)
def _fresh() -> None:
    torch._dynamo.reset()


@pytest.mark.parametrize("path", sorted(GOLDEN.glob("rewrite_*.tsr")), ids=lambda p: p.stem)
def test_golden(path: Path) -> None:
    head, separator, expected = path.read_text().partition("// ---\n")
    assert separator, f"{path.name} has no '// ---' separator"
    run, _, graph = head.partition("\n")
    assert run.startswith("// RUN:"), f"{path.name} does not start with a RUN line"
    assert _C.run_passes(graph, run.removeprefix("// RUN:").strip()) == expected


def test_each_pattern_has_a_pass_of_its_own() -> None:
    assert tessiq.library.rewrites() == {
        "tessiq::fast_gelu": "rewrite-fast-gelu",
        "tessiq::gemm_bias_gelu": "rewrite-gemm-bias-gelu",
        "tessiq::gemm_bias_gelu_tanh": "rewrite-gemm-bias-gelu-tanh",
        "tessiq::rms_norm": "rewrite-rms-norm",
    }


def test_a_rewrite_runs_only_when_a_compile_asks_for_it() -> None:
    assert not any(name.startswith("rewrite-") for name in tessiq.default_pipeline())


def test_a_rewrite_is_asked_for_by_its_operation_or_its_pass() -> None:
    assert tessiq.library.rewrite_passes(["rms_norm", "tessiq::fast_gelu", "rewrite-rms-norm"]) == [
        "rewrite-rms-norm",
        "rewrite-fast-gelu",
    ]


def test_asking_for_a_rewrite_nothing_has_names_the_ones_there_are() -> None:
    with pytest.raises(
        ValueError, match="these do: fast_gelu, gemm_bias_gelu, gemm_bias_gelu_tanh, rms_norm"
    ):
        tessiq.library.rewrite_passes(["layer_norm"])


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
def test_compiled_each_written_out_form_is_one_launch() -> None:
    graph = tessiq.capture(RMSNormBlock().eval(), torch.randn(2, 8, 64))
    recipe = build_recipe(graph, tessiq.gpu, use_cache=False, rewrites=BOTH)

    (norm,) = [k for k in recipe.kernels.values() if isinstance(k, _C.CustomKernel)]
    assert norm.op == "tessiq::rms_norm"
    assert norm.members == ["pow_1", "mean", "add", "rsqrt", "mul", "mul_1"]
    (gelu,) = [k for k in recipe.kernels.values() if isinstance(k, _C.FusedKernel)]
    assert set(gelu.members) == {"mul_2", "sigmoid", "mul_3"}
    # Every node either kernel absorbed is skipped rather than run a second time.
    assert {"pow_1", "mean", "add", "rsqrt", "mul", "mul_2", "sigmoid"} <= recipe.fused_away


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
def test_without_the_rewrites_nothing_changes() -> None:
    graph = tessiq.capture(RMSNormBlock().eval(), torch.randn(2, 8, 64))
    recipe = build_recipe(graph, tessiq.gpu, use_cache=False)
    assert not any(isinstance(k, _C.CustomKernel) for k in recipe.kernels.values())
    assert len(recipe.kernels) > 3


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
def test_the_cache_tells_a_compile_with_rewrites_from_one_without() -> None:
    graph = tessiq.capture(RMSNormBlock().eval(), torch.randn(2, 8, 64))
    plain = build_recipe(graph, tessiq.gpu)
    rewritten = build_recipe(graph, tessiq.gpu, rewrites=BOTH)
    assert len(rewritten.kernels) < len(plain.kernels)


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
def test_compiled_it_matches_eager_and_runs_the_kernel() -> None:
    model = RMSNormBlock().eval()
    x = torch.randn(4, 16, 64)
    before = tessiq.library.fallbacks("tessiq::rms_norm")
    with torch.no_grad():
        torch.testing.assert_close(compiled(model, BOTH)(x), model(x))
    assert tessiq.library.fallbacks("tessiq::rms_norm") == before, "the kernel declined the call"


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
def test_compiled_on_a_shape_that_changes() -> None:
    model = RMSNormBlock().eval()
    run = compiled(model, BOTH, dynamic=True)
    with torch.no_grad():
        for rows in (3, 11, 29):
            x = torch.randn(rows, 64)
            torch.testing.assert_close(run(x), model(x))


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
def test_compiled_for_a_type_the_norm_kernel_does_not_take() -> None:
    # rms_norm's kernel is float32 only, so the norm stays written out. fast_gelu takes float64,
    # so the GELU is still rewritten. The linear layer between them runs on the host, since the
    # matmul is float32 only too.
    model = RMSNormBlock().double().eval()
    x = torch.randn(2, 8, 64, dtype=torch.float64)
    graph = tessiq.capture(model, x)
    recipe = build_recipe(graph, tessiq.gpu, use_cache=False, rewrites=BOTH)
    assert not any(isinstance(k, _C.CustomKernel) for k in recipe.kernels.values())
    assert {"mul_2", "sigmoid"} <= recipe.fused_away
    with torch.no_grad():
        torch.testing.assert_close(compiled(model, BOTH)(x), model(x))


def test_explain_shows_the_rewrite() -> None:
    report = tessiq.explain(
        RMSNormBlock().eval(), torch.randn(2, 8, 64), sections=("passes",), rewrites=BOTH
    )
    assert "rewrite-rms-norm:" in report
    assert "+  %mul_1 = tessiq.rms_norm.default(%arg0_1, %arg1_1)" in report
    assert "rewrite-fast-gelu:" in report
    assert "+  %mul_3 = tessiq.fast_gelu.default(%view_1)" in report


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
def test_training_with_the_rewrites_named_reaches_the_same_parameters_as_eager() -> None:
    # The forward graph of a training step returns the values the backward graph reads, the norm's
    # reciprocal among them, so the patterns covering them do not match, and the step trains
    # exactly as it would without the rewrites.
    inputs = torch.randn(32, 64)
    targets = torch.randn(32, 64)

    def train(rewrites: list[str] | None) -> nn.Module:
        torch.manual_seed(0)
        net = RMSNormBlock()
        run = net if rewrites is None else compiled(net, rewrites)
        optimiser = torch.optim.SGD(net.parameters(), lr=0.05)
        for _ in range(20):
            optimiser.zero_grad()
            nn.functional.mse_loss(run(inputs), targets).backward()  # type: ignore[no-untyped-call]
            optimiser.step()
        return net

    eager = train(None)
    torch._dynamo.reset()
    ours = train(BOTH)
    for theirs, mine in zip(eager.parameters(), ours.parameters(), strict=True):
        torch.testing.assert_close(mine, theirs, rtol=1e-4, atol=1e-5)
