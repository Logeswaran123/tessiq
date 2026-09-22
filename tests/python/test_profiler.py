# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Measuring where a compiled call spends its time.

Phase 7 ended with a recipe reporting twenty-eight transfers per call and no way to weigh them
against the kernels. These check that the answer is collected, that it adds up, and that it costs
nothing when nobody is asking.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

import tessiq
from tessiq.profiler import Event, Profile


def chain(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(torch.relu(a + b) * 2.0)


@pytest.fixture(autouse=True)
def _fresh_dynamo() -> None:
    torch._dynamo.reset()


def compiled_chain(device: torch.device) -> Callable[..., torch.Tensor]:
    compiled: Callable[..., torch.Tensor] = torch.compile(
        chain, backend="tessiq", dynamic=False, options={"device": str(device)}
    )
    return compiled


def test_nothing_is_recorded_without_a_profile() -> None:
    # The check runs on every step of every call, so it has to be the cheap path when off.
    assert tessiq.profiler.current() is None
    compiled_chain(tessiq.cpu)(torch.randn(8), torch.randn(8))
    assert tessiq.profiler.current() is None


def test_a_call_records_its_steps() -> None:
    run = compiled_chain(tessiq.cpu)
    run(torch.randn(8), torch.randn(8))

    with tessiq.profile() as collected:
        run(torch.randn(8), torch.randn(8))

    assert collected.events, "a compiled call recorded nothing"
    assert set(collected.by_category()) <= {"kernel", "interpreted", "transfer"}


def test_the_profile_stops_collecting_after_the_block() -> None:
    run = compiled_chain(tessiq.cpu)
    with tessiq.profile() as collected:
        run(torch.randn(8), torch.randn(8))
    before = len(collected.events)

    run(torch.randn(8), torch.randn(8))
    assert len(collected.events) == before


def test_time_is_exclusive() -> None:
    # A host operation wraps the transfers that feed it. Counting those twice makes the operation
    # look expensive when moving its operands was what cost, and the shares stop adding up.
    collected = Profile()
    # Written nested on purpose, because the nesting is the thing under test. Combining them into
    # one `with` means the same thing and stops it reading as one region inside another.
    with tessiq.profiler.measure(collected, "outer", "interpreted", tessiq.cpu):  # noqa: SIM117
        with tessiq.profiler.measure(collected, "inner", "transfer", tessiq.cpu) as nbytes:
            nbytes.append(4096)
            torch.randn(256, 256).sum()

    inner = next(e for e in collected.events if e.name == "inner")
    outer = next(e for e in collected.events if e.name == "outer")
    assert inner.microseconds > 0
    assert outer.microseconds < inner.microseconds, "the inner region was counted twice"
    assert inner.nbytes == 4096


def test_the_summary_adds_up() -> None:
    collected = Profile()
    for name, category, micros in [
        ("a", "kernel", 10.0),
        ("b", "transfer", 30.0),
        ("c", "interpreted", 60.0),
    ]:
        collected.record(Event(name=name, category=category, device="cpu", microseconds=micros))

    assert collected.microseconds == pytest.approx(100.0)
    counts = collected.by_category()
    assert counts["interpreted"][1] == pytest.approx(60.0)
    assert list(counts) == ["interpreted", "transfer", "kernel"], "slowest first"
    assert "60%" in collected.summary()


def test_a_chrome_trace_is_valid_json_with_one_row_per_category(tmp_path: Path) -> None:
    collected = Profile()
    collected.record(Event("k", "kernel", "tessiq:1", 10.0))
    collected.record(Event("m", "transfer", "tessiq:1", 5.0, nbytes=2048))

    path = tmp_path / "trace.json"
    collected.write_chrome_trace(path)
    loaded = json.loads(path.read_text())

    events = loaded["traceEvents"]
    assert [e["name"] for e in events] == ["k", "m"]
    assert all(e["ph"] == "X" for e in events)
    assert events[0]["tid"] != events[1]["tid"], "kernels and transfers on separate rows"
    assert events[1]["ts"] == events[0]["dur"], "laid out end to end"
    assert events[1]["args"]["bytes"] == 2048


@pytest.mark.cuda
def test_a_gpu_call_reports_kernels_and_transfers() -> None:
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")

    run = compiled_chain(tessiq.gpu)
    inputs = (torch.randn(4096), torch.randn(4096))
    run(*inputs)

    with tessiq.profile() as collected:
        run(*inputs)

    categories = collected.by_category()
    assert "kernel" in categories, collected.summary()
    assert "transfer" in categories, collected.summary()
    moved = categories["transfer"][2]
    assert moved >= 4096 * 4, f"the inputs alone are 16 KB, saw {moved}"


def test_the_running_total_matches_the_events() -> None:
    # The total is kept rather than summed, because every measured region reads it twice to work
    # out how much of itself belongs to regions inside it. Summing made the profiler quadratic in
    # the number of steps and doubled the call it was measuring.
    collected = Profile()
    for i in range(500):
        collected.record(Event(f"e{i}", "kernel", "cpu", float(i)))

    assert collected.microseconds == pytest.approx(sum(float(i) for i in range(500)))


def test_only_a_kernel_drains_the_device() -> None:
    # A launch is asynchronous, so its region has to be drained to have measured anything. A
    # transfer and an interpreted step have already finished, and draining after them measures
    # nothing new at two microseconds a time, which with a third of the graph interpreted was most
    # of what the profiler cost.
    from tessiq.profiler import _NEEDS_DRAINING

    assert {"kernel"} == _NEEDS_DRAINING
