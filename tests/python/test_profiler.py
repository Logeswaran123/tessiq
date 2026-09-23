# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Measuring where a compiled call spends its time.

Phase 7 ended with a recipe reporting twenty-eight transfers per call and no way to weigh them
against the kernels. These check that the answer is collected, that it adds up, and that it costs
nothing when nobody is asking.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

import tessiq
from tessiq.profiler import Event, Profile, measure, profile


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


def test_only_a_kernel_asks_the_device_clock() -> None:
    # A launch is the one region whose work outlives it, so the host clock around it measures the
    # asking rather than the doing. A transfer and an interpreted step have finished by the time
    # their region ends, and marking the stream for them would measure nothing new.
    from tessiq.profiler import _TIMED_ON_THE_DEVICE

    assert {"kernel"} == _TIMED_ON_THE_DEVICE


def test_a_region_with_no_device_clock_records_none() -> None:
    # Most regions cannot ask, and the difference between "did not ask" and "took no time" has to
    # survive into the event. A zero would read as a kernel that ran instantly.
    collected = Profile()
    with measure(collected, "n", "interpreted", torch.device("cpu")):
        pass
    collected.resolve()

    assert collected.events[0].device_microseconds is None
    assert collected.device_microseconds == 0.0


@pytest.mark.cuda
def test_a_kernel_records_what_the_device_spent() -> None:
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")

    source = torch.randn(256, 256, device=str(tessiq.gpu))
    with profile(device_time=True) as collected, measure(collected, "n", "kernel", tessiq.gpu):
        source.transpose(0, 1).contiguous()

    event = collected.events[0]
    assert event.device_microseconds is not None
    assert event.device_microseconds > 0.0
    assert collected.device_microseconds == pytest.approx(event.device_microseconds)


@pytest.mark.cuda
def test_the_device_clock_is_not_read_unless_it_is_asked_for() -> None:
    # Marking the stream costs about what a small kernel costs on this card, so a profile that
    # only wants to know where the host time went must not pay for it.
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")

    source = torch.randn(256, 256, device=str(tessiq.gpu))
    with profile() as collected, measure(collected, "n", "kernel", tessiq.gpu):
        source.transpose(0, 1).contiguous()

    assert collected.events[0].device_microseconds is None


@pytest.mark.cuda
def test_profiling_does_not_double_the_call() -> None:
    # The old profiler drained after every kernel, which turned a pipeline the device overlaps
    # into a serial one and more than doubled a transformer block. Marking the stream costs the
    # host a command in a queue, so the shape of the call survives being measured.
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")

    source = torch.randn(512, 512, device=str(tessiq.gpu))

    def work() -> None:
        for _ in range(20):
            source.transpose(0, 1).contiguous()

    work()
    bare = time.perf_counter()
    work()
    bare = time.perf_counter() - bare

    watched = time.perf_counter()
    with profile():
        work()
    watched = time.perf_counter() - watched

    # Generous, because this is a wall clock on a shared machine. What it catches is a profiler
    # that serialises, which was a factor of two rather than a few percent.
    assert watched < bare * 4 + 0.05


def test_an_input_already_on_the_device_is_not_reported_as_moved() -> None:
    # A module's weights and an input passed again unchanged both stay on the device between
    # calls. Nothing crosses for them, so nothing may be reported. The profiler used to count every
    # one of them as a transfer on every call, which put twelve megabytes a call on an encoder
    # block that only ever moved its two megabyte output.
    torch.manual_seed(0)
    layer = torch.nn.Linear(64, 64).eval()
    example = torch.randn(8, 64)
    run = torch.compile(layer, backend="tessiq", options={"device": str(tessiq.cpu)})

    with torch.no_grad():
        run(example)
        with tessiq.profile() as collected:
            result = run(example)

    moves = [event for event in collected.events if event.category == "transfer"]
    moved = sum(event.nbytes for event in moves)
    assert moved == result.numel() * result.element_size(), (
        f"only the output crosses; reported {moved} bytes over {len(moves)} transfers"
    )
