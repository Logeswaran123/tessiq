# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Where a compiled call spends its time.

A recipe says how many kernels a graph became and how many of its values have to cross between the
host and the device. It does not say what either costs, and those are the two things that decide
whether a backend is worth using. This measures them.

Two clocks, because a launch returns long before the work does and one clock cannot tell you both
things. The host's says what issuing cost. The device's own, read from a pair of marks on the
stream, says what running cost.

The obvious alternative is to wait for the device before stopping the host clock, which is what
this did until phase 10. It measures the right thing and it changes the program while measuring
it: waiting after every kernel turns a pipeline the device would overlap into a serial one, and on
a transformer block that more than doubled the call. Marking the stream costs a command in a
queue, and the reading happens once at the end when the work has finished anyway.

Neither number is a share of the other. They overlap, which is the point: a device figure well
below the call's own duration is a device that spent the difference waiting to be fed.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

from tessiq import _C
from tessiq.device import cuda_ordinal

#: The profile collecting events on this thread, if any. Thread-local because a profile belongs to
#: the call being measured, and two threads calling the same compiled function are two calls.
_active = threading.local()


@dataclasses.dataclass(frozen=True)
class Event:
    """One measured region of a call.

    Attributes:
        name: The graph node this was, or a description for anything that is not one.
        category: ``kernel``, ``interpreted`` or ``transfer``.
        device: Where it ran, as a string.
        microseconds: Host time, exclusive of anything measured inside it. For a kernel this is
            what issuing it cost, not what running it cost, because a launch returns long before
            the work does.
        device_microseconds: What the device spent on it, from the device's own clock, or None
            where there was nothing to ask. Only a kernel has one.
        nbytes: Bytes moved, for a transfer. Zero otherwise.
    """

    name: str
    category: str
    device: str
    microseconds: float
    nbytes: int = 0
    device_microseconds: float | None = None


class Profile:
    """The events of one or more calls, and what they add up to."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total = 0.0
        self.events: list[Event] = []
        self._pending: list[tuple[int, Any]] = []

    def record(self, event: Event, timer: Any = None) -> None:
        with self._lock:
            self.events.append(event)
            self._total += event.microseconds
            if timer is not None:
                self._pending.append((len(self.events) - 1, timer))

    def resolve(self) -> None:
        """Read the device clock for every region that marked it.

        Left until the run is over on purpose. Reading waits for the mark, and waiting during the
        run is what made the old profiler more than double the call it was describing. By the time
        this runs the work has finished anyway, so the wait costs nothing.
        """
        with self._lock:
            pending, self._pending = self._pending, []
        for index, timer in pending:
            try:
                elapsed = timer.microseconds()
            except (TypeError, ValueError, RuntimeError):
                # A region that queued no device work has nothing to report, which is not a
                # failure. Leaving it as None says so, rather than a zero that reads as instant.
                continue
            with self._lock:
                self.events[index] = dataclasses.replace(
                    self.events[index], device_microseconds=elapsed
                )

    @property
    def device_microseconds(self) -> float:
        """What the device spent, over the regions that could say.

        Not comparable with `microseconds` as a share of one whole. That one is host time and this
        one is device time, and the point of having both is that they overlap.
        """
        return sum(e.device_microseconds or 0.0 for e in self.events)

    @property
    def microseconds(self) -> float:
        """Everything measured, which is most but not all of the call.

        Events hold exclusive time, so this is a running total rather than an outermost duration.
        It is kept rather than summed because every measured region reads it twice, to work out
        how much of itself belongs to regions inside it, and summing made the profiler quadratic
        in the number of steps.
        """
        with self._lock:
            return self._total

    def by_category(self) -> dict[str, tuple[int, float, int]]:
        """Count, microseconds and bytes for each of kernel, interpreted and transfer."""
        totals: dict[str, tuple[int, float, int]] = {}
        for event in self.events:
            count, micros, nbytes = totals.get(event.category, (0, 0.0, 0))
            totals[event.category] = (count + 1, micros + event.microseconds, nbytes + event.nbytes)
        return dict(sorted(totals.items(), key=lambda item: -item[1][1]))

    def summary(self) -> str:
        """A table of where the time went, which is the question this exists to answer."""
        if not self.events:
            return "no events recorded"

        total = self.microseconds
        on_device = {
            category: sum(
                e.device_microseconds or 0.0 for e in self.events if e.category == category
            )
            for category in self.by_category()
        }
        lines = [f"{'':12}{'count':>7}{'host':>12}{'share':>8}{'device':>12}{'moved':>12}"]
        for category, (count, micros, nbytes) in self.by_category().items():
            moved = f"{nbytes / 2**20:.1f} MB" if nbytes else ""
            spent = f"{on_device[category] / 1000:.2f} ms" if on_device[category] else ""
            lines.append(
                f"{category:12}{count:>7}{micros / 1000:>10.2f} ms"
                f"{100 * micros / total:>7.0f}%{spent:>12}{moved:>12}"
            )
        device_total = self.device_microseconds
        spent = f"{device_total / 1000:.2f} ms" if device_total else ""
        lines.append(f"{'total':12}{len(self.events):>7}{total / 1000:>10.2f} ms{'':>8}{spent:>12}")
        return "\n".join(lines)

    def slowest(self, count: int = 10) -> list[Event]:
        """The events that took longest, which is where to look first."""
        return sorted(self.events, key=lambda event: -event.microseconds)[:count]

    def chrome_trace(self) -> list[dict[str, Any]]:
        """The events in the Trace Event format, for chrome://tracing or Perfetto.

        One row per category rather than per thread, because the interesting grouping is what kind
        of work it was and everything here ran in sequence anyway.
        """
        rows = {"kernel": 1, "transfer": 2, "interpreted": 3}
        trace: list[dict[str, Any]] = []
        at = 0.0
        for event in self.events:
            entry: dict[str, Any] = {
                "name": event.name,
                "cat": event.category,
                "ph": "X",
                "ts": at,
                "dur": event.microseconds,
                "pid": 1,
                "tid": rows.get(event.category, 9),
                "args": {"device": event.device},
            }
            if event.nbytes:
                entry["args"]["bytes"] = event.nbytes
            trace.append(entry)
            at += event.microseconds
        return trace

    def write_chrome_trace(self, path: str | Path) -> None:
        """Write the trace where chrome://tracing and Perfetto can open it."""
        Path(path).write_text(json.dumps({"traceEvents": self.chrome_trace()}), encoding="utf-8")


@contextlib.contextmanager
def profile(*, device_time: bool = False) -> Iterator[Profile]:
    """Collect a profile of every compiled call made on this thread inside the block.

    ```python
    with tessiq.profile() as p:
        compiled(x)
    print(p.summary())
    ```

    Args:
        device_time: Also mark the stream around every kernel, so that each one reports what the
            device spent as well as what issuing it cost. Off by default because a mark costs
            about what a small kernel costs, so asking changes the call it is asking about. Turn
            it on when the question is which kernel is slow, and read the numbers knowing each is
            biased upward by the gap between the mark and the launch.

    Nesting is not supported: the inner block takes over and the outer one sees nothing from it.
    """
    collected = Profile()
    previous = getattr(_active, "profile", None)
    previous_detail = getattr(_device_detail, "on", False)
    _active.profile = collected
    _device_detail.on = device_time
    try:
        yield collected
    finally:
        _active.profile = previous
        _device_detail.on = previous_detail
        collected.resolve()


def current() -> Profile | None:
    """The profile collecting on this thread, or None. Called on every step, so it stays cheap."""
    return getattr(_active, "profile", None)


#: Categories whose work the device may still be doing when the call returns.
#:
#: The one category whose work outlives its region. A launch returns as soon as the command is
#: queued, so the host clock around it measures the asking. What the device spent comes from the
#: device, through a pair of marks on the stream that are read once the run is over.
_TIMED_ON_THE_DEVICE = frozenset({"kernel"})

#: Whether the profile collecting on this thread asked for the device clock.
#:
#: Off by default, because marking the stream is not free. On this card `cudaEventRecord` costs
#: about eight microseconds, which is what a small kernel takes to launch and about what one takes
#: to run, so marking every kernel changes the call by half again and biases each reading upward
#: by the gap between the mark and the launch. Worth paying when the question is which kernel is
#: slow. Not worth paying to find out where a call's host time goes, which is the usual question.
_device_detail = threading.local()


@functools.cache
def _timing_ordinal(device: str) -> int:
    """The CUDA device behind `device`, or -1 when there is no clock to read.

    Cached on the spelling because this is asked once per measured region, and the answer is a
    property of the device rather than of the call. Without the cache it was two calls across the
    binding per region, which on a graph with ninety of them is most of what a profiler costs.
    """
    if not _C.device_timing_available():
        return -1
    return int(cuda_ordinal(device))


def _device_timer(category: str, device: torch.device) -> Any:
    """A timer for this region, or None when there is no device clock to read.

    None covers three cases that are all ordinary: a category whose work is finished by the time
    its region ends, a build with no CUDA toolkit, and the host reference device, which is a
    tessiq device whose memory is host memory.
    """
    if category not in _TIMED_ON_THE_DEVICE or not getattr(_device_detail, "on", False):
        return None
    ordinal = _timing_ordinal(str(device))
    if ordinal < 0:
        return None
    timer = _C.DeviceTimer(ordinal)
    return timer if timer.ready else None


@contextlib.contextmanager
def measure(
    profile_: Profile, name: str, category: str, device: torch.device
) -> Iterator[list[int]]:
    """Time a region and record it. The yielded list takes a byte count, for transfers.

    What is recorded is exclusive time: anything measured inside this region is subtracted, so a
    host operation does not also count the transfers that fed it. Without that the shares do not
    add up, and an operation looks expensive when moving its operands was what cost.
    """
    nbytes: list[int] = []
    timer = _device_timer(category, device)
    before = profile_.microseconds
    started = time.perf_counter()
    if timer is not None:
        timer.start()
    try:
        yield nbytes
    finally:
        if timer is not None:
            timer.stop()
        elapsed = (time.perf_counter() - started) * 1e6
        nested = profile_.microseconds - before
        profile_.record(
            Event(
                name=name,
                category=category,
                device=str(device),
                microseconds=max(elapsed - nested, 0.0),
                nbytes=sum(nbytes),
            ),
            timer,
        )
