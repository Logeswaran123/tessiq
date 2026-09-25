# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Where a compiled call spends its time.

A recipe says how many kernels a graph became and how many values cross between the host and the
device, but not what either costs. This measures both, with two clocks: the host's says what
issuing work cost, and the device's, read from marks on the stream, says what running it cost. The
marks are read once at the end of a run, so measuring does not serialise the pipeline.

The two numbers overlap rather than one being a share of the other. Device time well below the
call's duration means the device spent the difference waiting for work.
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

#: The profile collecting events on this thread, if any. Thread-local, since two threads calling
#: the same compiled function make two calls.
_active = threading.local()


@dataclasses.dataclass(frozen=True)
class Event:
    """One measured region of a call.

    Attributes:
        name: The graph node this was, or a description for anything that is not one.
        category: ``kernel``, ``interpreted`` or ``transfer``.
        device: Where it ran, as a string.
        microseconds: Host time, exclusive of anything measured inside it. For a kernel this is
            what issuing it cost, not what running it cost.
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

        Left until the run is over, because reading waits for the mark and waiting during the run
        would serialise it.
        """
        with self._lock:
            pending, self._pending = self._pending, []
        for index, timer in pending:
            try:
                elapsed = timer.microseconds()
            except (TypeError, ValueError, RuntimeError):
                # No device work was queued. None, rather than a zero that reads as instant.
                continue
            with self._lock:
                self.events[index] = dataclasses.replace(
                    self.events[index], device_microseconds=elapsed
                )

    @property
    def device_microseconds(self) -> float:
        """What the device spent, over the regions that could say.

        Not a share of `microseconds`, which is host time; the two overlap.
        """
        return sum(e.device_microseconds or 0.0 for e in self.events)

    @property
    def microseconds(self) -> float:
        """Everything measured, which is most but not all of the call.

        Events hold exclusive time, so this is a running total rather than an outermost duration.
        Kept rather than summed because every measured region reads it twice, and summing each
        time would be quadratic in the number of steps.
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
        """A table of where the time went."""
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

        One row per category rather than per thread, since everything here ran in sequence.
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
        device_time: Also mark the stream around every kernel, so each one reports device time.
            Off by default because a mark costs about as much as a small kernel, which slows the
            call and biases each reading upward. Use it to find which kernel is slow.

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


#: Categories whose work the device may still be doing when the region ends, so their device time
#: comes from marks on the stream.
_TIMED_ON_THE_DEVICE = frozenset({"kernel"})

#: Whether the profile collecting on this thread asked for the device clock. See `profile`.
_device_detail = threading.local()


@functools.cache
def _timing_ordinal(device: str) -> int:
    """The CUDA device behind `device`, or -1 when there is no clock to read.

    Cached on the spelling, since this is asked once per measured region and each miss is two
    calls across the binding.
    """
    if not _C.device_timing_available():
        return -1
    return int(cuda_ordinal(device))


def _device_timer(category: str, device: torch.device) -> Any:
    """A timer for this region, or None when there is no device clock to read.

    None for a category whose work finishes within its region, a build with no CUDA toolkit, and
    the CPU reference device.
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

    Records exclusive time: anything measured inside this region is subtracted, so a host
    operation does not also count the transfers that fed it.
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
