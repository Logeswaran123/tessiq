# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Where a compiled call spends its time.

A recipe says how many kernels a graph became and how many of its values have to cross between the
host and the device. It does not say what either costs, and those are the two things that decide
whether a backend is worth using. This measures them.

The clock is the host's, and every measured region is drained before it stops, so a kernel's time
is the time until it finished rather than the time to ask for it. That serialises work the device
would otherwise overlap, which is the honest trade for attributing it: a profile is slower than
the call it describes, and the proportions are what it is for.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

#: The profile collecting events on this thread, if any. Thread-local because a profile belongs to
#: the call being measured, and two threads calling the same compiled function are two calls.
_active = threading.local()


@dataclasses.dataclass(frozen=True)
class Event:
    """One measured region of a call.

    Attributes:
        name: The graph node this was, or a description for anything that is not one.
        category: ``kernel``, ``host`` or ``transfer``.
        device: Where it ran, as a string.
        microseconds: How long it took, with the device drained before the clock stopped.
        nbytes: Bytes moved, for a transfer. Zero otherwise.
    """

    name: str
    category: str
    device: str
    microseconds: float
    nbytes: int = 0


class Profile:
    """The events of one or more calls, and what they add up to."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.events: list[Event] = []

    def record(self, event: Event) -> None:
        with self._lock:
            self.events.append(event)

    @property
    def microseconds(self) -> float:
        """Everything measured, which is most but not all of the call.

        Events hold exclusive time, so this is a sum rather than an outermost duration.
        """
        with self._lock:
            return sum(event.microseconds for event in self.events)

    def by_category(self) -> dict[str, tuple[int, float, int]]:
        """Count, microseconds and bytes for each of kernel, host and transfer."""
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
        lines = [f"{'':12}{'count':>7}{'time':>12}{'share':>8}{'moved':>12}"]
        for category, (count, micros, nbytes) in self.by_category().items():
            moved = f"{nbytes / 2**20:.1f} MB" if nbytes else ""
            lines.append(
                f"{category:12}{count:>7}{micros / 1000:>10.2f} ms"
                f"{100 * micros / total:>7.0f}%{moved:>12}"
            )
        lines.append(f"{'total':12}{len(self.events):>7}{total / 1000:>10.2f} ms")
        return "\n".join(lines)

    def slowest(self, count: int = 10) -> list[Event]:
        """The events that took longest, which is where to look first."""
        return sorted(self.events, key=lambda event: -event.microseconds)[:count]

    def chrome_trace(self) -> list[dict[str, Any]]:
        """The events in the Trace Event format, for chrome://tracing or Perfetto.

        One row per category rather than per thread, because the interesting grouping is what kind
        of work it was and everything here ran in sequence anyway.
        """
        rows = {"kernel": 1, "transfer": 2, "host": 3}
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
def profile() -> Iterator[Profile]:
    """Collect a profile of every compiled call made on this thread inside the block.

    ```python
    with tessiq.profile() as p:
        compiled(x)
    print(p.summary())
    ```

    Nesting is not supported: the inner block takes over and the outer one sees nothing from it.
    """
    collected = Profile()
    previous = getattr(_active, "profile", None)
    _active.profile = collected
    try:
        yield collected
    finally:
        _active.profile = previous


def current() -> Profile | None:
    """The profile collecting on this thread, or None. Called on every step, so it stays cheap."""
    return getattr(_active, "profile", None)


def _synchronise(device: torch.device) -> None:
    """Drain `device` so that a measured region has finished rather than been queued."""
    if device.type != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


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
    before = profile_.microseconds
    started = time.perf_counter()
    try:
        yield nbytes
    finally:
        _synchronise(device)
        elapsed = (time.perf_counter() - started) * 1e6
        nested = profile_.microseconds - before
        profile_.record(
            Event(
                name=name,
                category=category,
                device=str(device),
                microseconds=max(elapsed - nested, 0.0),
                nbytes=sum(nbytes),
            )
        )
