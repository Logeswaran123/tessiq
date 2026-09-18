# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""The Python logging API round-trips and validates its input."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

import tessiq

LEVELS = ["error", "warn", "info", "debug", "trace"]


@pytest.fixture(autouse=True)
def restore_log_level() -> Iterator[None]:
    """Restore the log level after each test, so ordering cannot affect results."""
    saved = tessiq.get_log_level()
    yield
    tessiq.set_log_level(saved)


@pytest.mark.parametrize("level", LEVELS)
def test_set_then_get_round_trips(level: str) -> None:
    tessiq.set_log_level(level)
    assert tessiq.get_log_level() == level


def test_level_names_are_case_insensitive() -> None:
    tessiq.set_log_level("DEBUG")
    assert tessiq.get_log_level() == "debug"


def test_unknown_level_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unknown log level"):
        tessiq.set_log_level("verbose")


def test_error_message_lists_the_valid_levels() -> None:
    # PLAN.md section 7.2: an error message names the fix, not just the problem.
    with pytest.raises(ValueError, match="unknown log level") as excinfo:
        tessiq.set_log_level("verbose")

    message = str(excinfo.value)
    for level in LEVELS:
        assert level in message, f"error message should list {level!r}: {message!r}"
