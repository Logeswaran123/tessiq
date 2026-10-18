#!/usr/bin/env python3
# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Runs clang-format and clang-tidy over the C++ sources, as CI does.

clang-tidy needs a compilation database, so configure a build first. Files run in parallel, and a
file's output is printed only when it fails.

Run with: python tools/lint_cpp.py [--format] [--tidy] [--build-dir build] [-j N]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ["csrc/*.cpp", "csrc/*.h", "csrc/*.cu", "tests/cpp/*.cpp", "tests/cpp/*.cu"]


def tracked(patterns: list[str]) -> list[str]:
    """Files git tracks that match `patterns`, relative to the repository root."""
    result = subprocess.run(
        ["git", "ls-files", *patterns], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return sorted(result.stdout.split())


def check_format(files: list[str]) -> bool:
    result = subprocess.run(
        ["clang-format", "--dry-run", "--Werror", *files], cwd=ROOT, check=False
    )
    return result.returncode == 0


def check_tidy(files: list[str], build_dir: Path, jobs: int) -> bool:
    database = build_dir / "compile_commands.json"
    if not database.is_file():
        sys.exit(f"{database} not found: configure a build first, e.g. cmake -S . -B {build_dir}")

    def tidy(file: str) -> tuple[str, subprocess.CompletedProcess[str]]:
        command = ["clang-tidy", "--quiet", "-p", str(build_dir), file]
        return file, subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)

    failed = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for file, result in pool.map(tidy, files):
            if result.returncode != 0:
                failed.append(file)
                print(result.stdout, result.stderr, sep="", end="")
    if failed:
        print(f"clang-tidy failed on {len(failed)} of {len(files)} files: {' '.join(failed)}")
    return not failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--format", action="store_true", help="run clang-format only")
    parser.add_argument("--tidy", action="store_true", help="run clang-tidy only")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build", help="for clang-tidy")
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    run_all = not (args.format or args.tidy)

    files = tracked(SOURCES)
    ok = True
    if run_all or args.format:
        ok &= check_format(files)
    if run_all or args.tidy:
        ok &= check_tidy(
            [f for f in files if f.endswith(".cpp")], args.build_dir.resolve(), args.jobs
        )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
