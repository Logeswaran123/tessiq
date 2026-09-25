# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""One model's whole compilation, in one place.

`import_graph`, `run_passes`, `build_recipe` and `emit_cuda` each answer part of "why is this
operation on the host". This lines their answers up: what the compiler saw, what each pass changed,
what became a kernel, and where every step ends up.
"""

from __future__ import annotations

import difflib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from tessiq.device import DEVICE_NAME, cpu
from tessiq.fx import capture, default_pipeline, emit_cuda, import_graph, run_passes
from tessiq.library import rewrite_passes
from tessiq.recipe import Recipe, build_recipe
from tessiq.shapes import is_static

#: Sections a report can hold, in the order they are printed.
SECTIONS = ("summary", "graph", "passes", "kernels", "schedule")

#: Where a step runs when the recipe does not say. Not `tessiq.cpu`, whose type is `tessiq`. A
#: node is missing from the map when its first result was named after the getitem that reads it,
#: which happens to anything returning more than one tensor.
_HOST = torch.device("cpu")


@dataclass(frozen=True)
class Report:
    """Everything the sections read, worked out once.

    One value, because each section reads a different few of these.
    """

    graph_module: torch.fx.GraphModule
    imported: str
    changes: list[PassChange]
    recipe: Recipe
    target: torch.device
    source: bool
    name: str
    pipeline: list[str]


@dataclass(frozen=True)
class PassChange:
    """What one pass did to the graph.

    Attributes:
        name: The pass.
        changed: Whether the IR came out different from how it went in.
        diff: The changed lines in unified form, empty when nothing changed.
    """

    name: str
    changed: bool
    diff: str


def pass_by_pass(ir: str, pipeline: list[str] | None = None) -> list[PassChange]:
    """Run each pass on its own and record what it changed.

    Running the pipeline a pass at a time gives the same result as running it whole, and shows
    which pass made each change.

    Args:
        ir: The graph in textual IR, as the importer produced it.
        pipeline: Passes to run, in order. Defaults to the ones the backend runs.

    Returns:
        One entry per pass, in order.
    """
    changes: list[PassChange] = []
    before = ir
    for name in pipeline if pipeline is not None else default_pipeline():
        after = run_passes(before, name)
        changes.append(PassChange(name=name, changed=after != before, diff=_diff(before, after)))
        before = after
    return changes


def _diff(before: str, after: str) -> str:
    """The lines a pass changed, each old line followed by the new one that replaced it.

    Paired rather than grouped as a unified diff would. Every pass here annotates, so a grouped
    diff prints the whole graph twice and hides the attribute that changed.
    """
    old = before.splitlines()
    new = after.splitlines()
    lines: list[str] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new).get_opcodes():
        if tag == "equal":
            continue
        for offset in range(max(i2 - i1, j2 - j1)):
            if i1 + offset < i2:
                lines.append("-" + old[i1 + offset])
            if j1 + offset < j2:
                lines.append("+" + new[j1 + offset])
    return "\n".join(lines)


def explain(
    model: torch.nn.Module,
    example: torch.Tensor,
    *,
    device: torch.device | None = None,
    dynamic: bool = False,
    sections: tuple[str, ...] = SECTIONS,
    source: bool = False,
    rewrites: Sequence[str] = (),
) -> str:
    """Report how `model` compiles for `example`.

    Args:
        model: Any module. Put it in eval mode unless training is what you mean to report on.
        example: One input, used to trace. Only its shape and dtype matter.
        device: Where clustered operations would run. Defaults to ``tessiq.cpu``, which has no
            kernels, so pass a GPU to see any.
        dynamic: True to report on the graph Dynamo captures once a dimension has changed, whose
            extents are not known until the call.
        sections: Which parts to include, from `SECTIONS`.
        source: True to print the generated CUDA for each kernel, which is long.
        rewrites: Custom operations whose rewrites to run, as a compile's ``rewrites`` option
            names them. See `tessiq.library.rewrites`.

    Returns:
        The report, ready to print.
    """
    return explain_graph(
        capture(model, example, dynamic=dynamic),
        device=device,
        sections=sections,
        source=source,
        name=type(model).__name__,
        rewrites=rewrites,
    )


def explain_graph(
    graph_module: torch.fx.GraphModule,
    *,
    device: torch.device | None = None,
    sections: tuple[str, ...] = SECTIONS,
    source: bool = False,
    name: str | None = None,
    rewrites: Sequence[str] = (),
) -> str:
    """Report how one already captured graph compiles.

    `name` is what to call the model in the report. An FX graph module is a generated class, so
    its own name says nothing. See `explain` for the rest.
    """
    unknown = sorted(set(sections) - set(SECTIONS))
    if unknown:
        raise ValueError(f"unknown section(s): {', '.join(unknown)}; known: {', '.join(SECTIONS)}")

    target = device if device is not None else cpu
    imported = import_graph(graph_module)
    pipeline = [*rewrite_passes(rewrites), *default_pipeline()]
    changes = pass_by_pass(imported, pipeline)
    recipe = build_recipe(graph_module, target, use_cache=False, rewrites=rewrites)

    parts: list[str] = []
    for section in SECTIONS:
        if section not in sections:
            continue
        body = _SECTIONS[section](
            Report(
                graph_module,
                imported,
                changes,
                recipe,
                target,
                source,
                name or "the graph",
                pipeline,
            )
        )
        parts.append(f"== {section} ==\n\n{body.rstrip()}")
    return "\n\n".join(parts) + "\n"


def _summary(report: Report) -> str:
    recipe = report.recipe
    steps = [node for node in report.graph_module.graph.nodes if node.op == "call_function"]
    on_device = [n for n in steps if recipe.devices.get(n.name, _HOST).type == DEVICE_NAME]
    covered = _covered_by_kernels(recipe)
    shapes = "known" if is_static(report.graph_module) else "known at the call"
    rows = [
        ("model", f"{report.name} on {report.target}"),
        ("shapes", shapes),
        ("operations", str(len(steps))),
        ("on the device", str(len(on_device))),
        ("kernels", f"{len(recipe.kernels)}, covering {len(covered)} operation(s)"),
        ("arena", f"{recipe.arena_bytes} bytes, against {recipe.naive_bytes} with no reuse"),
        (
            "passes",
            f"{sum(1 for c in report.changes if c.changed)} of {len(report.changes)}"
            f" changed something",
        ),
    ]
    width = max(len(label) for label, _ in rows)
    lines = [f"{label:<{width}}   {value}" for label, value in rows]
    if not recipe.kernels and report.target == cpu:
        lines.append("")
        lines.append("No kernels because the target is the host reference. Pass a gpu to see any.")
    return "\n".join(lines)


def _graph(report: Report) -> str:
    return report.imported


#: Changed lines to show per pass before summarising the rest.
_DIFF_LIMIT = 4


def _passes(report: Report) -> str:
    parts = []
    for change in report.changes:
        if not change.changed:
            parts.append(f"{change.name}: nothing changed")
            continue
        edits = change.diff.splitlines()
        # Counted on the lines that came out, because a pass that rewrites one line produces one.
        written = sum(1 for line in edits if line.startswith("+"))
        head = f"{change.name}: {written} line(s) rewritten"
        shown = edits[: _DIFF_LIMIT * 2]
        rest = written - sum(1 for line in shown if line.startswith("+"))
        body = "\n".join("  " + line for line in shown)
        parts.append(head + "\n" + body + (f"\n  ... {rest} more" if rest > 0 else ""))
    return "\n\n".join(parts)


def _kernels_by_kind(recipe: Recipe) -> list[tuple[str, Any]]:
    """Every kernel with the node it writes, in a stable order."""
    return sorted(recipe.kernels.items())


def _covered_by_kernels(recipe: Recipe) -> dict[str, str]:
    """The node each kernel computes, mapped to the kernel that computes it.

    A fusion group covers every operation in it, not only the one whose value escapes.
    """
    return {
        member: name
        for name, kernel in recipe.kernels.items()
        for member in (getattr(kernel, "members", None) or [name])
    }


def _kernels(report: Report) -> str:
    recipe = report.recipe
    if not recipe.kernels:
        return "none"

    parts = []
    for name, kernel in _kernels_by_kind(recipe):
        kind = kernel.kind
        covered = getattr(kernel, "members", None) or [name]
        extents = "at the launch" if getattr(kernel, "has_dynamic_extent", False) else "known here"
        parts.append(
            f"%{name}  {kind}, extents {extents}\n"
            f"    reads   {', '.join(kernel.inputs)}\n"
            f"    covers  {', '.join(covered)}"
        )
    body = "\n".join(parts)

    if report.source:
        through = run_passes(report.imported, ",".join(report.pipeline))
        body += "\n\n" + emit_cuda(through)
    return body


def _schedule(report: Report) -> str:
    """Every step in the order it runs, and what happens to it.

    A step is covered by a kernel, placed on the device with no kernel (cheap if it only moves
    data, a host round trip otherwise), or on the host.
    """
    recipe = report.recipe
    covers = _covered_by_kernels(recipe)

    lines = []
    for node in report.graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        where = recipe.devices.get(node.name, _HOST)
        if node.name in covers:
            fate = f"kernel %{covers[node.name]}"
        elif where.type == DEVICE_NAME:
            fate = "interpreted on the device"
        else:
            fate = "host"
        moved = " (moves its inputs)" if node.name in recipe.needs_move else ""
        lines.append(f"  {node.name:<24} {node.target!s:<34} {fate}{moved}")
    return "\n".join(lines)


_SECTIONS = {
    "summary": _summary,
    "graph": _graph,
    "passes": _passes,
    "kernels": _kernels,
    "schedule": _schedule,
}
