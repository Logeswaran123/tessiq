# ADR 0002: output structure belongs to the calling convention

- **Status:** accepted
- **Date:** 2026-09-18

## Context

AOTAutograd's backward graph returns one value per forward input: a gradient, or `None` where that
input takes no gradient. A `Linear` with a frozen bias produces this output node:

```
output args[0] = (t_3, None, None)
```

The importer used to require every element of that tuple to be an FX node, so it rejected the graph
and training failed on the first backward pass. It now skips `None`, which leaves the IR describing
only the values the graph computes:

```
FX output arity : 3 -> (t_3, None, None)

graph @backward(%primals_3: f32[2,4], %detach: f32[2,3], %tangents_1: f32[2,3]) -> f32[3,4] {
  ...
  return %t_3
}
```

Three outputs in the graph PyTorch handed over, one result in the IR. Execution is unaffected today,
because the executor walks the FX graph and passes non-tensors through untouched, so the tuple it
returns still has three slots in the original order.

That stops being true in phase 5. Once a `Recipe` is the thing that runs, the returned tuple is
built from the recipe rather than from FX, and the recipe is built from the IR. A recipe that
returns one value where the engine expects three is not a crash. The autograd engine matches
gradients to inputs by position, so the failure mode is a gradient landing on the wrong parameter.

Backward graphs are not a later concern that phase 8 can own. They reach the backend now, because
`aot_autograd` defaults `bw_compiler` to `fw_compiler` and tessiq passes only the latter. Whatever
phase 5 builds receives them immediately.

## Options

**A none or unit type in the IR.** The most faithful record: the graph would say it returns three
things, one of which is nothing.

`TensorType` is non-optional throughout the IR. A value carrying no shape and no dtype would have to
be handled by every pass, the verifier, the printer and the parser, and every golden file would
change. That is a large amount of surface to move in order to represent something that is not a
computation.

**Result positions stored on `Graph`.** Keep the results as they are and record the total arity plus
which slot each result fills.

Smaller, and exact. It still puts calling-convention data inside the IR, and the textual form has to
carry it or printing and parsing stop round-tripping.

**An output spec at the boundary.** Chosen.

## Decision

The IR records the values a graph computes. The shape of the object handed back to the caller is
calling convention, and it travels with the compiled callable rather than with the graph.

A `Recipe` carries an output spec captured at compile time from the FX output node: the total arity,
and which slot each IR result fills. Reassembling the tuple happens at the boundary, which is the
same place that already moves results back to the host.

This is what AOTAutograd itself does. Output metadata travels beside the graph, not inside it.

The spec is a required argument when a recipe is constructed, not an optional field with a default.
A forgotten spec produces a misassigned gradient rather than an error, so it must not be possible to
leave out.

## Consequences

`Graph::results()` means the values the graph computes, and nothing more. Anything reconstructing a
return value needs the spec as well, which is one more thing to thread through lowering.

The rule generalises past `None`. Nested tuples, dicts and any other structure PyTorch returns are
all calling convention, so the spec is the place they go, and the IR stays a list of typed values.

Nothing changes in phase 4. The executor walks the FX graph and already returns the right shape.

Revisit if the IR ever needs to represent a graph whose results are not all tensors for reasons
other than the calling convention, since that would be a genuine argument for option A.

## See also

`import_output` in `csrc/tessiq/torch/importer.cpp`, `compile_graph` in `python/tessiq/backend.py`,
phase 5 in `PHASES.md`.
