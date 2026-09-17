# Tools

| Tool | Purpose | Phase |
|---|---|---|
| `gen_ops.py` | Generates C++ registrations, shape functions and docs from `ops/*.yaml` | 6 |
| `tessera-opt` | Reads textual IR, runs a named pass pipeline, prints the result | 2 |
| `tessera-explain` | Shows a model's whole compilation: graph, per-pass diffs, kernel schedule | 7 |

`tessera-opt` is the one that makes the pass suite testable: passes are exercised as text in, text
out, with no PyTorch and no GPU involved. See [`PLAN.md` §5.1](../PLAN.md).
