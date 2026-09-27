# Passes

A pass reads the IR graph and changes or annotates it. This page covers the pipeline a compile
runs, the other registered passes, running passes by hand, and adding one.

## The pipeline a compile runs

Each graph compiled with `backend="tessiq"` goes through these, in order:

```
rewrite-<name>      only the ones the compile asks for, in the order it names them
override-<name>     one per custom operation that replaces a torch one, sorted by name
mark-supported
layout-assign
cluster
fuse-elementwise
```

With what ships today, a compile that asks for every rewrite runs:

```
rewrite-rms-norm, rewrite-fast-gelu, rewrite-gemm-bias-gelu, rewrite-gemm-bias-gelu-tanh,
override-softmax, mark-supported, layout-assign, cluster, fuse-elementwise
```

The pipeline is assembled in `csrc/tessiq/torch/partition.cpp`: `kPipeline` is the fixed tail
from `mark-supported` down, `default_pipeline()` prepends the overrides (listed by
`override_passes()` in `custom.cpp`), and `prepare()` prepends the requested rewrites.
`tessiq.default_pipeline()` returns the first two; `tessiq.explain(model, x, rewrites=[...])` runs
all of it and prints each pass's changes.

### Why each one sits where it does

| pass | what it does | why here |
|---|---|---|
| `rewrite-<name>` | Replaces a computation written out op by op, such as RMSNorm, with one call to a custom operation. | First, so every later pass sees the custom operation and none sees the operations it absorbed. A pattern is written against the torch operations AOTAutograd emits, including one an override would replace. |
| `override-<name>` | Makes each node of a torch operation, such as `aten._softmax`, a node of the custom operation that replaces it, where the custom operation declares the types. | Before `mark-supported`, so the node is claimed as the custom operation. |
| `mark-supported` | Records on each node, as `tessiq.supported`, whether the backend can run it: the operation has a kernel, the dtypes are ones it takes, its extents are ones it can handle, and an elementwise or matmul node has no keyword argument, such as `alpha` or GELU's `approximate`, that its kernel would ignore. It changes nothing else. | Everything after it reads that annotation. It needs nothing before it. |
| `layout-assign` | Inserts a `tessiq.contiguous` copy wherever a claimed elementwise or reduction operation reads a value that is not packed. | After `mark-supported`, because only claimed readers need the copy. Before `cluster`, because the copy is a node that clustering has to see. |
| `cluster` | Groups neighbouring claimed nodes into clusters, which run on the device as a unit, and keeps the graph of clusters free of cycles. | Needs `tessiq.supported`. |
| `fuse-elementwise` | Within a cluster, merges chains of elementwise operations whose intermediate values have one reader into fusion groups. Each group becomes one kernel. | Needs `tessiq.cluster`. It is last because nothing after it changes the graph. |

After the pipeline the graph is fixed. The compiler plans memory and builds kernels: one per
fusion group, one per reduction, matmul or convolution, and one per custom-operation call.

Pass sources document their `Precondition:` and `Postcondition:`; the order above follows from
them.

### What a compile can change

Only which rewrites run, via `options={"rewrites": ["rms_norm"]}`. Overrides always run, and the
rest of the pipeline is fixed.

An elementwise custom operation such as `fast_gelu` called outside `torch.compile` compiles a
one-node graph with `mark-supported,cluster,fuse-elementwise`. `layout-assign` is skipped because
its operands are already packed.

## Passes that exist and do not run in a compile

| pass | what it does |
|---|---|
| `dead-node-elimination` | Removes nodes whose results nothing reads. |
| `cse` | Replaces a repeated computation with the first one. |
| `const-fold` | Evaluates scalar arithmetic on constants. |
| `algebraic-simplify` | Replaces an operation whose result is already available, such as `x * 1`. |
| `shape-infer` | Propagates shapes and dtypes, and checks the ones already recorded. |

They are registered and tested, and run when a pipeline names them.

## Running passes yourself

A pipeline is a comma-separated list of pass names. `tessiq-opt` runs one over textual IR:

```
$ build/tools/tessiq-opt --list
$ build/tools/tessiq-opt --passes=mark-supported,layout-assign,cluster,fuse-elementwise graph.tsr
$ build/tools/tessiq-opt --passes=mark-supported,cluster,fuse-elementwise --emit-cuda graph.tsr
```

`tessiq-opt` is built without PyTorch, so it has only the core passes. `override-<name>` and
`rewrite-<name>` are registered by the Python extension. From Python:

```python
import tessiq

graph = tessiq.capture(model, x)  # the FX graph a compile receives
ir = tessiq.import_graph(graph)  # the same graph as textual IR
tessiq.pass_names()  # every registered pass
tessiq.run_passes(ir, "rewrite-rms-norm,mark-supported")
tessiq.explain(model, x, sections=("passes",), rewrites=["rms_norm"])
```

The `passes` section of `explain` shows each pass's changed lines, which is the quickest way to see
why a node ended up where it did.

## Adding a pass

To replace a pattern with a custom operation, give the operation a `.pattern.tsr` file and
`.pattern()`; that registers `rewrite-<name>`. To make a custom operation stand in for a PyTorch
operation, add `.replaces("aten::<op>")`; that registers `override-<name>`. Both are covered in
[csrc/custom_ops/README.md](../csrc/custom_ops/README.md) and need no changes outside the
operation's directory.

For anything else, write a C++ pass:

1. Create `csrc/tessiq/passes/<name>.cpp` with a class that derives from `Pass` and implements
   `name()`, `description()` and `run()`. `run()` returns whether it changed the graph. Above the
   class, write what the pass assumes on entry, what it guarantees on exit, and why the rewrite
   keeps the graph computing the same thing:

   ```cpp
   /// Removes nodes whose results nothing reads.
   ///
   /// Precondition: use lists agree with operands, which the verifier checks.
   /// Postcondition: every remaining node has at least one result that is used or returned.
   ///
   /// Legality: removing a node whose result nobody reads cannot change what the graph computes.
   class DeadNodeElimination final : public Pass {
    public:
     [[nodiscard]] std::string_view name() const override { return "dead-node-elimination"; }
     [[nodiscard]] std::string_view description() const override {
       return "remove nodes whose results are never used";
     }
     Result<bool> run(ir::Graph& graph) override;
   };

   TESSIQ_REGISTER_PASS(DeadNodeElimination)
   ```

2. Add the file to `tessiq_ir` in `csrc/CMakeLists.txt`. The library is linked whole, so the
   registration survives even though nothing calls the pass by symbol.
3. Add golden files in `tests/cpp/golden/`, named `<pass>_<what the case shows>.tsr`. Each is a
   `// RUN:` line naming a pipeline, an input graph, `// ---`, and the output exactly as it prints.
   The test runner finds them on its own. Include cases where the pass must leave the graph alone.
4. Rebuild. The pass is now in `tessiq-opt --list` and `tessiq.pass_names()`, and any pipeline can
   name it.
5. To have every compile run it, add it to `kPipeline` in `csrc/tessiq/torch/partition.cpp`, at the
   place its precondition needs, and say why there in the comment.

Passes are C++17 and don't include PyTorch. Debug builds run the verifier after every pass, so a
pass that leaves the graph malformed fails right there.
