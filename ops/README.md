# Op definitions

*Arrives in phase 6 — see [`PLAN.md` §6](../PLAN.md).*

This directory will hold the YAML op definitions that are the single source of truth for code
generation. `tools/gen_ops.py` reads them and emits dispatcher registrations, importer entries,
shape functions, lowering boilerplate, the kernel registry, and `docs/op_coverage.md`.

Defining ops as data rather than as hand-written C++ is also what keeps a future MLIR dialect
cheap: the same YAML can emit ODS/TableGen ([`PLAN.md` §9.3](../PLAN.md)).
