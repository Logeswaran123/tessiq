# Op definitions

Empty until phase 6.

Ops get defined here in YAML: aten schema, kernel name, supported dtypes and backends, shape rule,
fusion category. `tools/gen_ops.py` reads them and emits the dispatcher registrations, importer
entries, shape functions, lowering boilerplate, kernel registry and coverage docs.

Hand-writing four files per op stops scaling somewhere around op 40, which is why this is data
rather than code. It also means a future MLIR dialect can be generated from the same source
([PLAN.md §9.3](../PLAN.md)).
