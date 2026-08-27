# Current Rewrite Checkpoint

## State

- Active milestone: Foundation leading to M0.
- Last completed capability: MOR-001.
- Current capability: TOP-001, not started and ready.
- Rewrite implementation: isolated `simesh_rewrite` package with Cython foundation, access-region, and dense level-1 Morton primitives plus independent references.
- Stable contracts: `contracts/FND-001.md`, `contracts/FND-002.md`, `contracts/MOR-001.md`.
- Unresolved differences: none.

## Decisions Already Established

- Compatibility means supported numerical behavior, not API or byte identity.
- The order is 3D level-1, 3D refined, 2D, then periodic meshes.
- Staggered data is not currently supported.
- Kernels are added through normal Cython rebuilds; no dynamic kernel system.
- Data larger than available memory is a design target.
- Performance is multi-objective.
- Agents may revise contracts autonomously through independent sub-agent review.
- The workflow deliberately avoids hashes and heavy governance.
- Completed capabilities become cohesive Git checkpoints on the continuing rewrite branch.
- Non-trivial capabilities use a short design note before their stable contract; simple capabilities go directly to a concise contract.
- A capability starts only after its dependencies are complete unless the ledger records a narrower exception.
- Every capability checkpoint runs the accumulated rewrite regression suite and stages only capability-related files.
- Optimization stops after a correct integrated implementation reaches a useful trade-off and further credible variants no longer materially improve it.
- M0 spatial axes are `(x, y, z)` and rewrite-visible indices are zero-based signed `int64`.
- Canonical block interchange is C-contiguous `float64` in `(slot, field, x, y, z)` order; slots map to explicit global block IDs.
- Spatial regions are half-open boxes in storage coordinates and may express asymmetric halo storage.
- Inputs are borrowed read-only and callers own explicit output/workspace buffers; compiled boundaries do not repair layouts or allocate payload scratch.
- Rewrite extensions build only through `rewrite/build_ext.py`; root build and clean paths are not used for rewrite work.
- M0 access patterns are pointwise, fixed local stencil, and streaming reduction.
- Per-input halo requirements use tight asymmetric lower/upper `int64[3]` reach; a boolean `requires_ghosts` is not sufficient.
- Required-input expansion and valid-output contraction use exact half-open region algebra with explicit canonical empty boxes.
- Level-1 Morton rank is standard x/y/z bit-interleaved order filtered to the positive rectangular root grid and compacted to dense zero-based ranks.
- M0 uses Morton rank as the global level-1 block ID; arbitrary chunk slots remain distinct.
- Morton maps are caller-owned C-contiguous `int64` outputs with no raw-code width limit or heap scratch.

## Next Work

Specify and implement TOP-001 as validated non-periodic level-1 topology over
the MOR-001 maps. Establish exact face-neighbor IDs and physical-boundary
classification without adding refined, periodic, geometry, or halo behavior.

## Latest Reproduction Commands

```text
.venv/bin/python rewrite/build_ext.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_mor_001.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/mor_001.py --repeats 15 --reference-limit 50000 --current-shapes 32x32x32,31x29x27,33x33x33,64x64x64 --current-repeats 3
```

MOR-001 evidence is recorded in `evidence/MOR-001.md`.
