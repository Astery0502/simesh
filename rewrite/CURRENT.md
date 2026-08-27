# Current Rewrite Checkpoint

## State

- Active milestone: Foundation leading to M0.
- Last completed capability: FND-002.
- Current capability: MOR-001, not started and ready.
- Rewrite implementation: isolated `simesh_rewrite` package with local Cython foundation and access-region primitives plus independent references.
- Stable contracts: `contracts/FND-001.md`, `contracts/FND-002.md`.
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

## Next Work

Specify and implement MOR-001 for Cartesian 3D level-1 blocks. Recover the
current dense Morton ranking for both power-of-two and clipped rectangular
root grids, but do not introduce topology or forest state.

## Latest Reproduction Commands

```text
.venv/bin/python rewrite/build_ext.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_fnd_002.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests
PYTHONPATH=rewrite/src .venv/bin/python rewrite/benchmarks/fnd_002.py --iterations 100000 --repeats 9
```

FND-002 evidence is recorded in `evidence/FND-002.md`.
