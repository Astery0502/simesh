# Current Rewrite Checkpoint

## State

- Active milestone: Foundation leading to M0.
- Last completed capability: HAL-001.
- Current capability: HAL-002, contract work ready after its STO-002 integration prerequisite.
- Rewrite implementation: isolated `simesh_rewrite` package with topology/geometry, exact bounded storage, and bit-exact physical-envelope halo provision plus independent references.
- Stable contracts: `contracts/FND-001.md`, `contracts/FND-002.md`, `contracts/MOR-001.md`, `contracts/TOP-001.md`, `contracts/GEO-001.md`, `contracts/STO-001.md`, `contracts/STO-002.md`, `contracts/HAL-001.md`.
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
- M0 topology stores only `(xlo,xhi,ylo,yhi,zlo,zhi)` face IDs in zero-based `int64`; exact sentinel `-1` denotes a physical face.
- Face topology structurally validates the supplied MOR maps before output mutation.
- Edge/corner relations are derived through commuting face steps instead of a 4.5x larger 27-direction table.
- Level-1 geometry is slot-aligned to explicit block IDs and emits selected `(slot,lower/upper,x/y/z)` bounds plus one global spacing triplet.
- Physical faces come from global integer cell-face indices, giving bit-identical sibling faces and exact supplied endpoints.
- Cell centers use global integer cell indices; center arrays and per-block spacing are not materialized.
- GEO rejects subnormal spacing and numeric domains whose outer canonical centers collapse onto domain faces.
- In-memory backing is canonical `(global_block,field,x,y,z)` C-contiguous `float64`; gathers/scatters use explicit block/field selectors and translated valid boxes.
- Storage transfers are caller-buffered and bit-exact; duplicate scatter targets use deterministic slot-major/field-slot last-write semantics.
- STO-001 uses slab, plane, or row copies without selector sorting, hidden allocation, or source/sink classes.
- Managed workspace bytes are exactly `8*S*(F*V+1)` for payload plus block IDs; capacity counts the full allocation, including unused final slots.
- Chunk primaries are maximal ascending contiguous ID prefixes; optional unique direct-face support follows in first-discovery order.
- Full-halo chunks preserve the same primary-prefix contract and add the clipped one-block `3x3x3` support closure in canonical x-fast direction order; capacity 27 guarantees progress.
- Primary coverage is exactly once, while support may recur; no hash, bitmap, or primary-slot map is used.
- Read-only C-order `numpy.memmap` backing demonstrates that full payload need not be eagerly resident, while OS page cache remains outside the managed budget.
- Physical halo modes are explicit per local field and face: continuous, symmetric, antisymmetric, and no-inflow with explicit normal-field slots.
- HAL-001 maps only from valid interior cells, composes incident rules x then y then z, and fills each slot's exact physical envelope in place.
- The common FND valid box is the intersection of slot envelopes; reflected depth beyond the interior and missing no-inflow metadata are rejected.

## Next Work

Specify and implement HAL-002 same-level sibling halo provision over TOP-001
faces and STO-002 full-halo-closed workspaces. Complete remaining halo cells
without changing interiors, preserve HAL-001 physical composition at mixed
edges, and declare a full padded valid region for processed primary slots.

## Latest Reproduction Commands

```text
.venv/bin/python rewrite/build_ext.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_sto_002.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests
PYTHONPATH=rewrite/src .venv/bin/python rewrite/benchmarks/sto_002.py --capacities 64,256,1024 --memmap-blocks 2048 --budget-mib 2
```

The independently reviewed STO-002 halo-closure extension is recorded in
`evidence/STO-002.md`.
