# Current Rewrite Checkpoint

## State

- Active milestone: Foundation leading to M0.
- Last completed capability: OPR-002.
- Current capability: RED-001, not started and ready.
- Rewrite implementation: isolated `simesh_rewrite` package with topology/geometry, exact bounded storage, complete level-1 halos/sampling, and concrete pointwise/axis-stencil operators plus independent references.
- Stable contracts: `contracts/FND-001.md`, `contracts/FND-002.md`, `contracts/MOR-001.md`, `contracts/TOP-001.md`, `contracts/GEO-001.md`, `contracts/STO-001.md`, `contracts/STO-002.md`, `contracts/HAL-001.md`, `contracts/HAL-002.md`, `contracts/SAM-001.md`, `contracts/SAM-002.md`, `contracts/SAM-003.md`, `contracts/OPR-001.md`, `contracts/OPR-002.md`.
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
- HAL-002 maps each sibling-dependent primary halo cell directly from one selected interior, uses fixed 27-slot stack lookup, and never reads or mutates support halos.
- HAL-002 requires complete STO-002 one-block closure and halo widths no larger than the interior extent, then establishes the full padded valid box for the primary prefix.
- Mixed sibling/physical transforms retain HAL-001's deterministic x/y/z order; this intentionally differs from the current pass-ordered kernel only where no-inflow and antisymmetry do not commute.
- SAM-001 places a selected valid block region by exact integer cell boxes into field-first native uniform output; no floating geometry comparison is used.
- Reordered and repeated placement is deterministic in slot order, while bounded STO-002 primary traversal gives exactly-once disjoint native-grid coverage.
- SAM-002 defines output centers with separate binary64 operations and assigns exact face ties to the highest canonical native cell, giving one deterministic block owner.
- Sample bounds must lie inside the domain; outer-center-collapsed grids are rejected, while GEO-valid coincident internal faces use the explicit highest-face rule.
- SAM-001 native dispatch requires a separately amortized exact owner proof and is not repeated inside bounded SAM-002 chunk calls.
- SAM-003 reuses canonical ownership, uses current-style block-local center coordinates, and fixes a separately rounded z/y/x eight-corner blend tree.
- Trilinear input reach is exactly one valid lower and upper layer on every axis; all eight corners are read even when a weight is zero.
- Bounded trilinear execution samples only HAL-002-completed primaries after full-halo planning and gather/physical/sibling composition.
- OPR-001 is a concrete `left - scale*right` transform with explicit source/output fields and translated regions; scale accepts only exact binary64 scalar types.
- The pointwise access contract has exact zero reach, separate multiply/subtract rounding, no input/output overlap, and caller-owned output.
- Bounded no-closure gather/compute/scatter approaches whole-array NumPy throughput without its full product temporary, so no fused storage operator is retained.
- OPR-002 computes one axis-centered first derivative with exact reach one only on that axis and zero transverse reach.
- The stencil fixes inverse-spacing, neighbor subtraction, and multiplication order; current batching signed-zero behavior is deliberately outside the direct stencil.
- Bounded stencil execution uses full-halo primary chunks and separate output scatter; one valid halo layer is sufficient even though current dataset policy requests two.

## Next Work

Specify and implement RED-001 as one associative streaming reduction over
exactly-once no-closure primary chunks. Define accumulator/merge/finalize order,
empty and nonfinite behavior, selected field/region semantics, bounded memory,
current/NumPy comparison, chunk-size sensitivity, and deterministic serial
evidence without pretending floating addition is mathematically associative.

## Latest Reproduction Commands

```text
.venv/bin/python rewrite/build_ext.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_opr_002.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/opr_002.py --repeats 11
```

OPR-002 evidence is recorded in `evidence/OPR-002.md`.
