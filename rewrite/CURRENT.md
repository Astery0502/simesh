# Current Rewrite Checkpoint

## State

- Active milestone: M1 Cartesian 3D refined AMR.
- Last completed capability: HAX-001 explicit plan-consuming same-level halo application.
- Current capability: re-audit and split the preserved TOP-002 drafts before implementation.
- Rewrite implementation: isolated `simesh_rewrite` package with the complete non-periodic Cartesian 3D level-1 numerical and bounded-memory path, functional block substitution, and explicit flat refined-octree reconstruction with dense leaf/SFC maps.
- Stable contracts: `contracts/FND-001.md`, `contracts/FND-002.md`, `contracts/MIG-001.md`, `contracts/PERF-001.md`, `contracts/MOR-001.md`, `contracts/TOP-001.md`, `contracts/FST-001.md`, `contracts/GEO-001.md`, `contracts/STO-001.md`, `contracts/STO-002.md`, `contracts/STO-003.md`, `contracts/HAL-001.md`, `contracts/HAL-002.md`, `contracts/HPL-001.md`, `contracts/PBC-001.md`, `contracts/HAX-001.md`, `contracts/SAM-001.md`, `contracts/SAM-002.md`, `contracts/SAM-003.md`, `contracts/OPR-001.md`, `contracts/OPR-002.md`, `contracts/RED-001.md`, `contracts/INT-001.md`.
- Unresolved differences: HAL-002's Red finding is resolved and it is retained Fused over HPL/PBC/HAX semantics.  The preserved TOP draft is the only Red finding and remains paused until its FST conformance, contact semantics, BAL-001, and optional materialization decisions are split.  STO-002 is Yellow before refined support planning; SAM-002/SAM-003 are retained Fused until refined sampling.  The only available real refined Cartesian 3D `.dat` is staggered, so it remains forest/tree metadata evidence without broadening payload support.

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
- RED-001 uses one caller-owned binary64 state and fixed slot/x/y/z additions; persistent ascending no-closure chunks are bit-identical across capacities.
- Partial merge trees are explicitly strategy-dependent because binary64 addition is not associative; NumPy and `math.fsum` are descriptive/accuracy references, not alternate contract results.
- Empty updates perform no state write, while signed zero, NaN, infinity, and overflow follow the complete sequential IEEE stream without skipping or early exit.
- INT-001 uses separate no-closure and full-halo passes with one padded payload, one ID vector, one reused output workspace, and one persistent reduction scalar.
- Managed raw bytes are exactly `8*S*(F*Pvol+Bvol+1)+8`; source mapping/page cache, fixed controls, topology, and five final outputs are outside that budget and reported separately.
- Full semantic topology and actual-ID dry-plan validation complete before any result mutation; successful execution fully overwrites every sink/grid.
- External storage/framework state belongs behind coarse-grained functional
  reader/writer adapters; numerical kernels receive only canonical buffers and
  explicit metadata.
- A block adapter is frozen explicit state, canonical shape, one transfer
  function, and exposed memory-alias arrays; no global backend registry or
  hidden dispatch is used.
- Python adapter functions run once per planned transfer and never inside
  Cython cell, stencil, or block hot loops.
- `execute_level1_m0(...)` remains the stable array/memmap compatibility
  surface and delegates to `execute_level1_m0_from_blocks(...)`.
- Full-capacity resident and bounded functional strategies use the same
  numerical contracts and produce bit-identical M0 artifacts.
- HAL-001 and HAL-002 remain physical and level-1 same-level primitives; M1
  must separately compose access reach, support planning, relation
  classification, same-level copy, prolongation, and restriction.
- M0--M3 are the numerical-core axis, not the complete source migration.
  M4--M7 close scientific/derived behavior, I/O/export, dataset/public API,
  packaging/parallelism, and production cutover.
- Migration is tracked by supported feature disposition, parity tests, and
  public workflow evidence; files, classes, and source lines are not migration
  units.
- Every canonical `src/simesh` feature must end as rewrite, adapter, retain,
  replace, retire, or intentionally unsupported before cutover.
- High performance is a correctness-constrained multi-objective result across
  kernel, composition, real workflow, scaling, memory, and I/O measurements.
- Hot paths declare comparator, workload, hypothesis, metrics, and material
  regression before final optimization; raw runs stay ignored while compact
  benchmark evidence is committed.
- A refined 3D forest is a Boolean preorder stream containing exactly one
  complete octree per level-1 MOR rank; stream positions are node IDs and
  preorder leaves are canonical zero-based field-block IDs.
- Refined child columns are `x + 2*y + 4*z`; logical coordinates use
  `2*parent + child_bit`, levels start at one, and AMRVAC one-based conversion
  remains at the file boundary.
- FST-001 uses flat caller-owned parent/child, level/coordinate, root, and
  node/leaf maps.  It rejects malformed streams and unrepresentable complete
  level grids before output mutation and deliberately leaves two-to-one
  balance and neighbor classes to TOP-002.
- Every non-trivial capability passes the five questions in `DECOMPOSITION.md`:
  singular semantics, independent validation, independent substitution,
  independent measurement, and separation of semantic meaning from
  policy/execution.
- A function that makes an independently variable decision outside its stated
  semantic responsibility must split that decision into an explicit semantic,
  planning, policy, adapter, or execution boundary.
- Earlier nonconforming implementations are refined progressively when they are
  depended on, extended, substituted, optimized, integrated, found defective,
  or audited for cutover. Preserve behavior and compatibility wrappers, prove
  equivalence, measure, and migrate consumers incrementally; do not perform
  stylistic big-bang rewrites.
- Independent decomposition audit found FST-001 broadly conforming: forest
  reconstruction is one semantic responsibility, validation/fill are atomic
  phases, and its allocating/caller-buffered functions are valid separate
  boundaries. Do not reopen it wholesale. If multiple consumers need arbitrary
  FST artifact validation, add one reusable conformance boundary rather than
  duplicating structural validation.
- The uncommitted TOP-002 draft predates the decomposition gate. Before
  implementation, separately assess FST artifact revalidation, exact contact
  location/classification, global 26-direction two-to-one balance, and retained
  six-face topology materialization. Split independently variable decisions
  (for example BAL-001 versus TOP-002) instead of freezing them as one contract.
- The complete Phase A audit is `evidence/DECOMPOSITION-AUDIT.md`: sixteen
  capabilities are Green, STO-002 is Yellow, SAM-002/SAM-003 are retained
  Fused, and HAL-002 plus the in-progress TOP draft are Red.
- HAL-002's Red finding is semantic, not a correctness failure: preserve its
  tested direct-cell Cython entrypoint while extracting explicit relation/slot
  planning, pure same-level transfer, shared physical transforms, and a thin
  composition boundary.
- The TOP draft's FST conformance, raw contact lookup, global all-touch balance,
  and retained face cache vary independently.  Phase B will preserve its useful
  analysis but split those decisions; six kinds plus six `int64` IDs cost
  `54*L`, correcting the draft's `49*L` calculation.
- HPL-001 maps each primary's full x-fast 27-direction cube to an explicit
  selected source slot and three-bit physical-axis mask.  Center/pure physical
  directions use source `-1`; mixed directions retain their full direction and
  have both a valid source slot and physical mask.
- A full-direction plan-consuming HAL-002 hot-loop variant was bitwise correct
  but regressed multiple broad-halo kernels by more than 20%, so it was rejected.
  The public plan is retained for new composition while the existing optimized
  HAL-002 wrapper remains unchanged as the aggregate comparator.
- PBC-001 defines one explicit physical face/mode through a safe layer-based
  source index and an exact binary64 value transform.  Face eligibility and
  multi-axis order remain HAL responsibilities; aggregate Cython loops consume
  shared `inline noexcept nogil` rules without runtime callbacks.
- HAX-001 applies HPL plans without topology/global IDs and composes PBC rules
  in x/y/z order.  It is the simple substitutable level-1 value path; HAL-002
  remains the measured Fused wrapper because explicit checked plan+apply ranges
  from 0.49x to 1.31x its runtime across standard cases.

## Next Work

M0 and the pre-M1 functional/migration/performance protocols are complete, and
FST-001 reconstructs refined 3D hierarchy and leaf order.  Phase A decomposition
audit is complete and HAL-002's Red finding is resolved through HPL/PBC/HAX.
The next work re-audits/splits the preserved TOP drafts into
separately owned FST-conformance, contact, balance, and optional materialization
capabilities, followed by refined geometry, directional relation/support planning,
restriction/prolongation, refined halos, sampling, a native selective `.dat`
adapter, and real-data bounded integration.

## Latest Reproduction Commands

```text
.venv/bin/python rewrite/build_ext.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_hpl_001.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_pbc_001.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_hax_001.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_fst_001.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_sto_003.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_int_001.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests
PYTHONPATH=rewrite/src .venv/bin/python rewrite/benchmarks/sto_003.py --repeats 31
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/fst_001.py --repeats 15 --reference-limit 50000 --dat data/weno509_sub_0000.dat --dat-repeats 31 --current-repeats 5
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/hpl_001.py --repeats 31
PYTHONPATH=rewrite/src .venv/bin/python rewrite/benchmarks/pbc_001.py --calls 10000 --repeats 9
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/hax_001.py --repeats 21
```

Migration/performance audit evidence is recorded in `evidence/MIG-001.md` and
`evidence/PERF-001.md`. STO-003 composition evidence remains in
`evidence/STO-003.md`; FST-001 refined reconstruction evidence is in
`evidence/FST-001.md`; HPL-001 decomposition evidence is in
`evidence/HPL-001.md`; PBC-001 evidence is in `evidence/PBC-001.md`;
HAX-001 strategy evidence is in `evidence/HAX-001.md`; INT-001 and M0 evidence
remains in `evidence/INT-001.md`.
