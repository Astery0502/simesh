# Current Rewrite Checkpoint

## State

- Active milestone: M1 Cartesian 3D refined AMR.
- Last completed capability: TGT-001 directed halo target boxes.
- Current capability: RPH-001 refined relation child-phase codes, proposed.
- Rewrite implementation: isolated `simesh_rewrite` package with the complete non-periodic Cartesian 3D level-1 numerical and bounded-memory path, functional block substitution, and explicit flat refined-octree reconstruction with dense leaf/SFC maps.
- Stable contracts: `contracts/FND-001.md`, `contracts/FND-002.md`, `contracts/MIG-001.md`, `contracts/PERF-001.md`, `contracts/MOR-001.md`, `contracts/TOP-001.md`, `contracts/FST-001.md`, `contracts/FST-002.md`, `contracts/TOP-002.md`, `contracts/BAL-001.md`, `contracts/REL-001.md`, `contracts/GEO-001.md`, `contracts/GEO-002.md`, `contracts/WSP-001.md`, `contracts/PRI-001.md`, `contracts/FCL-001.md`, `contracts/HCL-001.md`, `contracts/STO-001.md`, `contracts/STO-002.md`, `contracts/STO-003.md`, `contracts/STO-004.md`, `contracts/RST-001.md`, `contracts/LIM-001.md`, `contracts/PRL-001.md`, `contracts/RSL-001.md`, `contracts/TGT-001.md`, `contracts/HAL-001.md`, `contracts/HAL-002.md`, `contracts/HPL-001.md`, `contracts/PBC-001.md`, `contracts/HAX-001.md`, `contracts/SAM-001.md`, `contracts/SAM-002.md`, `contracts/SAM-003.md`, `contracts/OPR-001.md`, `contracts/OPR-002.md`, `contracts/RED-001.md`, `contracts/INT-001.md`.
- Unresolved differences: no Red capability remains.  HAL-002 is retained Fused over HPL/PBC/HAX semantics, and the TOP draft is split into FST-002 conformance, TOP-002 raw contact lookup, BAL-001 admissibility, and optional TOP-003 materialization.  The STO-002 Yellow finding is resolved by WSP/PRI/FCL/HCL with wrappers retained.  SAM-002/SAM-003 remain Fused until refined sampling.  The only available real refined Cartesian 3D `.dat` is staggered, so it remains forest/tree metadata evidence without broadening payload support.

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
- HAL-002 requires complete HCL-001 one-block closure and halo widths no larger than the interior extent, then establishes the full padded valid box for the primary prefix.
- Mixed sibling/physical transforms retain HAL-001's deterministic x/y/z order; this intentionally differs from the current pass-ordered kernel only where no-inflow and antisymmetry do not commute.
- SAM-001 places a selected valid block region by exact integer cell boxes into field-first native uniform output; no floating geometry comparison is used.
- Reordered and repeated placement is deterministic in slot order, while bounded PRI-001 primary traversal gives exactly-once disjoint native-grid coverage.
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
- The pre-decomposition TOP draft combined FST conformance, raw contact lookup,
  global all-touch balance, and retained face caching.  Phase B preserved its
  analysis under `designs/TOP-003-*` and split those decisions into FST-002,
  TOP-002, BAL-001, and optional TOP-003; a six-face cache costs `54*L`, not the
  draft's `49*L`.
- At the Phase A checkpoint, sixteen capabilities were Green, STO-002 Yellow,
  SAM-002/SAM-003 retained Fused, and HAL-002 plus the TOP draft Red.  HPL/PBC/
  HAX resolved HAL-002 to retained Fused, and the TOP re-audit split its Red
  boundary before implementation.  `evidence/DECOMPOSITION-AUDIT.md` records
  both resolutions.
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
- TOP boundary re-audit preserves the original cache drafts under
  `designs/TOP-003-*` and assigns independent decisions to FST-002 conformance,
  TOP-002 raw contact lookup, BAL-001 all-touch balance, and optional TOP-003
  face materialization.  REL/STO retain operation/support planning.
- FST-002 validates flat preorder/root/child/level/coordinate/leaf-map
  conformance once per unchanged artifact lifecycle and returns exact maximum
  level.  MOR mathematical key order remains an upstream MOR-001 precondition;
  FST-002 proves dense inverse/root-row alignment without overclaiming it.
- TOP-002 returns raw adjacent covering nodes for explicit leaf/direction
  queries after one FST-002 lifecycle validation.  It accepts unbalanced
  contacts, assigns no kind, retains no cache, and leaves balance/mixed
  operation/support policy to BAL/REL/STO.
- BAL-001 checks all 26 x-fast directions for every canonical leaf and owns only
  the closed-box face/edge/corner level-gap policy.  A coarser target may differ
  by one level; a subdivided target is admitted only when its four/two/one
  direction-touching immediate children are leaves.
- BAL-001 reuses TOP-002's shared inline contact primitive, returns no token or
  cache, and retains no leaf-sized state.  Its result belongs to the same
  unchanged FST-002 artifact lifecycle and must be re-established after
  mutation.
- Optional TOP-003 remains deferred until an immediate relation consumer shows
  that retaining `54*L` bytes of faces materially improves the measured
  runtime-memory-complexity trade-off over on-demand contacts.
- GEO-002 maps explicit canonical leaf IDs to caller-owned bounds and
  slot-aligned spacing through level-global integer cell-face indices and
  `ldexp` dyadic scaling.  It reduces bitwise to GEO-001 at level one, snaps
  outer endpoints, and gives bit-identical same/coarse/fine shared faces.
- GEO-002 validates only selected numerical geometry after one unchanged
  FST-002 lifecycle.  It retains no full table or centers, does not require
  balance/topology, and leaves center representability and owner/tie policy to
  refined sampling.
- GEO-002 output is exactly `72*S` bytes.  The real checked pass is `0.621 ms`
  for 22,614 leaves; a 4,096-leaf bounded pass reuses 294,912 bytes.  All
  140,870 expanded real face comparisons are bit-identical, versus 98,342
  current `rnode` mismatches.
- REL-001 emits kind, three-bit physical mask, exact source count, and up to
  four canonical source leaf IDs for a selected leaf-by-direction product.
  Physical components are neutralized before TOP lookup; pure physical records
  have no external source, while mixed records retain their in-domain relation.
- FINER records filter immediate children with the reduced direction in
  canonical child order.  Physical-neutralized axes admit both phases; PBC
  mode/reach-specific narrowing belongs to later transfer planning.  REL owns
  no support union, selected-slot map, capacity, transfer formula, or cache.
- REL-001 output is exactly `35*P*D` bytes.  The real all-direction pass is
  `8.078 ms` for 587,964 records; a 1,024-leaf bounded buffer is 931,840 bytes.
  Current loses all 26,784 mixed physical/source records by calling them purely
  physical.  Optional TOP-003 remains deferred because faces cover only 24.8%
  of reduced real records and cache construction cannot amortize one pass.
- WSP-001 owns the exact canonical payload-plus-ID formula
  `8*S*(F*V+1)` and its block-clamped inverse.  It may return zero and owns no
  workspace allocation, traversal, progress, closure, backend/RSS, or
  executor-specific output/coarse/reduction scratch.
- `simesh_rewrite.workspace` is canonical; package-root and `chunking` paths
  remain bit/error-identical.  The 2 MiB composition still returns 31 slots and
  exactly 2,031,864 allocated bytes.  Forward/inverse calls remain about
  2.60/2.75 us with a fixed 1,053-byte traced peak.
- PRI-001 fills the maximal dense ascending primary prefix from explicit
  `first`, `block_count`, and caller capacity.  It owns end/zero-capacity,
  prefix/suffix, maximality, and exactly-once identity but no topology, budget,
  support, transfer, or scheduling.
- The legacy no-closure STO wrapper preserves every validation and returns
  `(count,count)` after delegation.  M0 dry preflight and pass one use PRI
  directly; all result arrays and reduction bits remain identical.  Canonical
  throughput is 110/438/1,404 million IDs/s at capacities 64/256/1,024, with
  zero retained traced bytes and a fixed 248-byte peak.
- FCL-001 jointly grows PRI candidates with their unique nonrecursive TOP face
  union.  It owns face0..5 discovery, promotion, stable support order, greedy
  maximality/first-fit, suffix/end atomicity, and the exact direct-face
  minimum-progress helper; it owns no diagonal/full/refined closure or payload.
- Canonical and legacy `True` paths retain identical 1,797/293/55 chunks and
  3.457/2.279/1.702 amplification at capacities 64/256/1,024.  Runtime remains
  about 3.85/2.20/0.94 million primaries/s with zero retained traced bytes and
  a fixed 576--600-byte peak.
- HCL-001 owns complete clipped level-1 one-block `3x3x3` closure: 26 x-fast
  directions, fixed x/y/z face walks, physical clipping, promotion/order,
  greedy maximality, and exact topology-specific progress up to 27 slots.  HAL
  still owns width/reach and values; refined REL-source union belongs STO-004.
- A fixed 26-ID C stack removes the redundant accepted-trial second walk while
  preserving every selected ID.  HCL reaches 1.59/0.80/0.39 million primaries/s
  at capacities 64/256/1,024, about 1.7--1.9x the recorded baseline, with zero
  retained traced bytes and a fixed 576--600-byte peak.  M0 pass-two artifacts
  and reduction remain bit-identical within the composed runtime gate.
- STO-004 greedily plans the maximal dense refined primary prefix whose ordered
  unique union of REL source leaves fits explicit selected capacity.  Exactly
  `min(capacity, remaining)` candidate rows prove global maximality; accepted
  primaries precede stable first-discovery support, and future support is
  promoted without disturbing the rest of the order.
- STO-004 validates the complete counts/source representation before mutation,
  keeps first-fit failure atomic, and exposes the exact bounded one-primary
  progress maximum.  Canonical balanced all-26 Cartesian 3D input has achieved
  universal bound 57; the real WENO forest's exact maximum is 53.
- The fixed-stack canonical planner retains no size-dependent scratch.  A
  bounded sliding REL/plan/gather composition generates each real relation row
  once, retains 52,326--940,032 metadata bytes at measured capacities, and is
  10.9--23.3% faster than exact-workload naive regeneration.  A whole-tree REL
  cache is rejected because its 20,578,740 bytes exceed current retained
  connectivity and are unnecessary for bounded execution.
- RST-001 maps an explicit even fine box to a translated half-size coarse box
  and evaluates current Cartesian 3D restriction as eight values in
  `000,100,010,110,001,101,011,111` order, seven additions, then exact
  multiplication by `0.125`.  It owns no relation, child phase, scratch,
  prolongation, or transfer policy.
- RST validates complete canonical payload/region/overlap state before
  mutation, accepts contained empty half-open axes, preserves every cell outside
  the requested output box, and retains no allocation.  Current eligible
  `datac` interiors, the scalar reference, checked/unchecked kernels, and the
  exact-order NumPy comparator are bitwise identical.
- Replacing volatile temporaries with ordinary explicit-statement locals keeps
  exact bits and improves the standard checked kernel about 5x to 0.199 ms,
  659.5 million coarse field-cells/s and 47.48 effective GB/s.  A call retains
  zero traced bytes and peaks at 1,056 bytes.
- Bounded RST composition extracts only accepted FINER source IDs, maps them to
  STO-selected slots outside the numerical kernel, and restricts compact
  canonical inputs.  At capacity 256 the exact 528-leaf path uses 3.78 MB
  working memory; Python source extraction dominates at 4.20 ms while gather
  and RST take 0.236/0.189 ms, so later transfer planning owns that optimization.
- LIM-001 freezes the current scalar limiter as separate `center-left` and
  `right-center` differences, their left-to-right sum and `0.5` multiplication,
  strict comparison-based sign/minimum branches, nonpositive clamp, and literal
  positive-zero fallback.  It is not a doubled-slope monotonized-central rule.
- All NaN inputs and every zero-return path produce exact positive zero;
  same-sign infinities may survive, centered overflow can still clip to a finite
  one-sided value, and limiting subnormals are preserved.  Production/reference
  outputs match across the complete branch/IEEE evidence matrix.
- LIM exposes one exact-type-validated scalar operation and one shared
  `inline noexcept nogil` rule for PRL.  It owns no axis, reach, eta, region,
  reconstruction, slope array, cache, or workspace decision.
- The descriptive scalar boundary reaches 2.525 million validated calls/s,
  versus 5.471 million unchecked and 0.660 million reference calls/s, with zero
  retained traced bytes and a 48-byte peak.  Meaningful inline throughput and
  the normal regression gate begin in PRL; no array slope strategy is retained.
- PRL-001 maps explicit fine indices through mathematical floor division and
  aligned origins, selects exact binary64 phase eta `+/-0.25`, reads only the
  mapped center and six axial coarse neighbors, applies shared LIM slopes, and
  reconstructs with separate x/y/z products and ordered additions.
- PRL requires the mapped center range expanded by one coarse cell on every
  axis to lie in the declared valid box, but never reads diagonals.  It mutates
  only the explicit fine target and owns no relation, source-slot, workspace
  assembly, physical, or scheduling decision.
- Exact phase intentionally removes current coordinate-rounding noise.  WENO's
  814,104 mapping cases have zero index differences, 27 current eta patterns,
  and maximum eta delta `2.842e-14`; dyadic current output is bitwise equal and
  non-dyadic output satisfies the frozen local slope/roundoff bound.
- Coarse-centric reuse of three LIM results for up to eight children preserves
  every child's products/additions and is 2.53x faster than fine-centric:
  225.5 million fine field-cells/s and 14.43 effective GB/s standard unchecked.
  Checked overhead is 1.2%, retained traced bytes zero, and peak 1,312 bytes.
- Resident assembly/RST/PRL takes 0.051/0.254/4.731 ms; capacities 1/8/64 remain
  bitwise exact and PRL reaches 197.5/209.4/214.2 million cells/s without
  relations or halo policy.  No slope array or parallel variant is retained.
- RSL-001 maps every active accepted-REL global source ID to its unique local
  STO-selected slot and writes exact `-1` for inactive columns.  It preserves
  `[P,D,4]` order/shape, requires accepted provenance `P<=S`, and interprets no
  kind, mask, action, target, workspace, or numerical policy.
- Checked RSL validates selected uniqueness, the full REL active/trailing
  representation, output separation, and complete membership before mutation;
  it therefore performs two linear lookup passes.  Production/reference/HPL
  reductions and WENO bounded roundtrips are exact without a hash, inverse map,
  sort, bitmap, or destroyed REL input.
- The standard `S=256,P=128,D=26` path resolves 6,653 sources in 0.817 ms
  checked/0.389 ms unchecked versus 17.406 ms for the list reference, retains
  zero traced bytes, and peaks at 848 bytes.  One lookup pass uses 854,305
  comparisons; full checked identity work is 1,741,250 comparisons.
- One 106,496-byte RSL artifact plus ten native scans is 9.31x faster than ten
  native repeated resolutions.  WENO peak slot bytes remain 19,136--549,952;
  larger capacities lower selected amplification but deepen per-pass lookup,
  leaving the capacity/action trade-off to later transfer planning.
- RAC-001 was retired before implementation because REL kind/mask already
  bijectively determines the base transfer action.  Combined RTP-001 then
  failed the five-question gate: direction boxes and refinement phases have
  disjoint inputs, outputs, validation, substitution, and reasons to change.
- TGT-001 is the extracted direction-box function.  For each direction axis it
  selects requested-lower/interior/interior-upper intervals, giving exact
  half-open face/edge/corner boxes.  Canonical 26 boxes are pairwise disjoint
  and cover the requested envelope outside the interior; asymmetric, empty,
  and valid singleton axes are supported.
- Canonical current level-one behavior matches all 26 TGT boxes on the interior
  leaf of a `3x3x3` forest with unique neighbor constants.  The unsafe current
  singleton physical-status table access is excluded, not valid rewrite input.
- Canonical TGT writes exactly 1,248 bytes in 5.708 us checked/1.000 us
  unchecked versus 29.333 us for the reference, with zero retained traced bytes
  and a 1,360-byte peak.  One native materialization plus 100,000 scans is 2.09x
  faster than repeated native derivation, so the tiny artifact is retained.

## Next Work

M0 and the pre-M1 functional/migration/performance protocols are complete, and
FST-001 reconstructs refined 3D hierarchy and leaf order.  Phase A decomposition
audit is complete and FST-002/TOP-002/BAL-001/GEO-002/REL-001 separately
establish conformance, raw contacts, all-touch admissibility, selected refined
geometry, and selected relation records.  The STO-002 Yellow trigger is resolved:
WSP-001 accounting, PRI-001 primary traversal, FCL-001 direct-face closure, and
HCL-001 full-halo closure, STO-004 refined support union/bounded planning, and
RST-001 ratio-two restriction, LIM-001 limiting, PRL-001 prolongation, and
RSL-001 accepted source-slot resolution are complete.  RAC-001 was retired as a
redundant rename of REL kind/mask, and combined RTP-001 split at the
five-question gate.  TGT-001 directed boxes are complete.  Next is RPH-001
refined child phases, followed by workspace/value application, complete refined
halos, sampling, a native selective `.dat` adapter, and real-data bounded
integration.

## Latest Reproduction Commands

```text
.venv/bin/python rewrite/build_ext.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_fst_002.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_top_002.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_bal_001.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_geo_002.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_rel_001.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_wsp_001.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_pri_001.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_fcl_001.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_hcl_001.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_sto_004.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_rst_001.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_lim_001.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_prl_001.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_rsl_001.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_tgt_001.py
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
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/fst_002.py --repeats 31 --reference-limit 50000 --dat data/weno509_sub_0000.dat
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/top_002.py --max-level 5 --repeats 15 --reference-limit 20000 --dat data/weno509_sub_0000.dat
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/bal_001.py --max-level 5 --repeats 15 --reference-leaf-limit 600 --dat data/weno509_sub_0000.dat
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/geo_002.py --max-level 5 --repeats 15 --reference-leaf-limit 600 --chunk-leaves 4096 --dat data/weno509_sub_0000.dat
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/rel_001.py --max-level 5 --repeats 15 --reference-leaf-limit 100 --chunk-leaves 1024 --dat data/weno509_sub_0000.dat
PYTHONPATH=rewrite/src .venv/bin/python rewrite/benchmarks/wsp_001.py --calls 10000 --repeats 15
PYTHONPATH=rewrite/src .venv/bin/python rewrite/benchmarks/pri_001.py --block-count 32768 --capacities 1,8,64,256,1024,32768 --repeats 15
PYTHONPATH=rewrite/src .venv/bin/python rewrite/benchmarks/fcl_001.py --capacities 64,256,1024 --repeats 15
PYTHONPATH=rewrite/src .venv/bin/python rewrite/benchmarks/hcl_001.py --capacities 64,256,1024 --repeats 15
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/sto_004.py --capacities 57,64,128,256,1024 --repeats 3 --dat data/weno509_sub_0000.dat
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/rst_001.py --repeats 31 --composition-repeats 7 --composition-capacities 64,128,256
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/lim_001.py --calls 1000000 --repeats 3
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/prl_001.py --repeats 5 --composition-repeats 3
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/rsl_001.py --repeats 5 --action-repeats 10 --weno-capacities 57,64,128,256,512,1024
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/tgt_001.py --repeats 9 --reuse-scans 100000 --reuse-repeats 5
```

Migration/performance audit evidence is recorded in `evidence/MIG-001.md` and
`evidence/PERF-001.md`. STO-003 composition evidence remains in
`evidence/STO-003.md`; FST-001 refined reconstruction evidence is in
`evidence/FST-001.md`; HPL-001 decomposition evidence is in
`evidence/HPL-001.md`; PBC-001 evidence is in `evidence/PBC-001.md`;
HAX-001 strategy evidence is in `evidence/HAX-001.md`; INT-001 and M0 evidence
remains in `evidence/INT-001.md`.  FST-002 evidence is in
`evidence/FST-002.md`; the TOP boundary split is recorded in
`evidence/TOP-BOUNDARY-REAUDIT.md`; TOP-002 evidence is in
`evidence/TOP-002.md`; BAL-001 evidence is in `evidence/BAL-001.md`; GEO-002
evidence is in `evidence/GEO-002.md`; REL-001 evidence is in
`evidence/REL-001.md`; WSP-001 evidence is in `evidence/WSP-001.md`.
PRI-001 evidence is in `evidence/PRI-001.md`.
FCL-001 evidence is in `evidence/FCL-001.md`.
HCL-001 evidence is in `evidence/HCL-001.md`.
STO-004 evidence is in `evidence/STO-004.md`.
RST-001 evidence is in `evidence/RST-001.md`.
LIM-001 evidence is in `evidence/LIM-001.md`.
PRL-001 evidence is in `evidence/PRL-001.md`.
RSL-001 evidence is in `evidence/RSL-001.md`.
TGT-001 evidence is in `evidence/TGT-001.md`; the rejected combined boundary is
recorded in `designs/RTP-001.md`.
