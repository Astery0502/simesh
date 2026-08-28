# Rewrite Capability Decomposition Audit

## Scope And Method

This Phase A audit applies `DECOMPOSITION.md` to every completed capability and
the untracked TOP-002 design/contract drafts.  It changes no implementation or
stable contract.  The drafts `designs/TOP-002.md` and `contracts/TOP-002.md`
remain preserved, untracked, and unstaged.

Five-question results use `S/V/U/M/P`: singular semantic responsibility,
independent validation, independent substitution, meaningful measurement, and
separation of semantic meaning from policy/execution.  `Y*` denotes a justified
fused implementation whose simple semantics/reference remain independently
available.  Evidence abbreviations are `T` tests, `R` reference, `B` benchmark,
and `E` committed evidence.

## Capability Rows

| ID | One-sentence responsibility | Decisions owned | Decisions explicitly not owned | 5Q `S/V/U/M/P` | Classification and proposed split | Consumers, risk, and refinement trigger | Existing evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TOP-002 draft | As drafted, revalidate FST arrays, classify contacts, enforce global balance, and retain six face relations. | FST conformance; contact descent/kinds; all-touch level-gap policy; face-cache layout/codes. | Geometry; value transfer; edge/corner operation plans; support union/capacity; storage; periodicity. | `N/Y/N/N/N` | **Red**. Split into reusable FST conformance, exact contact queries, BAL-001, and optional face materialization; keep REL/STO planning later. | GEO must not inherit topology policy; REL must not inherit one balance/cache choice. **Immediate trigger before any TOP implementation.** | Current `forest.pyx`/tests and real WENO recovery; independent common-lattice/contact/balance references and benchmarks are planned, not yet implemented. |
| FST-001 | Reconstruct one valid AMRVAC Boolean-preorder octree per MOR root into explicit flat node/leaf arrays. | Stream grammar; node/leaf IDs; root/child order; levels/coordinates; flat relations; atomic construction errors. | Artifact consumers' topology/balance; geometry; storage; halos; file one-based conversion. | `Y/Y/Y/Y/Y` | **Green**; no split. Add a reusable conformance operation as a new FST-family boundary now that TOP is a second consumer. | GEO, contact lookup, native I/O; risk is divergent hand-written validators. Trigger: first external-array consumer, now reached by TOP. | T `test_fst_001.py`; R `forest_reference.py`; B `fst_001.py`; E `FST-001.md`. |
| FND-002 | Describe one input's spatial access and compute exact half-open required/valid regions. | Pattern taxonomy; tight asymmetric reach; expansion/contraction/support and overflow rules. | Operator formulas; topology relations; halo values/storage; chunk/scheduling policy. | `Y/Y/Y/Y/Y` | **Green**; no split. | OPR-001/002, RED-001, future support planners; low risk. Trigger a new capability if reach becomes relation-aware. | T `test_fnd_002.py`; R `access_reference.py`; B `fnd_002.py`; E `FND-002.md`. |
| STO-002 | Convert a byte budget to capacity and plan deterministic level-1 primary/support chunks. | Byte formula; ascending primaries; greedy maximality/promotion/order; no/face/full-halo closure policies. | Transfers; halo values; backend/RSS/file policy; operators. | `N/Y/Y/Y/Y*` | **Yellow**. Later separate workspace accounting, primary traversal, direct-face support, and full-halo support; retain wrappers. | RED uses no closure; HAL/SAM/INT use full closure. Risk is adding refined closure as another mode. **Trigger before STO-004/refined support planning.** | T `test_sto_002.py`; R `chunking_reference.py`; B `sto_002.py`; E `STO-002.md`. |
| STO-003 | Provide one explicit coarse-grained substitution seam for canonical block reads/writes. | Descriptor state/shape/callable/aliases; transfer granularity; STO-001 conformance; callback failure boundary. | Backend choice; file format; cache/async/transaction policy; chunking and numerics. | `Y/Y/Y/Y/Y` | **Green**; no split. | INT-001 and native `.dat` reader; medium risk if adapters misreport aliases/atomicity. Trigger backend-specific conformance at first native/cache adapter. | T `test_sto_003.py`, `test_int_001.py`; R array/STO-001 adapter; B `sto_003.py`; E `STO-003.md`. |
| HAL-001 | Apply explicit non-periodic physical-boundary rules to every physically eligible halo cell. | Physical eligibility from explicit sentinel; four boundary rules; normal fields; x/y/z transform order; produced validity. | Topology construction; name inference; sibling/coarse/fine/periodic semantics; workspace/scheduling. | `Y/Y/Y/Y/Y` | **Green**; no split. | HAL-002/INT and future HAL-003; mixed-order risk is frozen/tested. Reuse these semantics rather than add refined modes; 2D/periodic are separate capabilities. | T `test_hal_001.py`; R `fill_physical_halos_reference`; B `hal_001.py`; E `HAL-001.md`. |
| HAL-002 | Discover same-level sources and transfer/post-transform values to complete M0 primary halos. | Direction walks; closure/slot lookup; source coordinates; sibling/physical cases; physical post-transforms; prefix/support mutation. | Claims not to own chunk planning, refined transfer, periodicity, or allocation, but currently embeds relation planning and HAL-001 semantics. | `N/Y*/N/Y/N` | **Red**, semantically correct fused M0 path. Extract relation/source-slot plan, pure same-level copy, shared physical transform, and composition wrapper; retain optimized entrypoint. | SAM-003, OPR-002, INT-001. High extension risk: adding coarse/fine would create the prohibited dispatcher. **Refine first in Phase B before resuming TOP.** | T `test_hal_002.py`; R aggregate `fill_same_level_halos_reference`; B `hal_002.py`; E `HAL-002.md`. |
| INT-001 | Execute one concrete bounded M0 workflow by scheduling already contracted primitives under an exact temporary budget. | Two-pass schedule; workspace reuse/capacity; preflight/atomicity; stage order and complete result ownership; adapter wrapper. | Numerical/topology/halo/operator/reduction meaning; backend choice; file I/O; concurrency/cache/persistent state. | `Y/Y/Y/Y/Y` | **Green**, justified large executor; no split. | M0 integration/public facade; risk is accidental growth into an M1 dispatcher. Trigger reusable plan/preflight extraction only when a second executor duplicates it. | T `test_int_001.py` full-resident composition; B `int_001.py`; E `INT-001.md`. |
| SAM-001 | Place selected level-1 interiors into exact integer native-grid boxes. | Integer placement; translated copy; duplicate last-write; preservation/errors. | Selection; allocation/chunking; physical coordinates; interpolation/refined ownership; backend/parallel policy. | `Y/Y/Y/Y/Y` | **Green**; no split. | SAM-002 comparator and INT-001; low risk. Trigger only refined overlap, new layout/backend, or parallel public placement. | T `test_sam_001.py`; R placement reference; B `sam_001.py`; E `SAM-001.md`. |
| SAM-002 | Apply piecewise-constant sampling under one canonical output-center/source-cell owner rule. | Center/face operation order; highest-face tie; cell/block/local owner; exact copy/duplicate behavior. | Field selection; allocation/capacity; halos; refinement/exterior/backend/parallel policy. | `Y*/Y/Y/Y/Y` | **Fused**, retain. At refined sampling extract shared canonical owner and optional selected-block range planner; keep wrapper. | SAM-003/INT; high tie/coincident-face sensitivity but strong evidence. **Trigger SAM-004/refined owner or second cache/parallel strategy.** | T `test_sam_002.py`; R zero-order/owner reference; B `sam_002.py`; E `SAM-002.md`. |
| SAM-003 | Trilinearly interpolate canonically owned centers from a completed one-layer workspace. | Axis stencil/index/weight; exact reach; eight loads; z/y/x blend tree; duplicate/preservation behavior. | Owner rule; geometry; halo values; field/chunk/allocation/backend policy. | `Y*/Y/Y/Y/Y` | **Fused**, retain. Expose reusable stencil/blend semantics and consume shared owner at refined-sampling trigger; preserve fused wrapper. | INT-001; medium risk that refined ownership changes while blend semantics should not. **Trigger SAM-005 or cached/parallel strategy.** | T `test_sam_003.py`; R trilinear/lerp reference; B `sam_003.py`; E `SAM-003.md`. |
| OPR-001 | Compute one scaled pointwise difference over an explicit translated region. | Field/region mapping; exact scalar gate; multiply-then-subtract order; zero reach; mutation/errors. | Names/dependencies; allocation/chunk/storage fusion; registry/scheduling/in-place policy. | `Y/Y/Y/Y/Y` | **Green**; no split. | INT and M4 derived recipes; low risk. New multi-output/in-place/fused backing work is a separate strategy. | T `test_opr_001.py`; R `operators_reference.py`; B `opr_001.py`; E `OPR-001.md`. |
| OPR-002 | Compute one centered first derivative along one explicit Cartesian axis. | Sparse axis reach; spacing; arithmetic order; field/region mapping; mutation/errors. | Term batching; halo production; geometry; chunk/support/fusion/scheduling. | `Y/Y/Y/Y/Y` | **Green**; no split. | INT and M4 diagnostics; medium numerical-order risk is frozen. Trigger a separate batch/term planner when real operators require it. | T `test_opr_002.py`; R `operators_reference.py`; B `opr_002.py`; E `OPR-002.md`. |
| RED-001 | Implement a fixed-order one-scalar binary64 streaming-sum state machine. | State/update order; merge/finalize algebra; empty/IEEE/nonassociative semantics. | Chunk/merge tree; compensated/pairwise/SIMD/parallel strategy; weighting/geometry/storage. | `Y/Y/Y/Y/Y` | **Green**; no split. | INT; low-bit/accuracy risk explicit. Trigger a new strategy capability, not a refactor, for compensated/parallel reduction. | T `test_red_001.py`; R `reductions_reference.py`; B `red_001.py`; E `RED-001.md`. |
| MOR-001 | Assign every level-1 Cartesian coordinate its exact dense Morton rank and inverse. | Mathematical order; gap filtering; output maps; representability/atomicity. | Forest/topology/geometry/refinement/storage. | `Y/Y/Y/Y/Y` | **Green**; no split. | TOP/FST; low risk. Boundary validation in consumers is allowed; add public conformance only if multiple external map producers appear. | T `test_mor_001.py`; R `morton_reference.py`; B `mor_001.py`; E `MOR-001.md`. |
| TOP-001 | Materialize exact non-periodic level-1 six-face neighbor IDs. | Face order/displacements; physical sentinel; reciprocity; compact table. | MOR ordering; refinement/periodicity; geometry; halo/storage/chunk policy. | `Y/Y/Y/Y/Y` | **Green**; no split. MOR inverse checks are owned input validation. | GEO/STO/HAL; low risk. New refined/periodic topology remains separate. | T `test_top_001.py`; R `topology_reference.py`; B `top_001.py`; E `TOP-001.md`. |
| GEO-001 | Map selected level-1 IDs to canonical Cartesian bounds, spacing, and center semantics. | Dyadic-free level-1 face/spacing formulas; endpoints/shared faces; representability; selected geometry. | Topology; storage; halos; sampling ownership; refinement. | `Y/Y/Y/Y/Y` | **Green**; no split. | HAL/SAM and GEO-002 reference; numerical representability risk is explicit. Trigger separate GEO-002 for refinement and dimension-specific work for M2. | T `test_geo_001.py`; R `geometry_reference.py`; B `geo_001.py`; E `GEO-001.md`. |
| FND-001 | Define the canonical 3D index/layout/ownership/region interchange conventions and their primitive transforms. | Axes/IDs/layout; valid-region/ownership rules; ravel/unravel/interior/copy semantics. | Mesh topology; geometry; numerical operators; backend/scheduling policy. | `Y/Y/Y/Y/Y` | **Green**; functions already separate coherent primitives; no split. | All core capabilities; broad but stable protocol risk. Trigger new dimension/layout contracts rather than silently mutate M0 conventions. | T `test_fnd_001.py`; R `foundation_reference.py`; B `fnd_001.py`; E `FND-001.md`. |
| STO-001 | Transfer explicit selected block/field regions between resident canonical storage and caller buffers. | Gather/scatter mapping; selector/region translation; duplicate order; bit preservation/atomicity. | Budget/chunking; file I/O; workspace lifetime; topology/halo/operator semantics. | `Y/Y/Y/Y/Y` | **Green**; gather/scatter are paired separate primitives; no split. | STO-002/003, SAM/OPR/INT; low risk. Trigger alternative storage through STO-003, not changes here. | T `test_sto_001.py`; R `storage_reference.py`; B `sto_001.py`; E `STO-001.md`. |
| MIG-001 | Define the feature migration unit, statuses/dispositions, milestone mapping, and final cutover gate. | Behavior-as-unit ledger; closure dispositions; feature/workflow evidence requirements. | Numerical/API implementation; runtime scheduling; performance formulas. | `Y/Y/Y/N/A->milestone/Y` | **Green** policy capability; no split. Isolated runtime timing is meaningless; effectiveness is measured by milestone/final gate closure. | All M1-M7/cutover; risk is unclosed/inventoried rows. Trigger updates as each feature family activates. | R `SOURCE_MIGRATION.md`; E `MIG-001.md`; no kernel test/benchmark by design. |
| PERF-001 | Define benchmark levels, comparators, recording, and material-regression gates. | Profiles/metrics/baselines; recording; correctness/performance gate policy. | Specific optimization/backend/runtime; absolute universal timings. | `Y/Y/Y/N/A->reports/Y` | **Green** policy capability; no split. Cost is observed through capability/milestone benchmark conformance. | Every hot path and cutover; risk is incomparable or incomplete evidence. Trigger only when real benchmark practice disproves the protocol. | R `PERFORMANCE.md`; E `PERF-001.md` plus all retained benchmarks; no isolated kernel test by design. |

## Expanded Findings

### TOP-002 Draft — Red And Blocking

The draft responsibility joins four independently variable decisions:

1. reusable conformance of caller-supplied FST arrays;
2. exact contact location/classification for an arbitrary direction;
3. global face/edge/corner two-to-one admissibility;
4. retention of a six-face cache and its kind/ID layout.

Later edge/corner operation records and support closure are a fifth and sixth
decision already excluded by the draft and must remain REL/STO responsibilities.
The correct Phase B boundaries are:

- an FST-family conformance operation returning the exact maximum level;
- TOP-002 as exact batched contact queries that accept balanced or unbalanced
  forests and return the raw covering target node (or physical sentinel);
- BAL-001 as the explicit global all-touch rule
  `abs(level_a-level_b) <= 1`;
- optional TOP-003 six-face materialization, consuming contact semantics and a
  BAL-001 precondition, only if a consumer/benchmark justifies retaining it;
- later REL-001 edge/corner/mixed-physical operation planning and STO-004
  support/capacity policy.

The drafts also contain a concrete retained-memory error: six `uint8` kinds
plus six `int64` node IDs occupy `54*L`, not `49*L`, bytes.  A synthetic
read-only probe found a forest with maximum face level gap one but edge/corner
gap two; current connectivity misclassified the deep interior diagonal as
physical.  Face topology therefore neither proves nor owns the separate global
balance policy.

Risk if left combined is structural: GEO would gain an unnecessary topology
policy dependency, REL would inherit one balance/cache representation, support
planning could mistake six faces for complete directional semantics, and a
single benchmark could not attribute conformance, lookup, balance, or cache
cost.  No TOP implementation may begin until Phase B preserves the useful
analysis while splitting these boundaries.

### HAL-002 — Red, Preserve The Optimized M0 Path

HAL-002's output is correct and extensively frozen, but one call currently:

- discovers diagonal topology relations;
- validates closure and maps global IDs to selected slots;
- chooses same-level source coordinates and transfers values;
- reimplements HAL-001 physical coordinate/value transformations at mixed
  cells;
- owns primary/support mutation and validity.

Topology discovery, source/slot planning, same-level transfer, and physical
value application can vary independently.  Phase B must first freeze the
existing suite/benchmark, then extract one decision at a time from the outside
in.  Keep `fill_same_level_halos(...)` as the compatibility/composed wrapper
and retain its direct-cell Cython implementation when it remains materially
faster.  Prove old/new/reference equivalence, including the documented
noncommuting current divergence, before migrating any M0 consumer.

This refinement is not permission to add refined modes to HAL-002.  Restriction,
prolongation, refined relation plans, and periodic behavior remain new
capabilities.

### STO-002 — Yellow

STO-002 is semantically correct and its public functions are already mostly
separable, but the capability record groups byte accounting, primary traversal,
and three independently variable support policies.  Do not reopen it now.
Before STO-004, expose explicit no-closure/direct-face/full-halo policies and a
shared primary traversal/capacity boundary, retaining existing wrappers and
measurements.  Refined support must not become another mode flag.

### SAM-002 And SAM-003 — Fused, Retain

The block-centric loops validly fuse owner-range search with value work and
avoid resolution-sized scratch.  Their scalar/reference semantics are frozen,
so replacing them with slower runtime composition would be counterproductive.
At refined sampling, extract a shared canonical owner boundary and reusable
trilinear stencil/blend semantics, then compare the retained fused kernels with
the simple composition.  Until that concrete trigger, both remain unchanged.

## Phase B Order

1. Refine the completed Red HAL-002 incrementally, preserving its entrypoint,
   behavior, and performance evidence.
2. Re-audit and split the preserved TOP-002 drafts into FST conformance,
   contact, balance, and optional face-materialization capabilities.
3. Update capability IDs/dependencies and only then resume M1 implementation.

No other completed capability blocks TOP work.  Green capabilities remain
unchanged; Yellow/Fused triggers above are durable requirements for their later
consumers and final cutover audit.

## Phase B HAL-002 Resolution

The Red HAL-002 finding is resolved incrementally without replacing the fast
entrypoint: HPL-001 extracts relation/source-slot planning, PBC-001 extracts
physical coordinate/value rules, and HAX-001 extracts explicit plan application.
Old/fused, new/composed, and independent references agree bitwise; kernel and
composition trade-offs are recorded in their evidence.  HAL-002 is now retained
Fused rather than Red.  The in-progress TOP draft is the only remaining Red
finding and is next for boundary re-audit/split.

## Phase B TOP Resolution

The preserved TOP draft has been re-audited and split before implementation:
FST-002 owns reusable flat-artifact conformance, TOP-002 owns raw contact lookup,
BAL-001 owns global all-touch admissibility, and optional TOP-003 owns only
six-face cache materialization.  Original draft analysis is preserved under
`designs/TOP-003-*`; the incorrect `49*L` formula is superseded by `54*L`.
No Red capability remains.  FST-002, TOP-002, and BAL-001 are now complete as
independently validated and measured conformance, contact, and admissibility
functions.  TOP-003 remains an
optional materialization boundary rather than an implied continuation: it may
activate only when a concrete relation consumer and cached-versus-on-demand
measurement justify `54*L` retained bytes.  GEO-002 is the next
dependency-ready capability and does not inherit balance or cache policy.
