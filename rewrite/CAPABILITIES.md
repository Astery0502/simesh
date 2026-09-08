# Capability Ledger

This is the lightweight dependency ledger for the rewrite. Add detail only when
a capability becomes active.

[baseline.md](../docs/analysis-core/archive/core-development-2026-09-08/baseline.md) now owns the current requirement-to-capability/evidence
map and recommended first-outcome readiness. The statuses below retain their
original historical scope; `complete` does not mean preferred for new workloads,
publicly integrated, or a mandate to follow the old milestone order. The proposed
three-pipeline result scope supersedes the repeated-sampling phase suggestion.
F tracing/Q/twist, independent D global derivatives/slices and L full-domain LOS
have no newly selected implementation group or capability IDs; define only
demonstrated missing responsibilities after the selected outcome's entry gate.

The 2026-09-07 [intent catalog](../docs/analysis-core/archive/core-development-2026-09-08/workflows.md) expands user outcomes and
resident/bounded acceptance. Catalog entries are not capability IDs or completed
features. Slice, surface and volume products need future consumer-specific
contracts; existing M0/M1 and optimization statuses retain their original scope.
No implementation group is activated by the documentation change.

## Status Values

- `proposed` -- identified but not specified;
- `specified` -- contract and acceptance behavior are clear;
- `reference` -- independent reference and focused tests pass;
- `implemented` -- the simple Cython implementation passes;
- `integrated` -- composed with its immediate consumers;
- `complete` -- relevant correctness, performance, and memory evidence is recorded.

## Foundation And M0

| ID | Capability | Depends on | Status |
| --- | --- | --- | --- |
| FND-001 | Layout, index, ownership, and valid-region conventions | -- | complete |
| FND-002 | Operator access-pattern and halo-requirement vocabulary | FND-001 | complete |
| MIG-001 | Supported-feature migration ledger and final cutover gate | FND-001 | complete |
| PERF-001 | Benchmark levels, baseline, recording, and regression protocol | FND-001 | complete |
| MOR-001 | Cartesian 3D level-1 Morton mapping | FND-001 | complete |
| TOP-001 | Validated level-1 topology | MOR-001 | complete |
| GEO-001 | Cartesian 3D block bounds and spacing | TOP-001 | complete |
| STO-001 | In-memory block source and sink | FND-001 | complete |
| WSP-001 | Explicit workspace byte and slot-capacity accounting | FND-001 | complete |
| STO-002 | Bounded workspace and chunk compatibility wrappers | WSP-001, PRI-001, FCL-001, HCL-001, STO-001, TOP-001 | complete |
| STO-003 | Functional block reader/writer adapters and execution substitution | STO-001 | complete |
| HAL-001 | Non-periodic physical-boundary halo provision | GEO-001, STO-001 | complete |
| HAL-002 | Same-level sibling halo provision | TOP-001, HCL-001, HAL-001 | complete |
| HPL-001 | Explicit level-1 halo relation and selected-source plan | TOP-001, HCL-001 | complete |
| PBC-001 | Per-axis Cartesian physical boundary coordinate/value rules | HAL-001 | complete |
| HAX-001 | Apply an explicit level-1 same-level halo plan | HPL-001, PBC-001 | complete |
| SAM-001 | Exact level-1 block placement | GEO-001, STO-001 | complete |
| SAM-002 | Zero-order uniform sampling | GEO-001, STO-001 | complete |
| SAM-003 | Trilinear uniform sampling | HAL-002, GEO-001 | complete |
| OPR-001 | Pointwise operator contract and implementation | FND-002, STO-001 | complete |
| OPR-002 | Local stencil operator contract and implementation | FND-002, HAL-002 | complete |
| RED-001 | Streaming associative reduction | FND-002, PRI-001, STO-001 | complete |
| INT-001 | M0 end-to-end numerical and bounded-memory path | SAM-003, OPR-001, OPR-002, RED-001 | complete |

All Foundation, functional-composition checkpoint, and M0 capabilities are
complete.

## M1: Cartesian 3D Refined AMR

Status: complete. See `evidence/M1-ARCHITECTURE-HORIZON.md`.

| ID | Capability | Depends on | Status |
| --- | --- | --- | --- |
| FST-001 | Validated Cartesian 3D refined forest reconstruction | MOR-001 | complete |
| FST-002 | Reusable refined-forest artifact conformance | FST-001 | complete |
| TOP-002 | Exact batched refined contact target lookup | FST-002 | complete |
| BAL-001 | Global Cartesian 3D all-touch two-to-one balance | TOP-002 | complete |
| TOP-003 | Optional balanced refined six-face materialization | TOP-002, BAL-001 | proposed |
| GEO-002 | Cartesian 3D refined leaf bounds and spacing | FST-002, GEO-001 | complete |
| REL-001 | Balanced refined directional relation records | TOP-002, BAL-001 | complete |
| PRI-001 | Deterministic ascending primary-prefix planning | WSP-001 | complete |
| FCL-001 | Deterministic direct-face support closure | PRI-001, TOP-001 | complete |
| HCL-001 | Deterministic complete one-block halo support closure | PRI-001, TOP-001 | complete |
| STO-004 | Balanced refined support union and bounded primary planning | WSP-001, PRI-001, REL-001 | complete |
| SPR-001 | Explicit selected-primary refined support planning | STO-004 | complete |
| RST-001 | Cartesian 3D ratio-two cell-average restriction | FND-001, STO-001, STO-004 | complete |
| LIM-001 | Cartesian three-point limited-slope primitive | FND-001, RST-001 | complete |
| PRL-001 | Cartesian 3D ratio-two limited prolongation | FND-001, RST-001, LIM-001 | complete |
| RSL-001 | Accepted refined relation source-slot resolution | REL-001, STO-004 | complete |
| TGT-001 | Directed halo target boxes | FND-001 | complete |
| RPH-001 | Refined relation child-phase codes | FST-002, REL-001, RSL-001 | complete |
| SLB-001 | Cartesian same-level source-box translation | FND-001, TGT-001 | complete |
| FRP-001 | Cartesian FINER restriction placement | RST-001, TGT-001, RPH-001 | complete |
| CWP-001 | Cartesian COARSER explicit-workspace placement and reach | PRL-001, TGT-001, RPH-001 | complete |
| CSP-001 | Cartesian COARSER slope-support planning | BAL-001, REL-001, SPR-001, RSL-001, RPH-001, CWP-001 | complete |
| PWA-001 | Cartesian physical widening application | FND-001, PBC-001, TGT-001 | complete |
| CWA-001 | Explicit COARSER workspace application | CSP-001, PWA-001, CWP-001, RST-001 | complete |
| RHE-001 | Bounded selected refined halo execution | FND-001, BAL-001, REL-001, RSL-001, RPH-001, TGT-001, CSP-001, PWA-001, CWA-001, STO-003, WSP-001, SPR-001, SLB-001, FRP-001, CWP-001, RST-001, PRL-001 | complete |
| RHC-001 | Synchronous completed refined-primary consumption | RHE-001, STO-003 | complete |
| LOC-001 | Exact Cartesian 3D refined point ownership | FST-002, GEO-002 | complete |
| SAM-004 | Refined zero-order point-group sampling | LOC-001, GEO-002, STO-001 | complete |
| SAM-005 | Refined trilinear point-group sampling | LOC-001, GEO-002, SAM-003, RHE-001 | complete |
| RPS-001 | Bounded repeated refined point-sampling execution | LOC-001, SAM-004, SAM-005, STO-003, RHC-001, WSP-001 | complete |
| DAT-001 | Safe AMRVAC v5 3D snapshot metadata index | FND-001 | complete |
| DAT-002 | Canonical AMRVAC leaf-order forest binding | DAT-001, MOR-001, FST-001, FST-002 | complete |
| DAT-003 | Native selective non-staggered AMRVAC block reader | DAT-001, DAT-002, STO-003 | complete |
| ROI-001 | Refined cell-center physical-region windows | FST-002, GEO-002 | complete |
| OPR-003 | Cartesian 3D refined curl | FND-002, OPR-001, OPR-002, GEO-002 | complete |
| LFE-001 | Bounded selected refined curl and regional sum execution | ROI-001, OPR-003, RHC-001, RED-001, GEO-002, DAT-003 | complete |
| HLO-001 | Exact last-owner hinted refined point ownership | LOC-001, GEO-002 | complete |
| CHS-001 | Completed refined owner-halo sampling session | HLO-001, RHC-001, SAM-005, DAT-003, WSP-001 | complete |
| FLN-001 | Normalized magnetic field-line RHS | FND-001 | complete |
| RKS-001 | Fixed classical RK4 augmented-state arithmetic | FND-001 | complete |
| TRM-001 | Nonperiodic field-line termination policy | FLN-001, RKS-001, LOC-001 | complete |
| SLE-001 | Cached native refined field-line execution | CHS-001, FLN-001, RKS-001, TRM-001 | complete |

Historical composition and decomposition decisions are in
[M1 capability history](evidence/M1-CAPABILITY-HISTORY.md),
[M1 analysis decisions](evidence/M1-ANALYSIS-DECISIONS.md), and the
[M1 horizon](evidence/M1-ARCHITECTURE-HORIZON.md). These preserve prior scope,
qualified fixture evidence, rejected alternatives, and measured reopen triggers.

## Post-M1 Native Derived Analysis

Status: optimization round complete; broader product scope remains planned.
Completed M0/M1 rows retain their
original contracts and evidence. See the post-M1 stage in ROADMAP and
[optimization scope](designs/M1-ANALYSIS-OPTIMIZATION.md).

| ID | Capability | Depends on | Status |
| --- | --- | --- | --- |
| HPR-001 | Invocation-local owned halo preflight | RHE-001, RHC-001, CSP-001, CWP-001 | complete |
| CQP-001 | Invocation-local indexed cache query planning | CHS-001, HPR-001 | complete |

The completed round reduced selected native-analysis halo preparation cost and
large-batch cache lookup cost. The [execution plan](designs/M1-ANALYSIS-OPTIMIZATION.md#execution-order)
owns the resolved sequence; [round evidence](evidence/M1-OPTIMIZATION-ROUND.md)
records the baseline, gates, results and every deferred direction's trigger.

DVA-001 was withdrawn at the
user's direction before contract freeze: field-specific validity exploration
is not a separate deliverable. Access propagation, produced validity, and
downstream interpolation are general requirements of the actual consumers.
Reuse existing region algebra and introduce a new capability only for a
demonstrated missing responsibility or useful execution strategy.

The optimization round assessed halo preparation, cache scaling,
storage/output amplification, and conditional compute improvements. Direction
dispositions live in the execution plan or linked evidence, not as artificial
completed capability rows. Repeated raw/derived analysis and additional
along-field diagnostics remain broader product outcomes; optimization closure
does not mark them implemented. Storage/cache, requested-direction, and compute
variants remain measured choices.

## M2: Cartesian 2D

| ID | Capability | Depends on | Status |
| --- | --- | --- | --- |
| DIM-001 | Explicit active-dimension and singleton-z artifact provenance | FND-001 | proposed |

The queued first group is **Active-Dimension Foundation**, initially singleton
DIM-001; post-M1 analysis planning currently has priority. It must keep
`active_ndim` distinct from `ndir`, retain the canonical
five-dimensional singleton-z payload layout, and prevent valid 3D singleton-
axis grids from being inferred as 2D. Quadtree, variable-width DAT, geometry,
halo, bilinear sampling, and operator capabilities follow only after this
provenance integrates. Activate this prerequisite earlier if a post-M1 shared
representation change requires it; full 2D implementation is not a prerequisite
for composing existing valid 3D contracts.

## Later Milestones

The post-M1 analysis stage currently precedes broad M2 work; it advances only
the concrete M4/M6 pieces needed by the user workflows. The retained dimensional
and migration dependency sequence is:

1. M1: refined forest/topology -> refined geometry -> support/relation planning
   -> restriction/prolongation -> refined halos -> refined sampling -> native
   selective `.dat` adapter -> real-data bounded integration.
2. M2: active-dimension conventions -> quadtree/Morton generalization -> 2D
   topology/geometry -> refined halos -> bilinear sampling/operators -> real 2D
   integration.
3. M3: periodic topology -> periodic support/transfer -> refined periodic
   sampling/operators -> periodic integration.
4. M4: concrete scientific operators -> derived dependency/materialization
   lifecycle -> diagnostics/traversal -> independent helper disposition.
5. M5: complete AMRVAC parse/source -> writer/roundtrip -> uniform
   construction/layout -> export -> format workflow integration.
6. M6: boundary/name adapters -> dataset lifecycle -> canonical public API ->
   compatibility and backend-selection integration.
7. M7: package build -> parallel strategies -> supported-platform evidence ->
   fallback/rollback -> canonical cutover and old-path retirement.

Each arrow is a default semantic dependency, not permission to predeclare all
detailed contracts. Add concrete IDs and exact dependencies when a family
becomes active. Every milestone also closes the corresponding rows in
`SOURCE_MIGRATION.md` and records the benchmark levels required by
`PERFORMANCE.md`.

When selecting work, choose a capability whose dependencies are sufficiently
complete and whose result unlocks a real consumer. Dependencies outside the
active group must be `complete`; an earlier member inside the active group may
be consumed at `integrated` after focused and immediate-composition checks pass.
State any different readiness rule explicitly in the ledger. Do not create
isolated utilities without a place in the dependency chain.
