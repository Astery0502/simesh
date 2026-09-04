# Capability Ledger

This is the lightweight dependency ledger for the rewrite. Add detail only when
a capability becomes active.

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
| LOC-001 | Exact Cartesian 3D refined point ownership | FST-002, GEO-002 | complete |
| SAM-004 | Refined zero-order point-group sampling | LOC-001, GEO-002, STO-001 | complete |
| SAM-005 | Refined trilinear point-group sampling | LOC-001, GEO-002, SAM-003, RHE-001 | complete |
| RPS-001 | Bounded repeated refined point-sampling execution | LOC-001, SAM-004, SAM-005, STO-003, RHE-001, WSP-001 | complete |

FST-001 supplies explicit flat preorder hierarchy and leaf maps, and FST-002
validates an unchanged artifact lifecycle once.  TOP-002 owns raw contact
semantics, and BAL-001 independently owns all-touch admissibility; both are
complete.  GEO-002 is also complete and consumes validated FST artifacts
without depending on balance or face caches.  REL-001 is complete and owns
selected directional relation records without support or value-transfer
policy; its consumer evidence keeps optional TOP-003 deferred.  The STO-002
Yellow trigger activated the now-complete extraction sequence: WSP-001
accounting, PRI-001 primary traversal, FCL-001 direct-face closure, and HCL-001
full-halo closure.  STO-002 is now a retained compatibility composition.
STO-004 now supplies exact bounded refined source union, dense-primary progress,
and one-row capacity bounds without owning transfer semantics.  The analysis
priority re-audit retains that operation for dense traversal but requires
SPR-001 before its next executor consumer so an explicit sparse/ROI selection
does not scan or read intervening primary leaves.  RST-001's numerical kernel
depends only on FND layout, while STO-001/STO-004 provide its completed dense
bounded gather evidence.
RST-001 now freezes the current eight-value ratio-two average independently of
relation/slot/halo policy and composes it with explicit FINER-source gathering.
LIM-001 now extracts the independently variable three-point slope rule as a
validated scalar plus allocation-free shared inline implementation.  PRL-001
owns ratio-two indexing, exact phase eta, limited reconstruction, regions, and
coarse reach without a slope cache.  RSL-001 provides bounded source-slot
identity without kind policy.  The proposed RAC-001 action enum was retired
because REL kind/mask already determines it.  Combined RTP-001 failed the
five-question gate and split into now-complete TGT-001 directed boxes and
now-complete RPH-001 ratio-two phases.  Proposed RSG-001 failed decomposition
and split into now-complete SLB-001 same-level translation, FRP-001 FINER
restriction placement, and CWP-001 COARSER workspace geometry; physical
widening remains PBC-owned.  SLB-001 is a topology-free affine translation from
explicit target boxes and reduced directions to equal-level source boxes.
SPR-001 now preserves STO-004 support semantics for explicit sparse ascending
primary selections without admitting gaps.  FRP-001 maps active finer phases to exact
RST source/target boxes, and CWP-001 exposes normalized PRL workspace placement
plus uncovered slope reach.  Their group evidence retains STO-004 for dense
full-domain traversal and records selected plan-to-gather bytes.
The selected refined halo completion group scopes its first complete consumer
to block extents at least four and per-side reach at most half a block.
CSP-001 plans the full CWP slope rectangle from the existing all-26 selected
support, PWA-001 owns physical widening, CWA-001 assembles one PRL workspace,
and RHE-001 owns bounded reader/writer scheduling.  The complete bounded path is
bitwise equal to its independent resident composition and safe dyadic current
comparison; its optimized preflight preserves calls/bytes/memory while materially
improving small, medium, and full WENO-metadata compositions.  CWP now accepts
every RPH-valid COARSER phase/direction after its first consumer exposed an old
restriction-send policy leak.  A proven `B=2`/wide-reach
counterexample is deferred to secondary closure or a separately contracted
cross-valid PRL strategy rather than hidden in the executor.

The completed bounded repeated-point sampling group separates exact leaf
ownership, zero-order cell selection, trilinear stencil arithmetic, and
reader/workspace scheduling. LOC-001 descends the existing flat forest under
canonical GEO face comparisons. SAM-004 and SAM-005 consume explicit
slot-to-point groups and do not select a reader, cache, or traversal policy.
RPS-001 locates once, groups repeated owners, and executes bounded batches;
zero-order batches read only owner interiors, while trilinear batches consume
RHE-001-completed one-cell halos. The existing SAM-002/SAM-003 block-centric
uniform wrappers remain fused compatibility/performance strategies. Exact
half-open point ownership, scalar point kernels, bounded array/non-array reader
execution, all refined relation/PBC kinds, public-RHE preservation, and
capacity/backend invariance are recorded in the group evidence. Measured
all-26 amplification and checked-halo planning remain explicit streamline and
native-reader optimization triggers rather than cache policy hidden in the
sampler.

Decomposition audit checkpoint: HAL-002's Red finding is resolved by explicit
HPL-001 relation planning, PBC-001 physical rules, and HAX-001 plan application;
HAL-002 is retained as the measured Fused compatibility implementation.  The
preserved TOP drafts have now been split into the M1 capabilities above; no Red
combined TOP contract remains.
The STO-002 Yellow trigger is resolved by WSP-001, PRI-001, FCL-001, and
HCL-001 with compatibility wrappers preserved.  SAM-002 and SAM-003 are
retained Fused implementations whose missing
semantic boundaries are extracted at refined sampling.  See
`evidence/DECOMPOSITION-AUDIT.md`.

## Later Milestones

The planned dependency sequence is:

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
