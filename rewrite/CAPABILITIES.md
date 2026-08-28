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
| STO-002 | Bounded workspace with direct-face and full-halo chunk plans | STO-001, TOP-001 | complete |
| STO-003 | Functional block reader/writer adapters and execution substitution | STO-001 | complete |
| HAL-001 | Non-periodic physical-boundary halo provision | GEO-001, STO-001 | complete |
| HAL-002 | Same-level sibling halo provision | TOP-001, HAL-001 | complete |
| HPL-001 | Explicit level-1 halo relation and selected-source plan | TOP-001, STO-002 | complete |
| PBC-001 | Per-axis Cartesian physical boundary coordinate/value rules | HAL-001 | complete |
| HAX-001 | Apply an explicit level-1 same-level halo plan | HPL-001, PBC-001 | proposed |
| SAM-001 | Exact level-1 block placement | GEO-001, STO-001 | complete |
| SAM-002 | Zero-order uniform sampling | GEO-001, STO-001 | complete |
| SAM-003 | Trilinear uniform sampling | HAL-002, GEO-001 | complete |
| OPR-001 | Pointwise operator contract and implementation | FND-002, STO-001 | complete |
| OPR-002 | Local stencil operator contract and implementation | FND-002, HAL-002 | complete |
| RED-001 | Streaming associative reduction | FND-002, STO-002 | complete |
| INT-001 | M0 end-to-end numerical and bounded-memory path | SAM-003, OPR-001, OPR-002, RED-001 | complete |

All Foundation, functional-composition checkpoint, and M0 capabilities are
complete.

## M1: Cartesian 3D Refined AMR

| ID | Capability | Depends on | Status |
| --- | --- | --- | --- |
| FST-001 | Validated Cartesian 3D refined forest reconstruction | MOR-001 | complete |
| TOP-002 | Refined leaf face topology and two-to-one balance | FST-001 | proposed |

FST-001 supplies explicit flat preorder hierarchy and leaf maps.  TOP-002 is
the next dependency-ready capability; refined geometry can then proceed from
FST-001 while directional relation planning proceeds from TOP-002.

Decomposition audit checkpoint: HAL-002 is Red and is being incrementally
refined before M1 implementation resumes.  HPL-001 has extracted its
relation/source-slot planning decision and PBC-001 has extracted shared
physical coordinate/value rules; the explicit plan-application boundary
HAX-001 remains.  The preserved untracked TOP-002
draft is also Red and must split reusable FST conformance, exact contact
semantics, BAL-001, and optional face materialization before contract freeze.
STO-002 is Yellow with a mandatory split trigger before refined support
planning.  SAM-002 and SAM-003 are retained Fused implementations whose missing
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
complete and whose result unlocks a real consumer. By default, every dependency
must be `complete`. If a capability needs only an earlier dependency status,
state that exception explicitly in its ledger entry. Do not create isolated
utilities without a place in the dependency chain.
