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
| MOR-001 | Cartesian 3D level-1 Morton mapping | FND-001 | complete |
| TOP-001 | Validated level-1 topology | MOR-001 | complete |
| GEO-001 | Cartesian 3D block bounds and spacing | TOP-001 | complete |
| STO-001 | In-memory block source and sink | FND-001 | complete |
| STO-002 | Bounded workspace with direct-face and full-halo chunk plans | STO-001, TOP-001 | complete |
| HAL-001 | Non-periodic physical-boundary halo provision | GEO-001, STO-001 | complete |
| HAL-002 | Same-level sibling halo provision | TOP-001, HAL-001 | complete |
| SAM-001 | Exact level-1 block placement | GEO-001, STO-001 | complete |
| SAM-002 | Zero-order uniform sampling | GEO-001, STO-001 | complete |
| SAM-003 | Trilinear uniform sampling | HAL-002, GEO-001 | complete |
| OPR-001 | Pointwise operator contract and implementation | FND-002, STO-001 | complete |
| OPR-002 | Local stencil operator contract and implementation | FND-002, HAL-002 | proposed |
| RED-001 | Streaming associative reduction | FND-002, STO-002 | proposed |
| INT-001 | M0 end-to-end numerical and bounded-memory path | SAM-003, OPR-001, OPR-002, RED-001 | proposed |

## Later Milestones

M1 adds refined topology, coarse/fine connectivity, restriction, prolongation,
refined halos, and refined sampling. M2 generalizes the proven model to 2D. M3
adds periodic topology and halo behavior.

When selecting work, choose a capability whose dependencies are sufficiently
complete and whose result unlocks a real consumer. By default, every dependency
must be `complete`. If a capability needs only an earlier dependency status,
state that exception explicitly in its ledger entry. Do not create isolated
utilities without a place in the dependency chain.
