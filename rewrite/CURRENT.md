# Current Rewrite Checkpoint

## Active State

- Milestone: M1 Cartesian 3D refined AMR.
- Last completed group: **Bounded Refined Repeated Point Sampling**; LOC-001,
  SAM-004, SAM-005, and RPS-001 are complete.
- Completed outcome: finite points flow through exact half-open refined leaf
  ownership and zero/trilinear grouped kernels in caller order. Zero order reads
  only unique owner interiors; trilinear samples synchronous RHE-completed
  primary chunks with bounded all-26 support and no completed-block copy.
- One-time analysis-workload reorientation audit: complete; see
  `evidence/ANALYSIS-PRIORITY-REAUDIT.md`.
- Group evidence: `evidence/BOUNDED-REFINED-REPEATED-POINT-SAMPLING.md`; 98
  focused and all 765 rewrite tests pass. Clustered `P=4096,U=4` location takes
  0.088 ms; bounded zero/trilinear compositions take 0.404/4.201 ms at smaller
  capacities with exact resident equality and explicit calls/bytes/memory.
- Next work: define the native selective non-staggered AMRVAC `.dat` reader
  adapter and prove it substitutes for array readers in bounded refined halo
  and repeated-point paths. Do not start local-field or streamline stepping
  until that storage boundary passes.
- Readiness: dependencies outside the group must be `complete`; earlier members
  inside the group may be consumed at `integrated` after focused and immediate
  composition checks pass.

The implementation remains isolated under `simesh_rewrite`. M0 supplies a
complete non-periodic Cartesian 3D level-1 numerical and bounded-memory path.
M1 currently supplies refined forest reconstruction/conformance, contact and
balance semantics, geometry, relation/support planning, restriction, limiter,
prolongation, selected-primary planning, complete refined value application,
bounded selected halo execution, exact refined point ownership, and bounded
repeated zero/trilinear sampling. Native real-data and local-field/streamline
analysis integration remain incomplete.

## Durable Decisions

- Preserve supported numerical behavior, not the current classes, internal
  algorithms, memory layout, or byte-identical file output.
- Treat the current implementation as evidence rather than unquestionable
  truth; explain every intentional difference.
- Use zero-based signed `int64` IDs and C-contiguous `float64` canonical block
  interchange in `(slot, field, x, y, z)` order.
- Use explicit half-open valid regions, borrowed read-only inputs, and
  caller-owned outputs/workspaces. Checked standalone wrappers validate complete
  contracts; validated executors may use explicit unchecked kernels internally.
- Keep storage/framework state behind coarse-grained functional reader/writer
  adapters. Numerical kernels receive canonical buffers and explicit metadata.
- Treat resident, bounded, cached, and framework-specific execution as
  replaceable strategies over shared numerical contracts.
- Keep topology, geometry, access requirements, support planning, physical
  rules, same-level transfer, restriction, prolongation, and execution policy
  independently owned.
- Conceptual decomposition does not require many runtime calls. Retain a fused
  kernel when it remains contract-equivalent and gives a useful measured trade-off.
- Field payloads must support bounded block/chunk flow; report managed workspace
  separately from outputs, mappings, page cache, and process RSS.
- Performance is correctness-constrained and multi-objective. Classify work as
  `cold/control`, `composition-only`, `hot-kernel`, or `milestone-workflow` and
  gather only the evidence required by that class.
- Optimize for selected-region local-field and streamline analysis: avoid I/O
  and payload movement first, reuse explicit metadata/plans second, batch
  compatible operators third, and tune arithmetic/parallel kernels after
  composed profiling.
- A time-boxed prototype may precede a contract only to answer one concrete
  feasibility, layout, cache, or complexity question. It cannot become a package
  import or downstream dependency.
- Default to one agent. Use sub-agents for genuinely independent bounded work or
  a materially useful independent challenge. Material stable-contract changes
  require one independent reviewer; editorial/private changes do not.
- A model or reasoning-effort setting alone does not require sub-agent use.
- Track migration by supported feature and public workflow, not source files or
  line counts. Use ordinary Git history as the recovery mechanism.

## Capability Status

The dependency ledger and exact status authority are in `CAPABILITIES.md`.

- Foundation, migration/performance protocol, functional storage substitution,
  and all M0 capabilities are complete.
- M1 complete through: FST-001/002, TOP-002, BAL-001, GEO-002, REL-001,
  WSP-001, PRI-001, FCL-001, HCL-001, STO-004, RST-001, LIM-001, PRL-001,
  RSL-001, TGT-001, RPH-001, SLB-001, SPR-001, FRP-001, CWP-001, CSP-001,
  PWA-001, CWA-001, RHE-001, LOC-001, SAM-004/005, and RPS-001.
- TOP-003 optional face-cache materialization remains proposed and must be
  justified by a repeated real consumer.
- The selected transfer-planning group is complete. Full-domain work retains
  STO-004; selected work uses SPR-001 and the same downstream slot layout.
- Complete selected refined halos retain a checked preflight reference and use
  an equivalent optimized proof over internal plans.  CWP/CSP accept every
  RPH-valid COARSER phase/direction; the old restriction-send gate is removed.

Both audit reopen findings are resolved: SPR-001 removes dense sparse-selection
work, while LOC/SAM/RPS extract refined point semantics and execution without
removing the measured SAM-002/SAM-003 fused wrappers. INT-001 remains M0
evidence rather than becoming the selected analysis executor. Historical
decisions and rejected alternatives live in `designs/` and `evidence/`.

## Known Limits And Migration Gaps

- The only available real refined Cartesian 3D `.dat` fixture is staggered, so
  it currently supplies forest/topology/geometry evidence but not supported
  refined payload evidence.
- A native selective `.dat` adapter and real bounded local-field/streamline M1
  vertical slices remain. `B=2`, reach above half a block, direction-subset
  support, last-leaf/neighbor lookup, and persistent caching have explicit
  consumer-driven reopen triggers.
- Cartesian 2D, periodic meshes, broader scientific/derived workflows, complete
  AMRVAC I/O/write/export, Dataset/public API integration, packaging,
  parallelism, fallback, and cutover remain assigned to M2--M7.
- Staggered payloads and non-Cartesian geometries are outside the current plan.

## Resume Procedure

1. Read this file, `CAPABILITIES.md`, and `ANALYSIS_WORKLOADS.md`. Read
   `CHARTER.md` only on a new agent's first rewrite cycle, at a milestone
   boundary, or for a material project-direction change.
2. Apply DECOMPOSITION and the analysis algorithm-selection gate to the native
   selective `.dat` reader. Keep format parsing, block transfer, cache policy,
   numerical consumers, and dataset lifecycle separate.
3. Compose the native reader with refined halos and RPS before adding the
   local-field and streamline vertical slices and M1 horizon review.

## Current Reproduction Commands

```text
.venv/bin/python rewrite/build_ext.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_loc_001.py rewrite/tests/test_sam_004.py rewrite/tests/test_sam_005.py rewrite/tests/test_rps_001.py rewrite/tests/test_sam_002.py rewrite/tests/test_sam_003.py rewrite/tests/test_rhe_001.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/rps_001.py --profile standard --output rewrite/benchmark-results/rps-001-standard.json
```

Historical commands and measurements remain with their capability/group
evidence. Declare the next active group before implementation.
