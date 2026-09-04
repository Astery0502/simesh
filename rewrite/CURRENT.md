# Current Rewrite Checkpoint

## Active State

- Milestone: M1 Cartesian 3D refined AMR.
- Last completed group: **Selected Refined Transfer Planning**; SPR-001,
  FRP-001, and CWP-001 are complete.
- One-time analysis-workload reorientation audit: complete; see
  `evidence/ANALYSIS-PRIORITY-REAUDIT.md`.
- Completed outcome: explicit sparse ascending selected leaves are closed over
  requested REL directions and mapped to SAME/FINER/COARSER boxes without gaps.
- Next work: define the smallest PBC-aware refined support/application group
  that fills CWP uncovered slope reach and produces complete selected halos.
- Readiness: dependencies outside the group must be `complete`; earlier members
  inside the group may be consumed at `integrated` after focused and immediate
  composition checks pass.
- Group evidence: `evidence/SELECTED-REFINED-TRANSFER-PLANNING.md`; 166 focused
  checks and all 632 rewrite tests pass. The standard composition records
  selected/read bytes, support amplification, reader calls, and dense fallback.

The implementation remains isolated under `simesh_rewrite`. M0 supplies a
complete non-periodic Cartesian 3D level-1 numerical and bounded-memory path.
M1 currently supplies refined forest reconstruction/conformance, contact and
balance semantics, geometry, relation/support planning, restriction, limiter,
prolongation, selected-primary planning, source-slot mapping, and SAME/FINER/
COARSER transfer geometry. Refined value application and sampling remain
incomplete.

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
  RSL-001, TGT-001, RPH-001, SLB-001, SPR-001, FRP-001, and CWP-001.
- TOP-003 optional face-cache materialization remains proposed and must be
  justified by a repeated real consumer.
- The selected transfer-planning group is complete. Full-domain work retains
  STO-004; selected work uses SPR-001 and the same downstream slot layout.

The audit accepted two focused reopen findings. First, preserve STO-004 dense
semantics but add SPR-001 before an executor so sparse/ROI primary selection
does not process gaps. Second, preserve SAM-002/SAM-003 wrappers and numerical
rules but extract point ownership/interpolation at refined sampling; add an
exact locator and repeated point sampler instead of using dense uniform output
for streamlines. INT-001 remains M0 evidence and is not extended into the
selected analysis executor. No stable numerical contract changed.

HAL-002 remains a measured fused compatibility implementation over the explicit
HPL/PBC/HAX semantics. SAM-002 and SAM-003 remain validated fused level-1
implementations; refined sampling is the trigger for extracting any additional
shared ownership/stencil boundaries. Historical decisions, performance numbers,
and rejected alternatives live in `designs/`, `contracts/`, and `evidence/`.

## Known Limits And Migration Gaps

- The only available real refined Cartesian 3D `.dat` fixture is staggered, so
  it currently supplies forest/topology/geometry evidence but not supported
  refined payload evidence.
- PBC-aware completion of CWP slope reach, refined value application, complete
  selected halos, refined sampling, a native selective `.dat` adapter, and a
  real bounded M1 vertical slice remain.
- Cartesian 2D, periodic meshes, broader scientific/derived workflows, complete
  AMRVAC I/O/write/export, Dataset/public API integration, packaging,
  parallelism, fallback, and cutover remain assigned to M2--M7.
- Staggered payloads and non-Cartesian geometries are outside the current plan.

## Resume Procedure

1. Read this file, `CAPABILITIES.md`, and `ANALYSIS_WORKLOADS.md`. Read
   `CHARTER.md` only on a new agent's first rewrite cycle, at a milestone
   boundary, or for a material project-direction change.
2. Decompose PBC-aware support completion, physical widening, value application,
   and execution; do not combine them into a refined halo dispatcher.
3. Record group alternatives and the analysis algorithm-selection gate, then
   freeze one-to-four dependency-adjacent members with explicit performance
   classes and closing checks.
4. Preserve STO-004 for dense traversal, use SPR-001 for explicit selections,
   and include CWP's uncovered slope reach before reading payload.
5. After complete refined halos create the refined point-owner/sampler boundary;
   retain SAM-002/SAM-003 fused uniform wrappers until that real consumer.

## Current Reproduction Commands

```text
.venv/bin/python rewrite/build_ext.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests
```

Run focused member tests and relevant current comparisons named in the active
group declaration before the group-closing command. Benchmark commands and
historical measurements remain with their capability or group evidence.
