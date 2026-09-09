# M1 Analysis Optimization Scope

Status: optimization round complete; HPR-001 and CQP-001 complete. See
[round evidence](../evidence/M1-OPTIMIZATION-ROUND.md) for the frozen baseline,
budgets, independent review and measurements.
The first operational objective is to reduce halo preparation cost in selected
native analysis. This document owns the order and round-exit criteria; WORKFLOW
and PERFORMANCE continue to own development and measurement rules.
The user withdrew the proposed field-specific DVA-001
exploration. General operator access, produced validity, and downstream sampling
requirements cover that behavior through the actual consumers.

## Candidate Algorithms

Use existing M1 contracts and evidence to identify one material bottleneck or
missing boundary. Declare a concrete workload and a group only after that choice.
Reuse established region algebra and numerical kernels where sufficient; no new
capability is required merely to restate a general requirement.

| Direction | Existing boundary/evidence | Decision to assess |
| --- | --- | --- |
| Selective reads and raw-data reuse | DAT-003, STO-003, M1 horizon | Are repeated native transfers material after existing completed-field hits? |
| Halo support reduction | SPR/CSP/RHE and measured all-26 amplification | Can requested targets reduce work while preserving extra slope/base closure? |
| Halo storage | RHE workspace and CHS entries | Does interior-only support storage or another retained representation improve complete memory/runtime cost? |
| Shared plans and completed-field reuse | RHC/CHS, repeated-analysis workloads | Which repeated planning/halo work is actually removed by retaining an artifact? |
| Operator composition and batching | FND-002, OPR/LFE, actual new operators | Do compatible consumers share reads/halos and preserve required output validity without extra materialization? |
| Bounded output | LFE/SLE full-output strategies | Does the real request need chunk delivery or accumulator-only execution? |
| Along-field evaluation | LOC/HLO/SAM/CHS/FLN/RKS/SLE | Can diagnostics share location/preparation, and which derivative/integrand contracts are missing? |
| Compute strategies | Existing CPU baselines and portability boundaries | Does measured arithmetic or independent-query work justify fusion, OpenMP, or an active device backend? |

These are candidate algorithms within the execution order below, not a
requirement to implement all cache layers. Compare costs and reopen assumptions
only where the current workload differs materially from historical evidence. Select the smallest
discriminating proof/probe only when it can change the decision; existing
evidence may already be sufficient to proceed to a contract.

## Execution Order

Use this default order. At each step, inspect only the affected consumer and
record whether to implement, retain, defer, or reject the candidate. Do not
restart the whole assessment after every group. Reorder when a demonstrated
dependency, accuracy failure, or resource limit makes another step necessary;
record that reason in CURRENT and this note.

| Step | Work | Entry and exit decision | Current state |
| --- | --- | --- | --- |
| 0 | Establish the bounded comparison set | Reuse existing evidence; freeze exact cases, baseline revision/build, metrics, resource limits, and gain criteria. Refresh only measurements needed to select step 1. | frozen in round evidence |
| 1 | Reduce halo preparation cost | Separate support reads from relation/action planning, validation, and application in the same consumer. Choose support projection or planning/check reuse from that attribution; evaluate the other against the retained result. | HPR-001 complete; projection deferred with trigger |
| 2 | Improve cache/query scaling | After step 1's disposition, check lookup, metadata copying, grouping, and miss costs over realistic working sets. Add raw/plan/derived retention only when it removes material remaining work. | CQP-001 complete; persistent retention deferred |
| 3 | Reduce workspace and output amplification | Compare support interiors versus padded storage, copies and retained representations, and bounded output. Move earlier if the declared workload cannot fit its resource limit. | assessed; storage/output retained under complete resource bounds |
| 4 | Improve remaining compute/scheduling | Consider fusion/batching and independent-query or OpenMP parallelism only for remaining measured costs. A GPU backend requires its own active workload and validation; it is not a round-exit requirement. | existing batching retained; new compute/backends deferred |
| 5 | Close the optimization round | Compare the final retained composition to step 0 and resolve every direction's disposition under the exit criteria below. | complete; see summary and timing limitations |

Update each step's state and link its disposition evidence as work proceeds;
CURRENT names only the next action, and CAPABILITIES tracks actual members.
These steps are not capability IDs or five mandatory implementation groups.
Each selected implementation uses WORKFLOW's group size and readiness rules.
Within step 1, reducing bytes is not automatically better than reducing planning
time. Planning reuse must not entrench a representation that a justified support
change immediately replaces. Check DIM-001 implications before freezing shared
direction/phase/session representations.

## Starting Evidence And First Objective

The first objective is **lower selected-analysis halo preparation cost while
preserving numerical results and useful bounded-memory behavior**. Use local
field analysis as the initial consumer and cached native sampling/field lines
as the adjacent compatibility and reuse consumer. It is not a new field-specific
numerical experiment or a requirement to create another physical operator.

Existing evidence identifies two competing costs:

- [Selected halo completion](../evidence/SELECTED-REFINED-HALO-COMPLETION.md)
  attributes 81.6% of one instrumented medium run to preflight even after its
  initial optimization. This is a historical workload result, not a universal
  fraction or a fresh measurement.
- [M1 horizon](../evidence/M1-ARCHITECTURE-HORIZON.md) records 27 loads for one
  small tdm primary and substantial all-26 support amplification. Projection
  must preserve COARSER slope and mixed physical-base support.
- [Cache implementation](../src/simesh_rewrite/completed_halo_sampling.py)
  currently copies key/recency arrays and linearly searches slots in its access
  plan. Assess large-working-set cost, not only small-cache hit rate.
- Raw-cache, eager-header, neighbor-table, and ROI-index alternatives retain
  their historically limited benefit until changed workloads justify reopening.

Step 0's design record names one structurally different candidate for the first
choice, a bounded probe only if needed, and exact focused/group commands before
implementation. Do not prescribe a specific replacement algorithm from these
observations alone. Ordinary preflight guarantees remain contractual; reuse or
unchecked execution needs an equivalent proof and explicit lifetime.

## Comparison Set And Promotion

Freeze a small comparison set before optimization; reuse it across this round:

- local analysis: small/thin ROI, a mixed-refinement medium region, and a dense
  full-domain control; include physical and coarse/fine boundaries;
- repeated analysis: coherent and divergent owners, cold/cleared and warm
  sessions, cache below/at/above the observed working set, and a supported
  uncached reference; larger capacities must reflect a plausible query;
- resources: minimum useful and larger bounded capacities, selected useful/read
  bytes, retained plans/cache, backend scratch, output, and total live footprint.

Use the existing synthetic refined native fixtures and real tdm; the WENO
regular-field bridge can add refined field evidence with its setup cost and
staggered origin reported. Keep the missing direct real refined non-staggered
fixture qualification. A large profile is required for new out-of-core claims,
not merely because a small bounded workspace is measured.

The primary performance comparator is the pre-optimization M1 composition on
the same runner. Record its Git revision plus relevant working diff, build,
configuration, and request. Also compare the best applicable original canonical
path where workload and outputs match; document nonmatching work explicitly.
An early rewrite reference is not the original implementation, and warm versus
cold or different cache capacities are not equivalent old/new speedup claims.

Record first-result and total latency, planning/read/apply/compute/output costs,
support loads and bytes, cache work, and complete resources as relevant. Separate
instrumented attribution from headline timing; do not sum overlapping stages.
At later groups compare both with the previous retained variant and the fixed
round baseline so cumulative regressions are visible.

Before each implementation group, freeze its minimum useful gain and protected
cases using PERFORMANCE. For requested-direction projection, retain the recorded
candidate gate of at least 20% composed improvement or 2x support/read-byte
reduction on two representative refined cases unless evidence justifies a
documented revision before final measurement. Other candidates define their
own material runtime, memory, I/O, or scaling gain before optimization; do not
apply a kernel-only gain to the whole workflow or choose thresholds afterward.
Resource-only gains need an explicit acceptable latency trade-off. PERFORMANCE's
regression rules still apply to sparse/dense and cold/warm control cases.

## Group And Round Completion

An implementation group completes only through the existing WORKFLOW gate:
contract correctness, required scientific acceptance, immediate composition,
focused/full regression checks, applicable performance/resource evidence, and
an executable checkpoint. Rejected alternatives are not marked as completed
software capabilities.

This optimization round closes when:

1. Each ordered direction has a recorded disposition: implemented and validated,
   retained because current behavior is sufficient, deferred with a concrete
   trigger/dependency, or rejected by a bounded comparison. No active experiment
   or selected implementation group remains unfinished.
2. The final selected-analysis and repeated/trajectory comparison cases meet the
   frozen numerical, validity, ownership, failure, and resource requirements.
   An unmet hard requirement cannot be relabeled as an optional deferral.
3. Adopted changes meet their predeclared useful-gain criteria, and no material
   regression remains unresolved under PERFORMANCE. Report sparse/dense and
   cold/warm trade-offs, including any cases where the original path is faster.
4. The final retained composition has same-runner evidence against the fixed
   baseline for both primary workflow families. Reuse a final group run if it
   already covers the complete retained state; otherwise run only the missing
   end-to-end cases, not all historical kernel benchmarks again.
5. One concise round summary records before/after costs, complete memory scope,
   fixture limitations, retained alternatives, and reopen conditions. Update
   CAPABILITIES/CURRENT and perform the focused horizon review using that same
   summary rather than another audit campaign.

If no candidate improves the baseline under its requirements, close the bounded
assessment as **baseline retained**, not as a delivered speedup. No proof of
global optimality, mandatory GPU implementation, or exhaustive cache/algorithm
search is required. A remaining hard blocker means the round is still open.

Round closure preserves the original M1 completion and does not automatically
close the broader post-M1 product stage. Then select the next justified missing
analysis capability from ROADMAP, or resume DIM-001/M2 if no such capability is
ready. Record remaining product gaps explicitly; new diagnostics/integrals are
not implicitly delivered by performance work. Later consumers may reopen a
specific optimization when their assumptions change.

## Constraints On Every Selected Consumer

- Propagate requested output regions backward through actual per-input access
  and produced validity forward. Allocation alone never proves valid values.
- Preserve field interpretation, transfer/derivative order, interface/boundary
  accuracy, and the chosen numerical strategy. Resolve missing scientific
  semantics in the owning contract, not an unrelated preliminary experiment.
- Retain strict M1 arithmetic and all-26 reference behavior. Existing RHE limits
  remain even B>=4 and side reach<=B/2; reject unsupported requests explicitly.
- Retained artifacts need explicit identity, dependency lifetime, and valid
  coverage. Account for complete live resources, including output and scratch.
- Assess DIM-001 before a shared direction/phase/session representation change.
  Avoid new 3D-only shared assumptions without requiring full M2 implementation
  for an otherwise valid existing 3D composition.
- Keep functional ownership and storage/execution/compute substitution intact.
  GPU support is not claimed or required by this assessment.

## Resume Into Development

Record the chosen outcome, affected boundaries, and concrete evidence in the
selected design using WORKFLOW's single record. Create only justified capability
IDs, freeze contracts and acceptance, then add exact focused/group commands to
CURRENT before implementation. Preserve completed M1 status for its original
scope; use normal defect/contract review if an existing guarantee is found false.

This note replaces the withdrawn initial design scope. Historical M1 decisions
and benchmark records remain unchanged. Current results and limits are in the
linked round evidence. No new feature stage is activated.
