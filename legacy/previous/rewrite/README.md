# Functional AMR Rewrite

The independent N4 core is fixed; application development starts at
[docs/analysis-core](../../core-development/documentation-before-application-guide/development/README.md). This tree retains the older
rewrite implementation, contracts and evidence. Its historical plans and the
archived project procedures below are not the active application-development workflow.

An isolated development and validation tree for a functional AMR computational
core and eventual supported-feature migration into `simesh`.

The product is native-AMR analysis of large, mostly immutable DAT snapshots:
selective reads and reuse, efficient ghost data, halo-valid derived quantities,
and diagnostics/derivatives/integrals along magnetic field lines. Cython CPU is
the current compute baseline. Explicit semantic, storage, and execution
boundaries support measured OpenMP and future GPU implementations without
silently weakening numerical contracts.

The corrected user priorities are independent magnetic tracing with Q/twist
and optional retained lines, whole-domain current/gradient analysis followed by slices,
and full-domain local-response LOS synthesis. Target scale and their minimal
result contracts are in [THREE-PIPELINE-RESULTS](../../core-development/documentation-before-application-guide/development/archive/core-development-2026-09-08/pipeline-results.md).

The [analysis intent catalog](../../core-development/documentation-before-application-guide/development/archive/core-development-2026-09-08/workflows.md) starts from user results:
fast local/global reads and halos, magnetic lines, slices, isosurfaces and
isovolumes, plus explicitly labeled adjacent analysis candidates. It spells out
fields, spatial access, reuse and acceptance for twelve workflow families.
Resident throughput and bounded-memory feasibility are both product goals;
one execution strategy is not a default answer to every request.

## Start And Resume

Read [AGENTS.md](AGENTS.md) for the task-based authority map. Each development
cycle starts with [CURRENT.md](CURRENT.md). For spec convergence or selection,
use [baseline.md](../../core-development/documentation-before-application-guide/development/archive/core-development-2026-09-08/baseline.md) before [CAPABILITIES.md](CAPABILITIES.md) and the
selected design/contract. New contributors also read
[intent.md](../../core-development/documentation-before-application-guide/development/archive/core-development-2026-09-08/intent.md). Do not reconstruct all historical work on every turn.

The active task is spec convergence, with implementation paused by the user's
scope. BASELINE now recommends specifying the three mainlines before a bounded
independent-seed parallel tracing outcome; it supersedes the earlier sampling-phase
suggestion and activates no implementation.
Existing M0/M1 completion remains historical evidence; future outcomes and M2
are not an automatic execution queue. CURRENT is the live checkpoint authority.

## Documentation Map

- [intent.md](../../core-development/documentation-before-application-guide/development/archive/core-development-2026-09-08/intent.md): product intent, durable engineering constraints, scope.
- [baseline.md](../../core-development/documentation-before-application-guide/development/archive/core-development-2026-09-08/baseline.md): current development baseline, asset dispositions,
  requirement-to-capability/evidence mapping, open decisions and entry conditions.
- [workflows.md](../../core-development/documentation-before-application-guide/development/archive/core-development-2026-09-08/workflows.md): representative user requests and
  scientific/resource acceptance.
- [technique-candidates.md](../../core-development/documentation-before-application-guide/development/archive/core-development-2026-09-08/technique-candidates.md): open mapping from usage
  conditions and required properties to candidate techniques, source evidence
  and gaps; no combined architecture or implementation selection.
- [Analysis lifetimes](../../core-development/documentation-before-application-guide/development/archive/core-development-2026-09-08/lifetime-sketches.md): draft ownership/reuse
  constraints, three continuous usage sequences, candidate execution sketches
  and optionality costs; no frozen architecture or implementation.
- [Three pipeline results](../../core-development/documentation-before-application-guide/development/archive/core-development-2026-09-08/pipeline-results.md): corrected F/D/L
  requirements, scientific dependencies, known scale and remaining spec decisions.
- [FUNCTIONAL_COMPOSITION.md](FUNCTIONAL_COMPOSITION.md): fields/validity,
  storage, sessions, execution, and compute substitution boundaries.
- [DECOMPOSITION.md](DECOMPOSITION.md): semantic responsibility and refinement.
- [WORKFLOW.md](WORKFLOW.md): one design record, exploration, development,
  optional targeted review, focused validation, and checkpoints.
- [performance.md](../../core-development/documentation-before-application-guide/development/archive/core-development-2026-09-08/performance.md): evidence classes, baselines, complete
  resources, host/device comparisons, and regression/stopping rules.
- [WENO-REFERENCE.md](WENO-REFERENCE.md): the explicit feature-completion real-data
  profile, fixed cases, comparator scope and feasibility budgets; not routine tests.
- [ROADMAP.md](ROADMAP.md): stage outcomes and sequencing.
- [SOURCE_MIGRATION.md](SOURCE_MIGRATION.md): feature dispositions and cutover.
- [Designs](designs/README.md): alternatives and pending decisions.
- [Contracts](contracts/README.md): frozen capability behavior.
- [Documentation realignment](evidence/DOCUMENTATION-REALIGNMENT.md): requirement
  preservation and the new planning checkpoint.
- [M1 horizon](evidence/M1-ARCHITECTURE-HORIZON.md) and
  [M1 decisions](evidence/M1-ANALYSIS-DECISIONS.md): scoped historical evidence.

Keep rules in their owning documents and use links elsewhere. Simple
capabilities need no separate design/evidence file. Historical evidence records
what was demonstrated; it does not override current intent or silently redefine
an existing contract.

## Suggested Long-Running Goal Prompt

Future opt-in example only. It is not authorization for the current spec-only
task; implementation requires BASELINE's readiness and a later user instruction.

```text
Continue the functional AMR rewrite under rewrite/ toward the product outcomes
and supported-feature cutover in CHARTER, ROADMAP, and SOURCE_MIGRATION.

Follow rewrite/AGENTS.md and resume rewrite/CURRENT.md. Use rewrite/BASELINE.md
for current requirements, asset roles and implementation readiness. Apply the current
workload acceptance, functional/compute boundaries, and WORKFLOW development
cycle. Preserve completed contracts and historical evidence. For the selected
outcome, use existing evidence to select a concrete optimization or missing
boundary; apply general access/validity requirements through its consumers.
Do not create a field-specific prerequisite experiment. Continue through
dependency-ready groups and the queued roadmap.

Use bounded exploration and measured consumer evidence to choose optimizations.
Use independent review only for a concrete unresolved core question. Keep useful references,
document intentional differences, and create cohesive validated group
checkpoints. Do not stop solely because a group or milestone has completed;
continue within this goal until cutover is complete or no work can make
meaningful progress.
```
