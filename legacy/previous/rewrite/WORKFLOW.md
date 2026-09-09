# Capability Workflow

These detailed rules describe development of existing rewrite capabilities.
New project-level analysis-core planning uses the
[historical project decision workflow](../../core-development/documentation-before-application-guide/development/archive/core-development-2026-09-08/historical-index.md#how-a-decision-advances);
it does not automatically inherit this tree's groups or milestone sequence.

This document owns the development cycle, exploration record, review, and
checkpoint rules. CHARTER owns intent; ANALYSIS_WORKLOADS owns user acceptance;
DECOMPOSITION owns boundary analysis; PERFORMANCE owns measurement policy.

## Select And Resume Work

For a new outcome, use [baseline.md](../../core-development/documentation-before-application-guide/development/archive/core-development-2026-09-08/baseline.md) to select by confirmed user
requirements and current gaps, then map the selected design to its affected R
rows. State delivered/protected/deferred scope and the matching evidence there;
reuse this single record rather than add a reporting layer. A historical
`complete` capability is eligible evidence, not automatic product acceptance.
The latest user instruction and CURRENT define active work; BASELINE supplies
requirements and readiness rather than a separate permission gate.

Select a small set of related changes delivering one concrete useful outcome.
Record the active work and next action in CURRENT. Use capability/group labels
only where they help track a real boundary; there is no fixed member quota or
requirement to give every helper its own contract, test or approval step.

Advance as dependencies become usable. An immediate consumer may establish a
producer's behavior without a separate isolated test. Update CAPABILITIES when
its actual scope or status changes; ordinary edits need no ledger ceremony.

For a long-running goal, continue across completed groups and milestones within
the authorized goal. Update the checkpoint, select the next ready outcome, and
continue until that goal is satisfied or no work can make meaningful progress.
Ordinary milestone completion is a checkpoint, not an automatic stopping point.

## One Design Record

For a substantive design choice, record the intended result, affected behavior,
chosen boundary, reason and evidence in one place. Link existing numerical and
ownership contracts. A small readable change can be explained in its diff or
checkpoint; it does not need a separate design document.

Use [DECOMPOSITION](DECOMPOSITION.md) as optional design questions when ownership
or coupling is unclear. There is no mandatory five-question audit, split rule,
independent-review gate or requirement to invent a competing implementation.
Choose checks through the project
[verification policy](../../core-development/documentation-before-application-guide/development/archive/core-development-2026-09-08/performance.md#choose-the-smallest-useful-check).
Direct inspection is enough when it resolves the question. Measure a real
consumer when performance is at issue; do not create a benchmark merely to
complete a form.

## Exploration And Contract Freeze

Recover current behavior and domain meaning before choosing an implementation.
Explore only uncertainties that can change semantics, a boundary, complexity,
layout, transfer count, or memory model. A prototype has one recorded question
and a time or variant bound. Keep it outside package imports, stable tests, and
downstream dependencies. It needs enough evidence to answer the question, not
a production evidence suite. Carry the conclusion into the design, then delete
the prototype or keep it only under an ignored experiment path.

A design records alternatives; a contract freezes selected behavior. Use
[contracts/README.md](contracts/README.md) for contract contents. Do not freeze
an implementation accident as universal semantics or add defensive checks
without an owned invariant. Conversely, no new policy document silently changes
an existing strict arithmetic, representation, ownership, or failure contract.

## Implement And Integrate

Read the relevant code, contract and existing evidence, then implement the useful
change with explicit buffers and owned state. Check the behavior actually affected:
core halo/AMR numerical behavior, valid data, selection/identity, live memory,
parallel execution or complete consumer costs as appropriate.

Prefer an existing focused consumer check. Add a test only for a meaningful
uncovered core case; metadata descriptions, incidental helper-call order and
simple unchanged quantity formulas normally need direct reading only. Multiple
helpers can be protected by one representative composition. Follow the project
[smallest-useful-check rule](../../core-development/documentation-before-application-guide/development/archive/core-development-2026-09-08/performance.md#choose-the-smallest-useful-check).

Update CURRENT and continue once the relevant evidence is adequate. Reuse passing
checks; broaden only for new changes, failures, cross-consumer impact or unresolved
concerns. Preserve the existing arithmetic/ownership/failure contracts and do not
hide a discrepancy with wider tolerances. PERFORMANCE owns cost comparisons;
FUNCTIONAL_COMPOSITION owns numerical-strategy and buffer meanings.

## Changed Requirements And Existing Work

For new user steering, identify the affected requirement and implementation,
update the primary definition and next action, and continue within scope. Reuse
existing decisions/evidence. Update ROADMAP/CAPABILITIES only when their actual
priority, dependency or status changes; no routine architecture-horizon audit or
reassessment of every completed capability is required.

Reopen a design for a concrete consumer need, incompatibility or measured cost,
not stylistic purity. Historical tests and measurements keep their original scope.

## Independent Review And Autonomy

Default to one agent and direct source/contract/diff inspection. Use an independent
sub-agent review only for a specific unresolved core question where another
technical perspective is useful, following the project
[review rule](../../core-development/documentation-before-application-guide/development/archive/core-development-2026-09-08/development.md#autonomous-decisions-and-user-judgments).
Contract, layout or milestone changes do not automatically require review.

State the question and resolve real findings. Review does not substitute for
meaningful core tests or performance measurements, and its absence is not an
artificial blocker when direct evidence is sufficient. Earlier user authorization
persists; seek new input only for an actual missing scope or scientific judgment.

## Implementation Group Gate

Close a coherent delivery when the intended behavior and relevant costs are
established. Rebuild changed compiled sources with the supported helper; a Python
or documentation edit does not automatically require a clean extension build.
Use affected focused/integration checks and useful current-path comparisons.
A full regression suite is justified by broad integration impact or an unresolved
failure, not required at every group boundary. Likewise, run only benchmarks
that address the changed hot path or requested consumer outcome.

Resolve failures, record the useful result/evidence and update the active status.
Passing unrelated checks are not repeated without a reason. Use ordinary Git
checkpoints for recoverable owned work, inspecting the diff and staging only the
selected files. No fixed group size, separate approval packet, content manifest
or per-helper commit is required. Existing numerical and public guarantees remain
in force; this changes verification selection, not scientific meaning.

## Feature-Completion Reference Assessment

At group selection, decide whether the changed feature affects WENO-supported
reading, topology/geometry, halos, local fields, sampling, caching, trajectories
or output. For an applicable feature, name an explicit completion assessment
using [WENO-REFERENCE.md](WENO-REFERENCE.md): affected component cases and their
immediate consumer, comparators, numerical scope, resource limits and command.
Run it after implementation and focused validation, alongside the existing
delivery assessment; share overlapping useful runs. A cross-layer change may
select a broader matrix, not automatically all historical benchmarks.

This is a deliberate feature-completion activity, not a default pytest/Make/CI
hook, per-edit check or per-commit trigger. Small unrelated changes and features
the fixture cannot represent record a concise not-applicable reason and use
their appropriate evidence. Reuse existing runs when code, workload, build and
required metrics still match; refresh only the missing comparisons.

An assessment closes with functional pass/fail or a documented semantic
difference, component costs, complete resources, feasibility, and implement/
retain/defer recommendations. An applicable hard correctness/resource failure
cannot be waived as a missing fixture. Missing data prevents claims about that
case, but does not run expensive substitutes silently. Preserve prior capability
scope; assess new evidence before selecting another optimization group.

## Documentation Checkpoint

Documentation-only work validates requirement preservation, local links,
contract/status consistency, and the diff by direct inspection. Check affected
links when needed; do not create a documentation test suite or routine review gate.
Do not build extensions or run numerical/performance suites solely for prose
changes. Do not mark proposed software complete or present historical tests as
fresh results. A documentation commit, when made, is labeled as such and never
represented as an executable capability-group completion.

## Checkpoint Surface

CURRENT contains the active stage/group, last executable checkpoint, current
capability/status, unresolved constraints, next concrete action, and exact
applicable check commands. Keep it short, normally under 150 lines. CAPABILITIES
owns exact member status; ROADMAP owns stage goals. Link durable rules and
historical evidence instead of copying their full contents into CURRENT.
