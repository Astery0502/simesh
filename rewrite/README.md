# Functional AMR Rewrite

This directory is an isolated, from-scratch rewrite of the computational core
of `simesh`. It exists beside the current implementation and does not replace
or modify it.

The isolated core is the first stage of a supported-feature migration. The
long-term roadmap closes every canonical source feature through rewrite,
adapter, retention, replacement, retirement, or explicit unsupported status,
then integrates the validated implementation behind the public `simesh`
package and performs a controlled cutover.

The rewrite uses small, explicit data transformations to recover and implement
the semantics of AMR topology, geometry, storage, ghost cells, sampling, and
scientific operators. Correctness is established incrementally by comparison
with the current implementation, simple independent references, and focused
domain invariants.

The project also treats runtime, peak memory, throughput, and parallel scaling
as joint design concerns. Data larger than available memory is an intended use
case, not an afterthought.

## Read Order

1. `CHARTER.md` -- purpose, scope, and design principles.
2. `AGENTS.md` -- durable instructions for agents working in this directory.
3. `CURRENT.md` -- the current checkpoint and next ready work.
4. `CAPABILITIES.md` -- the current dependency-ordered building blocks.
5. `ROADMAP.md` -- staged support matrix and milestone outcomes.
6. `WORKFLOW.md` -- how one capability is designed, specified, implemented, composed, and optimized.
7. `DECOMPOSITION.md` -- the five-question capability gate, decision ownership,
   and progressive refinement of earlier implementations.
8. `ANALYSIS_WORKLOADS.md` -- analysis-first workload, algorithm, data-structure,
   benchmark priorities, and the one-time completed-work reorientation audit.
9. `FUNCTIONAL_COMPOSITION.md` -- stable boundaries for storage, execution,
   halo planning, compute kernels, and alternative framework implementations.
10. `SOURCE_MIGRATION.md` -- feature-parity inventory from current source to
   final canonical replacement.
11. `PERFORMANCE.md` -- benchmark levels, baselines, metrics, recording, and
   regression policy.

Non-trivial design notes are added under `designs/` while alternatives are
still being explored. Stable capability specifications are added under
`contracts/` immediately before implementation. Simple capabilities may use a
single short contract without a separate design note. Keep the control surface
small and update existing files instead of creating process documents without
a concrete use.

## Suggested Long-Running Goal Prompt

```text
Continue the functional AMR rewrite under rewrite/.

Follow rewrite/AGENTS.md and the current checkpoint in rewrite/CURRENT.md.
Use rewrite/ANALYSIS_WORKLOADS.md when selecting algorithms, data structures,
execution strategies, and benchmark metrics. Complete its one-time
reorientation audit before resuming new M1 implementation.
Apply the five-question and decision-ownership gate in rewrite/DECOMPOSITION.md
before implementing or extending each non-trivial capability. Audit in-progress
work created under earlier rules before continuing it; preserve valid work and
refine it incrementally rather than resetting it.
Before freezing each capability group, record the current approach, one
structurally different credible alternative, the uncertainty most likely to
change the choice, and the evidence or consumer that would reopen it. Explore
only uncertainties that can materially change a boundary, complexity, layout,
transfer count, or memory model.
Work through dependency-ready capabilities across M1--M7 toward the complete
supported-feature migration and final cutover gates in SOURCE_MIGRATION.md.
Preserve numerical behavior, keep the new implementation isolated until its
integration stage, and follow PERFORMANCE.md for every hot path and milestone
workflow. Record enough correctness, performance, memory, I/O, and real-data
evidence appropriate to each capability's performance class. Material stable
contract changes use the independent sub-agent review described in
rewrite/WORKFLOW.md; routine clarifications do not.
Do not stop merely because one milestone is complete; update the feature
ledger, select the next dependency-ready capability, and continue until final
cutover is complete or no capability can make meaningful progress. Make one
cohesive Git commit after each completed capability group so the branch remains
an executable sequence of validated checkpoints. Before each commit, run the
group closing gate and stage only files belonging to the active group and its
checkpoint. At each milestone close, perform the concise architecture-horizon
review in WORKFLOW.md and use an independent sub-agent only for a material
cross-layer choice with credible alternatives or unresolved uncertainty.
```
