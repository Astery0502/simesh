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
7. `FUNCTIONAL_COMPOSITION.md` -- stable boundaries for storage, execution,
   halo planning, compute kernels, and alternative framework implementations.
8. `SOURCE_MIGRATION.md` -- feature-parity inventory from current source to
   final canonical replacement.
9. `PERFORMANCE.md` -- benchmark levels, baselines, metrics, recording, and
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
Work through dependency-ready capabilities across M1--M7 toward the complete
supported-feature migration and final cutover gates in SOURCE_MIGRATION.md.
Preserve numerical behavior, keep the new implementation isolated until its
integration stage, and follow PERFORMANCE.md for every hot path and milestone
workflow. Record enough correctness, performance, memory, I/O, and real-data
evidence to justify each completed capability. Stable contracts may change
only through the independent sub-agent review described in rewrite/WORKFLOW.md.
Do not stop merely because one milestone is complete; update the feature
ledger, select the next dependency-ready capability, and continue until final
cutover is complete or no capability can make meaningful progress. Make a
cohesive Git commit after each completed capability so the branch remains an
executable sequence of validated checkpoints. Before each commit, run the
accumulated rewrite regression suite and stage only files belonging to the
active rewrite capability and its checkpoint.
```
