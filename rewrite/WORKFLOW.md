# Capability Workflow

## Capability Group Cycle

Before implementation, select one to four dependency-adjacent capabilities
that deliver one concrete composed outcome and record the active group in
`CURRENT.md`. The group declaration names its member IDs, dependency order,
outcome, focused checks, full regression command, relevant current-path
comparisons, performance classes, and any required standard benchmark set.
Exact later member IDs may be refined by the five-question gate before the first
member is implemented, but the group must not expand beyond four members or
absorb unrelated work.

Before freezing the group, add a short exploration-coverage record:

```text
Current approach:
Structurally different credible alternative:
Most decision-changing uncertainty:
Reopen evidence or consumer:
```

This is a coverage check, not a requirement to implement every alternative. If
the uncertainty can materially change a boundary, complexity class, data
layout, transfer count, or memory model, use the disposable-exploration rule.
Otherwise record the deferral and proceed.

When the group selects or optimizes an algorithm, data structure, cache, local
operator, sampler, or streamline path, also complete the algorithm-selection
record in `ANALYSIS_WORKLOADS.md`. The representative consumer and metrics must
match the primary analysis workload unless the group explicitly owns a
migration-only or simulation-style compatibility requirement.

Implement one ready member at a time. Dependencies outside the active group
must be `complete`; an earlier member of the same group may be consumed at
`integrated` after its focused and immediate-composition checks pass. A member
remains `integrated` until the group gate passes. Do not rerun the accumulated
suite or standard benchmark matrix after each member.

Close the group by:

1. building the rewrite extensions once from the complete group state;
2. running all member-focused tests and the accumulated rewrite regression and
   integration suite once;
3. running the relevant comparisons with the current implementation once;
4. when required by a member's performance class, running one standard
   benchmark pass for the group's new hot paths and immediate composed outcome,
   without rerunning unaffected benchmarks;
5. resolving any failure at its owning capability and rerunning the failed gate;
6. recording any credible deferred alternative with its reason and concrete
   reopen trigger;
7. marking the members `complete`, updating the ledgers and evidence, inspecting
   the working and staged diffs, and creating one cohesive group checkpoint.

Use a singleton group when no other capability naturally shares the same
composed outcome and closing checks. Do not invent a companion capability or
prematurely specify downstream behavior to avoid a singleton.

## Capability Cycle

For one ready capability:

1. Recover the relevant behavior from the current implementation and domain meaning.
2. Apply the five-question and decision-ownership gate in `DECOMPOSITION.md`; split independently variable meanings before implementation.
3. For a non-trivial capability, write a short design note under `designs/` that compares the real alternatives and their composition, memory, and performance trade-offs.
4. Select an approach and write or refine its stable contract under `contracts/`.
5. Define exact outputs, floating-point comparisons, and focused invariants.
6. Build the smallest clear Python or NumPy reference when it adds independent evidence.
7. Implement the simplest Cython form with explicit inputs, outputs, and workspace.
8. Compare intermediate artifacts, not only the final user-visible result.
9. Integrate with the immediate lower-level producer and next higher-level consumer.
10. Assign the capability a performance class and gather only the evidence required by that class.
11. Explore optimized or fused variants only after the simple implementation and composition are correct.
12. Run the capability's focused tests and immediate reference/composition comparisons; use smoke benchmarks only when useful during iteration.
13. Update the capability ledger to `integrated` and update the active-group checkpoint.
14. Continue to the next ready member without rerunning the accumulated suite or standard benchmark matrix.
15. Mark members `complete` and commit only after the capability group gate passes.

Keep this cycle proportional. A simple index transform does not need a report;
a refined ghost algorithm or out-of-core executor does.

## Disposable Exploration

Before freezing a non-trivial contract, a short prototype is allowed when a
concrete feasibility, layout, cache, or complexity question requires executable
evidence. Record the question and a time or variant bound. Experimental code
must remain outside package imports, stable tests, and downstream dependencies.
It may use lightweight measurements but does not need production validation.
Carry only the result into the design/contract, then remove the prototype or
leave it under an ignored experiment path.

## Design And Specification

A design note is exploratory. It explains the problem, the few credible
approaches, and the trade-offs that affect later composition. It may change or
be discarded as evidence improves.

A contract is the selected specification. It defines the behavior that the
implementation and its consumers may rely on. Implementation starts only after
that contract is clear enough to test.

Capability size is semantic, not textual. Use `DECOMPOSITION.md` to record the
owned and non-owned decisions and to distinguish coherent validation/loop cases
from independently variable policy, planning, storage, or execution choices.
If the five-question gate fails, split before implementation or record a
specific exception in the design with evidence.

Existing work is refined progressively at dependency, substitution,
optimization, integration, defect, or cutover boundaries. Preserve validated
behavior and compatibility wrappers while extracting functions; do not reopen
completed work for stylistic purity alone.

Do not require separate design and contract files for trivial capabilities. A
simple capability may append its concise final evidence to its contract. Prefer
one evidence summary for a capability group instead of one file per member when
the checks and benchmark are shared. Do not turn design notes into long reports.
Their purpose is to preserve reasoning that a later agent would otherwise have
to rediscover.

## Contract Contents

Each active contract should answer:

- What concept and responsibility does this capability represent?
- What are its inputs and outputs?
- Who owns each buffer and where may mutation occur?
- What layout, valid region, and halo are required?
- What access pattern does it use?
- What numerical behavior is exact and what uses tolerance?
- Which old behavior, independent reference, or invariant establishes correctness?
- Which immediate capability consumes the result?

Use prose, small tables, and signatures. Avoid framework-like schemas unless
the same structure is repeatedly useful.

## Numerical Comparison

Use exact comparison for discrete structures and exactly specified placement.
For floating-point work, choose metrics from the operation's meaning, such as
maximum absolute error, relative error away from zero, norm error, conservation
error, or convergence behavior. Record the worst discrepancy and explain why
the chosen comparison is sufficient.

Do not use hashes for numerical comparison or artifact governance.

## Source Migration Closure

Before implementing a current-source behavior, identify its feature-family row
in `SOURCE_MIGRATION.md`, current authority, user-observable contract, and
intended disposition. Migrate semantics rather than classes or lines. When a
feature is integrated, update its parity tests, benchmark/workflow evidence,
and disposition together.

Do not retire or bypass a canonical path merely because lower kernels exist.
Replacement requires the corresponding real public workflow, format behavior,
failure behavior, and fallback/rollback evidence. Legacy modules are evidence
unless a supported behavior still uniquely depends on them.

## Optimization

An optimization begins with a concrete hypothesis. Keep the simple validated
implementation available while testing variants such as loop fusion, layout
changes, workspace reuse, tiling, mapping, caching, streaming, or OpenMP.

Classify the work before benchmarking:

- `cold/control`: validate complexity, allocation, and correctness; do not
  create an isolated timing merely because the implementation is in Cython;
- `composition-only`: measure at the first real immediate consumer rather than
  constructing a repeated synthetic consumer;
- `hot-kernel`: measure the kernel and its immediate composed consumer;
- `milestone-workflow`: measure a representative real-data or public workflow,
  including runtime, memory, I/O, and scaling dimensions that matter.

A capability moves to a more expensive class only when a cost model, profiler,
scaling risk, or real consumer justifies it.

For a hot path, record the `PERFORMANCE.md` workload/profile, comparator,
hypothesis, metrics, and material-regression rule before the final experiment.
The active group runs the standard kernel and immediate composed benchmarks
once at its closing gate. Milestones also require a real-data/public-workflow
benchmark at the level reached so far.

Choose among useful variants by the relevant combination of runtime,
throughput, peak memory, allocation behavior, scaling, and conceptual cost.
It is acceptable to retain separate in-memory and low-memory strategies.

Optimization occurs at two levels:

- kernel optimization improves one established transformation;
- composition optimization reduces transfers, temporary storage, repeated
  traversal, or call overhead across several correct transformations.

Keep the separate reference kernels even when the production path uses a fused
implementation. A fused path must preserve the contracts of the capabilities it
combines.

## Alternative Implementations And Adapters

An implementation is replaceable only when the substitution boundary is
explicit and the alternatives share one conformance suite. Use coarse-grained
Python function adapters for storage or framework transfer at chunk boundaries;
keep Cython cell and block loops free of Python callbacks.

For a new backend or execution strategy:

1. identify the existing semantic contract it must preserve;
2. keep backend state, shape, ownership, alias information, and transfer
   callable explicit;
3. transfer into or out of canonical caller-owned buffers unless the backend
   independently implements the complete compute contract;
4. run the same exact/numerical tests through every implementation;
5. measure adapter overhead, transfer amplification, retained state, peak
   memory, and failure behavior separately from kernel runtime;
6. retain a simple resident-array path as the reference composition.

Do not add global backend registries, inheritance frameworks, dynamic kernel
dispatch in hot loops, or framework-specific branches inside domain kernels.
Add a small adapter only when a concrete implementation needs it.

Halo evolution follows the same rule. Operator reach determines requirements;
topology and geometry determine source relations; a support planner determines
what must be resident; physical, same-level, prolongation, restriction, and
periodic functions determine values; the executor composes them. Each layer is
tested independently and no layer infers another from hidden state.

Stop the current optimization cycle when the implementation is correct,
integrated, and useful on the runtime-memory-complexity trade-off, and the next
credible variant does not provide a material improvement. Record valuable
future ideas briefly and move to the next layer. Optimization is part of
completion, not an unbounded search for a theoretical maximum.

## Independent Contract Review

Use an independent sub-agent review when evidence suggests a material stable
contract change affecting numerical meaning, ownership/mutation/atomicity, a
representation consumed elsewhere, or supported public failure behavior:

1. The primary agent writes a short proposed change with the conflicting evidence and affected capabilities.
2. The primary agent starts an independent sub-agent reviewer and asks it to inspect the contract, raw evidence, reference behavior, and invariants before relying on the primary conclusion.
3. The reviewer either agrees, rejects the change, or requests one concrete experiment.
4. If both agree, the primary agent updates the contract, implementation, tests, capability ledger, and checkpoint, then continues without user approval.
5. If they disagree, run the smallest experiment likely to resolve the disagreement and review again. Keep the existing contract until agreement is reached; continue other ready work when possible.

The review is a technical second perspective, not an approval ceremony. Keep
the proposal and conclusion concise. Editorial clarification, private naming,
benchmark tuning, and stronger tests for unchanged behavior use ordinary review
and do not require a sub-agent.

For parallel implementation or investigation, delegate only bounded workstreams
with enough independence to save wall-clock time or add a genuinely different
technical perspective. Give each sub-agent explicit file or evidence ownership;
the primary agent integrates the result. Do not delegate tiny sequential steps,
duplicate the same review, or infer delegation from the selected model or
reasoning-effort level.

## Milestone Architecture Horizon Review

At each milestone close, inspect the composed implementation before selecting
the next milestone's detailed capabilities. Keep the review focused on:

- data structures or layouts leaking across multiple semantic layers;
- validation, traversal, transfer, or materialization repeated across consumers;
- measured runtime, memory, I/O, and allocation hotspots;
- storage/backend boundaries that force avoidable full-payload movement;
- assumptions about dimension, geometry, refinement, ownership, or scheduling
  that the next milestone may invalidate;
- completed capabilities that still lack a real consumer.

Apply the workload order and local-field/streamline metrics in
`ANALYSIS_WORKLOADS.md`; explicitly identify any optimization retained mainly
for migration parity or a simulation-style workload.

Record only decisions, deferred questions, and reopen triggers in the milestone
evidence. Do not repeat every completed capability result. Use one independent
sub-agent when a material cross-layer choice has at least two credible
architectures or the primary analysis has an unresolved technical uncertainty.
A routine milestone with no such decision uses one concise primary-agent review.

## Checkpoint Discipline

During an active capability group and after each material decision, `CURRENT.md`
should say:

- active milestone;
- active group outcome and member IDs;
- last completed group and last integrated capability;
- current capability and status;
- important unresolved differences;
- next ready capability;
- exact commands for focused checks and the closing group gate;
- source-migration rows and benchmark baselines changed by the checkpoint.

Keep `CURRENT.md` as a short resume surface, normally no more than 150 lines.
Move completed milestone narratives and historical measurements to capability
or milestone evidence instead of duplicating them in the checkpoint.

The Git commit created after a completed capability group is the durable
recovery point. Keep group commits cohesive and executable. Intermediate
experiments and integrated members do not need their own commits unless an
external interruption requires a clearly labeled recovery checkpoint. Before
the group commit, inspect `git status`, the working diff, and the staged diff;
run the group gate; and ensure that unrelated repository changes remain
unstaged.
