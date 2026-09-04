# AGENTS.md

## Scope

These instructions apply to all work under `rewrite/`.

The current implementation under `src/simesh/`, its tests, data, and reports is
a read-only source of behavior and evidence unless a task explicitly authorizes
changes outside `rewrite/`. Do not gradually refactor the current implementation
as part of this rewrite.

## Start Every Work Cycle

On every work cycle, read in order:

1. `CURRENT.md`
2. `CAPABILITIES.md`
3. the design note and contract for the selected capability, if they exist

Read `CHARTER.md` on an agent's first rewrite cycle, when entering or closing a
milestone, or when proposing a material change to project direction. Do not
reread it for an ordinary continuation whose scope and constraints are already
captured by `CURRENT.md` and the selected contract.
Read `ANALYSIS_WORKLOADS.md` when selecting or optimizing an algorithm, data
structure, cache, query/execution strategy, local scientific operator, sampler,
or streamline capability, and at the one-time reorientation audit or a
milestone horizon review.

Read `DECOMPOSITION.md` before selecting, splitting, extending, or implementing
any non-trivial capability, and before refining work created under earlier
rules. No non-trivial implementation begins until the five-question gate and
decision-ownership inventory pass or a justified exception is recorded.
Read `FUNCTIONAL_COMPOSITION.md` before adding a storage backend, execution
strategy, framework adapter, halo family, or cross-capability executor.
Read `SOURCE_MIGRATION.md` when selecting a milestone, adding/removing public
behavior, or preparing production integration. Read `PERFORMANCE.md` before
implementing or declaring completion of a hot path, benchmark, parallel path,
or milestone workflow.

Resume from `CURRENT.md`; do not reconstruct or redesign the whole project on
every turn. Read `ROADMAP.md` when choosing or revising a milestone. Read
`WORKFLOW.md` when starting a capability, changing a contract, or preparing a
checkpoint. Do not reread every process document on every work cycle.

## Working Rules

- Work on one ready capability at a time inside one active capability group.
- A dependency outside the active group must be `complete`. A dependency inside
  the active group may be `integrated` when its focused checks and immediate
  composition checks pass. Record any different readiness rule in the ledger.
- For a non-trivial capability, explore alternatives in a short design note, then freeze the selected behavior in a contract before implementation.
- For a simple capability, use one concise contract and start implementation without extra design ceremony.
- Apply `DECOMPOSITION.md` before freezing a non-trivial contract. If a
  function makes an independently variable decision outside its stated
  semantics, split that decision into an explicit semantic, planning, policy,
  adapter, or execution boundary.
- Refine earlier nonconforming implementations incrementally: freeze behavior,
  inventory decisions, extract boundaries, retain compatibility wrappers,
  prove equivalence, measure, then migrate consumers. Do not perform stylistic
  big-bang rewrites.
- Keep inputs, outputs, ownership, mutation, valid regions, and error behavior explicit.
- Keep external storage/framework state at adapter boundaries. Numerical
  kernels receive canonical buffers and explicit metadata, never file handles,
  memory maps, dataset objects, or implicit backend dispatch.
- Prefer free functions over stateful service objects. When a callable backend
  needs state, carry the callable, state, shape, and alias information in a
  small immutable descriptor and pass that descriptor explicitly.
- Treat resident, bounded, cached, and framework-specific execution as
  replaceable strategies. Validate every strategy against the same semantic
  reference before optimizing it independently.
- Decompose halos into access requirements, support planning, topology
  relations, value-transfer rules, and workspace mutation. Do not grow the
  level-1 same-level kernel into a refined/periodic dispatcher.
- Close migration by feature disposition and public-workflow evidence. Do not
  mechanically port legacy files or claim completion from source/line counts.
- Optimize for selected-region local-field and streamline analysis before
  simulation-style full-domain updates when the contracts cannot serve both
  equally. Follow `ANALYSIS_WORKLOADS.md` for workload and metric priority.
- For a hot path, declare the benchmark workload, current/reference
  comparator, performance hypothesis, metrics, and material-regression rule
  before final optimization. Record both the kernel and its immediate composed
  consumer.
- Classify performance work as `cold/control`, `composition-only`, `hot-kernel`,
  or `milestone-workflow`. Cold/control capabilities need complexity and
  allocation reasoning, not an isolated timing. Composition-only capabilities
  are timed first in a real consumer. Only hot kernels require an independent
  kernel benchmark.
- Store raw benchmark runs outside Git under the ignored benchmark-results
  tree; commit concise environment, command, raw-summary, comparison, and
  trade-off evidence under `evidence/`.
- Use Cython typed memoryviews, C structs, pointers, or contiguous arrays where appropriate.
- Do not use Python callbacks in hot loops.
- Do not allocate inside cell, stencil, or block hot loops unless measurement justifies it.
- Keep a simple reference implementation until optimized variants are validated.
- Checked standalone wrappers validate their complete contract. After an
  executor has completed equivalent preflight, its hot loop may call explicit
  unchecked kernels rather than repeating invariant validation per chunk.
- Do not confuse a small conceptual capability with a requirement for many runtime calls; fuse kernels only after their separate meanings are established.
- Do not introduce a generic framework before multiple concrete capabilities need it.
- Keep documentation proportional to the decision. Prefer short contracts and evidence summaries.
- Do not use hashes as workflow, validation, or governance machinery.
- Avoid speculative defensive checks. Validate demonstrated domain invariants at the boundary that owns them.

## Capability Groups

- A capability group contains one to four dependency-adjacent capabilities that
  produce one concrete composed outcome. Record the group outcome, member IDs,
  and closing checks in `CURRENT.md` before implementing its first member.
- Before freezing the group, record four concise exploration answers: the
  current approach, one structurally different credible alternative, the
  uncertainty most likely to change the choice, and the evidence or consumer
  that would reopen the decision. Listing an alternative does not require
  implementing it.
- Keep semantic decomposition and contracts capability-sized. Grouping changes
  validation and checkpoint cadence; it does not merge independently variable
  decisions back into one function or contract.
- For each member, complete its contract, reference or invariants, production
  implementation, focused tests, and immediate composition checks. A member may
  reach `integrated` before the group gate and reaches `complete` only when the
  group gate passes.
- Close the group with one clean rewrite extension build, one accumulated
  rewrite regression/integration run, one set of relevant current-path
  comparisons, and, when the group contains hot-kernel or milestone-workflow
  work, one standard benchmark pass covering those paths and their immediate
  composed outcome. Do not rerun unaffected standard benchmarks.
- Use a singleton group when no second capability naturally shares its composed
  outcome, build, and benchmark gate. Do not add a member merely to satisfy a
  group-size target.
- If a group gate fails, fix the owning capability, rerun its focused checks,
  then rerun the failed group gate. Do not repeat already-passing unrelated
  benchmark profiles.
- When group evidence is needed, end it with concise deferred alternatives:
  name, reason deferred, and concrete reopen trigger. Omit the section when no
  credible alternative remains.

## Correctness

Use the minimum combination of evidence that establishes the capability:

- comparison with the current implementation;
- a small independent Python or NumPy reference;
- focused mathematical or structural invariants;
- a representative real-data check when it adds information.

The current implementation is evidence, not unquestionable truth. Never hide an
unexplained difference by weakening a comparison or increasing a tolerance.

## Performance And Memory

Correctness comes first, then measured optimization. Compare useful variants by
runtime, throughput, peak memory, allocation behavior, and parallel scaling as
relevant. There is no requirement that one variant be best for every workload.

Field payloads must be able to flow through bounded chunks or blocks. Operators
must declare their access pattern and halo requirements so an executor can
decide whether to stream, cache neighbors, map storage, or keep data in memory.

Stop optimizing the active capability when it has a correct integrated
implementation on a useful runtime-memory-complexity trade-off and another
credible variant no longer produces a material improvement. Record promising
deferred ideas and continue upward instead of optimizing indefinitely.

Do not invent an extreme synthetic repetition count solely to justify a cache,
plan, or materialized artifact. Before a real consumer exists, record the cost
model and defer optimization unless complexity or memory safety itself is at
risk.

## Exploration

A time-boxed disposable prototype may precede a stable contract when feasibility,
layout, cache value, or algorithmic complexity cannot be resolved from analysis.
The prototype must answer one recorded question, remain outside package imports
and downstream dependencies, and require no production-grade evidence suite.
Summarize the result in the eventual design or contract, then delete the
prototype or keep it only under an ignored experiment path.

Before the next M1 implementation group, perform the one-time completed-work
reorientation audit in `ANALYSIS_WORKLOADS.md`. Preserve validated semantics and
reopen only a boundary that can materially affect the primary analysis paths,
complexity, transfer volume, peak memory, or cross-layer representation. This
audit is not permission to rebenchmark or refactor every completed capability.

## Sub-agent Use

- Default to one agent for simple contracts, local implementation, focused
  tests, documentation, and mechanical checks.
- Use sub-agents when two or more bounded workstreams are genuinely independent
  and can proceed with disjoint file ownership, or when an independent technical
  challenge materially improves a complex design, numerical proof, or benchmark
  interpretation.
- One independent reviewer is required for a material stable-contract change:
  numerical meaning, ownership/mutation/atomicity, a representation consumed by
  other capabilities, or a supported public failure behavior. Editorial fixes,
  private renames, benchmark-parameter changes, and tests that only strengthen
  an unchanged contract do not require a reviewer.
- Do not create sub-agents to restate the primary analysis, perform tiny serial
  steps, or satisfy a ceremony. The primary agent owns integration and resolves
  conflicting conclusions with the smallest discriminating experiment.
- A model name or reasoning-effort setting does not by itself require delegation;
  use these task and risk conditions.

At each milestone close, perform one architecture-horizon review across data
layout, repeated validation/materialization, memory and runtime hotspots,
backend boundaries, and assumptions needed by the next milestone. Use one
independent sub-agent only when the review contains a material cross-layer
decision, two or more credible architectures, or an unresolved technical
disagreement; otherwise keep the review single-agent and concise.

## Contract Changes

Do not silently make a material stable-contract change. Follow the independent
sub-agent review in `WORKFLOW.md` when the change affects numerical meaning,
ownership/mutation/atomicity, a consumed representation, or supported public
failure behavior. If the primary agent and reviewer agree and the evidence
supports the change, update the contract, affected tests, capability state, and
`CURRENT.md`, then continue without user approval. Non-material clarifications
need ordinary focused review only.

## Completion And Checkpoints

A capability is complete only when its contract, implementation, focused tests,
integration point, evidence appropriate to its performance class, and active
group gate agree. Update `CAPABILITIES.md` and `CURRENT.md` after each member
reaches `integrated`, after a material change of direction, and when the group
closes.

After the group gate passes, create one cohesive Git commit containing the
group's design and contract updates, implementations, tests, integration
changes, one concise group evidence summary when evidence is needed, and
checkpoint updates. Simple members do not need separate design and evidence
files. Use ordinary Git history as the recovery mechanism. Do not maintain
separate content hashes, artifact hashes, or integrity manifests.

Before that commit:

- run every member's focused tests;
- build the rewrite extensions once from the complete group state;
- run all established rewrite regression and integration tests once;
- run the group's relevant current-path comparisons and performance-class
  checks once;
- inspect `git status`, the working diff, and the staged diff;
- stage only files belonging to the active capability group and its checkpoint;
- leave unrelated repository changes unstaged and untouched.

Commit only an executable group checkpoint whose staged contents match the
completed members and recorded group gate.
