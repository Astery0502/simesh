# AGENTS.md

## Scope

These instructions apply to all work under `rewrite/`.

The current implementation under `src/simesh/`, its tests, data, and reports is
a read-only source of behavior and evidence unless a task explicitly authorizes
changes outside `rewrite/`. Do not gradually refactor the current implementation
as part of this rewrite.

## Start Every Work Cycle

Read, in order:

1. `CHARTER.md`
2. `CURRENT.md`
3. `CAPABILITIES.md`
4. the design note and contract for the selected capability, if they exist

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

- Work on one ready capability at a time.
- A capability is ready only when every dependency is `complete`, unless its ledger entry explicitly states a lower required status.
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
- For a hot path, declare the benchmark workload, current/reference
  comparator, performance hypothesis, metrics, and material-regression rule
  before final optimization. Record both the kernel and its immediate composed
  consumer.
- Store raw benchmark runs outside Git under the ignored benchmark-results
  tree; commit concise environment, command, raw-summary, comparison, and
  trade-off evidence under `evidence/`.
- Use Cython typed memoryviews, C structs, pointers, or contiguous arrays where appropriate.
- Do not use Python callbacks in hot loops.
- Do not allocate inside cell, stencil, or block hot loops unless measurement justifies it.
- Keep a simple reference implementation until optimized variants are validated.
- Do not confuse a small conceptual capability with a requirement for many runtime calls; fuse kernels only after their separate meanings are established.
- Do not introduce a generic framework before multiple concrete capabilities need it.
- Keep documentation proportional to the decision. Prefer short contracts and evidence summaries.
- Do not use hashes as workflow, validation, or governance machinery.
- Avoid speculative defensive checks. Validate demonstrated domain invariants at the boundary that owns them.

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

## Contract Changes

Do not silently change a stable contract. Follow the independent sub-agent
review in `WORKFLOW.md`. If the primary agent and reviewer agree and the evidence
supports the change, update the contract, affected tests, capability state, and
`CURRENT.md`, then continue without user approval.

## Completion And Checkpoints

A capability is complete only when its contract, implementation, focused tests,
integration point, and relevant performance or memory evidence agree. Update
`CAPABILITIES.md` and `CURRENT.md` immediately after completion or a material
change of direction.

After a capability is complete, create one cohesive Git commit containing its
design or contract updates, implementation, tests, integration changes, and
checkpoint updates. Use ordinary Git history as the recovery mechanism. Do not
maintain separate content hashes, artifact hashes, or integrity manifests.

Before that commit:

- run the capability's focused tests;
- run all established rewrite regression and integration tests;
- run relevant comparisons against the current implementation;
- inspect `git status`, the working diff, and the staged diff;
- stage only files belonging to the active rewrite capability and its checkpoint;
- leave unrelated repository changes unstaged and untouched.

Commit only an executable checkpoint whose staged contents match the completed
capability.
