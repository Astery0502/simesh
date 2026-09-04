# Capability Decomposition And Refinement

## Purpose

The rewrite builds small semantic functions that can be validated, replaced,
and composed. "Small" does not mean few lines or one runtime call. It means the
smallest coherent unit that owns one independently meaningful domain decision.

Every non-trivial capability passes this decomposition gate before its contract
is frozen or implementation begins. Existing implementations pass the same gate
before they are extended, fused, exposed through a new public workflow, or used
as the foundation of another capability.

## Five Required Questions

Answer all five in the design or concise contract. A `no` requires a split or a
specific written justification showing why the apparent coupling is one
indivisible semantic responsibility.

### 1. Is The Semantic Responsibility Singular?

State the responsibility in one sentence. The capability should have one
primary reason to change. If the sentence joins independently variable domain
meanings with "and", split them.

Validation that establishes the owned input contract and loop mechanics that
implement the one meaning are not extra responsibilities.

### 2. Can It Be Validated Independently?

The capability has a focused invariant, exact comparison, mathematical
reference, or small independent implementation. Its correctness must not be
observable only through a final end-to-end workflow.

### 3. Can Its Implementation Be Substituted Independently?

A simple, optimized, resident, bounded, cached, or framework-specific
implementation can replace another without changing the consumer's semantic
contract. If replacement requires consumers to understand the internal
algorithm, the boundary is wrong or incomplete.

### 4. Is Its Cost Evaluated At The Right Boundary?

Assign the least expensive `PERFORMANCE.md` class that can guide a decision. For
a hot path, identify a kernel metric and the immediate composed metric. For a
composition-only helper, identify the first real consumer where its cost is
visible. For cold/control work, state the relevant complexity and allocation
bound instead of creating an isolated timing. Inability to locate any useful
cost boundary or bound is evidence that the responsibility may be mixed or too
small.

### 5. Are Semantic Meaning And Policy/Execution Separate?

The function owns domain meaning, or it owns a planning/execution policy, but
does not silently own both. Storage backend selection, chunk capacity, cache
policy, traversal strategy, parallel scheduling, fusion, and framework dispatch
remain explicit inputs or separate planning/execution functions unless they are
the capability's stated responsibility.

## Decision Ownership Rule

Inventory the decisions made by the proposed function or capability. A
decision is a choice that could vary independently while the other behavior
remains meaningful: a numerical rule, ownership/tie rule, topology relation,
boundary policy, storage policy, algorithm strategy, cache/scheduling policy,
or failure policy.

If a function makes a decision outside its stated semantic responsibility, and
that decision can change independently, split it into an explicit semantic,
planning, policy, adapter, or execution function. Pass the selected decision as
plain data when the consumer only needs to apply it.

The following do not by themselves require a split:

- branches that directly implement the one owned semantic rule;
- bounds, dtype, layout, alias, and invariant validation owned by the boundary;
- axis/direction cases of one dimension-independent meaning;
- loop tiling, vectorization, or fused execution that preserves already frozen
  semantic contracts;
- error cases that are part of the same input/output contract.

The following normally require a split:

- inferring a backend, cache, traversal, or parallel policy inside a numerical
  kernel;
- choosing among same-level copy, restriction, prolongation, physical, or
  periodic meaning inside one unplanned halo dispatcher;
- combining topology discovery with value mutation when either can change
  independently;
- selecting numerical ownership/tie behavior as a side effect of iteration
  order;
- using hidden object history to select a semantic or execution mode;
- changing multiple unrelated stable contracts to implement one capability.

## Required Boundary Record

Before implementation, record:

```text
Responsibility:
Owned semantic decisions:
Explicitly non-owned decisions:
Inputs and outputs:
Mutation and ownership:
Access pattern / halo reach:
Independent reference or invariant:
Immediate producer and consumer:
Performance class and measurement boundary or cost bound:
Five-question result and any justified exception:
```

This record belongs in the capability design note or concise contract; do not
create a separate form file for every capability.

## Large Executors And Fused Kernels

A large executor is acceptable when it composes already contracted functions
and owns only scheduling, workspace, caching, or another explicitly named
execution strategy. It may not introduce new numerical, topology, boundary,
ownership, or failure semantics.

A fused kernel is acceptable after the separate semantic functions and simple
composition are validated. Keep the simple reference composition and verify the
fused implementation against the same contract suite. Fusion is an
implementation strategy, not permission to erase semantic boundaries.

Do not split a coherent hot loop into Python callbacks or payload-sized
temporaries merely to create more functions. Conceptual decomposition and
runtime call structure are separate decisions.

## Refining Existing Implementations

Do not launch a big-bang rewrite of every completed capability. Refine an
existing implementation when one of these triggers occurs:

- a new capability will depend on or extend it;
- a second implementation/backend/strategy needs the same behavior;
- profiling motivates fusion, caching, parallelism, or layout changes;
- it is about to enter a public or real-data integration path;
- a bug or design review reveals hidden state or mixed decisions;
- final migration/cutover audit reaches its feature family.

Use this sequence:

1. **Freeze behavior.** Run and, if necessary, add focused tests, references,
   invariants, and current-comparison evidence before structural edits.
2. **Inventory decisions.** List semantic, planning, storage, execution,
   ownership, error, and optimization decisions currently combined.
3. **Apply the five questions.** Identify independently variable meanings and
   the stable data passed between them.
4. **Extract from the outside inward.** Separate policy/planning and adapters
   first, then semantic primitives where evidence supports the split. Avoid
   rewriting validated arithmetic without need.
5. **Keep compatibility.** Retain the old entrypoint as a thin composition or
   wrapper while consumers migrate.
6. **Prove equivalence.** Run old versus new, independent reference, edge/error,
   multiple-strategy, and preservation tests under the existing metrics.
7. **Measure the change.** Compare kernel and composed runtime, allocation,
   memory, I/O, and scaling. Retain a fused/optimized path when it is valuable
   and contract-equivalent.
8. **Migrate consumers incrementally.** Move one immediate consumer at a time,
   then remove or retire the old mixed implementation only after no supported
   path depends on it.
9. **Update the checkpoint.** Record changed boundaries, intentional
   differences, deferred refinements, evidence, and feature disposition.

Classify earlier work as:

- **conforming:** no structural action;
- **semantically correct but combined:** extract boundaries with a compatibility
  wrapper;
- **hidden-state dependent:** expose state and call order as explicit data;
- **policy-coupled:** move selection into a planner/strategy/adapter;
- **optimized/fused:** retain it and add the missing semantic reference;
- **obsolete or duplicate:** retain only as evidence, then retire through the
  migration workflow.

Completed capabilities are not reopened solely for stylistic purity. Record a
deferred refinement when it has no current consumer, correctness risk,
substitution need, or measurable performance value. Every supported feature
must nevertheless pass this audit before final cutover.

## In-Progress Work Created Under Earlier Rules

Preserve existing work. Before continuing implementation:

1. inspect the working tree and identify files belonging to the active work;
2. read this document and audit the current design/contract with the five
   questions and decision inventory;
3. amend or split the design/contract without discarding valid analysis;
4. update `CAPABILITIES.md` and `CURRENT.md` if dependencies or IDs change;
5. only then implement, test, benchmark, and checkpoint.

Never use a decomposition audit as a reason to reset or overwrite unrelated
user or prior-agent work.
