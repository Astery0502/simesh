# Capability Decomposition And Refinement

## Purpose

Use this page when a concrete design has unclear responsibility or coupling.
It is a design aid, not a prerequisite audit for every capability, extension,
fusion or resumed task. Keep coherent numerical work together and make decisions
explicit where that helps a real consumer or implementation change.

<a id="five-required-questions"></a>

## Design Questions

Use only questions that clarify the current choice. There is no required form,
pass/fail score or automatic split after a negative answer.

### 1. Is The Semantic Responsibility Singular?

Describe the useful responsibility. Separate independently changing decisions
when doing so improves clarity or reuse. The word "and" in a description, an
axis branch or a combined hot loop does not itself justify another abstraction.

### 2. Can It Be Validated Independently?

Identify evidence that can establish the behavior: direct reasoning, an existing
reference or a representative consumer. A helper need not have its own test if
inspection or an existing composed case adequately covers it. Core numerical or
memory uncertainty may justify a focused independent example.

### 3. Can Its Implementation Be Substituted Independently?

When an actual second strategy or backend is relevant, identify the meaning its
consumer relies on. Keep algorithm details local where useful, without building
an abstract interface or duplicate implementation for a hypothetical replacement.

### 4. Is Its Cost Evaluated At The Right Boundary?

Measure the operation the user needs when the change affects performance.
Isolated kernel timing is useful only when it explains that cost or helps choose
an implementation. Cold metadata or simple configuration work may be settled by
reading. No performance class or benchmark is required for every helper; use the
[verification policy](../docs/analysis-core/performance.md#choose-the-smallest-useful-check).

### 5. Are Semantic Meaning And Policy/Execution Separate?

Make numerical meaning, ownership and chosen execution policy explicit. They can
be represented as data and executed together in a compiled loop. Do not add
Python callbacks, per-cell objects or intermediate arrays solely to enforce a
conceptual decomposition.

## Decision Ownership Rule

Inspect the decisions relevant to the change rather than inventory every branch.
Separate source interpretation, numerical strategy, resource policy or mutable
state when their coupling causes a real ambiguity, correctness risk or cost.

Keep ordinary axis/direction cases, boundary-owned validation and arithmetic
belonging to one operation together. A fused planned transfer executor can contain
copy/restriction/prolongation cases under explicit meanings. An internal iteration
order must not silently choose a user's field ownership or numerical strategy,
but its incidental call sequence does not need a dedicated test.

## Record The Boundary Once

For a substantive choice, use [WORKFLOW's short record](WORKFLOW.md#one-design-record)
and link the existing contract. Record the decision and why it matters, not an
audit transcript. Small readable edits need no separate decomposition document,
justification packet or reviewer sign-off.

## Large Executors And Fused Kernels

Judge an executor by explicit inputs, behavior, ownership and actual consumer
cost, not line count or the number of helper calls. Preserve the selected
arithmetic, validity and failure meaning through fusion. Existing source and
references can establish simple behavior; use focused core checks where fusion
creates a numerical or memory question. It is not necessary to implement and
test every constituent as a stand-alone API before writing a useful fused kernel.

## Refining Existing Implementations

Refine code for an actual consumer, a concrete bug or a demonstrated cost.
Start by reading the relevant implementation/contract and existing evidence.
Make the smallest useful structural change and verify the affected behavior
using the [smallest useful check](../docs/analysis-core/performance.md#choose-the-smallest-useful-check).
Measure only relevant performance/resource claims and reuse adequate prior runs.

Preserve public behavior or state an intentional scoped change. Update the active
checkpoint and any genuinely changed contract/status. There is no mandatory
nine-step refinement audit, separate tests for metadata/order/each quantity, or
reclassification of every existing feature. Production cutover still establishes
its supported observable scope using appropriate evidence.

## In-Progress Work Created Under Earlier Rules

Inspect the working tree, read CURRENT and resume the selected task. Revisit a
decision only when the request, dependency, evidence or a real failure changes it.
Do not restart a five-question audit merely because the context changed, or
reset valid user/agent work to comply with a process template.
