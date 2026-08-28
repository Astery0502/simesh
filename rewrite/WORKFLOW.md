# Capability Workflow

## Capability Cycle

For one ready capability:

1. Recover the relevant behavior from the current implementation and domain meaning.
2. For a non-trivial capability, write a short design note under `designs/` that compares the real alternatives and their composition, memory, and performance trade-offs.
3. Select an approach and write or refine its stable contract under `contracts/`.
4. Define exact outputs, floating-point comparisons, and focused invariants.
5. Build the smallest clear Python or NumPy reference when it adds independent evidence.
6. Implement the simplest Cython form with explicit inputs, outputs, and workspace.
7. Compare intermediate artifacts, not only the final user-visible result.
8. Integrate with the immediate lower-level producer and next higher-level consumer.
9. Measure the performance and memory dimensions relevant to this capability and its new composed path.
10. Explore optimized or fused variants only after the simple implementation and composition are correct.
11. Run the capability tests, all established rewrite regression and integration tests, and relevant current-implementation comparisons.
12. Update the capability ledger and current checkpoint.
13. Inspect the working and staged diffs, then stage only the active capability and checkpoint files.
14. Create one cohesive Git commit for the completed capability.

Keep this cycle proportional. A simple index transform does not need a report;
a refined ghost algorithm or out-of-core executor does.

## Design And Specification

A design note is exploratory. It explains the problem, the few credible
approaches, and the trade-offs that affect later composition. It may change or
be discarded as evidence improves.

A contract is the selected specification. It defines the behavior that the
implementation and its consumers may rely on. Implementation starts only after
that contract is clear enough to test.

Do not require separate design and contract files for trivial capabilities. Do
not turn design notes into long reports. Their purpose is to preserve the
reasoning that a later agent would otherwise have to rediscover.

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

## Optimization

An optimization begins with a concrete hypothesis. Keep the simple validated
implementation available while testing variants such as loop fusion, layout
changes, workspace reuse, tiling, mapping, caching, streaming, or OpenMP.

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

When evidence shows that a stable contract may be wrong or incomplete:

1. The primary agent writes a short proposed change with the conflicting evidence and affected capabilities.
2. The primary agent starts an independent sub-agent reviewer and asks it to inspect the contract, raw evidence, reference behavior, and invariants before relying on the primary conclusion.
3. The reviewer either agrees, rejects the change, or requests one concrete experiment.
4. If both agree, the primary agent updates the contract, implementation, tests, capability ledger, and checkpoint, then continues without user approval.
5. If they disagree, run the smallest experiment likely to resolve the disagreement and review again. Keep the existing contract until agreement is reached; continue other ready work when possible.

The review is a technical second perspective, not an approval ceremony. Keep
the proposal and conclusion concise.

## Checkpoint Discipline

After each completed capability or material decision, `CURRENT.md` should say:

- active milestone;
- last completed capability;
- current capability and status;
- important unresolved differences;
- next ready capability;
- exact commands needed to reproduce the latest checks.

The Git commit created after a completed capability is the durable recovery
point. Keep commits cohesive and executable. Intermediate experiments do not
need their own process artifacts or commits unless retaining them is useful for
an active comparison. Before committing, inspect `git status`, the working diff,
and the staged diff; run the established rewrite regression suite; and ensure
that unrelated repository changes remain unstaged.
