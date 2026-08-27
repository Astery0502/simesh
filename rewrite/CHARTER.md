# Rewrite Charter

## Purpose

Build a new Cython AMR computational core from first principles in an isolated
directory. The rewrite should make domain concepts explicit, keep individual
responsibilities small, compose them efficiently, and remain understandable to
agents working over a long period.

The rewrite is also an investigation. Differences from the current
implementation should be used to clarify the real semantics of the project,
not merely copied or suppressed.

## Required Outcome

The rewrite must preserve supported numerical behavior. It does not need to
preserve the current classes, public API, internal algorithms, memory layout,
or byte-for-byte file output.

Discrete artifacts such as Morton mappings, topology, neighbor classes, field
ordering, and exact block placement should agree exactly when their contracts
are exact. Floating-point operators should use an explicit, operation-specific
comparison appropriate to their numerical meaning.

## Functional Core

"Functional" means explicit transformations rather than hidden object state:

- every dependency is an explicit input;
- output buffers and scratch workspaces have explicit ownership;
- mutation is limited to declared output or workspace buffers;
- kernels do not depend on an undocumented call history;
- the same inputs and execution strategy produce deterministic results;
- each function or capability represents one coherent domain responsibility.

This does not require immutable large arrays. In-place writes and reusable
buffers are expected where they improve performance and preserve a clear
contract.

## Design Principles

1. Establish semantics before optimization.
2. Separate logical fields, physical storage, operator requirements, and execution plans.
3. Separate topology, geometry, ghost-value rules, and ghost storage.
4. Keep Python orchestration out of Cython hot loops.
5. Prefer simple typed kernels with explicit buffers over hidden mutable classes.
6. Keep a clear reference path before introducing fused or parallel variants.
7. Treat runtime, throughput, peak memory, and scaling as a multi-objective trade-off.
8. Design field payload processing so the complete dataset need not fit in memory.
9. Add abstractions only after concrete capabilities demonstrate a shared need.
10. Keep the development process light; do not add hashes, signatures, authentication, or defensive ceremony.
11. Build from low-level semantic functions upward, then optimize both individual kernels and their composed execution paths.

## Support Sequence

1. Cartesian 3D, level-1, non-periodic, non-staggered.
2. Cartesian 3D refined AMR, non-periodic.
3. Cartesian 2D, including refined AMR and the singleton-z external convention.
4. Cartesian periodic meshes.

Staggered data and non-Cartesian geometries are outside the current plan. They
may be recognized and rejected clearly, but should not influence early core
design beyond avoiding an unnecessary dead end.

## Kernel Extension

New scientific kernels may require rebuilding the Cython extension. Runtime
callbacks, a dynamic expression language, JIT compilation, and automatic code
generation are out of scope until several real kernels demonstrate a stable
common interface.

## Autonomy

Agents may explore implementations and revise development contracts
autonomously. Contract changes require independent sub-agent review and
supporting technical evidence, but do not require user approval. The same rule
may be used to revise this charter when the rewrite itself reveals a better
model.
