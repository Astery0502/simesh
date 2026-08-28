# Functional Composition Boundaries

## Purpose

The rewrite uses functions with explicit inputs, outputs, state, and workspace
so scientific semantics can be composed independently of storage, scheduling,
and compute framework choices. This document defines where implementations may
be substituted without changing those semantics.

Functional does not mean that large arrays are immutable. It means mutation is
limited to declared caller-owned outputs or workspaces, dependencies are
visible in the function signature, and correctness does not depend on hidden
object history or global backend selection.

## Stable Layering

```text
external storage or framework
        |
        v
functional block reader/writer adapter
        |
        v
resident / bounded / cached execution strategy
        |
        v
canonical block IDs + canonical payload workspace
        |
        +--> topology / geometry / access requirements
        +--> halo support and transfer functions
        +--> sampling / operators / reductions
```

The canonical workspace is the interoperability boundary for the current
Cython core. An adapter may use a resident array, `numpy.memmap`, native
AMRVAC reads, HDF5, Zarr, a cache, or another framework, but it must implement
the same selected-region transfer contract. A framework that wants to avoid
the canonical host workspace must independently implement and validate the
complete affected compute contracts; array similarity is not enough.

## Storage Boundary

A block reader or writer is an immutable descriptor containing:

- explicit backend state;
- canonical `(block, field, x, y, z)` shape;
- one coarse-grained read or write function;
- arrays that participate in memory-alias validation, when applicable.

The function is invoked once per planned transfer, never once per cell. Reader
and writer functions do not choose primary blocks, support closure, halo rules,
or operators. The array adapter is the reference implementation and accepts
both resident NumPy arrays and memory maps. A future native AMRVAC adapter may
use `pread` or another block-aware mechanism without changing consumers.

Storage failures are distinct from contract failures. Array adapters retain
the current preflight and overlap guarantees. External adapters must document
whether an I/O failure can leave an external sink partially written; the
numerical executor is not an implicit distributed transaction.

## Execution Strategies

An execution strategy selects traversal and workspace policy:

- resident: all useful blocks are available and chunk capacity may cover the
  complete working set;
- bounded: primary/support chunks fit an explicit managed byte budget;
- cached or hybrid: a bounded executor retains selected support data across
  chunks;
- framework-specific: another compute backend implements the same contracts
  and conformance suite.

Strategies may differ in runtime, I/O volume, caching, parallel scheduling, and
temporary allocation. They may not silently change block ownership, floating
operation order where fixed, halo semantics, or reduction order where the
contract requires capacity invariance.

The current M0 executor remains the validated bounded reference. A resident
strategy may initially use full capacity and the same workspace; zero-copy or
fused resident paths require separate evidence because halo padding and output
ownership can still require explicit buffers.

## Halo Decomposition

Halo behavior is not one dispatcher. It is the composition of:

1. operator access requirements describing lower/upper reach;
2. topology and geometry relations classifying physical, same-level,
   coarse/fine, and periodic sources;
3. support planning deciding which block interiors must be available;
4. value functions for physical transforms, same-level copies,
   prolongation, restriction, and periodic transfers;
5. an executor applying those functions to explicit primary outputs.

HAL-001 remains the physical-value primitive and HAL-002 remains the validated
Cartesian 3D level-1 same-level primitive. Refined, 2D, and periodic work adds
new plans and functions rather than turning HAL-002 into a mode-dependent
framework. Common behavior is factored only after the concrete contracts show
that it is truly shared.

## Conformance Requirements

Every alternative backend or strategy must demonstrate:

- exact block/field/region selection and preservation outside declared writes;
- the same topology, halo, sampling, operator, and reduction results under the
  metrics of their existing contracts;
- explicit state lifetime, ownership, alias, and failure behavior;
- bounded-memory accounting where a budget is claimed;
- measured transfer calls, loaded/support amplification, runtime, and memory;
- resident-array comparison as the reference composition.

Tests should include reordered and repeated fields, nontrivial block order,
multiple capacities, physical and sibling halos, and a backend whose state is
not itself a NumPy array. This proves that substitution is structural rather
than an accidental consequence of `numpy.memmap` inheriting from `ndarray`.
