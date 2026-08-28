# Rewrite Roadmap

The roadmap is ordered by semantic dependency. It may evolve through the
contract-change workflow, but each milestone must remain independently usable
and verifiable.

## Foundation

Establish the common vocabulary and the smallest executable contracts:

- scalar, index, and layout conventions;
- explicit array ownership and valid regions;
- operator access patterns;
- block sources, block sinks, workspaces, and memory budgets;
- a minimal build and test path isolated from the current package.

## M0: Cartesian 3D Level-1

Status: complete.

Target: a complete non-periodic, non-staggered 3D path without refinement.

Required capabilities:

- layout and block-index primitives;
- level-1 Morton ordering;
- validated level-1 topology and Cartesian block geometry;
- in-memory and bounded block/chunk field sources;
- physical-boundary and same-level halo provision;
- exact level-1 placement, zero-order sampling, and trilinear sampling;
- one pointwise operator, one stencil operator, and one streaming reduction;
- numerical comparison with the current implementation;
- a bounded-memory path whose complete field payload need not reside in memory.

The existing AMRVAC reader may initially feed the new core through an adapter.
A native block source should be added when it is needed to demonstrate genuine
out-of-core execution.

## M1: Cartesian 3D Refined AMR

Before refined halo work, complete the functional-composition checkpoint:

- expose block readers and writers as explicit coarse-grained function
  adapters over canonical buffers;
- retain array/memmap behavior as one backend rather than a semantic
  dependency;
- keep resident and bounded traversal as interchangeable execution strategies;
- separate halo requirements, support closure, relation planning, and value
  transfer before adding coarse/fine behavior.

Add:

- validated parent/child reconstruction;
- coarse, sibling, and fine neighbor relations;
- restriction and prolongation;
- refined ghost provision;
- refined zero-order and trilinear sampling;
- refined real-data comparison.

M1 storage and halo implementations must compose through the same canonical
contracts. Native AMRVAC, mapped, resident, cached, or other adapters may alter
I/O and scheduling but not refined numerical semantics.

## M2: Cartesian 2D

Generalize established concepts to active x/y dimensions:

- quadtree traversal;
- singleton-z external arrays;
- 2D halo provision and refined interfaces;
- exact placement, bilinear sampling, and operator behavior.

Do not implement 2D as unrelated duplicate logic. Generalize only after the 3D
contracts make the shared and dimension-specific parts visible.

## M3: Periodic Cartesian Meshes

Add periodicity as an explicit topology and halo input, then validate periodic
connectivity, refined periodic interfaces, sampling, and stencil behavior.

## Later Scientific Operators

Once the operator contracts are established, add representative families:

- pointwise field transforms;
- local stencils;
- AMR block operators;
- associative reductions;
- geometry-aware sampling;
- field-line traversal and integration.

Field-line work should compose a spatial locator, field sampler, stepper,
termination policy, and reducer. It should not be forced into the local stencil
kernel interface.
