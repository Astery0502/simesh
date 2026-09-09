# TOP-003 Preserved Draft: Refined Face Cache

This is the preserved pre-decomposition TOP-002 design analysis.  Its compact
six-face representation remains a candidate TOP-003 materialization strategy,
but FST conformance, raw contact lookup, and global balance now have separate
capability boundaries.  Statements below that combine them are historical
draft context, not a frozen contract.

## Problem

M1 consumers need to distinguish physical, same-level, coarser, and finer
contacts for every refined leaf.  The current forest stores two 27-direction
leaf tables plus a 64-position fine-child table, using one-based leaf IDs and
zero with kind-dependent meaning.  That representation combines topology with
the later halo transfer layout and costs 472 bytes per leaf with current
`uint32` entries.

FST-001 now provides a flat octree.  TOP-002 should expose the smallest stable
contact artifact needed by refined geometry, directional relation planning,
support closure, and transfer kernels, while proving the two-to-one assumption
those consumers require.

## Alternatives

A direct `(leaf, 26, 4)` source table makes halo planning simple but repeats
edge/corner policy globally, stores physical composition prematurely, and
costs at least 832 bytes per leaf with rewrite IDs.  A `(leaf, 6, 4)` face
table is smaller but still repeats four fine leaf IDs whose common parent is
already present in FST-001.  Resolving all contacts on demand stores nothing,
but repeats variable-depth tree location in every geometry, planner, and halo
consumer.

A compact six-face kind plus one node ID retains the stable topological fact.
For same/coarser contacts the node is the neighboring leaf.  For a finer
contact it is the internal neighboring node at the source leaf's level.  Under
two-to-one balance, the four child nodes touching the face are exactly the four
fine leaves, in filtered canonical child-column order.

Balance could be checked only on faces.  The current 27-direction refined halo
logic also assumes that edge- and corner-touching leaves differ by at most one
level, so a face-only check would allow later relation planning to encounter an
unsupported diagonal.  TOP-002 therefore validates all 26 touching directions
even though it retains only six face results.

## Selected Direction

Locate a target logical block at the source leaf's level by descending from
the MOR-001 root through FST-001 child columns.  A leaf reached early is a
coarser neighbor, a leaf reached at the target level is same-level, and an
internal node reached at that level is finer.  Physical contacts are detected
from exact integer domain extents.

First validate all FST arrays and every one of the 26 contacts without mutating
outputs.  A coarser leaf may be only one level lower.  For an internal target,
every child that touches the source direction must be a leaf; the number is
four across a face, two across an edge, and one across a corner.  Then fill the
six face columns in TOP-001 order.

The retained output is `uint8[L,6]` kinds plus `int64[L,6]` node IDs, exactly
49 bytes per leaf.  Directional edges/corners, physical post-transforms,
support sets, and fine leaf expansion remain REL-001 responsibilities.

## Performance Decision

The standard workload includes mixed level-1--4 synthetic forests and the
available real level-3--6 Cartesian tree metadata.  The composed comparator is
FST-001 plus TOP-002 versus current `AMRForest` construction, which eagerly
builds its pointer forest and full connectivity tables.  Measure construction
runtime, leaves/s, retained topology bytes, fixed allocation, and depth/size
scaling.

The hypothesis is that flat traversal is faster while reducing retained
topology storage by about 90%.  A reproducible median regression above 20% in
two representative cases blocks completion unless the measured memory
reduction is explicitly judged material.  Independent pairwise box topology
is the semantic reference, not the performance comparator.
