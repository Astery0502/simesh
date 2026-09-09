# TOP-003 Preserved Contract Draft: Balanced Refined Face Cache

This unimplemented draft is preserved from the pre-decomposition TOP-002 work.
It is design input for optional TOP-003 only and is not a stable contract.  The
audited corrections are: reusable FST conformance, raw contact lookup, and
all-touch balance are separate capabilities; retained bytes are `54*L`, not
`49*L`; and face materialization must be justified against on-demand contact
composition before implementation.

## Responsibility

TOP-002 validates the complete FST-001 flat forest, proves Cartesian 3D
two-to-one balance across every face-, edge-, and corner-touching leaf pair,
and emits one compact relation for each leaf face.  It defines no floating
geometry, directional halo plan, support closure, field values, transfer
weights, periodicity, or physical boundary transform.

## Stable Operation

```text
fill_refined_face_topology(
    root_shape,
    coord_to_rank,
    root_node_ids,
    node_levels,
    node_coords,
    parent_node_ids,
    child_node_ids,
    node_leaf_ids,
    leaf_node_ids,
    face_kinds,
    face_neighbor_node_ids,
) -> None
```

Let `R = product(root_shape)`, `K = len(node_levels)`, and
`L = len(leaf_node_ids)`.

- `root_shape` and `coord_to_rank` retain MOR-001 shapes and meanings.
- The seven forest arrays retain their exact FST-001 meanings and shapes:
  roots `(R,)`, node vectors `(K,)`, node coordinates `(K,3)`, children
  `(K,8)`, and leaves `(L,)`.
- All input index arrays are borrowed read-only, native-endian C-contiguous
  signed `int64`.
- `face_kinds` is caller-owned writable C-contiguous `uint8` with shape
  `(L,6)`.
- `face_neighbor_node_ids` is caller-owned writable native-endian C-contiguous
  `int64` with shape `(L,6)`.
- Outputs may not overlap an input or each other.  Successful execution
  overwrites both outputs completely, changes no input, and retains nothing.

`refined_face_topology(...)` allocates and returns both output arrays after the
same validation.

## Face Order, Kinds, And Node IDs

Columns are exactly TOP-001 order `(xlo,xhi,ylo,yhi,zlo,zhi)`, with
`face = 2*axis + side` and opposite face `face ^ 1`.

Kinds are exact unsigned values matching current semantic classes:

| Value | Name | Meaning | Neighbor node ID |
| ---: | --- | --- | --- |
| 1 | `PHYSICAL` | face lies on the non-periodic domain boundary | `-1` |
| 2 | `COARSER` | one neighboring leaf is one level coarser | that leaf node |
| 3 | `SAME` | one neighboring leaf is at the same level | that leaf node |
| 4 | `FINER` | four neighboring leaves are one level finer | their common internal parent at the source level |

Zero is not a valid face kind.  Every non-physical node ID is in `[0,K)`.
For SAME and COARSER, `node_leaf_ids[id] >= 0`.  For FINER,
`node_leaf_ids[id] == -1`, its level equals the source level, and the four
children touching the source face are leaves.  Those leaves are obtained by
filtering child columns `0..7` in increasing order: the target-side bit on the
face axis is fixed opposite the source displacement, while both transverse
bits vary.

## Structural Validation And Balance

Before output mutation, TOP-002 recursively proves the supplied arrays form
the exact FST-001 preorder reconstruction:

- roots follow dense `coord_to_rank` order and partition all nodes;
- levels, coordinates, parents, and canonical child columns are reciprocal;
- leaf and node maps are dense exact inverses;
- leaf children and internal node/leaf sentinels are exact;
- complete logical level extents remain representable in signed `int64`.

For every leaf and every nonzero direction in `{-1,0,1}^3`, locate the
touching target region at the source level.  An out-of-domain direction is
physical.  Otherwise:

- a target leaf reached below the source level must be exactly one level
  coarser;
- a target leaf at the source level is same-level;
- a target internal node at the source level must have every direction-touching
  child as a leaf one level finer.

This is equivalent to `abs(level_a-level_b) <= 1` for all leaves whose closed
boxes touch at a face, edge, or corner.  A larger difference raises
`ValueError` identifying the source leaf and direction.  Balance validation is
global even though only face relations are retained.

## Exactness And Invariants

All TOP-002 outputs are discrete and exact.

- Physical occurs exactly on a domain face.
- SAME is reciprocal with SAME and the opposite face points back.
- A COARSER face's opposite coarse face is FINER; the FINER child set contains
  the source leaf node exactly once.
- A FINER face expands to exactly four leaves, each having a reciprocal
  COARSER face to the source node.
- Face contact areas cover the source face exactly without overlap or gaps.
- An all-level-1 forest reduces exactly to TOP-001 after SAME node IDs are
  converted through `node_leaf_ids`.

Correctness is established by an independent common-finest-lattice pairwise
box reference, current 27/64-table expansion, synthetic mixed-depth cases, the
representative real refined tree, malformed FST tables, and explicit
face-balanced/diagonally-unbalanced cases.

## Errors, Mutation, Memory, And Performance

Type/dtype errors raise `TypeError`.  Invalid shape, contiguity, writability,
overlap, forest structure, relation, or balance raises `ValueError`.
Unrepresentable root volume or logical level extent raises `OverflowError`.
Every failure precedes output mutation.

Retained output storage is exactly `49*L` bytes, compared with 472 bytes per
leaf for the current two 27-entry `uint32` tables plus 64-entry fine-child
table.  Validation uses `O(max_level)` C stack and no heap or per-leaf scratch.
Construction is `O(K + 26*L*max_level)` in the worst case.  Runtime, leaves/s,
fixed allocation, retained bytes, depth scaling, current composed comparison,
and the immediate fine-child expansion are measured.  Parallel scaling is not
relevant to this one-time metadata capability.

## Immediate Composition

FST-001 produces every structural input.  GEO-002 consumes leaf levels and
coordinates independently.  REL-001 consumes TOP-002 face kinds and node IDs
together with the FST hierarchy to build bounded 26-direction source plans;
it owns edge/corner expansion and mixed physical/source composition.
