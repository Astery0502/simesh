# AMR Forest and Mesh Notes

## Purpose

This document summarizes how the repository models adaptive mesh refinement
internally, especially the relationship between forest structure, mesh storage,
neighbor connectivity, and ghost-cell operations.

## Core concepts

### Forest

The AMR hierarchy is represented as a forest of level-1 blocks. Each level-1
block acts as the root of an octree in the 3D case and as a quadtree stored in
the first four octree child slots in the Cartesian 2D case.

Key responsibilities of the forest layer:

- reconstruct tree structure from leaf/non-leaf flags
- map traversal order to leaf nodes
- compute neighbor relationships
- distinguish boundary, coarse, sibling, and fine neighbors

In the legacy Python-side implementation, these ideas are mainly in:

- `src/simesh/legacy/geometry/amr/amr_forest.py`
- `src/simesh/utils/octree.py`

In the Cython-backed implementation, the corresponding core lives in:

- `src/simesh/utils/lib/amr/forest.pyx`

### Morton ordering

Morton order is used to map between structured block indices and a space-filling
curve ordering. This is important for:

- storing level-1 blocks consistently
- reconstructing the forest
- matching AMRVAC traversal assumptions
- building leaf lookup tables

Relevant modules:

- `src/simesh/legacy/geometry/amr/morton_order.py`
- `src/simesh/utils/lib/amr/morton.pyx`

### Mesh

The AMR mesh layer stores block-level field data, coarse representations, and
derived coordinate information for each leaf block.

Key responsibilities:

- allocate block data arrays
- map leaves to physical coordinates
- maintain interior and ghost-cell index ranges
- support boundary fill, restriction, and prolongation logic

Primary legacy Python module:

- `src/simesh/legacy/meshes/amr_mesh.py`

Primary Cython module:

- `src/simesh/utils/lib/amr/mesh.pyx`

## Neighbor types

The forest connectivity logic distinguishes several neighbor classes:

- unknown or uninitialized
- physical boundary
- coarse neighbor
- sibling neighbor at the same level
- fine neighbor

This classification drives ghost-cell exchange and coarse/fine interpolation
behavior in the mesh layer.

## Ghost cells and AMR updates

The mesh implementation allocates block storage with ghost cells and uses
neighbor connectivity to fill those regions.

For Cartesian 2D data, the mesh keeps a singleton physical z extent while the
padded storage still has z ghost slots. Ghost exchange, restriction, and
prolongation operate on the active x/y directions and pin data movement to the
single interior z plane.

Important operations in the legacy Python-side mesh include:

- physical boundary fills
- coarsening to temporary coarse storage
- restriction from fine to coarse interfaces
- prolongation from coarse to fine interfaces
- final ghost-cell update passes

This logic is concentrated in methods such as `getbc(...)` and its helper
methods in `src/simesh/legacy/meshes/amr_mesh.py`.

## Coordinate bookkeeping

Each leaf block stores physical extents and cell spacing derived from:

- domain bounds
- domain cell counts
- block cell counts
- leaf refinement level
- block grid indices

This bookkeeping allows later export, interpolation, and block placement on
uniform grids.

The Python AMRVAC API preserves a singleton-z array convention for Cartesian
2D: uniform data is `(nx, ny, 1, nw)` and block data is
`(nleafs, nw, bx, by, 1)`, even though the AMRVAC file metadata uses `ndim=2`.

## Dataset relationship

Datasets bind together:

- header metadata
- forest structure
- tree/block indexing
- AMR mesh data arrays

That means the dataset is the high-level container, while the forest and mesh
are the structural and numerical engines underneath it.

## Practical note

If you need to understand behavior, read in this order:

1. how the forest is reconstructed
2. how connectivity is built
3. how the mesh allocates storage
4. how ghost-cell update logic consumes neighbor types
