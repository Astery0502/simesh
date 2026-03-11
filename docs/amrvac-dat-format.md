# AMRVAC `.dat` Format Notes

## Scope

This document explains how this repository handles the AMRVAC-style `.dat`
snapshot format. It is not meant to be a complete external specification of the
format; it is a code-oriented guide to how `simesh` reads and writes it.

## Main ownership

The primary file-format implementation lives in:

- `src/simesh/amrvac/datio.py`
- `src/simesh/amrvac/amrvac_dataset.py`

The first module handles low-level binary parsing and writing. The second module
provides the canonical dataset object built on top of that format layer.

## File structure as modeled here

The code treats an AMRVAC `.dat` file as four main components:

1. header metadata
2. forest leaf/parent flags
3. tree information for leaf blocks
4. block field payloads

### Header

The header includes metadata such as:

- file version
- offsets to tree and block sections
- number of variables and dimensions
- refinement limits
- number of leaves and parents
- iteration/time info
- physical domain bounds
- domain and block cell counts
- periodicity and geometry
- field names
- physics parameters

`get_header(...)` in `datio.py` is the main parser.

### Forest

The forest is read as a flat boolean-like sequence indicating whether each node
in traversal order is a leaf. `get_forest(...)` returns this as the structural
input used to reconstruct the AMR tree.

### Tree information

`get_tree(...)` reads leaf-level information:

- block refinement levels
- block indices
- block byte offsets

These offsets are later used to locate per-block data payloads in the file.

### Block data

Each leaf block stores field values, optionally with ghost-cell metadata and
staggered data depending on the file header. `get_single_block_data(...)` and
`get_single_block_field_data(...)` provide block-level access.

## Read path

The typical read flow is:

1. open file
2. parse header
3. parse forest flags
4. parse tree info
5. reconstruct forest connectivity
6. create mesh object
7. create dataset object
8. load block data into mesh storage

In the canonical path, the dataset entrypoint is `AMRVACDataSet(...)` in
`src/simesh/amrvac/amrvac_dataset.py`.

## Write path

The typical write flow is split into metadata and field writes:

1. create or prepare an AMR mesh
2. assemble header values
3. compute tree and block offsets
4. write header
5. write forest and tree sections
6. write field payloads block by block

Main canonical helpers:

- `write_header(...)`
- `write_forest_tree(...)`
- `write_blocks(...)`

This split is useful when metadata and data arrays are prepared at different
stages.

## Construction from arrays

The repository also contains legacy Python-first AMRVAC construction helpers
under `src/simesh/legacy/frontends/amrvac/`. Those are preserved as reference
and fallback code rather than the canonical current path.

## Caveats

- The implementation is specialized toward current supported use cases rather
  than a broad AMRVAC compatibility matrix.
- Several code paths assume Cartesian 3D data.
- Staggered support exists in the low-level parser, but not every higher-level
  workflow appears equally mature.
