"""Independent small AMR data and binary-v5 fixtures for composed checks."""

import struct
import numpy as np
import simesh as sm


def mixed_source(function=None):
    mesh = sm.mesh_from_forest((2, 1, 1), np.array([False]+[True]*9),
                              lower=(0, 0, 0), upper=(2, 1, 1), block_shape=(8, 8, 8))
    if function is None:
        function = lambda x, y, z: np.array([y+2*z, 3*z+4*x, 5*x+6*y])
    local = np.indices(mesh.block_shape)+.5
    values = np.empty((mesh.leaf_count, 3, *mesh.block_shape))
    for leaf in range(mesh.leaf_count):
        xyz = mesh.bounds[leaf, 0, :, None, None, None] + local*mesh.spacing[leaf, :, None, None, None]
        values[leaf] = function(*xyz)
    return sm.source_from_arrays(mesh, values, ("b1", "b2", "b3"), copy=False), values


def write_dat(path, mesh, values, *, byte_order="<", staggered=False, saved_ghosts=False,
              geometry="Cartesian_3D"):
    """Write a minimal v5 fixture directly from the documented binary layout."""
    flags = mesh.node_leaves >= 0
    nleaf, fields = values.shape[:2]
    names = [f"b{i+1}" for i in range(fields)]
    fixed_format = byte_order+"10id6d9i16si"
    offset_tree = struct.calcsize(fixed_format)+16*fields+32
    offset_blocks = offset_tree+len(flags)*4+nleaf*24
    records = []
    for leaf in range(nleaf):
        lo = np.array((leaf % 2, 0, 1) if saved_ghosts else (0, 0, 0))
        hi = np.array((0, leaf % 2, 0) if saved_ghosts else (0, 0, 0))
        shape = np.array(mesh.block_shape)+lo+hi
        backing = np.full((fields, *shape), 0x0123456789ABCDEF, dtype=np.uint64)
        box = tuple(slice(int(a), int(a+b)) for a, b in zip(lo, mesh.block_shape))
        backing[(slice(None), *box)] = values[leaf].view(np.uint64)
        raw = np.asarray([*lo, *hi], dtype=byte_order+"i4").tobytes()
        raw += np.asarray(backing.transpose(0, 3, 2, 1), dtype=byte_order+"u8").tobytes()
        if staggered:
            raw += bytes(3*int(np.prod(shape+1))*8)
        records.append(raw)
    offsets = offset_blocks+np.r_[0, np.cumsum([len(record) for record in records[:-1]])]
    header = struct.pack(fixed_format, 5, offset_tree, offset_blocks, fields, 3, 3,
                         mesh.forest.max_level, nleaf, len(flags)-nleaf, 0, 0.,
                         *mesh.lower, *mesh.upper, *(mesh.root_shape*np.array(mesh.block_shape)),
                         *mesh.block_shape, *(int(flag) for flag in mesh.periodic), geometry.encode().ljust(16, b" "), int(staggered))
    header += b"".join(name.encode().ljust(16, b" ") for name in names)
    header += b"mhd".ljust(16, b" ")+struct.pack(byte_order+"4i", 0, 0, 0, 0)
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(flags.astype(byte_order+"i4").tobytes())
        stream.write(mesh.forest.node_levels[mesh.leaf_nodes].astype(byte_order+"i4").tobytes())
        stream.write((mesh.forest.node_coords[mesh.leaf_nodes]+1).astype(byte_order+"i4").tobytes())
        stream.write(offsets.astype(byte_order+"i8").tobytes())
        for record in records:
            stream.write(record)
