"""Private Cartesian 3D v5 serializer adapted from the retained AMRVAC writer."""

import struct

import numpy as np


_FIXED_HEADER = struct.Struct("=10id6d9i16si")
_NAME_BYTES = 16
_GHOST_HEADER = bytes(24)


def _name_bytes(value):
    encoded = value.encode("utf-8")
    if len(encoded) > _NAME_BYTES:
        raise ValueError("AMRVAC header names must fit in 16 bytes")
    return encoded.ljust(_NAME_BYTES, b" ")


def _header_bytes(header):
    count = header["n_par"]
    parameters = np.asarray(header["params"], dtype="=f8")
    if parameters.shape != (count,) or len(header["param_names"]) != count:
        raise ValueError("AMRVAC parameter values and names must match n_par")
    tail = (b"".join(_name_bytes(name) for name in header["w_names"])
            + _name_bytes(header["physics_type"])
            + struct.pack("=i", count) + parameters.tobytes()
            + b"".join(_name_bytes(name) for name in header["param_names"])
            + struct.pack("=3i", header["snapshotnext"], header["slicenext"],
                          header["collapsenext"]))
    header["offset_tree"] = _FIXED_HEADER.size + len(tail)
    # Forest flags, then each leaf's level, three coordinates and int64 offset.
    header["offset_blocks"] = (header["offset_tree"]
                               + 4*(header["nleafs"] + header["nparents"])
                               + 24*header["nleafs"])
    if header["offset_blocks"] > np.iinfo(np.int32).max:
        raise ValueError("AMRVAC tree/header offsets exceed the signed 32-bit representation")
    fixed = _FIXED_HEADER.pack(
        5, header["offset_tree"], header["offset_blocks"], header["nw"],
        header["ndir"], 3, header["levmax"], header["nleafs"], header["nparents"],
        header["it"], header["time"], *header["xmin"], *header["xmax"],
        *header["domain_nx"], *header["block_nx"], *(int(v) for v in header["periodic"]),
        _name_bytes(header["geometry"]), 0,
    )
    return fixed + tail


def _check_position(stream, expected):
    if stream.tell() != expected:
        raise OSError(f"AMRVAC write ended at byte {stream.tell()}, expected {expected}")


def _write_blocks(stream, data):
    position = stream.tell()
    for block in data:
        stream.write(_GHOST_HEADER)
        # AMRVAC stores x fastest, followed by y, z and field; retain float bits.
        stream.write(block.transpose(0, 3, 2, 1).tobytes(order="C"))
        position += len(_GHOST_HEADER) + block.nbytes
        _check_position(stream, position)


def write_datfile_from_sfc(stream, data, header, is_leaf, tree):
    """Write prepared SFC arrays to a new binary stream; the caller publishes it.

    ``data`` is contiguous native float64 in (leaf, field, x, y, z) order.
    ``tree`` contains the leaf levels and one-based three-dimensional coordinates.
    The returned header owns its calculated offsets; caller metadata is unchanged.
    """
    return write_datfile_from_batches(stream, (data,), header, is_leaf, tree)


def write_datfile_from_batches(stream, batches, header, is_leaf, tree):
    """Serialize ordered field-major batches, checking the declared total coverage.

    Batches may reuse their backing after consumption. At most one block's disk
    layout copy is retained; the caller owns publication and batch storage.
    """
    if header.get("staggered", False):
        raise ValueError("ordinary-field writer does not serialize staggered face values")
    if (header.get("datfile_version") != 5 or header.get("ndim") != 3
            or header.get("geometry") != "Cartesian_3D"
            or np.any(header.get("periodic", True))):
        raise ValueError("writer requires nonperiodic Cartesian 3D v5 data")
    leaves, fields = header["nleafs"], header["nw"]
    if leaves < 1 or fields < 1 or len(header["w_names"]) != fields:
        raise ValueError("nonempty blocks and field names must match the header")
    if not isinstance(tree, tuple) or len(tree) != 2:
        raise ValueError("tree must contain block levels and coordinates")
    flags = np.asarray(is_leaf, dtype="=i4")
    levels = np.asarray(tree[0], dtype="=i4")
    coordinates = np.asarray(tree[1], dtype="=i4")
    if (flags.shape != (leaves + header["nparents"],)
            or levels.shape != (leaves,) or coordinates.shape != (leaves, 3)
            or np.any(levels < 1) or np.any(coordinates < 1)):
        raise ValueError("forest and one-based block metadata must match the header")
    written = header.copy()
    encoded_header = _header_bytes(written)
    record_bytes = len(_GHOST_HEADER) + 8*fields*int(np.prod(header["block_nx"]))
    if written["offset_blocks"] + leaves*record_bytes > np.iinfo(np.int64).max:
        raise ValueError("AMRVAC block offsets exceed the signed 64-bit representation")
    offsets = written["offset_blocks"] + np.arange(leaves, dtype="=i8")*record_bytes
    stream.write(encoded_header)
    _check_position(stream, written["offset_tree"])
    for array in (flags, levels, coordinates, offsets):
        stream.write(memoryview(np.ascontiguousarray(array)).cast("B"))
    _check_position(stream, written["offset_blocks"])
    del flags, levels, coordinates, offsets, array
    count = 0
    for data in batches:
        if (not isinstance(data, np.ndarray) or data.dtype != np.float64 or
                not data.flags.c_contiguous or data.ndim != 5 or
                data.shape[1:] != (fields, *header["block_nx"]) or
                not len(data) or count + len(data) > leaves):
            raise ValueError("SFC batches must match the declared block shape and count")
        _write_blocks(stream, data)
        count += len(data)
    if count != leaves:
        raise ValueError("SFC batches do not cover all declared blocks")
    _check_position(stream, written["offset_blocks"] + leaves*record_bytes)
    return written
