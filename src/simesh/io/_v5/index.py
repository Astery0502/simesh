"""Safe AMRVAC v5 metadata indexing and canonical forest binding."""

from __future__ import annotations

import os
import stat
import struct
from typing import Final, NamedTuple

import numpy as np

from simesh._amr.forest import RefinedForest, refined_forest
from simesh._amr.forest_conformance import validate_refined_forest_arrays
from simesh._amr.morton import level1_morton


_INT64_MAX: Final = int(np.iinfo(np.int64).max)
_NAME_BYTES: Final = 16
_VERSION_BYTES: Final = 4
_FIXED_HEADER_BYTES: Final = 152
_GHOST_HEADER_BYTES: Final = 24


class AMRVACV5Index(NamedTuple):
    """Owned canonical metadata for one unchanged AMRVAC v5 file."""

    byte_order: str
    file_identity: tuple[int, int, int, int, int]
    offset_tree: int
    offset_blocks: int
    field_count: int
    direction_count: int
    dimension_count: int
    declared_max_level: int
    leaf_count: int
    parent_count: int
    iteration: int
    time: float
    domain_lower: np.ndarray
    domain_upper: np.ndarray
    domain_cell_counts: np.ndarray
    block_cell_counts: np.ndarray
    periodic: np.ndarray
    geometry: str
    staggered: bool
    field_names: tuple[str, ...]
    physics_type: str
    parameter_values: np.ndarray
    parameter_names: tuple[str, ...]
    snapshot_next: int
    slice_next: int
    collapse_next: int
    forest_flags: np.ndarray
    block_levels: np.ndarray
    block_coordinates: np.ndarray
    block_offsets: np.ndarray


class AMRVACV5ForestBinding(NamedTuple):
    """Canonical FST artifact proven to use the index's file leaf order."""

    source_file_identity: tuple[int, int, int, int, int]
    root_shape: np.ndarray
    coord_to_rank: np.ndarray
    rank_to_coord: np.ndarray
    forest: RefinedForest


def _checked_nonnegative_i64(value: int, description: str) -> int:
    if value < 0 or value > _INT64_MAX:
        raise OverflowError(f"{description} does not fit in signed int64")
    return value


def _checked_add(left: int, right: int, description: str) -> int:
    left = _checked_nonnegative_i64(left, description)
    right = _checked_nonnegative_i64(right, description)
    if left > _INT64_MAX - right:
        raise OverflowError(f"{description} does not fit in signed int64")
    return left + right


def _checked_mul(left: int, right: int, description: str) -> int:
    left = _checked_nonnegative_i64(left, description)
    right = _checked_nonnegative_i64(right, description)
    if left != 0 and right > _INT64_MAX // left:
        raise OverflowError(f"{description} does not fit in signed int64")
    return left * right


def _checked_product(values: np.ndarray, description: str) -> int:
    result = 1
    for raw_value in values:
        result = _checked_mul(result, int(raw_value), description)
    return result


def _pread_exact(file_descriptor: int, byte_count: int, offset: int) -> bytes:
    """Read an exact regular-file range without changing the fd cursor."""
    byte_count = _checked_nonnegative_i64(byte_count, "read byte count")
    offset = _checked_nonnegative_i64(offset, "read offset")
    _checked_add(offset, byte_count, "read range")
    chunks: list[bytes] = []
    received = 0
    while received < byte_count:
        chunk = os.pread(file_descriptor, byte_count - received, offset + received)
        if not chunk:
            raise ValueError(
                f"AMRVAC metadata is truncated at byte {offset + received}"
            )
        chunks.append(chunk)
        received += len(chunk)
    if len(chunks) == 1:
        return chunks[0]
    return b"".join(chunks)


def _file_identity(file_descriptor: int) -> tuple[int, int, int, int, int]:
    result = os.fstat(file_descriptor)
    if not stat.S_ISREG(result.st_mode):
        raise ValueError("file_descriptor must name an open regular file")
    size = int(result.st_size)
    _checked_nonnegative_i64(size, "file size")
    return (
        int(result.st_dev),
        int(result.st_ino),
        size,
        int(result.st_mtime_ns),
        int(result.st_ctime_ns),
    )


def _decode_fixed_string(raw: bytes, description: str) -> str:
    try:
        return raw.rstrip(b" \x00").decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError(f"{description} is not valid UTF-8") from error


def _native_float64(raw: bytes, byte_order: str) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.dtype(f"{byte_order}f8")).astype(
        np.float64,
        copy=True,
    )


def _require_positive_count(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validated_root_shape(
    domain_cell_counts: np.ndarray,
    block_cell_counts: np.ndarray,
) -> np.ndarray:
    if np.any(domain_cell_counts <= 0):
        raise ValueError("domain_cell_counts entries must be positive")
    if np.any(block_cell_counts <= 0):
        raise ValueError("block_cell_counts entries must be positive")
    if np.any(domain_cell_counts % block_cell_counts != 0):
        raise ValueError(
            "domain_cell_counts must be exactly divisible by block_cell_counts"
        )
    root_shape = np.empty(3, dtype=np.int64)
    for axis in range(3):
        root_shape[axis] = (
            int(domain_cell_counts[axis]) // int(block_cell_counts[axis])
        )
    _checked_product(root_shape, "root-grid volume")
    return root_shape


def _validate_leaf_coordinates(
    disk_coordinates: np.ndarray,
    block_levels: np.ndarray,
    root_shape: np.ndarray,
) -> np.ndarray:
    if np.any(disk_coordinates <= 0):
        raise ValueError("on-disk block coordinates must be positive")
    coordinates = disk_coordinates.astype(np.int64, copy=True)
    coordinates -= 1
    for leaf in range(block_levels.shape[0]):
        level = int(block_levels[leaf])
        shift = level - 1
        if shift >= 63:
            raise OverflowError(
                f"logical grid at block {leaf}, level {level} does not fit in int64"
            )
        scale = 1 << shift
        for axis in range(3):
            root_extent = int(root_shape[axis])
            if root_extent > _INT64_MAX // scale:
                raise OverflowError(
                    f"logical grid at block {leaf}, level {level} "
                    "does not fit in int64"
                )
            extent = root_extent * scale
            if int(coordinates[leaf, axis]) >= extent:
                raise ValueError(
                    f"block {leaf} coordinate on axis {axis} lies outside "
                    f"the level-{level} logical grid"
                )
    return coordinates


def read_amrvac_v5_index(file_descriptor: int) -> AMRVACV5Index:
    """Decode one checked AMRVAC v5 3D metadata and leaf-record index."""
    if type(file_descriptor) is not int:
        raise TypeError("file_descriptor must be an exact Python int")
    if file_descriptor < 0:
        raise ValueError("file_descriptor must be nonnegative")

    identity_before = _file_identity(file_descriptor)
    file_size = identity_before[2]

    version_raw = _pread_exact(file_descriptor, _VERSION_BYTES, 0)
    little_version = struct.unpack("<i", version_raw)[0]
    big_version = struct.unpack(">i", version_raw)[0]
    orders = []
    if little_version == 5:
        orders.append("<")
    if big_version == 5:
        orders.append(">")
    if len(orders) != 1:
        raise ValueError("unsupported or ambiguous AMRVAC file version")
    byte_order = orders[0]

    fixed_raw = _pread_exact(
        file_descriptor,
        _FIXED_HEADER_BYTES - _VERSION_BYTES,
        _VERSION_BYTES,
    )
    fixed = struct.unpack(f"{byte_order}9id6d9i16si", fixed_raw)
    (
        offset_tree,
        offset_blocks,
        field_count,
        direction_count,
        dimension_count,
        declared_max_level,
        leaf_count,
        parent_count,
        iteration,
    ) = (int(value) for value in fixed[:9])
    time = float(fixed[9])
    domain_lower = np.asarray(fixed[10:13], dtype=np.float64)
    domain_upper = np.asarray(fixed[13:16], dtype=np.float64)
    domain_cell_counts = np.asarray(fixed[16:19], dtype=np.int64)
    block_cell_counts = np.asarray(fixed[19:22], dtype=np.int64)
    periodic = np.asarray(
        tuple(int(value) != 0 for value in fixed[22:25]),
        dtype=np.bool_,
    )
    geometry_raw = fixed[25]
    staggered = bool(fixed[26])

    _require_positive_count("field_count", field_count)
    _require_positive_count("direction_count", direction_count)
    if dimension_count != 3:
        raise ValueError("only spatially 3D AMRVAC v5 files are supported")
    _require_positive_count("declared_max_level", declared_max_level)
    _require_positive_count("leaf_count", leaf_count)
    if parent_count < 0:
        raise ValueError("parent_count must be nonnegative")
    if offset_tree <= 0 or offset_blocks <= 0:
        raise ValueError("metadata offsets must be positive")
    _checked_nonnegative_i64(offset_tree, "tree offset")
    _checked_nonnegative_i64(offset_blocks, "block offset")
    if offset_tree > file_size or offset_blocks > file_size:
        raise ValueError("metadata offsets lie outside the file")
    if not np.all(np.isfinite(domain_lower)) or not np.all(
        np.isfinite(domain_upper)
    ):
        raise ValueError("domain bounds must be finite")
    if np.any(domain_upper <= domain_lower):
        raise ValueError("domain bounds must be strictly increasing")
    root_shape = _validated_root_shape(
        domain_cell_counts,
        block_cell_counts,
    )
    node_count = _checked_add(leaf_count, parent_count, "forest node count")

    cursor = _FIXED_HEADER_BYTES
    field_name_bytes = _checked_mul(
        field_count,
        _NAME_BYTES,
        "field-name byte count",
    )
    before_parameter_values = _checked_add(
        cursor,
        field_name_bytes,
        "header cursor",
    )
    before_parameter_values = _checked_add(
        before_parameter_values,
        _NAME_BYTES + 4,
        "header cursor",
    )
    if before_parameter_values > offset_tree:
        raise ValueError("field metadata extends beyond offset_tree")

    field_names_raw = _pread_exact(
        file_descriptor,
        field_name_bytes,
        cursor,
    )
    field_names = tuple(
        _decode_fixed_string(
            field_names_raw[start : start + _NAME_BYTES],
            f"field name {field}",
        )
        for field, start in enumerate(range(0, field_name_bytes, _NAME_BYTES))
    )
    cursor += field_name_bytes
    physics_type = _decode_fixed_string(
        _pread_exact(file_descriptor, _NAME_BYTES, cursor),
        "physics type",
    )
    cursor += _NAME_BYTES
    parameter_count = int(
        struct.unpack(
            f"{byte_order}i",
            _pread_exact(file_descriptor, 4, cursor),
        )[0]
    )
    cursor += 4
    if parameter_count < 0:
        raise ValueError("parameter count must be nonnegative")

    parameter_value_bytes = _checked_mul(
        parameter_count,
        8,
        "parameter-value byte count",
    )
    parameter_name_bytes = _checked_mul(
        parameter_count,
        _NAME_BYTES,
        "parameter-name byte count",
    )
    expected_header_end = _checked_add(
        cursor,
        parameter_value_bytes,
        "header cursor",
    )
    expected_header_end = _checked_add(
        expected_header_end,
        parameter_name_bytes,
        "header cursor",
    )
    expected_header_end = _checked_add(
        expected_header_end,
        12,
        "header cursor",
    )
    if expected_header_end != offset_tree:
        raise ValueError(
            f"decoded header ends at {expected_header_end}, "
            f"not offset_tree {offset_tree}"
        )

    parameter_values = _native_float64(
        _pread_exact(file_descriptor, parameter_value_bytes, cursor),
        byte_order,
    )
    cursor += parameter_value_bytes
    parameter_names_raw = _pread_exact(
        file_descriptor,
        parameter_name_bytes,
        cursor,
    )
    parameter_names = tuple(
        _decode_fixed_string(
            parameter_names_raw[start : start + _NAME_BYTES],
            f"parameter name {parameter}",
        )
        for parameter, start in enumerate(
            range(0, parameter_name_bytes, _NAME_BYTES)
        )
    )
    cursor += parameter_name_bytes
    snapshot_next, slice_next, collapse_next = (
        int(value)
        for value in struct.unpack(
            f"{byte_order}3i",
            _pread_exact(file_descriptor, 12, cursor),
        )
    )
    cursor += 12
    if cursor != offset_tree:
        raise RuntimeError("internal AMRVAC header cursor mismatch")

    forest_bytes = _checked_mul(node_count, 4, "forest-flag byte count")
    level_bytes = _checked_mul(leaf_count, 4, "block-level byte count")
    coordinate_values = _checked_mul(
        leaf_count,
        3,
        "block-coordinate value count",
    )
    coordinate_bytes = _checked_mul(
        coordinate_values,
        4,
        "block-coordinate byte count",
    )
    offset_bytes = _checked_mul(leaf_count, 8, "block-offset byte count")
    tree_bytes = _checked_add(forest_bytes, level_bytes, "tree byte count")
    tree_bytes = _checked_add(tree_bytes, coordinate_bytes, "tree byte count")
    tree_bytes = _checked_add(tree_bytes, offset_bytes, "tree byte count")
    expected_tree_end = _checked_add(offset_tree, tree_bytes, "tree cursor")
    if expected_tree_end != offset_blocks:
        raise ValueError(
            f"decoded tree ends at {expected_tree_end}, "
            f"not offset_blocks {offset_blocks}"
        )

    tree_raw = _pread_exact(file_descriptor, tree_bytes, offset_tree)
    tree_cursor = 0
    disk_int32 = np.dtype(f"{byte_order}i4")
    disk_int64 = np.dtype(f"{byte_order}i8")
    forest_flags = np.frombuffer(
        tree_raw,
        dtype=disk_int32,
        count=node_count,
        offset=tree_cursor,
    ).astype(np.bool_, copy=True)
    tree_cursor += forest_bytes
    block_levels = np.frombuffer(
        tree_raw,
        dtype=disk_int32,
        count=leaf_count,
        offset=tree_cursor,
    ).astype(np.int64, copy=True)
    tree_cursor += level_bytes
    disk_coordinates = np.frombuffer(
        tree_raw,
        dtype=disk_int32,
        count=coordinate_values,
        offset=tree_cursor,
    ).reshape(leaf_count, 3)
    tree_cursor += coordinate_bytes
    block_offsets = np.frombuffer(
        tree_raw,
        dtype=disk_int64,
        count=leaf_count,
        offset=tree_cursor,
    ).astype(np.int64, copy=True)
    tree_cursor += offset_bytes
    if tree_cursor != tree_bytes:
        raise RuntimeError("internal AMRVAC tree cursor mismatch")

    true_count = int(np.count_nonzero(forest_flags))
    if true_count != leaf_count or node_count - true_count != parent_count:
        raise ValueError("forest leaf/parent flags do not match declared counts")
    invalid_level = np.flatnonzero(
        (block_levels < 1) | (block_levels > declared_max_level)
    )
    if invalid_level.size:
        leaf = int(invalid_level[0])
        raise ValueError(
            f"block level at leaf {leaf} lies outside the declared range"
        )
    block_coordinates = _validate_leaf_coordinates(
        disk_coordinates,
        block_levels,
        root_shape,
    )

    block_cell_volume = _checked_product(
        block_cell_counts,
        "block-cell volume",
    )
    value_count = _checked_mul(
        field_count,
        block_cell_volume,
        "minimum block value count",
    )
    value_bytes = _checked_mul(
        value_count,
        8,
        "minimum block payload bytes",
    )
    minimum_record_bytes = _checked_add(
        _GHOST_HEADER_BYTES,
        value_bytes,
        "minimum block record bytes",
    )
    for leaf in range(leaf_count):
        offset = int(block_offsets[leaf])
        if offset <= 0:
            raise ValueError(f"block offset at leaf {leaf} must be positive")
        if leaf == 0:
            if offset != offset_blocks:
                raise ValueError("the first block offset must equal offset_blocks")
        else:
            prior = int(block_offsets[leaf - 1])
            if offset <= prior:
                raise ValueError("block offsets must be strictly increasing")
            if offset - prior < minimum_record_bytes:
                raise ValueError(
                    f"block offset interval before leaf {leaf} is too short"
                )
    if file_size - int(block_offsets[-1]) < minimum_record_bytes:
        raise ValueError("the final block record is truncated")

    identity_after = _file_identity(file_descriptor)
    if identity_after != identity_before:
        raise ValueError("file identity changed while AMRVAC metadata was parsed")

    return AMRVACV5Index(
        byte_order=byte_order,
        file_identity=identity_before,
        offset_tree=offset_tree,
        offset_blocks=offset_blocks,
        field_count=field_count,
        direction_count=direction_count,
        dimension_count=dimension_count,
        declared_max_level=declared_max_level,
        leaf_count=leaf_count,
        parent_count=parent_count,
        iteration=iteration,
        time=time,
        domain_lower=domain_lower,
        domain_upper=domain_upper,
        domain_cell_counts=domain_cell_counts,
        block_cell_counts=block_cell_counts,
        periodic=periodic,
        geometry=_decode_fixed_string(geometry_raw, "geometry"),
        staggered=staggered,
        field_names=field_names,
        physics_type=physics_type,
        parameter_values=parameter_values,
        parameter_names=parameter_names,
        snapshot_next=snapshot_next,
        slice_next=slice_next,
        collapse_next=collapse_next,
        forest_flags=forest_flags,
        block_levels=block_levels,
        block_coordinates=block_coordinates,
        block_offsets=block_offsets,
    )


def _require_index_array(
    name: str,
    value: object,
    dtype: np.dtype,
    shape: tuple[int, ...],
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype.name}")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def _require_exact_index_int(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact Python int")
    return value


def bind_amrvac_v5_forest(index: AMRVACV5Index) -> AMRVACV5ForestBinding:
    """Reconstruct FST-001 and prove exact AMRVAC leaf-row identity."""
    if type(index) is not AMRVACV5Index:
        raise TypeError("index must be exactly an AMRVACV5Index")

    leaf_count = _require_exact_index_int("leaf_count", index.leaf_count)
    parent_count = _require_exact_index_int("parent_count", index.parent_count)
    declared_max_level = _require_exact_index_int(
        "declared_max_level",
        index.declared_max_level,
    )
    if leaf_count <= 0:
        raise ValueError("leaf_count must be positive")
    if parent_count < 0:
        raise ValueError("parent_count must be nonnegative")
    if declared_max_level <= 0:
        raise ValueError("declared_max_level must be positive")
    node_count = _checked_add(leaf_count, parent_count, "forest node count")

    domain_cell_counts = _require_index_array(
        "domain_cell_counts",
        index.domain_cell_counts,
        np.dtype(np.int64),
        (3,),
    )
    block_cell_counts = _require_index_array(
        "block_cell_counts",
        index.block_cell_counts,
        np.dtype(np.int64),
        (3,),
    )
    forest_flags = _require_index_array(
        "forest_flags",
        index.forest_flags,
        np.dtype(np.bool_),
        (node_count,),
    )
    block_levels = _require_index_array(
        "block_levels",
        index.block_levels,
        np.dtype(np.int64),
        (leaf_count,),
    )
    block_coordinates = _require_index_array(
        "block_coordinates",
        index.block_coordinates,
        np.dtype(np.int64),
        (leaf_count, 3),
    )
    _require_index_array(
        "block_offsets",
        index.block_offsets,
        np.dtype(np.int64),
        (leaf_count,),
    )

    identity = index.file_identity
    if type(identity) is not tuple or len(identity) != 5:
        raise TypeError("file_identity must be a five-int tuple")
    if any(type(value) is not int for value in identity):
        raise TypeError("file_identity must be a five-int tuple")
    source_file_identity = tuple(value for value in identity)

    root_shape = _validated_root_shape(
        domain_cell_counts,
        block_cell_counts,
    )
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    forest = refined_forest(
        root_shape,
        coord_to_rank,
        rank_to_coord,
        forest_flags,
    )
    conformed_max_level = validate_refined_forest_arrays(
        root_shape,
        coord_to_rank,
        rank_to_coord,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.parent_node_ids,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
    )
    if conformed_max_level != forest.max_level:
        raise RuntimeError("constructed FST maximum level is inconsistent")
    if conformed_max_level > declared_max_level:
        raise ValueError(
            "reconstructed forest maximum exceeds declared_max_level"
        )
    if forest.leaf_node_ids.shape[0] != leaf_count:
        raise ValueError("forest stream does not match declared leaf_count")

    leaf_nodes = forest.leaf_node_ids
    expected_levels = forest.node_levels[leaf_nodes]
    unequal_level = np.flatnonzero(block_levels != expected_levels)
    if unequal_level.size:
        leaf = int(unequal_level[0])
        raise ValueError(
            f"on-disk level does not match canonical forest leaf {leaf}"
        )
    expected_coordinates = forest.node_coords[leaf_nodes]
    unequal_coordinate = np.argwhere(block_coordinates != expected_coordinates)
    if unequal_coordinate.size:
        leaf, axis = (int(value) for value in unequal_coordinate[0])
        raise ValueError(
            "on-disk coordinate does not match canonical forest "
            f"leaf {leaf}, axis {axis}"
        )

    return AMRVACV5ForestBinding(
        source_file_identity=source_file_identity,
        root_shape=root_shape,
        coord_to_rank=coord_to_rank,
        rank_to_coord=rank_to_coord,
        forest=forest,
    )
