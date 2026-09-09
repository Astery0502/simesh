from __future__ import annotations

import os
from pathlib import Path
import struct

import numpy as np
import pytest

import simesh_rewrite.amrvac_dat as amrvac_dat
from simesh.amrvac.datio import get_metadata
from simesh_rewrite.amrvac_dat import AMRVACV5Index, read_amrvac_v5_index


ROOT = Path(__file__).resolve().parents[2]
PARAMETER_BITS = (
    0x0000000000000000,
    0x8000000000000000,
    0x7FF0000000000000,
    0xFFF0000000000000,
    0x7FF0000000000042,
    0xFFF8000000001234,
)


def _fixed(value: bytes) -> bytes:
    if len(value) > 16:
        raise ValueError("test string is too long")
    return value + b" " * (16 - len(value))


def _fixture_bytes(byte_order: str) -> tuple[bytes, dict[str, object]]:
    flags = (False,) + (True,) * 8 + (True,)
    block_levels = (2,) * 8 + (1,)
    block_coordinates = (
        (1, 1, 1),
        (2, 1, 1),
        (1, 2, 1),
        (2, 2, 1),
        (1, 1, 2),
        (2, 1, 2),
        (1, 2, 2),
        (2, 2, 2),
        (2, 1, 1),
    )
    field_raw = (b"", b" rho")
    parameter_raw = tuple(f"p{i}".encode() for i in range(len(PARAMETER_BITS)))
    field_count = len(field_raw)
    leaf_count = len(block_levels)
    parent_count = len(flags) - leaf_count
    offset_tree = (
        152
        + 16 * field_count
        + 16
        + 4
        + 8 * len(PARAMETER_BITS)
        + 16 * len(PARAMETER_BITS)
        + 12
    )
    tree_bytes = 4 * len(flags) + 4 * leaf_count + 12 * leaf_count + 8 * leaf_count
    offset_blocks = offset_tree + tree_bytes
    block_shape = (2, 2, 2)
    record_bytes = 24 + field_count * int(np.prod(block_shape)) * 8
    block_offsets = tuple(
        offset_blocks + leaf * record_bytes for leaf in range(leaf_count)
    )

    header = bytearray()
    header += struct.pack(f"{byte_order}i", 5)
    header += struct.pack(
        f"{byte_order}9id",
        offset_tree,
        offset_blocks,
        field_count,
        3,
        3,
        4,
        leaf_count,
        parent_count,
        17,
        -0.0,
    )
    header += struct.pack(f"{byte_order}3d", -2.0, -1.0, -0.5)
    header += struct.pack(f"{byte_order}3d", 2.0, 1.0, 0.5)
    header += struct.pack(f"{byte_order}3i", 4, 2, 2)
    header += struct.pack(f"{byte_order}3i", *block_shape)
    header += struct.pack(f"{byte_order}3i", 0, -7, 2)
    header += _fixed(b"Cartesian_3D")
    header += struct.pack(f"{byte_order}i", -9)
    header += b"".join(_fixed(value) for value in field_raw)
    header += _fixed(b"mhd\x00inner")
    header += struct.pack(f"{byte_order}i", len(PARAMETER_BITS))
    header += b"".join(
        struct.pack(f"{byte_order}Q", value) for value in PARAMETER_BITS
    )
    header += b"".join(_fixed(value) for value in parameter_raw)
    header += struct.pack(f"{byte_order}3i", 23, -4, 9)
    assert len(header) == offset_tree

    tree = bytearray()
    tree += struct.pack(
        f"{byte_order}{len(flags)}i", *(1 if value else 0 for value in flags)
    )
    tree += struct.pack(f"{byte_order}{leaf_count}i", *block_levels)
    tree += struct.pack(
        f"{byte_order}{3 * leaf_count}i",
        *(value for row in block_coordinates for value in row),
    )
    tree += struct.pack(f"{byte_order}{leaf_count}q", *block_offsets)
    assert len(header) + len(tree) == offset_blocks

    records = b"\x00" * (leaf_count * record_bytes)
    expected = {
        "flags": np.asarray(flags, dtype=np.bool_),
        "levels": np.asarray(block_levels, dtype=np.int64),
        "coordinates": np.asarray(block_coordinates, dtype=np.int64) - 1,
        "offsets": np.asarray(block_offsets, dtype=np.int64),
        "offset_tree": offset_tree,
        "offset_blocks": offset_blocks,
        "record_bytes": record_bytes,
        "field_count": field_count,
    }
    return bytes(header + tree + records), expected


def _open_fixture(tmp_path: Path, byte_order: str) -> tuple[int, Path, dict[str, object]]:
    raw, expected = _fixture_bytes(byte_order)
    path = tmp_path / ("little.dat" if byte_order == "<" else "big.dat")
    path.write_bytes(raw)
    return os.open(path, os.O_RDONLY), path, expected


def _patch_i32(raw: bytes, byte_order: str, offset: int, value: int) -> bytes:
    changed = bytearray(raw)
    struct.pack_into(f"{byte_order}i", changed, offset, value)
    return bytes(changed)


def _patch_i64(raw: bytes, byte_order: str, offset: int, value: int) -> bytes:
    changed = bytearray(raw)
    struct.pack_into(f"{byte_order}q", changed, offset, value)
    return bytes(changed)


def _read_raw(tmp_path: Path, raw: bytes) -> AMRVACV5Index:
    path = tmp_path / "case.dat"
    path.write_bytes(raw)
    descriptor = os.open(path, os.O_RDONLY)
    try:
        return read_amrvac_v5_index(descriptor)
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("byte_order", ["<", ">"])
def test_byte_crafted_v5_metadata_is_exact_owned_and_cursor_preserving(
    tmp_path: Path,
    byte_order: str,
) -> None:
    descriptor, path, expected = _open_fixture(tmp_path, byte_order)
    try:
        os.lseek(descriptor, 11, os.SEEK_SET)
        index = read_amrvac_v5_index(descriptor)
        assert os.lseek(descriptor, 0, os.SEEK_CUR) == 11
        assert os.fstat(descriptor).st_size == path.stat().st_size
    finally:
        os.close(descriptor)

    assert index.byte_order == byte_order
    assert index.file_identity == (
        path.stat().st_dev,
        path.stat().st_ino,
        path.stat().st_size,
        path.stat().st_mtime_ns,
        path.stat().st_ctime_ns,
    )
    assert index.offset_tree == expected["offset_tree"]
    assert index.offset_blocks == expected["offset_blocks"]
    assert index.field_count == 2
    assert index.direction_count == 3
    assert index.dimension_count == 3
    assert index.declared_max_level == 4
    assert index.leaf_count == 9
    assert index.parent_count == 1
    assert index.iteration == 17
    assert np.asarray(index.time, dtype=np.float64).view(np.uint64) == (
        0x8000000000000000
    )
    assert np.array_equal(index.domain_lower, [-2.0, -1.0, -0.5])
    assert np.array_equal(index.domain_upper, [2.0, 1.0, 0.5])
    assert np.array_equal(index.domain_cell_counts, [4, 2, 2])
    assert np.array_equal(index.block_cell_counts, [2, 2, 2])
    assert np.array_equal(index.periodic, [False, True, True])
    assert index.geometry == "Cartesian_3D"
    assert index.staggered is True
    assert index.field_names == ("", " rho")
    assert index.physics_type == "mhd\x00inner"
    assert tuple(int(value) for value in index.parameter_values.view(np.uint64)) == (
        PARAMETER_BITS
    )
    assert index.parameter_names == tuple(f"p{i}" for i in range(6))
    assert (index.snapshot_next, index.slice_next, index.collapse_next) == (
        23,
        -4,
        9,
    )
    assert np.array_equal(index.forest_flags, expected["flags"])
    assert np.array_equal(index.block_levels, expected["levels"])
    assert np.array_equal(index.block_coordinates, expected["coordinates"])
    assert np.array_equal(index.block_offsets, expected["offsets"])

    arrays = (
        index.domain_lower,
        index.domain_upper,
        index.domain_cell_counts,
        index.block_cell_counts,
        index.periodic,
        index.parameter_values,
        index.forest_flags,
        index.block_levels,
        index.block_coordinates,
        index.block_offsets,
    )
    for value in arrays:
        assert value.flags.owndata
        assert value.flags.c_contiguous
        assert value.dtype.isnative
    for position, value in enumerate(arrays):
        assert not any(np.shares_memory(value, prior) for prior in arrays[:position])


@pytest.mark.parametrize("byte_order", ["<", ">"])
@pytest.mark.parametrize(
    "time_bits",
    [0x7FF0000000000042, 0xFFF8000000001234],
    ids=("signaling-nan", "negative-quiet-nan"),
)
def test_header_time_preserves_nan_sign_and_payload_bits(
    tmp_path: Path,
    byte_order: str,
    time_bits: int,
) -> None:
    raw, _ = _fixture_bytes(byte_order)
    changed = bytearray(raw)
    struct.pack_into(f"{byte_order}Q", changed, 40, time_bits)
    index = _read_raw(tmp_path, bytes(changed))
    assert int(np.asarray(index.time).view(np.uint64)) == time_bits


def test_metadata_read_stops_before_first_block_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor, _, expected = _open_fixture(tmp_path, "<")
    original = os.pread
    calls: list[tuple[int, int]] = []

    def recording_pread(fd: int, count: int, offset: int) -> bytes:
        calls.append((offset, count))
        return original(fd, count, offset)

    monkeypatch.setattr(amrvac_dat.os, "pread", recording_pread)
    try:
        read_amrvac_v5_index(descriptor)
    finally:
        os.close(descriptor)
    assert calls
    assert max(offset + count for offset, count in calls) == expected["offset_blocks"]


@pytest.mark.parametrize("value", [True, np.int64(3), 3.0, "3", None])
def test_file_descriptor_requires_an_exact_python_int(value: object) -> None:
    with pytest.raises(TypeError, match="exact Python int"):
        read_amrvac_v5_index(value)  # type: ignore[arg-type]


def test_descriptor_lifecycle_and_regular_file_requirements(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        read_amrvac_v5_index(-1)

    path = tmp_path / "closed.dat"
    path.write_bytes(b"\x05\x00\x00\x00")
    descriptor = os.open(path, os.O_RDONLY)
    os.close(descriptor)
    with pytest.raises(OSError):
        read_amrvac_v5_index(descriptor)

    read_descriptor, write_descriptor = os.pipe()
    try:
        with pytest.raises(ValueError, match="regular file"):
            read_amrvac_v5_index(read_descriptor)
    finally:
        os.close(read_descriptor)
        os.close(write_descriptor)


@pytest.mark.parametrize(
    "version_raw",
    [
        struct.pack("<i", 4),
        struct.pack(">i", 6),
        b"\x00\x00\x00\x00",
    ],
)
def test_wrong_or_unsupported_version_is_rejected(
    tmp_path: Path,
    version_raw: bytes,
) -> None:
    path = tmp_path / "version.dat"
    path.write_bytes(version_raw + b"\x00" * 200)
    descriptor = os.open(path, os.O_RDONLY)
    try:
        with pytest.raises(ValueError, match="version"):
            read_amrvac_v5_index(descriptor)
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(
    ("offset", "value", "message"),
    [
        (12, 0, "field_count"),
        (16, 0, "direction_count"),
        (20, 2, "spatially 3D"),
        (24, 0, "declared_max_level"),
        (28, 0, "leaf_count"),
        (32, -1, "parent_count"),
    ],
)
def test_invalid_fixed_counts_are_rejected_before_allocation(
    tmp_path: Path,
    offset: int,
    value: int,
    message: str,
) -> None:
    raw, _ = _fixture_bytes("<")
    with pytest.raises(ValueError, match=message):
        _read_raw(tmp_path, _patch_i32(raw, "<", offset, value))


def test_invalid_bounds_counts_and_divisibility_are_rejected(tmp_path: Path) -> None:
    raw, _ = _fixture_bytes("<")
    changed = bytearray(raw)
    struct.pack_into("<d", changed, 48, float("nan"))
    with pytest.raises(ValueError, match="finite"):
        _read_raw(tmp_path, bytes(changed))

    changed = bytearray(raw)
    struct.pack_into("<d", changed, 72, -2.0)
    with pytest.raises(ValueError, match="increasing"):
        _read_raw(tmp_path, bytes(changed))

    with pytest.raises(ValueError, match="domain_cell_counts"):
        _read_raw(tmp_path, _patch_i32(raw, "<", 96, 0))
    with pytest.raises(ValueError, match="block_cell_counts"):
        _read_raw(tmp_path, _patch_i32(raw, "<", 108, -1))
    with pytest.raises(ValueError, match="divisible"):
        _read_raw(tmp_path, _patch_i32(raw, "<", 96, 5))


def test_checked_root_block_and_refinement_arithmetic(tmp_path: Path) -> None:
    raw, expected = _fixture_bytes("<")
    changed = bytearray(raw)
    for axis in range(3):
        struct.pack_into("<i", changed, 96 + 4 * axis, 2_147_483_647)
        struct.pack_into("<i", changed, 108 + 4 * axis, 1)
    with pytest.raises(OverflowError, match="root-grid volume"):
        _read_raw(tmp_path, bytes(changed))

    changed = bytearray(raw)
    for axis in range(3):
        struct.pack_into("<i", changed, 96 + 4 * axis, 2_147_483_647)
        struct.pack_into("<i", changed, 108 + 4 * axis, 2_147_483_647)
    coordinate_start = int(expected["offset_tree"]) + 4 * 10 + 4 * 9
    struct.pack_into("<i", changed, coordinate_start + 12 * 8, 1)
    with pytest.raises(OverflowError, match="block-cell volume"):
        _read_raw(tmp_path, bytes(changed))

    changed = bytearray(raw)
    struct.pack_into("<i", changed, 24, 64)
    level_start = int(expected["offset_tree"]) + 4 * 10
    for leaf in range(9):
        struct.pack_into("<i", changed, level_start + 4 * leaf, 64)
    with pytest.raises(OverflowError, match="logical grid"):
        _read_raw(tmp_path, bytes(changed))


def test_header_cursor_utf8_and_short_sections_are_rejected(tmp_path: Path) -> None:
    raw, expected = _fixture_bytes("<")
    with pytest.raises(ValueError, match="decoded header ends"):
        _read_raw(tmp_path, _patch_i32(raw, "<", 4, int(expected["offset_tree"]) + 1))
    with pytest.raises(ValueError, match="decoded tree ends"):
        _read_raw(
            tmp_path,
            _patch_i32(raw, "<", 8, int(expected["offset_blocks"]) + 4),
        )

    changed = bytearray(raw)
    changed[152] = 0xFF
    with pytest.raises(ValueError, match="UTF-8"):
        _read_raw(tmp_path, bytes(changed))

    with pytest.raises(ValueError, match="outside the file|truncated"):
        _read_raw(tmp_path, raw[: int(expected["offset_tree"]) - 1])
    with pytest.raises(ValueError, match="outside the file|truncated"):
        _read_raw(tmp_path, raw[: int(expected["offset_blocks"]) - 1])


def test_tree_counts_levels_coordinates_and_offsets_are_checked(tmp_path: Path) -> None:
    raw, expected = _fixture_bytes("<")
    tree = int(expected["offset_tree"])
    level_start = tree + 4 * 10
    coordinate_start = level_start + 4 * 9
    offset_start = coordinate_start + 12 * 9

    with pytest.raises(ValueError, match="leaf/parent flags"):
        _read_raw(tmp_path, _patch_i32(raw, "<", tree, 1))
    with pytest.raises(ValueError, match="declared range"):
        _read_raw(tmp_path, _patch_i32(raw, "<", level_start, 0))
    with pytest.raises(ValueError, match="must be positive"):
        _read_raw(tmp_path, _patch_i32(raw, "<", coordinate_start, 0))
    with pytest.raises(ValueError, match="outside"):
        _read_raw(tmp_path, _patch_i32(raw, "<", coordinate_start, 5))
    with pytest.raises(ValueError, match="first block offset"):
        _read_raw(
            tmp_path,
            _patch_i64(
                raw,
                "<",
                offset_start,
                int(expected["offset_blocks"]) + 1,
            ),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        _read_raw(
            tmp_path,
            _patch_i64(
                raw,
                "<",
                offset_start + 8,
                int(expected["offset_blocks"]),
            ),
        )
    with pytest.raises(ValueError, match="interval"):
        _read_raw(
            tmp_path,
            _patch_i64(
                raw,
                "<",
                offset_start + 8,
                int(expected["offset_blocks"]) + int(expected["record_bytes"]) - 1,
            ),
        )
    with pytest.raises(ValueError, match="final block record"):
        _read_raw(tmp_path, raw[:-1])


def test_identity_change_during_parse_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor, _, _ = _open_fixture(tmp_path, "<")
    original = amrvac_dat._file_identity
    calls = 0

    def changing_identity(fd: int) -> tuple[int, int, int, int, int]:
        nonlocal calls
        calls += 1
        identity = original(fd)
        if calls == 2:
            return identity[:3] + (identity[3] + 1, identity[4])
        return identity

    monkeypatch.setattr(amrvac_dat, "_file_identity", changing_identity)
    try:
        with pytest.raises(ValueError, match="identity changed"):
            read_amrvac_v5_index(descriptor)
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(
    "relative_path",
    ["data/tdm.dat", "data/weno509_sub_0000.dat", "reference/bw.dat"],
)
def test_available_real_v5_metadata_matches_current_raw_reader(
    relative_path: str,
) -> None:
    path = ROOT / relative_path
    if not path.exists():
        pytest.skip(f"optional real fixture is unavailable: {relative_path}")
    header, flags, tree = get_metadata(str(path))
    descriptor = os.open(path, os.O_RDONLY)
    try:
        index = read_amrvac_v5_index(descriptor)
    finally:
        os.close(descriptor)

    assert index.byte_order == "<"
    assert index.offset_tree == header["offset_tree"]
    assert index.offset_blocks == header["offset_blocks"]
    assert index.field_count == header["nw"]
    assert index.direction_count == header["ndir"]
    assert index.dimension_count == header["ndim"]
    assert index.declared_max_level == header["levmax"]
    assert index.leaf_count == header["nleafs"]
    assert index.parent_count == header["nparents"]
    assert index.iteration == header["it"]
    assert np.asarray(index.time).view(np.uint64) == np.asarray(
        header["time"]
    ).view(np.uint64)
    assert np.array_equal(index.domain_lower, header["xmin"])
    assert np.array_equal(index.domain_upper, header["xmax"])
    assert np.array_equal(index.domain_cell_counts, header["domain_nx"])
    assert np.array_equal(index.block_cell_counts, header["block_nx"])
    assert np.array_equal(index.periodic, header["periodic"])
    assert index.geometry == header["geometry"]
    assert index.staggered == header["staggered"]
    assert index.field_names == tuple(header["w_names"])
    assert index.physics_type == header["physics_type"]
    assert np.array_equal(
        index.parameter_values.view(np.uint64),
        np.asarray(header["params"], dtype=np.float64).view(np.uint64),
    )
    assert index.parameter_names == tuple(header["param_names"])
    assert index.snapshot_next == header["snapshotnext"]
    assert index.slice_next == header["slicenext"]
    assert index.collapse_next == header["collapsenext"]
    assert np.array_equal(index.forest_flags, flags)
    assert np.array_equal(index.block_levels, tree[0])
    assert np.array_equal(index.block_coordinates, tree[1] - 1)
    assert np.array_equal(index.block_offsets, tree[2])
