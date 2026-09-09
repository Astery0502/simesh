from __future__ import annotations

import os
import struct
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

import simesh_rewrite.amrvac_dat_reader as dat_reader_module
from simesh_rewrite.amrvac_dat import AMRVACV5ForestBinding, AMRVACV5Index
from simesh_rewrite.amrvac_dat_reader import make_amrvac_v5_block_reader
from simesh_rewrite.blockio import read_blocks_into


_SPECIAL_BITS = np.asarray(
    [
        0x0000000000000000,
        0x8000000000000000,
        0x3FF0000000000000,
        0xBFF0000000000000,
        0x7FF0000000000000,
        0xFFF0000000000000,
        0x7FF8000000000123,
        0xFFF8000000000456,
        0x7FF0000000000789,
        0xFFF0000000000ABC,
        0x0000000000000001,
        0x8000000000000001,
    ],
    dtype=np.uint64,
)
_SENTINEL_BITS = np.uint64(0xDEADBEEFCAFE1234)


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def _identity(file_descriptor: int) -> tuple[int, int, int, int, int]:
    result = os.fstat(file_descriptor)
    return (
        int(result.st_dev),
        int(result.st_ino),
        int(result.st_size),
        int(result.st_mtime_ns),
        int(result.st_ctime_ns),
    )


@dataclass(frozen=True)
class SyntheticSource:
    path: Path
    file_descriptor: int
    index: AMRVACV5Index
    binding: AMRVACV5ForestBinding
    stored_bits: tuple[np.ndarray, ...]
    ghosts: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]


def _stored_bits(
    block_id: int,
    field_count: int,
    stored_shape: tuple[int, int, int],
) -> np.ndarray:
    size = field_count * int(np.prod(stored_shape))
    indices = np.arange(size, dtype=np.int64)
    values = _SPECIAL_BITS[(indices + 5 * block_id) % _SPECIAL_BITS.size]
    return values.reshape((field_count,) + stored_shape).copy()


def _make_index(
    byte_order: str,
    file_identity: tuple[int, int, int, int, int],
    offsets: np.ndarray,
    field_count: int,
    block_shape: tuple[int, int, int],
) -> AMRVACV5Index:
    leaf_count = int(offsets.size)
    root_shape = i3(leaf_count, 1, 1)
    coordinates = np.column_stack(
        (
            np.arange(leaf_count, dtype=np.int64),
            np.zeros(leaf_count, dtype=np.int64),
            np.zeros(leaf_count, dtype=np.int64),
        )
    )
    return AMRVACV5Index(
        byte_order=byte_order,
        file_identity=file_identity,
        offset_tree=1,
        offset_blocks=int(offsets[0]),
        field_count=field_count,
        direction_count=3,
        dimension_count=3,
        declared_max_level=1,
        leaf_count=leaf_count,
        parent_count=0,
        iteration=7,
        time=0.0,
        domain_lower=np.zeros(3, dtype=np.float64),
        domain_upper=np.ones(3, dtype=np.float64),
        domain_cell_counts=root_shape * i3(*block_shape),
        block_cell_counts=i3(*block_shape),
        periodic=np.zeros(3, dtype=np.bool_),
        geometry="Cartesian_3D",
        staggered=False,
        field_names=tuple(f"f{field}" for field in range(field_count)),
        physics_type="test",
        parameter_values=np.empty(0, dtype=np.float64),
        parameter_names=(),
        snapshot_next=0,
        slice_next=0,
        collapse_next=0,
        forest_flags=np.ones(leaf_count, dtype=np.bool_),
        block_levels=np.ones(leaf_count, dtype=np.int64),
        block_coordinates=coordinates,
        block_offsets=offsets,
    )


@contextmanager
def synthetic_source(
    tmp_path: Path,
    *,
    byte_order: str = "<",
    block_shape: tuple[int, int, int] = (3, 2, 4),
    field_count: int = 4,
    ghosts: tuple[
        tuple[tuple[int, int, int], tuple[int, int, int]], ...
    ] = (
        ((1, 0, 2), (0, 2, 1)),
        ((0, 0, 0), (0, 0, 0)),
        ((2, 1, 0), (1, 0, 2)),
    ),
    padding_after: dict[int, bytes] | None = None,
    name: str = "synthetic.dat",
) -> Iterator[SyntheticSource]:
    padding_after = {} if padding_after is None else padding_after
    raw_file = bytearray(b"\xCC" * 64)
    offsets: list[int] = []
    records: list[np.ndarray] = []
    for block_id, (lower_ghost, upper_ghost) in enumerate(ghosts):
        stored_shape = tuple(
            block_shape[axis] + lower_ghost[axis] + upper_ghost[axis]
            for axis in range(3)
        )
        values = _stored_bits(block_id, field_count, stored_shape)
        offsets.append(len(raw_file))
        raw_file.extend(
            struct.pack(
                f"{byte_order}6i",
                *(lower_ghost + upper_ghost),
            )
        )
        disk_values = values.transpose(0, 3, 2, 1).astype(
            np.dtype(f"{byte_order}u8"),
            copy=False,
        )
        raw_file.extend(disk_values.tobytes(order="C"))
        raw_file.extend(padding_after.get(block_id, b""))
        records.append(values)

    path = tmp_path / name
    path.write_bytes(raw_file)
    file_descriptor = os.open(path, os.O_RDONLY)
    try:
        offsets_array = np.asarray(offsets, dtype=np.int64)
        file_identity = _identity(file_descriptor)
        index = _make_index(
            byte_order,
            file_identity,
            offsets_array,
            field_count,
            block_shape,
        )
        leaf_count = len(records)
        binding = AMRVACV5ForestBinding(
            source_file_identity=file_identity,
            root_shape=i3(leaf_count, 1, 1),
            coord_to_rank=np.arange(leaf_count, dtype=np.int64).reshape(
                leaf_count, 1, 1
            ),
            rank_to_coord=index.block_coordinates.copy(),
            forest=None,  # DAT-003 consumes only the proven source provenance.
        )
        yield SyntheticSource(
            path=path,
            file_descriptor=file_descriptor,
            index=index,
            binding=binding,
            stored_bits=tuple(records),
            ghosts=ghosts,
        )
    finally:
        os.close(file_descriptor)


def _new_destination(
    shape: tuple[int, int, int, int, int],
) -> np.ndarray:
    return np.full(shape, _SENTINEL_BITS, dtype=np.uint64).view(np.float64)


def _expected_transfer(
    source: SyntheticSource,
    block_ids: np.ndarray,
    field_ids: np.ndarray,
    source_lower: np.ndarray,
    source_upper: np.ndarray,
    destination: np.ndarray,
    destination_lower: np.ndarray,
) -> np.ndarray:
    expected = destination.view(np.uint64).copy()
    extent = source_upper - source_lower
    destination_slices = tuple(
        slice(
            int(destination_lower[axis]),
            int(destination_lower[axis] + extent[axis]),
        )
        for axis in range(3)
    )
    for slot, block_id_raw in enumerate(block_ids):
        block_id = int(block_id_raw)
        lower_ghost = source.ghosts[block_id][0]
        stored_slices = tuple(
            slice(
                int(source_lower[axis]) + lower_ghost[axis],
                int(source_upper[axis]) + lower_ghost[axis],
            )
            for axis in range(3)
        )
        for field_position, field_id_raw in enumerate(field_ids):
            field_id = int(field_id_raw)
            expected[
                slot,
                field_position,
                destination_slices[0],
                destination_slices[1],
                destination_slices[2],
            ] = source.stored_bits[block_id][
                field_id,
                stored_slices[0],
                stored_slices[1],
                stored_slices[2],
            ]
    return expected


def _envelope_range(
    source: SyntheticSource,
    block_id: int,
    field_id: int,
    source_lower: np.ndarray,
    source_upper: np.ndarray,
) -> tuple[int, int]:
    lower_ghost = source.ghosts[block_id][0]
    stored_shape = source.stored_bits[block_id].shape[1:]
    x0, y0, z0 = (
        int(source_lower[axis]) + lower_ghost[axis] for axis in range(3)
    )
    x1, y1, z1 = (
        int(source_upper[axis]) + lower_ghost[axis] for axis in range(3)
    )
    nx, ny, _ = stored_shape
    cell_count = int(np.prod(stored_shape))
    first = x0 + nx * (y0 + ny * z0)
    last = (x1 - 1) + nx * ((y1 - 1) + ny * (z1 - 1)) + 1
    offset = int(source.index.block_offsets[block_id]) + 24
    offset += 8 * (field_id * cell_count + first)
    return offset, 8 * (last - first)


@pytest.mark.parametrize("byte_order", ["<", ">"])
def test_partial_saved_ghost_reads_are_bit_exact_deduplicated_and_minimal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    byte_order: str,
) -> None:
    with synthetic_source(tmp_path, byte_order=byte_order) as source:
        reader = make_amrvac_v5_block_reader(
            source.file_descriptor,
            source.index,
            source.binding,
        )
        block_ids = np.asarray([2, 0, 2, 1], dtype=np.int64)
        field_ids = np.asarray([3, 1, 3, 0], dtype=np.int64)
        source_lower = i3(1, 0, 1)
        source_upper = i3(3, 2, 4)
        destination_lower = i3(1, 1, 2)
        destination = _new_destination((4, 4, 5, 4, 6))
        expected = _expected_transfer(
            source,
            block_ids,
            field_ids,
            source_lower,
            source_upper,
            destination,
            destination_lower,
        )

        pread_calls: list[tuple[int, int]] = []
        real_pread = os.pread

        def tracking_pread(fd: int, count: int, offset: int) -> bytes:
            pread_calls.append((offset, count))
            return real_pread(fd, count, offset)

        monkeypatch.setattr(dat_reader_module.os, "pread", tracking_pread)
        original_cursor = os.lseek(source.file_descriptor, 7, os.SEEK_SET)
        read_blocks_into(
            reader,
            source_lower,
            source_upper,
            block_ids,
            field_ids,
            destination,
            destination_lower,
        )

        assert np.array_equal(destination.view(np.uint64), expected)
        assert os.lseek(source.file_descriptor, 0, os.SEEK_CUR) == original_cursor
        expected_headers = [
            (int(source.index.block_offsets[block_id]), 24)
            for block_id in (0, 1, 2)
        ]
        expected_payloads = [
            _envelope_range(
                source,
                block_id,
                field_id,
                source_lower,
                source_upper,
            )
            for block_id in (0, 1, 2)
            for field_id in (0, 1, 3)
        ]
        assert pread_calls == expected_headers + expected_payloads
        for bits in _SPECIAL_BITS:
            assert np.any(expected == bits)


@pytest.mark.parametrize("byte_order", ["<", ">"])
def test_complete_zero_ghost_fields_use_exact_consecutive_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    byte_order: str,
) -> None:
    ghosts = (
        ((0, 0, 0), (0, 0, 0)),
        ((0, 0, 0), (0, 0, 0)),
    )
    block_shape = (3, 2, 2)
    with synthetic_source(
        tmp_path,
        byte_order=byte_order,
        block_shape=block_shape,
        field_count=5,
        ghosts=ghosts,
    ) as source:
        reader = make_amrvac_v5_block_reader(
            source.file_descriptor,
            source.index,
            source.binding,
        )
        assert reader.shape == (2, 5, 3, 2, 2)
        assert len(reader.memory_arrays) == 1
        retained_offsets = reader.memory_arrays[0]
        assert retained_offsets.dtype == np.int64
        assert retained_offsets.flags.c_contiguous
        assert retained_offsets.flags.owndata
        assert not retained_offsets.flags.writeable
        assert not np.shares_memory(retained_offsets, source.index.block_offsets)

        block_ids = np.asarray([1, 0, 1], dtype=np.int64)
        field_ids = np.asarray([4, 2, 1, 4], dtype=np.int64)
        lower = i3(0, 0, 0)
        upper = i3(*block_shape)
        destination_lower = i3(0, 0, 0)
        destination = _new_destination((3, 4) + block_shape)
        expected = _expected_transfer(
            source,
            block_ids,
            field_ids,
            lower,
            upper,
            destination,
            destination_lower,
        )

        pread_calls: list[tuple[int, int]] = []
        real_pread = os.pread

        def tracking_pread(fd: int, count: int, offset: int) -> bytes:
            pread_calls.append((offset, count))
            return real_pread(fd, count, offset)

        monkeypatch.setattr(dat_reader_module.os, "pread", tracking_pread)
        read_blocks_into(
            reader,
            lower,
            upper,
            block_ids,
            field_ids,
            destination,
            destination_lower,
        )
        assert np.array_equal(destination.view(np.uint64), expected)

        cell_count = int(np.prod(block_shape))
        expected_calls = [
            (int(source.index.block_offsets[block_id]), 24)
            for block_id in (0, 1)
        ]
        for block_id in (0, 1):
            payload_start = int(source.index.block_offsets[block_id]) + 24
            expected_calls.extend(
                [
                    (payload_start + cell_count * 8, 2 * cell_count * 8),
                    (payload_start + 4 * cell_count * 8, cell_count * 8),
                ]
            )
        assert pread_calls == expected_calls


def test_multiple_field_runs_release_each_payload_before_the_next_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ghosts = (((0, 0, 0), (0, 0, 0)),)
    block_shape = (16, 8, 4)
    with synthetic_source(
        tmp_path,
        block_shape=block_shape,
        field_count=3,
        ghosts=ghosts,
    ) as source:
        reader = make_amrvac_v5_block_reader(
            source.file_descriptor,
            source.index,
            source.binding,
        )
        real_pread = os.pread

        class TrackedPayload(bytes):
            live = 0
            peak = 0

            def __new__(cls, value: bytes) -> TrackedPayload:
                result = super().__new__(cls, value)
                cls.live += 1
                cls.peak = max(cls.peak, cls.live)
                return result

            def __del__(self) -> None:
                type(self).live -= 1

        def tracked_pread(fd: int, count: int, offset: int) -> bytes:
            result = real_pread(fd, count, offset)
            if count > 24:
                return TrackedPayload(result)
            return result

        monkeypatch.setattr(dat_reader_module.os, "pread", tracked_pread)
        destination = _new_destination((1, 2) + block_shape)
        read_blocks_into(
            reader,
            i3(0, 0, 0),
            i3(*block_shape),
            np.asarray([0], dtype=np.int64),
            np.asarray([2, 0], dtype=np.int64),
            destination,
            i3(0, 0, 0),
        )
        assert TrackedPayload.peak == 1
        assert TrackedPayload.live == 0


def test_request_failures_and_empty_requests_perform_no_file_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with synthetic_source(tmp_path) as source:
        reader = make_amrvac_v5_block_reader(
            source.file_descriptor,
            source.index,
            source.binding,
        )
        block_shape = tuple(int(value) for value in source.index.block_cell_counts)

        def forbidden(*args: object, **kwargs: object) -> object:
            raise AssertionError("empty or invalid request performed file I/O")

        monkeypatch.setattr(dat_reader_module.os, "fstat", forbidden)
        monkeypatch.setattr(dat_reader_module.os, "pread", forbidden)

        destination = _new_destination((1, 1) + block_shape)
        before = destination.view(np.uint64).copy()
        with pytest.raises(ValueError, match="block_ids entry 0"):
            read_blocks_into(
                reader,
                i3(0, 0, 0),
                i3(*block_shape),
                np.asarray([source.index.leaf_count], dtype=np.int64),
                np.asarray([0], dtype=np.int64),
                destination,
                i3(0, 0, 0),
            )
        assert np.array_equal(destination.view(np.uint64), before)

        empty_cases = [
            (
                np.empty(0, dtype=np.int64),
                np.asarray([0], dtype=np.int64),
                i3(0, 0, 0),
                i3(*block_shape),
                _new_destination((0, 1) + block_shape),
            ),
            (
                np.asarray([0], dtype=np.int64),
                np.empty(0, dtype=np.int64),
                i3(0, 0, 0),
                i3(*block_shape),
                _new_destination((1, 0) + block_shape),
            ),
            (
                np.asarray([0], dtype=np.int64),
                np.asarray([0], dtype=np.int64),
                i3(1, 0, 0),
                i3(1, 2, 4),
                _new_destination((1, 1) + block_shape),
            ),
        ]
        for block_ids, field_ids, lower, upper, output in empty_cases:
            output_before = output.view(np.uint64).copy()
            read_blocks_into(
                reader,
                lower,
                upper,
                block_ids,
                field_ids,
                output,
                i3(0, 0, 0),
            )
            assert np.array_equal(output.view(np.uint64), output_before)


def test_destination_alias_is_rejected_before_file_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ghosts = (((0, 0, 0), (0, 0, 0)),)
    with synthetic_source(tmp_path, ghosts=ghosts) as source:
        reader = make_amrvac_v5_block_reader(
            source.file_descriptor,
            source.index,
            source.binding,
        )
        destination = _new_destination((1, 1, 3, 2, 4))
        source_lower = destination.view(np.int64).reshape(-1)[:3]
        source_lower[:] = 0
        before = destination.view(np.uint64).copy()

        def forbidden(*args: object, **kwargs: object) -> object:
            raise AssertionError("alias failure performed file I/O")

        monkeypatch.setattr(dat_reader_module.os, "fstat", forbidden)
        monkeypatch.setattr(dat_reader_module.os, "pread", forbidden)
        with pytest.raises(ValueError, match="must not overlap"):
            read_blocks_into(
                reader,
                source_lower,
                i3(3, 2, 4),
                np.asarray([0], dtype=np.int64),
                np.asarray([0], dtype=np.int64),
                destination,
                i3(0, 0, 0),
            )
        assert np.array_equal(destination.view(np.uint64), before)


def test_later_malformed_selected_record_is_header_only_and_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with synthetic_source(tmp_path, padding_after={1: b"\x00"}) as source:
        reader = make_amrvac_v5_block_reader(
            source.file_descriptor,
            source.index,
            source.binding,
        )
        block_ids = np.asarray([1, 0], dtype=np.int64)
        field_ids = np.asarray([0], dtype=np.int64)
        destination = _new_destination((2, 1, 3, 2, 4))
        before = destination.view(np.uint64).copy()
        pread_calls: list[tuple[int, int]] = []
        real_pread = os.pread

        def tracking_pread(fd: int, count: int, offset: int) -> bytes:
            pread_calls.append((offset, count))
            return real_pread(fd, count, offset)

        monkeypatch.setattr(dat_reader_module.os, "pread", tracking_pread)
        with pytest.raises(ValueError, match="block 1 record end"):
            read_blocks_into(
                reader,
                i3(0, 0, 0),
                i3(3, 2, 4),
                block_ids,
                field_ids,
                destination,
                i3(0, 0, 0),
            )
        assert np.array_equal(destination.view(np.uint64), before)
        assert pread_calls == [
            (int(source.index.block_offsets[block_id]), 24)
            for block_id in (0, 1)
        ]

        # Record 1 remains malformed, but an independent selection of record 2
        # neither reads nor validates that unselected record.
        pread_calls.clear()
        selected_blocks = np.asarray([2], dtype=np.int64)
        selected_destination = _new_destination((1, 1, 3, 2, 4))
        selected_expected = _expected_transfer(
            source,
            selected_blocks,
            field_ids,
            i3(0, 0, 0),
            i3(3, 2, 4),
            selected_destination,
            i3(0, 0, 0),
        )
        read_blocks_into(
            reader,
            i3(0, 0, 0),
            i3(3, 2, 4),
            selected_blocks,
            field_ids,
            selected_destination,
            i3(0, 0, 0),
        )
        assert np.array_equal(
            selected_destination.view(np.uint64),
            selected_expected,
        )
        assert pread_calls == [
            (int(source.index.block_offsets[2]), 24),
            _envelope_range(
                source,
                2,
                0,
                i3(0, 0, 0),
                i3(3, 2, 4),
            ),
        ]


def test_short_payload_identifies_range_and_may_leave_prior_block_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ghosts = (
        ((0, 0, 0), (0, 0, 0)),
        ((0, 0, 0), (0, 0, 0)),
    )
    block_shape = (2, 2, 2)
    with synthetic_source(
        tmp_path,
        block_shape=block_shape,
        field_count=1,
        ghosts=ghosts,
    ) as source:
        reader = make_amrvac_v5_block_reader(
            source.file_descriptor,
            source.index,
            source.binding,
        )
        block_ids = np.asarray([1, 0], dtype=np.int64)
        field_ids = np.asarray([0], dtype=np.int64)
        destination = _new_destination((2, 1) + block_shape)
        real_pread = os.pread
        second_payload_offset = int(source.index.block_offsets[1]) + 24
        expected_bytes = int(np.prod(block_shape)) * 8

        def short_second_payload(fd: int, count: int, offset: int) -> bytes:
            result = real_pread(fd, count, offset)
            if offset == second_payload_offset and count == expected_bytes:
                return result[:-8]
            return result

        monkeypatch.setattr(
            dat_reader_module.os,
            "pread",
            short_second_payload,
        )
        with pytest.raises(
            OSError,
            match=rf"offset {second_payload_offset}.*expected {expected_bytes}.*got {expected_bytes - 8}",
        ):
            read_blocks_into(
                reader,
                i3(0, 0, 0),
                i3(*block_shape),
                block_ids,
                field_ids,
                destination,
                i3(0, 0, 0),
            )
        assert np.all(destination.view(np.uint64)[0] == _SENTINEL_BITS)
        assert np.array_equal(
            destination.view(np.uint64)[1, 0],
            source.stored_bits[0][0],
        )


def test_short_later_selected_header_precedes_all_payload_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ghosts = (
        ((0, 0, 0), (0, 0, 0)),
        ((0, 0, 0), (0, 0, 0)),
    )
    with synthetic_source(tmp_path, ghosts=ghosts) as source:
        reader = make_amrvac_v5_block_reader(
            source.file_descriptor,
            source.index,
            source.binding,
        )
        destination = _new_destination((2, 1, 3, 2, 4))
        before = destination.view(np.uint64).copy()
        later_header = int(source.index.block_offsets[1])
        real_pread = os.pread

        def short_later_header(fd: int, count: int, offset: int) -> bytes:
            result = real_pread(fd, count, offset)
            if offset == later_header and count == 24:
                return result[:-4]
            return result

        monkeypatch.setattr(dat_reader_module.os, "pread", short_later_header)
        with pytest.raises(
            OSError,
            match=rf"offset {later_header}.*expected 24.*got 20",
        ):
            read_blocks_into(
                reader,
                i3(0, 0, 0),
                i3(3, 2, 4),
                np.asarray([0, 1], dtype=np.int64),
                np.asarray([0], dtype=np.int64),
                destination,
                i3(0, 0, 0),
            )
        assert np.array_equal(destination.view(np.uint64), before)


def test_changed_file_identity_is_rejected_without_payload_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ghosts = (((0, 0, 0), (0, 0, 0)),)
    with synthetic_source(tmp_path, ghosts=ghosts) as source:
        reader = make_amrvac_v5_block_reader(
            source.file_descriptor,
            source.index,
            source.binding,
        )
        with source.path.open("ab") as stream:
            stream.write(b"\x00")
        destination = _new_destination((1, 1, 3, 2, 4))
        before = destination.view(np.uint64).copy()
        pread_calls = 0
        real_pread = os.pread

        def tracking_pread(fd: int, count: int, offset: int) -> bytes:
            nonlocal pread_calls
            pread_calls += 1
            return real_pread(fd, count, offset)

        monkeypatch.setattr(dat_reader_module.os, "pread", tracking_pread)
        with pytest.raises(OSError, match="identity changed"):
            read_blocks_into(
                reader,
                i3(0, 0, 0),
                i3(3, 2, 4),
                np.asarray([0], dtype=np.int64),
                np.asarray([0], dtype=np.int64),
                destination,
                i3(0, 0, 0),
            )
        assert pread_calls == 0
        assert np.array_equal(destination.view(np.uint64), before)


def test_final_identity_change_is_reported_after_payload_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ghosts = (((0, 0, 0), (0, 0, 0)),)
    with synthetic_source(tmp_path, ghosts=ghosts) as source:
        reader = make_amrvac_v5_block_reader(
            source.file_descriptor,
            source.index,
            source.binding,
        )
        destination = _new_destination((1, 1, 3, 2, 4))
        real_pread = os.pread
        identity_changed = False

        def mutate_identity_after_payload(
            fd: int,
            count: int,
            offset: int,
        ) -> bytes:
            nonlocal identity_changed
            result = real_pread(fd, count, offset)
            if count > 24 and not identity_changed:
                file_stat = os.stat(source.path)
                os.utime(
                    source.path,
                    ns=(
                        file_stat.st_atime_ns,
                        file_stat.st_mtime_ns + 1_000_000_000,
                    ),
                )
                identity_changed = True
            return result

        monkeypatch.setattr(
            dat_reader_module.os,
            "pread",
            mutate_identity_after_payload,
        )
        with pytest.raises(OSError, match="changed during transfer"):
            read_blocks_into(
                reader,
                i3(0, 0, 0),
                i3(3, 2, 4),
                np.asarray([0], dtype=np.int64),
                np.asarray([0], dtype=np.int64),
                destination,
                i3(0, 0, 0),
            )
        assert identity_changed
        assert np.array_equal(
            destination.view(np.uint64)[0, 0],
            source.stored_bits[0][0],
        )


def test_reused_borrowed_fd_as_nonregular_raises_oserror(
    tmp_path: Path,
) -> None:
    ghosts = (((0, 0, 0), (0, 0, 0)),)
    with synthetic_source(tmp_path, ghosts=ghosts) as source:
        reader = make_amrvac_v5_block_reader(
            source.file_descriptor,
            source.index,
            source.binding,
        )
        directory_fd = os.open(tmp_path, os.O_RDONLY)
        try:
            os.dup2(directory_fd, source.file_descriptor)
        finally:
            if directory_fd != source.file_descriptor:
                os.close(directory_fd)

        destination = _new_destination((1, 1, 3, 2, 4))
        before = destination.view(np.uint64).copy()
        with pytest.raises(OSError, match="descriptor was replaced"):
            read_blocks_into(
                reader,
                i3(0, 0, 0),
                i3(3, 2, 4),
                np.asarray([0], dtype=np.int64),
                np.asarray([0], dtype=np.int64),
                destination,
                i3(0, 0, 0),
            )
        assert np.array_equal(destination.view(np.uint64), before)


def test_factory_validates_exact_types_provenance_staggering_and_fd(
    tmp_path: Path,
) -> None:
    with synthetic_source(tmp_path) as source:
        with pytest.raises(TypeError, match="exact Python int"):
            make_amrvac_v5_block_reader(
                np.int64(source.file_descriptor),
                source.index,
                source.binding,
            )
        with pytest.raises(ValueError, match="non-negative"):
            make_amrvac_v5_block_reader(-1, source.index, source.binding)
        with pytest.raises(TypeError, match="AMRVACV5Index"):
            make_amrvac_v5_block_reader(
                source.file_descriptor,
                object(),  # type: ignore[arg-type]
                source.binding,
            )
        with pytest.raises(TypeError, match="AMRVACV5ForestBinding"):
            make_amrvac_v5_block_reader(
                source.file_descriptor,
                source.index,
                object(),  # type: ignore[arg-type]
            )

        wrong_identity = source.index.file_identity[:2] + (
            source.index.file_identity[2] + 1,
        ) + source.index.file_identity[3:]
        wrong_binding = source.binding._replace(
            source_file_identity=wrong_identity
        )
        with pytest.raises(ValueError, match="provenance"):
            make_amrvac_v5_block_reader(
                source.file_descriptor,
                source.index,
                wrong_binding,
            )
        staggered_index = source.index._replace(staggered=True)
        with pytest.raises(ValueError, match="staggered"):
            make_amrvac_v5_block_reader(
                source.file_descriptor,
                staggered_index,
                source.binding,
            )

        duplicate_fd = os.dup(source.file_descriptor)
        os.close(duplicate_fd)
        with pytest.raises(OSError):
            make_amrvac_v5_block_reader(
                duplicate_fd,
                source.index,
                source.binding,
            )


@pytest.mark.parametrize(
    ("mutator", "error", "match"),
    [
        (
            lambda value: np.asarray(value, dtype=np.int32),
            TypeError,
            "dtype int64",
        ),
        (
            lambda value: np.asarray([0, 0], dtype=np.int64),
            ValueError,
            r"shape \(3,\)",
        ),
        (
            lambda value: np.asarray(value, dtype=np.int64)[::-1],
            ValueError,
            "C-contiguous",
        ),
    ],
)
def test_request_metadata_types_are_checked_before_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutator: object,
    error: type[Exception],
    match: str,
) -> None:
    with synthetic_source(tmp_path) as source:
        reader = make_amrvac_v5_block_reader(
            source.file_descriptor,
            source.index,
            source.binding,
        )
        destination = _new_destination((1, 1, 3, 2, 4))
        lower = mutator([0, 0, 0])  # type: ignore[operator]

        def forbidden(*args: object, **kwargs: object) -> object:
            raise AssertionError("invalid request performed file I/O")

        monkeypatch.setattr(dat_reader_module.os, "fstat", forbidden)
        monkeypatch.setattr(dat_reader_module.os, "pread", forbidden)
        with pytest.raises(error, match=match):
            read_blocks_into(
                reader,
                lower,
                i3(3, 2, 4),
                np.asarray([0], dtype=np.int64),
                np.asarray([0], dtype=np.int64),
                destination,
                i3(0, 0, 0),
            )


def test_destination_translation_overflow_precedes_file_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with synthetic_source(tmp_path) as source:
        reader = make_amrvac_v5_block_reader(
            source.file_descriptor,
            source.index,
            source.binding,
        )
        destination = _new_destination((1, 1, 3, 2, 4))

        def forbidden(*args: object, **kwargs: object) -> object:
            raise AssertionError("overflowing request performed file I/O")

        monkeypatch.setattr(dat_reader_module.os, "fstat", forbidden)
        monkeypatch.setattr(dat_reader_module.os, "pread", forbidden)
        with pytest.raises(OverflowError, match="destination upper"):
            read_blocks_into(
                reader,
                i3(0, 0, 0),
                i3(1, 1, 1),
                np.asarray([0], dtype=np.int64),
                np.asarray([0], dtype=np.int64),
                destination,
                i3(np.iinfo(np.int64).max, 0, 0),
            )
