"""Native selective reads from ordinary AMRVAC v5 block records."""

from __future__ import annotations

import errno
import os
import stat
import struct
from dataclasses import dataclass
from typing import Final

import numpy as np

from .amrvac_dat import AMRVACV5ForestBinding, AMRVACV5Index
from .blockio import BlockReader, make_block_reader


_INDEX_DTYPE: Final = np.dtype(np.int64)
_PAYLOAD_DTYPE: Final = np.dtype(np.float64)
_NATIVE_BITS_DTYPE: Final = np.dtype(np.uint64)
_INDEX_MAX: Final = int(np.iinfo(np.int64).max)
_GHOST_HEADER_BYTES: Final = 6 * 4


@dataclass(frozen=True, slots=True)
class _AMRVACV5BlockReaderState:
    file_descriptor: int
    byte_order: str
    file_identity: tuple[int, int, int, int, int]
    shape: tuple[int, int, int, int, int]
    block_offsets: np.ndarray
    staggered: bool = False
    zero_ghost_fast: bool = True
    zero_cells: int = 0
    zero_ordinary_bytes: int = 0
    zero_complete_bytes: int = 0


@dataclass(frozen=True, slots=True)
class _SelectedRecord:
    block_id: int
    block_offset: int
    record_end: int
    stored_shape: tuple[int, int, int]
    stored_cell_count: int
    stored_lower: tuple[int, int, int]
    stored_upper: tuple[int, int, int]
    complete_zero_ghost: bool


def _checked_add(left: int, right: int, *, what: str) -> int:
    if left < 0 or right < 0:
        raise ValueError(f"{what} inputs must be non-negative")
    if left > _INDEX_MAX - right:
        raise OverflowError(f"{what} does not fit in int64")
    return left + right


def _checked_mul(left: int, right: int, *, what: str) -> int:
    if left < 0 or right < 0:
        raise ValueError(f"{what} inputs must be non-negative")
    if left != 0 and right > _INDEX_MAX // left:
        raise OverflowError(f"{what} does not fit in int64")
    return left * right


def _file_identity(file_stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(file_stat.st_dev),
        int(file_stat.st_ino),
        int(file_stat.st_size),
        int(file_stat.st_mtime_ns),
        int(file_stat.st_ctime_ns),
    )


def _current_identity(
    file_descriptor: int,
    *,
    callback_lifecycle: bool = False,
) -> tuple[int, int, int, int, int]:
    file_stat = os.fstat(file_descriptor)
    if not stat.S_ISREG(file_stat.st_mode):
        if callback_lifecycle:
            raise OSError(
                errno.EBADF,
                "borrowed AMRVAC file descriptor was replaced",
            )
        raise ValueError("file_descriptor must name a regular file")
    if file_stat.st_size < 0 or file_stat.st_size > _INDEX_MAX:
        raise OverflowError("file size does not fit in int64")
    return _file_identity(file_stat)


def _require_index_triplet(name: str, value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != _INDEX_DTYPE:
        raise TypeError(f"{name} must have dtype int64")
    if value.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def _require_index_vector(name: str, value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != _INDEX_DTYPE:
        raise TypeError(f"{name} must have dtype int64")
    if value.ndim != 1:
        raise ValueError(f"{name} must have rank one, got shape {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def _require_destination(value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("destination must be a NumPy array")
    if value.dtype != _PAYLOAD_DTYPE:
        raise TypeError("destination must have native dtype float64")
    if value.ndim != 5:
        raise ValueError(
            "destination must have layout (slot, field, x, y, z), "
            f"got rank {value.ndim}"
        )
    if not value.flags.c_contiguous:
        raise ValueError("destination must be C-contiguous")
    if not value.flags.writeable:
        raise ValueError("destination must be writable")
    return value


def _validate_selectors(
    name: str,
    selectors: np.ndarray,
    extent: int,
) -> None:
    for position, raw_value in enumerate(selectors):
        value = int(raw_value)
        if value < 0 or value >= extent:
            raise ValueError(f"{name} entry {position} is out of range")


def _selector_positions(selectors: np.ndarray) -> dict[int, list[int]]:
    positions: dict[int, list[int]] = {}
    for position, raw_value in enumerate(selectors):
        positions.setdefault(int(raw_value), []).append(position)
    return positions


def _consecutive_runs(sorted_ids: list[int]) -> list[tuple[int, int]]:
    if not sorted_ids:
        return []
    runs: list[tuple[int, int]] = []
    start = sorted_ids[0]
    previous = start
    for value in sorted_ids[1:]:
        if value != previous + 1:
            runs.append((start, previous + 1))
            start = value
        previous = value
    runs.append((start, previous + 1))
    return runs


def _pread_exact(
    file_descriptor: int,
    byte_count: int,
    offset: int,
    *,
    section: str,
) -> bytes:
    raw = os.pread(file_descriptor, byte_count, offset)
    actual = len(raw)
    if actual != byte_count:
        raise OSError(
            errno.EIO,
            f"short {section} read at offset {offset}: "
            f"expected {byte_count} bytes, got {actual}",
        )
    return raw


def _linear_stored_index(
    x: int,
    y: int,
    z: int,
    shape: tuple[int, int, int],
    *,
    what: str,
) -> int:
    nx, ny, _ = shape
    nyz = _checked_mul(ny, z, what=what)
    y_nyz = _checked_add(y, nyz, what=what)
    nx_yz = _checked_mul(nx, y_nyz, what=what)
    return _checked_add(x, nx_yz, what=what)


def _full_run_location(
    record: _SelectedRecord,
    field_start: int,
    field_stop: int,
) -> tuple[int, int]:
    field_cells = _checked_mul(
        field_start,
        record.stored_cell_count,
        what="field offset",
    )
    field_bytes = _checked_mul(field_cells, 8, what="field byte offset")
    payload_start = _checked_add(
        record.block_offset,
        _GHOST_HEADER_BYTES,
        what="record payload offset",
    )
    offset = _checked_add(payload_start, field_bytes, what="field address")
    run_cells = _checked_mul(
        field_stop - field_start,
        record.stored_cell_count,
        what="field-run cells",
    )
    byte_count = _checked_mul(run_cells, 8, what="field-run bytes")
    end = _checked_add(offset, byte_count, what="field-run end")
    if end > record.record_end:
        raise ValueError(
            f"block {record.block_id} field run exceeds its record"
        )
    return offset, byte_count


def _envelope_location(
    record: _SelectedRecord,
    field_id: int,
) -> tuple[int, int]:
    x0, y0, z0 = record.stored_lower
    x1, y1, z1 = record.stored_upper
    first = _linear_stored_index(
        x0,
        y0,
        z0,
        record.stored_shape,
        what="envelope first index",
    )
    last_inclusive = _linear_stored_index(
        x1 - 1,
        y1 - 1,
        z1 - 1,
        record.stored_shape,
        what="envelope last index",
    )
    last = _checked_add(last_inclusive, 1, what="envelope last index")
    if first >= last or last > record.stored_cell_count:
        raise ValueError(f"block {record.block_id} has an invalid read envelope")

    field_cells = _checked_mul(
        field_id,
        record.stored_cell_count,
        what="field offset",
    )
    cell_offset = _checked_add(field_cells, first, what="field envelope offset")
    byte_offset = _checked_mul(cell_offset, 8, what="field envelope byte offset")
    payload_start = _checked_add(
        record.block_offset,
        _GHOST_HEADER_BYTES,
        what="record payload offset",
    )
    offset = _checked_add(payload_start, byte_offset, what="field address")
    byte_count = _checked_mul(last - first, 8, what="field envelope bytes")
    end = _checked_add(offset, byte_count, what="field envelope end")
    if end > record.record_end:
        raise ValueError(
            f"block {record.block_id} field envelope exceeds its record"
        )
    return offset, byte_count


def _decode_native_bits(raw: bytes, byte_order: str) -> np.ndarray:
    disk_dtype = np.dtype(f"{byte_order}u8")
    return np.frombuffer(raw, dtype=disk_dtype).astype(
        _NATIVE_BITS_DTYPE,
        copy=False,
    )


def _read_selected_headers(
    state: _AMRVACV5BlockReaderState,
    sorted_block_ids: list[int],
    source_lower: tuple[int, int, int],
    source_upper: tuple[int, int, int],
) -> list[_SelectedRecord]:
    raw_headers: list[tuple[int, bytes]] = []
    for block_id in sorted_block_ids:
        block_offset = int(state.block_offsets[block_id])
        raw_headers.append(
            (
                block_id,
                _pread_exact(
                    state.file_descriptor,
                    _GHOST_HEADER_BYTES,
                    block_offset,
                    section="record header",
                ),
            )
        )

    block_shape = state.shape[2:]
    field_count = state.shape[1]
    file_size = state.file_identity[2]
    header_struct = struct.Struct(f"{state.byte_order}6i")
    records: list[_SelectedRecord] = []
    source_is_complete = source_lower == (0,0,0) and source_upper == block_shape
    for block_id, raw_header in raw_headers:
        ghost_values = header_struct.unpack(raw_header)
        if state.zero_ghost_fast and state.zero_cells and not any(ghost_values):
            block_offset = int(state.block_offsets[block_id])
            expected_end = (int(state.block_offsets[block_id+1])
                            if block_id+1 < state.shape[0] else file_size)
            complete_end = block_offset+state.zero_complete_bytes
            if complete_end != expected_end:
                raise ValueError(f"block {block_id} record end {complete_end} does not equal expected offset {expected_end}")
            # Equality to a validated in-file next offset proves these positive
            # prefix addresses fit int64. Request boxes were already validated.
            records.append(_SelectedRecord(block_id,block_offset,
                block_offset+state.zero_ordinary_bytes,block_shape,state.zero_cells,
                source_lower,source_upper,source_is_complete))
            continue
        lower_ghost = tuple(int(value) for value in ghost_values[:3])
        upper_ghost = tuple(int(value) for value in ghost_values[3:])
        if any(value < 0 for value in lower_ghost + upper_ghost):
            raise ValueError(f"block {block_id} has negative saved ghost extents")

        stored_shape_values: list[int] = []
        stored_lower_values: list[int] = []
        stored_upper_values: list[int] = []
        for axis in range(3):
            stored_extent = _checked_add(
                block_shape[axis],
                lower_ghost[axis],
                what="stored shape",
            )
            stored_extent = _checked_add(
                stored_extent,
                upper_ghost[axis],
                what="stored shape",
            )
            if stored_extent <= 0:
                raise ValueError(f"block {block_id} has a nonpositive stored shape")
            translated_lower = _checked_add(
                source_lower[axis],
                lower_ghost[axis],
                what="translated source lower",
            )
            translated_upper = _checked_add(
                source_upper[axis],
                lower_ghost[axis],
                what="translated source upper",
            )
            if (
                translated_lower > translated_upper
                or translated_upper > stored_extent
            ):
                raise ValueError(
                    f"block {block_id} source region is outside stored shape"
                )
            stored_shape_values.append(stored_extent)
            stored_lower_values.append(translated_lower)
            stored_upper_values.append(translated_upper)

        stored_shape = tuple(stored_shape_values)
        cell_count = _checked_mul(
            stored_shape[0],
            stored_shape[1],
            what="stored cell count",
        )
        cell_count = _checked_mul(
            cell_count,
            stored_shape[2],
            what="stored cell count",
        )
        value_count = _checked_mul(
            field_count,
            cell_count,
            what="record value count",
        )
        payload_bytes = _checked_mul(value_count, 8, what="record payload bytes")
        block_offset = int(state.block_offsets[block_id])
        payload_start = _checked_add(
            block_offset,
            _GHOST_HEADER_BYTES,
            what="record payload offset",
        )
        record_end = _checked_add(
            payload_start,
            payload_bytes,
            what="record end",
        )
        expected_end = (
            int(state.block_offsets[block_id + 1])
            if block_id + 1 < state.shape[0]
            else file_size
        )
        complete_record_end = record_end
        if state.staggered:
            tail_cells = 1
            for stored_extent in stored_shape:
                tail_cells = _checked_mul(tail_cells,
                    _checked_add(stored_extent, 1, what="staggered tail shape"),
                    what="staggered tail cells")
            tail_bytes = _checked_mul(tail_cells, 24, what="three staggered tail bytes")
            complete_record_end = _checked_add(record_end, tail_bytes, what="complete staggered record end")
        if complete_record_end != expected_end:
            raise ValueError(
                f"block {block_id} record end {complete_record_end} does not equal "
                f"expected offset {expected_end}"
            )
        records.append(
            _SelectedRecord(
                block_id=block_id,
                block_offset=block_offset,
                record_end=record_end,
                stored_shape=stored_shape,  # type: ignore[arg-type]
                stored_cell_count=cell_count,
                stored_lower=tuple(stored_lower_values),  # type: ignore[arg-type]
                stored_upper=tuple(stored_upper_values),  # type: ignore[arg-type]
                complete_zero_ghost=(
                    source_lower == (0, 0, 0)
                    and source_upper == block_shape
                    and lower_ghost == (0, 0, 0)
                    and upper_ghost == (0, 0, 0)
                ),
            )
        )
    return records


def _copy_complete_runs(
    state: _AMRVACV5BlockReaderState,
    record: _SelectedRecord,
    block_positions: dict[int, list[int]],
    field_positions: dict[int, list[int]],
    field_runs: list[tuple[int, int]],
    destination_bits: np.ndarray,
    destination_lower: tuple[int, int, int],
    packed_fields: dict[tuple[int,int],int] | None = None,
) -> None:
    dx, dy, dz = state.shape[2:]
    destination_slices = tuple(
        slice(destination_lower[axis], destination_lower[axis] + extent)
        for axis, extent in enumerate((dx, dy, dz))
    )
    for field_start, field_stop in field_runs:
        offset, byte_count = _full_run_location(record, field_start, field_stop)
        raw = _pread_exact(
            state.file_descriptor,
            byte_count,
            offset,
            section="payload",
        )
        bits = _decode_native_bits(raw, state.byte_order)
        disk_fields = bits.reshape(field_stop - field_start, dz, dy, dx)
        canonical_fields = disk_fields.transpose(0, 3, 2, 1)
        packed_start = None if packed_fields is None else packed_fields.get((field_start,field_stop))
        if packed_start is not None:
            for slot in block_positions[record.block_id]:
                destination_bits[slot,packed_start:packed_start+field_stop-field_start,
                    destination_slices[0],destination_slices[1],destination_slices[2]] = canonical_fields
            del canonical_fields,disk_fields,bits,raw
            continue
        for field_id in range(field_start, field_stop):
            values = canonical_fields[field_id - field_start]
            for slot in block_positions[record.block_id]:
                for field_position in field_positions[field_id]:
                    destination_bits[
                        slot,
                        field_position,
                        destination_slices[0],
                        destination_slices[1],
                        destination_slices[2],
                    ] = values
        # Do not carry one run's bytes/views into allocation of the next run.
        del values, canonical_fields, disk_fields, bits, raw


def _copy_partial_fields(
    state: _AMRVACV5BlockReaderState,
    record: _SelectedRecord,
    block_positions: dict[int, list[int]],
    field_positions: dict[int, list[int]],
    sorted_field_ids: list[int],
    extent: tuple[int, int, int],
    destination_bits: np.ndarray,
    destination_lower: tuple[int, int, int],
) -> None:
    nx, ny, _ = record.stored_shape
    destination_slices = tuple(
        slice(destination_lower[axis], destination_lower[axis] + extent[axis])
        for axis in range(3)
    )
    for field_id in sorted_field_ids:
        offset, byte_count = _envelope_location(record, field_id)
        raw = _pread_exact(
            state.file_descriptor,
            byte_count,
            offset,
            section="payload",
        )
        bits = _decode_native_bits(raw, state.byte_order)
        selected = np.lib.stride_tricks.as_strided(
            bits,
            shape=extent,
            strides=(8, 8 * nx, 8 * nx * ny),
            writeable=False,
        )
        for slot in block_positions[record.block_id]:
            for field_position in field_positions[field_id]:
                destination_bits[
                    slot,
                    field_position,
                    destination_slices[0],
                    destination_slices[1],
                    destination_slices[2],
                ] = selected
        # Bound live payload scratch to one envelope plus endian conversion.
        del selected, bits, raw


def _read_amrvac_v5_blocks_into(
    state: _AMRVACV5BlockReaderState,
    source_valid_lower: np.ndarray,
    source_valid_upper: np.ndarray,
    block_ids: np.ndarray,
    field_ids: np.ndarray,
    destination: np.ndarray,
    destination_lower: np.ndarray,
) -> None:
    source_valid_lower = _require_index_triplet(
        "source_valid_lower", source_valid_lower
    )
    source_valid_upper = _require_index_triplet(
        "source_valid_upper", source_valid_upper
    )
    block_ids = _require_index_vector("block_ids", block_ids)
    field_ids = _require_index_vector("field_ids", field_ids)
    destination = _require_destination(destination)
    destination_lower = _require_index_triplet(
        "destination_lower", destination_lower
    )

    if destination.shape[:2] != (block_ids.shape[0], field_ids.shape[0]):
        raise ValueError(
            "destination slot/field shape must match block_ids and field_ids"
        )
    source_lower = tuple(int(value) for value in source_valid_lower)
    source_upper = tuple(int(value) for value in source_valid_upper)
    block_shape = state.shape[2:]
    for axis in range(3):
        if source_lower[axis] < 0:
            raise ValueError("source_valid lower bound must be non-negative")
        if source_lower[axis] > source_upper[axis]:
            raise ValueError("source_valid lower bound exceeds its upper bound")
        if source_upper[axis] > block_shape[axis]:
            raise ValueError("source_valid exceeds the source spatial shape")

    extent = tuple(source_upper[axis] - source_lower[axis] for axis in range(3))
    destination_start = tuple(int(value) for value in destination_lower)
    destination_shape = destination.shape[2:]
    for axis in range(3):
        if destination_start[axis] < 0:
            raise ValueError("destination lower bound must be non-negative")
        destination_end = _checked_add(
            destination_start[axis],
            extent[axis],
            what="destination upper bound",
        )
        if destination_end > destination_shape[axis]:
            raise ValueError("destination region exceeds the output spatial shape")

    _validate_selectors("block_ids", block_ids, state.shape[0])
    _validate_selectors("field_ids", field_ids, state.shape[1])
    if any(
        np.shares_memory(destination, source)
        for source in (
            source_valid_lower,
            source_valid_upper,
            block_ids,
            field_ids,
            destination_lower,
            state.block_offsets,
        )
    ):
        raise ValueError("destination must not overlap reader or request inputs")

    if block_ids.size == 0 or field_ids.size == 0 or 0 in extent:
        return

    current_identity = _current_identity(
        state.file_descriptor,
        callback_lifecycle=True,
    )
    if current_identity != state.file_identity:
        raise OSError("AMRVAC source file identity changed")

    block_positions = _selector_positions(block_ids)
    field_positions = _selector_positions(field_ids)
    sorted_block_ids = sorted(block_positions)
    sorted_field_ids = sorted(field_positions)
    field_runs = _consecutive_runs(sorted_field_ids)
    packed_fields = {}
    if state.zero_ghost_fast:
        for first,last in field_runs:
            places = [field_positions[field] for field in range(first,last)]
            if all(len(place)==1 for place in places):
                begin = places[0][0]
                if all(place[0]==begin+offset for offset,place in enumerate(places)):
                    packed_fields[first,last] = begin
    records = _read_selected_headers(
        state,
        sorted_block_ids,
        source_lower,
        source_upper,
    )

    # Finish every record-derived address check before the first payload byte is
    # read or the destination is mutated.
    for record in records:
        if record.complete_zero_ghost:
            for field_start, field_stop in field_runs:
                _full_run_location(record, field_start, field_stop)
        else:
            for field_id in sorted_field_ids:
                _envelope_location(record, field_id)

    destination_bits = destination.view(_NATIVE_BITS_DTYPE)
    for record in records:
        if record.complete_zero_ghost:
            _copy_complete_runs(
                state,
                record,
                block_positions,
                field_positions,
                field_runs,
                destination_bits,
                destination_start,
                packed_fields,
            )
        else:
            _copy_partial_fields(
                state,
                record,
                block_positions,
                field_positions,
                sorted_field_ids,
                extent,
                destination_bits,
                destination_start,
            )

    if _current_identity(
        state.file_descriptor,
        callback_lifecycle=True,
    ) != state.file_identity:
        raise OSError("AMRVAC source file identity changed during transfer")


def make_amrvac_v5_block_reader(
    file_descriptor: int,
    index: AMRVACV5Index,
    forest_binding: AMRVACV5ForestBinding,
) -> BlockReader:
    """Create a cache-free STO-003 reader over ordinary AMRVAC v5 records."""
    return _make_amrvac_v5_block_reader(file_descriptor,index,forest_binding,allow_staggered=False)


def make_amrvac_v5_ordinary_block_reader(
    file_descriptor: int,
    index: AMRVACV5Index,
    forest_binding: AMRVACV5ForestBinding,
) -> BlockReader:
    """Read ordinary 3D v5 fields, validating any three-component staggered tail.

    This separate profile does not expose CT/staggered values. It preserves
    DAT-003 transfer/preflight rules and requires complete record sizes including
    tails. The original factory still rejects every staggered index.
    """
    return _make_amrvac_v5_block_reader(file_descriptor,index,forest_binding,allow_staggered=True)


def _make_amrvac_v5_block_reader(file_descriptor,index,forest_binding,*,allow_staggered):
    if type(file_descriptor) is not int:
        raise TypeError("file_descriptor must be an exact Python int")
    if file_descriptor < 0:
        raise ValueError("file_descriptor must be non-negative")
    if type(index) is not AMRVACV5Index:
        raise TypeError("index must be an AMRVACV5Index")
    if type(forest_binding) is not AMRVACV5ForestBinding:
        raise TypeError("forest_binding must be an AMRVACV5ForestBinding")
    if forest_binding.source_file_identity != index.file_identity:
        raise ValueError("forest binding source provenance does not match index")
    if index.staggered and not allow_staggered:
        raise ValueError("staggered AMRVAC block records are unsupported")
    if allow_staggered and index.dimension_count != 3:
        raise ValueError("ordinary staggered-tail profile requires spatially 3D records")
    if index.byte_order not in ("<", ">"):
        raise ValueError("index byte order must be '<' or '>'")

    block_count = int(index.leaf_count)
    field_count = int(index.field_count)
    if block_count < 0 or field_count < 0:
        raise ValueError("index block and field counts must be non-negative")
    block_shape_array = index.block_cell_counts
    if not isinstance(block_shape_array, np.ndarray):
        raise TypeError("index block_cell_counts must be a NumPy array")
    if block_shape_array.dtype != _INDEX_DTYPE:
        raise TypeError("index block_cell_counts must have dtype int64")
    if block_shape_array.shape != (3,) or not block_shape_array.flags.c_contiguous:
        raise ValueError("index block_cell_counts must be C-contiguous shape (3,)")
    block_shape = tuple(int(value) for value in block_shape_array)
    if any(value <= 0 for value in block_shape):
        raise ValueError("index block cell counts must be positive")

    source_offsets = index.block_offsets
    if not isinstance(source_offsets, np.ndarray):
        raise TypeError("index block_offsets must be a NumPy array")
    if source_offsets.dtype != _INDEX_DTYPE:
        raise TypeError("index block_offsets must have dtype int64")
    if source_offsets.shape != (block_count,):
        raise ValueError("index block_offsets shape does not match leaf_count")
    if not source_offsets.flags.c_contiguous:
        raise ValueError("index block_offsets must be C-contiguous")
    if block_count and (
        np.any(source_offsets <= 0) or np.any(source_offsets[1:] <= source_offsets[:-1])
    ):
        raise ValueError("index block_offsets must be positive and increasing")

    expected_identity = tuple(index.file_identity)
    if len(expected_identity) != 5:
        raise ValueError("index file identity must have five entries")
    current_identity = _current_identity(file_descriptor)
    if current_identity != expected_identity:
        raise OSError("file_descriptor does not match the indexed AMRVAC file")

    block_offsets = np.array(
        source_offsets,
        dtype=np.int64,
        order="C",
        copy=True,
    )
    block_offsets.flags.writeable = False
    shape = (
        block_count,
        field_count,
        block_shape[0],
        block_shape[1],
        block_shape[2],
    )
    # Cache geometry-only byte facts. Preserve the old factory's behavior for
    # forged/unrepresentable shapes by leaving those on checked general reads.
    zero_cells = block_shape[0]*block_shape[1]*block_shape[2]
    ordinary_bytes = _GHOST_HEADER_BYTES+zero_cells*field_count*8
    complete_bytes = ordinary_bytes
    if index.staggered:
        complete_bytes += 24*(block_shape[0]+1)*(block_shape[1]+1)*(block_shape[2]+1)
    if zero_cells > _INDEX_MAX or complete_bytes > _INDEX_MAX:
        zero_cells = ordinary_bytes = complete_bytes = 0
    state = _AMRVACV5BlockReaderState(
        file_descriptor=file_descriptor,
        byte_order=index.byte_order,
        file_identity=current_identity,
        shape=shape,
        block_offsets=block_offsets,
        staggered=bool(index.staggered),
        zero_cells=zero_cells,
        zero_ordinary_bytes=ordinary_bytes,
        zero_complete_bytes=complete_bytes,
    )
    return make_block_reader(
        state,
        shape,
        _read_amrvac_v5_blocks_into,
        memory_arrays=(block_offsets,),
    )
