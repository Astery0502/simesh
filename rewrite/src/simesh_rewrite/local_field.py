"""LFE-001 bounded selected refined curl and regional reduction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from ._curl import cartesian_curl_unchecked
from ._reductions import accumulate_field_sum_unchecked
from .blockio import BlockReader, _require_block_reader
from .completed_primary import (
    execute_selected_refined_halos_with_consumer,
    make_completed_primary_consumer,
)
from .foundation import INDEX_DTYPE, _require_payload
from .reductions import _require_accumulator
from .refined_geometry import refined_leaf_geometry
from .selected_refined_support import (
    _require_primary_selection,
    _validate_primary_selection,
)
from .storage import _require_index_vector
from .workspace import _require_nonnegative_integer


_INDEX_MAX = int(np.iinfo(np.int64).max)


class SelectedCurlExecutionStats(NamedTuple):
    primary_count: int
    output_cell_count: int
    output_value_count: int
    chunk_count: int
    reader_call_count: int
    consumer_call_count: int
    operator_call_count: int
    reduction_call_count: int
    selected_load_count: int
    support_load_count: int
    maximum_selected_slots: int
    logical_reader_bytes: int
    managed_array_bytes: int


@dataclass(slots=True)
class _SelectedCurlConsumerState:
    primary_leaf_ids: np.ndarray
    cell_lower: np.ndarray
    cell_upper: np.ndarray
    slot_cell_spacing: np.ndarray
    curl_values: np.ndarray
    reduction_component: int
    accumulator: np.ndarray
    field_positions: np.ndarray
    source_lower: np.ndarray
    source_upper: np.ndarray
    call_counts: np.ndarray


def _checked_add(name: str, left: int, right: int) -> int:
    result = left + right
    if result < 0 or result > _INDEX_MAX:
        raise OverflowError(f"{name} does not fit in int64")
    return result


def _checked_multiply(name: str, left: int, right: int) -> int:
    if left < 0 or right < 0:
        raise ValueError(f"{name} factors must be nonnegative")
    if left and right > _INDEX_MAX // left:
        raise OverflowError(f"{name} does not fit in int64")
    return left * right


def _require_cell_boxes(
    name: str,
    value: np.ndarray,
    primary_count: int,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != INDEX_DTYPE:
        raise TypeError(f"{name} must have dtype int64")
    if value.shape != (primary_count, 3):
        raise ValueError(
            f"{name} must have shape {(primary_count, 3)}, got {value.shape}"
        )
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def _require_reduction_component(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("reduction_component must be an integer")
    result = int(value)
    if result < 0 or result >= 3:
        raise ValueError("reduction_component must be in [0, 3)")
    return result


def _require_max_level(value: int, node_levels: np.ndarray) -> int:
    if type(value) is not int:
        raise TypeError("max_level must be an exact Python int")
    if value <= 0:
        raise ValueError("max_level must be positive")
    if node_levels.size and int(np.max(node_levels)) != value:
        raise ValueError("max_level must equal the maximum forest node level")
    return value


def _validate_output_nonoverlap(
    reader: BlockReader,
    curl_values: np.ndarray,
    accumulator: np.ndarray,
    metadata: tuple[np.ndarray, ...],
) -> None:
    outputs = (curl_values, accumulator)
    inputs = (*reader.memory_arrays, *metadata)
    if np.shares_memory(curl_values, accumulator):
        raise ValueError("curl_values and accumulator must not overlap")
    for output in outputs:
        if any(np.shares_memory(output, value) for value in inputs):
            raise ValueError(
                "mutable outputs must not overlap reader memory or metadata"
            )


def _validate_boxes_and_count_cells(
    cell_lower: np.ndarray,
    cell_upper: np.ndarray,
    block_shape: tuple[int, int, int],
) -> int:
    output_cell_count = 0
    for row in range(cell_lower.shape[0]):
        volume = 1
        for axis in range(3):
            lower = int(cell_lower[row, axis])
            upper = int(cell_upper[row, axis])
            if lower < 0 or lower >= upper or upper > block_shape[axis]:
                raise ValueError(
                    "cell boxes must be nonempty and contained in the block"
                )
            volume = _checked_multiply("cell-box volume", volume, upper - lower)
        output_cell_count = _checked_add(
            "output cell count", output_cell_count, volume
        )
    return output_cell_count


def _consume_selected_curl_runs(
    state: _SelectedCurlConsumerState,
    input_primary_offset: int,
    primary_leaf_ids: np.ndarray,
    completed_payload: np.ndarray,
    payload_valid_lower: np.ndarray,
    payload_valid_upper: np.ndarray,
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
) -> None:
    del payload_valid_lower, payload_valid_upper, interior_upper
    primary_count = int(primary_leaf_ids.shape[0])
    expected_ids = state.primary_leaf_ids[
        input_primary_offset : input_primary_offset + primary_count
    ]
    if not np.array_equal(primary_leaf_ids, expected_ids):
        raise RuntimeError("RHC completed primary IDs do not match the input slice")

    run_first = 0
    while run_first < primary_count:
        global_first = input_primary_offset + run_first
        run_stop = run_first + 1
        while run_stop < primary_count:
            global_stop = input_primary_offset + run_stop
            if not np.array_equal(
                state.cell_lower[global_stop], state.cell_lower[global_first]
            ) or not np.array_equal(
                state.cell_upper[global_stop], state.cell_upper[global_first]
            ):
                break
            run_stop += 1

        global_stop = input_primary_offset + run_stop
        np.add(
            interior_lower,
            state.cell_lower[global_first],
            out=state.source_lower,
        )
        np.add(
            interior_lower,
            state.cell_upper[global_first],
            out=state.source_upper,
        )
        cartesian_curl_unchecked(
            completed_payload[run_first:run_stop],
            state.source_lower,
            state.source_upper,
            state.field_positions,
            state.slot_cell_spacing[global_first:global_stop],
            state.curl_values[global_first:global_stop],
            state.field_positions,
            state.cell_lower[global_first],
        )
        state.call_counts[0] += 1
        accumulate_field_sum_unchecked(
            state.curl_values[global_first:global_stop],
            state.cell_lower[global_first],
            state.cell_upper[global_first],
            state.reduction_component,
            state.accumulator,
        )
        state.call_counts[1] += 1
        run_first = run_stop


def execute_selected_refined_curl_from_blocks(
    reader: BlockReader,
    primary_leaf_ids: np.ndarray,
    cell_lower: np.ndarray,
    cell_upper: np.ndarray,
    magnetic_field_ids: np.ndarray,
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    domain_cell_counts: np.ndarray,
    block_cell_counts: np.ndarray,
    max_level: int,
    root_shape: np.ndarray,
    coord_to_rank: np.ndarray,
    root_node_ids: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    child_node_ids: np.ndarray,
    node_leaf_ids: np.ndarray,
    leaf_node_ids: np.ndarray,
    boundary_modes: np.ndarray,
    normal_field_slots: np.ndarray,
    slot_capacity: int,
    curl_values: np.ndarray,
    reduction_component: int,
    accumulator: np.ndarray,
) -> SelectedCurlExecutionStats:
    """Compute compact selected curl values and one serial component sum."""
    reader = _require_block_reader(reader)
    primary_leaf_ids = _require_primary_selection(primary_leaf_ids)
    primary_count = int(primary_leaf_ids.shape[0])
    cell_lower = _require_cell_boxes("cell_lower", cell_lower, primary_count)
    cell_upper = _require_cell_boxes("cell_upper", cell_upper, primary_count)
    magnetic_field_ids = _require_index_vector(
        "magnetic_field_ids", magnetic_field_ids
    )
    if magnetic_field_ids.shape != (3,):
        raise ValueError("magnetic_field_ids must have shape (3,)")
    leaf_node_ids = _require_index_vector("leaf_node_ids", leaf_node_ids)
    if reader.shape[0] != leaf_node_ids.shape[0]:
        raise ValueError("reader block axis must equal leaf count")
    _validate_primary_selection(primary_leaf_ids, int(reader.shape[0]))
    block_shape = tuple(int(value) for value in reader.shape[2:])
    if not isinstance(block_cell_counts, np.ndarray):
        raise TypeError("block_cell_counts must be a NumPy array")
    if block_cell_counts.dtype != INDEX_DTYPE:
        raise TypeError("block_cell_counts must have dtype int64")
    if (
        block_cell_counts.shape != (3,)
        or not block_cell_counts.flags.c_contiguous
    ):
        raise ValueError("block_cell_counts must be a C-contiguous triplet")
    if tuple(int(value) for value in block_cell_counts) != block_shape:
        raise ValueError("block_cell_counts must equal reader spatial extents")
    for position, field_id in enumerate(magnetic_field_ids):
        if int(field_id) < 0 or int(field_id) >= reader.shape[1]:
            raise ValueError(
                f"magnetic_field_ids entry {position} is out of range"
            )
    slot_capacity = _require_nonnegative_integer("slot_capacity", slot_capacity)
    if slot_capacity > reader.shape[0]:
        raise ValueError("slot_capacity exceeds leaf count")
    reduction_component = _require_reduction_component(reduction_component)

    output_cell_count = _validate_boxes_and_count_cells(
        cell_lower, cell_upper, block_shape
    )
    output_value_count = _checked_multiply(
        "output value count", output_cell_count, 3
    )
    curl_values = _require_payload("curl_values", curl_values, writable=True)
    expected_shape = (primary_count, 3, *block_shape)
    if curl_values.shape != expected_shape:
        raise ValueError(
            f"curl_values must have shape {expected_shape}, got {curl_values.shape}"
        )
    accumulator = _require_accumulator(
        "accumulator", accumulator, writable=True
    )

    metadata = tuple(
        value
        for value in (
            primary_leaf_ids,
            cell_lower,
            cell_upper,
            magnetic_field_ids,
            domain_lower,
            domain_upper,
            domain_cell_counts,
            block_cell_counts,
            root_shape,
            coord_to_rank,
            root_node_ids,
            node_levels,
            node_coords,
            child_node_ids,
            node_leaf_ids,
            leaf_node_ids,
            boundary_modes,
            normal_field_slots,
        )
        if isinstance(value, np.ndarray)
    )
    _validate_output_nonoverlap(reader, curl_values, accumulator, metadata)

    leaf_bounds, slot_cell_spacing = refined_leaf_geometry(
        domain_lower,
        domain_upper,
        root_shape,
        domain_cell_counts,
        block_cell_counts,
        node_levels,
        node_coords,
        leaf_node_ids,
        primary_leaf_ids,
    )
    _require_max_level(max_level, node_levels)
    leaf_bounds.setflags(write=False)
    slot_cell_spacing.setflags(write=False)

    block_volume = 1
    for extent in block_shape:
        block_volume = _checked_multiply(
            "reader block volume", block_volume, extent
        )
    maximum_loads = _checked_multiply(
        "maximum selected load count", primary_count, slot_capacity
    )
    maximum_reader_values = _checked_multiply(
        "maximum logical reader values", maximum_loads, 3
    )
    maximum_reader_values = _checked_multiply(
        "maximum logical reader values", maximum_reader_values, block_volume
    )
    _checked_multiply("maximum logical reader bytes", maximum_reader_values, 8)

    one = np.ones(3, dtype=np.int64)
    field_positions = np.arange(3, dtype=np.int64)
    source_lower = np.empty(3, dtype=np.int64)
    source_upper = np.empty(3, dtype=np.int64)
    call_counts = np.zeros(2, dtype=np.int64)
    additional_arrays = (
        leaf_bounds,
        slot_cell_spacing,
        one,
        field_positions,
        source_lower,
        source_upper,
        call_counts,
    )
    additional_managed_array_bytes = sum(
        int(value.nbytes) for value in additional_arrays
    )
    if additional_managed_array_bytes > _INDEX_MAX:
        raise OverflowError("LFE managed raw-array bytes do not fit in int64")

    state = _SelectedCurlConsumerState(
        primary_leaf_ids,
        cell_lower,
        cell_upper,
        slot_cell_spacing,
        curl_values,
        reduction_component,
        accumulator,
        field_positions,
        source_lower,
        source_upper,
        call_counts,
    )
    consumer = make_completed_primary_consumer(
        state,
        _consume_selected_curl_runs,
        output_arrays=(
            curl_values,
            accumulator,
            source_lower,
            source_upper,
            call_counts,
        ),
    )
    execution = execute_selected_refined_halos_with_consumer(
        reader,
        consumer,
        primary_leaf_ids,
        magnetic_field_ids,
        root_shape,
        coord_to_rank,
        root_node_ids,
        node_levels,
        node_coords,
        child_node_ids,
        node_leaf_ids,
        leaf_node_ids,
        one,
        one,
        boundary_modes,
        normal_field_slots,
        slot_capacity,
        _additional_managed_array_bytes=additional_managed_array_bytes,
    )

    logical_reader_values = _checked_multiply(
        "logical reader values", execution.selected_load_count, 3
    )
    logical_reader_values = _checked_multiply(
        "logical reader values", logical_reader_values, block_volume
    )
    logical_reader_bytes = _checked_multiply(
        "logical reader bytes", logical_reader_values, 8
    )
    managed_array_bytes = _checked_add(
        "managed raw-array bytes",
        execution.managed_array_bytes,
        additional_managed_array_bytes,
    )
    return SelectedCurlExecutionStats(
        primary_count,
        output_cell_count,
        output_value_count,
        execution.chunk_count,
        execution.reader_call_count,
        execution.consumer_call_count,
        int(call_counts[0]),
        int(call_counts[1]),
        execution.selected_load_count,
        execution.selected_load_count - primary_count,
        execution.maximum_selected_slots,
        logical_reader_bytes,
        managed_array_bytes,
    )
