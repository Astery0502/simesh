"""RPS-001 bounded repeated refined point-sampling execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from ._refined_sampling import (
    sample_refined_trilinear_point_groups_unchecked,
    sample_refined_zero_order_point_groups_unchecked,
)
from .blockio import BlockReader, _require_block_reader, read_blocks_into
from .completed_primary import (
    execute_selected_refined_halos_with_consumer,
    make_completed_primary_consumer,
)
from .foundation import PAYLOAD_DTYPE
from .point_location import (
    _require_points,
    fill_refined_point_leaf_ids,
)
from .refined_sampling import _preflight_refined_point_groups
from .storage import _require_index_vector
from .workspace import _require_nonnegative_integer


_INDEX_MAX = int(np.iinfo(np.int64).max)


class RepeatedPointExecutionStats(NamedTuple):
    point_count: int
    inside_point_count: int
    owner_count: int
    chunk_count: int
    reader_call_count: int
    sampler_call_count: int
    selected_load_count: int
    maximum_selected_slots: int
    managed_array_bytes: int


@dataclass(frozen=True, slots=True)
class _RepeatedPointPlan:
    point_owner_leaf_ids: np.ndarray
    grouped_point_indices: np.ndarray
    owner_leaf_ids: np.ndarray
    owner_offsets: np.ndarray


@dataclass(frozen=True, slots=True)
class _TrilinearPointConsumerState:
    domain_lower: np.ndarray
    domain_upper: np.ndarray
    domain_cell_counts: np.ndarray
    block_cell_counts: np.ndarray
    node_levels: np.ndarray
    node_coords: np.ndarray
    leaf_node_ids: np.ndarray
    base_spacing: np.ndarray
    points: np.ndarray
    plan: _RepeatedPointPlan
    point_values: np.ndarray


def _consume_trilinear_point_groups(
    state: _TrilinearPointConsumerState,
    primary_offset: int,
    primary_leaf_ids: np.ndarray,
    payload: np.ndarray,
    _valid_lower: np.ndarray,
    _valid_upper: np.ndarray,
    interior_lower: np.ndarray,
    _interior_upper: np.ndarray,
) -> None:
    stop = primary_offset + int(primary_leaf_ids.shape[0])
    sample_refined_trilinear_point_groups_unchecked(
        payload,
        interior_lower,
        primary_leaf_ids,
        state.domain_lower,
        state.domain_upper,
        state.domain_cell_counts,
        state.block_cell_counts,
        state.node_levels,
        state.node_coords,
        state.leaf_node_ids,
        state.base_spacing,
        state.points,
        state.plan.grouped_point_indices,
        state.plan.owner_offsets[primary_offset : stop + 1],
        state.point_values,
    )


def _require_point_values(
    point_values: np.ndarray,
    point_count: int,
    field_count: int,
) -> np.ndarray:
    if not isinstance(point_values, np.ndarray):
        raise TypeError("point_values must be a NumPy array")
    if point_values.dtype != PAYLOAD_DTYPE:
        raise TypeError("point_values must have dtype float64")
    expected = (point_count, field_count)
    if point_values.shape != expected:
        raise ValueError(
            f"point_values must have shape {expected}, got {point_values.shape}"
        )
    if not point_values.flags.c_contiguous:
        raise ValueError("point_values must be C-contiguous")
    if not point_values.flags.writeable:
        raise ValueError("point_values must be writable")
    return point_values


def _validate_point_output_nonoverlap(
    reader: BlockReader,
    point_values: np.ndarray,
    metadata: tuple[np.ndarray, ...],
) -> None:
    if any(
        np.shares_memory(point_values, source)
        for source in (*reader.memory_arrays, *metadata)
    ):
        raise ValueError("point_values must not overlap reader memory or metadata")


def _validate_reader_request(
    reader: BlockReader,
    field_ids: np.ndarray,
    leaf_node_ids: np.ndarray,
) -> tuple[int, tuple[int, int, int]]:
    if any(value > _INDEX_MAX for value in reader.shape):
        raise OverflowError("reader shape does not fit in int64")
    leaf_count = int(leaf_node_ids.shape[0])
    if reader.shape[0] != leaf_count:
        raise ValueError("reader block axis must equal leaf count")
    for position, field_id in enumerate(field_ids):
        if int(field_id) < 0 or int(field_id) >= reader.shape[1]:
            raise ValueError(f"field_ids entry {position} is out of range")
    return leaf_count, tuple(int(value) for value in reader.shape[2:])


def _make_point_plan(point_owner_leaf_ids: np.ndarray) -> _RepeatedPointPlan:
    point_count = int(point_owner_leaf_ids.shape[0])
    inside_point_count = int(np.count_nonzero(point_owner_leaf_ids >= 0))
    if inside_point_count:
        # NumPy's sorting workspace is transient benchmark state.  The retained
        # plan below contains only the four arrays frozen by the contract.
        order = np.argsort(point_owner_leaf_ids, kind="stable")
        grouped_point_indices = np.array(
            order[point_count - inside_point_count :],
            dtype=np.int64,
            order="C",
            copy=True,
        )
        del order
        grouped_owners = point_owner_leaf_ids[grouped_point_indices]
        starts = np.flatnonzero(
            np.concatenate(
                (
                    np.ones(1, dtype=np.bool_),
                    grouped_owners[1:] != grouped_owners[:-1],
                )
            )
        ).astype(np.int64, copy=False)
        owner_leaf_ids = np.array(
            grouped_owners[starts],
            dtype=np.int64,
            order="C",
            copy=True,
        )
        owner_offsets = np.empty(owner_leaf_ids.shape[0] + 1, dtype=np.int64)
        owner_offsets[:-1] = starts
        owner_offsets[-1] = inside_point_count
    else:
        grouped_point_indices = np.empty(0, dtype=np.int64)
        owner_leaf_ids = np.empty(0, dtype=np.int64)
        owner_offsets = np.zeros(1, dtype=np.int64)
    return _RepeatedPointPlan(
        point_owner_leaf_ids,
        grouped_point_indices,
        owner_leaf_ids,
        owner_offsets,
    )


def _plan_array_bytes(plan: _RepeatedPointPlan) -> int:
    result = sum(
        value.nbytes
        for value in (
            plan.point_owner_leaf_ids,
            plan.grouped_point_indices,
            plan.owner_leaf_ids,
            plan.owner_offsets,
        )
    )
    if result > _INDEX_MAX:
        raise OverflowError("point-plan bytes do not fit in int64")
    return result


def _checked_payload_bytes(
    slot_count: int,
    field_count: int,
    block_shape: tuple[int, int, int],
) -> int:
    spatial_volume = 1
    for extent in block_shape:
        if spatial_volume > _INDEX_MAX // extent:
            raise OverflowError("reader spatial volume does not fit in int64")
        spatial_volume *= extent
    if field_count and spatial_volume > _INDEX_MAX // field_count:
        raise OverflowError("sampling workspace value count does not fit in int64")
    values = field_count * spatial_volume
    if slot_count and values > _INDEX_MAX // slot_count:
        raise OverflowError("sampling workspace value count does not fit in int64")
    values *= slot_count
    if values > _INDEX_MAX // 8:
        raise OverflowError("sampling workspace bytes do not fit in int64")
    return 8 * values


def _prepare_repeated_request(
    reader: BlockReader,
    points: np.ndarray,
    field_ids: np.ndarray,
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    root_shape: np.ndarray,
    domain_cell_counts: np.ndarray,
    block_cell_counts: np.ndarray,
    max_level: int,
    coord_to_rank: np.ndarray,
    root_node_ids: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    child_node_ids: np.ndarray,
    node_leaf_ids: np.ndarray,
    leaf_node_ids: np.ndarray,
    slot_capacity: int,
    point_values: np.ndarray,
    *,
    trilinear: bool,
    extra_metadata: tuple[np.ndarray, ...] = (),
) -> tuple[
    BlockReader,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    tuple[int, int, int],
    _RepeatedPointPlan,
]:
    reader = _require_block_reader(reader)
    points = _require_points(points)
    field_ids = _require_index_vector("field_ids", field_ids)
    leaf_node_ids = _require_index_vector("leaf_node_ids", leaf_node_ids)
    slot_capacity = _require_nonnegative_integer("slot_capacity", slot_capacity)
    point_values = _require_point_values(
        point_values,
        int(points.shape[0]),
        int(field_ids.shape[0]),
    )
    leaf_count, block_shape = _validate_reader_request(
        reader,
        field_ids,
        leaf_node_ids,
    )
    if slot_capacity > leaf_count:
        raise ValueError("slot_capacity exceeds leaf count")

    metadata = tuple(
        value
        for value in (
            points,
            field_ids,
            domain_lower,
            domain_upper,
            root_shape,
            domain_cell_counts,
            block_cell_counts,
            coord_to_rank,
            root_node_ids,
            node_levels,
            node_coords,
            child_node_ids,
            node_leaf_ids,
            leaf_node_ids,
            *extra_metadata,
        )
        if isinstance(value, np.ndarray)
    )
    _validate_point_output_nonoverlap(reader, point_values, metadata)

    point_owner_leaf_ids = np.empty(points.shape[0], dtype=np.int64)
    fill_refined_point_leaf_ids(
        domain_lower,
        domain_upper,
        root_shape,
        domain_cell_counts,
        block_cell_counts,
        max_level,
        coord_to_rank,
        root_node_ids,
        child_node_ids,
        node_leaf_ids,
        points,
        point_owner_leaf_ids,
    )
    if tuple(int(value) for value in block_cell_counts) != block_shape:
        raise ValueError("block_cell_counts must equal reader spatial extents")

    plan = _make_point_plan(point_owner_leaf_ids)
    inside_point_count = int(plan.grouped_point_indices.shape[0])
    if inside_point_count and slot_capacity == 0:
        raise ValueError("slot_capacity must be positive for interior points")
    if trilinear and inside_point_count and slot_capacity < min(57, leaf_count):
        raise ValueError("slot_capacity cannot fit the universal all-26 closure")

    normalized = _preflight_refined_point_groups(
        domain_lower,
        domain_upper,
        domain_cell_counts,
        block_cell_counts,
        node_levels,
        node_coords,
        leaf_node_ids,
        plan.owner_leaf_ids,
        points,
        plan.grouped_point_indices,
        plan.owner_offsets,
        trilinear=trilinear,
    )
    base_spacing = normalized[-1]
    return (
        reader,
        points,
        field_ids,
        base_spacing,
        slot_capacity,
        block_shape,
        plan,
    )


def execute_refined_zero_order_points_from_blocks(
    reader: BlockReader,
    points: np.ndarray,
    field_ids: np.ndarray,
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    root_shape: np.ndarray,
    domain_cell_counts: np.ndarray,
    block_cell_counts: np.ndarray,
    max_level: int,
    coord_to_rank: np.ndarray,
    root_node_ids: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    child_node_ids: np.ndarray,
    node_leaf_ids: np.ndarray,
    leaf_node_ids: np.ndarray,
    slot_capacity: int,
    point_values: np.ndarray,
) -> RepeatedPointExecutionStats:
    """Read and sample stable owner batches without requesting halo support."""
    (
        reader,
        points,
        field_ids,
        base_spacing,
        slot_capacity,
        block_shape,
        plan,
    ) = _prepare_repeated_request(
        reader,
        points,
        field_ids,
        domain_lower,
        domain_upper,
        root_shape,
        domain_cell_counts,
        block_cell_counts,
        max_level,
        coord_to_rank,
        root_node_ids,
        node_levels,
        node_coords,
        child_node_ids,
        node_leaf_ids,
        leaf_node_ids,
        slot_capacity,
        point_values,
        trilinear=False,
    )

    owner_count = int(plan.owner_leaf_ids.shape[0])
    plan_array_bytes = _plan_array_bytes(plan)
    workspace_slots = min(slot_capacity, owner_count)
    _checked_payload_bytes(workspace_slots, int(field_ids.shape[0]), block_shape)
    payload = np.empty(
        (workspace_slots, int(field_ids.shape[0]), *block_shape),
        dtype=np.float64,
    )
    zero = np.zeros(3, dtype=np.int64)
    managed_array_bytes = plan_array_bytes + sum(
        value.nbytes for value in (base_spacing, payload, zero)
    )
    if managed_array_bytes > _INDEX_MAX:
        raise OverflowError("managed raw-array bytes do not fit in int64")

    read_blocks_into(
        reader,
        zero,
        block_cell_counts,
        plan.owner_leaf_ids[:0],
        field_ids,
        payload[:0],
        zero,
    )

    first = 0
    chunk_count = 0
    maximum_selected_slots = 0
    while first < owner_count:
        stop = min(first + slot_capacity, owner_count)
        selected = plan.owner_leaf_ids[first:stop]
        read_blocks_into(
            reader,
            zero,
            block_cell_counts,
            selected,
            field_ids,
            payload[: stop - first],
            zero,
        )
        sample_refined_zero_order_point_groups_unchecked(
            payload[: stop - first],
            zero,
            selected,
            domain_lower,
            domain_upper,
            domain_cell_counts,
            block_cell_counts,
            node_levels,
            node_coords,
            leaf_node_ids,
            base_spacing,
            points,
            plan.grouped_point_indices,
            plan.owner_offsets[first : stop + 1],
            point_values,
        )
        maximum_selected_slots = max(maximum_selected_slots, stop - first)
        first = stop
        chunk_count += 1

    return RepeatedPointExecutionStats(
        int(points.shape[0]),
        int(plan.grouped_point_indices.shape[0]),
        owner_count,
        chunk_count,
        chunk_count,
        chunk_count,
        owner_count,
        maximum_selected_slots,
        managed_array_bytes,
    )


def execute_refined_trilinear_points_from_blocks(
    reader: BlockReader,
    points: np.ndarray,
    field_ids: np.ndarray,
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    root_shape: np.ndarray,
    domain_cell_counts: np.ndarray,
    block_cell_counts: np.ndarray,
    max_level: int,
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
    point_values: np.ndarray,
) -> RepeatedPointExecutionStats:
    """Sample point groups synchronously from completed RHE primary chunks."""
    (
        reader,
        points,
        field_ids,
        base_spacing,
        slot_capacity,
        _,
        plan,
    ) = _prepare_repeated_request(
        reader,
        points,
        field_ids,
        domain_lower,
        domain_upper,
        root_shape,
        domain_cell_counts,
        block_cell_counts,
        max_level,
        coord_to_rank,
        root_node_ids,
        node_levels,
        node_coords,
        child_node_ids,
        node_leaf_ids,
        leaf_node_ids,
        slot_capacity,
        point_values,
        trilinear=True,
        extra_metadata=(boundary_modes, normal_field_slots),
    )
    one = np.ones(3, dtype=np.int64)
    point_managed_array_bytes = (
        _plan_array_bytes(plan) + base_spacing.nbytes + one.nbytes
    )
    if point_managed_array_bytes > _INDEX_MAX:
        raise OverflowError("point managed raw-array bytes do not fit in int64")

    consumer_state = _TrilinearPointConsumerState(
        domain_lower,
        domain_upper,
        domain_cell_counts,
        block_cell_counts,
        node_levels,
        node_coords,
        leaf_node_ids,
        base_spacing,
        points,
        plan,
        point_values,
    )
    consumer = make_completed_primary_consumer(
        consumer_state,
        _consume_trilinear_point_groups,
        output_arrays=(point_values,),
    )
    rhe_stats = execute_selected_refined_halos_with_consumer(
        reader,
        consumer,
        plan.owner_leaf_ids,
        field_ids,
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
        _additional_managed_array_bytes=point_managed_array_bytes,
    )

    managed_array_bytes = point_managed_array_bytes + rhe_stats.managed_array_bytes
    if managed_array_bytes > _INDEX_MAX:
        raise OverflowError("managed raw-array bytes do not fit in int64")
    return RepeatedPointExecutionStats(
        int(points.shape[0]),
        int(plan.grouped_point_indices.shape[0]),
        int(plan.owner_leaf_ids.shape[0]),
        rhe_stats.chunk_count,
        rhe_stats.reader_call_count,
        rhe_stats.consumer_call_count,
        rhe_stats.selected_load_count,
        rhe_stats.maximum_selected_slots,
        managed_array_bytes,
    )
