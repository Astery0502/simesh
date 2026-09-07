"""CHS-001 completed refined owner-halo trilinear sampling session."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from ._point_location import (
    fill_refined_point_leaf_ids_with_hints_unchecked,
    validate_refined_point_hints_unchecked,
)
from ._refined_sampling import (
    sample_refined_trilinear_point_groups_unchecked,
    validate_refined_point_groups_unchecked,
)
from .blockio import BlockReader, _require_block_reader, read_blocks_into
from .coarser_support import CANONICAL_DIRECTIONS
from .completed_primary import (
    execute_selected_refined_halos_with_consumer,
    make_completed_primary_consumer,
)
from .foundation import INDEX_DTYPE, _require_index_triplet
from .hinted_location import fill_refined_point_leaf_ids_with_hints
from .point_location import (
    _require_point_leaf_ids,
    _require_points,
    fill_refined_point_leaf_ids,
)
from .refined_halo import (
    _allocate_workspace,
    _apply_chunk_actions_unchecked,
    _prepare_refined_halo_chunk,
    _validate_boundary_configuration,
)
from .refined_sampling import _preflight_refined_point_groups
from .repeated_sampling import (
    _make_point_plan,
    _plan_array_bytes,
    _require_point_values,
)
from .storage import _require_index_vector
from .target_boxes import fill_directed_halo_target_boxes
from .workspace import _require_nonnegative_integer


_INDEX_MAX = int(np.iinfo(np.int64).max)


class UnrepresentableRefinedSampleError(ValueError):
    """A finite owned point has no representable SAM-005 stencil."""

    def __init__(
        self,
        point_index: int,
        owner_slot: int,
        call_managed_array_bytes: int,
    ) -> None:
        self.point_index = int(point_index)
        self.owner_slot = int(owner_slot)
        self.call_managed_array_bytes = int(call_managed_array_bytes)
        super().__init__(
            "unrepresentable trilinear stencil for "
            f"point {self.point_index} at owner slot {self.owner_slot}"
        )


class CachedVectorSamplingStats(NamedTuple):
    point_count: int
    inside_point_count: int
    owner_count: int
    hint_candidate_count: int
    hint_hit_count: int
    hierarchy_fallback_count: int
    cache_lookup_count: int
    cache_hit_count: int
    cache_miss_count: int
    cache_eviction_count: int
    halo_fill_count: int
    reader_call_count: int
    selected_load_count: int
    support_load_count: int
    maximum_selected_slots: int
    logical_reader_bytes: int
    call_managed_array_bytes: int
    session_managed_array_bytes: int


@dataclass(slots=True)
class _CompletedHaloSamplingState:
    reader: BlockReader
    source_field_ids: np.ndarray
    domain_lower: np.ndarray
    domain_upper: np.ndarray
    domain_cell_counts: np.ndarray
    block_cell_counts: np.ndarray
    max_level: int
    root_shape: np.ndarray
    coord_to_rank: np.ndarray
    root_node_ids: np.ndarray
    node_levels: np.ndarray
    node_coords: np.ndarray
    child_node_ids: np.ndarray
    node_leaf_ids: np.ndarray
    leaf_node_ids: np.ndarray
    boundary_modes: np.ndarray
    normal_field_slots: np.ndarray
    rhe_slot_capacity: int
    cache_byte_budget: int
    cache_capacity: int
    cache_entry_bytes: int
    cache_payload: np.ndarray
    cache_leaf_ids: np.ndarray
    cache_recency: np.ndarray
    clock: int
    workspace: object
    workspace_arrays: tuple[np.ndarray, ...]
    zero: np.ndarray
    one: np.ndarray
    block_shape_array: np.ndarray
    padded_shape_array: np.ndarray
    interior_upper: np.ndarray
    base_spacing: np.ndarray
    miss_primary_ids: np.ndarray
    owned_arrays: tuple[np.ndarray, ...]
    borrowed_arrays: tuple[np.ndarray, ...]
    session_managed_array_bytes: int
    active: bool


@dataclass(frozen=True, slots=True)
class CompletedHaloSamplingSession:
    """Explicit serial lifecycle for completed-owner halo reuse."""

    _state: _CompletedHaloSamplingState

    @property
    def cache_capacity(self) -> int:
        return self._state.cache_capacity

    @property
    def cache_entry_bytes(self) -> int:
        return self._state.cache_entry_bytes

    @property
    def cache_payload_bytes(self) -> int:
        return int(self._state.cache_payload.nbytes)

    @property
    def session_managed_array_bytes(self) -> int:
        return self._state.session_managed_array_bytes


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


def _checked_volume(name: str, shape: tuple[int, int, int]) -> int:
    result = 1
    for extent in shape:
        if extent <= 0:
            raise ValueError(f"{name} entries must be positive")
        result = _checked_multiply(name, result, extent)
    return result


def _require_cache_byte_budget(value: int) -> int:
    if type(value) is not int:
        raise TypeError("cache_byte_budget must be an exact Python int")
    if value < 0:
        raise ValueError("cache_byte_budget must be nonnegative")
    if value > _INDEX_MAX:
        raise OverflowError("cache_byte_budget does not fit in int64")
    return value


def _unexpected_empty_consumer(*_args) -> None:
    raise RuntimeError("empty CHS conformance unexpectedly invoked its consumer")


def _require_session(
    session: CompletedHaloSamplingSession,
) -> _CompletedHaloSamplingState:
    if not isinstance(session, CompletedHaloSamplingSession):
        raise TypeError("session must be a CompletedHaloSamplingSession")
    state = session._state
    if not isinstance(state, _CompletedHaloSamplingState):
        raise TypeError("session contains invalid internal state")
    if type(state.active) is not bool:
        raise ValueError("session active flag is invalid")
    capacity = state.cache_capacity
    expected_payload = (
        capacity,
        3,
        *(int(value) for value in state.padded_shape_array),
    )
    if (
        state.cache_payload.dtype != np.dtype(np.float64)
        or state.cache_payload.shape != expected_payload
        or not state.cache_payload.flags.c_contiguous
        or not state.cache_payload.flags.writeable
    ):
        raise ValueError("session cache payload state is invalid")
    for name, value in (
        ("cache_leaf_ids", state.cache_leaf_ids),
        ("cache_recency", state.cache_recency),
    ):
        if (
            value.dtype != INDEX_DTYPE
            or value.shape != (capacity,)
            or not value.flags.c_contiguous
            or not value.flags.writeable
        ):
            raise ValueError(f"session {name} state is invalid")
    valid = state.cache_leaf_ids >= 0
    if np.any(state.cache_leaf_ids < -1) or np.any(
        state.cache_leaf_ids >= state.leaf_node_ids.shape[0]
    ):
        raise ValueError("session cache contains an invalid leaf key")
    valid_keys = state.cache_leaf_ids[valid]
    if np.unique(valid_keys).shape[0] != valid_keys.shape[0]:
        raise ValueError("session cache contains duplicate valid leaf keys")
    if np.any(state.cache_recency < 0):
        raise ValueError("session cache recency must be nonnegative")
    if state.clock < 0 or (
        valid.any() and state.clock < int(np.max(state.cache_recency[valid]))
    ):
        raise ValueError("session cache clock is inconsistent with recency")
    return state


def _static_sampling_preflight(
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
) -> np.ndarray:
    empty_points = np.empty((0, 3), dtype=np.float64)
    empty_ids = np.empty(0, dtype=np.int64)
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
        empty_points,
        empty_ids,
    )
    fill_refined_point_leaf_ids_with_hints(
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
        empty_points,
        empty_ids,
        empty_ids,
    )
    normalized = _preflight_refined_point_groups(
        domain_lower,
        domain_upper,
        domain_cell_counts,
        block_cell_counts,
        node_levels,
        node_coords,
        leaf_node_ids,
        empty_ids,
        empty_points,
        empty_ids,
        np.zeros(1, dtype=np.int64),
        trilinear=True,
    )
    return normalized[-1]


def make_completed_halo_sampling_session(
    reader: BlockReader,
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
    rhe_slot_capacity: int,
    cache_byte_budget: int,
) -> CompletedHaloSamplingSession:
    """Validate and allocate one serial completed-owner sampling session."""
    reader = _require_block_reader(reader)
    magnetic_field_ids = _require_index_vector(
        "magnetic_field_ids", magnetic_field_ids
    )
    if magnetic_field_ids.shape != (3,):
        raise ValueError("magnetic_field_ids must have shape (3,)")
    leaf_node_ids = _require_index_vector("leaf_node_ids", leaf_node_ids)
    leaf_count = int(leaf_node_ids.shape[0])
    if leaf_count == 0:
        raise ValueError("leaf_node_ids must contain at least one leaf")
    if reader.shape[0] != leaf_count:
        raise ValueError("reader block axis must equal leaf count")
    for position, field_id in enumerate(magnetic_field_ids):
        if int(field_id) < 0 or int(field_id) >= reader.shape[1]:
            raise ValueError(
                f"magnetic_field_ids entry {position} is out of range"
            )
    block_cell_counts = _require_index_triplet(
        "block_cell_counts", block_cell_counts
    )
    block_shape = tuple(int(value) for value in reader.shape[2:])
    if tuple(int(value) for value in block_cell_counts) != block_shape:
        raise ValueError("block_cell_counts must equal reader spatial extents")
    rhe_slot_capacity = _require_nonnegative_integer(
        "rhe_slot_capacity", rhe_slot_capacity
    )
    if rhe_slot_capacity > leaf_count:
        raise ValueError("rhe_slot_capacity exceeds leaf count")
    if rhe_slot_capacity < min(57, leaf_count):
        raise ValueError(
            "rhe_slot_capacity cannot fit the universal all-26 closure"
        )
    cache_byte_budget = _require_cache_byte_budget(cache_byte_budget)

    padded_shape = tuple(
        _checked_add("padded shape", value, 2) for value in block_shape
    )
    padded_volume = _checked_volume("padded shape", padded_shape)
    payload_entry_bytes = _checked_multiply(
        "cache payload entry bytes",
        _checked_multiply("cache payload entry values", 3, padded_volume),
        8,
    )
    cache_entry_bytes = _checked_add(
        "cache entry bytes", payload_entry_bytes, 16
    )
    cache_capacity = min(leaf_count, cache_byte_budget // cache_entry_bytes)

    source_field_ids = magnetic_field_ids.copy()
    validated_modes, validated_normals = _validate_boundary_configuration(
        boundary_modes, normal_field_slots, 3
    )
    copied_modes = validated_modes.copy()
    copied_normals = validated_normals.copy()
    one = np.ones(3, dtype=np.int64)
    cache_payload = np.empty(
        (cache_capacity, 3, *padded_shape), dtype=np.float64
    )
    cache_leaf_ids = np.full(cache_capacity, -1, dtype=np.int64)
    cache_recency = np.zeros(cache_capacity, dtype=np.int64)

    base_spacing = _static_sampling_preflight(
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
    )

    empty_primary_ids = np.empty(0, dtype=np.int64)
    consumer = make_completed_primary_consumer(
        None,
        _unexpected_empty_consumer,
        output_arrays=(cache_payload, cache_leaf_ids, cache_recency),
    )
    additional_bytes = sum(
        int(value.nbytes)
        for value in (
            source_field_ids,
            copied_modes,
            copied_normals,
            one,
            cache_payload,
            cache_leaf_ids,
            cache_recency,
            base_spacing,
        )
    )
    execute_selected_refined_halos_with_consumer(
        reader,
        consumer,
        empty_primary_ids,
        source_field_ids,
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
        copied_modes,
        copied_normals,
        rhe_slot_capacity,
        _additional_managed_array_bytes=additional_bytes,
    )

    workspace, workspace_arrays = _allocate_workspace(
        rhe_slot_capacity, 3, block_shape, padded_shape
    )
    zero = np.zeros(3, dtype=np.int64)
    block_shape_array = np.asarray(block_shape, dtype=np.int64)
    padded_shape_array = np.asarray(padded_shape, dtype=np.int64)
    interior_upper = one + block_shape_array
    fill_directed_halo_target_boxes(
        one,
        interior_upper,
        zero,
        padded_shape_array,
        CANONICAL_DIRECTIONS,
        workspace.target_lower,
        workspace.target_upper,
    )
    miss_primary_ids = np.empty(1, dtype=np.int64)
    for value in (
        source_field_ids,
        copied_modes,
        copied_normals,
        one,
        zero,
        block_shape_array,
        padded_shape_array,
        interior_upper,
        base_spacing,
    ):
        value.setflags(write=False)

    owned_arrays = (
        *workspace_arrays,
        zero,
        one,
        block_shape_array,
        padded_shape_array,
        interior_upper,
        base_spacing,
        miss_primary_ids,
        source_field_ids,
        copied_modes,
        copied_normals,
        cache_payload,
        cache_leaf_ids,
        cache_recency,
    )
    session_managed_array_bytes = sum(int(value.nbytes) for value in owned_arrays)
    if session_managed_array_bytes > _INDEX_MAX:
        raise OverflowError("session managed raw-array bytes do not fit in int64")
    borrowed_arrays = tuple(
        value
        for value in (
            *reader.memory_arrays,
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
        )
        if isinstance(value, np.ndarray)
    )
    state = _CompletedHaloSamplingState(
        reader,
        source_field_ids,
        domain_lower,
        domain_upper,
        domain_cell_counts,
        block_cell_counts,
        max_level,
        root_shape,
        coord_to_rank,
        root_node_ids,
        node_levels,
        node_coords,
        child_node_ids,
        node_leaf_ids,
        leaf_node_ids,
        copied_modes,
        copied_normals,
        rhe_slot_capacity,
        cache_byte_budget,
        cache_capacity,
        cache_entry_bytes,
        cache_payload,
        cache_leaf_ids,
        cache_recency,
        0,
        workspace,
        workspace_arrays,
        zero,
        one,
        block_shape_array,
        padded_shape_array,
        interior_upper,
        base_spacing,
        miss_primary_ids,
        owned_arrays,
        borrowed_arrays,
        session_managed_array_bytes,
        False,
    )
    return CompletedHaloSamplingSession(state)


def clear_completed_halo_sampling_session(
    session: CompletedHaloSamplingSession,
) -> None:
    """Invalidate all completed entries without clearing payload bytes."""
    state = _require_session(session)
    if state.active:
        raise ValueError("completed halo sampling session is already active")
    state.cache_leaf_ids.fill(-1)
    state.cache_recency.fill(0)
    state.clock = 0


def _validate_dynamic_arrays(
    state: _CompletedHaloSamplingState,
    points: np.ndarray,
    hint_leaf_ids: np.ndarray,
    point_values: np.ndarray,
    point_leaf_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = _require_points(points)
    hint_leaf_ids = _require_index_vector("hint_leaf_ids", hint_leaf_ids)
    if hint_leaf_ids.shape != (points.shape[0],):
        raise ValueError("hint_leaf_ids must have one entry per point")
    point_values = _require_point_values(
        point_values, int(points.shape[0]), 3
    )
    point_leaf_ids = _require_point_leaf_ids(
        point_leaf_ids, int(points.shape[0])
    )
    outputs = (point_values, point_leaf_ids)
    for dynamic_input in (points, hint_leaf_ids):
        if any(
            np.shares_memory(dynamic_input, value)
            for value in state.owned_arrays
        ):
            raise ValueError(
                "dynamic inputs must not overlap mutable session memory"
            )
    readonly = (
        points,
        hint_leaf_ids,
        *state.borrowed_arrays,
        *state.owned_arrays,
    )
    if np.shares_memory(point_values, point_leaf_ids):
        raise ValueError("point outputs must not overlap each other")
    for output in outputs:
        if any(np.shares_memory(output, value) for value in readonly):
            raise ValueError(
                "point outputs must not overlap inputs or session memory"
            )
    return points, hint_leaf_ids, point_values, point_leaf_ids


def _validate_and_locate(
    state: _CompletedHaloSamplingState,
    points: np.ndarray,
    hint_leaf_ids: np.ndarray,
    owner_ids: np.ndarray,
) -> tuple[int, int, int, int]:
    (
        status,
        bad_point,
        bad_axis,
        inside_count,
        _outside_count,
        hint_candidate_count,
    ) = validate_refined_point_hints_unchecked(
        state.domain_lower,
        state.domain_upper,
        state.root_shape,
        state.domain_cell_counts,
        state.block_cell_counts,
        state.max_level,
        state.node_levels,
        state.node_coords,
        state.node_leaf_ids,
        state.leaf_node_ids,
        state.base_spacing,
        points,
        hint_leaf_ids,
    )
    if status == 1:
        raise ValueError(
            "hint_leaf_ids entries must be -1 or valid leaf IDs; "
            f"first invalid entry is {bad_point}"
        )
    if status == 2:
        raise ValueError(
            f"invalid hinted refined geometry at point {bad_point}, axis {bad_axis}"
        )
    if status == 3:
        raise OverflowError(
            f"hinted refined geometry overflows at point {bad_point}, axis {bad_axis}"
        )
    if status == 4:
        raise ValueError(
            f"invalid hinted floating geometry at point {bad_point}, axis {bad_axis}"
        )
    if status == 5:
        raise ValueError(
            f"points must be finite; point {bad_point}, axis {bad_axis} is not"
        )
    if status != 0:
        raise RuntimeError(f"unexpected hinted location status {status}")
    hint_hits = int(
        fill_refined_point_leaf_ids_with_hints_unchecked(
            state.domain_lower,
            state.domain_upper,
            state.domain_cell_counts,
            state.block_cell_counts,
            state.coord_to_rank,
            state.root_node_ids,
            state.node_levels,
            state.node_coords,
            state.child_node_ids,
            state.node_leaf_ids,
            state.leaf_node_ids,
            state.base_spacing,
            points,
            hint_leaf_ids,
            owner_ids,
        )
    )
    return int(inside_count), int(hint_candidate_count), hint_hits, int(inside_count) - hint_hits


def _validate_point_plan(state, points, plan) -> None:
    status, bad_slot, bad_point = validate_refined_point_groups_unchecked(
        state.domain_lower,
        state.domain_upper,
        state.domain_cell_counts,
        state.block_cell_counts,
        state.node_levels,
        state.node_coords,
        state.leaf_node_ids,
        plan.owner_leaf_ids,
        points,
        plan.grouped_point_indices,
        plan.owner_offsets,
        state.base_spacing,
        True,
    )
    if status == 1:
        raise ValueError(f"invalid selected refined geometry at slot {bad_slot}")
    if status == 2:
        raise OverflowError(
            f"selected refined cell indices overflow at slot {bad_slot}"
        )
    if status == 3:
        raise ValueError(f"point {bad_point} is not owned by slot {bad_slot}")
    if status == 4:
        raise UnrepresentableRefinedSampleError(
            int(bad_point),
            int(bad_slot),
            _plan_array_bytes(plan),
        )
    if status != 0:
        raise RuntimeError(f"unexpected refined sampling status {status}")


def _cache_access_plan(state, owner_leaf_ids: np.ndarray, *, _indexed: bool = True):
    owner_count = int(owner_leaf_ids.shape[0])
    capacity = state.cache_capacity
    if capacity and state.clock > _INDEX_MAX - owner_count:
        raise OverflowError("cache recency clock would overflow int64")
    simulated_keys = state.cache_leaf_ids.copy()
    simulated_recency = state.cache_recency.copy()
    # Amortize the index only across larger owner batches; no state escapes.
    lookup = (
        {int(key): slot for slot, key in enumerate(simulated_keys) if key >= 0}
        if _indexed and capacity >= 16 and owner_count >= 8 else None
    )
    planned_slots = np.full(owner_count, -1, dtype=np.int64)
    planned_hits = np.zeros(owner_count, dtype=np.uint8)
    planned_selected_counts = np.zeros(owner_count, dtype=np.int64)
    simulated_clock = state.clock
    workspace = state.workspace
    for position, owner_value in enumerate(owner_leaf_ids):
        owner = int(owner_value)
        slot = -1
        if lookup is not None:
            slot = lookup.get(owner, -1)
        else:
            for candidate in range(capacity):
                if int(simulated_keys[candidate]) == owner:
                    slot = candidate
                    break
        if slot >= 0:
            planned_hits[position] = 1
        else:
            state.miss_primary_ids[0] = owner
            primary_count, selected_count = _prepare_refined_halo_chunk(
                workspace,
                state.miss_primary_ids,
                state.root_shape,
                state.coord_to_rank,
                state.root_node_ids,
                state.node_levels,
                state.node_coords,
                state.child_node_ids,
                state.node_leaf_ids,
                state.leaf_node_ids,
                state.one,
                state.interior_upper,
                state.boundary_modes,
                state.normal_field_slots,
                validate_actions=True,
            )
            if primary_count != 1:
                raise RuntimeError("one-owner RHE preflight made no progress")
            planned_selected_counts[position] = selected_count
            if capacity:
                for candidate in range(capacity):
                    if int(simulated_keys[candidate]) < 0:
                        slot = candidate
                        break
                if slot < 0:
                    slot = min(
                        range(capacity),
                        key=lambda value: (
                            int(simulated_recency[value]), value
                        ),
                    )
                if lookup is not None:
                    previous = int(simulated_keys[slot])
                    if previous >= 0:
                        del lookup[previous]
                    lookup[owner] = slot
                simulated_keys[slot] = owner
        planned_slots[position] = slot
        if capacity:
            simulated_clock += 1
            simulated_recency[slot] = simulated_clock
    arrays = (
        simulated_keys,
        simulated_recency,
        planned_slots,
        planned_hits,
        planned_selected_counts,
    )
    return arrays


def _sample_owner_group(
    state: _CompletedHaloSamplingState,
    payload: np.ndarray,
    owner_position: int,
    points: np.ndarray,
    plan,
    point_values: np.ndarray,
) -> None:
    sample_refined_trilinear_point_groups_unchecked(
        payload,
        state.one,
        state.miss_primary_ids,
        state.domain_lower,
        state.domain_upper,
        state.domain_cell_counts,
        state.block_cell_counts,
        state.node_levels,
        state.node_coords,
        state.leaf_node_ids,
        state.base_spacing,
        points,
        plan.grouped_point_indices,
        plan.owner_offsets[owner_position : owner_position + 2],
        point_values,
    )


def _complete_owner(state: _CompletedHaloSamplingState) -> int:
    primary_count, selected_count = _prepare_refined_halo_chunk(
        state.workspace,
        state.miss_primary_ids,
        state.root_shape,
        state.coord_to_rank,
        state.root_node_ids,
        state.node_levels,
        state.node_coords,
        state.child_node_ids,
        state.node_leaf_ids,
        state.leaf_node_ids,
        state.one,
        state.interior_upper,
        state.boundary_modes,
        state.normal_field_slots,
        validate_actions=True,
    )
    if primary_count != 1:
        raise RuntimeError("one-owner RHE execution made no progress")
    read_blocks_into(
        state.reader,
        state.zero,
        state.block_shape_array,
        state.workspace.selected_leaf_ids[:selected_count],
        state.source_field_ids,
        state.workspace.payload[:selected_count],
        state.one,
    )
    _apply_chunk_actions_unchecked(
        state.workspace,
        1,
        selected_count,
        state.one,
        state.interior_upper,
        state.boundary_modes,
        state.normal_field_slots,
    )
    return selected_count


def _sample_refined_trilinear_vectors_cached(
    state: _CompletedHaloSamplingState,
    points: np.ndarray,
    hint_leaf_ids: np.ndarray,
    point_values: np.ndarray,
    point_leaf_ids: np.ndarray,
) -> CachedVectorSamplingStats:
    points, hint_leaf_ids, point_values, point_leaf_ids = _validate_dynamic_arrays(
        state, points, hint_leaf_ids, point_values, point_leaf_ids
    )
    owner_scratch = np.empty(points.shape[0], dtype=np.int64)
    (
        inside_count,
        hint_candidate_count,
        hint_hit_count,
        hierarchy_fallback_count,
    ) = _validate_and_locate(state, points, hint_leaf_ids, owner_scratch)
    plan = _make_point_plan(owner_scratch)
    _validate_point_plan(state, points, plan)
    cache_plan_arrays = _cache_access_plan(state, plan.owner_leaf_ids)
    (
        _simulated_keys,
        _simulated_recency,
        planned_slots,
        planned_hits,
        planned_selected_counts,
    ) = cache_plan_arrays

    selected_load_count = 0
    for value in planned_selected_counts:
        selected_load_count = _checked_add(
            "selected load count", selected_load_count, int(value)
        )
    owner_count = int(plan.owner_leaf_ids.shape[0])
    cache_hit_count = int(np.count_nonzero(planned_hits))
    cache_miss_count = owner_count - cache_hit_count
    block_volume = _checked_volume(
        "block shape", tuple(int(value) for value in state.block_shape_array)
    )
    logical_values = _checked_multiply(
        "logical reader values", selected_load_count, 3
    )
    logical_values = _checked_multiply(
        "logical reader values", logical_values, block_volume
    )
    logical_reader_bytes = _checked_multiply(
        "logical reader bytes", logical_values, 8
    )
    call_managed_array_bytes = _plan_array_bytes(plan) + sum(
        int(value.nbytes) for value in cache_plan_arrays
    )
    if call_managed_array_bytes > _INDEX_MAX:
        raise OverflowError("call managed raw-array bytes do not fit in int64")

    np.copyto(point_leaf_ids, owner_scratch)
    cache_eviction_count = 0
    maximum_selected_slots = 0
    for owner_position, owner_value in enumerate(plan.owner_leaf_ids):
        owner = int(owner_value)
        state.miss_primary_ids[0] = owner
        slot = int(planned_slots[owner_position])
        if int(planned_hits[owner_position]):
            if slot < 0 or int(state.cache_leaf_ids[slot]) != owner:
                raise RuntimeError("cache access plan diverged before a hit")
            _sample_owner_group(
                state,
                state.cache_payload[slot : slot + 1],
                owner_position,
                points,
                plan,
                point_values,
            )
            state.clock += 1
            state.cache_recency[slot] = state.clock
            continue

        selected_count = _complete_owner(state)
        if selected_count != int(planned_selected_counts[owner_position]):
            raise RuntimeError("RHE selected count changed after preflight")
        maximum_selected_slots = max(maximum_selected_slots, selected_count)
        _sample_owner_group(
            state,
            state.workspace.payload[:1],
            owner_position,
            points,
            plan,
            point_values,
        )
        if state.cache_capacity:
            if slot < 0:
                raise RuntimeError("cache miss has no planned victim")
            evicted = int(state.cache_leaf_ids[slot]) >= 0
            state.cache_leaf_ids[slot] = -1
            np.copyto(state.cache_payload[slot], state.workspace.payload[0])
            state.cache_leaf_ids[slot] = owner
            state.clock += 1
            state.cache_recency[slot] = state.clock
            if evicted:
                cache_eviction_count += 1

    return CachedVectorSamplingStats(
        int(points.shape[0]),
        inside_count,
        owner_count,
        hint_candidate_count,
        hint_hit_count,
        hierarchy_fallback_count,
        owner_count,
        cache_hit_count,
        cache_miss_count,
        cache_eviction_count,
        cache_miss_count,
        cache_miss_count,
        selected_load_count,
        selected_load_count - cache_miss_count,
        maximum_selected_slots,
        logical_reader_bytes,
        call_managed_array_bytes,
        state.session_managed_array_bytes,
    )


def sample_refined_trilinear_vectors_cached(
    session: CompletedHaloSamplingSession,
    points: np.ndarray,
    hint_leaf_ids: np.ndarray,
    point_values: np.ndarray,
    point_leaf_ids: np.ndarray,
) -> CachedVectorSamplingStats:
    """Sample one ordered point batch through a completed-owner halo LRU."""
    state = _require_session(session)
    if state.active:
        raise ValueError("completed halo sampling session is already active")
    state.active = True
    try:
        return _sample_refined_trilinear_vectors_cached(
            state,
            points,
            hint_leaf_ids,
            point_values,
            point_leaf_ids,
        )
    finally:
        state.active = False
