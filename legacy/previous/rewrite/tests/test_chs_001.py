from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

import numpy as np
import pytest

import simesh_rewrite.completed_halo_sampling as chs_module
from simesh_rewrite.amrvac_dat import bind_amrvac_v5_forest, read_amrvac_v5_index
from simesh_rewrite.amrvac_dat_reader import make_amrvac_v5_block_reader
from simesh_rewrite.balance import validate_refined_all_touch_2to1
from simesh_rewrite.blockio import array_block_reader, make_block_reader
from simesh_rewrite.coarser_support import CANONICAL_DIRECTIONS
from simesh_rewrite.completed_halo_sampling import (
    CachedVectorSamplingStats,
    clear_completed_halo_sampling_session,
    make_completed_halo_sampling_session,
    sample_refined_trilinear_vectors_cached,
)
from simesh_rewrite.forest import refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.point_location import refined_point_leaf_ids
from simesh_rewrite.refined_geometry import refined_leaf_geometry
from simesh_rewrite.relations import balanced_refined_relations
from simesh_rewrite.repeated_sampling import (
    execute_refined_trilinear_points_from_blocks,
)
from simesh_rewrite.storage import gather_blocks_into


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


@dataclass(frozen=True)
class Artifact:
    root_shape: np.ndarray
    coord_to_rank: np.ndarray
    root_node_ids: np.ndarray
    node_levels: np.ndarray
    node_coords: np.ndarray
    child_node_ids: np.ndarray
    node_leaf_ids: np.ndarray
    leaf_node_ids: np.ndarray
    max_level: int
    lower: np.ndarray
    upper: np.ndarray
    domain_counts: np.ndarray
    block_counts: np.ndarray
    backing: np.ndarray


def make_artifact() -> Artifact:
    root_shape = i3(2, 1, 1)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    flags = np.asarray([False, *([True] * 8), True], dtype=np.bool_)
    forest = refined_forest(root_shape, coord_to_rank, rank_to_coord, flags)
    max_level = validate_refined_forest_arrays(
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
    validate_refined_all_touch_2to1(
        root_shape,
        coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
    )
    block_counts = i3(4, 4, 4)
    lower = np.asarray((0.0, 0.0, 0.0), dtype=np.float64)
    upper = root_shape.astype(np.float64)
    domain_counts = root_shape * block_counts
    leaf_count = int(forest.leaf_node_ids.size)
    field_count = 4
    x, y, z = np.indices((4, 4, 4), dtype=np.float64)
    backing = np.empty((leaf_count, field_count, 4, 4, 4), dtype=np.float64)
    for leaf in range(leaf_count):
        for field_position in range(field_count):
            backing[leaf, field_position] = (
                100000.0 * leaf
                + 10000.0 * field_position
                + 100.0 * x
                + 10.0 * y
                + z
                + 1.0
            )
    return Artifact(
        root_shape,
        coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
        max_level,
        lower,
        upper,
        domain_counts,
        block_counts,
        backing,
    )


@dataclass
class ReaderState:
    backing: np.ndarray
    calls: list[np.ndarray] = field(default_factory=list)
    field_calls: list[np.ndarray] = field(default_factory=list)
    nonempty_calls: int = 0
    bytes_read: int = 0
    fail_nonempty_call: int | None = None
    fail_empty: bool = False

    def reset(self, *, keep_failure: bool = False) -> None:
        self.calls.clear()
        self.field_calls.clear()
        self.nonempty_calls = 0
        self.bytes_read = 0
        if not keep_failure:
            self.fail_nonempty_call = None


def counting_read(
    state: ReaderState,
    source_lower: np.ndarray,
    source_upper: np.ndarray,
    block_ids: np.ndarray,
    field_ids: np.ndarray,
    destination: np.ndarray,
    destination_lower: np.ndarray,
) -> None:
    state.calls.append(block_ids.copy())
    state.field_calls.append(field_ids.copy())
    if not block_ids.size and state.fail_empty:
        raise OSError("injected empty reader failure")
    if block_ids.size:
        state.nonempty_calls += 1
        if state.nonempty_calls == state.fail_nonempty_call:
            raise OSError("injected completed-halo read failure")
        extent = source_upper - source_lower
        state.bytes_read += (
            int(block_ids.size)
            * int(field_ids.size)
            * int(np.prod(extent, dtype=np.int64))
            * 8
        )
    gather_blocks_into(
        state.backing,
        source_lower,
        source_upper,
        block_ids,
        field_ids,
        destination,
        destination_lower,
    )


def counting_reader(state: ReaderState):
    return make_block_reader(
        state,
        state.backing.shape,
        counting_read,
        memory_arrays=(state.backing,),
    )


@dataclass
class ReentrantReaderState:
    backing: np.ndarray
    session: object | None = None
    nested_errors: list[tuple[str, Exception]] = field(default_factory=list)
    attempted: bool = False


def reentrant_read(
    state: ReentrantReaderState,
    source_lower: np.ndarray,
    source_upper: np.ndarray,
    block_ids: np.ndarray,
    field_ids: np.ndarray,
    destination: np.ndarray,
    destination_lower: np.ndarray,
) -> None:
    if block_ids.size and state.session is not None and not state.attempted:
        state.attempted = True
        empty_points = np.empty((0, 3), dtype=np.float64)
        try:
            sample_refined_trilinear_vectors_cached(
                state.session,
                empty_points,
                np.empty(0, dtype=np.int64),
                np.empty((0, 3), dtype=np.float64),
                np.empty(0, dtype=np.int64),
            )
        except Exception as error:  # assertions below freeze the public type/message
            state.nested_errors.append(("sample", error))
        try:
            clear_completed_halo_sampling_session(state.session)
        except Exception as error:  # assertions below freeze the public type/message
            state.nested_errors.append(("clear", error))
    gather_blocks_into(
        state.backing,
        source_lower,
        source_upper,
        block_ids,
        field_ids,
        destination,
        destination_lower,
    )


def reentrant_reader(state: ReentrantReaderState):
    return make_block_reader(
        state,
        state.backing.shape,
        reentrant_read,
        memory_arrays=(state.backing,),
    )


def boundary_configuration() -> tuple[np.ndarray, np.ndarray]:
    modes = np.zeros((3, 6), dtype=np.uint8)
    modes[0, 0] = 2
    modes[1, 2] = 3
    modes[2, 4] = 1
    return modes, i3(-1, 1, -1)


def entry_bytes(artifact: Artifact) -> int:
    return 3 * int(np.prod(artifact.block_counts + 2, dtype=np.int64)) * 8 + 16


def make_session(
    artifact: Artifact,
    reader,
    cache_capacity: int,
    *,
    fields: np.ndarray | None = None,
):
    modes, normals = boundary_configuration()
    if fields is None:
        fields = i3(2, 0, 2)
    session = make_completed_halo_sampling_session(
        reader,
        fields,
        artifact.lower,
        artifact.upper,
        artifact.domain_counts,
        artifact.block_counts,
        artifact.max_level,
        artifact.root_shape,
        artifact.coord_to_rank,
        artifact.root_node_ids,
        artifact.node_levels,
        artifact.node_coords,
        artifact.child_node_ids,
        artifact.node_leaf_ids,
        artifact.leaf_node_ids,
        modes,
        normals,
        len(artifact.leaf_node_ids),
        cache_capacity * entry_bytes(artifact),
    )
    assert session.cache_capacity == cache_capacity
    assert session.cache_entry_bytes == entry_bytes(artifact)
    assert session.cache_payload_bytes == (
        cache_capacity * (entry_bytes(artifact) - 16)
    )
    return session


def factory_arguments(
    artifact: Artifact,
    reader,
    *,
    fields: np.ndarray | None = None,
    modes: np.ndarray | None = None,
    normals: np.ndarray | None = None,
    rhe_capacity: int | None = None,
    cache_budget: int | None = None,
) -> list[object]:
    default_modes, default_normals = boundary_configuration()
    return [
        reader,
        i3(2, 0, 2) if fields is None else fields,
        artifact.lower,
        artifact.upper,
        artifact.domain_counts,
        artifact.block_counts,
        artifact.max_level,
        artifact.root_shape,
        artifact.coord_to_rank,
        artifact.root_node_ids,
        artifact.node_levels,
        artifact.node_coords,
        artifact.child_node_ids,
        artifact.node_leaf_ids,
        artifact.leaf_node_ids,
        default_modes if modes is None else modes,
        default_normals if normals is None else normals,
        len(artifact.leaf_node_ids) if rhe_capacity is None else rhe_capacity,
        entry_bytes(artifact) if cache_budget is None else cache_budget,
    ]


def points_for_leaves(artifact: Artifact, leaf_ids: list[int]) -> np.ndarray:
    selected = np.asarray(leaf_ids, dtype=np.int64)
    bounds, spacing = refined_leaf_geometry(
        artifact.lower,
        artifact.upper,
        artifact.root_shape,
        artifact.domain_counts,
        artifact.block_counts,
        artifact.node_levels,
        artifact.node_coords,
        artifact.leaf_node_ids,
        selected,
    )
    fractions = np.asarray(
        ((0.25, 1.25, 2.25), (2.25, 0.25, 1.25), (1.25, 2.25, 0.25)),
        dtype=np.float64,
    )
    result = np.empty((selected.size, 3), dtype=np.float64)
    for row in range(selected.size):
        result[row] = bounds[row, 0] + fractions[row % 3] * spacing[row]
    return result


def public_rps(
    artifact: Artifact,
    points: np.ndarray,
    fields: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    modes, normals = boundary_configuration()
    values = np.full((points.shape[0], 3), -919.0)
    owners = refined_point_leaf_ids(
        artifact.lower,
        artifact.upper,
        artifact.root_shape,
        artifact.domain_counts,
        artifact.block_counts,
        artifact.max_level,
        artifact.coord_to_rank,
        artifact.root_node_ids,
        artifact.child_node_ids,
        artifact.node_leaf_ids,
        points,
    )
    execute_refined_trilinear_points_from_blocks(
        array_block_reader(artifact.backing),
        points,
        fields,
        artifact.lower,
        artifact.upper,
        artifact.root_shape,
        artifact.domain_counts,
        artifact.block_counts,
        artifact.max_level,
        artifact.coord_to_rank,
        artifact.root_node_ids,
        artifact.node_levels,
        artifact.node_coords,
        artifact.child_node_ids,
        artifact.node_leaf_ids,
        artifact.leaf_node_ids,
        modes,
        normals,
        len(artifact.leaf_node_ids),
        values,
    )
    return values, owners


def call_bytes(point_count: int, inside_count: int, owner_count: int, capacity: int) -> int:
    plan_bytes = 8 * (point_count + inside_count + owner_count + owner_count + 1)
    cache_plan_bytes = 16 * capacity + 17 * owner_count
    return plan_bytes + cache_plan_bytes


def expected_session_bytes(
    block_counts: np.ndarray,
    rhe_capacity: int,
    cache_capacity: int,
) -> int:
    field_count = 3
    block_shape = tuple(int(value) for value in block_counts)
    padded_shape = tuple(value + 2 for value in block_shape)
    padded_volume = int(np.prod(padded_shape, dtype=np.int64))
    coarse_volume = int(
        np.prod(tuple(value + 1 for value in block_shape), dtype=np.int64)
    )
    rhe_managed = (
        rhe_capacity * (8 * field_count * padded_volume + 1854)
        + 8 * field_count * coarse_volume
        + 11778
        + 8 * field_count
    )
    # RHE's five executor arrays are not part of _allocate_workspace.
    rhe_executor_arrays = 4 * 3 * 8 + field_count * 8
    chs_fixed_arrays = (
        6 * 3 * 8  # zero/one/block/padded/interior/base-spacing triplets
        + 8  # one miss-primary leaf ID
        + field_count * 8  # copied source fields
        + field_count * 6  # copied uint8 boundary modes
        + 3 * 8  # copied normal slots
    )
    return (
        rhe_managed
        - rhe_executor_arrays
        + chs_fixed_arrays
        + cache_capacity * (3 * padded_volume * 8 + 16)
    )


def assert_exact_stats_and_io(
    stats: CachedVectorSamplingStats,
    state: ReaderState,
    session,
    *,
    point_count: int,
    inside_count: int,
    owner_count: int,
    hint_candidates: int,
    hint_hits: int,
    cache_hits: int,
    cache_misses: int,
    evictions: int,
) -> None:
    assert stats.point_count == point_count
    assert stats.inside_point_count == inside_count
    assert stats.owner_count == owner_count
    assert stats.hint_candidate_count == hint_candidates
    assert stats.hint_hit_count == hint_hits
    assert stats.hierarchy_fallback_count == inside_count - hint_hits
    assert stats.cache_lookup_count == owner_count
    assert stats.cache_hit_count == cache_hits
    assert stats.cache_miss_count == cache_misses
    assert stats.cache_eviction_count == evictions
    assert stats.halo_fill_count == cache_misses
    assert stats.reader_call_count == cache_misses == state.nonempty_calls
    selected_from_calls = sum(int(call.size) for call in state.calls if call.size)
    assert stats.selected_load_count == selected_from_calls
    assert stats.support_load_count == selected_from_calls - cache_misses
    assert stats.maximum_selected_slots == max(
        (int(call.size) for call in state.calls if call.size), default=0
    )
    assert stats.logical_reader_bytes == selected_from_calls * 3 * 4**3 * 8
    assert state.bytes_read == stats.logical_reader_bytes
    assert stats.call_managed_array_bytes == call_bytes(
        point_count, inside_count, owner_count, session.cache_capacity
    )
    assert stats.session_managed_array_bytes == session.session_managed_array_bytes
    assert stats.session_managed_array_bytes == expected_session_bytes(
        session._state.block_cell_counts,
        session._state.rhe_slot_capacity,
        session.cache_capacity,
    )


def test_factory_empty_call_and_empty_exterior_interior_semantics() -> None:
    artifact = make_artifact()
    state = ReaderState(artifact.backing)
    session = make_session(artifact, counting_reader(state), 1)
    assert len(state.calls) == 1
    assert state.calls[0].shape == (0,)
    assert state.nonempty_calls == 0

    empty_points = np.empty((0, 3), dtype=np.float64)
    empty_values = np.empty((0, 3), dtype=np.float64)
    empty_owners = np.empty(0, dtype=np.int64)
    stats = sample_refined_trilinear_vectors_cached(
        session, empty_points, np.empty(0, dtype=np.int64), empty_values, empty_owners
    )
    assert_exact_stats_and_io(
        stats,
        state,
        session,
        point_count=0,
        inside_count=0,
        owner_count=0,
        hint_candidates=0,
        hint_hits=0,
        cache_hits=0,
        cache_misses=0,
        evictions=0,
    )
    assert len(state.calls) == 1

    exterior = np.asarray(
        ((-0.1, 0.5, 0.5), (2.0, 0.5, 0.5)), dtype=np.float64
    )
    hints = i3(0, 8)
    values = np.asarray(((np.nan, -0.0, np.inf), (7.0, 8.0, 9.0)))
    before = values.copy()
    owners = np.full(2, -77, dtype=np.int64)
    stats = sample_refined_trilinear_vectors_cached(
        session, exterior, hints, values, owners
    )
    assert owners.tolist() == [-1, -1]
    assert np.array_equal(values.view(np.uint64), before.view(np.uint64))
    assert_exact_stats_and_io(
        stats,
        state,
        session,
        point_count=2,
        inside_count=0,
        owner_count=0,
        hint_candidates=2,
        hint_hits=0,
        cache_hits=0,
        cache_misses=0,
        evictions=0,
    )


@pytest.mark.parametrize(
    ("failure", "error", "match"),
    [
        ("reader_type", TypeError, "BlockReader"),
        ("field_type", TypeError, "int64"),
        ("field_shape", ValueError, "shape"),
        ("field_range", ValueError, "out of range"),
        ("boundary_shape", ValueError, "shape"),
        ("boundary_code", ValueError, "unknown mode"),
        ("normal_range", ValueError, "field positions"),
        ("capacity_low", ValueError, "universal all-26"),
        ("capacity_high", ValueError, "exceeds leaf count"),
        ("capacity_type", TypeError, "integer"),
        ("budget_low", ValueError, "nonnegative"),
        ("budget_type", TypeError, "exact Python int"),
        ("budget_overflow", OverflowError, "int64"),
    ],
)
def test_factory_rejects_invalid_boundaries_before_reader_call(
    failure: str,
    error: type[Exception],
    match: str,
) -> None:
    artifact = make_artifact()
    state = ReaderState(artifact.backing)
    arguments = factory_arguments(artifact, counting_reader(state))
    if failure == "reader_type":
        arguments[0] = object()
    elif failure == "field_type":
        arguments[1] = np.asarray((2, 0, 2), dtype=np.int32)
    elif failure == "field_shape":
        arguments[1] = i3(2, 0)
    elif failure == "field_range":
        arguments[1] = i3(2, 0, artifact.backing.shape[1])
    elif failure == "boundary_shape":
        arguments[15] = np.zeros((3, 5), dtype=np.uint8)
    elif failure == "boundary_code":
        modes = np.zeros((3, 6), dtype=np.uint8)
        modes[1, 3] = 4
        arguments[15] = modes
    elif failure == "normal_range":
        arguments[16] = i3(-1, 3, -1)
    elif failure == "capacity_low":
        arguments[17] = len(artifact.leaf_node_ids) - 1
    elif failure == "capacity_high":
        arguments[17] = len(artifact.leaf_node_ids) + 1
    elif failure == "capacity_type":
        arguments[17] = True
    elif failure == "budget_low":
        arguments[18] = -1
    elif failure == "budget_type":
        arguments[18] = True
    else:
        arguments[18] = int(np.iinfo(np.int64).max) + 1

    with pytest.raises(error, match=match):
        make_completed_halo_sampling_session(*arguments)
    assert state.calls == []


def test_factory_propagates_its_single_empty_reader_failure() -> None:
    artifact = make_artifact()
    state = ReaderState(artifact.backing, fail_empty=True)
    with pytest.raises(OSError, match="empty reader failure"):
        make_completed_halo_sampling_session(
            *factory_arguments(artifact, counting_reader(state))
        )
    assert len(state.calls) == 1
    assert state.calls[0].shape == (0,)
    assert state.nonempty_calls == 0


def test_factory_copies_field_and_physical_boundary_provenance() -> None:
    artifact = make_artifact()
    state = ReaderState(artifact.backing)
    fields = i3(2, 0, 2)
    modes, normals = boundary_configuration()
    expected_fields = fields.copy()
    expected_modes = modes.copy()
    expected_normals = normals.copy()
    session = make_completed_halo_sampling_session(
        *factory_arguments(
            artifact,
            counting_reader(state),
            fields=fields,
            modes=modes,
            normals=normals,
        )
    )
    assert not np.shares_memory(session._state.source_field_ids, fields)
    assert not np.shares_memory(session._state.boundary_modes, modes)
    assert not np.shares_memory(session._state.normal_field_slots, normals)

    fields[:] = 3
    modes[:] = 0
    normals[:] = -1
    assert np.array_equal(session._state.source_field_ids, expected_fields)
    assert np.array_equal(session._state.boundary_modes, expected_modes)
    assert np.array_equal(session._state.normal_field_slots, expected_normals)

    point = points_for_leaves(artifact, [0])
    expected_values, expected_owners = public_rps(
        artifact, point, expected_fields
    )
    state.reset()
    values = np.full((1, 3), -1511.0)
    owners = np.full(1, -1512, dtype=np.int64)
    sample_refined_trilinear_vectors_cached(
        session, point, i3(-1), values, owners
    )
    assert np.array_equal(values.view(np.uint64), expected_values.view(np.uint64))
    assert np.array_equal(owners, expected_owners)
    assert state.field_calls
    assert all(
        np.array_equal(call, expected_fields) for call in state.field_calls
    )


def test_coherent_same_owner_run_uses_mixed_hint_quality_and_one_cache_lookup() -> None:
    artifact = make_artifact()
    point = points_for_leaves(artifact, [8])[0]
    points = np.ascontiguousarray(np.repeat(point[None, :], 6, axis=0))
    hints = i3(8, 8, -1, 0, -1, 8)
    expected_values, expected_owners = public_rps(
        artifact, points, i3(2, 0, 2)
    )
    state = ReaderState(artifact.backing)
    session = make_session(artifact, counting_reader(state), 1)
    state.reset()
    values = np.full((6, 3), -727.0)
    owners = np.full(6, -728, dtype=np.int64)
    stats = sample_refined_trilinear_vectors_cached(
        session, points, hints, values, owners
    )
    assert np.array_equal(owners, expected_owners)
    assert np.array_equal(values.view(np.uint64), expected_values.view(np.uint64))
    assert_exact_stats_and_io(
        stats,
        state,
        session,
        point_count=6,
        inside_count=6,
        owner_count=1,
        hint_candidates=4,
        hint_hits=3,
        cache_hits=0,
        cache_misses=1,
        evictions=0,
    )


@pytest.mark.parametrize("capacity", [0, 1, 3, 9])
def test_cache_capacities_hints_fields_and_public_rps_bits(capacity: int) -> None:
    artifact = make_artifact()
    fields = i3(2, 0, 2)
    base = points_for_leaves(artifact, [8, 0, 1])
    points = np.ascontiguousarray(
        [base[0], base[1], base[0], base[2], base[1]], dtype=np.float64
    )
    hints = i3(8, -1, 0, 8, 0)  # correct, absent, wrong, wrong, correct
    expected_values, expected_owners = public_rps(artifact, points, fields)
    state = ReaderState(artifact.backing)
    session = make_session(artifact, counting_reader(state), capacity, fields=fields)
    state.reset()
    values = np.full((points.shape[0], 3), -919.0)
    owners = np.full(points.shape[0], -5, dtype=np.int64)
    stats = sample_refined_trilinear_vectors_cached(
        session, points, hints, values, owners
    )
    assert np.array_equal(owners, expected_owners)
    assert np.array_equal(values.view(np.uint64), expected_values.view(np.uint64))
    assert set(int(value) for value in owners) == {0, 1, 8}
    assert [int(call[0]) for call in state.calls] == [0, 1, 8]
    assert_exact_stats_and_io(
        stats,
        state,
        session,
        point_count=5,
        inside_count=5,
        owner_count=3,
        hint_candidates=4,
        hint_hits=2,
        cache_hits=0,
        cache_misses=3,
        evictions=max(0, 3 - capacity) if capacity else 0,
    )

    state.reset()
    second = np.full_like(values, -919.0)
    second_owners = np.full_like(owners, -5)
    second_stats = sample_refined_trilinear_vectors_cached(
        session, points, expected_owners.copy(), second, second_owners
    )
    expected_hits = 3 if capacity >= 3 else 0
    expected_misses = 3 - expected_hits
    assert np.array_equal(second_owners, expected_owners)
    assert np.array_equal(second.view(np.uint64), expected_values.view(np.uint64))
    assert_exact_stats_and_io(
        second_stats,
        state,
        session,
        point_count=5,
        inside_count=5,
        owner_count=3,
        hint_candidates=5,
        hint_hits=5,
        cache_hits=expected_hits,
        cache_misses=expected_misses,
        evictions=expected_misses if capacity == 1 else 0,
    )


def test_mixed_relations_boundary_modes_lru_eviction_and_clear() -> None:
    artifact = make_artifact()
    relation_ids = i3(8, 0, 1)
    kinds, masks, _, _ = balanced_refined_relations(
        artifact.root_shape,
        artifact.coord_to_rank,
        artifact.root_node_ids,
        artifact.node_levels,
        artifact.node_coords,
        artifact.child_node_ids,
        artifact.node_leaf_ids,
        artifact.leaf_node_ids,
        relation_ids,
        CANONICAL_DIRECTIONS,
    )
    assert set(int(value) for value in np.unique(kinds)) == {1, 2, 3, 4}
    assert np.any(masks != 0)
    modes, _ = boundary_configuration()
    assert set(int(value) for value in np.unique(modes)) == {0, 1, 2, 3}

    state = ReaderState(artifact.backing)
    session = make_session(artifact, counting_reader(state), 2)
    state.reset()
    first_points = points_for_leaves(artifact, [1, 0])
    first_values = np.empty((2, 3), dtype=np.float64)
    first_owners = np.empty(2, dtype=np.int64)
    first_stats = sample_refined_trilinear_vectors_cached(
        session, first_points, i3(-1, -1), first_values, first_owners
    )
    assert first_owners.tolist() == [1, 0]
    assert [int(call[0]) for call in state.calls] == [0, 1]
    assert session._state.cache_leaf_ids.tolist() == [0, 1]
    assert session._state.cache_recency.tolist() == [1, 2]
    assert first_stats.cache_eviction_count == 0

    point0 = points_for_leaves(artifact, [0])
    sample_refined_trilinear_vectors_cached(
        session,
        point0,
        i3(0),
        np.empty((1, 3), dtype=np.float64),
        np.empty(1, dtype=np.int64),
    )
    assert session._state.cache_recency.tolist() == [3, 2]

    point8 = points_for_leaves(artifact, [8])
    miss = sample_refined_trilinear_vectors_cached(
        session,
        point8,
        i3(-1),
        np.empty((1, 3), dtype=np.float64),
        np.empty(1, dtype=np.int64),
    )
    assert miss.cache_eviction_count == 1
    assert session._state.cache_leaf_ids.tolist() == [0, 8]
    assert session._state.cache_recency.tolist() == [3, 4]

    payload_before = session._state.cache_payload.copy()
    calls_before = len(state.calls)
    clear_completed_halo_sampling_session(session)
    assert session._state.cache_leaf_ids.tolist() == [-1, -1]
    assert session._state.cache_recency.tolist() == [0, 0]
    assert session._state.clock == 0
    assert np.array_equal(
        session._state.cache_payload.view(np.uint64), payload_before.view(np.uint64)
    )
    assert len(state.calls) == calls_before


@pytest.mark.parametrize(
    ("failure", "error", "match"),
    [
        ("nonfinite", ValueError, "finite"),
        ("bad_hint", ValueError, "hint_leaf_ids"),
        ("bad_values", ValueError, "point_values"),
        ("bad_owner_dtype", TypeError, "int64"),
    ],
)
def test_dynamic_ordinary_failure_is_atomic_and_has_no_io(
    failure: str,
    error: type[Exception],
    match: str,
) -> None:
    artifact = make_artifact()
    state = ReaderState(artifact.backing)
    session = make_session(artifact, counting_reader(state), 2)
    seed = points_for_leaves(artifact, [8])
    sample_refined_trilinear_vectors_cached(
        session,
        seed,
        i3(8),
        np.empty((1, 3), dtype=np.float64),
        np.empty(1, dtype=np.int64),
    )
    state.reset()
    keys_before = session._state.cache_leaf_ids.copy()
    recency_before = session._state.cache_recency.copy()
    clock_before = session._state.clock
    points = points_for_leaves(artifact, [0, 1])
    hints = i3(-1, -1)
    values = np.full((2, 3), -771.0)
    owners = np.full(2, -772, dtype=np.int64)
    if failure == "nonfinite":
        points = points.copy()
        points[1, 2] = np.nan
    elif failure == "bad_hint":
        hints = i3(-1, len(artifact.leaf_node_ids))
    elif failure == "bad_values":
        values = np.full((2, 2), -771.0)
    else:
        owners = owners.astype(np.int32)
    values_before = values.copy()
    owners_before = owners.copy()

    with pytest.raises(error, match=match):
        sample_refined_trilinear_vectors_cached(
            session, points, hints, values, owners
        )
    assert state.calls == []
    assert np.array_equal(values.view(np.uint64), values_before.view(np.uint64))
    assert np.array_equal(owners, owners_before)
    assert np.array_equal(session._state.cache_leaf_ids, keys_before)
    assert np.array_equal(session._state.cache_recency, recency_before)
    assert session._state.clock == clock_before


@pytest.mark.parametrize(
    "alias_kind",
    ["workspace_points", "miss_ids", "cache_keys", "recency", "cache_payload"],
)
def test_dynamic_inputs_cannot_alias_mutable_persistent_session_memory(
    alias_kind: str,
) -> None:
    artifact = make_artifact()
    state = ReaderState(artifact.backing)
    session = make_session(artifact, counting_reader(state), 1)
    ordinary_point = points_for_leaves(artifact, [0])
    points = ordinary_point.copy()
    hints = i3(-1)
    if alias_kind == "workspace_points":
        points = session._state.workspace.payload.reshape(-1)[:3].reshape(1, 3)
        points[:] = ordinary_point
    elif alias_kind == "miss_ids":
        hints = session._state.miss_primary_ids
        hints[0] = -1
    elif alias_kind == "cache_keys":
        hints = session._state.cache_leaf_ids
        hints[0] = -1
    elif alias_kind == "recency":
        hints = session._state.cache_recency
        hints[0] = 0
    else:
        hints = session._state.cache_payload.view(np.int64).reshape(-1)[:1]
        hints[0] = -1

    state.reset()
    values = np.full((1, 3), -1401.0)
    owners = np.full(1, -1402, dtype=np.int64)
    values_before = values.copy()
    owners_before = owners.copy()
    keys_before = session._state.cache_leaf_ids.copy()
    recency_before = session._state.cache_recency.copy()
    clock_before = session._state.clock
    dynamic_bits_before = (
        points.view(np.uint64).copy()
        if alias_kind == "workspace_points"
        else hints.view(np.uint64).copy()
    )

    with pytest.raises(ValueError, match="dynamic inputs must not overlap"):
        sample_refined_trilinear_vectors_cached(
            session, points, hints, values, owners
        )
    assert state.calls == []
    assert np.array_equal(values.view(np.uint64), values_before.view(np.uint64))
    assert np.array_equal(owners, owners_before)
    assert np.array_equal(session._state.cache_leaf_ids, keys_before)
    assert np.array_equal(session._state.cache_recency, recency_before)
    assert session._state.clock == clock_before
    dynamic_bits_after = (
        points.view(np.uint64)
        if alias_kind == "workspace_points"
        else hints.view(np.uint64)
    )
    assert np.array_equal(dynamic_bits_after, dynamic_bits_before)


def test_reader_callback_cannot_reenter_sampling_or_clear_session() -> None:
    artifact = make_artifact()
    reader_state = ReentrantReaderState(artifact.backing)
    session = make_session(artifact, reentrant_reader(reader_state), 1)
    reader_state.session = session
    point = points_for_leaves(artifact, [0])
    values = np.full((1, 3), -1601.0)
    owners = np.full(1, -1602, dtype=np.int64)
    stats = sample_refined_trilinear_vectors_cached(
        session, point, i3(-1), values, owners
    )
    assert stats.cache_miss_count == 1
    assert reader_state.attempted
    assert [name for name, _ in reader_state.nested_errors] == ["sample", "clear"]
    for _, error in reader_state.nested_errors:
        assert isinstance(error, ValueError)
        assert "already active" in str(error)
    assert session._state.active is False
    clear_completed_halo_sampling_session(session)
    assert session._state.cache_leaf_ids.tolist() == [-1]


def test_first_reader_failure_publishes_owners_preserves_values_and_resets_busy() -> None:
    artifact = make_artifact()
    state = ReaderState(artifact.backing)
    session = make_session(artifact, counting_reader(state), 1)
    point = points_for_leaves(artifact, [0])
    values = np.full((1, 3), -1611.0)
    owners = np.full(1, -1612, dtype=np.int64)
    state.reset()
    state.fail_nonempty_call = 1
    with pytest.raises(OSError, match="completed-halo read failure"):
        sample_refined_trilinear_vectors_cached(
            session, point, i3(-1), values, owners
        )
    assert owners.tolist() == [0]
    assert np.all(values == -1611.0)
    assert session._state.cache_leaf_ids.tolist() == [-1]
    assert session._state.clock == 0
    assert session._state.active is False

    state.reset()
    stats = sample_refined_trilinear_vectors_cached(
        session, point, i3(0), values, owners
    )
    assert stats.cache_miss_count == 1
    assert stats.hint_hit_count == 1
    assert session._state.cache_leaf_ids.tolist() == [0]
    assert session._state.active is False


def test_later_reader_failure_preserves_prior_group_and_failed_victim() -> None:
    artifact = make_artifact()
    state = ReaderState(artifact.backing)
    session = make_session(artifact, counting_reader(state), 1)
    seed = points_for_leaves(artifact, [8])
    sample_refined_trilinear_vectors_cached(
        session,
        seed,
        i3(8),
        np.empty((1, 3), dtype=np.float64),
        np.empty(1, dtype=np.int64),
    )
    assert session._state.cache_leaf_ids.tolist() == [8]

    points = points_for_leaves(artifact, [1, 0])
    expected, expected_owners = public_rps(artifact, points, i3(2, 0, 2))
    values = np.full((2, 3), -881.0)
    owners = np.full(2, -882, dtype=np.int64)
    state.reset()
    state.fail_nonempty_call = 2
    with pytest.raises(OSError, match="injected"):
        sample_refined_trilinear_vectors_cached(
            session, points, i3(-1, -1), values, owners
        )
    assert np.array_equal(owners, expected_owners)
    assert np.array_equal(values[1].view(np.uint64), expected[1].view(np.uint64))
    assert np.all(values[0] == -881.0)
    assert session._state.cache_leaf_ids.tolist() == [0]
    assert state.nonempty_calls == 2
    assert [int(call[0]) for call in state.calls] == [0, 1]


def test_copy_failure_invalidates_victim_and_cannot_create_false_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = make_artifact()
    state = ReaderState(artifact.backing)
    session = make_session(artifact, counting_reader(state), 1)
    seed = points_for_leaves(artifact, [8])
    sample_refined_trilinear_vectors_cached(
        session,
        seed,
        i3(8),
        np.empty((1, 3), dtype=np.float64),
        np.empty(1, dtype=np.int64),
    )
    assert session._state.cache_leaf_ids.tolist() == [8]
    state.reset()
    point0 = points_for_leaves(artifact, [0])
    original_copyto = np.copyto

    def fail_cache_copy(destination, source, *args, **kwargs):
        if np.shares_memory(destination, session._state.cache_payload):
            destination.reshape(-1)[0] = -12345.0
            raise MemoryError("injected cache copy failure")
        return original_copyto(destination, source, *args, **kwargs)

    monkeypatch.setattr(chs_module.np, "copyto", fail_cache_copy)
    with pytest.raises(MemoryError, match="injected cache copy"):
        sample_refined_trilinear_vectors_cached(
            session,
            point0,
            i3(-1),
            np.full((1, 3), -991.0),
            np.full(1, -992, dtype=np.int64),
        )
    assert session._state.cache_leaf_ids.tolist() == [-1]
    assert state.nonempty_calls == 1

    monkeypatch.setattr(chs_module.np, "copyto", original_copyto)
    values = np.full((1, 3), -993.0)
    owners = np.full(1, -994, dtype=np.int64)
    stats = sample_refined_trilinear_vectors_cached(
        session, point0, i3(0), values, owners
    )
    assert stats.cache_hit_count == 0
    assert stats.cache_miss_count == 1
    assert state.nonempty_calls == 2
    assert session._state.cache_leaf_ids.tolist() == [0]


def test_real_tdm_native_reader_matches_public_rps_for_one_owner() -> None:
    path = REPOSITORY_ROOT / "data/tdm.dat"
    if not path.exists():
        pytest.skip("real non-staggered tdm.dat fixture is unavailable")
    descriptor = os.open(path, os.O_RDONLY)
    try:
        index = read_amrvac_v5_index(descriptor)
        if index.staggered:
            pytest.skip("real tdm.dat fixture is staggered")
        binding = bind_amrvac_v5_forest(index)
        forest = binding.forest
        reader = make_amrvac_v5_block_reader(descriptor, index, binding)
        fields = i3(2, 0, 2)
        modes, normals = boundary_configuration()
        rhe_capacity = min(57, index.leaf_count)
        padded_volume = int(
            np.prod(index.block_cell_counts + 2, dtype=np.int64)
        )
        native_session = make_completed_halo_sampling_session(
            reader,
            fields,
            index.domain_lower,
            index.domain_upper,
            index.domain_cell_counts,
            index.block_cell_counts,
            forest.max_level,
            binding.root_shape,
            binding.coord_to_rank,
            forest.root_node_ids,
            forest.node_levels,
            forest.node_coords,
            forest.child_node_ids,
            forest.node_leaf_ids,
            forest.leaf_node_ids,
            modes,
            normals,
            rhe_capacity,
            3 * padded_volume * 8 + 16,
        )
        bounds, spacing = refined_leaf_geometry(
            index.domain_lower,
            index.domain_upper,
            binding.root_shape,
            index.domain_cell_counts,
            index.block_cell_counts,
            forest.node_levels,
            forest.node_coords,
            forest.leaf_node_ids,
            i3(0),
        )
        point = np.ascontiguousarray(
            bounds[:, 0] + np.asarray((0.25, 1.25, 2.25)) * spacing
        )
        cached_values = np.full((1, 3), -1701.0)
        cached_owners = np.full(1, -1702, dtype=np.int64)
        cached_stats = sample_refined_trilinear_vectors_cached(
            native_session,
            point,
            i3(-1),
            cached_values,
            cached_owners,
        )
        rps_values = np.full((1, 3), -1701.0)
        execute_refined_trilinear_points_from_blocks(
            reader,
            point,
            fields,
            index.domain_lower,
            index.domain_upper,
            binding.root_shape,
            index.domain_cell_counts,
            index.block_cell_counts,
            forest.max_level,
            binding.coord_to_rank,
            forest.root_node_ids,
            forest.node_levels,
            forest.node_coords,
            forest.child_node_ids,
            forest.node_leaf_ids,
            forest.leaf_node_ids,
            modes,
            normals,
            rhe_capacity,
            rps_values,
        )
        expected_owner = refined_point_leaf_ids(
            index.domain_lower,
            index.domain_upper,
            binding.root_shape,
            index.domain_cell_counts,
            index.block_cell_counts,
            forest.max_level,
            binding.coord_to_rank,
            forest.root_node_ids,
            forest.child_node_ids,
            forest.node_leaf_ids,
            point,
        )
        assert cached_stats.owner_count == cached_stats.cache_miss_count == 1
        assert np.array_equal(cached_owners, expected_owner)
        assert np.array_equal(
            cached_values.view(np.uint64), rps_values.view(np.uint64)
        )
    finally:
        os.close(descriptor)
