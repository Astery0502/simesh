from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from simesh_rewrite.balance import validate_refined_all_touch_2to1
from simesh_rewrite.blockio import (
    array_block_reader,
    array_block_writer,
    make_block_reader,
)
from simesh_rewrite.forest import refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.point_location import fill_refined_point_leaf_ids
from simesh_rewrite.refined_geometry import refined_leaf_geometry
from simesh_rewrite.refined_halo import (
    _execute_selected_refined_halos_with_consumer,
    execute_selected_refined_halos_from_blocks,
)
from simesh_rewrite.refined_sampling import (
    sample_refined_trilinear_point_groups,
    sample_refined_zero_order_point_groups,
)
from simesh_rewrite.repeated_sampling import (
    RepeatedPointExecutionStats,
    _make_point_plan,
    execute_refined_trilinear_points_from_blocks,
    execute_refined_zero_order_points_from_blocks,
)
from simesh_rewrite.coarser_support import CANONICAL_DIRECTIONS
from simesh_rewrite.relations import balanced_refined_relations
from simesh_rewrite.storage import gather_blocks_into


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


@dataclass(frozen=True)
class Artifact:
    root_shape: np.ndarray
    coord_to_rank: np.ndarray
    rank_to_coord: np.ndarray
    root_node_ids: np.ndarray
    node_levels: np.ndarray
    node_coords: np.ndarray
    parent_node_ids: np.ndarray
    child_node_ids: np.ndarray
    node_leaf_ids: np.ndarray
    leaf_node_ids: np.ndarray
    max_level: int


def make_artifact(
    root_shape: tuple[int, int, int],
    refined_roots: set[tuple[int, int, int]],
) -> Artifact:
    root = i3(*root_shape)
    coord_to_rank, rank_to_coord = level1_morton(root)
    flags: list[bool] = []
    for coordinate in rank_to_coord:
        refined = tuple(int(value) for value in coordinate) in refined_roots
        flags.append(not refined)
        if refined:
            flags.extend([True] * 8)
    forest = refined_forest(
        root,
        coord_to_rank,
        rank_to_coord,
        np.asarray(flags, dtype=np.bool_),
    )
    max_level = validate_refined_forest_arrays(
        root,
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
        root,
        coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
    )
    return Artifact(
        root,
        coord_to_rank,
        rank_to_coord,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.parent_node_ids,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
        max_level,
    )


def axis_coded_backing(
    leaf_count: int,
    field_count: int,
    block_shape: tuple[int, int, int],
) -> np.ndarray:
    x, y, z = np.indices(block_shape, dtype=np.float64)
    backing = np.empty(
        (leaf_count, field_count, *block_shape),
        dtype=np.float64,
    )
    for leaf in range(leaf_count):
        for field_index in range(field_count):
            backing[leaf, field_index] = (
                100000.0 * leaf
                + 10000.0 * field_index
                + 100.0 * x
                + 10.0 * y
                + z
                + 1.0
            )
    return backing


@dataclass
class CountingReaderState:
    backing: np.ndarray
    calls: list[np.ndarray] = field(default_factory=list)
    fail_nonempty_call: int | None = None
    nonempty_calls: int = 0


def counting_read_into(
    state: CountingReaderState,
    source_valid_lower: np.ndarray,
    source_valid_upper: np.ndarray,
    block_ids: np.ndarray,
    field_ids: np.ndarray,
    destination: np.ndarray,
    destination_lower: np.ndarray,
) -> None:
    state.calls.append(block_ids.copy())
    if block_ids.shape[0]:
        state.nonempty_calls += 1
        if state.fail_nonempty_call == state.nonempty_calls:
            raise RuntimeError("injected read failure")
    gather_blocks_into(
        state.backing,
        source_valid_lower,
        source_valid_upper,
        block_ids,
        field_ids,
        destination,
        destination_lower,
    )


def counting_reader(state: CountingReaderState):
    return make_block_reader(
        state,
        state.backing.shape,
        counting_read_into,
        memory_arrays=(state.backing,),
    )


def geometry_inputs(
    artifact: Artifact,
    block_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lower = np.asarray((0.0, 0.0, 0.0), dtype=np.float64)
    upper = artifact.root_shape.astype(np.float64)
    block_counts = i3(*block_shape)
    domain_counts = artifact.root_shape * block_counts
    return lower, upper, domain_counts, block_counts


def points_for_leaves(
    artifact: Artifact,
    leaf_ids: list[int],
    lower: np.ndarray,
    upper: np.ndarray,
    domain_counts: np.ndarray,
    block_counts: np.ndarray,
) -> np.ndarray:
    selected = np.asarray(leaf_ids, dtype=np.int64)
    bounds, spacing = refined_leaf_geometry(
        lower,
        upper,
        artifact.root_shape,
        domain_counts,
        block_counts,
        artifact.node_levels,
        artifact.node_coords,
        artifact.leaf_node_ids,
        selected,
    )
    fractions = ((0.25, 1.25, 2.25), (2.25, 0.25, 1.25), (1.25, 2.25, 0.25))
    result = np.empty((len(leaf_ids), 3), dtype=np.float64)
    for row, fraction in enumerate(fractions[: len(leaf_ids)]):
        for axis in range(3):
            result[row, axis] = (
                float(bounds[row, 0, axis])
                + float(fraction[axis]) * float(spacing[row, axis])
            )
    return result


def locate_and_group(
    artifact: Artifact,
    points: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    domain_counts: np.ndarray,
    block_counts: np.ndarray,
):
    owners = np.empty(points.shape[0], dtype=np.int64)
    fill_refined_point_leaf_ids(
        lower,
        upper,
        artifact.root_shape,
        domain_counts,
        block_counts,
        artifact.max_level,
        artifact.coord_to_rank,
        artifact.root_node_ids,
        artifact.child_node_ids,
        artifact.node_leaf_ids,
        points,
        owners,
    )
    return _make_point_plan(owners)


def zero_args(
    artifact: Artifact,
    reader,
    points: np.ndarray,
    fields: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    domain_counts: np.ndarray,
    block_counts: np.ndarray,
    capacity: int,
    output: np.ndarray,
) -> tuple:
    return (
        reader,
        points,
        fields,
        lower,
        upper,
        artifact.root_shape,
        domain_counts,
        block_counts,
        artifact.max_level,
        artifact.coord_to_rank,
        artifact.root_node_ids,
        artifact.node_levels,
        artifact.node_coords,
        artifact.child_node_ids,
        artifact.node_leaf_ids,
        artifact.leaf_node_ids,
        capacity,
        output,
    )


def trilinear_args(
    artifact: Artifact,
    reader,
    points: np.ndarray,
    fields: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    domain_counts: np.ndarray,
    block_counts: np.ndarray,
    modes: np.ndarray,
    normals: np.ndarray,
    capacity: int,
    output: np.ndarray,
) -> tuple:
    return (
        reader,
        points,
        fields,
        lower,
        upper,
        artifact.root_shape,
        domain_counts,
        block_counts,
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
        capacity,
        output,
    )


def test_zero_order_groups_are_stable_bounded_and_read_each_owner_once() -> None:
    artifact = make_artifact((2, 1, 1), {(0, 0, 0)})
    block_shape = (4, 4, 4)
    lower, upper, domain_counts, block_counts = geometry_inputs(
        artifact, block_shape
    )
    leaf_count = int(artifact.leaf_node_ids.shape[0])
    backing = axis_coded_backing(leaf_count, 2, block_shape)
    base_points = points_for_leaves(
        artifact,
        [8, 0, 3],
        lower,
        upper,
        domain_counts,
        block_counts,
    )
    points = np.ascontiguousarray(
        np.asarray(
            [
                base_points[0],
                base_points[1],
                base_points[0],
                (-0.25, 0.5, 0.5),
                base_points[2],
                base_points[1],
            ],
            dtype=np.float64,
        )
    )
    fields = np.asarray((1, 0, 1), dtype=np.int64)
    actual = np.full((points.shape[0], fields.shape[0]), -777.0)
    expected = actual.copy()
    state = CountingReaderState(backing)

    stats = execute_refined_zero_order_points_from_blocks(
        *zero_args(
            artifact,
            counting_reader(state),
            points,
            fields,
            lower,
            upper,
            domain_counts,
            block_counts,
            2,
            actual,
        )
    )

    plan = locate_and_group(
        artifact, points, lower, upper, domain_counts, block_counts
    )
    zero = i3(0, 0, 0)
    sample_refined_zero_order_point_groups(
        np.ascontiguousarray(backing[plan.owner_leaf_ids][:, fields]),
        zero,
        block_counts,
        zero,
        block_counts,
        plan.owner_leaf_ids,
        lower,
        upper,
        domain_counts,
        block_counts,
        artifact.node_levels,
        artifact.node_coords,
        artifact.leaf_node_ids,
        points,
        plan.grouped_point_indices,
        plan.owner_offsets,
        expected,
    )

    assert isinstance(stats, RepeatedPointExecutionStats)
    assert stats[:8] == (6, 5, 3, 2, 2, 2, 3, 2)
    assert stats.managed_array_bytes == (
        8 * (6 + 5 + 3 + 4) + 24 + 2 * 3 * 4 * 4 * 4 * 8 + 24
    )
    assert len(state.calls) == 3
    assert state.calls[0].shape == (0,)
    assert [call.tolist() for call in state.calls[1:]] == [[0, 3], [8]]
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))
    assert np.all(actual[3] == -777.0)


def test_zero_order_all_exterior_accepts_zero_capacity_and_odd_blocks() -> None:
    artifact = make_artifact((1, 1, 1), set())
    block_shape = (3, 3, 3)
    lower, upper, domain_counts, block_counts = geometry_inputs(
        artifact, block_shape
    )
    backing = axis_coded_backing(1, 1, block_shape)
    state = CountingReaderState(backing)
    points = np.asarray(((-1.0, 0.0, 0.0), (1.0, 0.5, 0.5)))
    output = np.asarray(((7.0,), (-0.0,)), dtype=np.float64)
    before = output.copy()

    stats = execute_refined_zero_order_points_from_blocks(
        *zero_args(
            artifact,
            counting_reader(state),
            points,
            i3(0),
            lower,
            upper,
            domain_counts,
            block_counts,
            0,
            output,
        )
    )

    assert stats[:8] == (2, 0, 0, 0, 0, 0, 0, 0)
    assert stats.managed_array_bytes == 72
    assert len(state.calls) == 1 and state.calls[0].shape == (0,)
    assert np.array_equal(output.view(np.uint64), before.view(np.uint64))


def test_nonfinite_and_output_alias_reject_before_reader_or_mutation() -> None:
    artifact = make_artifact((1, 1, 1), set())
    block_shape = (4, 4, 4)
    lower, upper, domain_counts, block_counts = geometry_inputs(
        artifact, block_shape
    )
    backing = axis_coded_backing(1, 1, block_shape)
    state = CountingReaderState(backing)
    points = np.asarray(((0.25, 0.25, 0.25), (np.nan, 0.5, 0.5)))
    output = np.full((2, 1), -13.0)

    with pytest.raises(ValueError, match="finite"):
        execute_refined_zero_order_points_from_blocks(
            *zero_args(
                artifact,
                counting_reader(state),
                points,
                i3(0),
                lower,
                upper,
                domain_counts,
                block_counts,
                1,
                output,
            )
        )
    assert state.calls == []
    assert np.all(output == -13.0)

    alias_points = backing.reshape(-1)[:6].reshape(2, 3)
    alias_output = backing.reshape(-1)[6:8].reshape(2, 1)
    with pytest.raises(ValueError, match="overlap"):
        execute_refined_zero_order_points_from_blocks(
            *zero_args(
                artifact,
                counting_reader(state),
                alias_points,
                i3(0),
                lower,
                upper,
                domain_counts,
                block_counts,
                1,
                alias_output,
            )
        )
    assert state.calls == []


@pytest.mark.parametrize(
    ("argument_index", "replacement", "error", "message"),
    [
        (2, np.asarray((1,), dtype=np.int64), ValueError, "field_ids"),
        (7, i3(2, 4, 4), ValueError, "block_cell_counts"),
        (16, 0, ValueError, "positive"),
        (16, True, TypeError, "integer"),
        (16, 2, ValueError, "exceeds leaf count"),
        (17, np.empty((2, 2), dtype=np.float64), ValueError, "point_values"),
    ],
)
def test_zero_order_request_errors_precede_reader_and_output_mutation(
    argument_index: int,
    replacement,
    error: type[Exception],
    message: str,
) -> None:
    artifact = make_artifact((1, 1, 1), set())
    block_shape = (4, 4, 4)
    lower, upper, domain_counts, block_counts = geometry_inputs(
        artifact, block_shape
    )
    backing = axis_coded_backing(1, 1, block_shape)
    state = CountingReaderState(backing)
    points = np.asarray(((0.25, 0.25, 0.25),), dtype=np.float64)
    output = np.full((1, 1), -17.0)
    arguments = list(
        zero_args(
            artifact,
            counting_reader(state),
            points,
            i3(0),
            lower,
            upper,
            domain_counts,
            block_counts,
            1,
            output,
        )
    )
    arguments[argument_index] = replacement

    with pytest.raises(error, match=message):
        execute_refined_zero_order_points_from_blocks(*arguments)

    assert state.calls == []
    assert output[0, 0] == -17.0


def test_trilinear_private_consumer_matches_public_rhe_resident_composition() -> None:
    artifact = make_artifact((2, 1, 1), {(0, 0, 0)})
    block_shape = (4, 4, 4)
    lower, upper, domain_counts, block_counts = geometry_inputs(
        artifact, block_shape
    )
    leaf_count = int(artifact.leaf_node_ids.shape[0])
    backing = axis_coded_backing(leaf_count, 2, block_shape)
    base_points = points_for_leaves(
        artifact,
        [8, 0, 1],
        lower,
        upper,
        domain_counts,
        block_counts,
    )
    points = np.ascontiguousarray(
        np.asarray(
            [
                base_points[0],
                base_points[1],
                base_points[0],
                base_points[2],
                (2.0, 0.5, 0.5),
            ],
            dtype=np.float64,
        )
    )
    fields = np.asarray((1, 0, 1), dtype=np.int64)
    modes = np.zeros((fields.shape[0], 6), dtype=np.uint8)
    modes[0, 0] = 2
    modes[1, 2] = 3
    modes[2, 4] = 1
    normals = i3(-1, 1, -1)
    actual = np.full((points.shape[0], fields.shape[0]), -999.0)
    expected = actual.copy()
    state = CountingReaderState(backing)

    stats = execute_refined_trilinear_points_from_blocks(
        counting_reader(state),
        points,
        fields,
        lower,
        upper,
        artifact.root_shape,
        domain_counts,
        block_counts,
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
        leaf_count,
        actual,
    )

    plan = locate_and_group(
        artifact, points, lower, upper, domain_counts, block_counts
    )
    kinds, masks, _, _ = balanced_refined_relations(
        artifact.root_shape,
        artifact.coord_to_rank,
        artifact.root_node_ids,
        artifact.node_levels,
        artifact.node_coords,
        artifact.child_node_ids,
        artifact.node_leaf_ids,
        artifact.leaf_node_ids,
        plan.owner_leaf_ids,
        CANONICAL_DIRECTIONS,
    )
    assert set(int(value) for value in np.unique(kinds)) == {1, 2, 3, 4}
    assert np.any((masks != 0) & (kinds == 2))
    assert np.any((masks != 0) & (kinds == 3))
    assert np.any((masks != 0) & (kinds == 4))
    assert set(int(value) for value in np.unique(modes)) == {0, 1, 2, 3}
    padded_shape = tuple(value + 2 for value in block_shape)
    completed = np.full(
        (leaf_count, fields.shape[0], *padded_shape),
        np.nan,
        dtype=np.float64,
    )
    rhe_stats = execute_selected_refined_halos_from_blocks(
        array_block_reader(backing),
        array_block_writer(completed),
        plan.owner_leaf_ids,
        fields,
        artifact.root_shape,
        artifact.coord_to_rank,
        artifact.root_node_ids,
        artifact.node_levels,
        artifact.node_coords,
        artifact.child_node_ids,
        artifact.node_leaf_ids,
        artifact.leaf_node_ids,
        i3(1, 1, 1),
        i3(1, 1, 1),
        modes,
        normals,
        leaf_count,
    )
    selected_completed = np.ascontiguousarray(completed[plan.owner_leaf_ids])
    sample_refined_trilinear_point_groups(
        selected_completed,
        i3(0, 0, 0),
        i3(*padded_shape),
        i3(1, 1, 1),
        i3(5, 5, 5),
        plan.owner_leaf_ids,
        lower,
        upper,
        domain_counts,
        block_counts,
        artifact.node_levels,
        artifact.node_coords,
        artifact.leaf_node_ids,
        points,
        plan.grouped_point_indices,
        plan.owner_offsets,
        expected,
    )

    assert stats.point_count == 5
    assert stats.inside_point_count == 4
    assert stats.owner_count == 3
    assert stats.chunk_count == stats.reader_call_count == stats.sampler_call_count == 1
    assert stats.selected_load_count == rhe_stats.selected_load_count
    assert stats.maximum_selected_slots == rhe_stats.maximum_selected_slots
    assert stats.managed_array_bytes == (
        sum(
            value.nbytes
            for value in (
                plan.point_owner_leaf_ids,
                plan.grouped_point_indices,
                plan.owner_leaf_ids,
                plan.owner_offsets,
            )
        )
        + 48
        + rhe_stats.managed_array_bytes
    )
    assert len(state.calls) == 2 and state.calls[0].shape == (0,)
    assert state.calls[1].tolist() == [0, 1, 8, 2, 3, 4, 5, 6, 7]
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))
    assert np.all(actual[4] == -999.0)


def test_trilinear_nonzero_chunk_offset_is_capacity_invariant() -> None:
    artifact = make_artifact((4, 4, 4), set())
    block_shape = (4, 4, 4)
    lower, upper, domain_counts, block_counts = geometry_inputs(
        artifact, block_shape
    )
    leaf_count = int(artifact.leaf_node_ids.shape[0])
    backing = axis_coded_backing(leaf_count, 1, block_shape)
    leaf_ids = np.arange(leaf_count, dtype=np.int64)
    bounds, spacing = refined_leaf_geometry(
        lower,
        upper,
        artifact.root_shape,
        domain_counts,
        block_counts,
        artifact.node_levels,
        artifact.node_coords,
        artifact.leaf_node_ids,
        leaf_ids,
    )
    points = np.ascontiguousarray(bounds[:, 0] + 0.75 * spacing)
    fields = i3(0)
    modes = np.zeros((1, 6), dtype=np.uint8)
    normals = i3(-1, -1, -1)
    bounded = np.full((leaf_count, 1), np.nan)
    resident = np.full_like(bounded, np.nan)
    bounded_state = CountingReaderState(backing)

    bounded_stats = execute_refined_trilinear_points_from_blocks(
        counting_reader(bounded_state),
        points,
        fields,
        lower,
        upper,
        artifact.root_shape,
        domain_counts,
        block_counts,
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
        57,
        bounded,
    )
    resident_stats = execute_refined_trilinear_points_from_blocks(
        array_block_reader(backing),
        points,
        fields,
        lower,
        upper,
        artifact.root_shape,
        domain_counts,
        block_counts,
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
        leaf_count,
        resident,
    )

    assert bounded_stats.chunk_count == bounded_stats.sampler_call_count == 2
    assert bounded_stats.selected_load_count == 102
    assert bounded_stats.maximum_selected_slots == 57
    assert resident_stats.chunk_count == 1
    assert [call.shape[0] for call in bounded_state.calls] == [0, 57, 45]
    assert np.array_equal(bounded.view(np.uint64), resident.view(np.uint64))


def test_trilinear_all_exterior_accepts_zero_capacity_and_only_empty_read() -> None:
    artifact = make_artifact((1, 1, 1), set())
    block_shape = (4, 4, 4)
    lower, upper, domain_counts, block_counts = geometry_inputs(
        artifact, block_shape
    )
    backing = axis_coded_backing(1, 1, block_shape)
    state = CountingReaderState(backing)
    points = np.asarray(((-1.0, 0.5, 0.5), (1.0, 0.5, 0.5)))
    output = np.asarray(((13.0,), (-0.0,)), dtype=np.float64)
    before = output.copy()

    stats = execute_refined_trilinear_points_from_blocks(
        *trilinear_args(
            artifact,
            counting_reader(state),
            points,
            i3(0),
            lower,
            upper,
            domain_counts,
            block_counts,
            np.zeros((1, 6), dtype=np.uint8),
            i3(-1, -1, -1),
            0,
            output,
        )
    )

    assert stats[:8] == (2, 0, 0, 0, 0, 0, 0, 0)
    assert len(state.calls) == 1 and state.calls[0].shape == (0,)
    assert np.array_equal(output.view(np.uint64), before.view(np.uint64))


@pytest.mark.parametrize("fail_nonempty_call", [1, 2])
def test_trilinear_reader_failure_preserves_only_completed_primary_groups(
    fail_nonempty_call: int,
) -> None:
    artifact = make_artifact((4, 4, 4), set())
    block_shape = (4, 4, 4)
    lower, upper, domain_counts, block_counts = geometry_inputs(
        artifact, block_shape
    )
    leaf_count = int(artifact.leaf_node_ids.shape[0])
    backing = axis_coded_backing(leaf_count, 1, block_shape)
    leaf_ids = np.arange(leaf_count, dtype=np.int64)
    bounds, spacing = refined_leaf_geometry(
        lower,
        upper,
        artifact.root_shape,
        domain_counts,
        block_counts,
        artifact.node_levels,
        artifact.node_coords,
        artifact.leaf_node_ids,
        leaf_ids,
    )
    points = np.ascontiguousarray(bounds[:, 0] + 0.75 * spacing)
    output = np.full((leaf_count, 1), -123.0)
    state = CountingReaderState(
        backing,
        fail_nonempty_call=fail_nonempty_call,
    )

    with pytest.raises(RuntimeError, match="injected read"):
        execute_refined_trilinear_points_from_blocks(
            *trilinear_args(
                artifact,
                counting_reader(state),
                points,
                i3(0),
                lower,
                upper,
                domain_counts,
                block_counts,
                np.zeros((1, 6), dtype=np.uint8),
                i3(-1, -1, -1),
                57,
                output,
            )
        )

    changed = np.flatnonzero(output[:, 0] != -123.0)
    assert [call.shape[0] for call in state.calls] == (
        [0, 57] if fail_nonempty_call == 1 else [0, 57, 45]
    )
    if fail_nonempty_call == 1:
        assert changed.shape[0] == 0
    else:
        assert changed.tolist() == list(range(40))


def test_private_rhe_consumer_receives_read_only_completed_primary_views() -> None:
    artifact = make_artifact((1, 1, 1), set())
    block_shape = (4, 4, 4)
    backing = axis_coded_backing(1, 1, block_shape)
    seen: list[tuple[int, np.ndarray, np.ndarray]] = []

    def consume(offset, leaf_ids, payload, valid_lower, valid_upper) -> None:
        assert not leaf_ids.flags.writeable
        assert not payload.flags.writeable
        assert not valid_lower.flags.writeable
        assert not valid_upper.flags.writeable
        seen.append((offset, leaf_ids.copy(), payload.copy()))
        assert valid_lower.tolist() == [0, 0, 0]
        assert valid_upper.tolist() == [6, 6, 6]

    arguments = (
        array_block_reader(backing),
        i3(0),
        i3(0),
        artifact.root_shape,
        artifact.coord_to_rank,
        artifact.root_node_ids,
        artifact.node_levels,
        artifact.node_coords,
        artifact.child_node_ids,
        artifact.node_leaf_ids,
        artifact.leaf_node_ids,
        i3(1, 1, 1),
        i3(1, 1, 1),
        np.zeros((1, 6), dtype=np.uint8),
        i3(-1, -1, -1),
        1,
    )
    stats = _execute_selected_refined_halos_with_consumer(*arguments, consume)

    assert stats.writer_calls == 0
    assert len(seen) == 1
    assert seen[0][0] == 0
    assert seen[0][1].tolist() == [0]
    assert seen[0][2].shape == (1, 1, 6, 6, 6)

    def fail(*_args) -> None:
        raise LookupError("injected consumer failure")

    with pytest.raises(LookupError, match="injected consumer"):
        _execute_selected_refined_halos_with_consumer(*arguments, fail)
    with pytest.raises(TypeError, match="must return None"):
        _execute_selected_refined_halos_with_consumer(
            *arguments,
            lambda *_args: 1,
        )


def test_trilinear_boundary_error_precedes_empty_reader_and_output_mutation() -> None:
    artifact = make_artifact((1, 1, 1), set())
    block_shape = (4, 4, 4)
    lower, upper, domain_counts, block_counts = geometry_inputs(
        artifact, block_shape
    )
    backing = axis_coded_backing(1, 1, block_shape)
    state = CountingReaderState(backing)
    points = np.asarray(((0.25, 0.25, 0.25),), dtype=np.float64)
    output = np.full((1, 1), -8.0)
    modes = np.full((1, 6), 255, dtype=np.uint8)

    with pytest.raises(ValueError, match="unknown mode"):
        execute_refined_trilinear_points_from_blocks(
            counting_reader(state),
            points,
            i3(0),
            lower,
            upper,
            artifact.root_shape,
            domain_counts,
            block_counts,
            artifact.max_level,
            artifact.coord_to_rank,
            artifact.root_node_ids,
            artifact.node_levels,
            artifact.node_coords,
            artifact.child_node_ids,
            artifact.node_leaf_ids,
            artifact.leaf_node_ids,
            modes,
            i3(-1, -1, -1),
            1,
            output,
        )
    assert state.calls == []
    assert output[0, 0] == -8.0


def test_zero_order_later_reader_failure_keeps_only_completed_owner_groups() -> None:
    artifact = make_artifact((2, 1, 1), {(0, 0, 0)})
    block_shape = (4, 4, 4)
    lower, upper, domain_counts, block_counts = geometry_inputs(
        artifact, block_shape
    )
    backing = axis_coded_backing(
        int(artifact.leaf_node_ids.shape[0]), 1, block_shape
    )
    base_points = points_for_leaves(
        artifact,
        [8, 0, 3],
        lower,
        upper,
        domain_counts,
        block_counts,
    )
    points = np.ascontiguousarray(base_points[[0, 1, 2, 0]])
    output = np.full((4, 1), -5.0)
    state = CountingReaderState(backing, fail_nonempty_call=2)

    with pytest.raises(RuntimeError, match="injected"):
        execute_refined_zero_order_points_from_blocks(
            *zero_args(
                artifact,
                counting_reader(state),
                points,
                i3(0),
                lower,
                upper,
                domain_counts,
                block_counts,
                1,
                output,
            )
        )

    plan = locate_and_group(
        artifact, points, lower, upper, domain_counts, block_counts
    )
    completed_point = int(plan.grouped_point_indices[0])
    assert output[completed_point, 0] != -5.0
    assert np.count_nonzero(output[:, 0] != -5.0) == 1
    assert len(state.calls) == 3
