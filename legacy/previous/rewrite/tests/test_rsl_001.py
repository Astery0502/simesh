from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from simesh_rewrite.balance import validate_refined_all_touch_2to1
from simesh_rewrite.forest import refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.halo_plans import level1_halo_relation_plan
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.refined_support import (
    maximum_balanced_refined_support_slots,
    plan_balanced_refined_support_prefix,
)
from simesh_rewrite.relation_slots import (
    resolve_refined_relation_source_slots,
)
from simesh_rewrite.relation_slots_reference import (
    resolve_refined_relation_source_slots_reference,
)
from simesh_rewrite.relations import (
    RELATION_COARSER,
    RELATION_FINER,
    balanced_refined_relations,
)
from simesh_rewrite.topology import level1_face_neighbors


ALL_DIRECTIONS = np.asarray(
    [
        (dx, dy, dz)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ],
    dtype=np.int64,
)


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def assert_slots_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    assert np.array_equal(actual, expected)


def test_selected_permutation_primary_support_active_trailing_and_repeats() -> None:
    selected = i3(7, 2, 9, 1, 5)
    counts = np.asarray([[1, 4, 0], [1, 2, 1]], dtype=np.uint8)
    sources = np.full((2, 3, 4), -1, dtype=np.int64)
    sources[0, 0, 0] = 7
    sources[0, 1] = [2, 9, 5, 2]
    sources[1, 0, 0] = 1
    sources[1, 1, :2] = [5, 7]
    sources[1, 2, 0] = 9
    actual = np.full((2, 3, 4), -77, dtype=np.int64)
    expected = actual.copy()
    resolve_refined_relation_source_slots_reference(
        10, selected, counts, sources, expected
    )
    resolve_refined_relation_source_slots(10, selected, counts, sources, actual)
    assert_slots_equal(actual, expected)
    assert actual[0, 0, 0] == 0
    assert actual[0, 1].tolist() == [1, 2, 4, 1]
    assert actual[0, 2].tolist() == [-1, -1, -1, -1]
    assert actual[1, 0, 0] == 3


def test_zero_directions_empty_rows_and_p_le_s_invariant() -> None:
    selected = i3(3, 1, 4)
    counts = np.empty((2, 0), dtype=np.uint8)
    sources = np.empty((2, 0, 4), dtype=np.int64)
    slots = np.empty_like(sources)
    resolve_refined_relation_source_slots(5, selected, counts, sources, slots)
    assert slots.shape == (2, 0, 4)

    empty_counts = np.empty((0, 3), dtype=np.uint8)
    empty_sources = np.empty((0, 3, 4), dtype=np.int64)
    empty_slots = np.empty_like(empty_sources)
    resolve_refined_relation_source_slots(
        5, selected, empty_counts, empty_sources, empty_slots
    )

    too_many_counts = np.empty((4, 0), dtype=np.uint8)
    too_many_sources = np.empty((4, 0, 4), dtype=np.int64)
    too_many_slots = np.empty_like(too_many_sources)
    with pytest.raises(ValueError, match="accepted relation rows"):
        resolve_refined_relation_source_slots(
            5,
            selected,
            too_many_counts,
            too_many_sources,
            too_many_slots,
        )


@pytest.mark.parametrize("missing_position", [(0, 0, 0), (1, 1, 1)])
def test_first_and_late_missing_source_are_atomic(missing_position) -> None:
    selected = i3(0, 2, 4)
    counts = np.asarray([[1, 1], [1, 2]], dtype=np.uint8)
    sources = np.full((2, 2, 4), -1, dtype=np.int64)
    sources[0, 0, 0] = 0
    sources[0, 1, 0] = 2
    sources[1, 0, 0] = 4
    sources[1, 1, :2] = [0, 2]
    sources[missing_position] = 3
    output = np.full((2, 2, 4), -91, dtype=np.int64)
    before = output.copy()
    with pytest.raises(ValueError, match="absent from selected"):
        resolve_refined_relation_source_slots(
            5, selected, counts, sources, output
        )
    assert_slots_equal(output, before)
    with pytest.raises(ValueError, match="absent from selected"):
        resolve_refined_relation_source_slots_reference(
            5, selected, counts, sources, output
        )
    assert_slots_equal(output, before)


def assert_atomic_error(error, operation, output: np.ndarray) -> None:
    before = output.copy()
    with pytest.raises(error):
        operation()
    assert_slots_equal(output, before)


def test_selected_count_source_and_trailing_corruption_are_atomic() -> None:
    selected = i3(0, 1, 2)
    counts = np.asarray([[1, 0]], dtype=np.uint8)
    sources = np.full((1, 2, 4), -1, dtype=np.int64)
    sources[0, 0, 0] = 1

    def check(error, selected_value=selected, count_value=counts, source_value=sources):
        output = np.full((1, 2, 4), -101, dtype=np.int64)
        assert_atomic_error(
            error,
            lambda: resolve_refined_relation_source_slots(
                3, selected_value, count_value, source_value, output
            ),
            output,
        )

    check(ValueError, selected_value=i3(0, 1, 1))
    check(ValueError, selected_value=i3(0, 1, 3))
    bad_count = counts.copy()
    bad_count[0, 0] = 5
    check(ValueError, count_value=bad_count)
    bad_active = sources.copy()
    bad_active[0, 0, 0] = -1
    check(ValueError, source_value=bad_active)
    bad_active[0, 0, 0] = 3
    check(ValueError, source_value=bad_active)
    bad_trailing = sources.copy()
    bad_trailing[0, 0, 1] = 0
    check(ValueError, source_value=bad_trailing)


def test_representation_read_only_and_overlap_validation() -> None:
    selected = i3(0, 1)
    counts = np.asarray([[1]], dtype=np.uint8)
    sources = np.full((1, 1, 4), -1, dtype=np.int64)
    sources[0, 0, 0] = 1
    for value in (selected, counts, sources):
        value.setflags(write=False)
    output = np.empty((1, 1, 4), dtype=np.int64)
    resolve_refined_relation_source_slots(2, selected, counts, sources, output)
    assert output.tolist() == [[[1, -1, -1, -1]]]

    def check(error, selected_value=selected, count_value=counts, source_value=sources, output_value=None):
        destination = (
            np.full((1, 1, 4), -17, dtype=np.int64)
            if output_value is None
            else output_value
        )
        assert_atomic_error(
            error,
            lambda: resolve_refined_relation_source_slots(
                2, selected_value, count_value, source_value, destination
            ),
            destination,
        )

    check(TypeError, selected_value=selected.astype(np.int32))
    check(ValueError, selected_value=np.arange(4, dtype=np.int64)[::2])
    check(TypeError, count_value=counts.astype(np.int64))
    check(ValueError, count_value=counts.reshape(1, 1, 1))
    check(TypeError, source_value=sources.astype(np.int32))
    check(ValueError, source_value=sources[:, :, :3])
    check(TypeError, output_value=np.empty((1, 1, 4), dtype=np.int32))
    readonly_output = np.empty((1, 1, 4), dtype=np.int64)
    readonly_output.setflags(write=False)
    check(ValueError, output_value=readonly_output)
    overlapping = sources.copy()
    overlapping.setflags(write=True)
    check(ValueError, source_value=overlapping, output_value=overlapping)
    with pytest.raises(TypeError, match="leaf_count"):
        resolve_refined_relation_source_slots(
            True, selected, counts, sources, output
        )


def make_level1_refined_flags(
    root: np.ndarray,
    refined_roots: tuple[tuple[int, int, int], ...] | list[tuple[int, int, int]],
) -> np.ndarray:
    _, coordinates = level1_morton(root)
    values: list[bool] = []
    for coordinate in coordinates:
        split = tuple(int(value) for value in coordinate) in refined_roots
        values.append(not split)
        if split:
            values.extend([True] * 8)
    return np.asarray(values, dtype=np.bool_)


def forest_artifact(root: np.ndarray, flags: np.ndarray):
    coord_to_rank, rank_to_coord = level1_morton(root)
    forest = refined_forest(root, coord_to_rank, rank_to_coord, flags)
    validate_refined_forest_arrays(
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
    return coord_to_rank, rank_to_coord, forest


def test_random_balanced_rel_sto_windows_match_list_reference() -> None:
    rng = np.random.default_rng(20260902)
    for _ in range(20):
        root = rng.integers(1, 4, size=3, dtype=np.int64)
        _, root_coordinates = level1_morton(root)
        refined = [
            tuple(int(value) for value in coordinate)
            for coordinate in root_coordinates
            if rng.random() < 0.3
        ]
        coord_to_rank, _, forest = forest_artifact(
            root, make_level1_refined_flags(root, refined)
        )
        leaf_count = forest.leaf_node_ids.size
        leaf_ids = np.arange(leaf_count, dtype=np.int64)
        _, _, source_counts, source_leaf_ids = balanced_refined_relations(
            root,
            coord_to_rank,
            forest.root_node_ids,
            forest.node_levels,
            forest.node_coords,
            forest.child_node_ids,
            forest.node_leaf_ids,
            forest.leaf_node_ids,
            leaf_ids,
            ALL_DIRECTIONS,
        )
        minimum = maximum_balanced_refined_support_slots(
            0, leaf_count, source_counts, source_leaf_ids
        )
        capacity = min(leaf_count, minimum + int(rng.integers(0, 8)))
        selected = np.empty(capacity, dtype=np.int64)
        first = 0
        while first < leaf_count:
            candidate_count = min(capacity, leaf_count - first)
            primary_count, selected_count = plan_balanced_refined_support_prefix(
                first,
                leaf_count,
                source_counts[first : first + candidate_count],
                source_leaf_ids[first : first + candidate_count],
                selected,
            )
            accepted_counts = source_counts[first : first + primary_count]
            accepted_sources = source_leaf_ids[first : first + primary_count]
            actual = np.empty((*accepted_counts.shape, 4), dtype=np.int64)
            expected = np.empty_like(actual)
            resolve_refined_relation_source_slots_reference(
                leaf_count,
                selected[:selected_count],
                accepted_counts,
                accepted_sources,
                expected,
            )
            resolve_refined_relation_source_slots(
                leaf_count,
                selected[:selected_count],
                accepted_counts,
                accepted_sources,
                actual,
            )
            assert_slots_equal(actual, expected)
            for primary, direction in np.ndindex(accepted_counts.shape):
                for source in range(int(accepted_counts[primary, direction])):
                    assert (
                        selected[actual[primary, direction, source]]
                        == accepted_sources[primary, direction, source]
                    )
            first += primary_count


def test_level_one_reduces_to_hpl_source_slots() -> None:
    root = i3(3, 2, 1)
    coord_to_rank, rank_to_coord, forest = forest_artifact(
        root,
        make_level1_refined_flags(root, ()),
    )
    faces = level1_face_neighbors(root, coord_to_rank, rank_to_coord)
    selected = np.arange(forest.leaf_node_ids.size, dtype=np.int64)[::-1].copy()
    kinds, masks, counts, sources = balanced_refined_relations(
        root,
        coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
        selected,
        ALL_DIRECTIONS,
    )
    del kinds, masks
    slots = np.empty_like(sources)
    resolve_refined_relation_source_slots(
        selected.size, selected, counts, sources, slots
    )
    hpl_slots, _ = level1_halo_relation_plan(
        selected, selected.size, faces
    )
    for primary in range(selected.size):
        for direction, (dx, dy, dz) in enumerate(ALL_DIRECTIONS):
            column = (dz + 1) * 9 + (dy + 1) * 3 + dx + 1
            if counts[primary, direction] == 0:
                assert hpl_slots[primary, column] == -1
            else:
                assert counts[primary, direction] == 1
                assert slots[primary, direction, 0] == hpl_slots[primary, column]


def test_finer_and_coarser_consumers_use_slots_without_rsl_kind_policy() -> None:
    root = i3(2, 1, 1)
    coord_to_rank, _, forest = forest_artifact(
        root,
        make_level1_refined_flags(root, {(0, 0, 0)}),
    )
    selected = np.arange(forest.leaf_node_ids.size, dtype=np.int64)[::-1].copy()
    kinds, _, counts, sources = balanced_refined_relations(
        root,
        coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
        selected,
        ALL_DIRECTIONS,
    )
    slots = np.empty_like(sources)
    resolve_refined_relation_source_slots(
        selected.size, selected, counts, sources, slots
    )
    payload_identity = selected.astype(np.float64) * 10.0 + 1.0
    finer_records = np.argwhere(kinds == RELATION_FINER)
    coarser_records = np.argwhere(kinds == RELATION_COARSER)
    assert finer_records.size and coarser_records.size
    for records, kind in (
        (finer_records, RELATION_FINER),
        (coarser_records, RELATION_COARSER),
    ):
        primary, direction = (int(value) for value in records[0])
        assert kinds[primary, direction] == kind
        count = int(counts[primary, direction])
        for source in range(count):
            slot = int(slots[primary, direction, source])
            global_source = int(sources[primary, direction, source])
            assert selected[slot] == global_source
            assert payload_identity[slot] == global_source * 10.0 + 1.0


def test_weno_bounded_active_roundtrip_when_available() -> None:
    path = Path(__file__).resolve().parents[2] / "data/weno509_sub_0000.dat"
    if not path.exists():
        pytest.skip("representative refined AMRVAC evidence file is unavailable")
    from simesh.amrvac.datio import get_metadata

    header, flags, _ = get_metadata(str(path))
    root = np.ascontiguousarray(
        header["domain_nx"] // header["block_nx"], dtype=np.int64
    )
    coord_to_rank, _, forest = forest_artifact(
        root, np.ascontiguousarray(flags, dtype=np.bool_)
    )
    leaf_count = forest.leaf_node_ids.size
    leaf_ids = np.arange(leaf_count, dtype=np.int64)
    _, _, counts, sources = balanced_refined_relations(
        root,
        coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
        leaf_ids,
        ALL_DIRECTIONS,
    )
    capacity = 256
    selected = np.empty(capacity, dtype=np.int64)
    first = 0
    covered = 0
    while first < leaf_count:
        candidate_count = min(capacity, leaf_count - first)
        primary_count, selected_count = plan_balanced_refined_support_prefix(
            first,
            leaf_count,
            counts[first : first + candidate_count],
            sources[first : first + candidate_count],
            selected,
        )
        accepted_counts = counts[first : first + primary_count]
        accepted_sources = sources[first : first + primary_count]
        slots = np.empty_like(accepted_sources)
        resolve_refined_relation_source_slots(
            leaf_count,
            selected[:selected_count],
            accepted_counts,
            accepted_sources,
            slots,
        )
        for primary, direction in np.ndindex(accepted_counts.shape):
            for source in range(int(accepted_counts[primary, direction])):
                assert (
                    selected[slots[primary, direction, source]]
                    == accepted_sources[primary, direction, source]
                )
        first += primary_count
        covered += primary_count
    assert covered == leaf_count
