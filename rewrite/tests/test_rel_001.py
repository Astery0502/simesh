from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from simesh.utils.lib.amr.forest import AMRForest
from simesh_rewrite.balance import validate_refined_all_touch_2to1
from simesh_rewrite.balance_reference import is_refined_all_touch_2to1_reference
from simesh_rewrite.forest import refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.halo_plans import level1_halo_relation_plan
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.relations import (
    balanced_refined_relations,
    fill_balanced_refined_relations,
)
from simesh_rewrite.topology import level1_face_neighbors


PHYSICAL = np.uint8(1)
COARSER = np.uint8(2)
SAME = np.uint8(3)
FINER = np.uint8(4)

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
ALL_COLUMNS = np.asarray(
    [
        (dz + 1) * 9 + (dy + 1) * 3 + dx + 1
        for dx, dy, dz in ALL_DIRECTIONS
    ],
    dtype=np.int64,
)
FACE_DIRECTIONS = np.asarray(
    [
        [-1, 0, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, 0, -1],
        [0, 0, 1],
    ],
    dtype=np.int64,
)
CURRENT_FACE_COLUMNS = np.asarray([12, 14, 10, 16, 4, 22], dtype=np.int64)


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def make_flags(root_shape: np.ndarray, refine) -> np.ndarray:
    _, roots = level1_morton(root_shape)
    flags: list[bool] = []

    def visit(level: int, coord: tuple[int, int, int]) -> None:
        split = bool(refine(level, coord))
        flags.append(not split)
        if not split:
            return
        for child in range(8):
            bits = (child & 1, (child >> 1) & 1, (child >> 2) & 1)
            visit(
                level + 1,
                tuple(2 * coord[axis] + bits[axis] for axis in range(3)),
            )

    for root in roots:
        visit(1, tuple(int(value) for value in root))
    return np.asarray(flags, dtype=bool)


def artifact(
    root_shape: tuple[int, int, int],
    refine,
    *,
    require_balance: bool = True,
):
    root = i3(*root_shape)
    coord_to_rank, rank_to_coord = level1_morton(root)
    flags = make_flags(root, refine)
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
    if require_balance:
        validate_refined_all_touch_2to1(
            *relation_forest_arguments(root, coord_to_rank, forest)
        )
    return root, coord_to_rank, rank_to_coord, flags, forest


def relation_forest_arguments(root, coord_to_rank, forest):
    return (
        root,
        coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
    )


def relation_arguments(root, coord_to_rank, forest, leaf_ids, directions):
    return (
        *relation_forest_arguments(root, coord_to_rank, forest),
        leaf_ids,
        directions,
    )


def leaf_boxes(forest):
    max_level = int(forest.node_levels[forest.leaf_node_ids].max())
    lowers: list[tuple[int, int, int]] = []
    uppers: list[tuple[int, int, int]] = []
    for node in forest.leaf_node_ids:
        level = int(forest.node_levels[node])
        scale = 1 << (max_level - level)
        lower = tuple(int(value) * scale for value in forest.node_coords[node])
        lowers.append(lower)
        uppers.append(tuple(value + scale for value in lower))
    return lowers, uppers


def balanced_relations_reference(root_shape, forest, leaf_ids, directions):
    """Independent common-lattice reference without tree lookup or maps."""
    primary_count = leaf_ids.size
    direction_count = directions.shape[0]
    kinds = np.empty((primary_count, direction_count), dtype=np.uint8)
    masks = np.zeros((primary_count, direction_count), dtype=np.uint8)
    counts = np.empty((primary_count, direction_count), dtype=np.uint8)
    sources = np.full((primary_count, direction_count, 4), -1, dtype=np.int64)
    lowers, uppers = leaf_boxes(forest)

    for primary, leaf_id_value in enumerate(leaf_ids):
        leaf_id = int(leaf_id_value)
        source_node = int(forest.leaf_node_ids[leaf_id])
        source_level = int(forest.node_levels[source_node])
        source_coord = tuple(
            int(value) for value in forest.node_coords[source_node]
        )
        extent = tuple(
            int(root_shape[axis]) * (1 << (source_level - 1))
            for axis in range(3)
        )
        for direction_index, direction_value in enumerate(directions):
            direction = tuple(int(value) for value in direction_value)
            mask = 0
            reduced = list(direction)
            for axis, delta in enumerate(direction):
                target = source_coord[axis] + delta
                if delta and (target < 0 or target >= extent[axis]):
                    mask |= 1 << axis
                    reduced[axis] = 0
            masks[primary, direction_index] = mask
            if reduced == [0, 0, 0]:
                kinds[primary, direction_index] = PHYSICAL
                counts[primary, direction_index] = 0
                continue

            source_lower = lowers[leaf_id]
            source_upper = uppers[leaf_id]
            source_scale = source_upper[0] - source_lower[0]
            target_lower = tuple(
                source_lower[axis] + reduced[axis] * source_scale
                for axis in range(3)
            )
            phase_lower = list(target_lower)
            phase_upper = [value + source_scale for value in target_lower]
            if source_scale > 1:
                for axis, delta in enumerate(reduced):
                    midpoint = target_lower[axis] + source_scale // 2
                    if delta < 0:
                        phase_lower[axis] = midpoint
                    elif delta > 0:
                        phase_upper[axis] = midpoint

            touching: list[int] = []
            for candidate in range(forest.leaf_node_ids.size):
                candidate_lower = lowers[candidate]
                candidate_upper = uppers[candidate]
                if all(
                    max(candidate_lower[axis], phase_lower[axis])
                    < min(candidate_upper[axis], phase_upper[axis])
                    for axis in range(3)
                ):
                    touching.append(candidate)

            touching.sort()
            assert 1 <= len(touching) <= 4
            target_levels = [
                int(forest.node_levels[forest.leaf_node_ids[value]])
                for value in touching
            ]
            if len(touching) == 1 and target_levels[0] == source_level - 1:
                kind = COARSER
            elif len(touching) == 1 and target_levels[0] == source_level:
                kind = SAME
            else:
                assert all(level == source_level + 1 for level in target_levels)
                kind = FINER
            count = len(touching)
            kinds[primary, direction_index] = kind
            counts[primary, direction_index] = count
            sources[primary, direction_index, :count] = touching
    return kinds, masks, counts, sources


def assert_relations_equal(actual, expected) -> None:
    assert len(actual) == len(expected) == 4
    for actual_value, expected_value in zip(actual, expected, strict=True):
        assert np.array_equal(actual_value, expected_value)


def fresh_outputs(primary_count: int, direction_count: int):
    return (
        np.full((primary_count, direction_count), 0xA1, dtype=np.uint8),
        np.full((primary_count, direction_count), 0xB2, dtype=np.uint8),
        np.full((primary_count, direction_count), 0xC3, dtype=np.uint8),
        np.full(
            (primary_count, direction_count, 4),
            -77,
            dtype=np.int64,
        ),
    )


def assert_unchanged(actual, before) -> None:
    for actual_value, before_value in zip(actual, before, strict=True):
        assert np.array_equal(actual_value, before_value)


def test_level1_relations_reduce_exactly_to_hpl_global_sources() -> None:
    root, coord_to_rank, rank_to_coord, _, forest = artifact(
        (3, 2, 1), lambda level, coord: False
    )
    leaf_ids = np.arange(forest.leaf_node_ids.size, dtype=np.int64)
    actual = balanced_refined_relations(
        *relation_arguments(
            root, coord_to_rank, forest, leaf_ids, ALL_DIRECTIONS
        )
    )
    kinds, masks, counts, sources = actual

    faces = level1_face_neighbors(root, coord_to_rank, rank_to_coord)
    hpl_sources, hpl_masks = level1_halo_relation_plan(
        leaf_ids,
        leaf_ids.size,
        faces,
    )
    assert np.array_equal(masks, hpl_masks[:, ALL_COLUMNS])
    for primary, direction_index in np.ndindex(kinds.shape):
        source = int(hpl_sources[primary, ALL_COLUMNS[direction_index]])
        if source < 0:
            assert kinds[primary, direction_index] == PHYSICAL
            assert counts[primary, direction_index] == 0
            assert np.all(sources[primary, direction_index] == -1)
        else:
            assert kinds[primary, direction_index] == SAME
            assert counts[primary, direction_index] == 1
            assert sources[primary, direction_index].tolist() == [source, -1, -1, -1]


@pytest.mark.parametrize("direction", [tuple(row) for row in ALL_DIRECTIONS])
def test_every_direction_has_exact_finer_count_and_child_order(direction) -> None:
    shape = tuple(2 if delta else 1 for delta in direction)
    source_coord = tuple(
        0 if delta >= 0 else 1 for delta in direction
    )
    target_coord = tuple(
        source_coord[axis] + direction[axis] for axis in range(3)
    )
    root, coord_to_rank, _, _, forest = artifact(
        shape,
        lambda level, coord: level == 1 and coord == target_coord,
    )
    source_node = int(
        forest.root_node_ids[int(coord_to_rank[source_coord])]
    )
    source_leaf = int(forest.node_leaf_ids[source_node])
    target_node = int(
        forest.root_node_ids[int(coord_to_rank[target_coord])]
    )
    allowed_columns = [
        child
        for child in range(8)
        if all(
            direction[axis] == 0
            or ((child >> axis) & 1) == (1 if direction[axis] < 0 else 0)
            for axis in range(3)
        )
    ]
    expected_sources = [
        int(forest.node_leaf_ids[forest.child_node_ids[target_node, child]])
        for child in allowed_columns
    ]

    kinds, masks, counts, sources = balanced_refined_relations(
        *relation_arguments(
            root,
            coord_to_rank,
            forest,
            i3(source_leaf),
            np.asarray([direction], dtype=np.int64),
        )
    )
    assert kinds[0, 0] == FINER
    assert masks[0, 0] == 0
    assert counts[0, 0] == len(allowed_columns)
    assert sources[0, 0, : len(allowed_columns)].tolist() == expected_sources
    assert np.all(sources[0, 0, len(allowed_columns) :] == -1)


def test_all_four_kinds_have_exact_counts_and_sentinels() -> None:
    root, coord_to_rank, _, _, forest = artifact(
        (2, 1, 1),
        lambda level, coord: level == 1 and coord == (0, 0, 0),
    )
    refined_root = int(forest.root_node_ids[int(coord_to_rank[0, 0, 0])])
    coarse_node = int(forest.root_node_ids[int(coord_to_rank[1, 0, 0])])
    coarse_leaf = int(forest.node_leaf_ids[coarse_node])
    child_zero = int(
        forest.node_leaf_ids[forest.child_node_ids[refined_root, 0]]
    )
    child_one = int(
        forest.node_leaf_ids[forest.child_node_ids[refined_root, 1]]
    )
    leaf_ids = i3(coarse_leaf, child_zero, child_one)
    directions = np.asarray([[-1, 0, 0], [1, 0, 0]], dtype=np.int64)
    actual = balanced_refined_relations(
        *relation_arguments(root, coord_to_rank, forest, leaf_ids, directions)
    )
    expected = balanced_relations_reference(root, forest, leaf_ids, directions)
    assert_relations_equal(actual, expected)
    kinds, _, counts, sources = actual
    assert set(int(value) for value in kinds.ravel()) == {1, 2, 3, 4}
    assert np.all(counts[kinds == PHYSICAL] == 0)
    assert np.all(sources[kinds == PHYSICAL] == -1)
    assert np.all(counts[(kinds == COARSER) | (kinds == SAME)] == 1)
    assert np.all(counts[kinds == FINER] == 4)
    for primary, direction_index in np.ndindex(counts.shape):
        count = int(counts[primary, direction_index])
        assert np.all(sources[primary, direction_index, count:] == -1)


def test_mixed_physical_finer_relation_admits_both_bits_on_masked_axes() -> None:
    root, coord_to_rank, _, _, forest = artifact(
        (1, 2, 1),
        lambda level, coord: level == 1 and coord == (0, 1, 0),
    )
    source_node = int(forest.root_node_ids[int(coord_to_rank[0, 0, 0])])
    source_leaf = int(forest.node_leaf_ids[source_node])
    target_node = int(forest.root_node_ids[int(coord_to_rank[0, 1, 0])])
    expected_columns = [0, 1, 4, 5]
    expected_sources = [
        int(forest.node_leaf_ids[forest.child_node_ids[target_node, child]])
        for child in expected_columns
    ]
    directions = np.asarray([[-1, 1, -1]], dtype=np.int64)

    actual = balanced_refined_relations(
        *relation_arguments(
            root, coord_to_rank, forest, i3(source_leaf), directions
        )
    )
    expected = balanced_relations_reference(
        root, forest, i3(source_leaf), directions
    )
    assert_relations_equal(actual, expected)
    kinds, masks, counts, sources = actual
    assert kinds[0, 0] == FINER
    assert masks[0, 0] == 0b101
    assert counts[0, 0] == 4
    assert sources[0, 0].tolist() == expected_sources


def test_random_balanced_forests_match_common_lattice_reference() -> None:
    rng = np.random.default_rng(20260828)
    accepted = 0
    for _ in range(100):
        shape = tuple(int(value) for value in rng.integers(1, 4, size=3))
        decisions: dict[tuple[int, tuple[int, int, int]], bool] = {}

        def refine(level, coord):
            key = (level, coord)
            if key not in decisions:
                decisions[key] = level < 3 and bool(rng.random() < 0.25)
            return decisions[key]

        root, coord_to_rank, _, _, forest = artifact(
            shape, refine, require_balance=False
        )
        if not is_refined_all_touch_2to1_reference(
            forest.node_levels,
            forest.node_coords,
            forest.leaf_node_ids,
        ):
            continue
        validate_refined_all_touch_2to1(
            *relation_forest_arguments(root, coord_to_rank, forest)
        )
        selected_count = min(12, forest.leaf_node_ids.size)
        leaf_ids = rng.integers(
            0,
            forest.leaf_node_ids.size,
            size=selected_count,
            dtype=np.int64,
        )
        directions = ALL_DIRECTIONS[rng.permutation(26)].copy()
        actual = balanced_refined_relations(
            *relation_arguments(
                root, coord_to_rank, forest, leaf_ids, directions
            )
        )
        expected = balanced_relations_reference(
            root, forest, leaf_ids, directions
        )
        assert_relations_equal(actual, expected)
        accepted += 1
        if accepted == 20:
            break
    assert accepted == 20


def test_unmasked_relations_are_reciprocal_across_all_kinds() -> None:
    refined_roots = {(0, 0, 0), (1, 1, 1), (2, 0, 1)}
    root, coord_to_rank, _, _, forest = artifact(
        (3, 2, 2),
        lambda level, coord: level == 1 and coord in refined_roots,
    )
    leaf_ids = np.arange(forest.leaf_node_ids.size, dtype=np.int64)
    kinds, masks, counts, sources = balanced_refined_relations(
        *relation_arguments(
            root, coord_to_rank, forest, leaf_ids, ALL_DIRECTIONS
        )
    )
    reverse_index = {
        tuple(int(value) for value in direction): index
        for index, direction in enumerate(ALL_DIRECTIONS)
    }
    seen = set()
    for leaf_id, direction_index in np.ndindex(kinds.shape):
        if masks[leaf_id, direction_index] != 0:
            continue
        kind = int(kinds[leaf_id, direction_index])
        direction = tuple(
            int(value) for value in ALL_DIRECTIONS[direction_index]
        )
        reverse = reverse_index[tuple(-value for value in direction)]
        for source in sources[
            leaf_id, direction_index, : int(counts[leaf_id, direction_index])
        ]:
            source_id = int(source)
            reverse_kind = int(kinds[source_id, reverse])
            reverse_sources = sources[
                source_id, reverse, : int(counts[source_id, reverse])
            ]
            if kind == int(SAME):
                assert reverse_kind == int(SAME)
                assert reverse_sources.tolist() == [leaf_id]
            elif kind == int(COARSER):
                reciprocal_finer = [
                    candidate_direction
                    for candidate_direction in range(26)
                    if masks[source_id, candidate_direction] == 0
                    and kinds[source_id, candidate_direction] == FINER
                    and leaf_id
                    in sources[
                        source_id,
                        candidate_direction,
                        : int(counts[source_id, candidate_direction]),
                    ]
                ]
                assert reciprocal_finer
            else:
                assert kind == int(FINER)
                assert reverse_kind == int(COARSER)
                assert reverse_sources.tolist() == [leaf_id]
            seen.add(kind)
    assert seen == {2, 3, 4}


def test_reordered_repeated_selections_and_fill_match_reference() -> None:
    root, coord_to_rank, _, _, forest = artifact(
        (2, 2, 2),
        lambda level, coord: level == 1 and coord == (0, 0, 0),
    )
    leaf_ids = i3(forest.leaf_node_ids.size - 1, 0, 3, 0, 3)
    directions = np.asarray(
        [[1, 0, 0], [-1, 1, 0], [1, 0, 0], [0, 0, -1]],
        dtype=np.int64,
    )
    before_inputs = leaf_ids.copy(), directions.copy()
    outputs = fresh_outputs(leaf_ids.size, directions.shape[0])
    fill_balanced_refined_relations(
        *relation_arguments(root, coord_to_rank, forest, leaf_ids, directions),
        *outputs,
    )
    allocating = balanced_refined_relations(
        *relation_arguments(root, coord_to_rank, forest, leaf_ids, directions)
    )
    expected = balanced_relations_reference(
        root, forest, leaf_ids, directions
    )
    assert_relations_equal(outputs, expected)
    assert_relations_equal(allocating, expected)
    assert np.array_equal(leaf_ids, before_inputs[0])
    assert np.array_equal(directions, before_inputs[1])


@pytest.mark.parametrize(("primary_count", "directions"), [
    (0, FACE_DIRECTIONS),
    (2, np.empty((0, 3), dtype=np.int64)),
])
def test_empty_selection_axes_are_valid(primary_count, directions) -> None:
    root, coord_to_rank, _, _, forest = artifact(
        (2, 1, 1), lambda level, coord: False
    )
    leaf_ids = np.arange(primary_count, dtype=np.int64)
    expected_shapes = (
        (primary_count, directions.shape[0]),
        (primary_count, directions.shape[0]),
        (primary_count, directions.shape[0]),
        (primary_count, directions.shape[0], 4),
    )
    actual = balanced_refined_relations(
        *relation_arguments(root, coord_to_rank, forest, leaf_ids, directions)
    )
    assert tuple(value.shape for value in actual) == expected_shapes
    assert tuple(value.dtype for value in actual) == (
        np.dtype(np.uint8),
        np.dtype(np.uint8),
        np.dtype(np.uint8),
        np.dtype(np.int64),
    )
    outputs = fresh_outputs(primary_count, directions.shape[0])
    fill_balanced_refined_relations(
        *relation_arguments(root, coord_to_rank, forest, leaf_ids, directions),
        *outputs,
    )
    assert_relations_equal(outputs, actual)


def test_maximum_level_63_masks_and_relations_are_exact() -> None:
    root, coord_to_rank, _, _, forest = artifact(
        (1, 1, 1),
        lambda level, coord: level <= 62 and coord == (0, 0, 0),
    )
    assert forest.max_level == 63
    source_nodes = np.flatnonzero(
        (forest.node_levels == 63)
        & np.all(forest.node_coords == np.asarray([0, 0, 0]), axis=1)
    )
    source_leaf = int(forest.node_leaf_ids[int(source_nodes[0])])
    directions = np.asarray(
        [[1, 0, 0], [-1, 0, 0], [-1, 1, 0], [-1, -1, -1]],
        dtype=np.int64,
    )
    actual = balanced_refined_relations(
        *relation_arguments(
            root, coord_to_rank, forest, i3(source_leaf), directions
        )
    )
    expected = balanced_relations_reference(
        root, forest, i3(source_leaf), directions
    )
    assert_relations_equal(actual, expected)
    kinds, masks, counts, sources = actual
    assert masks[0].tolist() == [0, 1, 1, 7]
    assert kinds[0].tolist() == [3, 1, 3, 1]
    assert counts[0].tolist() == [1, 0, 1, 0]
    assert sources[0, 0, 0] >= 0
    assert sources[0, 2, 0] >= 0


def test_invalid_selectors_directions_and_outputs_are_atomic() -> None:
    root, coord_to_rank, _, _, forest = artifact(
        (2, 2, 1), lambda level, coord: False
    )
    leaf_ids = i3(0, 1)
    directions = np.asarray([[-1, 0, 0], [0, 1, 0]], dtype=np.int64)

    def check_failure(error, match, selected, selected_directions, outputs):
        before = tuple(value.copy() for value in outputs)
        with pytest.raises(error, match=match):
            fill_balanced_refined_relations(
                *relation_arguments(
                    root,
                    coord_to_rank,
                    forest,
                    selected,
                    selected_directions,
                ),
                *outputs,
            )
        assert_unchanged(outputs, before)

    bad_leaf_ids = leaf_ids.copy()
    bad_leaf_ids[-1] = forest.leaf_node_ids.size
    check_failure(
        ValueError,
        "out of range",
        bad_leaf_ids,
        directions,
        fresh_outputs(2, 2),
    )
    for bad_row in ([0, 0, 0], [2, 0, 0]):
        bad_directions = directions.copy()
        bad_directions[-1] = bad_row
        check_failure(
            ValueError,
            "noncenter",
            leaf_ids,
            bad_directions,
            fresh_outputs(2, 2),
        )

    wrong_dtype = list(fresh_outputs(2, 2))
    wrong_dtype[0] = wrong_dtype[0].astype(np.int64)
    check_failure(
        TypeError, "uint8", leaf_ids, directions, tuple(wrong_dtype)
    )
    wrong_shape = list(fresh_outputs(2, 2))
    wrong_shape[3] = np.full((2, 2, 3), -77, dtype=np.int64)
    check_failure(
        ValueError, "shape", leaf_ids, directions, tuple(wrong_shape)
    )
    readonly = list(fresh_outputs(2, 2))
    readonly[1].setflags(write=False)
    check_failure(
        ValueError, "writable", leaf_ids, directions, tuple(readonly)
    )

    shared = np.full((2, 2), 0xA1, dtype=np.uint8)
    overlapping = (
        shared,
        shared,
        np.full((2, 2), 0xC3, dtype=np.uint8),
        np.full((2, 2, 4), -77, dtype=np.int64),
    )
    check_failure(
        ValueError, "overlap", leaf_ids, directions, overlapping
    )

    alias_leaf_ids = i3(0)
    alias_directions = np.asarray([[1, 0, 0]], dtype=np.int64)
    alias_outputs = list(fresh_outputs(1, 1))
    alias_outputs[3] = forest.child_node_ids.ravel()[:4].reshape(1, 1, 4)
    before_forest = forest.child_node_ids.copy()
    check_failure(
        ValueError,
        "overlap",
        alias_leaf_ids,
        alias_directions,
        tuple(alias_outputs),
    )
    assert np.array_equal(forest.child_node_ids, before_forest)


def test_unbalanced_forest_is_rejected_by_required_upstream_lifecycle() -> None:
    root, coord_to_rank, _, _, forest = artifact(
        (2, 1, 1),
        lambda level, coord: (
            level == 1 and coord == (0, 0, 0)
        )
        or (level == 2 and coord == (1, 0, 0)),
        require_balance=False,
    )
    with pytest.raises(ValueError, match="balance violation"):
        validate_refined_all_touch_2to1(
            *relation_forest_arguments(root, coord_to_rank, forest)
        )


def current_fine_positions(direction) -> list[int]:
    positions = []
    for child in range(8):
        bits = tuple((child >> axis) & 1 for axis in range(3))
        if not all(
            direction[axis] == 0
            or bits[axis] == (1 if direction[axis] < 0 else 0)
            for axis in range(3)
        ):
            continue
        shell = tuple(
            0
            if direction[axis] < 0
            else 3
            if direction[axis] > 0
            else 1 + bits[axis]
            for axis in range(3)
        )
        positions.append(shell[0] + 4 * shell[1] + 16 * shell[2])
    return positions


def test_current_supported_records_match_and_mixed_records_preserve_more() -> None:
    root_shape = (2, 2, 2)
    root, coord_to_rank, _, flags, forest = artifact(
        root_shape,
        lambda level, coord: level == 1 and coord == (0, 0, 0),
    )
    leaf_ids = np.arange(forest.leaf_node_ids.size, dtype=np.int64)
    kinds, masks, counts, sources = balanced_refined_relations(
        *relation_arguments(
            root, coord_to_rank, forest, leaf_ids, ALL_DIRECTIONS
        )
    )
    current = AMRForest(3, *root_shape, flags.astype(np.int32))
    current_types = np.asarray(current.neighbor_type)
    current_ids = np.asarray(current.neighbor_index)
    current_children = np.asarray(current.neighbor_children)
    mixed_nonphysical = 0
    mixed_finer = 0

    for leaf_id, direction_index in np.ndindex(kinds.shape):
        direction = tuple(
            int(value) for value in ALL_DIRECTIONS[direction_index]
        )
        column = int(ALL_COLUMNS[direction_index])
        kind = int(kinds[leaf_id, direction_index])
        count = int(counts[leaf_id, direction_index])
        if kind == int(PHYSICAL):
            assert current_types[leaf_id, column] == PHYSICAL
            assert current_ids[leaf_id, column] == 0
        elif masks[leaf_id, direction_index] == 0:
            assert current_types[leaf_id, column] == kind
            if kind in (int(COARSER), int(SAME)):
                assert current_ids[leaf_id, column] - 1 == sources[
                    leaf_id, direction_index, 0
                ]
            else:
                positions = current_fine_positions(direction)
                current_sources = current_children[leaf_id, positions] - 1
                assert current_sources.tolist() == sources[
                    leaf_id, direction_index, :count
                ].tolist()
        else:
            assert current_types[leaf_id, column] == PHYSICAL
            assert current_ids[leaf_id, column] == 0
            mixed_nonphysical += 1
            mixed_finer += kind == int(FINER)
    assert mixed_nonphysical > 0
    assert mixed_finer > 0


def test_representative_real_face_kind_counts_when_available() -> None:
    path = Path(__file__).resolve().parents[2] / "data/weno509_sub_0000.dat"
    if not path.exists():
        pytest.skip("representative refined AMRVAC evidence file is unavailable")
    from simesh.amrvac.datio import get_metadata

    header, flags_input, _ = get_metadata(str(path))
    root = np.ascontiguousarray(
        header["domain_nx"] // header["block_nx"], dtype=np.int64
    )
    coord_to_rank, rank_to_coord = level1_morton(root)
    forest = refined_forest(
        root,
        coord_to_rank,
        rank_to_coord,
        np.ascontiguousarray(flags_input, dtype=bool),
    )
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
        *relation_forest_arguments(root, coord_to_rank, forest)
    )
    leaf_ids = np.arange(forest.leaf_node_ids.size, dtype=np.int64)
    kinds, masks, counts, sources = balanced_refined_relations(
        *relation_arguments(
            root, coord_to_rank, forest, leaf_ids, FACE_DIRECTIONS
        )
    )
    values, frequencies = np.unique(kinds, return_counts=True)
    assert dict(zip(values.tolist(), frequencies.tolist(), strict=True)) == {
        1: 3430,
        2: 11488,
        3: 117894,
        4: 2872,
    }
    assert np.all(masks[kinds != PHYSICAL] == 0)
    assert np.all(counts[kinds == PHYSICAL] == 0)
    assert np.all(counts[(kinds == COARSER) | (kinds == SAME)] == 1)
    assert np.all(counts[kinds == FINER] == 4)
    assert np.all(sources[kinds == PHYSICAL] == -1)

    current = AMRForest(
        3,
        *tuple(int(value) for value in root),
        np.ascontiguousarray(flags_input, dtype=np.int32),
    )
    assert np.array_equal(
        kinds,
        np.asarray(current.neighbor_type)[:, CURRENT_FACE_COLUMNS],
    )
