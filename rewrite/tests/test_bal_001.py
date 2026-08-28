from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from simesh_rewrite.balance import validate_refined_all_touch_2to1
from simesh_rewrite.balance_reference import is_refined_all_touch_2to1_reference
from simesh_rewrite.forest import refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.morton import level1_morton


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


def artifact(root_shape: tuple[int, int, int], refine):
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
    return root, coord_to_rank, rank_to_coord, flags, forest


def balance_args(root, coord_to_rank, forest):
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


def reference_balanced(forest) -> bool:
    return is_refined_all_touch_2to1_reference(
        forest.node_levels,
        forest.node_coords,
        forest.leaf_node_ids,
    )


def leaf_boxes(forest):
    max_level = int(forest.node_levels[forest.leaf_node_ids].max())
    boxes = []
    for node in forest.leaf_node_ids:
        level = int(forest.node_levels[node])
        scale = 1 << (max_level - level)
        lower = tuple(int(value) * scale for value in forest.node_coords[node])
        upper = tuple(value + scale for value in lower)
        boxes.append((lower, upper, level))
    return boxes


def face_only_balanced(forest) -> bool:
    boxes = leaf_boxes(forest)
    for left in range(len(boxes)):
        lower_left, upper_left, level_left = boxes[left]
        for right in range(left + 1, len(boxes)):
            lower_right, upper_right, level_right = boxes[right]
            touching_axes = 0
            positive_overlap = True
            for axis in range(3):
                overlap = min(upper_left[axis], upper_right[axis]) - max(
                    lower_left[axis], lower_right[axis]
                )
                if overlap < 0:
                    positive_overlap = False
                    break
                if overlap == 0:
                    touching_axes += 1
            if (
                positive_overlap
                and touching_axes == 1
                and abs(level_left - level_right) > 1
            ):
                return False
    return True


@pytest.mark.parametrize(
    ("shape", "refine"),
    [
        ((3, 2, 1), lambda level, coord: False),
        (
            (2, 2, 2),
            lambda level, coord: level == 1 and coord == (0, 0, 0),
        ),
        (
            (2, 1, 1),
            lambda level, coord: (
                level == 1 and coord == (1, 0, 0)
            )
            or (level == 2 and coord == (3, 0, 0)),
        ),
    ],
)
def test_balanced_uniform_mixed_and_away_refinement_match_reference(
    shape,
    refine,
) -> None:
    root, coord_to_rank, _, _, forest = artifact(shape, refine)
    assert reference_balanced(forest)
    assert validate_refined_all_touch_2to1(
        *balance_args(root, coord_to_rank, forest)
    ) is None


def test_face_level_gap_two_is_rejected() -> None:
    root, coord_to_rank, _, _, forest = artifact(
        (2, 1, 1),
        lambda level, coord: (
            level == 1 and coord == (0, 0, 0)
        )
        or (level == 2 and coord == (1, 0, 0)),
    )
    assert not reference_balanced(forest)
    with pytest.raises(ValueError, match="balance violation"):
        validate_refined_all_touch_2to1(
            *balance_args(root, coord_to_rank, forest)
        )


def test_face_balanced_edge_corner_gap_two_is_rejected() -> None:
    root, coord_to_rank, _, _, forest = artifact(
        (2, 2, 1),
        lambda level, coord: (
            level == 1 and coord in {(0, 0, 0), (1, 0, 0), (0, 1, 0)}
        )
        or (level == 2 and coord == (1, 1, 0)),
    )
    assert face_only_balanced(forest)
    assert not reference_balanced(forest)
    with pytest.raises(ValueError, match="direction") as error:
        validate_refined_all_touch_2to1(
            *balance_args(root, coord_to_rank, forest)
        )
    assert "source leaf" in str(error.value)
    assert "target node" in str(error.value)


def test_random_complete_forests_match_pairwise_reference() -> None:
    rng = np.random.default_rng(20260828)
    for _ in range(100):
        shape = tuple(int(value) for value in rng.integers(1, 4, size=3))
        decisions: dict[tuple[int, tuple[int, int, int]], bool] = {}

        def refine(level, coord):
            key = (level, coord)
            if key not in decisions:
                decisions[key] = level < 3 and bool(rng.random() < 0.3)
            return decisions[key]

        root, coord_to_rank, _, _, forest = artifact(shape, refine)
        expected = reference_balanced(forest)
        if expected:
            validate_refined_all_touch_2to1(
                *balance_args(root, coord_to_rank, forest)
            )
        else:
            with pytest.raises(ValueError, match="balance violation"):
                validate_refined_all_touch_2to1(
                    *balance_args(root, coord_to_rank, forest)
                )


def test_first_violation_reporting_is_deterministic() -> None:
    root, coord_to_rank, _, _, forest = artifact(
        (2, 1, 1),
        lambda level, coord: (
            level == 1 and coord == (0, 0, 0)
        )
        or (level == 2 and coord == (1, 0, 0)),
    )
    expected = (
        "all-touch two-to-one balance violation at source leaf 2, "
        "direction 14, target node 17, offending node 17"
    )
    for _ in range(2):
        with pytest.raises(ValueError, match="balance violation") as error:
            validate_refined_all_touch_2to1(
                *balance_args(root, coord_to_rank, forest)
            )
        assert str(error.value) == expected


@pytest.mark.parametrize(
    "direction",
    [
        (dx, dy, dz)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ],
)
def test_all_directions_select_only_contact_touching_child_phases(direction) -> None:
    shape = tuple(2 if delta else 1 for delta in direction)
    source = tuple(0 if delta >= 0 else 1 for delta in direction)
    target = tuple(source[axis] + direction[axis] for axis in range(3))
    touching_bits = tuple(
        0 if delta >= 0 else 1 for delta in direction
    )

    def make_case(bits):
        deep_coord = tuple(2 * target[axis] + bits[axis] for axis in range(3))
        return artifact(
            shape,
            lambda level, coord: (
                level == 1 and coord != source
            )
            or (level == 2 and coord == deep_coord),
        )

    root, coord_to_rank, _, _, touching = make_case(touching_bits)
    assert not reference_balanced(touching)
    with pytest.raises(ValueError, match="balance violation"):
        validate_refined_all_touch_2to1(
            *balance_args(root, coord_to_rank, touching)
        )

    away_bits = list(touching_bits)
    away_axis = next(axis for axis, delta in enumerate(direction) if delta)
    away_bits[away_axis] = 1 - away_bits[away_axis]
    root, coord_to_rank, _, _, away = make_case(tuple(away_bits))
    assert reference_balanced(away)
    assert validate_refined_all_touch_2to1(
        *balance_args(root, coord_to_rank, away)
    ) is None


def test_maximum_depth_artifact_is_checked_without_shift_overflow() -> None:
    root, coord_to_rank, _, _, forest = artifact(
        (1, 1, 1),
        lambda level, coord: level <= 62 and coord == (0, 0, 0),
    )
    expected = reference_balanced(forest)
    if expected:
        validate_refined_all_touch_2to1(
            *balance_args(root, coord_to_rank, forest)
        )
    else:
        with pytest.raises(ValueError, match="balance violation"):
            validate_refined_all_touch_2to1(
                *balance_args(root, coord_to_rank, forest)
            )


def test_invalid_shapes_and_dtypes_are_nonmutating() -> None:
    root, coord_to_rank, _, _, forest = artifact(
        (2, 1, 1), lambda level, coord: False
    )
    before = tuple(value.copy() for value in balance_args(root, coord_to_rank, forest))
    arguments = list(balance_args(root, coord_to_rank, forest))
    arguments[3] = arguments[3].astype(np.int32)
    with pytest.raises(TypeError, match="int64"):
        validate_refined_all_touch_2to1(*arguments)
    arguments = list(balance_args(root, coord_to_rank, forest))
    arguments[4] = np.empty((2, 6), dtype=np.int64)[:, ::2]
    with pytest.raises(ValueError, match="C-contiguous"):
        validate_refined_all_touch_2to1(*arguments)
    for actual, expected in zip(
        balance_args(root, coord_to_rank, forest), before, strict=True
    ):
        assert np.array_equal(actual, expected)


def test_representative_real_tree_is_balanced_when_available() -> None:
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
    assert validate_refined_all_touch_2to1(
        *balance_args(root, coord_to_rank, forest)
    ) is None
