from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from simesh_rewrite.forest import RefinedForest, refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.forest_conformance_reference import (
    validate_refined_forest_arrays_reference,
)
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


def conformance_arguments(
    root_shape: np.ndarray,
    coord_to_rank: np.ndarray,
    rank_to_coord: np.ndarray,
    forest: RefinedForest,
) -> tuple[np.ndarray, ...]:
    return (
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


def mixed_artifact():
    root_shape = i3(3, 2, 2)
    flags = make_flags(
        root_shape,
        lambda level, coord: (
            level == 1 and coord in {(0, 0, 0), (2, 1, 1)}
        )
        or (level == 2 and coord in {(0, 0, 0), (5, 3, 3)}),
    )
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    forest = refined_forest(
        root_shape,
        coord_to_rank,
        rank_to_coord,
        flags,
    )
    return root_shape, coord_to_rank, rank_to_coord, forest


def manual_chain_artifact(internal_levels: int):
    levels: list[int] = []
    coords: list[tuple[int, int, int]] = []
    parents: list[int] = []
    children: list[list[int]] = []
    node_to_leaf: list[int] = []
    leaf_to_node: list[int] = []

    def visit(
        level: int,
        coord: tuple[int, int, int],
        parent: int,
        remaining: int,
    ) -> int:
        node = len(levels)
        levels.append(level)
        coords.append(coord)
        parents.append(parent)
        children.append([-1] * 8)
        node_to_leaf.append(-1)
        if remaining == 0:
            node_to_leaf[node] = len(leaf_to_node)
            leaf_to_node.append(node)
            return node
        for child in range(8):
            bits = (child & 1, (child >> 1) & 1, (child >> 2) & 1)
            children[node][child] = len(levels)
            visit(
                level + 1,
                tuple(2 * coord[axis] + bits[axis] for axis in range(3)),
                node,
                remaining - 1 if child == 0 else 0,
            )
        return node

    visit(1, (0, 0, 0), -1, internal_levels)
    root_shape = i3(1, 1, 1)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    forest = RefinedForest(
        np.asarray(levels, dtype=np.int64),
        np.asarray(coords, dtype=np.int64),
        np.asarray(parents, dtype=np.int64),
        np.asarray(children, dtype=np.int64),
        np.asarray(node_to_leaf, dtype=np.int64),
        np.asarray(leaf_to_node, dtype=np.int64),
        i3(0),
        max(levels),
    )
    return root_shape, coord_to_rank, rank_to_coord, forest


@pytest.mark.parametrize(
    "case",
    [
        "level1",
        "mixed",
        "deep",
    ],
)
def test_valid_artifacts_match_independent_reference(case: str) -> None:
    if case == "level1":
        root_shape = i3(3, 2, 1)
        coord_to_rank, rank_to_coord = level1_morton(root_shape)
        forest = refined_forest(
            root_shape,
            coord_to_rank,
            rank_to_coord,
            np.ones(6, dtype=bool),
        )
    elif case == "mixed":
        root_shape, coord_to_rank, rank_to_coord, forest = mixed_artifact()
    else:
        root_shape, coord_to_rank, rank_to_coord, forest = manual_chain_artifact(20)
    actual = validate_refined_forest_arrays(
        *conformance_arguments(root_shape, coord_to_rank, rank_to_coord, forest)
    )
    expected = validate_refined_forest_arrays_reference(
        root_shape,
        rank_to_coord,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.parent_node_ids,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
    )
    assert actual == expected == forest.max_level


@pytest.mark.parametrize(
    ("argument_index", "mutate"),
    [
        (1, lambda value: value.__setitem__((0, 0, 0), value.size - 1)),
        (2, lambda value: value.__setitem__(0, value[1])),
        (3, lambda value: value.__setitem__(0, value[0] + 1)),
        (4, lambda value: value.__setitem__(-1, value[-1] + 1)),
        (5, lambda value: value.__setitem__((-1, 0), value[-1, 0] + 1)),
        (6, lambda value: value.__setitem__(-1, -1)),
        (7, lambda value: value.__setitem__((0, 0), value[0, 0] + 1)),
        (8, lambda value: value.__setitem__(0, -2)),
        (9, lambda value: value.__setitem__(-1, 0)),
    ],
)
def test_each_artifact_family_corruption_is_rejected(
    argument_index: int,
    mutate,
) -> None:
    root_shape, coord_to_rank, rank_to_coord, forest = mixed_artifact()
    arguments = list(
        conformance_arguments(root_shape, coord_to_rank, rank_to_coord, forest)
    )
    arguments[argument_index] = arguments[argument_index].copy()
    mutate(arguments[argument_index])
    with pytest.raises(ValueError):
        validate_refined_forest_arrays(*arguments)


def test_leaf_child_and_internal_leaf_sentinels_are_exact() -> None:
    root_shape, coord_to_rank, rank_to_coord, forest = mixed_artifact()
    arguments = list(
        conformance_arguments(root_shape, coord_to_rank, rank_to_coord, forest)
    )
    leaf_node = int(forest.leaf_node_ids[0])
    children = forest.child_node_ids.copy()
    children[leaf_node, 7] = leaf_node
    arguments[7] = children
    with pytest.raises(ValueError, match="non-sentinel child"):
        validate_refined_forest_arrays(*arguments)

    arguments = list(
        conformance_arguments(root_shape, coord_to_rank, rank_to_coord, forest)
    )
    node_leaf = forest.node_leaf_ids.copy()
    internal_node = int(np.flatnonzero(node_leaf == -1)[0])
    node_leaf[internal_node] = -2
    arguments[8] = node_leaf
    with pytest.raises(ValueError, match="invalid leaf sentinel"):
        validate_refined_forest_arrays(*arguments)


def test_trailing_nodes_and_leaves_are_rejected() -> None:
    root_shape, coord_to_rank, rank_to_coord, forest = mixed_artifact()
    trailing = RefinedForest(
        np.r_[forest.node_levels, 1],
        np.vstack([forest.node_coords, [0, 0, 0]]),
        np.r_[forest.parent_node_ids, -1],
        np.vstack([forest.child_node_ids, np.full(8, -1, dtype=np.int64)]),
        np.r_[forest.node_leaf_ids, -1],
        forest.leaf_node_ids,
        forest.root_node_ids,
        forest.max_level,
    )
    with pytest.raises(ValueError, match="trailing nodes"):
        validate_refined_forest_arrays(
            *conformance_arguments(
                root_shape,
                coord_to_rank,
                rank_to_coord,
                trailing,
            )
        )

    trailing_leaf = forest._replace(
        leaf_node_ids=np.r_[forest.leaf_node_ids, forest.leaf_node_ids[-1]]
    )
    with pytest.raises(ValueError, match="trailing leaves"):
        validate_refined_forest_arrays(
            *conformance_arguments(
                root_shape,
                coord_to_rank,
                rank_to_coord,
                trailing_leaf,
            )
        )


def test_maximum_representable_depth_and_extent_overflow() -> None:
    valid = manual_chain_artifact(62)
    assert validate_refined_forest_arrays(
        *conformance_arguments(*valid)
    ) == 63
    overflow = manual_chain_artifact(63)
    with pytest.raises(OverflowError, match="logical grid"):
        validate_refined_forest_arrays(
            *conformance_arguments(*overflow)
        )


def test_read_only_overlap_and_lifecycle_revalidation() -> None:
    root_shape = i3(1, 1, 1)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    shared_leaf_map = i3(0)
    forest = RefinedForest(
        i3(1),
        np.asarray([[0, 0, 0]], dtype=np.int64),
        i3(-1),
        np.full((1, 8), -1, dtype=np.int64),
        shared_leaf_map,
        shared_leaf_map,
        i3(0),
        1,
    )
    arguments = conformance_arguments(
        root_shape,
        coord_to_rank,
        rank_to_coord,
        forest,
    )
    for value in arguments:
        value.setflags(write=False)
    assert validate_refined_forest_arrays(*arguments) == 1

    root_shape, coord_to_rank, rank_to_coord, forest = mixed_artifact()
    mutable = list(
        conformance_arguments(root_shape, coord_to_rank, rank_to_coord, forest)
    )
    assert validate_refined_forest_arrays(*mutable) == forest.max_level
    mutable[4][-1] += 1
    with pytest.raises(ValueError, match="metadata"):
        validate_refined_forest_arrays(*mutable)
    mutable[4][-1] -= 1
    assert validate_refined_forest_arrays(*mutable) == forest.max_level


def test_consistent_non_morton_permutation_is_explicit_upstream_precondition() -> None:
    root_shape = i3(2, 1, 1)
    _, morton_inverse = level1_morton(root_shape)
    rank_to_coord = morton_inverse[::-1].copy()
    coord_to_rank = np.empty((2, 1, 1), dtype=np.int64)
    for rank, coord in enumerate(rank_to_coord):
        coord_to_rank[tuple(coord)] = rank
    forest = refined_forest(
        root_shape,
        coord_to_rank,
        rank_to_coord,
        np.ones(2, dtype=bool),
    )
    assert validate_refined_forest_arrays(
        *conformance_arguments(root_shape, coord_to_rank, rank_to_coord, forest)
    ) == 1


def test_invalid_layout_dtype_shape_and_empty_artifacts_are_explicit() -> None:
    root_shape, coord_to_rank, rank_to_coord, forest = mixed_artifact()
    arguments = list(
        conformance_arguments(root_shape, coord_to_rank, rank_to_coord, forest)
    )
    wrong_dtype = arguments[4].astype(np.int32)
    arguments[4] = wrong_dtype
    with pytest.raises(TypeError, match="int64"):
        validate_refined_forest_arrays(*arguments)

    arguments = list(
        conformance_arguments(root_shape, coord_to_rank, rank_to_coord, forest)
    )
    arguments[5] = np.empty((forest.node_levels.size, 6), dtype=np.int64)[:, ::2]
    with pytest.raises(ValueError, match="C-contiguous"):
        validate_refined_forest_arrays(*arguments)

    arguments = list(
        conformance_arguments(root_shape, coord_to_rank, rank_to_coord, forest)
    )
    arguments[3] = arguments[3][:-1].copy()
    with pytest.raises(ValueError, match="shape"):
        validate_refined_forest_arrays(*arguments)

    empty = (
        root_shape,
        coord_to_rank,
        rank_to_coord,
        forest.root_node_ids,
        np.empty(0, dtype=np.int64),
        np.empty((0, 3), dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty((0, 8), dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
    )
    with pytest.raises(ValueError, match="at least one"):
        validate_refined_forest_arrays(*empty)


def test_representative_refined_dat_artifact_when_available() -> None:
    path = Path(__file__).resolve().parents[2] / "data/weno509_sub_0000.dat"
    if not path.exists():
        pytest.skip("representative refined AMRVAC evidence file is unavailable")
    from simesh.amrvac.datio import get_metadata

    header, flags_input, _ = get_metadata(str(path))
    root_shape = np.ascontiguousarray(
        header["domain_nx"] // header["block_nx"], dtype=np.int64
    )
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    forest = refined_forest(
        root_shape,
        coord_to_rank,
        rank_to_coord,
        np.ascontiguousarray(flags_input, dtype=bool),
    )
    assert validate_refined_forest_arrays(
        *conformance_arguments(root_shape, coord_to_rank, rank_to_coord, forest)
    ) == 6
