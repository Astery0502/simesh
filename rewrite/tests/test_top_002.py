from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from simesh.utils.lib.amr.forest import AMRForest
from simesh_rewrite.contacts import (
    fill_refined_contact_targets,
    refined_contact_targets,
)
from simesh_rewrite.contacts_reference import refined_contact_targets_reference
from simesh_rewrite.forest import refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.topology import level1_face_neighbors


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
ALL_DIRECTIONS = np.asarray(
    [
        (dx, dy, dz)
        for dz in range(-1, 2)
        for dy in range(-1, 2)
        for dx in range(-1, 2)
        if (dx, dy, dz) != (0, 0, 0)
    ],
    dtype=np.int64,
)


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


def lookup_arguments(root, coord_to_rank, forest):
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


def reference(root, forest, source_leaf_ids, directions):
    return refined_contact_targets_reference(
        root,
        forest.node_levels,
        forest.node_coords,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
        source_leaf_ids,
        directions,
    )


def test_all_level1_faces_reduce_exactly_to_top001() -> None:
    root, coord_to_rank, rank_to_coord, _, forest = artifact(
        (3, 2, 1), lambda level, coord: False
    )
    leaves = np.repeat(np.arange(forest.leaf_node_ids.size, dtype=np.int64), 6)
    directions = np.tile(FACE_DIRECTIONS, (forest.leaf_node_ids.size, 1))
    actual = refined_contact_targets(
        *lookup_arguments(root, coord_to_rank, forest),
        leaves,
        directions,
    ).reshape(-1, 6)
    expected = level1_face_neighbors(root, coord_to_rank, rank_to_coord)
    assert np.array_equal(actual, expected)


def test_balanced_coarser_same_and_subdivided_face_results_are_raw_nodes() -> None:
    root, coord_to_rank, _, _, forest = artifact(
        (2, 1, 1),
        lambda level, coord: level == 1 and coord == (0, 0, 0),
    )
    coarse_leaf = int(forest.node_leaf_ids[9])
    coarse_to_fine = refined_contact_targets(
        *lookup_arguments(root, coord_to_rank, forest),
        i3(coarse_leaf),
        np.asarray([[-1, 0, 0]], dtype=np.int64),
    )
    assert coarse_to_fine.tolist() == [0]
    assert forest.node_leaf_ids[0] == -1

    fine_leaf_ids = np.asarray([1, 3, 5, 7], dtype=np.int64)
    fine_to_coarse = refined_contact_targets(
        *lookup_arguments(root, coord_to_rank, forest),
        fine_leaf_ids,
        np.tile(np.asarray([[1, 0, 0]], dtype=np.int64), (4, 1)),
    )
    assert np.array_equal(fine_to_coarse, np.full(4, 9, dtype=np.int64))
    assert forest.node_levels[9] == 1
    assert forest.node_leaf_ids[9] == coarse_leaf


def test_unbalanced_arbitrarily_coarser_contact_is_valid() -> None:
    root, coord_to_rank, _, _, forest = artifact(
        (2, 1, 1),
        lambda level, coord: (
            level == 1 and coord == (0, 0, 0)
        )
        or (level == 2 and coord == (1, 0, 0)),
    )
    candidate_nodes = np.flatnonzero(
        (forest.node_levels == 3)
        & np.all(forest.node_coords == np.asarray([3, 0, 0]), axis=1)
    )
    source_node = int(candidate_nodes[0])
    source_leaf = int(forest.node_leaf_ids[source_node])
    target = refined_contact_targets(
        *lookup_arguments(root, coord_to_rank, forest),
        i3(source_leaf),
        np.asarray([[1, 0, 0]], dtype=np.int64),
    )[0]
    assert forest.node_levels[target] == 1
    assert forest.node_leaf_ids[target] >= 0


def test_all_26_directions_match_coordinate_dictionary_reference() -> None:
    root, coord_to_rank, _, _, forest = artifact(
        (3, 2, 2),
        lambda level, coord: (
            level == 1 and coord in {(0, 0, 0), (1, 1, 1), (2, 0, 1)}
        )
        or (level == 2 and coord in {(1, 0, 0), (2, 3, 3)}),
    )
    source_ids = np.repeat(
        np.arange(forest.leaf_node_ids.size, dtype=np.int64),
        26,
    )
    directions = np.tile(ALL_DIRECTIONS, (forest.leaf_node_ids.size, 1))
    actual = refined_contact_targets(
        *lookup_arguments(root, coord_to_rank, forest),
        source_ids,
        directions,
    )
    expected = reference(root, forest, source_ids, directions)
    assert np.array_equal(actual, expected)


def test_mixed_physical_direction_is_physical_but_neutralized_query_is_not() -> None:
    root, coord_to_rank, _, _, forest = artifact(
        (2, 2, 1), lambda level, coord: False
    )
    source_node = int(coord_to_rank[0, 0, 0])
    source_leaf = int(forest.node_leaf_ids[source_node])
    directions = np.asarray([[-1, 1, 0], [0, 1, 0]], dtype=np.int64)
    targets = refined_contact_targets(
        *lookup_arguments(root, coord_to_rank, forest),
        i3(source_leaf, source_leaf),
        directions,
    )
    assert targets[0] == -1
    assert targets[1] >= 0


def test_reordered_repeated_queries_and_caller_fill_match_reference() -> None:
    root, coord_to_rank, _, _, forest = artifact(
        (2, 2, 2),
        lambda level, coord: level == 1 and coord == (0, 0, 0),
    )
    source_ids = i3(0, 5, 0, forest.leaf_node_ids.size - 1, 3)
    directions = np.asarray(
        [[1, 0, 0], [-1, 1, 0], [0, 0, 1], [0, -1, -1], [1, 1, 1]],
        dtype=np.int64,
    )
    output = np.full(source_ids.size, -77, dtype=np.int64)
    fill_refined_contact_targets(
        *lookup_arguments(root, coord_to_rank, forest),
        source_ids,
        directions,
        output,
    )
    assert np.array_equal(output, reference(root, forest, source_ids, directions))


def test_current_face_kinds_and_leaf_indices_match_on_balanced_forest() -> None:
    root_shape = (2, 2, 2)
    root, coord_to_rank, _, flags, forest = artifact(
        root_shape,
        lambda level, coord: level == 1 and coord == (0, 0, 0),
    )
    source_ids = np.repeat(
        np.arange(forest.leaf_node_ids.size, dtype=np.int64), 6
    )
    directions = np.tile(FACE_DIRECTIONS, (forest.leaf_node_ids.size, 1))
    targets = refined_contact_targets(
        *lookup_arguments(root, coord_to_rank, forest),
        source_ids,
        directions,
    ).reshape(-1, 6)
    current = AMRForest(3, *root_shape, flags.astype(np.int32))
    current_types = np.asarray(current.neighbor_type)[:, CURRENT_FACE_COLUMNS]
    current_ids = np.asarray(current.neighbor_index)[:, CURRENT_FACE_COLUMNS]
    for leaf in range(forest.leaf_node_ids.size):
        source_node = int(forest.leaf_node_ids[leaf])
        for face, target in enumerate(targets[leaf]):
            if target < 0:
                assert current_types[leaf, face] == 1
                assert current_ids[leaf, face] == 0
            elif forest.node_leaf_ids[target] >= 0:
                target_leaf = int(forest.node_leaf_ids[target])
                expected_type = (
                    2
                    if forest.node_levels[target] < forest.node_levels[source_node]
                    else 3
                )
                assert current_types[leaf, face] == expected_type
                assert current_ids[leaf, face] == target_leaf + 1
            else:
                assert current_types[leaf, face] == 4
                assert current_ids[leaf, face] == 0


def test_invalid_queries_and_outputs_are_atomic() -> None:
    root, coord_to_rank, _, _, forest = artifact(
        (2, 2, 1), lambda level, coord: False
    )
    source_ids = i3(0, 1, 2)
    directions = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64)
    output = np.full(3, -77, dtype=np.int64)
    before = output.copy()

    bad_sources = source_ids.copy()
    bad_sources[-1] = forest.leaf_node_ids.size
    with pytest.raises(ValueError, match="out of range"):
        fill_refined_contact_targets(
            *lookup_arguments(root, coord_to_rank, forest),
            bad_sources,
            directions,
            output,
        )
    bad_directions = directions.copy()
    bad_directions[-1] = [0, 0, 0]
    with pytest.raises(ValueError, match="noncenter"):
        fill_refined_contact_targets(
            *lookup_arguments(root, coord_to_rank, forest),
            source_ids,
            bad_directions,
            output,
        )
    bad_directions[-1] = [2, 0, 0]
    with pytest.raises(ValueError, match="noncenter"):
        fill_refined_contact_targets(
            *lookup_arguments(root, coord_to_rank, forest),
            source_ids,
            bad_directions,
            output,
        )
    readonly = output.copy()
    readonly.setflags(write=False)
    with pytest.raises(ValueError, match="writable"):
        fill_refined_contact_targets(
            *lookup_arguments(root, coord_to_rank, forest),
            source_ids,
            directions,
            readonly,
        )
    with pytest.raises(TypeError, match="int64"):
        fill_refined_contact_targets(
            *lookup_arguments(root, coord_to_rank, forest),
            source_ids.astype(np.int32),
            directions,
            output,
        )
    with pytest.raises(ValueError, match="shape"):
        fill_refined_contact_targets(
            *lookup_arguments(root, coord_to_rank, forest),
            source_ids,
            directions[:, :2],
            output,
        )
    alias_base = source_ids.copy()
    with pytest.raises(ValueError, match="overlap"):
        fill_refined_contact_targets(
            *lookup_arguments(root, coord_to_rank, forest),
            alias_base,
            directions,
            alias_base,
        )
    assert np.array_equal(output, before)


def test_empty_query_is_a_valid_noop() -> None:
    root, coord_to_rank, _, _, forest = artifact(
        (1, 1, 1), lambda level, coord: False
    )
    output = np.empty(0, dtype=np.int64)
    fill_refined_contact_targets(
        *lookup_arguments(root, coord_to_rank, forest),
        np.empty(0, dtype=np.int64),
        np.empty((0, 3), dtype=np.int64),
        output,
    )
    assert output.shape == (0,)


def test_maximum_level_63_shift_and_descent_match_reference() -> None:
    root, coord_to_rank, _, _, forest = artifact(
        (1, 1, 1),
        lambda level, coord: level <= 62 and coord == (0, 0, 0),
    )
    source_nodes = np.flatnonzero(
        (forest.node_levels == 63)
        & np.all(forest.node_coords == np.asarray([0, 0, 0]), axis=1)
    )
    source_leaf = int(forest.node_leaf_ids[int(source_nodes[0])])
    source_ids = i3(source_leaf, source_leaf)
    directions = np.asarray([[1, 0, 0], [-1, 0, 0]], dtype=np.int64)
    actual = refined_contact_targets(
        *lookup_arguments(root, coord_to_rank, forest),
        source_ids,
        directions,
    )
    expected = reference(root, forest, source_ids, directions)
    assert np.array_equal(actual, expected)
    assert actual[0] >= 0
    assert forest.node_levels[actual[0]] == 63
    assert actual[1] == -1


def test_representative_real_face_kinds_match_current_when_available() -> None:
    path = Path(__file__).resolve().parents[2] / "data/weno509_sub_0000.dat"
    if not path.exists():
        pytest.skip("representative refined AMRVAC evidence file is unavailable")
    from simesh.amrvac.datio import get_metadata

    header, flags_input, _ = get_metadata(str(path))
    root_shape = tuple(
        int(value) for value in header["domain_nx"] // header["block_nx"]
    )
    root, coord_to_rank, _, flags, forest = artifact(
        root_shape,
        lambda level, coord: False,
    )
    flags = np.ascontiguousarray(flags_input, dtype=bool)
    forest = refined_forest(root, *level1_morton(root), flags)
    _, rank_to_coord = level1_morton(root)
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
    source_ids = np.repeat(
        np.arange(forest.leaf_node_ids.size, dtype=np.int64), 6
    )
    directions = np.tile(FACE_DIRECTIONS, (forest.leaf_node_ids.size, 1))
    targets = refined_contact_targets(
        *lookup_arguments(root, coord_to_rank, forest),
        source_ids,
        directions,
    ).reshape(-1, 6)
    current = AMRForest(3, *root_shape, flags.astype(np.int32))
    current_types = np.asarray(current.neighbor_type)[:, CURRENT_FACE_COLUMNS]
    for leaf, face in np.ndindex(targets.shape):
        target = int(targets[leaf, face])
        source_node = int(forest.leaf_node_ids[leaf])
        if target < 0:
            kind = 1
        elif forest.node_leaf_ids[target] < 0:
            kind = 4
        elif forest.node_levels[target] < forest.node_levels[source_node]:
            kind = 2
        else:
            kind = 3
        assert current_types[leaf, face] == kind
