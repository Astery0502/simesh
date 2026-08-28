from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from simesh.utils.lib.amr.forest import AMRForest
from simesh_rewrite.forest import fill_refined_forest, refined_forest
from simesh_rewrite.forest_reference import refined_forest_reference
from simesh_rewrite.morton import level1_morton


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def make_flags(
    root_shape: np.ndarray,
    should_refine: Callable[[int, tuple[int, int, int]], bool],
) -> np.ndarray:
    _, rank_to_coord = level1_morton(root_shape)
    flags: list[bool] = []

    def emit(level: int, coord: tuple[int, int, int]) -> None:
        refine = should_refine(level, coord)
        flags.append(not refine)
        if not refine:
            return
        for child in range(8):
            bits = (child & 1, (child >> 1) & 1, (child >> 2) & 1)
            emit(
                level + 1,
                tuple(2 * coord[axis] + bits[axis] for axis in range(3)),
            )

    for coordinate in rank_to_coord:
        emit(1, tuple(int(value) for value in coordinate))
    return np.asarray(flags, dtype=np.bool_)


def chain_flags(internal_levels: int) -> np.ndarray:
    def emit(remaining: int) -> list[bool]:
        if remaining == 0:
            return [True]
        return [False, *emit(remaining - 1), *([True] * 7)]

    return np.asarray(emit(internal_levels), dtype=np.bool_)


def output_arrays(
    node_count: int,
    leaf_count: int,
    root_count: int,
    fill: int = -77,
) -> tuple[np.ndarray, ...]:
    return (
        np.full(node_count, fill, dtype=np.int64),
        np.full((node_count, 3), fill, dtype=np.int64),
        np.full(node_count, fill, dtype=np.int64),
        np.full((node_count, 8), fill, dtype=np.int64),
        np.full(node_count, fill, dtype=np.int64),
        np.full(leaf_count, fill, dtype=np.int64),
        np.full(root_count, fill, dtype=np.int64),
    )


def assert_forest_invariants(
    root_shape: np.ndarray,
    rank_to_coord: np.ndarray,
    flags: np.ndarray,
    forest,
) -> None:
    node_count = flags.size
    leaf_count = int(np.count_nonzero(flags))
    parent_count = node_count - leaf_count
    root_count = rank_to_coord.shape[0]
    assert leaf_count == root_count + 7 * parent_count
    assert forest.node_levels.shape == (node_count,)
    assert forest.node_coords.shape == (node_count, 3)
    assert forest.parent_node_ids.shape == (node_count,)
    assert forest.child_node_ids.shape == (node_count, 8)
    assert forest.node_leaf_ids.shape == (node_count,)
    assert forest.leaf_node_ids.shape == (leaf_count,)
    assert forest.root_node_ids.shape == (root_count,)
    assert forest.max_level == int(forest.node_levels.max())
    assert np.array_equal(
        forest.node_coords[forest.root_node_ids], rank_to_coord
    )
    assert np.all(forest.node_levels[forest.root_node_ids] == 1)
    assert np.all(forest.parent_node_ids[forest.root_node_ids] == -1)
    assert np.array_equal(
        forest.node_leaf_ids[forest.leaf_node_ids],
        np.arange(leaf_count, dtype=np.int64),
    )
    assert np.array_equal(np.all(forest.child_node_ids == -1, axis=1), flags)

    root_ends = np.r_[forest.root_node_ids[1:], node_count]
    for root_node, root_end in zip(
        forest.root_node_ids, root_ends, strict=True
    ):
        assert root_node < root_end
        if root_node + 1 < root_end:
            parents = forest.parent_node_ids[root_node + 1 : root_end]
            assert np.all(parents >= root_node)
            assert np.all(parents < np.arange(root_node + 1, root_end))

    for node in range(node_count):
        if flags[node]:
            assert np.all(forest.child_node_ids[node] == -1)
            assert forest.node_leaf_ids[node] >= 0
            continue
        assert forest.node_leaf_ids[node] == -1
        children = forest.child_node_ids[node]
        assert np.all(children > node)
        assert np.all(forest.parent_node_ids[children] == node)
        assert np.all(forest.node_levels[children] == forest.node_levels[node] + 1)
        for child, child_node in enumerate(children):
            bits = np.array(
                [child & 1, (child >> 1) & 1, (child >> 2) & 1],
                dtype=np.int64,
            )
            assert np.array_equal(
                forest.node_coords[child_node],
                2 * forest.node_coords[node] + bits,
            )

    # The leaf boxes tile the complete domain when expressed at max level.
    scale = 1 << (forest.max_level - forest.node_levels[forest.leaf_node_ids])
    lower = forest.node_coords[forest.leaf_node_ids] * scale[:, None]
    upper = (forest.node_coords[forest.leaf_node_ids] + 1) * scale[:, None]
    finest_shape = root_shape * (1 << (forest.max_level - 1))
    assert np.all(lower >= 0)
    assert np.all(upper <= finest_shape)
    assert int(np.prod(upper - lower, axis=1).sum()) == int(
        np.prod(finest_shape, dtype=np.int64)
    )


def test_golden_refined_root_freezes_stream_and_child_order() -> None:
    root_shape = i3(2, 1, 1)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    flags = np.asarray([False, *([True] * 8), True], dtype=np.bool_)
    actual = refined_forest(root_shape, coord_to_rank, rank_to_coord, flags)

    assert actual.max_level == 2
    assert np.array_equal(actual.root_node_ids, [0, 9])
    assert np.array_equal(actual.child_node_ids[0], np.arange(1, 9))
    assert np.array_equal(actual.leaf_node_ids, np.arange(1, 10))
    assert np.array_equal(
        actual.node_coords,
        [
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [1, 0, 0],
        ],
    )
    assert_forest_invariants(root_shape, rank_to_coord, flags, actual)


def test_mixed_depth_compiled_matches_independent_reference() -> None:
    root_shape = i3(3, 2, 2)

    def refine(level: int, coord: tuple[int, int, int]) -> bool:
        return (level == 1 and coord in {(0, 0, 0), (2, 1, 1)}) or (
            level == 2 and coord in {(0, 0, 0), (5, 3, 3)}
        )

    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    flags = make_flags(root_shape, refine)
    actual = refined_forest(root_shape, coord_to_rank, rank_to_coord, flags)
    expected = refined_forest_reference(root_shape, rank_to_coord, flags)
    assert actual.max_level == expected.max_level == 3
    for actual_array, expected_array in zip(actual[:-1], expected[:-1], strict=True):
        assert np.array_equal(actual_array, expected_array)
    assert_forest_invariants(root_shape, rank_to_coord, flags, actual)


def test_nested_forest_matches_current_roundtrip_counts_and_level() -> None:
    root_shape = i3(2, 1, 1)
    flags = make_flags(
        root_shape,
        lambda level, coord: (level == 1 and coord == (0, 0, 0))
        or (level == 2 and coord == (0, 0, 0)),
    )
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    actual = refined_forest(root_shape, coord_to_rank, rank_to_coord, flags)
    current = AMRForest(3, 2, 1, 1, flags.astype(np.int32))

    assert np.array_equal(np.asarray(current.write_forest(), dtype=bool), flags)
    assert int(current.nleafs) == actual.leaf_node_ids.size == 16
    assert int(current.nparents) == np.count_nonzero(~flags) == 2
    assert int(current.max_level) == actual.max_level == 3
    assert np.array_equal(actual.root_node_ids, [0, 17])


def test_caller_owned_fill_overwrites_every_output() -> None:
    root_shape = i3(2, 2, 1)
    flags = make_flags(
        root_shape,
        lambda level, coord: level == 1 and coord in {(0, 0, 0), (1, 1, 0)},
    )
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    outputs = output_arrays(
        flags.size,
        int(np.count_nonzero(flags)),
        rank_to_coord.shape[0],
    )
    max_level = fill_refined_forest(
        root_shape,
        coord_to_rank,
        rank_to_coord,
        flags,
        *outputs,
    )
    expected = refined_forest_reference(root_shape, rank_to_coord, flags)
    assert max_level == expected.max_level
    for actual, reference in zip(outputs, expected[:-1], strict=True):
        assert np.array_equal(actual, reference)


@pytest.mark.parametrize(
    ("flags", "match"),
    [
        (np.asarray([], dtype=bool), "ends"),
        (np.asarray([False, *([True] * 7)], dtype=bool), "ends"),
        (np.asarray([True, True], dtype=bool), "trailing"),
    ],
)
def test_malformed_streams_are_rejected_before_output_mutation(
    flags: np.ndarray,
    match: str,
) -> None:
    root_shape = i3(1, 1, 1)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    outputs = output_arrays(flags.size, int(np.count_nonzero(flags)), 1)
    before = tuple(output.copy() for output in outputs)
    with pytest.raises(ValueError, match=match):
        fill_refined_forest(
            root_shape,
            coord_to_rank,
            rank_to_coord,
            flags,
            *outputs,
        )
    for output, original in zip(outputs, before, strict=True):
        assert np.array_equal(output, original)


def test_representable_depth_boundary_and_overflow_are_explicit() -> None:
    root_shape = i3(1, 1, 1)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    valid_flags = chain_flags(62)
    valid = refined_forest(
        root_shape,
        coord_to_rank,
        rank_to_coord,
        valid_flags,
    )
    assert valid.max_level == 63

    overflow_flags = chain_flags(63)
    outputs = output_arrays(
        overflow_flags.size,
        int(np.count_nonzero(overflow_flags)),
        1,
    )
    before = tuple(output.copy() for output in outputs)
    with pytest.raises(OverflowError, match="logical grid"):
        fill_refined_forest(
            root_shape,
            coord_to_rank,
            rank_to_coord,
            overflow_flags,
            *outputs,
        )
    for output, original in zip(outputs, before, strict=True):
        assert np.array_equal(output, original)


def test_invalid_maps_layouts_and_outputs_are_atomic() -> None:
    root_shape = i3(1, 1, 1)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    flags = np.asarray([True], dtype=bool)
    outputs = output_arrays(1, 1, 1)
    before = tuple(output.copy() for output in outputs)

    broken_inverse = rank_to_coord.copy()
    broken_inverse[0, 0] = 1
    with pytest.raises(ValueError, match="rank 0"):
        fill_refined_forest(
            root_shape,
            coord_to_rank,
            broken_inverse,
            flags,
            *outputs,
        )

    with pytest.raises(TypeError, match="dtype bool"):
        fill_refined_forest(
            root_shape,
            coord_to_rank,
            rank_to_coord,
            np.ones(1, dtype=np.uint8),
            *outputs,
        )

    noncontiguous_flags = np.ones(4, dtype=bool)[::2]
    assert not noncontiguous_flags.flags.c_contiguous
    with pytest.raises(ValueError, match="C-contiguous"):
        fill_refined_forest(
            root_shape,
            coord_to_rank,
            rank_to_coord,
            noncontiguous_flags,
            *outputs,
        )

    with pytest.raises(ValueError, match="shape"):
        fill_refined_forest(
            root_shape,
            coord_to_rank,
            rank_to_coord,
            flags,
            outputs[0],
            outputs[1],
            outputs[2],
            outputs[3][:, :7],
            *outputs[4:],
        )

    readonly = outputs[0].copy()
    readonly.setflags(write=False)
    with pytest.raises(ValueError, match="writable"):
        fill_refined_forest(
            root_shape,
            coord_to_rank,
            rank_to_coord,
            flags,
            readonly,
            *outputs[1:],
        )

    overlapping = list(output_arrays(1, 1, 1))
    overlapping[2] = overlapping[0]
    with pytest.raises(ValueError, match="overlap each other"):
        fill_refined_forest(
            root_shape,
            coord_to_rank,
            rank_to_coord,
            flags,
            *overlapping,
        )

    for output, original in zip(outputs, before, strict=True):
        assert np.array_equal(output, original)


def test_read_only_inputs_are_accepted_and_preserved() -> None:
    root_shape = i3(2, 1, 1)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    flags = np.asarray([True, True], dtype=bool)
    expected_inputs = tuple(
        value.copy() for value in (root_shape, coord_to_rank, rank_to_coord, flags)
    )
    for value in (root_shape, coord_to_rank, rank_to_coord, flags):
        value.setflags(write=False)
    forest = refined_forest(root_shape, coord_to_rank, rank_to_coord, flags)
    assert forest.max_level == 1
    assert np.array_equal(forest.root_node_ids, np.arange(2, dtype=np.int64))
    assert np.array_equal(forest.leaf_node_ids, np.arange(2, dtype=np.int64))
    assert np.array_equal(forest.node_levels, np.ones(2, dtype=np.int64))
    assert np.array_equal(forest.node_coords, rank_to_coord)
    for value, expected in zip(
        (root_shape, coord_to_rank, rank_to_coord, flags),
        expected_inputs,
        strict=True,
    ):
        assert np.array_equal(value, expected)


def test_representative_refined_dat_tree_metadata_when_available() -> None:
    path = Path(__file__).resolve().parents[2] / "data/weno509_sub_0000.dat"
    if not path.exists():
        pytest.skip("representative refined AMRVAC evidence file is unavailable")

    from simesh.amrvac.datio import get_metadata

    header, flags_input, tree = get_metadata(str(path))
    root_shape = np.ascontiguousarray(
        header["domain_nx"] // header["block_nx"], dtype=np.int64
    )
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    flags = np.ascontiguousarray(flags_input, dtype=np.bool_)
    forest = refined_forest(
        root_shape,
        coord_to_rank,
        rank_to_coord,
        flags,
    )
    leaf_nodes = forest.leaf_node_ids

    assert header["geometry"] == "Cartesian_3D"
    assert bool(header["staggered"])
    assert np.array_equal(
        forest.node_levels[leaf_nodes], np.asarray(tree[0], dtype=np.int64)
    )
    assert np.array_equal(
        forest.node_coords[leaf_nodes] + 1,
        np.asarray(tree[1], dtype=np.int64),
    )
    assert forest.root_node_ids.tolist() == [0, 6433, 12890, 19355]
    assert forest.max_level == 6
    assert sum(array.nbytes for array in forest[:-1]) == 3_075_472
