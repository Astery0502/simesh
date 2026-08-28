from __future__ import annotations

import numpy as np
import pytest

from simesh.utils.lib.amr.forest import AMRForest
from simesh.utils.lib.amr.mesh import AMRMesh
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.target_boxes import fill_directed_halo_target_boxes
from simesh_rewrite.target_boxes_reference import (
    directed_halo_target_boxes_reference,
)


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


def production_boxes(
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    requested_lower: np.ndarray,
    requested_upper: np.ndarray,
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(directions.shape, -77, dtype=np.int64)
    upper = np.full(directions.shape, -91, dtype=np.int64)
    fill_directed_halo_target_boxes(
        interior_lower,
        interior_upper,
        requested_lower,
        requested_upper,
        directions,
        lower,
        upper,
    )
    return lower, upper


def assert_reference_equal(
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    requested_lower: np.ndarray,
    requested_upper: np.ndarray,
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    actual = production_boxes(
        interior_lower,
        interior_upper,
        requested_lower,
        requested_upper,
        directions,
    )
    expected = directed_halo_target_boxes_reference(
        interior_lower,
        interior_upper,
        requested_lower,
        requested_upper,
        directions,
    )
    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])
    return actual


def assert_canonical_disjoint_union(
    requested_lower: np.ndarray,
    requested_upper: np.ndarray,
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    shape = tuple(int(value) for value in requested_upper - requested_lower)
    occupancy = np.zeros(shape, dtype=np.uint8)
    for row in range(lower.shape[0]):
        local_lower = lower[row] - requested_lower
        local_upper = upper[row] - requested_lower
        occupancy[
            int(local_lower[0]) : int(local_upper[0]),
            int(local_lower[1]) : int(local_upper[1]),
            int(local_lower[2]) : int(local_upper[2]),
        ] += np.uint8(1)

    expected = np.ones(shape, dtype=np.uint8)
    local_interior_lower = interior_lower - requested_lower
    local_interior_upper = interior_upper - requested_lower
    expected[
        int(local_interior_lower[0]) : int(local_interior_upper[0]),
        int(local_interior_lower[1]) : int(local_interior_upper[1]),
        int(local_interior_lower[2]) : int(local_interior_upper[2]),
    ] = 0
    assert int(occupancy.max(initial=0)) <= 1
    assert np.array_equal(occupancy, expected)


def test_all_26_exact_boxes_are_a_disjoint_halo_partition() -> None:
    requested_lower = i3(2, 3, 5)
    interior_lower = i3(4, 6, 6)
    interior_upper = i3(9, 8, 10)
    requested_upper = i3(12, 13, 12)
    lower, upper = assert_reference_equal(
        interior_lower,
        interior_upper,
        requested_lower,
        requested_upper,
        ALL_DIRECTIONS,
    )

    interval_lowers = (requested_lower, interior_lower, interior_upper)
    interval_uppers = (interior_lower, interior_upper, requested_upper)
    for row, direction in enumerate(ALL_DIRECTIONS):
        for axis in range(3):
            index = int(direction[axis]) + 1
            assert lower[row, axis] == interval_lowers[index][axis]
            assert upper[row, axis] == interval_uppers[index][axis]

    assert_canonical_disjoint_union(
        requested_lower,
        requested_upper,
        interior_lower,
        interior_upper,
        lower,
        upper,
    )


@pytest.mark.parametrize(
    ("requested_lower", "interior_lower", "interior_upper", "requested_upper"),
    [
        (i3(0, 4, 1), i3(3, 4, 6), i3(8, 11, 8), i3(9, 15, 13)),
        (i3(3, 0, 2), i3(3, 5, 2), i3(8, 9, 7), i3(10, 9, 11)),
        (i3(0, 0, 0), i3(2, 3, 4), i3(2, 8, 9), i3(6, 11, 12)),
        (i3(0, 0, 0), i3(2, 3, 4), i3(2, 3, 4), i3(6, 7, 8)),
        (i3(1, 3, 5), i3(2, 4, 6), i3(3, 5, 7), i3(4, 6, 8)),
        (i3(2, 3, 4), i3(2, 3, 4), i3(2, 7, 9), i3(2, 8, 11)),
    ],
)
def test_asymmetric_zero_width_empty_and_singleton_axes(
    requested_lower,
    interior_lower,
    interior_upper,
    requested_upper,
) -> None:
    lower, upper = assert_reference_equal(
        interior_lower,
        interior_upper,
        requested_lower,
        requested_upper,
        ALL_DIRECTIONS,
    )
    assert np.all(lower <= upper)
    assert_canonical_disjoint_union(
        requested_lower,
        requested_upper,
        interior_lower,
        interior_upper,
        lower,
        upper,
    )


def test_subsets_reorders_repeats_d0_and_read_only_inputs() -> None:
    requested_lower = i3(0, 1, 2)
    interior_lower = i3(2, 3, 5)
    interior_upper = i3(7, 9, 11)
    requested_upper = i3(10, 10, 15)
    directions = np.asarray(
        [
            (1, 0, 0),
            (-1, 1, 0),
            (1, 0, 0),
            (0, 0, -1),
            (1, 1, 1),
        ],
        dtype=np.int64,
    )
    inputs = (
        interior_lower,
        interior_upper,
        requested_lower,
        requested_upper,
        directions,
    )
    before = tuple(value.copy() for value in inputs)
    for value in inputs:
        value.setflags(write=False)
    lower, upper = assert_reference_equal(*inputs)
    assert np.array_equal(lower[0], lower[2])
    assert np.array_equal(upper[0], upper[2])
    for value, original in zip(inputs, before, strict=True):
        assert np.array_equal(value, original)

    empty_directions = np.empty((0, 3), dtype=np.int64)
    empty_lower = np.empty((0, 3), dtype=np.int64)
    empty_upper = np.empty((0, 3), dtype=np.int64)
    fill_directed_halo_target_boxes(
        interior_lower,
        interior_upper,
        requested_lower,
        requested_upper,
        empty_directions,
        empty_lower,
        empty_upper,
    )
    reference = directed_halo_target_boxes_reference(
        interior_lower,
        interior_upper,
        requested_lower,
        requested_upper,
        empty_directions,
    )
    assert reference[0].shape == reference[1].shape == (0, 3)
    assert np.array_equal(empty_lower, reference[0])
    assert np.array_equal(empty_upper, reference[1])


def test_overlapping_read_only_box_inputs_are_allowed() -> None:
    lower_boundary = i3(2, 4, 6)
    upper_boundary = i3(7, 9, 11)
    lower_boundary.setflags(write=False)
    upper_boundary.setflags(write=False)
    assert_reference_equal(
        lower_boundary,
        upper_boundary,
        lower_boundary,
        upper_boundary,
        ALL_DIRECTIONS,
    )


def test_current_level1_neighbor_constants_fill_all_target_boxes() -> None:
    root = i3(3, 3, 3)
    block_shape = i3(4, 6, 8)
    ghost_width = 2
    leaf_count = int(np.prod(root))
    forest = AMRForest(
        3,
        *tuple(int(value) for value in root),
        np.ones(leaf_count, dtype=np.int32),
    )
    mesh = AMRMesh(
        3,
        block_shape.astype(np.uint32),
        (root * block_shape).astype(np.uint32),
        np.zeros(3),
        np.ones(3),
        np.uint32(ghost_width),
        np.uint32(1),
        forest,
    )

    interior = np.empty((leaf_count, 1, *block_shape), dtype=np.float64)
    for leaf_id in range(leaf_count):
        interior[leaf_id].fill(np.float64(leaf_id) + np.float64(0.25))
    mesh.load_interior_data(interior)
    mesh.apply_ghost_cells()

    interior_lower = i3(ghost_width, ghost_width, ghost_width)
    interior_upper = interior_lower + block_shape
    requested_lower = i3(0, 0, 0)
    requested_upper = block_shape + 2 * ghost_width
    lower, upper = production_boxes(
        interior_lower,
        interior_upper,
        requested_lower,
        requested_upper,
        ALL_DIRECTIONS,
    )
    coord_to_rank, _ = level1_morton(root)
    center_coord = i3(1, 1, 1)
    center_leaf = int(coord_to_rank[1, 1, 1])
    padded = np.asarray(mesh.padded_view())[center_leaf, ..., 0]
    for row, direction in enumerate(ALL_DIRECTIONS):
        source_coord = center_coord + direction
        source_leaf = int(coord_to_rank[tuple(source_coord)])
        expected = np.float64(source_leaf) + np.float64(0.25)
        target = padded[
            int(lower[row, 0]) : int(upper[row, 0]),
            int(lower[row, 1]) : int(upper[row, 1]),
            int(lower[row, 2]) : int(upper[row, 2]),
        ]
        assert target.size > 0
        assert np.all(target == expected)


def assert_atomic_failure(
    error_type: type[Exception],
    arguments: tuple,
) -> None:
    lower = arguments[-2]
    upper = arguments[-1]
    lower_before = lower.copy() if isinstance(lower, np.ndarray) else None
    upper_before = upper.copy() if isinstance(upper, np.ndarray) else None
    with pytest.raises(error_type):
        fill_directed_halo_target_boxes(*arguments)
    if isinstance(lower, np.ndarray):
        assert np.array_equal(lower, lower_before)
    if isinstance(upper, np.ndarray):
        assert np.array_equal(upper, upper_before)


def valid_arguments(direction_count: int = 2) -> list:
    directions = np.asarray([(-1, 0, 1), (1, -1, 0)], dtype=np.int64)[
        :direction_count
    ].copy()
    return [
        i3(2, 3, 4),
        i3(7, 9, 11),
        i3(0, 1, 2),
        i3(10, 12, 15),
        directions,
        np.full((direction_count, 3), -71, dtype=np.int64),
        np.full((direction_count, 3), -83, dtype=np.int64),
    ]


@pytest.mark.parametrize("argument_index", range(5))
def test_input_object_and_dtype_errors_are_atomic(argument_index: int) -> None:
    arguments = valid_arguments()
    arguments[argument_index] = arguments[argument_index].tolist()
    assert_atomic_failure(TypeError, tuple(arguments))

    arguments = valid_arguments()
    arguments[argument_index] = arguments[argument_index].astype(np.int32)
    assert_atomic_failure(TypeError, tuple(arguments))

    nonnative = np.dtype(">i8" if np.little_endian else "<i8")
    arguments = valid_arguments()
    arguments[argument_index] = arguments[argument_index].astype(nonnative)
    assert_atomic_failure(TypeError, tuple(arguments))


@pytest.mark.parametrize("argument_index", range(4))
def test_box_shape_and_layout_errors_are_atomic(argument_index: int) -> None:
    arguments = valid_arguments()
    arguments[argument_index] = np.zeros((1, 3), dtype=np.int64)
    assert_atomic_failure(ValueError, tuple(arguments))

    arguments = valid_arguments()
    arguments[argument_index] = np.arange(6, dtype=np.int64)[::2]
    assert_atomic_failure(ValueError, tuple(arguments))


def test_direction_shape_layout_and_value_errors_are_atomic() -> None:
    invalid_directions = [
        np.zeros(3, dtype=np.int64),
        np.zeros((2, 2), dtype=np.int64),
        np.arange(12, dtype=np.int64).reshape(2, 6)[:, ::2],
        np.asarray([(-1, 0, 1), (2, 0, 0)], dtype=np.int64),
        np.asarray([(-1, 0, 1), (0, 0, 0)], dtype=np.int64),
    ]
    for directions in invalid_directions:
        arguments = valid_arguments()
        arguments[4] = directions
        assert_atomic_failure(ValueError, tuple(arguments))


@pytest.mark.parametrize("output_index", (5, 6))
def test_output_type_shape_layout_and_writability_are_atomic(
    output_index: int,
) -> None:
    cases = [
        (list, TypeError),
        (lambda value: value.astype(np.int32), TypeError),
        (
            lambda value: value.astype(
                np.dtype(">i8" if np.little_endian else "<i8")
            ),
            TypeError,
        ),
        (lambda value: np.empty((2, 2), dtype=np.int64), ValueError),
        (
            lambda value: np.empty((2, 6), dtype=np.int64)[:, ::2],
            ValueError,
        ),
        (
            lambda value: np.asarray(value).copy(),
            None,
        ),
    ]
    for transform, error_type in cases:
        arguments = valid_arguments()
        if error_type is None:
            output = transform(arguments[output_index])
            output.setflags(write=False)
            arguments[output_index] = output
            assert_atomic_failure(ValueError, tuple(arguments))
        else:
            arguments[output_index] = transform(arguments[output_index])
            assert_atomic_failure(error_type, tuple(arguments))


@pytest.mark.parametrize(
    ("argument_index", "replacement"),
    [
        (2, i3(-1, 1, 2)),
        (2, i3(3, 1, 2)),
        (0, i3(8, 3, 4)),
        (1, i3(7, 13, 11)),
        (3, i3(10, 8, 15)),
    ],
)
def test_box_order_errors_are_atomic(argument_index, replacement) -> None:
    arguments = valid_arguments()
    arguments[argument_index] = replacement
    assert_atomic_failure(ValueError, tuple(arguments))


def test_every_output_overlap_is_atomic() -> None:
    arguments = valid_arguments()
    arguments[6] = arguments[5]
    assert_atomic_failure(ValueError, tuple(arguments))

    for input_index in range(5):
        arguments = valid_arguments(direction_count=1)
        if input_index < 4:
            shared = np.asarray(arguments[input_index]).reshape(1, 3)
            arguments[input_index] = shared.reshape(3)
        else:
            shared = arguments[input_index]
        arguments[5] = shared
        assert_atomic_failure(ValueError, tuple(arguments))

        arguments = valid_arguments(direction_count=1)
        if input_index < 4:
            shared = np.asarray(arguments[input_index]).reshape(1, 3)
            arguments[input_index] = shared.reshape(3)
        else:
            shared = arguments[input_index]
        arguments[6] = shared
        assert_atomic_failure(ValueError, tuple(arguments))
