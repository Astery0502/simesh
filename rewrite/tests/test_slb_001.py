from __future__ import annotations

import numpy as np
import pytest

from simesh.utils.lib.amr.forest import AMRForest
from simesh.utils.lib.amr.mesh import AMRMesh
from simesh_rewrite.foundation import copy_region_into
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.same_level_boxes import fill_same_level_source_boxes
from simesh_rewrite.same_level_boxes_reference import (
    same_level_source_boxes_reference,
)
from simesh_rewrite.target_boxes import fill_directed_halo_target_boxes


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


def target_boxes(
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    requested_lower: np.ndarray,
    requested_upper: np.ndarray,
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.empty(directions.shape, dtype=np.int64)
    upper = np.empty(directions.shape, dtype=np.int64)
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


def production_boxes(
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    directions: np.ndarray,
    target_lower: np.ndarray,
    target_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(directions.shape, -71, dtype=np.int64)
    upper = np.full(directions.shape, -83, dtype=np.int64)
    fill_same_level_source_boxes(
        interior_lower,
        interior_upper,
        directions,
        target_lower,
        target_upper,
        lower,
        upper,
    )
    return lower, upper


def assert_reference_equal(
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    directions: np.ndarray,
    target_lower: np.ndarray,
    target_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    actual = production_boxes(
        interior_lower,
        interior_upper,
        directions,
        target_lower,
        target_upper,
    )
    expected = same_level_source_boxes_reference(
        interior_lower,
        interior_upper,
        directions,
        target_lower,
        target_upper,
    )
    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])
    return actual


def assert_translation_invariants(
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    directions: np.ndarray,
    target_lower: np.ndarray,
    target_upper: np.ndarray,
    source_lower: np.ndarray,
    source_upper: np.ndarray,
) -> None:
    widths = interior_upper - interior_lower
    assert np.array_equal(source_upper - source_lower, target_upper - target_lower)
    assert np.all(source_lower >= interior_lower)
    assert np.all(source_upper <= interior_upper)
    for row in range(directions.shape[0]):
        assert np.array_equal(
            source_lower[row],
            target_lower[row] - directions[row] * widths,
        )
        assert np.array_equal(
            source_upper[row],
            target_upper[row] - directions[row] * widths,
        )


def test_all_directions_with_asymmetric_tgt_boxes() -> None:
    requested_lower = i3(1, 0, 2)
    interior_lower = i3(3, 4, 5)
    interior_upper = i3(9, 11, 13)
    requested_upper = i3(12, 13, 15)
    target_lower, target_upper = target_boxes(
        interior_lower,
        interior_upper,
        requested_lower,
        requested_upper,
        ALL_DIRECTIONS,
    )
    source_lower, source_upper = assert_reference_equal(
        interior_lower,
        interior_upper,
        ALL_DIRECTIONS,
        target_lower,
        target_upper,
    )
    assert_translation_invariants(
        interior_lower,
        interior_upper,
        ALL_DIRECTIONS,
        target_lower,
        target_upper,
        source_lower,
        source_upper,
    )


def test_partial_and_zero_width_rows_on_every_axis() -> None:
    interior_lower = i3(5, 7, 11)
    interior_upper = i3(11, 14, 19)
    widths = interior_upper - interior_lower
    target_lower = np.empty_like(ALL_DIRECTIONS)
    target_upper = np.empty_like(ALL_DIRECTIONS)
    for row, direction in enumerate(ALL_DIRECTIONS):
        for axis in range(3):
            component = int(direction[axis])
            maximum_width = int(widths[axis])
            if component < 0:
                maximum_width = min(
                    maximum_width, int(interior_lower[axis])
                )
            width = (row + 2 * axis) % (maximum_width + 1)
            if component < 0:
                target_lower[row, axis] = interior_lower[axis] - width
                target_upper[row, axis] = interior_lower[axis]
            elif component == 0:
                start = (row + axis) % (int(widths[axis]) + 1)
                remaining = int(widths[axis]) - start
                extent = (row + 3 * axis) % (remaining + 1)
                target_lower[row, axis] = interior_lower[axis] + start
                target_upper[row, axis] = target_lower[row, axis] + extent
            else:
                target_lower[row, axis] = interior_upper[axis]
                target_upper[row, axis] = interior_upper[axis] + width

    source_lower, source_upper = assert_reference_equal(
        interior_lower,
        interior_upper,
        ALL_DIRECTIONS,
        target_lower,
        target_upper,
    )
    assert_translation_invariants(
        interior_lower,
        interior_upper,
        ALL_DIRECTIONS,
        target_lower,
        target_upper,
        source_lower,
        source_upper,
    )
    assert np.any(target_lower == target_upper)


def test_reduced_tgt_rows_repeats_reorders_and_read_only_inputs() -> None:
    interior_lower = i3(2, 3, 4)
    interior_upper = i3(8, 10, 12)
    requested_lower = i3(0, 1, 1)
    requested_upper = i3(10, 12, 15)
    reduced_directions = np.asarray(
        [
            (-1, 1, 0),
            (0, 1, 0),
            (1, 0, -1),
            (-1, 1, 0),
            (0, -1, 1),
        ],
        dtype=np.int64,
    )
    target_lower, target_upper = target_boxes(
        interior_lower,
        interior_upper,
        requested_lower,
        requested_upper,
        reduced_directions,
    )
    inputs = [
        interior_lower,
        interior_upper,
        reduced_directions,
        target_lower,
        target_upper,
    ]
    before = [value.copy() for value in inputs]
    for value in inputs:
        value.setflags(write=False)
    source_lower, source_upper = assert_reference_equal(*inputs)
    assert np.array_equal(source_lower[0], source_lower[3])
    assert np.array_equal(source_upper[0], source_upper[3])
    for value, original in zip(inputs, before, strict=True):
        assert np.array_equal(value, original)


def test_empty_row_axis_is_valid() -> None:
    lower = i3(2, 3, 4)
    upper = i3(7, 9, 11)
    directions = np.empty((0, 3), dtype=np.int64)
    targets = np.empty((0, 3), dtype=np.int64)
    source_lower = np.empty((0, 3), dtype=np.int64)
    source_upper = np.empty((0, 3), dtype=np.int64)
    fill_same_level_source_boxes(
        lower,
        upper,
        directions,
        targets,
        targets,
        source_lower,
        source_upper,
    )
    expected = same_level_source_boxes_reference(
        lower, upper, directions, targets, targets
    )
    assert expected[0].shape == expected[1].shape == (0, 3)
    assert np.array_equal(source_lower, expected[0])
    assert np.array_equal(source_upper, expected[1])


@pytest.mark.parametrize(
    ("interior_lower", "interior_upper", "requested_lower", "requested_upper"),
    [
        (i3(2, 4, 6), i3(3, 5, 7), i3(1, 3, 5), i3(4, 6, 8)),
        (i3(2, 4, 6), i3(2, 5, 6), i3(2, 3, 6), i3(2, 6, 6)),
        (i3(0, 0, 0), i3(0, 0, 0), i3(0, 0, 0), i3(0, 0, 0)),
    ],
)
def test_singleton_and_empty_interior_axes_match_reference(
    interior_lower,
    interior_upper,
    requested_lower,
    requested_upper,
) -> None:
    target_lower, target_upper = target_boxes(
        interior_lower,
        interior_upper,
        requested_lower,
        requested_upper,
        ALL_DIRECTIONS,
    )
    source_lower, source_upper = assert_reference_equal(
        interior_lower,
        interior_upper,
        ALL_DIRECTIONS,
        target_lower,
        target_upper,
    )
    assert_translation_invariants(
        interior_lower,
        interior_upper,
        ALL_DIRECTIONS,
        target_lower,
        target_upper,
        source_lower,
        source_upper,
    )


def test_near_int64_limit_translation_is_exact() -> None:
    maximum = int(np.iinfo(np.int64).max)
    interior_lower = i3(1, 1, 1)
    interior_upper = i3(maximum, maximum, maximum)
    directions = np.asarray([(-1, 1, 0)], dtype=np.int64)
    target_lower = np.asarray([[0, maximum, 1]], dtype=np.int64)
    target_upper = np.asarray([[1, maximum, maximum]], dtype=np.int64)
    source_lower, source_upper = assert_reference_equal(
        interior_lower,
        interior_upper,
        directions,
        target_lower,
        target_upper,
    )
    assert source_lower.tolist() == [[maximum - 1, 1, 1]]
    assert source_upper.tolist() == [[maximum, 1, maximum]]


def test_current_unmasked_level1_source_regions_are_exact() -> None:
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
    x, y, z = np.indices(tuple(int(value) for value in block_shape))
    for leaf_id in range(leaf_count):
        interior[leaf_id, 0] = (
            float(leaf_id) * 1_000_000.0
            + x * 10_000.0
            + y * 100.0
            + z
        )
    mesh.load_interior_data(interior)
    mesh.apply_ghost_cells()

    interior_lower = i3(ghost_width, ghost_width, ghost_width)
    interior_upper = interior_lower + block_shape
    requested_lower = i3(0, 0, 0)
    requested_upper = block_shape + 2 * ghost_width
    target_lower, target_upper = target_boxes(
        interior_lower,
        interior_upper,
        requested_lower,
        requested_upper,
        ALL_DIRECTIONS,
    )
    source_lower, source_upper = production_boxes(
        interior_lower,
        interior_upper,
        ALL_DIRECTIONS,
        target_lower,
        target_upper,
    )
    coord_to_rank, _ = level1_morton(root)
    primary_coord = i3(1, 1, 1)
    primary_leaf = int(coord_to_rank[1, 1, 1])
    padded = np.asarray(mesh.padded_view())[primary_leaf, ..., 0]
    for row, direction in enumerate(ALL_DIRECTIONS):
        source_coord = primary_coord + direction
        source_leaf = int(coord_to_rank[tuple(source_coord)])
        actual = padded[
            int(target_lower[row, 0]) : int(target_upper[row, 0]),
            int(target_lower[row, 1]) : int(target_upper[row, 1]),
            int(target_lower[row, 2]) : int(target_upper[row, 2]),
        ]
        local_lower = source_lower[row] - interior_lower
        local_upper = source_upper[row] - interior_lower
        expected = interior[
            source_leaf,
            0,
            int(local_lower[0]) : int(local_upper[0]),
            int(local_lower[1]) : int(local_upper[1]),
            int(local_lower[2]) : int(local_upper[2]),
        ]
        assert np.array_equal(actual, expected)


def test_one_slot_fnd_copy_composition_is_exact() -> None:
    requested_lower = i3(0, 0, 0)
    interior_lower = i3(2, 3, 1)
    interior_upper = i3(7, 9, 8)
    requested_upper = i3(9, 11, 10)
    target_lower, target_upper = target_boxes(
        interior_lower,
        interior_upper,
        requested_lower,
        requested_upper,
        ALL_DIRECTIONS,
    )
    source_lower, source_upper = production_boxes(
        interior_lower,
        interior_upper,
        ALL_DIRECTIONS,
        target_lower,
        target_upper,
    )
    spatial_shape = tuple(int(value) for value in requested_upper)
    source = np.empty((1, 2, *spatial_shape), dtype=np.float64)
    indices = np.indices(spatial_shape)
    for field in range(2):
        source[0, field] = (
            field * 1_000_000.0
            + indices[0] * 10_000.0
            + indices[1] * 100.0
            + indices[2]
        )
    destination = np.full_like(source, -1.0)
    expected = destination.copy()
    for row in range(ALL_DIRECTIONS.shape[0]):
        extent = target_upper[row] - target_lower[row]
        copy_region_into(
            source,
            source_lower[row],
            destination,
            target_lower[row],
            extent,
        )
        for field in range(2):
            expected[
                0,
                field,
                int(target_lower[row, 0]) : int(target_upper[row, 0]),
                int(target_lower[row, 1]) : int(target_upper[row, 1]),
                int(target_lower[row, 2]) : int(target_upper[row, 2]),
            ] = source[
                0,
                field,
                int(source_lower[row, 0]) : int(source_upper[row, 0]),
                int(source_lower[row, 1]) : int(source_upper[row, 1]),
                int(source_lower[row, 2]) : int(source_upper[row, 2]),
            ]
    assert np.array_equal(destination, expected)
    assert np.all(
        destination[
            :,
            :,
            int(interior_lower[0]) : int(interior_upper[0]),
            int(interior_lower[1]) : int(interior_upper[1]),
            int(interior_lower[2]) : int(interior_upper[2]),
        ]
        == -1.0
    )


def valid_arguments(row_count: int = 2) -> list[np.ndarray]:
    directions = np.asarray([(-1, 0, 1), (1, -1, 0)], dtype=np.int64)[
        :row_count
    ].copy()
    target_lower = np.asarray([[0, 3, 11], [7, 1, 4]], dtype=np.int64)[
        :row_count
    ].copy()
    target_upper = np.asarray([[2, 9, 14], [10, 3, 11]], dtype=np.int64)[
        :row_count
    ].copy()
    return [
        i3(2, 3, 4),
        i3(7, 9, 11),
        directions,
        target_lower,
        target_upper,
        np.full((row_count, 3), -41, dtype=np.int64),
        np.full((row_count, 3), -53, dtype=np.int64),
    ]


def assert_atomic_failure(
    error_type: type[Exception], arguments: list[np.ndarray]
) -> None:
    lower = arguments[5]
    upper = arguments[6]
    lower_before = lower.copy() if isinstance(lower, np.ndarray) else None
    upper_before = upper.copy() if isinstance(upper, np.ndarray) else None
    with pytest.raises(error_type):
        fill_same_level_source_boxes(*arguments)
    if isinstance(lower, np.ndarray):
        assert np.array_equal(lower, lower_before)
    if isinstance(upper, np.ndarray):
        assert np.array_equal(upper, upper_before)


def noncontiguous_copy(value: np.ndarray) -> np.ndarray:
    if value.ndim == 1:
        return np.repeat(value, 2)[::2]
    return np.repeat(value, 2, axis=-1)[..., ::2]


@pytest.mark.parametrize("input_index", range(5))
def test_input_type_dtype_shape_and_layout_errors_are_atomic(
    input_index: int,
) -> None:
    arguments = valid_arguments()
    arguments[input_index] = arguments[input_index].tolist()
    assert_atomic_failure(TypeError, arguments)

    arguments = valid_arguments()
    arguments[input_index] = arguments[input_index].astype(np.int32)
    assert_atomic_failure(TypeError, arguments)

    arguments = valid_arguments()
    nonnative = np.dtype(">i8" if np.little_endian else "<i8")
    arguments[input_index] = arguments[input_index].astype(nonnative)
    assert_atomic_failure(TypeError, arguments)

    arguments = valid_arguments()
    if input_index < 2:
        arguments[input_index] = np.empty((1, 3), dtype=np.int64)
    elif input_index == 2:
        arguments[input_index] = np.empty((2, 2), dtype=np.int64)
    else:
        arguments[input_index] = np.empty((2, 4), dtype=np.int64)
    assert_atomic_failure(ValueError, arguments)

    arguments = valid_arguments()
    arguments[input_index] = noncontiguous_copy(arguments[input_index])
    assert_atomic_failure(ValueError, arguments)


@pytest.mark.parametrize("output_index", (5, 6))
def test_output_type_dtype_shape_layout_and_writability_are_atomic(
    output_index: int,
) -> None:
    transforms = [
        (lambda value: value.tolist(), TypeError),
        (lambda value: value.astype(np.int32), TypeError),
        (lambda value: np.empty((2, 2), dtype=np.int64), ValueError),
        (noncontiguous_copy, ValueError),
    ]
    for transform, error_type in transforms:
        arguments = valid_arguments()
        arguments[output_index] = transform(arguments[output_index])
        assert_atomic_failure(error_type, arguments)
    arguments = valid_arguments()
    arguments[output_index].setflags(write=False)
    assert_atomic_failure(ValueError, arguments)


def test_interior_direction_target_and_formal_side_errors_are_atomic() -> None:
    mutations: list[tuple[int, tuple[int, ...] | tuple[int, int, int]]] = [
        (0, (-1, 3, 4)),
        (0, (8, 3, 4)),
    ]
    for index, values in mutations:
        arguments = valid_arguments()
        arguments[index] = i3(*values)
        assert_atomic_failure(ValueError, arguments)

    arguments = valid_arguments()
    arguments[2][1, 0] = 2
    assert_atomic_failure(ValueError, arguments)
    arguments = valid_arguments()
    arguments[2][1] = 0
    assert_atomic_failure(ValueError, arguments)
    arguments = valid_arguments()
    arguments[3][1, 1] = -1
    assert_atomic_failure(ValueError, arguments)
    arguments = valid_arguments()
    arguments[3][1, 1] = 4
    arguments[4][1, 1] = 3
    assert_atomic_failure(ValueError, arguments)

    side_mutations = [
        (3, (0, 0), -4),
        (4, (0, 0), 3),
        (3, (0, 1), 2),
        (4, (0, 1), 10),
        (3, (0, 2), 10),
        (4, (0, 2), 19),
    ]
    for argument_index, index, value in side_mutations:
        arguments = valid_arguments()
        arguments[argument_index][index] = value
        assert_atomic_failure(ValueError, arguments)


def test_complete_direction_validation_precedes_target_semantics() -> None:
    arguments = valid_arguments(row_count=1)
    arguments[2][0] = 0
    arguments[3][0, 0] = 1
    lower_before = arguments[5].copy()
    upper_before = arguments[6].copy()
    with pytest.raises(ValueError, match="must be noncenter"):
        fill_same_level_source_boxes(*arguments)
    assert np.array_equal(arguments[5], lower_before)
    assert np.array_equal(arguments[6], upper_before)

    arguments = valid_arguments()
    arguments[3][0, 0] = 1
    arguments[2][1, 2] = 2
    lower_before = arguments[5].copy()
    upper_before = arguments[6].copy()
    with pytest.raises(ValueError, match="outside"):
        fill_same_level_source_boxes(*arguments)
    assert np.array_equal(arguments[5], lower_before)
    assert np.array_equal(arguments[6], upper_before)


def test_inputs_may_overlap_and_outputs_may_not() -> None:
    arguments = valid_arguments(row_count=1)
    shared_target = arguments[3]
    shared_target.fill(2)
    arguments[4] = shared_target
    arguments[2][0] = (-1, -1, -1)
    arguments[0] = i3(2, 2, 2)
    arguments[1] = i3(2, 2, 2)
    fill_same_level_source_boxes(*arguments)

    arguments = valid_arguments()
    arguments[6] = arguments[5]
    assert_atomic_failure(ValueError, arguments)


def test_each_output_input_overlap_is_atomic() -> None:
    base = valid_arguments()
    output_shape = base[5].shape
    output_nbytes = int(np.prod(output_shape, dtype=np.int64)) * 8
    for input_index in range(5):
        for output_index in (5, 6):
            arguments = valid_arguments()
            source = arguments[input_index]
            raw = np.zeros(max(source.nbytes, output_nbytes), dtype=np.uint8)
            shared_source = np.ndarray(
                source.shape, dtype=source.dtype, buffer=raw, offset=0
            )
            np.copyto(shared_source, source)
            shared_output = np.ndarray(
                output_shape, dtype=np.int64, buffer=raw, offset=0
            )
            arguments[input_index] = shared_source
            arguments[output_index] = shared_output
            assert_atomic_failure(ValueError, arguments)
