from __future__ import annotations

import itertools

import numpy as np
import pytest

from simesh.utils.lib.amr.forest import AMRForest
from simesh.utils.lib.amr.mesh import AMRMesh
from simesh_rewrite.balance import validate_refined_all_touch_2to1
from simesh_rewrite.finer_boxes import fill_finer_restriction_boxes
from simesh_rewrite.finer_boxes_reference import (
    finer_restriction_boxes_reference,
)
from simesh_rewrite.forest import refined_forest
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.relations import RELATION_FINER, balanced_refined_relations
from simesh_rewrite.restriction import restrict_cartesian_2to1_into
from simesh_rewrite.restriction_reference import (
    restrict_cartesian_2to1_reference,
)
from simesh_rewrite.target_boxes import fill_directed_halo_target_boxes


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def active_rows():
    directions: list[tuple[int, int, int]] = []
    phases: list[int] = []
    for direction in itertools.product((-1, 0, 1), repeat=3):
        if direction == (0, 0, 0):
            continue
        for phase in range(8):
            compatible = all(
                component == 0
                or ((phase >> axis) & 1) == (1 if component < 0 else 0)
                for axis, component in enumerate(direction)
            )
            if compatible:
                directions.append(direction)
                phases.append(phase)
    return np.asarray(directions, dtype=np.int64), np.asarray(phases, dtype=np.uint8)


def boxes_for_rows(directions: np.ndarray):
    interior_lower = i3(3, 4, 5)
    interior_upper = i3(11, 14, 17)
    requested_lower = i3(1, 2, 3)
    requested_upper = i3(13, 16, 19)
    target_lower = np.empty_like(directions)
    target_upper = np.empty_like(directions)
    fill_directed_halo_target_boxes(
        interior_lower,
        interior_upper,
        requested_lower,
        requested_upper,
        directions,
        target_lower,
        target_upper,
    )
    return interior_lower, interior_upper, target_lower, target_upper


def production(*inputs):
    directions = inputs[2]
    outputs = tuple(np.full_like(directions, -77) for _ in range(4))
    fill_finer_restriction_boxes(*inputs, *outputs)
    return outputs


def test_all_active_direction_phases_match_reference_and_ratio() -> None:
    directions, phases = active_rows()
    lower, upper, target_lower, target_upper = boxes_for_rows(directions)
    actual = production(
        lower, upper, directions, phases, target_lower, target_upper
    )
    expected = finer_restriction_boxes_reference(
        lower, upper, directions, phases, target_lower, target_upper
    )
    for value, reference in zip(actual, expected, strict=True):
        assert np.array_equal(value, reference)
    placed_lower, placed_upper, source_lower, source_upper = actual
    assert np.array_equal(
        source_upper - source_lower,
        2 * (placed_upper - placed_lower),
    )
    assert np.all(source_lower >= lower)
    assert np.all(source_upper <= upper)


def test_active_phases_partition_each_direction_target() -> None:
    directions, phases = active_rows()
    lower, upper, target_lower, target_upper = boxes_for_rows(directions)
    placed_lower, placed_upper, _, _ = production(
        lower, upper, directions, phases, target_lower, target_upper
    )
    for direction in {
        tuple(int(value) for value in row) for row in directions
    }:
        row_ids = [
            row
            for row, value in enumerate(directions)
            if tuple(int(item) for item in value) == direction
        ]
        expected_cells = set(
            itertools.product(
                *(
                    range(int(target_lower[row_ids[0], axis]), int(target_upper[row_ids[0], axis]))
                    for axis in range(3)
                )
            )
        )
        actual_cells: set[tuple[int, int, int]] = set()
        total = 0
        for row in row_ids:
            cells = set(
                itertools.product(
                    *(
                        range(int(placed_lower[row, axis]), int(placed_upper[row, axis]))
                        for axis in range(3)
                    )
                )
            )
            assert actual_cells.isdisjoint(cells)
            actual_cells.update(cells)
            total += len(cells)
        assert total == len(actual_cells)
        assert actual_cells == expected_cells


def test_rst_composition_matches_independent_restriction() -> None:
    lower = i3(2, 2, 2)
    upper = i3(10, 10, 10)
    directions = np.asarray([(-1, 0, 0)], dtype=np.int64)
    phases = np.asarray([1 + 2 + 0], dtype=np.uint8)
    target_lower = np.asarray([[0, 2, 2]], dtype=np.int64)
    target_upper = np.asarray([[2, 10, 10]], dtype=np.int64)
    placed_lower, placed_upper, source_lower, source_upper = production(
        lower, upper, directions, phases, target_lower, target_upper
    )
    fine = np.arange(12**3, dtype=np.float64).reshape(1, 1, 12, 12, 12)
    actual = np.full((1, 1, 12, 12, 12), np.nan, dtype=np.float64)
    expected = actual.copy()
    restrict_cartesian_2to1_into(
        fine,
        source_lower[0],
        source_upper[0],
        actual,
        placed_lower[0],
    )
    restrict_cartesian_2to1_reference(
        fine,
        source_lower[0],
        source_upper[0],
        expected,
        placed_lower[0],
    )
    assert np.array_equal(actual, expected, equal_nan=True)
    target = tuple(
        slice(int(placed_lower[0, axis]), int(placed_upper[0, axis]))
        for axis in range(3)
    )
    assert not np.isnan(actual[(0, 0, *target)]).any()


def test_zero_normal_width_and_read_only_inputs() -> None:
    lower = i3(2, 2, 2)
    upper = i3(10, 10, 10)
    directions = np.asarray([(-1, 0, 0)], dtype=np.int64)
    phases = np.asarray([1], dtype=np.uint8)
    target_lower = np.asarray([[2, 2, 2]], dtype=np.int64)
    target_upper = np.asarray([[2, 10, 10]], dtype=np.int64)
    inputs = [lower, upper, directions, phases, target_lower, target_upper]
    for value in inputs:
        value.setflags(write=False)
    placed_lower, placed_upper, source_lower, source_upper = production(*inputs)
    assert placed_lower[0, 0] == placed_upper[0, 0] == 2
    assert source_lower[0, 0] == source_upper[0, 0] == 10
    assert np.array_equal(source_upper - source_lower, 2 * (placed_upper - placed_lower))


@pytest.mark.parametrize(
    ("direction", "phase", "target_lower", "target_upper", "message"),
    [
        ((-1, 0, 0), 8, (0, 2, 2), (2, 10, 10), "outside"),
        ((0, 0, 0), 0, (2, 2, 2), (10, 10, 10), "noncenter"),
        ((-1, 0, 0), 0, (0, 2, 2), (2, 10, 10), "lower FINER"),
        ((1, 0, 0), 1, (10, 2, 2), (12, 10, 10), "upper FINER"),
        ((-1, 0, 0), 1, (0, 3, 2), (2, 10, 10), "zero-direction"),
        ((-1, 0, 0), 1, (-3, 2, 2), (2, 10, 10), "ordered and nonnegative"),
    ],
)
def test_invalid_semantics_preserve_every_output(
    direction, phase, target_lower, target_upper, message
) -> None:
    lower = i3(2, 2, 2)
    upper = i3(10, 10, 10)
    directions = np.asarray([direction], dtype=np.int64)
    phases = np.asarray([phase], dtype=np.uint8)
    targets_lower = np.asarray([target_lower], dtype=np.int64)
    targets_upper = np.asarray([target_upper], dtype=np.int64)
    outputs = tuple(
        np.asarray([[71, 72, 73]], dtype=np.int64) for _ in range(4)
    )
    before = tuple(value.copy() for value in outputs)
    with pytest.raises(ValueError, match=message):
        fill_finer_restriction_boxes(
            lower,
            upper,
            directions,
            phases,
            targets_lower,
            targets_upper,
            *outputs,
        )
    for value, original in zip(outputs, before, strict=True):
        assert np.array_equal(value, original)


def test_odd_tangential_extent_empty_rows_and_alias_rejection() -> None:
    lower = i3(2, 2, 2)
    upper = i3(10, 9, 10)
    directions = np.asarray([(-1, 0, 0)], dtype=np.int64)
    phases = np.asarray([1], dtype=np.uint8)
    target_lower = np.asarray([[0, 2, 2]], dtype=np.int64)
    target_upper = np.asarray([[2, 9, 10]], dtype=np.int64)
    outputs = tuple(np.full((1, 3), -1, dtype=np.int64) for _ in range(4))
    with pytest.raises(ValueError, match="positive even full interior"):
        fill_finer_restriction_boxes(
            lower, upper, directions, phases, target_lower, target_upper, *outputs
        )

    empty_rows = np.empty((0, 3), dtype=np.int64)
    empty_phases = np.empty(0, dtype=np.uint8)
    empty_outputs = tuple(np.empty((0, 3), dtype=np.int64) for _ in range(4))
    fill_finer_restriction_boxes(
        lower,
        i3(10, 10, 10),
        empty_rows,
        empty_phases,
        empty_rows,
        empty_rows,
        *empty_outputs,
    )

    lower = i3(2, 2, 2)
    upper = i3(10, 10, 10)
    directions = np.asarray([(-1, 0, 0)], dtype=np.int64)
    phases = np.asarray([1], dtype=np.uint8)
    target_lower = np.asarray([[0, 2, 2]], dtype=np.int64)
    target_upper = np.asarray([[2, 10, 10]], dtype=np.int64)
    output = np.full((1, 3), -1, dtype=np.int64)
    with pytest.raises(ValueError, match="must not overlap"):
        fill_finer_restriction_boxes(
            lower,
            upper,
            directions,
            phases,
            target_lower,
            target_upper,
            output,
            output,
            np.empty_like(output),
            np.empty_like(output),
        )


def test_representation_readonly_and_input_alias_errors_are_atomic() -> None:
    lower = i3(2, 2, 2)
    upper = i3(10, 10, 10)
    directions = np.asarray([(-1, 0, 0)], dtype=np.int64)
    phases = np.asarray([1], dtype=np.uint8)
    target_lower = np.asarray([[0, 2, 2]], dtype=np.int64)
    target_upper = np.asarray([[2, 10, 10]], dtype=np.int64)

    for bad_directions, bad_phases, error in (
        (directions.astype(np.int32), phases, TypeError),
        (directions, phases.astype(np.int64), TypeError),
        (np.asarray([[0, 0, -1]], dtype=np.int64)[:, ::-1], phases, ValueError),
    ):
        outputs = tuple(np.full((1, 3), 73, dtype=np.int64) for _ in range(4))
        before = tuple(value.copy() for value in outputs)
        with pytest.raises(error):
            fill_finer_restriction_boxes(
                lower,
                upper,
                bad_directions,
                bad_phases,
                target_lower,
                target_upper,
                *outputs,
            )
        for output, original in zip(outputs, before, strict=True):
            assert np.array_equal(output, original)

    outputs = tuple(np.full((1, 3), 79, dtype=np.int64) for _ in range(4))
    outputs[0].setflags(write=False)
    before = tuple(value.copy() for value in outputs)
    with pytest.raises(ValueError, match="must be writable"):
        fill_finer_restriction_boxes(
            lower,
            upper,
            directions,
            phases,
            target_lower,
            target_upper,
            *outputs,
        )
    for output, original in zip(outputs, before, strict=True):
        assert np.array_equal(output, original)

    aliased_before = target_lower.copy()
    with pytest.raises(ValueError, match="must not overlap"):
        fill_finer_restriction_boxes(
            lower,
            upper,
            directions,
            phases,
            target_lower,
            target_upper,
            target_lower,
            *(np.empty_like(target_lower) for _ in range(3)),
        )
    assert np.array_equal(target_lower, aliased_before)


def test_current_unmasked_finer_face_values_are_bit_exact() -> None:
    root = i3(4, 4, 4)
    coord_to_rank, rank_to_coord = level1_morton(root)
    flags: list[bool] = []
    for coordinate in rank_to_coord:
        if tuple(int(value) for value in coordinate) == (1, 1, 1):
            flags.append(False)
            flags.extend([True] * 8)
        else:
            flags.append(True)
    flag_array = np.asarray(flags, dtype=np.bool_)
    forest = refined_forest(root, coord_to_rank, rank_to_coord, flag_array)
    relation_base = (
        root,
        coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
    )
    validate_refined_all_touch_2to1(*relation_base)
    face_directions = np.asarray(
        [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)],
        dtype=np.int64,
    )
    leaf_ids = np.arange(forest.leaf_node_ids.size, dtype=np.int64)
    kinds, masks, counts, sources = balanced_refined_relations(
        *relation_base, leaf_ids, face_directions
    )
    candidates = np.argwhere(
        (kinds == RELATION_FINER) & (masks == 0) & (counts == 4)
    )
    assert candidates.size
    primary, direction_index = (int(value) for value in candidates[0])
    direction = face_directions[direction_index]

    current_forest = AMRForest(
        3, 4, 4, 4, flag_array.astype(np.int32)
    )
    column = (
        (int(direction[2]) + 1) * 9
        + (int(direction[1]) + 1) * 3
        + int(direction[0])
        + 1
    )
    assert int(np.asarray(current_forest.neighbor_type)[primary, column]) == 4
    mesh = AMRMesh(
        3,
        np.asarray([4, 4, 4], dtype=np.uint32),
        np.asarray([16, 16, 16], dtype=np.uint32),
        np.zeros(3),
        np.ones(3),
        np.uint32(2),
        np.uint32(1),
        current_forest,
    )
    values = np.arange(current_forest.nleafs * 64, dtype=np.float64).reshape(
        current_forest.nleafs, 1, 4, 4, 4
    )
    mesh.load_interior_data(values)
    mesh.apply_ghost_cells()
    padded = mesh.padded_view()
    current = padded[primary].transpose(3, 0, 1, 2)[None].copy()

    interior_lower = i3(2, 2, 2)
    interior_upper = i3(6, 6, 6)
    requested_lower = i3(0, 0, 0)
    requested_upper = i3(8, 8, 8)
    one_direction = np.ascontiguousarray(direction[None])
    target_lower = np.empty((1, 3), dtype=np.int64)
    target_upper = np.empty((1, 3), dtype=np.int64)
    fill_directed_halo_target_boxes(
        interior_lower,
        interior_upper,
        requested_lower,
        requested_upper,
        one_direction,
        target_lower,
        target_upper,
    )
    source_ids = sources[primary, direction_index, :4]
    phases = np.asarray(
        [
            sum(
                (int(forest.node_coords[forest.leaf_node_ids[int(source)], axis]) & 1) << axis
                for axis in range(3)
            )
            for source in source_ids
        ],
        dtype=np.uint8,
    )
    directions = np.repeat(one_direction, 4, axis=0)
    targets_lower = np.repeat(target_lower, 4, axis=0)
    targets_upper = np.repeat(target_upper, 4, axis=0)
    placed_lower, placed_upper, source_lower, source_upper = production(
        interior_lower,
        interior_upper,
        directions,
        phases,
        targets_lower,
        targets_upper,
    )
    actual = np.full_like(current, np.nan)
    for row, source in enumerate(source_ids):
        fine = padded[int(source)].transpose(3, 0, 1, 2)[None]
        restrict_cartesian_2to1_into(
            fine,
            source_lower[row],
            source_upper[row],
            actual,
            placed_lower[row],
        )
    target = tuple(
        slice(int(target_lower[0, axis]), int(target_upper[0, axis]))
        for axis in range(3)
    )
    assert np.array_equal(actual[(0, 0, *target)], current[(0, 0, *target)])
