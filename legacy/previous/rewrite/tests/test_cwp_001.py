from __future__ import annotations

import itertools

import numpy as np
import pytest

from simesh_rewrite.coarser_workspace import fill_coarser_workspace_boxes
from simesh_rewrite.coarser_workspace_reference import (
    coarser_workspace_boxes_reference,
)
from simesh_rewrite.foundation import copy_region_into
from simesh_rewrite.prolongation import prolong_cartesian_2to1_into
from simesh_rewrite.prolongation_reference import (
    prolong_cartesian_2to1_reference,
)
from simesh_rewrite.target_boxes import fill_directed_halo_target_boxes


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def all_direction_phase_rows():
    directions: list[tuple[int, int, int]] = []
    phases: list[int] = []
    for direction in itertools.product((-1, 0, 1), repeat=3):
        if direction == (0, 0, 0):
            continue
        for phase in range(8):
            directions.append(direction)
            phases.append(phase)
    return np.asarray(directions, dtype=np.int64), np.asarray(phases, dtype=np.uint8)


def targets(directions: np.ndarray):
    lower = i3(3, 4, 5)
    upper = i3(11, 14, 17)
    requested_lower = i3(1, 2, 3)
    requested_upper = i3(13, 16, 19)
    target_lower = np.empty_like(directions)
    target_upper = np.empty_like(directions)
    fill_directed_halo_target_boxes(
        lower,
        upper,
        requested_lower,
        requested_upper,
        directions,
        target_lower,
        target_upper,
    )
    return lower, upper, target_lower, target_upper


def production(*inputs):
    outputs = tuple(np.full_like(inputs[2], -77) for _ in range(7))
    fill_coarser_workspace_boxes(*inputs, *outputs)
    return outputs


def test_all_208_direction_phase_rows_match_reference_and_containment() -> None:
    directions, phases = all_direction_phase_rows()
    assert directions.shape == (208, 3)
    assert phases.shape == (208,)
    lower, upper, target_lower, target_upper = targets(directions)
    actual = production(
        lower, upper, directions, phases, target_lower, target_upper
    )
    expected = coarser_workspace_boxes_reference(
        lower, upper, directions, phases, target_lower, target_upper
    )
    for value, reference in zip(actual, expected, strict=True):
        assert np.array_equal(value, reference)

    source_lower, source_upper, workspace_lower, workspace_upper, required_lower, required_upper, _ = actual
    assert np.all(source_lower >= lower)
    assert np.all(source_upper <= upper)
    assert np.array_equal(source_upper - source_lower, workspace_upper - workspace_lower)
    assert np.all(workspace_lower >= 0)
    assert np.all(workspace_lower >= required_lower)
    assert np.all(workspace_upper <= required_upper)


def test_source_copy_uses_only_intersection_and_preserves_uncovered_reach() -> None:
    lower = i3(2, 2, 2)
    upper = i3(10, 10, 10)
    directions = np.asarray([(-1, 0, 0)], dtype=np.int64)
    phases = np.asarray([0], dtype=np.uint8)
    target_lower = np.asarray([[0, 2, 2]], dtype=np.int64)
    target_upper = np.asarray([[2, 10, 10]], dtype=np.int64)
    source_lower, source_upper, workspace_lower, workspace_upper, _, required_upper, _ = production(
        lower, upper, directions, phases, target_lower, target_upper
    )
    source = np.arange(12**3, dtype=np.float64).reshape(1, 1, 12, 12, 12)
    shape = tuple(int(value) for value in required_upper[0])
    workspace = np.full((1, 1, *shape), np.nan, dtype=np.float64)
    copy_region_into(
        source,
        source_lower[0],
        workspace,
        workspace_lower[0],
        source_upper[0] - source_lower[0],
    )
    copied = tuple(
        slice(int(workspace_lower[0, axis]), int(workspace_upper[0, axis]))
        for axis in range(3)
    )
    assert not np.isnan(workspace[(0, 0, *copied)]).any()
    assert np.isnan(workspace).any()


@pytest.mark.parametrize(
    ("direction", "phase", "target_lower", "target_upper"),
    [
        ((-1, 0, 0), 4, (0, 2, 2), (2, 10, 10)),
        ((-1, 0, 0), 1, (0, 2, 2), (2, 10, 10)),
        ((1, 0, 0), 0, (10, 2, 2), (12, 10, 10)),
    ],
)
def test_emitted_origin_drives_prl_mapping_for_matching_and_mismatched_phases(
    direction, phase, target_lower, target_upper
) -> None:
    lower = i3(2, 2, 2)
    upper = i3(10, 10, 10)
    directions = np.asarray([direction], dtype=np.int64)
    phases = np.asarray([phase], dtype=np.uint8)
    target_lower = np.asarray([target_lower], dtype=np.int64)
    target_upper = np.asarray([target_upper], dtype=np.int64)
    _, _, _, _, required_lower, required_upper, origin = production(
        lower, upper, directions, phases, target_lower, target_upper
    )
    shape = tuple(int(value) for value in required_upper[0])
    coarse = np.empty((1, 1, *shape), dtype=np.float64)
    for i, j, k in itertools.product(*(range(value) for value in shape)):
        coarse[0, 0, i, j, k] = 3.0 * i + 5.0 * j + 7.0 * k
    actual = np.full((1, 1, 12, 12, 12), np.nan, dtype=np.float64)
    expected = actual.copy()
    prolong_cartesian_2to1_into(
        coarse,
        required_lower[0],
        required_upper[0],
        origin[0],
        actual,
        target_lower[0],
        target_upper[0],
        lower,
    )
    prolong_cartesian_2to1_reference(
        coarse,
        required_lower[0],
        required_upper[0],
        origin[0],
        expected,
        target_lower[0],
        target_upper[0],
        lower,
    )
    assert np.array_equal(actual, expected, equal_nan=True)


def test_empty_target_row_is_exact_zero_after_full_validation() -> None:
    lower = i3(2, 2, 2)
    upper = i3(10, 10, 10)
    directions = np.asarray([(-1, 0, 0)], dtype=np.int64)
    phases = np.asarray([0], dtype=np.uint8)
    target_lower = np.asarray([[2, 2, 2]], dtype=np.int64)
    target_upper = np.asarray([[2, 10, 10]], dtype=np.int64)
    outputs = production(
        lower, upper, directions, phases, target_lower, target_upper
    )
    for output in outputs:
        assert np.array_equal(output, np.zeros((1, 3), dtype=np.int64))


@pytest.mark.parametrize(
    ("upper", "direction", "phase", "target_lower", "target_upper", "message"),
    [
        ((10, 10, 10), (-1, 0, 0), 8, (0, 2, 2), (2, 10, 10), "outside"),
        ((9, 10, 10), (-1, 0, 0), 0, (0, 2, 2), (2, 10, 10), "positive and even"),
        ((10, 10, 10), (0, 0, 0), 0, (2, 2, 2), (10, 10, 10), "noncenter"),
        ((10, 10, 10), (-1, 0, 0), 0, (-1, 2, 2), (2, 10, 10), "ordered and nonnegative"),
    ],
)
def test_invalid_semantics_are_atomic(
    upper, direction, phase, target_lower, target_upper, message
) -> None:
    inputs = (
        i3(2, 2, 2),
        i3(*upper),
        np.asarray([direction], dtype=np.int64),
        np.asarray([phase], dtype=np.uint8),
        np.asarray([target_lower], dtype=np.int64),
        np.asarray([target_upper], dtype=np.int64),
    )
    outputs = tuple(np.asarray([[71, 72, 73]], dtype=np.int64) for _ in range(7))
    before = tuple(output.copy() for output in outputs)
    with pytest.raises(ValueError, match=message):
        fill_coarser_workspace_boxes(*inputs, *outputs)
    for output, original in zip(outputs, before, strict=True):
        assert np.array_equal(output, original)


def test_read_only_empty_rows_and_output_alias_rejection() -> None:
    lower = i3(2, 2, 2)
    upper = i3(10, 10, 10)
    directions = np.empty((0, 3), dtype=np.int64)
    phases = np.empty(0, dtype=np.uint8)
    inputs = [lower, upper, directions, phases, directions, directions]
    for value in inputs:
        value.setflags(write=False)
    outputs = tuple(np.empty((0, 3), dtype=np.int64) for _ in range(7))
    fill_coarser_workspace_boxes(*inputs, *outputs)

    directions = np.asarray([(-1, 0, 0)], dtype=np.int64)
    phases = np.asarray([0], dtype=np.uint8)
    target_lower = np.asarray([[0, 2, 2]], dtype=np.int64)
    target_upper = np.asarray([[2, 10, 10]], dtype=np.int64)
    shared = np.empty((1, 3), dtype=np.int64)
    with pytest.raises(ValueError, match="must not overlap"):
        fill_coarser_workspace_boxes(
            lower,
            upper,
            directions,
            phases,
            target_lower,
            target_upper,
            shared,
            shared,
            *(np.empty_like(shared) for _ in range(5)),
        )


def test_representation_readonly_and_input_alias_errors_are_atomic() -> None:
    lower = i3(2, 2, 2)
    upper = i3(10, 10, 10)
    directions = np.asarray([(-1, 0, 0)], dtype=np.int64)
    phases = np.asarray([0], dtype=np.uint8)
    target_lower = np.asarray([[0, 2, 2]], dtype=np.int64)
    target_upper = np.asarray([[2, 10, 10]], dtype=np.int64)

    for bad_directions, bad_phases, error in (
        (directions.astype(np.int32), phases, TypeError),
        (directions, phases.astype(np.int64), TypeError),
        (np.asarray([[0, 0, -1]], dtype=np.int64)[:, ::-1], phases, ValueError),
    ):
        outputs = tuple(np.full((1, 3), 73, dtype=np.int64) for _ in range(7))
        before = tuple(value.copy() for value in outputs)
        with pytest.raises(error):
            fill_coarser_workspace_boxes(
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

    outputs = tuple(np.full((1, 3), 79, dtype=np.int64) for _ in range(7))
    outputs[0].setflags(write=False)
    before = tuple(value.copy() for value in outputs)
    with pytest.raises(ValueError, match="must be writable"):
        fill_coarser_workspace_boxes(
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
        fill_coarser_workspace_boxes(
            lower,
            upper,
            directions,
            phases,
            target_lower,
            target_upper,
            target_lower,
            *(np.empty_like(target_lower) for _ in range(6)),
        )
    assert np.array_equal(target_lower, aliased_before)


def test_derived_int64_overflow_is_atomic() -> None:
    index_max = np.iinfo(np.int64).max
    lower = i3(2, int(index_max) - 8, 2)
    upper = i3(10, int(index_max), 10)
    directions = np.asarray([(-1, 0, 0)], dtype=np.int64)
    phases = np.asarray([2], dtype=np.uint8)
    target_lower = np.asarray(
        [[0, int(index_max) - 8, 2]], dtype=np.int64
    )
    target_upper = np.asarray([[2, int(index_max), 10]], dtype=np.int64)
    outputs = tuple(np.full((1, 3), 83, dtype=np.int64) for _ in range(7))
    before = tuple(value.copy() for value in outputs)

    with pytest.raises(OverflowError, match="coarse required upper"):
        fill_coarser_workspace_boxes(
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
