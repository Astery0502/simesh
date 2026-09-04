from __future__ import annotations

import math

import numpy as np
import pytest

from simesh_rewrite.curl import cartesian_curl_into
from simesh_rewrite.curl_reference import cartesian_curl_reference
from simesh_rewrite.operators import central_difference_into, scaled_difference_into


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def call_curl(
    source: np.ndarray,
    spacing: np.ndarray,
    destination: np.ndarray,
    *,
    output_lower: np.ndarray | None = None,
    output_upper: np.ndarray | None = None,
    source_fields: np.ndarray | None = None,
    destination_fields: np.ndarray | None = None,
    destination_lower: np.ndarray | None = None,
) -> None:
    if output_lower is None:
        output_lower = i3(1, 1, 1)
    if output_upper is None:
        output_upper = np.asarray(source.shape[2:], dtype=np.int64) - 1
    if source_fields is None:
        source_fields = i3(0, 1, 2)
    if destination_fields is None:
        destination_fields = i3(0, 1, 2)
    if destination_lower is None:
        destination_lower = i3(0, 0, 0)
    cartesian_curl_into(
        source,
        i3(0, 0, 0),
        np.asarray(source.shape[2:], dtype=np.int64),
        output_lower,
        output_upper,
        source_fields,
        spacing,
        destination,
        destination_fields,
        destination_lower,
    )


def test_mixed_spacing_translated_fields_match_scalar_reference_bitwise() -> None:
    rng = np.random.default_rng(20260904)
    source = rng.normal(size=(3, 4, 6, 7, 8)).astype(np.float64)
    spacing = np.asarray(
        ((0.5, 0.25, 1.0), (0.125, 0.75, 0.375), (2.0, 1.5, 0.25)),
        dtype=np.float64,
    )
    actual = np.full((3, 5, 7, 8, 9), -91.0)
    expected = actual.copy()
    output_lower = i3(1, 2, 1)
    output_upper = i3(5, 6, 7)
    source_fields = i3(3, 1, 2)
    destination_fields = i3(4, 0, 2)
    destination_lower = i3(2, 1, 2)
    cartesian_curl_reference(
        source,
        output_lower,
        output_upper,
        source_fields,
        spacing,
        expected,
        destination_fields,
        destination_lower,
    )
    cartesian_curl_into(
        source,
        i3(0, 0, 0),
        i3(6, 7, 8),
        output_lower,
        output_upper,
        source_fields,
        spacing,
        actual,
        destination_fields,
        destination_lower,
    )
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_mixed_spacing_matches_per_slot_opr_002_opr_001_composition() -> None:
    rng = np.random.default_rng(321)
    source = rng.normal(size=(3, 3, 6, 6, 6)).astype(np.float64)
    spacing = np.asarray(
        ((0.5, 0.25, 1.0), (0.25, 0.125, 0.5), (1.0, 0.5, 2.0)),
        dtype=np.float64,
    )
    actual = np.empty((3, 3, 4, 4, 4), dtype=np.float64)
    expected = np.empty_like(actual)
    call_curl(source, spacing, actual)
    zero = i3(0, 0, 0)
    extent = i3(4, 4, 4)
    derivative_terms = (
        (2, 1),
        (1, 2),
        (0, 2),
        (2, 0),
        (1, 0),
        (0, 1),
    )
    for slot in range(source.shape[0]):
        derivatives = np.empty((1, 6, 4, 4, 4), dtype=np.float64)
        for term, (field, axis) in enumerate(derivative_terms):
            central_difference_into(
                source[slot : slot + 1],
                zero,
                i3(6, 6, 6),
                i3(1, 1, 1),
                i3(5, 5, 5),
                field,
                axis,
                np.ascontiguousarray(spacing[slot]),
                derivatives,
                term,
                zero,
            )
        for component, (left, right) in enumerate(((0, 1), (2, 3), (4, 5))):
            scaled_difference_into(
                derivatives,
                zero,
                extent,
                left,
                right,
                1.0,
                expected[slot : slot + 1],
                component,
                zero,
            )
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_affine_vector_field_reproduces_constant_curl_for_mixed_slots() -> None:
    spacing = np.asarray(((0.5, 0.25, 1.0), (0.25, 0.125, 0.5)))
    source = np.empty((2, 3, 6, 6, 6), dtype=np.float64)
    for slot in range(2):
        x = np.arange(6)[:, None, None] * spacing[slot, 0]
        y = np.arange(6)[None, :, None] * spacing[slot, 1]
        z = np.arange(6)[None, None, :] * spacing[slot, 2]
        source[slot, 0] = 2.0 * y + 3.0 * z
        source[slot, 1] = 7.0 * x + 5.0 * z
        source[slot, 2] = 11.0 * x + 13.0 * y
    output = np.empty((2, 3, 4, 4, 4), dtype=np.float64)
    call_curl(source, spacing, output)
    expected = np.asarray((8.0, -8.0, 5.0))
    assert np.allclose(output, expected[None, :, None, None, None], atol=1e-14)


@pytest.mark.parametrize(
    ("positive_negative", "negative_negative", "expected_bits"),
    [
        (False, False, 0x0000000000000000),
        (False, True, 0x0000000000000000),
        (True, False, 0x8000000000000000),
        (True, True, 0x0000000000000000),
    ],
    ids=("+0-minus-+0", "+0-minus--0", "-0-minus-+0", "-0-minus--0"),
)
def test_all_derivative_signed_zero_pairs(
    positive_negative: bool,
    negative_negative: bool,
    expected_bits: int,
) -> None:
    source = np.zeros((1, 3, 3, 3, 3), dtype=np.float64)
    if positive_negative:
        source[0, 2, 1, 2, 1] = -0.0
    if negative_negative:
        source[0, 1, 1, 1, 2] = -0.0
    output = np.empty((1, 3, 1, 1, 1), dtype=np.float64)
    call_curl(
        source,
        np.ones((1, 3), dtype=np.float64),
        output,
        output_upper=i3(2, 2, 2),
    )
    assert int(output[0, 0, 0, 0, 0].view(np.uint64)) == expected_bits


@pytest.mark.parametrize(
    ("field", "delta", "component"),
    [
        (2, (0, 1, 0), 0),
        (2, (0, -1, 0), 0),
        (1, (0, 0, 1), 0),
        (1, (0, 0, -1), 0),
        (0, (0, 0, 1), 1),
        (0, (0, 0, -1), 1),
        (2, (1, 0, 0), 1),
        (2, (-1, 0, 0), 1),
        (1, (1, 0, 0), 2),
        (1, (-1, 0, 0), 2),
        (0, (0, 1, 0), 2),
        (0, (0, -1, 0), 2),
    ],
)
def test_every_neighbor_is_loaded(
    field: int,
    delta: tuple[int, int, int],
    component: int,
) -> None:
    source = np.zeros((1, 3, 3, 3, 3), dtype=np.float64)
    index = (1 + delta[0], 1 + delta[1], 1 + delta[2])
    source[(0, field, *index)] = np.nan
    output = np.empty((1, 3, 1, 1, 1), dtype=np.float64)
    call_curl(
        source,
        np.ones((1, 3), dtype=np.float64),
        output,
        output_upper=i3(2, 2, 2),
    )
    assert np.isnan(output[0, component, 0, 0, 0])


def test_smooth_field_has_second_order_convergence() -> None:
    errors = []
    for count in (12, 24, 48):
        spacing_value = 1.0 / count
        spacing = np.full((1, 3), spacing_value)
        coordinates = (np.arange(count + 2) - 1 + 0.5) * spacing_value
        source = np.zeros((1, 3, count + 2, count + 2, count + 2))
        source[0, 2] = np.sin(coordinates)[None, :, None]
        output = np.empty((1, 3, count, count, count))
        call_curl(
            source,
            spacing,
            output,
            output_upper=i3(count + 1, count + 1, count + 1),
        )
        expected = np.cos(coordinates[1:-1])[None, :, None]
        errors.append(float(np.max(np.abs(output[0, 0] - expected))))
    orders = [math.log(errors[i] / errors[i + 1], 2.0) for i in range(2)]
    assert min(orders) >= 1.8


@pytest.mark.parametrize(
    "failure",
    ("spacing", "source_field", "destination_field", "reach", "output", "alias"),
)
def test_validation_failures_preserve_destination(failure: str) -> None:
    source = np.ones((2, 3, 5, 5, 5), dtype=np.float64)
    spacing = np.ones((2, 3), dtype=np.float64)
    destination = np.full((2, 3, 3, 3, 3), -77.0)
    before = destination.copy()
    valid_lower = i3(0, 0, 0)
    valid_upper = i3(5, 5, 5)
    output_lower = i3(1, 1, 1)
    output_upper = i3(4, 4, 4)
    source_fields = i3(0, 1, 2)
    destination_fields = i3(0, 1, 2)
    destination_lower = i3(0, 0, 0)
    if failure == "spacing":
        spacing[-1, -1] = np.nan
    elif failure == "source_field":
        source_fields[-1] = 3
    elif failure == "destination_field":
        destination_fields[-1] = 1
    elif failure == "reach":
        valid_upper[-1] = 4
    elif failure == "output":
        destination_lower[0] = 1
    else:
        destination = source[:, :, 1:4, 1:4, 1:4]
        before = destination.copy()
    with pytest.raises(ValueError):
        cartesian_curl_into(
            source,
            valid_lower,
            valid_upper,
            output_lower,
            output_upper,
            source_fields,
            spacing,
            destination,
            destination_fields,
            destination_lower,
        )
    assert np.array_equal(destination.view(np.uint64), before.view(np.uint64))


def test_empty_box_and_slots_are_noops_after_metadata_validation() -> None:
    source = np.empty((0, 3, 3, 3, 3), dtype=np.float64)
    destination = np.empty((0, 3, 1, 1, 1), dtype=np.float64)
    call_curl(
        source,
        np.empty((0, 3), dtype=np.float64),
        destination,
        output_upper=i3(1, 2, 2),
    )
    nonempty_source = np.ones((1, 3, 3, 3, 3))
    preserved = np.full((1, 3, 1, 1, 1), -9.0)
    before = preserved.copy()
    call_curl(
        nonempty_source,
        np.ones((1, 3)),
        preserved,
        output_upper=i3(1, 2, 2),
    )
    assert np.array_equal(preserved, before)
