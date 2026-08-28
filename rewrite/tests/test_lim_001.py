from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from simesh_rewrite.limiter import three_point_limited_slope
from simesh_rewrite.limiter_reference import (
    three_point_limited_slope_reference,
)
from simesh_rewrite.restriction import restrict_cartesian_2to1_into


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def bits(value: float) -> int:
    return int(np.asarray([value], dtype=np.float64).view(np.uint64)[0])


def from_bits(value: int) -> np.float64:
    return np.asarray([value], dtype=np.uint64).view(np.float64)[0]


@pytest.mark.parametrize(
    ("left", "center", "right", "expected"),
    [
        (0.0, 1.0, 3.0, 1.0),
        (0.0, 2.0, 3.0, 1.0),
        (0.0, 1.0, 2.0, 1.0),
        (3.0, 2.0, 0.0, -1.0),
        (3.0, 1.0, 0.0, -1.0),
        (2.0, 1.0, 0.0, -1.0),
    ],
)
def test_monotone_branches_and_ties(
    left: float,
    center: float,
    right: float,
    expected: float,
) -> None:
    actual = three_point_limited_slope(left, center, right)
    reference = three_point_limited_slope_reference(left, center, right)
    assert bits(actual) == bits(reference) == bits(expected)


@pytest.mark.parametrize(
    ("left", "center", "right"),
    [
        (5.0, 5.0, 5.0),
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (0.0, 2.0, 1.0),
        (2.0, 0.0, 1.0),
    ],
)
def test_plateaus_extrema_and_one_sided_zero_return_positive_zero(
    left: float,
    center: float,
    right: float,
) -> None:
    actual = three_point_limited_slope(left, center, right)
    assert bits(actual) == 0
    assert bits(actual) == bits(
        three_point_limited_slope_reference(left, center, right)
    )


def test_large_offset_uses_separately_rounded_one_sided_differences() -> None:
    base = np.float64(1.0e16)
    left = base
    center = np.float64(base + np.float64(2.0))
    right = np.float64(base + np.float64(6.0))
    actual = three_point_limited_slope(left, center, right)
    assert bits(actual) == bits(np.float64(2.0))
    assert bits(actual) == bits(
        three_point_limited_slope_reference(left, center, right)
    )


@pytest.mark.parametrize("descending", [False, True])
def test_centered_sum_overflow_still_clips_to_finite_difference(
    descending: bool,
) -> None:
    maximum = np.float64(np.finfo(np.float64).max)
    half = np.float64(maximum * np.float64(0.5))
    if descending:
        left, center, right = half, np.float64(0.0), np.float64(-maximum)
        expected = np.float64(-half)
    else:
        left, center, right = np.float64(-maximum), np.float64(0.0), half
        expected = half
    with np.errstate(all="ignore"):
        actual = three_point_limited_slope(left, center, right)
        reference = three_point_limited_slope_reference(left, center, right)
    assert np.isfinite(actual)
    assert bits(actual) == bits(reference) == bits(expected)


def test_all_signed_zero_combinations_return_exact_positive_zero() -> None:
    zeros = (from_bits(0), from_bits(0x8000000000000000))
    for left, center, right in product(zeros, repeat=3):
        actual = three_point_limited_slope(left, center, right)
        reference = three_point_limited_slope_reference(left, center, right)
        assert bits(actual) == bits(reference) == 0


def test_smallest_subnormal_slopes_are_preserved() -> None:
    smallest = np.nextafter(np.float64(0.0), np.float64(1.0))
    cases = (
        (np.float64(0.0), smallest, np.float64(3.0) * smallest, smallest),
        (
            np.float64(0.0),
            -smallest,
            np.float64(-3.0) * smallest,
            -smallest,
        ),
    )
    for left, center, right, expected in cases:
        actual = three_point_limited_slope(left, center, right)
        reference = three_point_limited_slope_reference(left, center, right)
        assert bits(actual) == bits(reference) == bits(expected)


def test_nan_in_each_position_and_payload_returns_positive_zero() -> None:
    nan_patterns = (
        0x7FF8000000001234,
        0xFFF8000000005678,
        0x7FF0000000004321,
    )
    with np.errstate(all="ignore"):
        for pattern in nan_patterns:
            nan_value = from_bits(pattern)
            for position in range(3):
                values = [np.float64(0.0), np.float64(1.0), np.float64(2.0)]
                values[position] = nan_value
                actual = three_point_limited_slope(*values)
                reference = three_point_limited_slope_reference(*values)
                assert bits(actual) == bits(reference) == 0


@pytest.mark.parametrize(
    ("left", "center", "right", "expected"),
    [
        (-np.inf, 0.0, np.inf, np.inf),
        (np.inf, 0.0, -np.inf, -np.inf),
        (0.0, 1.0, np.inf, 1.0),
        (np.inf, 1.0, 0.0, -1.0),
        (0.0, np.inf, np.inf, 0.0),
        (-np.inf, -np.inf, 0.0, 0.0),
        (0.0, np.inf, 0.0, 0.0),
    ],
)
def test_infinity_combinations(
    left: float,
    center: float,
    right: float,
    expected: float,
) -> None:
    with np.errstate(all="ignore"):
        actual = three_point_limited_slope(left, center, right)
        reference = three_point_limited_slope_reference(left, center, right)
    assert bits(actual) == bits(reference) == bits(expected)


def test_random_finite_values_match_reference_and_minmod_invariant() -> None:
    rng = np.random.default_rng(20260830)
    values = rng.normal(size=(10_000, 3)).astype(np.float64)
    for left, center, right in values:
        actual = three_point_limited_slope(left, center, right)
        reference = three_point_limited_slope_reference(left, center, right)
        assert bits(actual) == bits(reference)

        slope_l = np.float64(center - left)
        slope_r = np.float64(right - center)
        if slope_l > 0.0 and slope_r > 0.0:
            expected = slope_l if slope_l < slope_r else slope_r
        elif slope_l < 0.0 and slope_r < 0.0:
            negative_l = np.float64(-slope_l)
            negative_r = np.float64(-slope_r)
            magnitude = (
                negative_l if negative_l < negative_r else negative_r
            )
            expected = np.float64(-magnitude)
        else:
            expected = np.float64(0.0)
        assert bits(actual) == bits(expected)


def test_exact_scalar_types_validation_precedence_and_output_type() -> None:
    assert type(three_point_limited_slope(0.0, 1.0, 2.0)) is float
    assert type(
        three_point_limited_slope(
            np.float64(0.0), np.float64(1.0), np.float64(2.0)
        )
    ) is float
    assert type(three_point_limited_slope_reference(0.0, 1.0, 2.0)) is float

    class FloatSubclass(float):
        pass

    invalid_values = (
        1,
        True,
        np.float32(1.0),
        np.longdouble(1.0),
        np.array(1.0, dtype=np.float64),
        FloatSubclass(1.0),
    )
    for invalid in invalid_values:
        with pytest.raises(TypeError, match="left_value"):
            three_point_limited_slope(invalid, invalid, invalid)
        with pytest.raises(TypeError, match="center_value"):
            three_point_limited_slope(0.0, invalid, invalid)
        with pytest.raises(TypeError, match="right_value"):
            three_point_limited_slope(0.0, 1.0, invalid)


def test_rst_produced_coarse_values_support_explicit_scalar_reconstruction() -> None:
    """Non-stable consumer evidence; PRL will own reconstruction semantics."""
    x = np.arange(8, dtype=np.float64)[:, None, None]
    fine = np.empty((1, 4, 8, 4, 4), dtype=np.float64)
    fine[0, 0].fill(5.0)
    fine[0, 1] = np.broadcast_to(x + 0.5, (8, 4, 4))
    fine[0, 2] = np.broadcast_to(np.where(x < 4, 0.0, 10.0), (8, 4, 4))
    fine[0, 3] = np.broadcast_to((x + 0.5) ** 2, (8, 4, 4))
    coarse = np.empty((1, 4, 4, 2, 2), dtype=np.float64)
    restrict_cartesian_2to1_into(
        fine,
        i3(0, 0, 0),
        i3(8, 4, 4),
        coarse,
        i3(0, 0, 0),
    )

    slopes = []
    reconstructions = []
    for field in range(4):
        left = coarse[0, field, 0, 0, 0]
        center = coarse[0, field, 1, 0, 0]
        right = coarse[0, field, 2, 0, 0]
        slope = three_point_limited_slope(left, center, right)
        reference = three_point_limited_slope_reference(left, center, right)
        assert bits(slope) == bits(reference)
        lower_child = np.float64(
            center + np.float64(np.float64(slope) * np.float64(-0.25))
        )
        upper_child = np.float64(
            center + np.float64(np.float64(slope) * np.float64(0.25))
        )
        slopes.append(slope)
        reconstructions.append((lower_child, upper_child))

    assert bits(slopes[0]) == 0
    assert reconstructions[0] == (np.float64(5.0), np.float64(5.0))
    assert slopes[1] == np.float64(2.0)
    assert reconstructions[1] == (
        fine[0, 1, 2, 0, 0],
        fine[0, 1, 3, 0, 0],
    )
    assert bits(slopes[2]) == 0
    assert reconstructions[2] == (np.float64(0.0), np.float64(0.0))
    assert slopes[3] == np.float64(8.0)
    assert coarse[0, 3, 0, 0, 0] <= reconstructions[3][0]
    assert reconstructions[3][1] <= coarse[0, 3, 2, 0, 0]
