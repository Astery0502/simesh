from __future__ import annotations

import math

import numpy as np
import pytest

from simesh_rewrite._rk4 import (
    field_line_rk4_finish_unchecked,
    field_line_rk4_stage_unchecked,
)
from simesh_rewrite.rk4 import (
    field_line_rk4_finish_into,
    field_line_rk4_stage_into,
)
from simesh_rewrite.rk4_reference import (
    field_line_rk4_finish_reference,
    field_line_rk4_stage_reference,
)


SENTINEL_BITS = np.uint64(0x7FF800000000C701)
SENTINEL = np.asarray([SENTINEL_BITS], dtype=np.uint64).view(np.float64)[0]


def assert_ieee_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.shape == expected.shape
    finite = np.isfinite(expected)
    assert np.array_equal(
        actual[finite].view(np.uint64), expected[finite].view(np.uint64)
    )
    assert np.array_equal(np.isnan(actual), np.isnan(expected))
    assert np.array_equal(np.isposinf(actual), np.isposinf(expected))
    assert np.array_equal(np.isneginf(actual), np.isneginf(expected))


@pytest.mark.parametrize("half_step", [True, False])
def test_empty_and_finite_batch_stage_match_scalar_tree_bitwise(
    half_step: bool,
) -> None:
    empty_base = np.empty((0, 4), dtype=np.float64)
    empty_rhs = np.empty((0, 4), dtype=np.float64)
    empty_output = np.empty((0, 4), dtype=np.float64)
    field_line_rk4_stage_into(
        empty_base, empty_rhs, 0.25, half_step, empty_output
    )

    rng = np.random.default_rng(20260904)
    base = rng.normal(size=(37, 4)).astype(np.float64)
    rhs = rng.normal(size=(37, 4)).astype(np.float64)
    actual = np.full((37, 4), SENTINEL)
    expected = actual.copy()
    field_line_rk4_stage_reference(base, rhs, 0.375, half_step, expected)
    field_line_rk4_stage_into(base, rhs, 0.375, half_step, actual)
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_empty_and_finite_batch_finish_match_scalar_tree_bitwise() -> None:
    empty_inputs = [np.empty((0, 4), dtype=np.float64) for _ in range(5)]
    empty_output = np.empty((0, 4), dtype=np.float64)
    field_line_rk4_finish_into(*empty_inputs, 0.25, empty_output)

    rng = np.random.default_rng(77123)
    arrays = [rng.normal(size=(41, 4)).astype(np.float64) for _ in range(5)]
    actual = np.full((41, 4), SENTINEL)
    expected = actual.copy()
    field_line_rk4_finish_reference(*arrays, 0.625, expected)
    field_line_rk4_finish_into(*arrays, 0.625, actual)
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("half_step", [True, False])
@pytest.mark.parametrize(
    "step_size",
    [np.finfo(np.float64).tiny, 0.5, np.finfo(np.float64).max],
    ids=("minimum-normal", "ordinary", "maximum-finite"),
)
def test_stage_signed_zero_cancellation_subnormal_and_large_values(
    half_step: bool,
    step_size: float,
) -> None:
    subnormal = np.nextafter(np.float64(0.0), np.float64(1.0))
    maximum = np.finfo(np.float64).max
    base = np.asarray(
        (
            (0.0, -0.0, 1.0, -1.0),
            (subnormal, -subnormal, maximum, -maximum),
        ),
        dtype=np.float64,
    )
    rhs = np.asarray(
        (
            (-0.0, 0.0, -2.0, 2.0),
            (1.0, -1.0, maximum, maximum),
        ),
        dtype=np.float64,
    )
    actual = np.full_like(base, SENTINEL)
    expected = actual.copy()
    field_line_rk4_stage_reference(
        base, rhs, step_size, half_step, expected
    )
    field_line_rk4_stage_into(base, rhs, step_size, half_step, actual)
    assert_ieee_equal(actual, expected)


@pytest.mark.parametrize("source_name", ["base", "rhs"])
@pytest.mark.parametrize("component", range(4))
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_stage_propagates_each_nonfinite_input_position(
    source_name: str,
    component: int,
    nonfinite: float,
) -> None:
    base = np.ones((1, 4), dtype=np.float64)
    rhs = np.full((1, 4), 0.25, dtype=np.float64)
    (base if source_name == "base" else rhs)[0, component] = nonfinite
    actual = np.full((1, 4), SENTINEL)
    expected = actual.copy()
    field_line_rk4_stage_reference(base, rhs, 0.5, True, expected)
    field_line_rk4_stage_into(base, rhs, 0.5, True, actual)
    assert_ieee_equal(actual, expected)


@pytest.mark.parametrize("source_position", range(5), ids=("base", "k1", "k2", "k3", "k4"))
@pytest.mark.parametrize("component", range(4))
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_finish_propagates_each_nonfinite_input_position(
    source_position: int,
    component: int,
    nonfinite: float,
) -> None:
    arrays = [np.full((1, 4), 0.25, dtype=np.float64) for _ in range(5)]
    arrays[0].fill(1.0)
    arrays[source_position][0, component] = nonfinite
    actual = np.full((1, 4), SENTINEL)
    expected = actual.copy()
    field_line_rk4_finish_reference(*arrays, 0.5, expected)
    field_line_rk4_finish_into(*arrays, 0.5, actual)
    assert_ieee_equal(actual, expected)


def test_finish_frozen_cancellation_and_large_tree_matches_reference() -> None:
    maximum = np.finfo(np.float64).max
    subnormal = np.nextafter(np.float64(0.0), np.float64(1.0))
    base = np.asarray(((0.0, -0.0, maximum, subnormal),), dtype=np.float64)
    k1 = np.asarray(((1.0, -1.0, maximum, subnormal),))
    k2 = np.asarray(((-0.5, 0.5, maximum, -subnormal),))
    k3 = np.asarray(((0.25, -0.25, -maximum, subnormal),))
    k4 = np.asarray(((-0.5, 0.5, maximum, -subnormal),))
    actual = np.full((1, 4), SENTINEL)
    expected = actual.copy()
    field_line_rk4_finish_reference(base, k1, k2, k3, k4, 0.75, expected)
    field_line_rk4_finish_into(base, k1, k2, k3, k4, 0.75, actual)
    assert_ieee_equal(actual, expected)


def test_unchecked_stage_and_finish_symbols_match_checked_boundaries() -> None:
    rng = np.random.default_rng(404)
    base, k1, k2, k3, k4 = [
        rng.normal(size=(9, 4)).astype(np.float64) for _ in range(5)
    ]
    checked_stage = np.empty_like(base)
    unchecked_stage = np.empty_like(base)
    field_line_rk4_stage_into(base, k1, 0.125, True, checked_stage)
    field_line_rk4_stage_unchecked(base, k1, 0.125, True, unchecked_stage)
    assert np.array_equal(
        checked_stage.view(np.uint64), unchecked_stage.view(np.uint64)
    )

    checked_finish = np.empty_like(base)
    unchecked_finish = np.empty_like(base)
    field_line_rk4_finish_into(base, k1, k2, k3, k4, 0.125, checked_finish)
    field_line_rk4_finish_unchecked(
        base, k1, k2, k3, k4, 0.125, unchecked_finish
    )
    assert np.array_equal(
        checked_finish.view(np.uint64), unchecked_finish.view(np.uint64)
    )


@pytest.mark.parametrize("bad_step", [1, True, np.int64(1), 1.0 + 0.0j, None])
def test_nonfloating_step_kind_is_atomic(bad_step) -> None:
    base = np.ones((2, 4), dtype=np.float64)
    output = np.full((2, 4), SENTINEL)
    before = output.copy()
    with pytest.raises(TypeError, match="floating scalar"):
        field_line_rk4_stage_into(base, base, bad_step, True, output)
    assert np.array_equal(output.view(np.uint64), before.view(np.uint64))


@pytest.mark.parametrize(
    "bad_step",
    [
        0.0,
        -1.0,
        np.nextafter(np.finfo(np.float64).tiny, 0.0),
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_invalid_floating_step_value_is_atomic(bad_step: float) -> None:
    base = np.ones((2, 4), dtype=np.float64)
    output = np.full((2, 4), SENTINEL)
    before = output.copy()
    with pytest.raises(ValueError, match="finite, positive, and normal"):
        field_line_rk4_finish_into(base, base, base, base, base, bad_step, output)
    assert np.array_equal(output.view(np.uint64), before.view(np.uint64))


@pytest.mark.parametrize("bad_half", [np.bool_(True), 1, 0.0, None])
def test_half_step_must_be_exact_bool_and_is_atomic(bad_half) -> None:
    base = np.ones((2, 4), dtype=np.float64)
    output = np.full((2, 4), SENTINEL)
    before = output.copy()
    with pytest.raises(TypeError, match="exact bool"):
        field_line_rk4_stage_into(base, base, 0.5, bad_half, output)
    assert np.array_equal(output.view(np.uint64), before.view(np.uint64))


@pytest.mark.parametrize(
    ("failure", "error", "match"),
    [
        ("base_type", TypeError, "NumPy array"),
        ("rhs_dtype", TypeError, "float64"),
        ("base_rank", ValueError, "shape"),
        ("rhs_width", ValueError, "shape"),
        ("rhs_rows", ValueError, "same shape"),
        ("base_layout", ValueError, "C-contiguous"),
        ("output_readonly", ValueError, "writable"),
    ],
)
def test_stage_array_validation_preserves_output(
    failure: str,
    error: type[Exception],
    match: str,
) -> None:
    base = np.ones((2, 4), dtype=np.float64)
    rhs = np.full((2, 4), 2.0)
    output = np.full((2, 4), SENTINEL)
    if failure == "base_type":
        base = [[1.0] * 4] * 2  # type: ignore[assignment]
    elif failure == "rhs_dtype":
        rhs = rhs.astype(np.float32)
    elif failure == "base_rank":
        base = base.reshape(8)
    elif failure == "rhs_width":
        rhs = np.ones((2, 3), dtype=np.float64)
    elif failure == "rhs_rows":
        rhs = np.ones((1, 4), dtype=np.float64)
    elif failure == "base_layout":
        base = np.asfortranarray(base)
    else:
        output.setflags(write=False)
    before = output.copy()
    with pytest.raises(error, match=match):
        field_line_rk4_stage_into(base, rhs, 0.5, True, output)
    assert np.array_equal(output.view(np.uint64), before.view(np.uint64))


@pytest.mark.parametrize("source", ["base", "rhs"])
def test_stage_output_alias_is_atomic(source: str) -> None:
    base = np.ones((2, 4), dtype=np.float64)
    rhs = np.full((2, 4), 2.0)
    output = base if source == "base" else rhs
    before = output.copy()
    with pytest.raises(ValueError, match="must not overlap"):
        field_line_rk4_stage_into(base, rhs, 0.5, False, output)
    assert np.array_equal(output.view(np.uint64), before.view(np.uint64))


@pytest.mark.parametrize("source_position", range(5), ids=("base", "k1", "k2", "k3", "k4"))
def test_finish_output_aliases_are_atomic(source_position: int) -> None:
    arrays = [np.full((2, 4), float(index + 1)) for index in range(5)]
    output = arrays[source_position]
    before = output.copy()
    with pytest.raises(ValueError, match="must not overlap"):
        field_line_rk4_finish_into(*arrays, 0.5, output)
    assert np.array_equal(output.view(np.uint64), before.view(np.uint64))


@pytest.mark.parametrize(
    ("failure", "error", "match"),
    [
        ("k_dtype", TypeError, "float64"),
        ("k_width", ValueError, "shape"),
        ("k_rows", ValueError, "same shape"),
        ("k_layout", ValueError, "C-contiguous"),
        ("output_shape", ValueError, "same shape"),
        ("output_readonly", ValueError, "writable"),
    ],
)
def test_finish_array_validation_preserves_output(
    failure: str,
    error: type[Exception],
    match: str,
) -> None:
    arrays = [np.full((2, 4), float(index + 1)) for index in range(5)]
    output = np.full((2, 4), SENTINEL)
    if failure == "k_dtype":
        arrays[3] = arrays[3].astype(np.float32)
    elif failure == "k_width":
        arrays[3] = np.ones((2, 3), dtype=np.float64)
    elif failure == "k_rows":
        arrays[3] = np.ones((1, 4), dtype=np.float64)
    elif failure == "k_layout":
        arrays[3] = np.asfortranarray(arrays[3])
    elif failure == "output_shape":
        output = np.full((1, 4), SENTINEL)
    else:
        output.setflags(write=False)
    before = output.copy()
    with pytest.raises(error, match=match):
        field_line_rk4_finish_into(*arrays, 0.5, output)
    assert np.array_equal(output.view(np.uint64), before.view(np.uint64))


def test_readonly_inputs_are_accepted_and_preserved() -> None:
    arrays = [np.full((3, 4), float(index + 1)) for index in range(5)]
    copies = [value.copy() for value in arrays]
    for value in arrays:
        value.setflags(write=False)
    stage = np.full((3, 4), SENTINEL)
    candidate = np.full((3, 4), SENTINEL)
    field_line_rk4_stage_into(arrays[0], arrays[1], 0.75, True, stage)
    field_line_rk4_finish_into(*arrays, 0.75, candidate)
    for value, expected in zip(arrays, copies, strict=True):
        assert np.array_equal(value.view(np.uint64), expected.view(np.uint64))
    assert not np.any(stage.view(np.uint64) == SENTINEL_BITS)
    assert not np.any(candidate.view(np.uint64) == SENTINEL_BITS)


def test_constant_augmented_rhs_has_exact_half_full_and_finish_states() -> None:
    base = np.asarray(((1.0, -2.0, 4.0, 8.0), (0.0, 1.0, 2.0, 3.0)))
    rhs = np.asarray(((2.0, -4.0, 8.0, 16.0), (-2.0, 4.0, -8.0, 16.0)))
    half = np.empty_like(base)
    full = np.empty_like(base)
    candidate = np.empty_like(base)
    field_line_rk4_stage_into(base, rhs, 0.75, True, half)
    field_line_rk4_stage_into(base, rhs, 0.75, False, full)
    field_line_rk4_finish_into(base, rhs, rhs, rhs, rhs, 0.75, candidate)
    assert np.array_equal(half, base + 0.375 * rhs)
    assert np.array_equal(full, base + 0.75 * rhs)
    assert np.array_equal(candidate, full)


def rotational_rhs(states: np.ndarray) -> np.ndarray:
    result = np.empty_like(states)
    result[:, 0] = -states[:, 1]
    result[:, 1] = states[:, 0]
    result[:, 2] = 0.0
    result[:, 3] = 1.0
    return result


def integrate_rotation(step_count: int) -> np.ndarray:
    step = float(2.0 * math.pi / step_count)
    state = np.asarray(((1.0, 0.0, 2.0, 0.0),), dtype=np.float64)
    for _ in range(step_count):
        k1 = rotational_rhs(state)
        stage2 = np.empty_like(state)
        field_line_rk4_stage_into(state, k1, step, True, stage2)
        k2 = rotational_rhs(stage2)
        stage3 = np.empty_like(state)
        field_line_rk4_stage_into(state, k2, step, True, stage3)
        k3 = rotational_rhs(stage3)
        stage4 = np.empty_like(state)
        field_line_rk4_stage_into(state, k3, step, False, stage4)
        k4 = rotational_rhs(stage4)
        candidate = np.empty_like(state)
        field_line_rk4_finish_into(state, k1, k2, k3, k4, step, candidate)
        state = candidate
    return state


def test_rotational_rhs_has_fourth_order_global_convergence() -> None:
    errors = []
    for step_count in (16, 32, 64):
        result = integrate_rotation(step_count)[0]
        errors.append(float(np.hypot(result[0] - 1.0, result[1])))
        assert result[2] == 2.0
        assert abs(result[3] - 2.0 * math.pi) < 2.0e-14
    orders = [math.log(errors[index] / errors[index + 1], 2.0) for index in range(2)]
    assert min(orders) >= 3.8
