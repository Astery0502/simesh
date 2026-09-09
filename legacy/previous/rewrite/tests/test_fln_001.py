from __future__ import annotations

import itertools

import numpy as np
import pytest

from simesh_rewrite._field_line_rhs import field_line_rhs_unchecked
from simesh_rewrite.field_line_rhs import (
    RHS_NONFINITE_FIELD,
    RHS_OK,
    RHS_UNREPRESENTABLE_NORM,
    RHS_ZERO_FIELD,
    field_line_rhs_into,
)
from simesh_rewrite.field_line_rhs_reference import field_line_rhs_reference


def assert_bits_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def reference_result(
    vectors: np.ndarray,
    signs: np.ndarray,
    initial_rhs: np.ndarray,
    initial_statuses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rhs = initial_rhs.copy()
    statuses = initial_statuses.copy()
    field_line_rhs_reference(vectors, signs, rhs, statuses)
    return rhs, statuses


def test_empty_and_axis_aligned_oriented_rows_match_exact_tree() -> None:
    empty_vectors = np.empty((0, 3), dtype=np.float64)
    empty_signs = np.empty(0, dtype=np.int8)
    empty_rhs = np.empty((0, 4), dtype=np.float64)
    empty_status = np.empty(0, dtype=np.uint8)
    assert field_line_rhs_into(
        empty_vectors, empty_signs, empty_rhs, empty_status
    ) is None

    vectors = np.asarray(
        (
            (2.0, 0.0, 0.0),
            (0.0, -3.0, 0.0),
            (0.0, 0.0, 4.0),
            (-5.0, 0.0, 0.0),
            (0.0, 6.0, 0.0),
            (0.0, 0.0, -7.0),
        ),
        dtype=np.float64,
    )
    signs = np.asarray((1, 1, 1, -1, -1, -1), dtype=np.int8)
    rhs = np.full((6, 4), -91.0, dtype=np.float64)
    statuses = np.full(6, 255, dtype=np.uint8)
    expected_rhs, expected_statuses = reference_result(
        vectors, signs, rhs, statuses
    )
    field_line_rhs_into(vectors, signs, rhs, statuses)
    assert_bits_equal(rhs, expected_rhs)
    assert np.array_equal(statuses, expected_statuses)
    assert np.all(statuses == RHS_OK)
    assert rhs[:, 3].tolist() == [2.0, 3.0, 4.0, -5.0, -6.0, -7.0]
    assert np.array_equal(np.abs(rhs[:, :3]), np.eye(3)[[0, 1, 2, 0, 1, 2]])


def test_random_finite_rows_and_unchecked_symbol_match_reference_bits() -> None:
    rng = np.random.default_rng(20260904)
    mantissas = rng.uniform(-1.0, 1.0, size=(800, 3))
    exponents = rng.integers(-900, 900, size=(800, 1))
    vectors = np.ascontiguousarray(np.ldexp(mantissas, exponents))
    zero_rows = np.all(vectors == 0.0, axis=1)
    vectors[zero_rows, 0] = 1.0
    signs = rng.choice(np.asarray((-1, 1), dtype=np.int8), size=800)
    signs = np.ascontiguousarray(signs, dtype=np.int8)
    initial_rhs = rng.normal(size=(800, 4)).astype(np.float64)
    initial_statuses = np.full(800, 211, dtype=np.uint8)
    expected_rhs, expected_statuses = reference_result(
        vectors, signs, initial_rhs, initial_statuses
    )

    checked_rhs = initial_rhs.copy()
    checked_statuses = initial_statuses.copy()
    field_line_rhs_into(vectors, signs, checked_rhs, checked_statuses)
    assert_bits_equal(checked_rhs, expected_rhs)
    assert np.array_equal(checked_statuses, expected_statuses)

    unchecked_rhs = initial_rhs.copy()
    unchecked_statuses = initial_statuses.copy()
    field_line_rhs_unchecked(
        vectors, signs, unchecked_rhs, unchecked_statuses
    )
    assert_bits_equal(unchecked_rhs, expected_rhs)
    assert np.array_equal(unchecked_statuses, expected_statuses)


def test_power_of_two_scaling_preserves_tangent_and_scales_integrand() -> None:
    base = np.asarray((1.0, -2.0, 4.0), dtype=np.float64)
    vectors = np.ascontiguousarray(
        np.asarray(
            [np.ldexp(base, exponent) for exponent in (-500, 0, 500)],
            dtype=np.float64,
        )
    )
    signs = np.ones(3, dtype=np.int8)
    rhs = np.empty((3, 4), dtype=np.float64)
    statuses = np.empty(3, dtype=np.uint8)
    field_line_rhs_into(vectors, signs, rhs, statuses)
    assert np.all(statuses == RHS_OK)
    assert_bits_equal(rhs[0, :3], rhs[1, :3])
    assert_bits_equal(rhs[1, :3], rhs[2, :3])
    assert rhs[0, 3] == np.ldexp(rhs[1, 3], -500)
    assert rhs[2, 3] == np.ldexp(rhs[1, 3], 500)


def test_normal_subnormal_and_overflowing_norm_cases() -> None:
    smallest = np.nextafter(np.float64(0.0), np.float64(1.0))
    minimum_normal = np.float64(np.finfo(np.float64).tiny)
    maximum = np.float64(np.finfo(np.float64).max)
    vectors = np.asarray(
        (
            (smallest, 0.0, 0.0),
            (smallest, smallest, smallest),
            (minimum_normal, minimum_normal, 0.0),
            (maximum, 0.0, 0.0),
            (maximum / 2.0, maximum / 2.0, maximum / 2.0),
            (maximum, maximum, 0.0),
        ),
        dtype=np.float64,
    )
    signs = np.asarray((1, -1, 1, -1, 1, -1), dtype=np.int8)
    rhs = np.arange(24, dtype=np.float64).reshape(6, 4)
    before = rhs.copy()
    statuses = np.full(6, 255, dtype=np.uint8)
    expected_rhs, expected_statuses = reference_result(
        vectors, signs, rhs, statuses
    )
    field_line_rhs_into(vectors, signs, rhs, statuses)
    assert_bits_equal(rhs, expected_rhs)
    assert np.array_equal(statuses, expected_statuses)
    assert statuses.tolist() == [RHS_OK] * 5 + [RHS_UNREPRESENTABLE_NORM]
    assert np.isfinite(rhs[:5]).all()
    assert_bits_equal(rhs[5], before[5])
    assert rhs[0, 3] == smallest
    assert rhs[3, 3] == -maximum


def test_every_signed_zero_combination_is_zero_field_and_preserves_rhs() -> None:
    vectors = np.asarray(
        list(itertools.product((np.float64(0.0), np.float64(-0.0)), repeat=3)),
        dtype=np.float64,
    )
    signs = np.resize(np.asarray((-1, 1), dtype=np.int8), vectors.shape[0])
    rhs = np.arange(vectors.shape[0] * 4, dtype=np.float64).reshape(-1, 4)
    before = rhs.copy()
    statuses = np.full(vectors.shape[0], 255, dtype=np.uint8)
    field_line_rhs_into(vectors, signs, rhs, statuses)
    assert np.all(statuses == RHS_ZERO_FIELD)
    assert_bits_equal(rhs, before)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_component_positions_are_classified_without_rhs_write(
    bad_value: float,
) -> None:
    vectors = np.ones((3, 3), dtype=np.float64)
    for component in range(3):
        vectors[component, component] = bad_value
    signs = np.asarray((1, -1, 1), dtype=np.int8)
    rhs = np.asarray(
        (
            (np.nan, -0.0, np.inf, -np.inf),
            (1.0, 2.0, 3.0, 4.0),
            (-5.0, -6.0, -7.0, -8.0),
        ),
        dtype=np.float64,
    )
    before = rhs.copy()
    statuses = np.full(3, 255, dtype=np.uint8)
    field_line_rhs_into(vectors, signs, rhs, statuses)
    assert np.all(statuses == RHS_NONFINITE_FIELD)
    assert_bits_equal(rhs, before)


def test_mixed_status_batch_overwrites_only_successful_rhs_rows() -> None:
    maximum = np.float64(np.finfo(np.float64).max)
    smallest = np.nextafter(np.float64(0.0), np.float64(1.0))
    vectors = np.asarray(
        (
            (3.0, 4.0, 0.0),
            (-0.0, 0.0, -0.0),
            (np.nan, 1.0, 2.0),
            (maximum, maximum, maximum),
            (smallest, 0.0, 0.0),
        ),
        dtype=np.float64,
    )
    signs = np.asarray((1, -1, 1, -1, -1), dtype=np.int8)
    rhs = np.asarray(
        [
            [100.0 * row + column for column in range(4)]
            for row in range(vectors.shape[0])
        ],
        dtype=np.float64,
    )
    before = rhs.copy()
    statuses = np.full(5, 199, dtype=np.uint8)
    expected_rhs, expected_statuses = reference_result(
        vectors, signs, rhs, statuses
    )
    field_line_rhs_into(vectors, signs, rhs, statuses)
    assert statuses.tolist() == [
        RHS_OK,
        RHS_ZERO_FIELD,
        RHS_NONFINITE_FIELD,
        RHS_UNREPRESENTABLE_NORM,
        RHS_OK,
    ]
    assert np.array_equal(statuses, expected_statuses)
    assert_bits_equal(rhs, expected_rhs)
    assert_bits_equal(rhs[1:4], before[1:4])


def test_unit_tangent_and_oriented_b_dot_dx_invariants() -> None:
    vectors = np.asarray(
        ((3.0, 4.0, 12.0), (-2.5, 7.0, -1.25), (1.0e-200, -2.0e-200, 3.0e-200)),
        dtype=np.float64,
    )
    signs = np.asarray((1, -1, -1), dtype=np.int8)
    rhs = np.empty((3, 4), dtype=np.float64)
    statuses = np.empty(3, dtype=np.uint8)
    field_line_rhs_into(vectors, signs, rhs, statuses)
    assert np.all(statuses == RHS_OK)
    tangent_norm = np.sqrt(np.sum(rhs[:, :3] * rhs[:, :3], axis=1))
    np.testing.assert_allclose(tangent_norm, 1.0, rtol=8.0 * np.finfo(float).eps)
    dot = np.sum(vectors * rhs[:, :3], axis=1)
    np.testing.assert_allclose(
        dot,
        rhs[:, 3],
        rtol=16.0 * np.finfo(float).eps,
        atol=np.nextafter(0.0, 1.0),
    )
    assert np.array_equal(np.signbit(rhs[:, 3]), signs < 0)


@pytest.mark.parametrize(
    "failure",
    (
        "vectors_type",
        "vectors_dtype",
        "vectors_shape",
        "vectors_layout",
        "signs_dtype",
        "signs_shape",
        "signs_layout",
        "sign_value",
        "rhs_dtype",
        "rhs_shape",
        "rhs_layout",
        "rhs_readonly",
        "status_dtype",
        "status_shape",
        "status_layout",
        "status_readonly",
    ),
)
def test_validation_failures_preserve_both_outputs(failure: str) -> None:
    vectors = np.asarray(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)))
    signs = np.asarray((1, -1), dtype=np.int8)
    rhs = np.full((2, 4), -701.0, dtype=np.float64)
    statuses = np.full(2, 177, dtype=np.uint8)
    if failure == "vectors_type":
        vectors = vectors.tolist()  # type: ignore[assignment]
    elif failure == "vectors_dtype":
        vectors = vectors.astype(np.float32)
    elif failure == "vectors_shape":
        vectors = np.ones((2, 2), dtype=np.float64)
    elif failure == "vectors_layout":
        vectors = np.empty((2, 6), dtype=np.float64)[:, ::2]
    elif failure == "signs_dtype":
        signs = signs.astype(np.int64)
    elif failure == "signs_shape":
        signs = signs[:1]
    elif failure == "signs_layout":
        storage = np.asarray((1, 9, -1, 9), dtype=np.int8)
        signs = storage[::2]
    elif failure == "sign_value":
        signs = np.asarray((1, 0), dtype=np.int8)
    elif failure == "rhs_dtype":
        rhs = rhs.astype(np.float32)
    elif failure == "rhs_shape":
        rhs = np.full((2, 3), -701.0)
    elif failure == "rhs_layout":
        rhs = np.empty((2, 8), dtype=np.float64)[:, ::2]
    elif failure == "rhs_readonly":
        rhs.setflags(write=False)
    elif failure == "status_dtype":
        statuses = statuses.astype(np.int8)
    elif failure == "status_shape":
        statuses = statuses[:1]
    elif failure == "status_layout":
        storage = np.full(4, 177, dtype=np.uint8)
        statuses = storage[::2]
    else:
        statuses.setflags(write=False)
    rhs_before = rhs.copy()
    statuses_before = statuses.copy()
    error = TypeError if failure.endswith(("type", "dtype")) else ValueError
    with pytest.raises(error):
        field_line_rhs_into(vectors, signs, rhs, statuses)
    assert_bits_equal(np.asarray(rhs), rhs_before)
    assert np.array_equal(statuses, statuses_before)


def test_input_and_output_aliases_are_rejected_atomically() -> None:
    shared_float = np.arange(8, dtype=np.float64)
    vectors = shared_float[:6].reshape(2, 3)
    rhs = shared_float.reshape(2, 4)
    signs = np.asarray((1, -1), dtype=np.int8)
    statuses = np.full(2, 131, dtype=np.uint8)
    float_before = shared_float.copy()
    status_before = statuses.copy()
    with pytest.raises(ValueError, match="overlap inputs"):
        field_line_rhs_into(vectors, signs, rhs, statuses)
    assert_bits_equal(shared_float, float_before)
    assert np.array_equal(statuses, status_before)

    vectors = np.ones((2, 3), dtype=np.float64)
    rhs = np.full((2, 4), -801.0, dtype=np.float64)
    statuses = rhs.view(np.uint8).reshape(-1)[:2]
    rhs_before = rhs.copy()
    with pytest.raises(ValueError, match="overlap each other"):
        field_line_rhs_into(vectors, signs, rhs, statuses)
    assert_bits_equal(rhs, rhs_before)

    sign_storage = np.asarray((1, -1), dtype=np.int8)
    statuses = sign_storage.view(np.uint8)
    rhs = np.full((2, 4), -901.0, dtype=np.float64)
    rhs_before = rhs.copy()
    signs_before = sign_storage.copy()
    with pytest.raises(ValueError, match="overlap inputs"):
        field_line_rhs_into(vectors, sign_storage, rhs, statuses)
    assert_bits_equal(rhs, rhs_before)
    assert np.array_equal(sign_storage, signs_before)


def test_readonly_inputs_are_preserved() -> None:
    vectors = np.asarray(((3.0, 4.0, 0.0), (1.0, -2.0, 2.0)))
    signs = np.asarray((1, -1), dtype=np.int8)
    vectors_before = vectors.copy()
    signs_before = signs.copy()
    vectors.setflags(write=False)
    signs.setflags(write=False)
    rhs = np.empty((2, 4), dtype=np.float64)
    statuses = np.empty(2, dtype=np.uint8)
    field_line_rhs_into(vectors, signs, rhs, statuses)
    assert np.array_equal(vectors, vectors_before)
    assert np.array_equal(signs, signs_before)
    assert np.all(statuses == RHS_OK)
