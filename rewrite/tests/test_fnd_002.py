from __future__ import annotations

import numpy as np
import pytest

from simesh.utils.lib.amr.forest import AMRForest
from simesh.utils.lib.amr.mesh import AMRMesh
from simesh_rewrite.access import (
    AccessPattern,
    required_input_region,
    supports_output_region,
    valid_output_region,
    validate_access_requirement,
)
from simesh_rewrite.access_reference import (
    required_input_region as required_input_region_reference,
)
from simesh_rewrite.access_reference import (
    supports_output_region as supports_output_region_reference,
)
from simesh_rewrite.access_reference import (
    valid_output_region as valid_output_region_reference,
)


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def test_access_patterns_have_disjoint_reach_contracts() -> None:
    zero = i3(0, 0, 0)
    x_reach = i3(1, 0, 0)
    validate_access_requirement(AccessPattern.POINTWISE, zero, zero)
    validate_access_requirement(AccessPattern.STREAMING_REDUCTION, zero, zero)
    validate_access_requirement(AccessPattern.LOCAL_STENCIL, x_reach, zero)

    with pytest.raises(ValueError, match="zero reach"):
        validate_access_requirement(AccessPattern.POINTWISE, x_reach, zero)
    with pytest.raises(ValueError, match="nonzero reach"):
        validate_access_requirement(AccessPattern.LOCAL_STENCIL, zero, zero)
    with pytest.raises(TypeError, match="AccessPattern"):
        validate_access_requirement(0, zero, zero)


def test_asymmetric_region_algebra_matches_reference_exactly() -> None:
    output_lower = i3(3, 5, 7)
    output_upper = i3(8, 11, 13)
    lower_reach = i3(2, 0, 3)
    upper_reach = i3(0, 4, 1)

    required = required_input_region(
        output_lower,
        output_upper,
        lower_reach,
        upper_reach,
    )
    expected = required_input_region_reference(
        output_lower,
        output_upper,
        lower_reach,
        upper_reach,
    )
    assert np.array_equal(required[0], i3(1, 5, 4))
    assert np.array_equal(required[1], i3(8, 15, 14))
    assert np.array_equal(required[0], expected[0])
    assert np.array_equal(required[1], expected[1])

    recovered = valid_output_region(
        required[0],
        required[1],
        lower_reach,
        upper_reach,
    )
    assert np.array_equal(recovered[0], output_lower)
    assert np.array_equal(recovered[1], output_upper)


def test_support_requires_all_six_faces() -> None:
    output_lower = i3(3, 5, 7)
    output_upper = i3(8, 11, 13)
    lower_reach = i3(2, 1, 3)
    upper_reach = i3(1, 4, 1)
    valid_lower, valid_upper = required_input_region(
        output_lower,
        output_upper,
        lower_reach,
        upper_reach,
    )
    assert supports_output_region(
        valid_lower,
        valid_upper,
        output_lower,
        output_upper,
        lower_reach,
        upper_reach,
    )

    for axis in range(3):
        short_lower = valid_lower.copy()
        short_lower[axis] += 1
        assert not supports_output_region(
            short_lower,
            valid_upper,
            output_lower,
            output_upper,
            lower_reach,
            upper_reach,
        )

        short_upper = valid_upper.copy()
        short_upper[axis] -= 1
        assert not supports_output_region(
            valid_lower,
            short_upper,
            output_lower,
            output_upper,
            lower_reach,
            upper_reach,
        )


def test_zero_reach_is_identity_and_inputs_are_not_mutated() -> None:
    lower = i3(2, 3, 4)
    upper = i3(9, 11, 15)
    zero = i3(0, 0, 0)
    snapshots = [array.copy() for array in (lower, upper, zero)]
    required = required_input_region(lower, upper, zero, zero)
    valid = valid_output_region(lower, upper, zero, zero)
    assert np.array_equal(required[0], lower)
    assert np.array_equal(required[1], upper)
    assert np.array_equal(valid[0], lower)
    assert np.array_equal(valid[1], upper)
    assert all(
        np.array_equal(array, snapshot)
        for array, snapshot in zip((lower, upper, zero), snapshots, strict=True)
    )


def test_empty_regions_require_no_reads_and_contract_to_canonical_empty() -> None:
    output_lower = i3(4, 2, 3)
    output_upper = i3(4, 9, 10)
    reach = i3(2, 1, 3)
    required = required_input_region(output_lower, output_upper, reach, reach)
    assert np.array_equal(required[0], output_lower)
    assert np.array_equal(required[1], output_lower)
    assert supports_output_region(
        i3(0, 0, 0),
        i3(1, 1, 1),
        output_lower,
        output_upper,
        reach,
        reach,
    )

    contracted = valid_output_region(i3(5, 6, 7), i3(8, 9, 10), reach, reach)
    assert np.array_equal(contracted[0], i3(5, 6, 7))
    assert np.array_equal(contracted[1], i3(5, 6, 7))


def test_reference_and_compiled_region_invariant_over_varied_boxes() -> None:
    rng = np.random.default_rng(20260828)
    for _ in range(100):
        lower_reach = rng.integers(0, 4, size=3, dtype=np.int64)
        upper_reach = rng.integers(0, 4, size=3, dtype=np.int64)
        output_lower = lower_reach + rng.integers(0, 5, size=3, dtype=np.int64)
        output_upper = output_lower + rng.integers(1, 8, size=3, dtype=np.int64)
        required = required_input_region(
            output_lower,
            output_upper,
            lower_reach,
            upper_reach,
        )
        expected_required = required_input_region_reference(
            output_lower,
            output_upper,
            lower_reach,
            upper_reach,
        )
        assert np.array_equal(required[0], expected_required[0])
        assert np.array_equal(required[1], expected_required[1])

        recovered = valid_output_region(
            required[0],
            required[1],
            lower_reach,
            upper_reach,
        )
        expected_recovered = valid_output_region_reference(
            required[0],
            required[1],
            lower_reach,
            upper_reach,
        )
        assert np.array_equal(recovered[0], output_lower)
        assert np.array_equal(recovered[1], output_upper)
        assert np.array_equal(recovered[0], expected_recovered[0])
        assert np.array_equal(recovered[1], expected_recovered[1])
        assert supports_output_region(
            required[0],
            required[1],
            output_lower,
            output_upper,
            lower_reach,
            upper_reach,
        ) == supports_output_region_reference(
            required[0],
            required[1],
            output_lower,
            output_upper,
            lower_reach,
            upper_reach,
        )


def test_invalid_regions_reaches_and_int64_overflow_are_rejected() -> None:
    zero = i3(0, 0, 0)
    with pytest.raises(TypeError, match="int64"):
        required_input_region(
            np.zeros(3, dtype=np.int32),
            i3(1, 1, 1),
            zero,
            zero,
        )
    with pytest.raises(ValueError, match="exceeds"):
        valid_output_region(i3(2, 0, 0), i3(1, 1, 1), zero, zero)
    with pytest.raises(ValueError, match="non-negative"):
        valid_output_region(zero, i3(1, 1, 1), i3(-1, 0, 0), zero)
    with pytest.raises(ValueError, match="below storage"):
        required_input_region(zero, i3(1, 1, 1), i3(1, 0, 0), zero)
    with pytest.raises(OverflowError, match="int64"):
        required_input_region(
            i3(1, 1, 1),
            i3(np.iinfo(np.int64).max, 2, 2),
            zero,
            i3(1, 0, 0),
        )

    mixed_lower_and_overflow = (
        i3(0, 0, 0),
        i3(10, 10, 10),
        i3(0, 1, 1),
        i3(2, np.iinfo(np.int64).max, 2),
        i3(1, 0, 0),
        i3(0, 1, 0),
    )
    with pytest.raises(OverflowError, match="int64"):
        supports_output_region(*mixed_lower_and_overflow)
    with pytest.raises(OverflowError, match="int64"):
        supports_output_region_reference(*mixed_lower_and_overflow)


def test_current_central_difference_matches_one_cell_region_contraction() -> None:
    block_shape = np.array([4, 4, 4], dtype=np.uint32)
    forest = AMRForest(3, 1, 1, 1, np.ones(1, dtype=np.int32))
    mesh = AMRMesh(
        3,
        block_shape,
        block_shape,
        np.zeros(3, dtype=np.float64),
        np.ones(3, dtype=np.float64),
        np.uint32(2),
        np.uint32(1),
        forest,
    )
    padded = mesh.padded_view()
    i = np.arange(8, dtype=np.float64)[:, None, None]
    j = np.arange(8, dtype=np.float64)[None, :, None]
    k = np.arange(8, dtype=np.float64)[None, None, :]
    padded[0, :, :, :, 0] = i + 10.0 * j + 100.0 * k

    output = np.empty((1, 8, 8, 8, 1), dtype=np.float64)
    mesh.first_derivative_fields(
        output,
        np.array([0, 0, 0], dtype=np.uint32),
        np.array([0, 0, 0], dtype=np.uint32),
        np.array([0, 1, 2], dtype=np.uint32),
        np.ones(3, dtype=np.float64),
    )

    reach = i3(1, 1, 1)
    predicted = valid_output_region(i3(0, 0, 0), i3(8, 8, 8), reach, reach)
    assert np.array_equal(predicted[0], i3(1, 1, 1))
    assert np.array_equal(predicted[1], i3(7, 7, 7))
    assert np.array_equal(output[0, 1:7, 1:7, 1:7, 0], np.full((6, 6, 6), 444.0))
    assert np.all(output[:, 0, :, :, :] == 0.0)
    assert np.all(output[:, -1, :, :, :] == 0.0)
    assert np.all(output[:, :, 0, :, :] == 0.0)
    assert np.all(output[:, :, -1, :, :] == 0.0)
    assert np.all(output[:, :, :, 0, :] == 0.0)
    assert np.all(output[:, :, :, -1, :] == 0.0)
