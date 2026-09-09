from __future__ import annotations

import numpy as np
import pytest

from simesh.amrvac import load_from_uniform
from simesh_rewrite.access import (
    AccessPattern,
    required_input_region,
    validate_access_requirement,
)
from simesh_rewrite.chunking import plan_level1_chunk
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.operators import scaled_difference_into
from simesh_rewrite.storage import gather_blocks_into, scatter_blocks_from
from simesh_rewrite.topology import level1_face_neighbors


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def assert_bits_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def patterned(shape: tuple[int, ...]) -> np.ndarray:
    slot, field, x, y, z = np.indices(shape, dtype=np.float64)
    return 10000.0 * slot + 1000.0 * field + 100.0 * x + 10.0 * y + z


def reference(
    source: np.ndarray,
    source_lower: np.ndarray,
    source_upper: np.ndarray,
    left_field: int,
    right_field: int,
    scale: float,
    destination: np.ndarray,
    destination_field: int,
    destination_lower: np.ndarray,
) -> None:
    source_region = tuple(
        slice(int(lower), int(upper))
        for lower, upper in zip(source_lower, source_upper, strict=True)
    )
    extent = source_upper - source_lower
    destination_region = tuple(
        slice(int(lower), int(lower + size))
        for lower, size in zip(destination_lower, extent, strict=True)
    )
    product = np.multiply(
        source[(slice(None), right_field, *source_region)],
        np.float64(scale),
    )
    np.subtract(
        source[(slice(None), left_field, *source_region)],
        product,
        out=destination[(slice(None), destination_field, *destination_region)],
    )


def run_scalar(left: float, right: float, scale: float) -> np.float64:
    source = np.asarray([left, right], dtype=np.float64).reshape(1, 2, 1, 1, 1)
    destination = np.empty((1, 1, 1, 1, 1), dtype=np.float64)
    with np.errstate(all="ignore"):
        scaled_difference_into(
            source,
            i3(0, 0, 0),
            i3(1, 1, 1),
            0,
            1,
            scale,
            destination,
            0,
            i3(0, 0, 0),
        )
    return destination[0, 0, 0, 0, 0]


def test_translated_asymmetric_regions_and_fields_match_reference() -> None:
    source = patterned((2, 3, 5, 6, 7))
    source_before = source.copy()
    source_lower = i3(1, 2, 1)
    source_upper = i3(5, 6, 7)
    destination_lower = i3(2, 1, 3)
    destination = np.full((2, 4, 8, 7, 10), -19.0)
    expected = destination.copy()
    reference(
        source,
        source_lower,
        source_upper,
        2,
        0,
        np.float64(0.5),
        expected,
        3,
        destination_lower,
    )
    scaled_difference_into(
        source,
        source_lower,
        source_upper,
        2,
        0,
        np.float64(0.5),
        destination,
        3,
        destination_lower,
    )
    assert_bits_equal(destination, expected)
    assert_bits_equal(source, source_before)
    assert np.all(destination[:, :3] == -19.0)
    assert np.all(destination[:, 3, :2] == -19.0)


def test_compiled_and_current_recipe_match_numpy_bitwise() -> None:
    x, y, z = np.indices((4, 4, 4), dtype=np.float64)
    uniform = np.empty((4, 4, 4, 3), dtype=np.float64)
    uniform[..., 0] = 1.0 + x + 2.0 * y + 3.0 * z
    uniform[..., 1] = 10.0 + y
    uniform[..., 2] = 100.0 + 5.0 * x - y + 0.25 * z
    dataset = load_from_uniform(
        uniform,
        ["rho", "m1", "e"],
        np.zeros(3),
        np.ones(3),
        np.array([2, 2, 2], dtype=np.int32),
    )
    source = np.ascontiguousarray(dataset.data)
    expected = np.subtract(
        source[:, 2],
        np.multiply(np.float64(0.5), source[:, 0]),
    )
    destination = np.empty((source.shape[0], 1, *source.shape[2:]))
    scaled_difference_into(
        source,
        i3(0, 0, 0),
        i3(*source.shape[2:]),
        2,
        0,
        0.5,
        destination,
        0,
        i3(0, 0, 0),
    )
    dataset.register_derived(
        "p",
        lambda ctx: ctx.field("e") - 0.5 * ctx.field("rho"),
        dependencies=["rho", "e"],
    )
    dataset.materialize_fields(["p"])
    assert_bits_equal(destination[:, 0], expected)
    assert_bits_equal(dataset.data[:, 3], expected)


def test_separate_multiply_then_subtract_is_not_fused() -> None:
    eps = np.float64(2.0**-52)
    result = run_scalar(
        np.float64(1.0),
        np.float64(1.0) - eps,
        np.float64(1.0) + eps,
    )
    assert result.view(np.uint64) == np.uint64(0x0000000000000000)


@pytest.mark.parametrize(
    ("scale_bits", "left_bits", "right_bits", "expected_bits"),
    [
        (0x0000000000000000, 0x8000000000000000, 0x0000000000000000, 0x8000000000000000),
        (0x8000000000000000, 0x8000000000000000, 0x0000000000000000, 0x0000000000000000),
        (0x8000000000000000, 0x8000000000000000, 0x8000000000000000, 0x8000000000000000),
    ],
)
def test_signed_zero_behavior_is_exact(
    scale_bits: int,
    left_bits: int,
    right_bits: int,
    expected_bits: int,
) -> None:
    scale = np.asarray([scale_bits], dtype=np.uint64).view(np.float64)[0]
    left = np.asarray([left_bits], dtype=np.uint64).view(np.float64)[0]
    right = np.asarray([right_bits], dtype=np.uint64).view(np.float64)[0]
    result = run_scalar(left, right, scale)
    assert result.view(np.uint64) == np.uint64(expected_bits)


def test_nonfinite_overflow_and_equal_field_classification() -> None:
    with np.errstate(all="ignore"):
        assert np.isnan(run_scalar(1.0, np.inf, 0.0))
        assert np.isnan(run_scalar(np.inf, np.inf, 1.0))
        assert np.isposinf(run_scalar(np.inf, -np.inf, 1.0))
        assert np.isneginf(
            run_scalar(np.finfo(np.float64).max, np.finfo(np.float64).max, 2.0)
        )
        assert np.isnan(run_scalar(np.nan, 1.0, 0.5))
        assert np.isnan(run_scalar(1.0, np.nan, 0.5))

    values = np.asarray([1.0, -2.0, -0.0, np.inf], dtype=np.float64)
    source = values.reshape(1, 1, 1, 1, 4)
    destination = np.empty_like(source)
    with np.errstate(all="ignore"):
        scaled_difference_into(
            source,
            i3(0, 0, 0),
            i3(1, 1, 4),
            0,
            0,
            1.0,
            destination,
            0,
            i3(0, 0, 0),
        )
    assert destination[0, 0, 0, 0, 0].view(np.uint64) == 0
    assert destination[0, 0, 0, 0, 1].view(np.uint64) == 0
    assert destination[0, 0, 0, 0, 2].view(np.uint64) == 0
    assert np.isnan(destination[0, 0, 0, 0, 3])


def test_empty_requests_still_validate_fields_and_scale() -> None:
    empty_source = np.empty((0, 2, 2, 2, 2), dtype=np.float64)
    empty_destination = np.empty((0, 1, 2, 2, 2), dtype=np.float64)
    scaled_difference_into(
        empty_source,
        i3(0, 0, 0),
        i3(2, 2, 2),
        0,
        1,
        0.5,
        empty_destination,
        0,
        i3(0, 0, 0),
    )
    with pytest.raises(ValueError, match="field axis"):
        scaled_difference_into(
            empty_source,
            i3(0, 0, 0),
            i3(2, 2, 2),
            2,
            1,
            0.5,
            empty_destination,
            0,
            i3(0, 0, 0),
        )

    source = np.ones((1, 2, 2, 2, 2), dtype=np.float64)
    destination = np.full((1, 1, 2, 2, 2), -3.0)
    before = destination.copy()
    scaled_difference_into(
        source,
        i3(1, 0, 0),
        i3(1, 2, 2),
        0,
        1,
        0.5,
        destination,
        0,
        i3(0, 0, 0),
    )
    assert_bits_equal(destination, before)
    with pytest.raises(TypeError, match="binary64"):
        scaled_difference_into(
            source,
            i3(1, 0, 0),
            i3(1, 2, 2),
            0,
            1,
            1,
            destination,
            0,
            i3(0, 0, 0),
        )


def test_invalid_scale_field_region_output_overlap_and_overflow_are_atomic() -> None:
    class FloatSubclass(float):
        pass

    source = patterned((1, 2, 2, 2, 2))
    destination = np.full((1, 1, 3, 3, 3), -7.0)
    before = destination.copy()

    for bad_scale in (
        1,
        True,
        np.float32(0.5),
        np.asarray(0.5),
        FloatSubclass(0.5),
        np.inf,
        np.nan,
    ):
        error = ValueError if type(bad_scale) in (float, np.float64) else TypeError
        with pytest.raises(error):
            scaled_difference_into(
                source,
                i3(0, 0, 0),
                i3(2, 2, 2),
                0,
                1,
                bad_scale,
                destination,
                0,
                i3(0, 0, 0),
            )
        assert_bits_equal(destination, before)

    with pytest.raises(ValueError, match="field axis"):
        scaled_difference_into(
            source,
            i3(0, 0, 0),
            i3(2, 2, 2),
            2,
            1,
            0.5,
            destination,
            0,
            i3(0, 0, 0),
        )
    assert_bits_equal(destination, before)

    with pytest.raises(ValueError, match="source valid region"):
        scaled_difference_into(
            source,
            i3(1, 0, 0),
            i3(0, 2, 2),
            0,
            1,
            0.5,
            destination,
            0,
            i3(0, 0, 0),
        )
    assert_bits_equal(destination, before)

    readonly = destination.copy()
    readonly.setflags(write=False)
    with pytest.raises(ValueError, match="writable"):
        scaled_difference_into(
            source,
            i3(0, 0, 0),
            i3(2, 2, 2),
            0,
            1,
            0.5,
            readonly,
            0,
            i3(0, 0, 0),
        )

    overlap = patterned((1, 2, 2, 2, 2))
    overlap_before = overlap.copy()
    with pytest.raises(ValueError, match="overlap"):
        scaled_difference_into(
            overlap,
            i3(0, 0, 0),
            i3(2, 2, 2),
            0,
            1,
            0.5,
            overlap,
            0,
            i3(0, 0, 0),
        )
    assert_bits_equal(overlap, overlap_before)

    with pytest.raises(OverflowError, match="int64"):
        scaled_difference_into(
            source,
            i3(0, 0, 0),
            i3(2, 2, 2),
            0,
            1,
            0.5,
            destination,
            0,
            i3(np.iinfo(np.int64).max, 0, 0),
        )
    assert_bits_equal(destination, before)


def test_fnd_pointwise_requirement_has_exact_zero_reach() -> None:
    zero = i3(0, 0, 0)
    validate_access_requirement(AccessPattern.POINTWISE, zero, zero)
    lower, upper = required_input_region(
        i3(2, 3, 4),
        i3(7, 9, 11),
        zero,
        zero,
    )
    assert np.array_equal(lower, i3(2, 3, 4))
    assert np.array_equal(upper, i3(7, 9, 11))
    with pytest.raises(ValueError, match="zero reach"):
        validate_access_requirement(
            AccessPattern.POINTWISE,
            i3(1, 0, 0),
            zero,
        )


def test_bounded_gather_compute_scatter_with_reordered_fields() -> None:
    root_shape = i3(5, 3, 2)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    faces = level1_face_neighbors(root_shape, coord_to_rank, rank_to_coord)
    block_count = rank_to_coord.shape[0]
    block_shape = i3(2, 3, 4)
    backing = patterned((block_count, 3, 2, 3, 4))
    sink = np.full((block_count, 1, 2, 3, 4), np.nan)
    field_ids = np.array([2, 0], dtype=np.int64)
    sink_fields = np.array([0], dtype=np.int64)
    capacity = 7
    chunk_ids = np.empty(capacity, dtype=np.int64)
    source_workspace = np.full((capacity, 2, 4, 5, 6), -101.0)
    destination_workspace = np.full((capacity, 1, 5, 4, 7), -103.0)
    source_lower = i3(1, 1, 1)
    source_upper = source_lower + block_shape
    destination_lower = i3(2, 0, 2)
    destination_upper = destination_lower + block_shape
    first = 0
    covered = []
    while first < block_count:
        primary_count, selected_count = plan_level1_chunk(
            first,
            faces,
            False,
            chunk_ids,
        )
        assert primary_count == selected_count
        gather_blocks_into(
            backing,
            i3(0, 0, 0),
            block_shape,
            chunk_ids[:selected_count],
            field_ids,
            source_workspace[:selected_count],
            source_lower,
        )
        scaled_difference_into(
            source_workspace[:primary_count],
            source_lower,
            source_upper,
            0,
            1,
            0.5,
            destination_workspace[:primary_count],
            0,
            destination_lower,
        )
        scatter_blocks_from(
            destination_workspace[:primary_count],
            destination_lower,
            destination_upper,
            chunk_ids[:primary_count],
            sink_fields,
            sink,
            i3(0, 0, 0),
        )
        covered.extend(chunk_ids[:primary_count].tolist())
        first += primary_count

    expected = np.subtract(backing[:, 2], np.multiply(0.5, backing[:, 0]))
    assert covered == list(range(block_count))
    assert_bits_equal(sink[:, 0], expected)
