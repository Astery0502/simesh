from __future__ import annotations

import numpy as np
import pytest

from simesh_rewrite.boundary_rules import (
    physical_halo_source_index,
    transform_physical_halo_value,
)
from simesh_rewrite.boundary_rules_reference import (
    physical_halo_source_index_reference,
    transform_physical_halo_value_reference,
)
from simesh_rewrite.halos import fill_physical_halos, fill_same_level_halos
from simesh_rewrite.halos_reference import (
    fill_physical_halos_reference,
    fill_same_level_halos_reference,
)
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.topology import level1_face_neighbors


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def bits(value: float) -> int:
    return int(np.asarray([value], dtype=np.float64).view(np.uint64)[0])


def test_all_faces_modes_and_reflected_layers_match_reference() -> None:
    lower = i3(3, 4, 5)
    upper = i3(7, 10, 13)
    for face in range(6):
        axis = face // 2
        extent = int(upper[axis] - lower[axis])
        for mode in range(4):
            for layer in range(1, extent + 1):
                target = (
                    int(lower[axis]) - layer
                    if face % 2 == 0
                    else int(upper[axis]) + layer - 1
                )
                actual = physical_halo_source_index(
                    target,
                    lower,
                    upper,
                    face,
                    mode,
                )
                expected = physical_halo_source_index_reference(
                    target,
                    int(lower[axis]),
                    int(upper[axis]),
                    face,
                    mode,
                )
                assert type(actual) is int
                assert actual == expected

        deep_lower = int(lower[axis]) - extent - 7
        deep_upper = int(upper[axis]) + extent + 6
        assert physical_halo_source_index(
            deep_lower,
            lower,
            upper,
            2 * axis,
            0,
        ) == lower[axis]
        assert physical_halo_source_index(
            deep_upper,
            lower,
            upper,
            2 * axis + 1,
            3,
        ) == upper[axis] - 1


def test_safe_int64_boundary_reflection_and_excess_depth_taxonomy() -> None:
    maximum = np.iinfo(np.int64).max
    minimum = np.iinfo(np.int64).min
    lower = i3(maximum - 1, 0, 0)
    upper = i3(maximum, 1, 1)
    assert physical_halo_source_index(
        np.int64(maximum),
        lower,
        upper,
        np.int64(1),
        np.uint8(1),
    ) == maximum - 1

    unit_lower = i3(0, 0, 0)
    unit_upper = i3(1, 1, 1)
    with pytest.raises(ValueError, match="depth"):
        physical_halo_source_index(
            minimum,
            unit_lower,
            unit_upper,
            0,
            1,
        )
    assert physical_halo_source_index(
        minimum,
        unit_lower,
        unit_upper,
        0,
        0,
    ) == 0
    with pytest.raises(OverflowError, match="int64"):
        physical_halo_source_index(
            int(maximum) + 1,
            unit_lower,
            unit_upper,
            1,
            0,
        )


def test_value_rules_preserve_or_toggle_exact_binary64_bits() -> None:
    bit_patterns = [
        0x0000000000000000,
        0x8000000000000000,
        0x3FF8000000000000,
        0xBFF8000000000000,
        0x7FF0000000000000,
        0xFFF0000000000000,
        0x7FF8000000001234,
        0xFFF8000000005678,
        0x7FF0000000001234,
        0xFFF0000000005678,
    ]
    with np.errstate(invalid="ignore"):
        for pattern in bit_patterns:
            value = np.asarray([pattern], dtype=np.uint64).view(np.float64)[0]
            for face in range(6):
                for mode in range(4):
                    for normal_slot in (-1, 2):
                        actual = transform_physical_halo_value(
                            value,
                            np.int64(2),
                            np.int64(normal_slot),
                            np.uint8(face),
                            np.uint8(mode),
                        )
                        expected = transform_physical_halo_value_reference(
                            value,
                            2,
                            normal_slot,
                            face,
                            mode,
                        )
                        assert type(actual) is float
                        assert bits(actual) == bits(expected)


def test_random_scalar_rules_match_independent_reference() -> None:
    rng = np.random.default_rng(20260828)
    for _ in range(1_000):
        lower_values = rng.integers(0, 1_000, size=3, dtype=np.int64)
        extents = rng.integers(1, 50, size=3, dtype=np.int64)
        upper_values = lower_values + extents
        face = int(rng.integers(0, 6))
        axis = face // 2
        layer = int(rng.integers(1, int(extents[axis]) + 1))
        target = (
            int(lower_values[axis]) - layer
            if face % 2 == 0
            else int(upper_values[axis]) + layer - 1
        )
        mode = int(rng.integers(0, 4))
        actual = physical_halo_source_index(
            target,
            lower_values,
            upper_values,
            face,
            mode,
        )
        expected = physical_halo_source_index_reference(
            target,
            int(lower_values[axis]),
            int(upper_values[axis]),
            face,
            mode,
        )
        assert actual == expected

        pattern = rng.integers(0, 1 << 64, dtype=np.uint64)
        value = np.asarray([pattern], dtype=np.uint64).view(np.float64)[0]
        field = int(rng.integers(0, 4))
        normal = int(rng.integers(-1, 4))
        with np.errstate(invalid="ignore"):
            actual_value = transform_physical_halo_value(
                value,
                field,
                normal,
                face,
                mode,
            )
            expected_value = transform_physical_halo_value_reference(
                value,
                field,
                normal,
                face,
                mode,
            )
        assert bits(actual_value) == bits(expected_value)


def test_invalid_source_inputs_and_read_only_triplets() -> None:
    lower = i3(1, 1, 1)
    upper = i3(3, 3, 3)
    lower.setflags(write=False)
    upper.setflags(write=False)
    assert physical_halo_source_index(0, lower, upper, 0, 1) == 1

    for invalid in (True, np.bool_(False), 1.5, np.array(0, dtype=np.int64)):
        with pytest.raises(TypeError, match="integer scalar"):
            physical_halo_source_index(invalid, lower, upper, 0, 0)
    with pytest.raises(TypeError, match="int64"):
        physical_halo_source_index(
            0,
            lower.astype(np.int32),
            upper,
            0,
            0,
        )
    with pytest.raises(ValueError, match="C-contiguous"):
        physical_halo_source_index(
            0,
            np.arange(6, dtype=np.int64)[::2],
            upper,
            0,
            0,
        )
    with pytest.raises(ValueError, match="nonempty"):
        physical_halo_source_index(0, i3(1, 1, 1), i3(1, 2, 2), 0, 0)
    with pytest.raises(ValueError, match="lower face"):
        physical_halo_source_index(1, lower, upper, 0, 0)
    with pytest.raises(ValueError, match="upper face"):
        physical_halo_source_index(2, lower, upper, 1, 0)
    with pytest.raises(ValueError, match="face"):
        physical_halo_source_index(0, lower, upper, 6, 0)
    with pytest.raises(ValueError, match="mode"):
        physical_halo_source_index(0, lower, upper, 0, 4)
    with pytest.raises(TypeError, match="integer scalar"):
        physical_halo_source_index(0, lower, upper, True, 0)
    with pytest.raises(TypeError, match="integer scalar"):
        physical_halo_source_index(0, lower, upper, 0, np.array(0))


def test_invalid_value_inputs_are_explicit() -> None:
    for invalid in (
        1,
        True,
        np.float32(1.0),
        np.longdouble(1.0),
        np.array(1.0, dtype=np.float64),
    ):
        with pytest.raises(TypeError, match="exact float"):
            transform_physical_halo_value(invalid, 0, -1, 0, 0)
    with pytest.raises(ValueError, match="field_position"):
        transform_physical_halo_value(1.0, -1, -1, 0, 0)
    with pytest.raises(ValueError, match="normal_field_slot"):
        transform_physical_halo_value(1.0, 0, -2, 0, 0)
    with pytest.raises(TypeError, match="integer scalar"):
        transform_physical_halo_value(1.0, True, -1, 0, 0)
    with pytest.raises(OverflowError, match="int64"):
        transform_physical_halo_value(
            1.0,
            0,
            int(np.iinfo(np.int64).max) + 1,
            0,
            0,
        )
    with pytest.raises(ValueError, match="face"):
        transform_physical_halo_value(1.0, 0, -1, -1, 0)
    with pytest.raises(ValueError, match="mode"):
        transform_physical_halo_value(1.0, 0, -1, 0, 9)


def test_inline_aggregate_hal_paths_remain_bitwise_equal_to_references() -> None:
    root = i3(2, 1, 1)
    coord_to_rank, rank_to_coord = level1_morton(root)
    faces = level1_face_neighbors(root, coord_to_rank, rank_to_coord)
    block_ids = i3(0, 1)
    lower = i3(2, 1, 2)
    upper = lower + i3(4, 3, 5)
    spatial_shape = tuple(int(value) for value in upper + i3(1, 2, 1))
    x, y, z = np.indices((4, 3, 5), dtype=np.float64)
    payload = np.full((2, 3, *spatial_shape), np.nan)
    for slot in range(2):
        for field in range(3):
            payload[
                slot,
                field,
                lower[0] : upper[0],
                lower[1] : upper[1],
                lower[2] : upper[2],
            ] = 1000.0 * slot + 100.0 * field + 10.0 * x + y + 0.01 * z
    payload[:, 1, lower[0] : upper[0], lower[1] : upper[1], lower[2] : upper[2]] = (
        1.5 - x
    )
    modes = np.array(
        [[3, 0, 2, 1, 0, 2], [3, 3, 1, 2, 0, 1], [2, 1, 3, 3, 2, 0]],
        dtype=np.uint8,
    )
    normals = i3(1, 2, -1)
    expected = payload.copy()
    fill_physical_halos_reference(
        expected,
        lower,
        upper,
        block_ids,
        faces,
        modes,
        normals,
    )
    fill_same_level_halos_reference(
        expected,
        lower,
        upper,
        block_ids,
        2,
        faces,
        modes,
        normals,
    )
    fill_physical_halos(
        payload,
        lower,
        upper,
        block_ids,
        faces,
        modes,
        normals,
    )
    fill_same_level_halos(
        payload,
        lower,
        upper,
        block_ids,
        2,
        faces,
        modes,
        normals,
    )
    assert np.array_equal(payload.view(np.uint64), expected.view(np.uint64))
