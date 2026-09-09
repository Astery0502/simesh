from __future__ import annotations

import itertools

import numpy as np
import pytest

from simesh_rewrite.halo_apply_reference import (
    apply_level1_same_level_halo_plan_reference,
)
from simesh_rewrite.physical_widening import (
    _validate_physical_widening,
    apply_cartesian_physical_widening,
)
from simesh_rewrite.physical_widening_reference import (
    apply_cartesian_physical_widening_reference,
)


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def rows(*values: tuple[int, int, int]) -> np.ndarray:
    return np.asarray(values, dtype=np.int64).reshape(-1, 3)


def assert_bits_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def axis_coded_payload(
    slots: int,
    fields: int,
    shape: tuple[int, int, int],
) -> np.ndarray:
    x, y, z = np.indices(shape, dtype=np.float64)
    payload = np.empty((slots, fields, *shape), dtype=np.float64)
    for slot in range(slots):
        for field in range(fields):
            payload[slot, field] = (
                100000.0 * slot
                + 10000.0 * field
                + 100.0 * x
                + 10.0 * y
                + z
                + 1.0
            )
    return payload


def invoke(case: dict) -> None:
    apply_cartesian_physical_widening(
        case["payload"],
        case["target_slot"],
        case["logical_interior_lower"],
        case["logical_interior_upper"],
        case["storage_logical_offsets"],
        case["directions"],
        case["physical_masks"],
        case["base_lower"],
        case["base_upper"],
        case["target_lower"],
        case["target_upper"],
        case["boundary_modes"],
        case["normal_field_slots"],
    )


def reference(case: dict) -> None:
    apply_cartesian_physical_widening_reference(
        case["payload"],
        case["target_slot"],
        case["logical_interior_lower"],
        case["logical_interior_upper"],
        case["storage_logical_offsets"],
        case["directions"],
        case["physical_masks"],
        case["base_lower"],
        case["base_upper"],
        case["target_lower"],
        case["target_upper"],
        case["boundary_modes"],
        case["normal_field_slots"],
    )


def valid_case(fields: int = 1) -> dict:
    payload = axis_coded_payload(2, fields, (8, 8, 8))
    return {
        "payload": payload,
        "target_slot": 0,
        "logical_interior_lower": rows((2, 2, 2)),
        "logical_interior_upper": rows((6, 6, 6)),
        "storage_logical_offsets": rows((0, 0, 0)),
        "directions": rows((-1, 0, 0)),
        "physical_masks": np.asarray([1], dtype=np.uint8),
        "base_lower": rows((2, 2, 2)),
        "base_upper": rows((6, 6, 6)),
        "target_lower": rows((0, 2, 2)),
        "target_upper": rows((2, 6, 6)),
        "boundary_modes": np.zeros((fields, 6), dtype=np.uint8),
        "normal_field_slots": i3(-1, -1, -1),
    }


def assert_atomic_error(case: dict, error, match: str) -> None:
    before = case["payload"].copy()
    with pytest.raises(error, match=match):
        invoke(case)
    assert_bits_equal(case["payload"], before)


def test_all_faces_modes_and_two_layers_match_reference() -> None:
    interior_lower = (2, 2, 2)
    interior_upper = (6, 7, 8)
    shape = (8, 9, 10)
    for face, mode in itertools.product(range(6), range(4)):
        axis = face // 2
        direction = [0, 0, 0]
        direction[axis] = -1 if face % 2 == 0 else 1
        target_lower = list(interior_lower)
        target_upper = list(interior_upper)
        if direction[axis] < 0:
            target_lower[axis] = 0
            target_upper[axis] = interior_lower[axis]
        else:
            target_lower[axis] = interior_upper[axis]
            target_upper[axis] = shape[axis]

        payload = axis_coded_payload(2, 1, shape)
        case = {
            "payload": payload,
            "target_slot": 0,
            "logical_interior_lower": rows(interior_lower),
            "logical_interior_upper": rows(interior_upper),
            "storage_logical_offsets": rows((0, 0, 0)),
            "directions": rows(tuple(direction)),
            "physical_masks": np.asarray([1 << axis], dtype=np.uint8),
            "base_lower": rows(interior_lower),
            "base_upper": rows(interior_upper),
            "target_lower": rows(tuple(target_lower)),
            "target_upper": rows(tuple(target_upper)),
            "boundary_modes": np.zeros((1, 6), dtype=np.uint8),
            "normal_field_slots": i3(0, 0, 0),
        }
        case["boundary_modes"][0, face] = mode
        expected = {**case, "payload": payload.copy()}
        reference(expected)
        invoke(case)
        assert_bits_equal(case["payload"], expected["payload"])


def test_mixed_noncommuting_transforms_follow_x_y_z_order() -> None:
    payload = np.full((2, 2, 6, 6, 6), np.float64(np.nan))
    payload[0, 0, 2:4, 2:4, 2:4] = 3.0
    payload[0, 1, 2:4, 2:4, 2:4] = 5.0
    payload[1] = 91.0
    modes = np.zeros((2, 6), dtype=np.uint8)
    modes[0, 0] = 3
    modes[0, 2] = 2
    modes[1, 0] = 2
    modes[1, 2] = 3
    case = {
        "payload": payload,
        "target_slot": 0,
        "logical_interior_lower": rows((2, 2, 2)),
        "logical_interior_upper": rows((4, 4, 4)),
        "storage_logical_offsets": rows((0, 0, 0)),
        "directions": rows((-1, -1, 0)),
        "physical_masks": np.asarray([3], dtype=np.uint8),
        "base_lower": rows((2, 2, 2)),
        "base_upper": rows((4, 4, 4)),
        "target_lower": rows((0, 0, 2)),
        "target_upper": rows((2, 2, 4)),
        "boundary_modes": modes,
        "normal_field_slots": i3(0, 1, -1),
    }
    expected = {**case, "payload": payload.copy()}
    reference(expected)
    invoke(case)
    assert_bits_equal(payload, expected["payload"])
    assert payload[0, 0, 1, 1, 2].view(np.uint64) == np.uint64(
        0x8000000000000000
    )
    assert np.all(payload[1] == 91.0)


def test_three_axis_physical_corner_matches_direct_reference() -> None:
    payload = axis_coded_payload(1, 3, (8, 8, 8))
    modes = np.zeros((3, 6), dtype=np.uint8)
    modes[:, 0] = (1, 2, 3)
    modes[:, 2] = (2, 3, 1)
    modes[:, 4] = (3, 1, 2)
    case = {
        "payload": payload,
        "target_slot": 0,
        "logical_interior_lower": rows((2, 2, 2)),
        "logical_interior_upper": rows((6, 6, 6)),
        "storage_logical_offsets": rows((0, 0, 0)),
        "directions": rows((-1, -1, -1)),
        "physical_masks": np.asarray([7], dtype=np.uint8),
        "base_lower": rows((2, 2, 2)),
        "base_upper": rows((6, 6, 6)),
        "target_lower": rows((0, 0, 0)),
        "target_upper": rows((2, 2, 2)),
        "boundary_modes": modes,
        "normal_field_slots": i3(2, 1, 0),
    }
    expected = {**case, "payload": payload.copy()}
    reference(expected)
    invoke(case)
    assert_bits_equal(payload, expected["payload"])


def test_antisymmetry_preserves_nan_payloads_and_toggles_sign_bits() -> None:
    payload = np.zeros((1, 1, 4, 4, 4), dtype=np.float64)
    patterns = np.asarray(
        [
            0x0000000000000000,
            0x8000000000000000,
            0x7FF8000000001234,
            0x7FF0000000005678,
        ],
        dtype=np.uint64,
    ).view(np.float64).reshape(2, 2)
    payload[0, 0, 1, 1:3, 1:3] = patterns
    case = {
        "payload": payload,
        "target_slot": 0,
        "logical_interior_lower": rows((1, 1, 1)),
        "logical_interior_upper": rows((3, 3, 3)),
        "storage_logical_offsets": rows((0, 0, 0)),
        "directions": rows((-1, 0, 0)),
        "physical_masks": np.asarray([1], dtype=np.uint8),
        "base_lower": rows((1, 1, 1)),
        "base_upper": rows((3, 3, 3)),
        "target_lower": rows((0, 1, 1)),
        "target_upper": rows((1, 3, 3)),
        "boundary_modes": np.asarray([[2, 0, 0, 0, 0, 0]], dtype=np.uint8),
        "normal_field_slots": i3(-1, -1, -1),
    }
    with np.errstate(invalid="ignore"):
        invoke(case)
    expected = patterns.view(np.uint64) ^ np.uint64(1 << 63)
    assert np.array_equal(payload[0, 0, 0, 1:3, 1:3].view(np.uint64), expected)


def test_noinflow_preserves_signed_zeros_and_nan_bits() -> None:
    payload = np.zeros((1, 1, 4, 4, 4), dtype=np.float64)
    source_bits = np.asarray(
        [
            0x0000000000000000,
            0x8000000000000000,
            0x7FF0000000000000,
            0xFFF8000000001234,
        ],
        dtype=np.uint64,
    ).reshape(2, 2)
    payload[0, 0, 1, 1:3, 1:3] = source_bits.view(np.float64)
    case = {
        "payload": payload,
        "target_slot": 0,
        "logical_interior_lower": rows((1, 1, 1)),
        "logical_interior_upper": rows((3, 3, 3)),
        "storage_logical_offsets": rows((0, 0, 0)),
        "directions": rows((-1, 0, 0)),
        "physical_masks": np.asarray([1], dtype=np.uint8),
        "base_lower": rows((1, 1, 1)),
        "base_upper": rows((3, 3, 3)),
        "target_lower": rows((0, 1, 1)),
        "target_upper": rows((1, 3, 3)),
        "boundary_modes": np.asarray([[3, 0, 0, 0, 0, 0]], dtype=np.uint8),
        "normal_field_slots": i3(0, -1, -1),
    }
    with np.errstate(invalid="ignore"):
        invoke(case)
    expected = source_bits.copy()
    expected[1, 0] = 0
    assert np.array_equal(payload[0, 0, 0, 1:3, 1:3].view(np.uint64), expected)


def test_translated_cwp_style_frame_matches_reference() -> None:
    payload = axis_coded_payload(1, 2, (3, 7, 7))
    modes = np.zeros((2, 6), dtype=np.uint8)
    modes[0, 2] = 1
    modes[1, 2] = 2
    case = {
        "payload": payload,
        "target_slot": 0,
        "logical_interior_lower": rows((2, 2, 2)),
        "logical_interior_upper": rows((10, 10, 10)),
        "storage_logical_offsets": rows((8, 1, 1)),
        "directions": rows((1, -1, 0)),
        "physical_masks": np.asarray([2], dtype=np.uint8),
        "base_lower": rows((0, 1, 1)),
        "base_upper": rows((2, 6, 5)),
        "target_lower": rows((0, 0, 1)),
        "target_upper": rows((2, 1, 5)),
        "boundary_modes": modes,
        "normal_field_slots": i3(-1, -1, -1),
    }
    expected = {**case, "payload": payload.copy()}
    reference(expected)
    invoke(case)
    assert_bits_equal(payload, expected["payload"])


def test_negative_fine_owner_offset_writes_only_target_slot_one() -> None:
    payload = axis_coded_payload(2, 4, (8, 8, 8))
    before = payload.copy()
    modes = np.zeros((4, 6), dtype=np.uint8)
    modes[:, 0] = (0, 1, 2, 3)
    case = {
        "payload": payload,
        "target_slot": 1,
        "logical_interior_lower": rows((0, 0, 0)),
        "logical_interior_upper": rows((4, 4, 4)),
        "storage_logical_offsets": rows((-2, -2, -2)),
        "directions": rows((-1, 1, 0)),
        "physical_masks": np.asarray([1], dtype=np.uint8),
        "base_lower": rows((2, 2, 2)),
        "base_upper": rows((6, 6, 6)),
        "target_lower": rows((0, 2, 2)),
        "target_upper": rows((2, 6, 6)),
        "boundary_modes": modes,
        "normal_field_slots": i3(3, -1, -1),
    }
    expected = {**case, "payload": payload.copy()}
    reference(expected)
    invoke(case)
    assert_bits_equal(payload, expected["payload"])
    assert_bits_equal(payload[:1], before[:1])
    assert_bits_equal(payload[1:, :, 2:6, 2:6, 2:6], before[1:, :, 2:6, 2:6, 2:6])


def test_disjoint_multirow_batch_preserves_other_slot_and_bases() -> None:
    payload = axis_coded_payload(2, 2, (8, 8, 8))
    before = payload.copy()
    case = {
        "payload": payload,
        "target_slot": 0,
        "logical_interior_lower": rows((2, 2, 2), (2, 2, 2)),
        "logical_interior_upper": rows((6, 6, 6), (6, 6, 6)),
        "storage_logical_offsets": rows((0, 0, 0), (0, 0, 0)),
        "directions": rows((-1, 0, 0), (1, 0, 0)),
        "physical_masks": np.asarray([1, 1], dtype=np.uint8),
        "base_lower": rows((2, 2, 2), (2, 2, 2)),
        "base_upper": rows((6, 6, 6), (6, 6, 6)),
        "target_lower": rows((0, 2, 2), (6, 2, 2)),
        "target_upper": rows((2, 6, 6), (8, 6, 6)),
        "boundary_modes": np.asarray(
            [[1, 2, 0, 0, 0, 0], [2, 1, 0, 0, 0, 0]], dtype=np.uint8
        ),
        "normal_field_slots": i3(-1, -1, -1),
    }
    for value in (
        case["logical_interior_lower"],
        case["logical_interior_upper"],
        case["storage_logical_offsets"],
        case["directions"],
        case["physical_masks"],
        case["base_lower"],
        case["base_upper"],
        case["target_lower"],
        case["target_upper"],
    ):
        value.setflags(write=False)
    expected = {**case, "payload": payload.copy()}
    reference(expected)
    invoke(case)
    assert_bits_equal(payload, expected["payload"])
    assert_bits_equal(payload[1:], before[1:])
    assert_bits_equal(payload[:, :, 2:6, 2:6, 2:6], before[:, :, 2:6, 2:6, 2:6])


def test_private_preflight_normalizes_without_mutating_payload() -> None:
    case = valid_case()
    before = case["payload"].copy()
    normalized = _validate_physical_widening(
        case["payload"],
        np.int64(case["target_slot"]),
        case["logical_interior_lower"],
        case["logical_interior_upper"],
        case["storage_logical_offsets"],
        case["directions"],
        case["physical_masks"],
        case["base_lower"],
        case["base_upper"],
        case["target_lower"],
        case["target_upper"],
        case["boundary_modes"],
        case["normal_field_slots"],
    )
    assert normalized[0] is case["payload"]
    assert normalized[1] == 0
    for actual, name in zip(
        normalized[2:],
        (
            "logical_interior_lower",
            "logical_interior_upper",
            "storage_logical_offsets",
            "directions",
            "physical_masks",
            "base_lower",
            "base_upper",
            "target_lower",
            "target_upper",
            "boundary_modes",
            "normal_field_slots",
        ),
        strict=True,
    ):
        assert actual is case[name]
    assert_bits_equal(case["payload"], before)


def test_complete_mixed_same_base_matches_hax_reference() -> None:
    payload = np.full((2, 2, 8, 8, 8), np.nan, dtype=np.float64)
    support = axis_coded_payload(1, 2, (8, 8, 8))[0]
    support[1, 2:6, 2:6, 2:6] -= 20000.0
    payload[1] = support
    for x, y, z in itertools.product(range(2, 6), range(0, 2), range(2, 6)):
        payload[0, :, x, y, z] = payload[1, :, x, y + 4, z]

    modes = np.zeros((2, 6), dtype=np.uint8)
    modes[0, 0] = 2
    modes[1, 0] = 3
    normals = i3(1, -1, -1)
    expected = payload.copy()
    source_slots = np.full((1, 27), -1, dtype=np.int64)
    hax_masks = np.zeros((1, 27), dtype=np.uint8)
    column = 9
    source_slots[0, column] = 1
    hax_masks[0, column] = 1
    apply_level1_same_level_halo_plan_reference(
        expected,
        i3(2, 2, 2),
        i3(6, 6, 6),
        source_slots,
        hax_masks,
        modes,
        normals,
    )

    apply_cartesian_physical_widening(
        payload,
        0,
        rows((2, 2, 2)),
        rows((6, 6, 6)),
        rows((0, 0, 0)),
        rows((-1, -1, 0)),
        np.asarray([1], dtype=np.uint8),
        rows((2, 0, 2)),
        rows((6, 2, 6)),
        rows((0, 0, 2)),
        rows((2, 2, 6)),
        modes,
        normals,
    )
    assert_bits_equal(payload, expected)


def test_empty_rows_fields_and_boundary_anchored_empty_target_are_noops() -> None:
    payload = axis_coded_payload(1, 1, (8, 8, 8))
    before = payload.copy()
    empty_rows = np.empty((0, 3), dtype=np.int64)
    apply_cartesian_physical_widening(
        payload,
        0,
        empty_rows,
        empty_rows,
        empty_rows,
        empty_rows,
        np.empty(0, dtype=np.uint8),
        empty_rows,
        empty_rows,
        empty_rows,
        empty_rows,
        np.zeros((1, 6), dtype=np.uint8),
        i3(-1, -1, -1),
    )
    assert_bits_equal(payload, before)

    empty_fields = np.empty((1, 0, 8, 8, 8), dtype=np.float64)
    case = valid_case(fields=0)
    case["payload"] = empty_fields
    invoke(case)
    bad_normal = {**case, "normal_field_slots": i3(0, -1, -1)}
    assert_atomic_error(bad_normal, ValueError, "normal_field_slots")

    case = valid_case()
    case["target_lower"] = rows((2, 2, 2))
    case["target_upper"] = rows((2, 6, 6))
    case["boundary_modes"][0, 0] = 1
    before = case["payload"].copy()
    invoke(case)
    assert_bits_equal(case["payload"], before)


def test_empty_rows_still_validate_slot_and_boundary_table() -> None:
    payload = np.zeros((1, 1, 2, 2, 2), dtype=np.float64)
    empty = np.empty((0, 3), dtype=np.int64)
    args = dict(
        payload=payload,
        target_slot=1,
        logical_interior_lower=empty,
        logical_interior_upper=empty,
        storage_logical_offsets=empty,
        directions=empty,
        physical_masks=np.empty(0, dtype=np.uint8),
        base_lower=empty,
        base_upper=empty,
        target_lower=empty,
        target_upper=empty,
        boundary_modes=np.zeros((1, 6), dtype=np.uint8),
        normal_field_slots=i3(-1, -1, -1),
    )
    assert_atomic_error(args, ValueError, "target_slot")
    args["target_slot"] = 0
    args["boundary_modes"][0, 0] = 4
    assert_atomic_error(args, ValueError, "unknown mode")


def test_noinflow_configuration_and_translation_overflow_are_atomic() -> None:
    case = valid_case()
    case["boundary_modes"][0, 0] = 3
    assert_atomic_error(case, ValueError, "no-inflow")

    case = valid_case()
    case["storage_logical_offsets"][0, 0] = np.iinfo(np.int64).max
    assert_atomic_error(case, OverflowError, "forward target translation")


def test_reflection_depth_and_mapped_image_errors_are_atomic() -> None:
    case = valid_case()
    case["logical_interior_upper"][0, 0] = 3
    case["boundary_modes"][0, 0] = 1
    assert_atomic_error(case, ValueError, "reflected target depth")

    case = valid_case()
    case["base_lower"][0, 0] = 3
    assert_atomic_error(case, ValueError, "mapped source image")


def test_side_direction_mask_and_box_errors_are_atomic() -> None:
    case = valid_case()
    case["target_upper"][0, 0] = 1
    assert_atomic_error(case, ValueError, "boundary-anchored")

    case = valid_case()
    case["directions"][0, 0] = 0
    assert_atomic_error(case, ValueError, "zero direction")

    case = valid_case()
    case["physical_masks"][0] = 0
    assert_atomic_error(case, ValueError, "nonzero")

    case = valid_case()
    case["base_upper"][0, 0] = case["base_lower"][0, 0]
    assert_atomic_error(case, ValueError, "base boxes")


def test_pairwise_waw_and_cross_row_raw_are_atomic() -> None:
    case = valid_case()
    for name in (
        "logical_interior_lower",
        "logical_interior_upper",
        "storage_logical_offsets",
        "directions",
        "base_lower",
        "base_upper",
        "target_lower",
        "target_upper",
    ):
        case[name] = np.concatenate((case[name], case[name]), axis=0)
    case["physical_masks"] = np.asarray([1, 1], dtype=np.uint8)
    assert_atomic_error(case, ValueError, "pairwise disjoint")

    case = valid_case()
    case.update(
        logical_interior_lower=rows((2, 2, 2), (2, 2, 2)),
        logical_interior_upper=rows((6, 6, 6), (6, 6, 6)),
        storage_logical_offsets=rows((0, 0, 0), (0, 0, 0)),
        directions=rows((-1, 0, 0), (-1, -1, 0)),
        physical_masks=np.asarray([1, 2], dtype=np.uint8),
        base_lower=rows((2, 2, 2), (0, 2, 2)),
        base_upper=rows((6, 4, 4), (2, 6, 4)),
        target_lower=rows((0, 2, 2), (0, 0, 2)),
        target_upper=rows((2, 4, 4), (2, 2, 4)),
    )
    assert_atomic_error(case, ValueError, "must not overlap any batch base")


def test_payload_metadata_alias_and_representations_are_atomic() -> None:
    case = valid_case()
    alias = case["payload"].view(np.int64).reshape(-1)[:3].reshape(1, 3)
    alias[:] = (-1, 0, 0)
    case["directions"] = alias
    assert_atomic_error(case, ValueError, "must not overlap")

    case = valid_case()
    case["directions"] = case["directions"].astype(np.int32)
    assert_atomic_error(case, TypeError, "dtype int64")

    case = valid_case()
    case["directions"] = np.asarray([[-1, 0, 0], [-1, 0, 0]], dtype=np.int64)[
        :1, ::-1
    ]
    assert_atomic_error(case, ValueError, "C-contiguous")

    case = valid_case()
    case["payload"].setflags(write=False)
    with pytest.raises(ValueError, match="writable"):
        invoke(case)


def test_target_slot_scalar_rules_are_explicit() -> None:
    case = valid_case()
    case["target_slot"] = True
    assert_atomic_error(case, TypeError, "integer scalar")

    case = valid_case()
    case["target_slot"] = np.array(0, dtype=np.int64)
    assert_atomic_error(case, TypeError, "integer scalar")

    case = valid_case()
    case["target_slot"] = int(np.iinfo(np.int64).max) + 1
    assert_atomic_error(case, OverflowError, "int64")
