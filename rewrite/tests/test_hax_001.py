from __future__ import annotations

import numpy as np
import pytest

from simesh_rewrite.chunking import plan_level1_halo_chunk
from simesh_rewrite.halo_apply import apply_level1_same_level_halo_plan
from simesh_rewrite.halo_apply_reference import (
    apply_level1_same_level_halo_plan_reference,
)
from simesh_rewrite.halo_plans import level1_halo_relation_plan
from simesh_rewrite.halos import fill_physical_halos, fill_same_level_halos
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.storage import gather_blocks_into
from simesh_rewrite.topology import level1_face_neighbors


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def topology(shape: tuple[int, int, int]):
    root = i3(*shape)
    coord_to_rank, rank_to_coord = level1_morton(root)
    return coord_to_rank, level1_face_neighbors(
        root,
        coord_to_rank,
        rank_to_coord,
    )


def axis_coded_backing(
    block_count: int,
    field_count: int,
    block_shape: tuple[int, int, int],
) -> np.ndarray:
    x, y, z = np.indices(block_shape, dtype=np.float64)
    backing = np.empty((block_count, field_count, *block_shape), dtype=np.float64)
    for block in range(block_count):
        for field in range(field_count):
            backing[block, field] = (
                100000.0 * block
                + 10000.0 * field
                + 100.0 * x
                + 10.0 * y
                + z
                + 1.0
            )
    return backing


def prepared_case(
    root_shape: tuple[int, int, int] = (3, 2, 2),
    block_shape: tuple[int, int, int] = (4, 3, 5),
):
    _, faces = topology(root_shape)
    block_ids = np.arange(faces.shape[0], dtype=np.int64)[::-1].copy()
    primary_count = block_ids.size
    source_slots, masks = level1_halo_relation_plan(
        block_ids,
        primary_count,
        faces,
    )
    lower = i3(2, 1, 2)
    upper = lower + i3(*block_shape)
    spatial = tuple(int(value) for value in upper + i3(1, 2, 1))
    backing = axis_coded_backing(faces.shape[0], 3, block_shape)
    x = np.indices(block_shape, dtype=np.float64)[0]
    backing[:, 1] = 1.5 - x
    payload = np.full((block_ids.size, 3, *spatial), np.nan)
    gather_blocks_into(
        backing,
        i3(0, 0, 0),
        i3(*block_shape),
        block_ids,
        i3(0, 1, 2),
        payload,
        lower,
    )
    modes = np.array(
        [
            [3, 0, 2, 1, 0, 2],
            [3, 3, 1, 2, 0, 1],
            [2, 1, 3, 3, 2, 0],
        ],
        dtype=np.uint8,
    )
    normals = i3(1, 2, -1)
    fill_physical_halos(
        payload,
        lower,
        upper,
        block_ids,
        faces,
        modes,
        normals,
    )
    return (
        payload,
        lower,
        upper,
        source_slots,
        masks,
        modes,
        normals,
        block_ids,
        faces,
    )


def assert_bits_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_explicit_apply_matches_reference_and_fused_wrapper_bitwise() -> None:
    (
        payload,
        lower,
        upper,
        source_slots,
        masks,
        modes,
        normals,
        block_ids,
        faces,
    ) = prepared_case()
    expected = payload.copy()
    apply_level1_same_level_halo_plan_reference(
        expected,
        lower,
        upper,
        source_slots,
        masks,
        modes,
        normals,
    )
    fused = payload.copy()
    fill_same_level_halos(
        fused,
        lower,
        upper,
        block_ids,
        source_slots.shape[0],
        faces,
        modes,
        normals,
    )
    apply_level1_same_level_halo_plan(
        payload,
        lower,
        upper,
        source_slots,
        masks,
        modes,
        normals,
    )
    assert_bits_equal(payload, expected)
    assert_bits_equal(payload, fused)


def test_bounded_plan_apply_preserves_support_and_matches_fused() -> None:
    coord_to_rank, faces = topology((4, 4, 3))
    ids = np.empty(64, dtype=np.int64)
    first = int(coord_to_rank[2, 2, 1])
    primary_count, selected_count = plan_level1_halo_chunk(first, faces, ids)
    block_ids = ids[:selected_count]
    source_slots, masks = level1_halo_relation_plan(
        block_ids,
        primary_count,
        faces,
    )
    block_shape = (4, 4, 4)
    lower = i3(1, 2, 1)
    upper = lower + i3(*block_shape)
    spatial = tuple(int(value) for value in upper + i3(2, 1, 2))
    payload = np.full((selected_count, 2, *spatial), np.nan)
    gather_blocks_into(
        axis_coded_backing(faces.shape[0], 2, block_shape),
        i3(0, 0, 0),
        i3(*block_shape),
        block_ids,
        i3(0, 1),
        payload,
        lower,
    )
    modes = np.array([[1, 2, 0, 3, 2, 0], [3, 0, 2, 1, 0, 2]], dtype=np.uint8)
    normals = i3(1, 0, -1)
    fill_physical_halos(
        payload,
        lower,
        upper,
        block_ids,
        faces,
        modes,
        normals,
    )
    before = payload.copy()
    fused = payload.copy()
    fill_same_level_halos(
        fused,
        lower,
        upper,
        block_ids,
        primary_count,
        faces,
        modes,
        normals,
    )
    apply_level1_same_level_halo_plan(
        payload,
        lower,
        upper,
        source_slots,
        masks,
        modes,
        normals,
    )
    assert_bits_equal(payload, fused)
    assert_bits_equal(payload[primary_count:], before[primary_count:])
    interior = (
        slice(None),
        slice(None),
        slice(lower[0], upper[0]),
        slice(lower[1], upper[1]),
        slice(lower[2], upper[2]),
    )
    assert_bits_equal(payload[:primary_count][interior], before[:primary_count][interior])


def test_special_values_and_read_only_plans_preserve_bits() -> None:
    case = list(prepared_case(root_shape=(2, 1, 1), block_shape=(2, 2, 2)))
    payload = case[0]
    patterns = np.array(
        [
            0x0000000000000000,
            0x8000000000000000,
            0x7FF0000000000000,
            0xFFF0000000000000,
            0x7FF8000000001234,
            0x7FF0000000005678,
            0x3FF0000000000000,
            0xBFF0000000000000,
        ],
        dtype=np.uint64,
    ).view(np.float64).reshape(2, 2, 2)
    payload[
        :,
        0,
        case[1][0] : case[2][0],
        case[1][1] : case[2][1],
        case[1][2] : case[2][2],
    ] = patterns
    case[3].setflags(write=False)
    case[4].setflags(write=False)
    expected = payload.copy()
    apply_level1_same_level_halo_plan_reference(
        expected,
        *case[1:7],
    )
    apply_level1_same_level_halo_plan(
        payload,
        *case[1:7],
    )
    assert_bits_equal(payload, expected)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda slots, masks, count: slots.__setitem__((0, 13), 0),
        lambda slots, masks, count: masks.__setitem__((0, 13), 1),
        lambda slots, masks, count: slots.__setitem__((0, 12), -1),
        lambda slots, masks, count: masks.__setitem__((0, 12), 1),
        lambda slots, masks, count: slots.__setitem__((0, 14), count),
        lambda slots, masks, count: masks.__setitem__((0, 14), 8),
        lambda slots, masks, count: masks.__setitem__((0, 16), 1),
    ],
)
def test_structurally_invalid_plans_are_atomic(mutation) -> None:
    case = list(prepared_case())
    payload = case[0]
    source_slots = case[3].copy()
    masks = case[4].copy()
    mutation(source_slots, masks, payload.shape[0])
    before = payload.copy()
    with pytest.raises(ValueError, match="structurally invalid"):
        apply_level1_same_level_halo_plan(
            payload,
            case[1],
            case[2],
            source_slots,
            masks,
            case[5],
            case[6],
        )
    assert_bits_equal(payload, before)


def test_range_safe_but_wrong_source_identity_is_hpl_precondition() -> None:
    case = list(prepared_case())
    payload = case[0]
    source_slots = case[3].copy()
    correct = payload.copy()
    apply_level1_same_level_halo_plan(
        correct,
        *case[1:7],
    )

    direction = 12
    original = int(source_slots[0, direction])
    replacement = (original + 1) % payload.shape[0]
    source_slots[0, direction] = replacement
    apply_level1_same_level_halo_plan(
        payload,
        case[1],
        case[2],
        source_slots,
        case[4],
        case[5],
        case[6],
    )
    assert not np.array_equal(payload.view(np.uint64), correct.view(np.uint64))


def test_zero_primary_and_zero_field_calls_validate_and_write_nothing() -> None:
    payload = np.arange(54, dtype=np.float64).reshape(2, 1, 3, 3, 3)
    before = payload.copy()
    apply_level1_same_level_halo_plan(
        payload,
        i3(1, 1, 1),
        i3(2, 2, 2),
        np.empty((0, 27), dtype=np.int64),
        np.empty((0, 27), dtype=np.uint8),
        np.zeros((1, 6), dtype=np.uint8),
        i3(-1, -1, -1),
    )
    assert_bits_equal(payload, before)

    empty_fields = np.empty((2, 0, 3, 3, 3), dtype=np.float64)
    source_slots = np.full((1, 27), -1, dtype=np.int64)
    masks = np.zeros((1, 27), dtype=np.uint8)
    for column in range(27):
        dx = column % 3 - 1
        dy = (column // 3) % 3 - 1
        dz = column // 9 - 1
        masks[0, column] = (dx != 0) | ((dy != 0) << 1) | ((dz != 0) << 2)
    apply_level1_same_level_halo_plan(
        empty_fields,
        i3(1, 1, 1),
        i3(2, 2, 2),
        source_slots,
        masks,
        np.empty((0, 6), dtype=np.uint8),
        i3(-1, -1, -1),
    )
    with pytest.raises(ValueError, match="normal_field_slots"):
        apply_level1_same_level_halo_plan(
            empty_fields,
            i3(1, 1, 1),
            i3(2, 2, 2),
            source_slots,
            masks,
            np.empty((0, 6), dtype=np.uint8),
            i3(0, -1, -1),
        )


def test_invalid_metadata_regions_modes_alias_and_writability_are_atomic() -> None:
    case = list(prepared_case())
    payload = case[0]
    before = payload.copy()
    readonly = payload.copy()
    readonly.setflags(write=False)
    with pytest.raises(ValueError, match="writable"):
        apply_level1_same_level_halo_plan(readonly, *case[1:7])

    with pytest.raises(ValueError, match="equal shapes"):
        apply_level1_same_level_halo_plan(
            payload,
            case[1],
            case[2],
            case[3],
            case[4][:-1].copy(),
            case[5],
            case[6],
        )
    bad_modes = case[5].copy()
    bad_modes[0, 0] = 4
    with pytest.raises(ValueError, match="unknown mode"):
        apply_level1_same_level_halo_plan(
            payload,
            case[1],
            case[2],
            case[3],
            case[4],
            bad_modes,
            case[6],
        )
    no_normal = case[6].copy()
    no_normal[0] = -1
    with pytest.raises(ValueError, match="no-inflow"):
        apply_level1_same_level_halo_plan(
            payload,
            case[1],
            case[2],
            case[3],
            case[4],
            case[5],
            no_normal,
        )
    too_wide_lower = i3(5, 1, 2)
    with pytest.raises(ValueError, match="halo width"):
        apply_level1_same_level_halo_plan(
            payload,
            too_wide_lower,
            case[2],
            case[3],
            case[4],
            case[5],
            case[6],
        )

    alias_payload = np.zeros((1, 1, 3, 3, 3), dtype=np.float64)
    alias_slots = alias_payload.reshape(1, 27).view(np.int64)
    alias_masks = np.zeros((1, 27), dtype=np.uint8)
    with pytest.raises(ValueError, match="must not overlap"):
        apply_level1_same_level_halo_plan(
            alias_payload,
            i3(1, 1, 1),
            i3(2, 2, 2),
            alias_slots,
            alias_masks,
            np.zeros((1, 6), dtype=np.uint8),
            i3(-1, -1, -1),
        )
    assert_bits_equal(payload, before)
