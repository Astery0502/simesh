from __future__ import annotations

import numpy as np
import pytest

from simesh_rewrite.chunking import plan_level1_halo_chunk
from simesh_rewrite.halo_plans import (
    fill_level1_halo_relation_plan,
    level1_halo_relation_plan,
)
from simesh_rewrite.halo_plans_reference import (
    level1_halo_relation_plan_reference,
)
from simesh_rewrite.halo_apply_reference import (
    apply_level1_same_level_halo_plan_reference,
)
from simesh_rewrite.halos import fill_physical_halos, fill_same_level_halos
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.topology import level1_face_neighbors


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def topology(shape: tuple[int, int, int]):
    root_shape = i3(*shape)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    faces = level1_face_neighbors(root_shape, coord_to_rank, rank_to_coord)
    return coord_to_rank, faces


def direction_column(dx: int, dy: int, dz: int) -> int:
    return (dz + 1) * 9 + (dy + 1) * 3 + dx + 1


def assert_plan_equal(actual, expected) -> None:
    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])


def test_golden_full_direction_mixed_physical_and_sibling_plan() -> None:
    coord_to_rank, faces = topology((3, 2, 1))
    block_ids = np.arange(faces.shape[0], dtype=np.int64)[::-1].copy()
    slots = {int(block_id): slot for slot, block_id in enumerate(block_ids)}
    actual = level1_halo_relation_plan(block_ids, len(block_ids), faces)
    expected = level1_halo_relation_plan_reference(
        block_ids, len(block_ids), faces
    )
    assert_plan_equal(actual, expected)

    primary = slots[int(coord_to_rank[0, 0, 0])]
    source_slots, masks = actual
    assert source_slots[primary, 13] == -1
    assert masks[primary, 13] == 0
    assert source_slots[primary, direction_column(-1, 0, 0)] == -1
    assert masks[primary, direction_column(-1, 0, 0)] == 1
    y_neighbor_slot = slots[int(coord_to_rank[0, 1, 0])]
    mixed = direction_column(-1, 1, 0)
    assert source_slots[primary, mixed] == y_neighbor_slot
    assert masks[primary, mixed] == 1
    xyz = direction_column(1, 1, -1)
    assert source_slots[primary, xyz] == slots[int(coord_to_rank[1, 1, 0])]
    assert masks[primary, xyz] == 4


def test_compiled_matches_reference_across_random_shapes_orders_and_prefixes() -> None:
    rng = np.random.default_rng(20260828)
    for _ in range(100):
        shape = tuple(int(value) for value in rng.integers(1, 5, size=3))
        _, faces = topology(shape)
        block_ids = rng.permutation(faces.shape[0]).astype(np.int64)
        primary_count = int(rng.integers(0, block_ids.size + 1))
        actual = level1_halo_relation_plan(
            block_ids,
            primary_count,
            faces,
        )
        expected = level1_halo_relation_plan_reference(
            block_ids,
            primary_count,
            faces,
        )
        assert_plan_equal(actual, expected)


def test_singleton_root_has_only_physical_masks_and_no_sources() -> None:
    _, faces = topology((1, 1, 1))
    source_slots, masks = level1_halo_relation_plan(i3(0), 1, faces)
    assert np.all(source_slots == -1)
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                expected = (dx != 0) | ((dy != 0) << 1) | ((dz != 0) << 2)
                assert masks[0, direction_column(dx, dy, dz)] == expected


def test_sto_full_halo_chunk_plan_is_sufficient_and_exact() -> None:
    coord_to_rank, faces = topology((5, 4, 3))
    chunk_ids = np.empty(64, dtype=np.int64)
    first = int(coord_to_rank[2, 2, 1])
    primary_count, selected_count = plan_level1_halo_chunk(
        first,
        faces,
        chunk_ids,
    )
    selected = chunk_ids[:selected_count]
    actual = level1_halo_relation_plan(selected, primary_count, faces)
    expected = level1_halo_relation_plan_reference(
        selected,
        primary_count,
        faces,
    )
    assert_plan_equal(actual, expected)
    assert np.all(actual[0][actual[0] >= 0] < selected_count)


def test_plan_reference_consumer_matches_retained_hal002_bitwise() -> None:
    coord_to_rank, faces = topology((3, 2, 2))
    block_ids = np.arange(faces.shape[0], dtype=np.int64)[::-1].copy()
    primary_count = block_ids.size
    source_slots, masks = level1_halo_relation_plan(
        block_ids,
        primary_count,
        faces,
    )

    block_shape = (4, 3, 5)
    lower = i3(2, 1, 2)
    upper = lower + i3(*block_shape)
    spatial_shape = tuple(int(value) for value in upper + i3(1, 2, 1))
    x, y, z = np.indices(block_shape, dtype=np.float64)
    payload = np.full((block_ids.size, 3, *spatial_shape), np.nan)
    for slot, block_id in enumerate(block_ids):
        for field in range(3):
            payload[
                slot,
                field,
                lower[0] : upper[0],
                lower[1] : upper[1],
                lower[2] : upper[2],
            ] = (
                100000.0 * block_id
                + 10000.0 * field
                + 100.0 * x
                + 10.0 * y
                + z
                + 1.0
            )
    payload[:, 1, lower[0] : upper[0], lower[1] : upper[1], lower[2] : upper[2]] = (
        1.5 - x
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
    fill_same_level_halos(
        payload,
        lower,
        upper,
        block_ids,
        primary_count,
        faces,
        modes,
        normals,
    )
    assert np.array_equal(payload.view(np.uint64), expected.view(np.uint64))


def test_missing_closure_and_invalid_face_range_are_atomic() -> None:
    coord_to_rank, faces = topology((3, 3, 3))
    block_ids = i3(int(coord_to_rank[1, 1, 1]))
    source_slots = np.full((1, 27), -77, dtype=np.int64)
    masks = np.full((1, 27), 0xA5, dtype=np.uint8)
    before = source_slots.copy(), masks.copy()
    with pytest.raises(ValueError, match="lacks selected closure"):
        fill_level1_halo_relation_plan(
            block_ids,
            1,
            faces,
            source_slots,
            masks,
        )
    assert_plan_equal((source_slots, masks), before)

    broken_faces = faces.copy()
    broken_faces[0, 1] = faces.shape[0]
    with pytest.raises(ValueError, match="outside"):
        fill_level1_halo_relation_plan(
            i3(0),
            1,
            broken_faces,
            source_slots,
            masks,
        )
    assert_plan_equal((source_slots, masks), before)


def test_invalid_ids_outputs_and_aliases_fail_before_mutation() -> None:
    _, faces = topology((1, 1, 1))
    block_ids = i3(0)
    source_slots = np.full((1, 27), -77, dtype=np.int64)
    masks = np.full((1, 27), 0xA5, dtype=np.uint8)
    before = source_slots.copy(), masks.copy()

    with pytest.raises(ValueError, match="duplicates"):
        fill_level1_halo_relation_plan(
            i3(0, 0),
            1,
            faces,
            source_slots,
            masks,
        )
    with pytest.raises(ValueError, match="out of range"):
        fill_level1_halo_relation_plan(
            i3(1),
            1,
            faces,
            source_slots,
            masks,
        )
    with pytest.raises(ValueError, match="shape"):
        fill_level1_halo_relation_plan(
            block_ids,
            1,
            faces,
            source_slots[:, :26],
            masks,
        )
    wrong_dtype = source_slots.astype(np.int32)
    with pytest.raises(TypeError, match="int64"):
        fill_level1_halo_relation_plan(
            block_ids,
            1,
            faces,
            wrong_dtype,
            masks,
        )
    readonly = source_slots.copy()
    readonly.setflags(write=False)
    with pytest.raises(ValueError, match="writable"):
        fill_level1_halo_relation_plan(
            block_ids,
            1,
            faces,
            readonly,
            masks,
        )

    shared = np.empty(27 * 8, dtype=np.uint8)
    shared_slots = shared.view(np.int64).reshape(1, 27)
    shared_masks = shared[:27].reshape(1, 27)
    with pytest.raises(ValueError, match="overlap each other"):
        fill_level1_halo_relation_plan(
            block_ids,
            1,
            faces,
            shared_slots,
            shared_masks,
        )
    input_base = np.zeros(27, dtype=np.int64)
    overlapping_ids = input_base[:1]
    overlapping_slots = input_base.reshape(1, 27)
    with pytest.raises(ValueError, match="overlap inputs"):
        fill_level1_halo_relation_plan(
            overlapping_ids,
            1,
            faces,
            overlapping_slots,
            masks,
        )
    assert_plan_equal((source_slots, masks), before)


def test_zero_primary_plan_is_empty_but_still_validates_selected_ids() -> None:
    _, faces = topology((2, 1, 1))
    source_slots = np.empty((0, 27), dtype=np.int64)
    masks = np.empty((0, 27), dtype=np.uint8)
    fill_level1_halo_relation_plan(
        i3(0, 1),
        0,
        faces,
        source_slots,
        masks,
    )
    assert source_slots.shape == masks.shape == (0, 27)
    with pytest.raises(ValueError, match="duplicates"):
        fill_level1_halo_relation_plan(
            i3(0, 0),
            0,
            faces,
            source_slots,
            masks,
        )
