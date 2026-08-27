from __future__ import annotations

import math

import numpy as np
import pytest

from simesh.utils.lib.amr.forest import AMRForest
from simesh.utils.lib.amr.mesh import AMRMesh
from simesh_rewrite.chunking import plan_level1_chunk
from simesh_rewrite.geometry import level1_block_geometry
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.sampling import (
    place_level1_blocks,
    sample_level1_zero_order,
)
from simesh_rewrite.storage import gather_blocks_into
from simesh_rewrite.topology import level1_face_neighbors


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def f3(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def assert_bits_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def maps(root_shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    return level1_morton(i3(*root_shape))


def canonical_face(
    lower: np.float64,
    upper: np.float64,
    spacing: np.float64,
    index: int,
    count: int,
) -> np.float64:
    if index == 0:
        return lower
    if index == count:
        return upper
    product = np.float64(index) * spacing
    return np.float64(lower + product)


def output_center(
    lower: np.float64,
    spacing: np.float64,
    index: int,
) -> np.float64:
    factor = np.float64(index) + np.float64(0.5)
    product = factor * spacing
    return np.float64(lower + product)


def source_cell_index(
    center: np.float64,
    domain_lower: np.float64,
    domain_upper: np.float64,
    native_spacing: np.float64,
    domain_count: int,
) -> int:
    owner = 0
    for face_index in range(1, domain_count):
        face = canonical_face(
            domain_lower,
            domain_upper,
            native_spacing,
            face_index,
            domain_count,
        )
        if face <= center:
            owner = face_index
    return owner


def owner_indices(
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    domain_counts: np.ndarray,
    sample_lower: np.ndarray,
    sample_upper: np.ndarray,
    output_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    native_spacing = (domain_upper - domain_lower) / domain_counts
    output_counts = np.asarray(output_shape, dtype=np.int64)
    output_spacing = (sample_upper - sample_lower) / output_counts
    owners = []
    for axis in range(3):
        axis_owners = np.empty(output_shape[axis], dtype=np.int64)
        for output_index in range(output_shape[axis]):
            center = output_center(
                np.float64(sample_lower[axis]),
                np.float64(output_spacing[axis]),
                output_index,
            )
            axis_owners[output_index] = source_cell_index(
                center,
                np.float64(domain_lower[axis]),
                np.float64(domain_upper[axis]),
                np.float64(native_spacing[axis]),
                int(domain_counts[axis]),
            )
        owners.append(axis_owners)
    return tuple(owners)


def sample_reference(
    payload: np.ndarray,
    payload_valid_lower: np.ndarray,
    block_ids: np.ndarray,
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    domain_counts: np.ndarray,
    block_counts: np.ndarray,
    coord_to_rank: np.ndarray,
    sample_lower: np.ndarray,
    sample_upper: np.ndarray,
    uniform_grid: np.ndarray,
) -> None:
    owners = owner_indices(
        domain_lower,
        domain_upper,
        domain_counts,
        sample_lower,
        sample_upper,
        uniform_grid.shape[1:],
    )
    for slot, block_id in enumerate(block_ids):
        for output_index in np.ndindex(uniform_grid.shape[1:]):
            global_index = tuple(
                int(owners[axis][output_index[axis]]) for axis in range(3)
            )
            block_coordinate = tuple(
                global_index[axis] // int(block_counts[axis])
                for axis in range(3)
            )
            if int(coord_to_rank[block_coordinate]) != int(block_id):
                continue
            local_index = tuple(
                global_index[axis] % int(block_counts[axis])
                for axis in range(3)
            )
            source = tuple(
                int(payload_valid_lower[axis]) + local_index[axis]
                for axis in range(3)
            )
            uniform_grid[(slice(None), *output_index)] = payload[
                (slot, slice(None), *source)
            ]


def patterned_blocks(
    block_count: int,
    field_count: int,
    block_shape: tuple[int, int, int],
) -> np.ndarray:
    x, y, z = np.indices(block_shape, dtype=np.float64)
    blocks = np.empty((block_count, field_count, *block_shape), dtype=np.float64)
    for block in range(block_count):
        for field in range(field_count):
            blocks[block, field] = (
                100000.0 * block
                + 10000.0 * field
                + 100.0 * x
                + 10.0 * y
                + z
            )
    return blocks


def current_zero_order(
    payload: np.ndarray,
    root_shape: tuple[int, int, int],
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    sample_lower: np.ndarray,
    sample_upper: np.ndarray,
    output_shape: tuple[int, int, int],
    *,
    fill_value: float = -999.0,
) -> np.ndarray:
    block_counts = np.asarray(payload.shape[2:], dtype=np.uint32)
    domain_counts = np.asarray(root_shape, dtype=np.uint32) * block_counts
    forest = AMRForest(
        3,
        *root_shape,
        np.ones(int(np.prod(root_shape)), dtype=np.int32),
    )
    mesh = AMRMesh(
        3,
        block_counts,
        domain_counts,
        domain_lower,
        domain_upper,
        np.uint32(0),
        np.uint32(payload.shape[1]),
        forest,
    )
    output = np.full(
        (payload.shape[1], *output_shape),
        fill_value,
        dtype=np.float64,
    )
    mesh.uniform_grid_zero_order(
        payload,
        output,
        np.asarray(output_shape, dtype=np.uint32),
        sample_lower,
        sample_upper,
    )
    return output


def x_cell_blocks(
    root_x: int,
    cells_per_block: int,
    *,
    y_cells: int = 2,
    z_cells: int = 2,
) -> np.ndarray:
    payload = np.empty(
        (root_x, 1, cells_per_block, y_cells, z_cells),
        dtype=np.float64,
    )
    for block in range(root_x):
        for local in range(cells_per_block):
            payload[block, 0, local] = block * cells_per_block + local
    return payload


def test_exact_native_cell_and_block_face_ties_choose_upper_owner() -> None:
    root_shape = (2, 1, 1)
    block_counts = i3(2, 2, 2)
    domain_counts = i3(4, 2, 2)
    domain_lower = f3(0.0, 0.0, 0.0)
    domain_upper = f3(4.0, 2.0, 2.0)
    coord_to_rank, rank_to_coord = maps(root_shape)
    payload = x_cell_blocks(2, 2)
    ids = np.arange(2, dtype=np.int64)

    cases = [
        (f3(0.5, 0.0, 0.0), f3(1.5, 1.0, 1.0), 1.0),
        (f3(1.5, 0.0, 0.0), f3(2.5, 1.0, 1.0), 2.0),
    ]
    for sample_lower, sample_upper, expected_value in cases:
        output = np.full((1, 1, 1, 1), -7.0)
        expected = output.copy()
        sample_reference(
            payload,
            i3(0, 0, 0),
            ids,
            domain_lower,
            domain_upper,
            domain_counts,
            block_counts,
            coord_to_rank,
            sample_lower,
            sample_upper,
            expected,
        )
        sample_level1_zero_order(
            payload,
            i3(0, 0, 0),
            block_counts,
            ids,
            domain_lower,
            domain_upper,
            domain_counts,
            block_counts,
            coord_to_rank,
            rank_to_coord,
            sample_lower,
            sample_upper,
            output,
        )
        assert output[0, 0, 0, 0] == expected_value
        assert_bits_equal(output, expected)

    current = current_zero_order(
        payload,
        root_shape,
        domain_lower,
        domain_upper,
        f3(1.5, 0.0, 0.0),
        f3(2.5, 1.0, 1.0),
        (1, 1, 1),
    )
    assert current[0, 0, 0, 0] == 2.0

    center = 2.0
    output_spacing = 1.0
    lower_stop = math.floor((2.0 - 1.5) / output_spacing + 0.5)
    upper_start = math.ceil((2.0 - 1.5) / output_spacing - 0.5)
    lower_local_index = math.floor((center - 0.0) / 1.0)
    assert lower_stop == 1 and upper_start == 0
    assert lower_local_index == block_counts[0]


@pytest.mark.parametrize(
    ("sample_lower_x", "sample_upper_x", "output_x", "expected_owners"),
    [
        (0.0, 4.0, 4, [0, 1, 2, 3]),
        (0.0, 4.0, 3, [0, 2, 3]),
        (0.0, 4.0, 7, [0, 0, 1, 2, 2, 3, 3]),
        (0.75, 3.25, 5, [1, 1, 2, 2, 3]),
    ],
)
def test_native_coarse_fine_and_subdomain_owner_sequences_match_current(
    sample_lower_x: float,
    sample_upper_x: float,
    output_x: int,
    expected_owners: list[int],
) -> None:
    root_shape = (2, 1, 1)
    block_counts = i3(2, 2, 2)
    domain_counts = i3(4, 2, 2)
    domain_lower = f3(0.0, 0.0, 0.0)
    domain_upper = f3(4.0, 2.0, 2.0)
    sample_lower = f3(sample_lower_x, 0.0, 0.0)
    sample_upper = f3(sample_upper_x, 1.0, 1.0)
    coord_to_rank, rank_to_coord = maps(root_shape)
    payload = x_cell_blocks(2, 2)
    ids = np.arange(2, dtype=np.int64)
    output = np.full((1, output_x, 1, 1), -13.0)
    sample_level1_zero_order(
        payload,
        i3(0, 0, 0),
        block_counts,
        ids,
        domain_lower,
        domain_upper,
        domain_counts,
        block_counts,
        coord_to_rank,
        rank_to_coord,
        sample_lower,
        sample_upper,
        output,
    )
    assert output[0, :, 0, 0].tolist() == expected_owners

    current = current_zero_order(
        payload,
        root_shape,
        domain_lower,
        domain_upper,
        sample_lower,
        sample_upper,
        (output_x, 1, 1),
    )
    assert_bits_equal(output, current)


def test_safe_large_origin_matches_reference_and_current() -> None:
    origin = np.float64(1.0e16)
    root_shape = (2, 1, 1)
    block_counts = i3(4, 2, 2)
    domain_counts = i3(8, 2, 2)
    domain_lower = f3(origin, 0.0, 0.0)
    domain_upper = f3(origin + 32.0, 2.0, 2.0)
    sample_lower = f3(origin + 4.0, 0.0, 0.0)
    sample_upper = f3(origin + 28.0, 1.0, 1.0)
    coord_to_rank, rank_to_coord = maps(root_shape)
    payload = x_cell_blocks(2, 4)
    output = np.full((1, 3, 1, 1), -5.0)
    expected = output.copy()
    ids = np.arange(2, dtype=np.int64)
    sample_reference(
        payload,
        i3(0, 0, 0),
        ids,
        domain_lower,
        domain_upper,
        domain_counts,
        block_counts,
        coord_to_rank,
        sample_lower,
        sample_upper,
        expected,
    )
    sample_level1_zero_order(
        payload,
        i3(0, 0, 0),
        block_counts,
        ids,
        domain_lower,
        domain_upper,
        domain_counts,
        block_counts,
        coord_to_rank,
        rank_to_coord,
        sample_lower,
        sample_upper,
        output,
    )
    assert output[0, :, 0, 0].tolist() == [2.0, 4.0, 6.0]
    assert_bits_equal(output, expected)
    current = current_zero_order(
        payload,
        root_shape,
        domain_lower,
        domain_upper,
        sample_lower,
        sample_upper,
        (3, 1, 1),
    )
    assert_bits_equal(output, current)


def test_current_collapsed_and_exterior_behavior_are_explicit_divergences() -> None:
    origin = np.float64(1.0e16)
    root_shape = (2, 1, 1)
    block_counts = i3(4, 2, 2)
    domain_counts = i3(8, 2, 2)
    coord_to_rank, rank_to_coord = maps(root_shape)
    payload = x_cell_blocks(2, 4)
    collapsed_lower = f3(origin, 0.0, 0.0)
    collapsed_upper = f3(origin + 8.0, 2.0, 2.0)
    collapsed_output = np.full((1, 8, 1, 1), -23.0)
    collapsed_before = collapsed_output.copy()
    with pytest.raises(ValueError, match="center|representable"):
        sample_level1_zero_order(
            payload,
            i3(0, 0, 0),
            block_counts,
            np.arange(2, dtype=np.int64),
            collapsed_lower,
            collapsed_upper,
            domain_counts,
            block_counts,
            coord_to_rank,
            rank_to_coord,
            collapsed_lower,
            collapsed_upper,
            collapsed_output,
        )
    assert_bits_equal(collapsed_output, collapsed_before)
    collapsed_spacing = np.float64(1.0)
    assert output_center(origin, collapsed_spacing, 0) == origin
    assert output_center(origin, collapsed_spacing, 7) == collapsed_upper[0]

    normal_lower = f3(0.0, 0.0, 0.0)
    normal_upper = f3(4.0, 2.0, 2.0)
    outside_lower = f3(-1.0, 0.0, 0.0)
    outside_upper = f3(5.0, 1.0, 1.0)
    rejected = np.full((1, 6, 1, 1), -999.0)
    rejected_before = rejected.copy()
    with pytest.raises(ValueError, match="contained"):
        sample_level1_zero_order(
            x_cell_blocks(2, 2),
            i3(0, 0, 0),
            i3(2, 2, 2),
            np.arange(2, dtype=np.int64),
            normal_lower,
            normal_upper,
            i3(4, 2, 2),
            i3(2, 2, 2),
            coord_to_rank,
            rank_to_coord,
            outside_lower,
            outside_upper,
            rejected,
        )
    assert_bits_equal(rejected, rejected_before)
    current = current_zero_order(
        x_cell_blocks(2, 2),
        root_shape,
        normal_lower,
        normal_upper,
        outside_lower,
        outside_upper,
        (6, 1, 1),
    )
    assert current[0, :, 0, 0].tolist() == [-999.0, 0.0, 1.0, 2.0, 3.0, -999.0]


def test_geo_valid_native_request_can_differ_from_exact_placement() -> None:
    domain_x0 = np.float64((1 << 53) + 2)
    domain_x1 = np.float64(domain_x0 + 4.0)
    domain_lower = f3(domain_x0, 0.0, 0.0)
    domain_upper = f3(domain_x1, 1.0, 1.0)
    domain_counts = i3(2, 1, 1)
    block_counts = i3(1, 1, 1)
    coord_to_rank, rank_to_coord = maps((2, 1, 1))
    ids = np.array([0, 1], dtype=np.int64)
    bounds, spacing = level1_block_geometry(
        domain_lower,
        domain_upper,
        domain_counts,
        block_counts,
        coord_to_rank,
        rank_to_coord,
        ids,
    )
    assert bounds[0, 1, 0] == bounds[1, 0, 0]
    face_one = domain_x0 + spacing[0]
    center_zero = domain_x0 + np.float64(0.5) * spacing[0]
    center_one = domain_x0 + np.float64(1.5) * spacing[0]
    assert center_zero == face_one == center_one

    payload = np.array([10.0, 20.0]).reshape(2, 1, 1, 1, 1)
    sampled = np.zeros((1, 2, 1, 1), dtype=np.float64)
    sample_level1_zero_order(
        payload,
        i3(0, 0, 0),
        block_counts,
        ids,
        domain_lower,
        domain_upper,
        domain_counts,
        block_counts,
        coord_to_rank,
        rank_to_coord,
        domain_lower,
        domain_upper,
        sampled,
    )
    placed = np.zeros_like(sampled)
    place_level1_blocks(
        payload,
        i3(0, 0, 0),
        block_counts,
        ids,
        domain_counts,
        block_counts,
        coord_to_rank,
        rank_to_coord,
        placed,
    )
    assert sampled[0, :, 0, 0].tolist() == [20.0, 20.0]
    assert placed[0, :, 0, 0].tolist() == [10.0, 20.0]


def test_padded_reordered_repeated_partial_and_empty_preserve_bits() -> None:
    root_shape = (3, 2, 1)
    block_counts = i3(2, 2, 2)
    domain_counts = i3(6, 4, 2)
    domain_lower = f3(-3.0, 2.0, 10.0)
    domain_upper = f3(3.0, 6.0, 12.0)
    sample_lower = f3(-2.5, 2.25, 10.0)
    sample_upper = f3(2.5, 5.75, 12.0)
    coord_to_rank, rank_to_coord = maps(root_shape)
    block_ids = np.array([4, 0, 4], dtype=np.int64)
    lower = i3(1, 2, 1)
    upper = lower + block_counts
    spatial_shape = tuple(int(value) for value in upper + i3(2, 1, 2))
    halo_nan = np.asarray([0x7FF8000000001234], dtype=np.uint64).view(np.float64)[0]
    payload = np.full((3, 1, *spatial_shape), halo_nan, dtype=np.float64)
    source_region = tuple(
        slice(int(start), int(stop))
        for start, stop in zip(lower, upper, strict=True)
    )
    for slot in range(3):
        payload[(slot, 0, *source_region)] = 100.0 * slot + np.arange(8).reshape(2, 2, 2)
    payload_before = payload.copy()
    untouched = np.asarray([0xFFF8000000005678], dtype=np.uint64).view(np.float64)[0]
    output = np.full((1, 9, 5, 3), untouched, dtype=np.float64)
    expected = output.copy()
    sample_reference(
        payload,
        lower,
        block_ids,
        domain_lower,
        domain_upper,
        domain_counts,
        block_counts,
        coord_to_rank,
        sample_lower,
        sample_upper,
        expected,
    )
    sample_level1_zero_order(
        payload,
        lower,
        upper,
        block_ids,
        domain_lower,
        domain_upper,
        domain_counts,
        block_counts,
        coord_to_rank,
        rank_to_coord,
        sample_lower,
        sample_upper,
        output,
    )
    assert_bits_equal(output, expected)
    assert_bits_equal(payload, payload_before)
    assert np.any(output.view(np.uint64) == np.uint64(0xFFF8000000005678))

    empty = np.full_like(output, untouched)
    empty_before = empty.copy()
    sample_level1_zero_order(
        np.empty((0, 1, *spatial_shape), dtype=np.float64),
        lower,
        upper,
        np.empty(0, dtype=np.int64),
        domain_lower,
        domain_upper,
        domain_counts,
        block_counts,
        coord_to_rank,
        rank_to_coord,
        sample_lower,
        sample_upper,
        empty,
    )
    assert_bits_equal(empty, empty_before)


def test_invalid_bounds_spacing_map_region_output_and_overlap_are_atomic() -> None:
    root_shape = (2, 1, 1)
    block_counts = i3(2, 2, 2)
    domain_counts = i3(4, 2, 2)
    domain_lower = f3(0.0, 0.0, 0.0)
    domain_upper = f3(4.0, 2.0, 2.0)
    coord_to_rank, rank_to_coord = maps(root_shape)
    payload = x_cell_blocks(2, 2)
    ids = np.array([0, 1], dtype=np.int64)
    output = np.full((1, 4, 2, 2), -41.0)
    before = output.copy()

    with pytest.raises(ValueError, match="contained"):
        sample_level1_zero_order(
            payload,
            i3(0, 0, 0),
            block_counts,
            ids,
            domain_lower,
            domain_upper,
            domain_counts,
            block_counts,
            coord_to_rank,
            rank_to_coord,
            f3(-1.0, 0.0, 0.0),
            domain_upper,
            output,
        )
    assert_bits_equal(output, before)

    tiny = np.finfo(np.float64).tiny / np.float64(2.0)
    tiny_output = np.full((1, 1, 1, 1), -3.0)
    tiny_before = tiny_output.copy()
    with pytest.raises(ValueError, match="normal"):
        sample_level1_zero_order(
            payload,
            i3(0, 0, 0),
            block_counts,
            ids,
            domain_lower,
            domain_upper,
            domain_counts,
            block_counts,
            coord_to_rank,
            rank_to_coord,
            f3(0.0, 0.0, 0.0),
            f3(tiny, 1.0, 1.0),
            tiny_output,
        )
    assert_bits_equal(tiny_output, tiny_before)

    broken_forward = coord_to_rank.copy()
    coordinate = tuple(int(value) for value in rank_to_coord[1])
    broken_forward[coordinate] = 0
    with pytest.raises(ValueError, match="slot 1"):
        sample_level1_zero_order(
            payload,
            i3(0, 0, 0),
            block_counts,
            ids,
            domain_lower,
            domain_upper,
            domain_counts,
            block_counts,
            broken_forward,
            rank_to_coord,
            domain_lower,
            domain_upper,
            output,
        )
    assert_bits_equal(output, before)

    with pytest.raises(ValueError, match="extent"):
        sample_level1_zero_order(
            payload,
            i3(0, 0, 0),
            i3(1, 2, 2),
            ids,
            domain_lower,
            domain_upper,
            domain_counts,
            block_counts,
            coord_to_rank,
            rank_to_coord,
            domain_lower,
            domain_upper,
            output,
        )
    assert_bits_equal(output, before)

    wrong_fields = np.full((2, 4, 2, 2), -7.0)
    wrong_before = wrong_fields.copy()
    with pytest.raises(ValueError, match="field counts"):
        sample_level1_zero_order(
            payload,
            i3(0, 0, 0),
            block_counts,
            ids,
            domain_lower,
            domain_upper,
            domain_counts,
            block_counts,
            coord_to_rank,
            rank_to_coord,
            domain_lower,
            domain_upper,
            wrong_fields,
        )
    assert_bits_equal(wrong_fields, wrong_before)

    readonly = output.copy()
    readonly.setflags(write=False)
    with pytest.raises(ValueError, match="writable"):
        sample_level1_zero_order(
            payload,
            i3(0, 0, 0),
            block_counts,
            ids,
            domain_lower,
            domain_upper,
            domain_counts,
            block_counts,
            coord_to_rank,
            rank_to_coord,
            domain_lower,
            domain_upper,
            readonly,
        )

    overlap_payload = x_cell_blocks(1, 2)
    overlap_output = overlap_payload.reshape(1, 2, 2, 2)
    overlap_before = overlap_payload.copy()
    one_forward, one_inverse = maps((1, 1, 1))
    with pytest.raises(ValueError, match="overlap"):
        sample_level1_zero_order(
            overlap_payload,
            i3(0, 0, 0),
            block_counts,
            np.array([0], dtype=np.int64),
            domain_lower,
            f3(2.0, 2.0, 2.0),
            block_counts,
            block_counts,
            one_forward,
            one_inverse,
            domain_lower,
            f3(2.0, 2.0, 2.0),
            overlap_output,
        )
    assert_bits_equal(overlap_payload, overlap_before)


def test_bounded_no_closure_gather_sample_matches_full_reference() -> None:
    root_shape = (5, 3, 2)
    block_counts = i3(2, 3, 4)
    domain_counts = i3(10, 9, 8)
    domain_lower = f3(-2.0, 1.0, 10.0)
    domain_upper = f3(8.0, 10.0, 18.0)
    sample_lower = f3(-1.0, 2.0, 11.0)
    sample_upper = f3(7.0, 9.0, 17.0)
    coord_to_rank, rank_to_coord = maps(root_shape)
    faces = level1_face_neighbors(
        i3(*root_shape),
        coord_to_rank,
        rank_to_coord,
    )
    backing = patterned_blocks(
        rank_to_coord.shape[0],
        3,
        tuple(int(value) for value in block_counts),
    )
    field_ids = np.array([2, 0], dtype=np.int64)
    capacity = 7
    chunk_ids = np.empty(capacity, dtype=np.int64)
    workspace = np.empty(
        (capacity, len(field_ids), *(int(value) for value in block_counts)),
        dtype=np.float64,
    )
    workspace_identity = id(workspace)
    output = np.full((len(field_ids), 13, 7, 9), np.nan)
    first = 0
    covered = []
    while first < faces.shape[0]:
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
            block_counts,
            chunk_ids[:selected_count],
            field_ids,
            workspace[:selected_count],
            i3(0, 0, 0),
        )
        sample_level1_zero_order(
            workspace[:primary_count],
            i3(0, 0, 0),
            block_counts,
            chunk_ids[:primary_count],
            domain_lower,
            domain_upper,
            domain_counts,
            block_counts,
            coord_to_rank,
            rank_to_coord,
            sample_lower,
            sample_upper,
            output,
        )
        covered.extend(chunk_ids[:primary_count].tolist())
        first += primary_count
        assert id(workspace) == workspace_identity

    expected = np.full_like(output, np.nan)
    sample_reference(
        backing[:, field_ids],
        i3(0, 0, 0),
        np.arange(rank_to_coord.shape[0], dtype=np.int64),
        domain_lower,
        domain_upper,
        domain_counts,
        block_counts,
        coord_to_rank,
        sample_lower,
        sample_upper,
        expected,
    )
    assert covered == list(range(rank_to_coord.shape[0]))
    assert_bits_equal(output, expected)
