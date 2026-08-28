from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.lib.format import open_memmap

from simesh.utils.lib.amr.forest import AMRForest
from simesh.utils.lib.amr.mesh import AMRMesh
from simesh_rewrite.blockio import make_block_reader, make_block_writer
from simesh_rewrite.halos import fill_physical_halos, fill_same_level_halos
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.operators import central_difference_into, scaled_difference_into
from simesh_rewrite.pipeline import execute_level1_m0, execute_level1_m0_from_blocks
from simesh_rewrite.storage import gather_blocks_into, scatter_blocks_from
from simesh_rewrite.reductions import accumulate_field_sum, finalize_field_sum
from simesh_rewrite.sampling import (
    place_level1_blocks,
    sample_level1_trilinear,
    sample_level1_zero_order,
)
from simesh_rewrite.topology import level1_face_neighbors


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def f3(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def assert_bits_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def managed_bytes_per_slot(field_count: int, block_counts: np.ndarray) -> int:
    padded_volume = int(np.prod(block_counts + 2, dtype=np.int64))
    block_volume = int(np.prod(block_counts, dtype=np.int64))
    return 8 * (field_count * padded_volume + block_volume + 1)


def make_outputs(
    block_count: int,
    field_count: int,
    block_counts: np.ndarray,
    domain_counts: np.ndarray,
    zero_shape: tuple[int, int, int],
    trilinear_shape: tuple[int, int, int],
    sentinel_bits: int = 0x7FF8000000004321,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sentinel = np.asarray([sentinel_bits], dtype=np.uint64).view(np.float64)[0]
    pointwise = np.full(
        (block_count, 1, *(int(value) for value in block_counts)), sentinel
    )
    stencil = np.full_like(pointwise, sentinel)
    native = np.full(
        (field_count, *(int(value) for value in domain_counts)), sentinel
    )
    zero = np.full((field_count, *zero_shape), sentinel)
    trilinear = np.full((field_count, *trilinear_shape), sentinel)
    return pointwise, stencil, native, zero, trilinear


def execute(
    backing: np.ndarray,
    field_ids: np.ndarray,
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    domain_counts: np.ndarray,
    block_counts: np.ndarray,
    coord_to_rank: np.ndarray,
    rank_to_coord: np.ndarray,
    neighbors: np.ndarray,
    modes: np.ndarray,
    normals: np.ndarray,
    sample_lower: np.ndarray,
    sample_upper: np.ndarray,
    budget: int,
    outputs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[float, int]:
    return execute_level1_m0(
        backing,
        field_ids,
        domain_lower,
        domain_upper,
        domain_counts,
        block_counts,
        coord_to_rank,
        rank_to_coord,
        neighbors,
        modes,
        normals,
        0,
        1,
        0.5,
        2,
        0,
        1,
        sample_lower,
        sample_upper,
        budget,
        *outputs,
    )


def full_reference(
    backing: np.ndarray,
    field_ids: np.ndarray,
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    domain_counts: np.ndarray,
    block_counts: np.ndarray,
    coord_to_rank: np.ndarray,
    rank_to_coord: np.ndarray,
    neighbors: np.ndarray,
    modes: np.ndarray,
    normals: np.ndarray,
    sample_lower: np.ndarray,
    sample_upper: np.ndarray,
    zero_shape: tuple[int, int, int],
    trilinear_shape: tuple[int, int, int],
) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    selected = np.ascontiguousarray(np.take(backing, field_ids, axis=1))
    block_count, field_count = selected.shape[:2]
    block_ids = np.arange(block_count, dtype=np.int64)
    padded_shape = tuple(int(value) for value in block_counts + 2)
    payload = np.full((block_count, field_count, *padded_shape), np.nan)
    interior_lower = i3(1, 1, 1)
    interior_upper = interior_lower + block_counts
    interior = tuple(
        slice(int(lower), int(upper))
        for lower, upper in zip(interior_lower, interior_upper, strict=True)
    )
    payload[(slice(None), slice(None), *interior)] = selected

    outputs = make_outputs(
        block_count,
        field_count,
        block_counts,
        domain_counts,
        zero_shape,
        trilinear_shape,
        sentinel_bits=0x7FF8000000009999,
    )
    pointwise, stencil, native, zero, trilinear = outputs
    scaled_difference_into(
        payload,
        interior_lower,
        interior_upper,
        0,
        1,
        0.5,
        pointwise,
        0,
        i3(0, 0, 0),
    )
    place_level1_blocks(
        payload,
        interior_lower,
        interior_upper,
        block_ids,
        domain_counts,
        block_counts,
        coord_to_rank,
        rank_to_coord,
        native,
    )
    sample_level1_zero_order(
        payload,
        interior_lower,
        interior_upper,
        block_ids,
        domain_lower,
        domain_upper,
        domain_counts,
        block_counts,
        coord_to_rank,
        rank_to_coord,
        sample_lower,
        sample_upper,
        zero,
    )
    state = np.array([0.0])
    accumulate_field_sum(
        payload,
        interior_lower,
        interior_upper,
        1,
        state,
    )

    fill_physical_halos(
        payload,
        interior_lower,
        interior_upper,
        block_ids,
        neighbors,
        modes,
        normals,
    )
    fill_same_level_halos(
        payload,
        interior_lower,
        interior_upper,
        block_ids,
        block_count,
        neighbors,
        modes,
        normals,
    )
    spacing = np.ascontiguousarray(
        (domain_upper - domain_lower) / domain_counts,
        dtype=np.float64,
    )
    central_difference_into(
        payload,
        i3(0, 0, 0),
        i3(*padded_shape),
        interior_lower,
        interior_upper,
        2,
        0,
        spacing,
        stencil,
        0,
        i3(0, 0, 0),
    )
    sample_level1_trilinear(
        payload,
        i3(0, 0, 0),
        i3(*padded_shape),
        interior_lower,
        interior_upper,
        block_ids,
        domain_lower,
        domain_upper,
        domain_counts,
        block_counts,
        coord_to_rank,
        rank_to_coord,
        sample_lower,
        sample_upper,
        trilinear,
    )
    return finalize_field_sum(state), outputs


def create_memmap_case(tmp_path: Path):
    root_shape = i3(8, 4, 4)
    block_counts = i3(2, 2, 2)
    domain_counts = root_shape * block_counts
    block_count = int(np.prod(root_shape))
    backing_fields = 8
    path = tmp_path / "m0_source.npy"
    writable = open_memmap(
        path,
        mode="w+",
        dtype=np.float64,
        shape=(block_count, backing_fields, 2, 2, 2),
    )
    x, y, z = np.indices((2, 2, 2), dtype=np.float64)
    for block in range(block_count):
        for field in range(backing_fields):
            writable[block, field] = (
                1000.0 * field + 10.0 * block + 4.0 * x + 2.0 * y + z
            )
    adversarial = np.resize(
        np.asarray([1.0e16, 1.0, -1.0e16, 1.0]),
        block_count * 8,
    )
    writable[:, 2] = adversarial.reshape(block_count, 2, 2, 2)
    writable.flush()
    del writable
    backing = np.load(path, mmap_mode="r")
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    neighbors = level1_face_neighbors(root_shape, coord_to_rank, rank_to_coord)
    field_ids = np.array([5, 2, 5], dtype=np.int64)
    modes = np.zeros((3, 6), dtype=np.uint8)
    modes[2, 0] = 2
    return {
        "backing": backing,
        "field_ids": field_ids,
        "domain_lower": f3(0.0, 0.0, 0.0),
        "domain_upper": f3(1.0, 1.0, 1.0),
        "domain_counts": domain_counts,
        "block_counts": block_counts,
        "coord_to_rank": coord_to_rank,
        "rank_to_coord": rank_to_coord,
        "neighbors": neighbors,
        "modes": modes,
        "normals": i3(-1, -1, -1),
        "sample_lower": f3(0.0, 0.17, 0.19),
        "sample_upper": f3(0.3, 0.83, 0.81),
        "zero_shape": (7, 6, 5),
        "trilinear_shape": (6, 5, 4),
    }


class CountingBlockStorage:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.calls = 0


def counting_read_blocks(
    state: CountingBlockStorage,
    source_lower: np.ndarray,
    source_upper: np.ndarray,
    block_ids: np.ndarray,
    field_ids: np.ndarray,
    destination: np.ndarray,
    destination_lower: np.ndarray,
) -> None:
    state.calls += 1
    gather_blocks_into(
        state.data,
        source_lower,
        source_upper,
        block_ids,
        field_ids,
        destination,
        destination_lower,
    )


def counting_write_blocks(
    state: CountingBlockStorage,
    source: np.ndarray,
    source_lower: np.ndarray,
    source_upper: np.ndarray,
    block_ids: np.ndarray,
    field_ids: np.ndarray,
    destination_lower: np.ndarray,
) -> None:
    state.calls += 1
    scatter_blocks_from(
        source,
        source_lower,
        source_upper,
        block_ids,
        field_ids,
        state.data,
        destination_lower,
    )


def execute_from_functional_adapters(
    case,
    budget: int,
    outputs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
):
    source_state = CountingBlockStorage(case["backing"])
    pointwise_state = CountingBlockStorage(outputs[0])
    stencil_state = CountingBlockStorage(outputs[1])
    reader = make_block_reader(
        source_state,
        source_state.data.shape,
        counting_read_blocks,
        memory_arrays=(source_state.data,),
    )
    pointwise_writer = make_block_writer(
        pointwise_state,
        pointwise_state.data.shape,
        counting_write_blocks,
        memory_arrays=(pointwise_state.data,),
    )
    stencil_writer = make_block_writer(
        stencil_state,
        stencil_state.data.shape,
        counting_write_blocks,
        memory_arrays=(stencil_state.data,),
    )
    result = execute_level1_m0_from_blocks(
        reader,
        case["field_ids"],
        case["domain_lower"],
        case["domain_upper"],
        case["domain_counts"],
        case["block_counts"],
        case["coord_to_rank"],
        case["rank_to_coord"],
        case["neighbors"],
        case["modes"],
        case["normals"],
        0,
        1,
        0.5,
        2,
        0,
        1,
        case["sample_lower"],
        case["sample_upper"],
        budget,
        pointwise_writer,
        stencil_writer,
        outputs[2],
        outputs[3],
        outputs[4],
    )
    return result, source_state.calls, pointwise_state.calls, stencil_state.calls


def test_functional_backends_and_resident_strategy_match_bounded_array_path(
    tmp_path: Path,
) -> None:
    case = create_memmap_case(tmp_path)
    backing = case["backing"]
    field_count = len(case["field_ids"])
    per_slot = managed_bytes_per_slot(field_count, case["block_counts"])

    bounded_outputs = make_outputs(
        backing.shape[0],
        field_count,
        case["block_counts"],
        case["domain_counts"],
        case["zero_shape"],
        case["trilinear_shape"],
    )
    bounded_result = execute(
        backing,
        case["field_ids"],
        case["domain_lower"],
        case["domain_upper"],
        case["domain_counts"],
        case["block_counts"],
        case["coord_to_rank"],
        case["rank_to_coord"],
        case["neighbors"],
        case["modes"],
        case["normals"],
        case["sample_lower"],
        case["sample_upper"],
        8 + 27 * per_slot,
        bounded_outputs,
    )

    resident_outputs = make_outputs(
        backing.shape[0],
        field_count,
        case["block_counts"],
        case["domain_counts"],
        case["zero_shape"],
        case["trilinear_shape"],
    )
    (resident_result, read_calls, point_calls, stencil_calls) = (
        execute_from_functional_adapters(
            case,
            8 + backing.shape[0] * per_slot,
            resident_outputs,
        )
    )
    assert bounded_result[0].hex() == resident_result[0].hex()
    assert bounded_result[1] == 27
    assert resident_result[1] == backing.shape[0]
    for actual, expected in zip(resident_outputs, bounded_outputs, strict=True):
        assert_bits_equal(actual, expected)
    # Empty preflight plus one read per resident pass; one empty and one real write.
    assert read_calls == 3
    assert point_calls == 2
    assert stencil_calls == 2


def test_memmap_two_capacities_match_full_reference_and_overwrite_all(tmp_path: Path) -> None:
    case = create_memmap_case(tmp_path)
    backing = case["backing"]
    assert isinstance(backing, np.memmap)
    assert not backing.flags.writeable
    source_edges = (
        backing[0, :, 0, 0, 0].copy(),
        backing[-1, :, -1, -1, -1].copy(),
    )
    reference_sum, reference_outputs = full_reference(
        backing,
        case["field_ids"],
        case["domain_lower"],
        case["domain_upper"],
        case["domain_counts"],
        case["block_counts"],
        case["coord_to_rank"],
        case["rank_to_coord"],
        case["neighbors"],
        case["modes"],
        case["normals"],
        case["sample_lower"],
        case["sample_upper"],
        case["zero_shape"],
        case["trilinear_shape"],
    )
    per_slot = managed_bytes_per_slot(len(case["field_ids"]), case["block_counts"])
    insufficient = make_outputs(
        backing.shape[0],
        len(case["field_ids"]),
        case["block_counts"],
        case["domain_counts"],
        case["zero_shape"],
        case["trilinear_shape"],
    )
    insufficient_before = tuple(value.copy() for value in insufficient)
    with pytest.raises(ValueError, match="minimum full-halo closure"):
        execute(
            backing,
            case["field_ids"],
            case["domain_lower"],
            case["domain_upper"],
            case["domain_counts"],
            case["block_counts"],
            case["coord_to_rank"],
            case["rank_to_coord"],
            case["neighbors"],
            case["modes"],
            case["normals"],
            case["sample_lower"],
            case["sample_upper"],
            8 + 27 * per_slot - 1,
            insufficient,
        )
    assert_outputs_unchanged(insufficient, insufficient_before)

    reductions = []
    tested_outputs = []
    for capacity in (27, 64):
        budget = 8 + capacity * per_slot
        outputs = make_outputs(
            backing.shape[0],
            len(case["field_ids"]),
            case["block_counts"],
            case["domain_counts"],
            case["zero_shape"],
            case["trilinear_shape"],
        )
        reduction, actual_capacity = execute(
            backing,
            case["field_ids"],
            case["domain_lower"],
            case["domain_upper"],
            case["domain_counts"],
            case["block_counts"],
            case["coord_to_rank"],
            case["rank_to_coord"],
            case["neighbors"],
            case["modes"],
            case["normals"],
            case["sample_lower"],
            case["sample_upper"],
            budget,
            outputs,
        )
        assert actual_capacity == capacity
        assert 8 + actual_capacity * per_slot == budget
        for actual, expected in zip(outputs, reference_outputs, strict=True):
            assert_bits_equal(actual, expected)
            assert not np.any(
                actual.view(np.uint64) == np.uint64(0x7FF8000000004321)
            )
        assert_bits_equal(np.asarray([reduction]), np.asarray([reference_sum]))
        reductions.append(reduction)
        tested_outputs.append(outputs)

    assert_bits_equal(np.asarray(reductions[:1]), np.asarray(reductions[1:]))
    assert backing.nbytes > 8 + 27 * per_slot
    assert_bits_equal(backing[0, :, 0, 0, 0], source_edges[0])
    assert_bits_equal(backing[-1, :, -1, -1, -1], source_edges[1])
    assert not np.array_equal(tested_outputs[0][4][0], tested_outputs[0][4][2])


def small_case(root_shape: tuple[int, int, int] = (1, 1, 1)):
    root = i3(*root_shape)
    block_counts = i3(2, 2, 2)
    domain_counts = root * block_counts
    block_count = int(np.prod(root))
    backing = np.empty((block_count, 3, 2, 2, 2), dtype=np.float64)
    x, y, z = np.indices((2, 2, 2), dtype=np.float64)
    for block in range(block_count):
        backing[block, 0] = 10.0 * block + x + y + z
        backing[block, 1] = 20.0 + backing[block, 0]
        backing[block, 2] = 2.0 * x
    coord_to_rank, rank_to_coord = level1_morton(root)
    neighbors = level1_face_neighbors(root, coord_to_rank, rank_to_coord)
    return {
        "backing": backing,
        "field_ids": np.array([1, 0, 2], dtype=np.int64),
        "domain_lower": f3(0.0, 0.0, 0.0),
        "domain_upper": f3(1.0, 1.0, 1.0),
        "domain_counts": domain_counts,
        "block_counts": block_counts,
        "coord_to_rank": coord_to_rank,
        "rank_to_coord": rank_to_coord,
        "neighbors": neighbors,
        "modes": np.zeros((3, 6), dtype=np.uint8),
        "normals": i3(-1, -1, -1),
        "sample_lower": f3(0.1, 0.1, 0.1),
        "sample_upper": f3(0.9, 0.9, 0.9),
        "zero_shape": (4, 4, 4),
        "trilinear_shape": (4, 4, 4),
    }


def test_budget_boundaries_capacity_clamp_types_and_overflow() -> None:
    case = small_case()
    per_slot = managed_bytes_per_slot(3, case["block_counts"])
    outputs = make_outputs(1, 3, case["block_counts"], case["domain_counts"], (4, 4, 4), (4, 4, 4))
    for bad_budget, error in (
        (True, TypeError),
        (1.5, TypeError),
        (-1, ValueError),
        (np.iinfo(np.int64).max + 1, OverflowError),
        (0, ValueError),
        (7, ValueError),
        (8, ValueError),
        (8 + per_slot - 1, ValueError),
    ):
        with pytest.raises(error):
            execute(
                case["backing"], case["field_ids"], case["domain_lower"], case["domain_upper"],
                case["domain_counts"], case["block_counts"], case["coord_to_rank"],
                case["rank_to_coord"], case["neighbors"], case["modes"], case["normals"],
                case["sample_lower"], case["sample_upper"], bad_budget, outputs,
            )
    outputs = make_outputs(1, 3, case["block_counts"], case["domain_counts"], (4, 4, 4), (4, 4, 4))
    _, capacity = execute(
        case["backing"], case["field_ids"], case["domain_lower"], case["domain_upper"],
        case["domain_counts"], case["block_counts"], case["coord_to_rank"],
        case["rank_to_coord"], case["neighbors"], case["modes"], case["normals"],
        case["sample_lower"], case["sample_upper"], 8 + 5 * per_slot, outputs,
    )
    assert capacity == 1


def assert_outputs_unchanged(
    actual: tuple[np.ndarray, ...],
    before: tuple[np.ndarray, ...],
) -> None:
    for value, expected in zip(actual, before, strict=True):
        assert_bits_equal(value, expected)


def test_corrupt_late_map_and_face_rows_are_atomic() -> None:
    case = small_case((4, 4, 4))
    per_slot = managed_bytes_per_slot(3, case["block_counts"])
    budget = 8 + 27 * per_slot
    for corruption in ("map", "face"):
        outputs = make_outputs(64, 3, case["block_counts"], case["domain_counts"], (4, 4, 4), (4, 4, 4))
        before = tuple(value.copy() for value in outputs)
        inverse = case["rank_to_coord"].copy()
        neighbors = case["neighbors"].copy()
        if corruption == "map":
            inverse[-1, 0] = 0
        else:
            neighbors[-1, 0] = -1 if neighbors[-1, 0] >= 0 else 0
        with pytest.raises(ValueError):
            execute(
                case["backing"], case["field_ids"], case["domain_lower"], case["domain_upper"],
                case["domain_counts"], case["block_counts"], case["coord_to_rank"], inverse,
                neighbors, case["modes"], case["normals"], case["sample_lower"],
                case["sample_upper"], budget, outputs,
            )
        assert_outputs_unchanged(outputs, before)


def test_all_result_pairs_and_result_input_overlap_are_rejected() -> None:
    case = small_case()
    per_slot = managed_bytes_per_slot(3, case["block_counts"])
    budget = 8 + per_slot
    result_shapes = [
        (1, 1, 2, 2, 2),
        (1, 1, 2, 2, 2),
        (3, 2, 2, 2),
        (3, 4, 4, 4),
        (3, 4, 4, 4),
    ]
    for first in range(5):
        for second in range(first + 1, 5):
            size = max(int(np.prod(result_shapes[first])), int(np.prod(result_shapes[second])))
            shared = np.full(size, -3.0)
            results = [np.full(shape, -7.0) for shape in result_shapes]
            results[first] = shared[: int(np.prod(result_shapes[first]))].reshape(result_shapes[first])
            results[second] = shared[: int(np.prod(result_shapes[second]))].reshape(result_shapes[second])
            with pytest.raises(ValueError, match="overlap"):
                execute(
                    case["backing"], case["field_ids"], case["domain_lower"], case["domain_upper"],
                    case["domain_counts"], case["block_counts"], case["coord_to_rank"],
                    case["rank_to_coord"], case["neighbors"], case["modes"], case["normals"],
                    case["sample_lower"], case["sample_upper"], budget, tuple(results),
                )

    backing_one = np.ascontiguousarray(case["backing"][:, :1])
    overlap_result = backing_one
    other = make_outputs(1, 1, case["block_counts"], case["domain_counts"], (4, 4, 4), (4, 4, 4))
    with pytest.raises(ValueError, match="overlap"):
        execute_level1_m0(
            backing_one,
            np.array([0], dtype=np.int64),
            case["domain_lower"], case["domain_upper"], case["domain_counts"], case["block_counts"],
            case["coord_to_rank"], case["rank_to_coord"], case["neighbors"],
            np.zeros((1, 6), dtype=np.uint8), case["normals"],
            0, 0, 0.5, 0, 0, 0, case["sample_lower"], case["sample_upper"],
            8 + managed_bytes_per_slot(1, case["block_counts"]),
            overlap_result, other[1],
            np.empty((1, 2, 2, 2)), np.empty((1, 4, 4, 4)), np.empty((1, 4, 4, 4)),
        )

    metadata_base = np.full(24, -5.0)
    metadata_native = metadata_base.reshape(3, 2, 2, 2)
    metadata_lower = metadata_base[:3]
    metadata_lower[:] = 0.0
    metadata_outputs = make_outputs(
        1,
        3,
        case["block_counts"],
        case["domain_counts"],
        (4, 4, 4),
        (4, 4, 4),
    )
    metadata_outputs = (
        metadata_outputs[0],
        metadata_outputs[1],
        metadata_native,
        metadata_outputs[3],
        metadata_outputs[4],
    )
    with pytest.raises(ValueError, match="overlap"):
        execute(
            case["backing"],
            case["field_ids"],
            metadata_lower,
            case["domain_upper"],
            case["domain_counts"],
            case["block_counts"],
            case["coord_to_rank"],
            case["rank_to_coord"],
            case["neighbors"],
            case["modes"],
            case["normals"],
            case["sample_lower"],
            case["sample_upper"],
            budget,
            metadata_outputs,
        )


def test_current_native_zero_and_trilinear_agree_for_safe_commuting_case() -> None:
    case = small_case((2, 2, 2))
    per_slot = managed_bytes_per_slot(3, case["block_counts"])
    outputs = make_outputs(8, 3, case["block_counts"], case["domain_counts"], (4, 4, 4), (4, 4, 4))
    execute(
        case["backing"], case["field_ids"], case["domain_lower"], case["domain_upper"],
        case["domain_counts"], case["block_counts"], case["coord_to_rank"],
        case["rank_to_coord"], case["neighbors"], case["modes"], case["normals"],
        case["sample_lower"], case["sample_upper"], 8 + 8 * per_slot, outputs,
    )
    selected = np.ascontiguousarray(np.take(case["backing"], case["field_ids"], axis=1))
    forest = AMRForest(3, 2, 2, 2, np.ones(8, dtype=np.int32))
    mesh = AMRMesh(
        3, np.array([2, 2, 2], dtype=np.uint32), np.array([4, 4, 4], dtype=np.uint32),
        case["domain_lower"], case["domain_upper"], 1, 3, forest,
        case["modes"].astype(np.int32), case["normals"].astype(np.int32),
    )
    current_native = np.empty_like(outputs[2])
    mesh.uniform_full_level1(selected, current_native)
    mesh.load_interior_data(selected)
    mesh.apply_ghost_cells()
    current_zero = np.empty_like(outputs[3])
    current_tri = np.empty_like(outputs[4])
    nx = np.array([4, 4, 4], dtype=np.uint32)
    mesh.uniform_grid_zero_order(selected, current_zero, nx, case["sample_lower"], case["sample_upper"])
    mesh.uniform_grid_linear(current_tri, nx, case["sample_lower"], case["sample_upper"], np.arange(3, dtype=np.uint32))
    assert_bits_equal(outputs[2], current_native)
    assert_bits_equal(outputs[3], current_zero)
    np.testing.assert_allclose(outputs[4], current_tri, rtol=64 * np.finfo(float).eps, atol=64 * np.finfo(float).eps)
