"""Kernel and bounded composed-path measurements for SAM-003."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc

import numpy as np

from simesh.utils.lib.amr.forest import AMRForest
from simesh.utils.lib.amr.mesh import AMRMesh
from simesh_rewrite.chunking import plan_level1_halo_chunk, workspace_nbytes
from simesh_rewrite.halos import fill_physical_halos, fill_same_level_halos
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.sampling import sample_level1_trilinear
from simesh_rewrite.sampling_reference import sample_level1_trilinear_reference
from simesh_rewrite.storage import gather_blocks_into
from simesh_rewrite.topology import level1_face_neighbors


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def median_seconds(operation, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def maximum_ulp_distance(left: np.ndarray, right: np.ndarray) -> int:
    left_bits = left.view(np.uint64)
    right_bits = right.view(np.uint64)
    sign_mask = np.uint64(1 << 63)
    left_ordered = np.where(
        left_bits & sign_mask,
        ~left_bits,
        left_bits | sign_mask,
    )
    right_ordered = np.where(
        right_bits & sign_mask,
        ~right_bits,
        right_bits | sign_mask,
    )
    distance = np.where(
        left_ordered >= right_ordered,
        left_ordered - right_ordered,
        right_ordered - left_ordered,
    )
    return int(np.max(distance, initial=np.uint64(0)))


def affine_backing(
    root_shape: np.ndarray,
    block_shape: np.ndarray,
    rank_to_coord: np.ndarray,
    field_count: int,
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
) -> np.ndarray:
    domain_counts = root_shape * block_shape
    spacing = (domain_upper - domain_lower) / domain_counts
    result = np.empty(
        (
            rank_to_coord.shape[0],
            field_count,
            *(int(value) for value in block_shape),
        ),
        dtype=np.float64,
    )
    local = np.indices(tuple(int(value) for value in block_shape), dtype=np.float64)
    for block_id, coordinate in enumerate(rank_to_coord):
        global_index = coordinate[:, None, None, None] * block_shape[:, None, None, None] + local
        centers = domain_lower[:, None, None, None] + (
            global_index + 0.5
        ) * spacing[:, None, None, None]
        base = centers[0] + 2.0 * centers[1] + 3.0 * centers[2]
        for field in range(field_count):
            result[block_id, field] = base + 10.0 * field
    return result


def completed_full_payload(
    root_shape: tuple[int, int, int],
    block_shape: tuple[int, int, int],
    field_count: int,
):
    root = i3(*root_shape)
    block = i3(*block_shape)
    domain = root * block
    domain_lower = np.array([-2.0, 1.0, 10.0])
    domain_upper = np.array([6.0, 7.0, 14.0])
    coord_to_rank, rank_to_coord = level1_morton(root)
    faces = level1_face_neighbors(root, coord_to_rank, rank_to_coord)
    backing = affine_backing(
        root,
        block,
        rank_to_coord,
        field_count,
        domain_lower,
        domain_upper,
    )
    lower = i3(1, 1, 1)
    upper = lower + block
    spatial_shape = tuple(int(value) for value in upper + 1)
    payload = np.full(
        (faces.shape[0], field_count, *spatial_shape),
        np.nan,
        dtype=np.float64,
    )
    ids = np.arange(faces.shape[0], dtype=np.int64)
    fields = np.arange(field_count, dtype=np.int64)
    zero = i3(0, 0, 0)
    gather_blocks_into(backing, zero, block, ids, fields, payload, lower)
    modes = np.zeros((field_count, 6), dtype=np.uint8)
    normals = i3(-1, -1, -1)
    fill_physical_halos(payload, lower, upper, ids, faces, modes, normals)
    fill_same_level_halos(
        payload,
        lower,
        upper,
        ids,
        len(ids),
        faces,
        modes,
        normals,
    )
    return (
        root,
        block,
        domain,
        domain_lower,
        domain_upper,
        coord_to_rank,
        rank_to_coord,
        faces,
        backing,
        payload,
        ids,
        lower,
        upper,
        modes,
        normals,
    )


def sampling_case(
    root_shape: tuple[int, int, int],
    block_shape: tuple[int, int, int],
    field_count: int,
    output_shape: tuple[int, int, int],
    sample_lower: np.ndarray,
    sample_upper: np.ndarray,
    repeats: int,
) -> dict:
    (
        root,
        block,
        domain,
        domain_lower,
        domain_upper,
        coord_to_rank,
        rank_to_coord,
        _,
        backing,
        payload,
        ids,
        lower,
        upper,
        modes,
        normals,
    ) = completed_full_payload(root_shape, block_shape, field_count)
    output = np.empty((field_count, *output_shape), dtype=np.float64)
    valid_lower = i3(0, 0, 0)
    valid_upper = np.asarray(payload.shape[2:], dtype=np.int64)

    def sample() -> None:
        sample_level1_trilinear(
            payload,
            valid_lower,
            valid_upper,
            lower,
            upper,
            ids,
            domain_lower,
            domain_upper,
            domain,
            block,
            coord_to_rank,
            rank_to_coord,
            sample_lower,
            sample_upper,
            output,
        )

    sample()
    seconds = median_seconds(sample, repeats)
    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    sample()
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    forest = AMRForest(
        3,
        *root_shape,
        np.ones(rank_to_coord.shape[0], dtype=np.int32),
    )
    mesh = AMRMesh(
        3,
        block.astype(np.uint32),
        domain.astype(np.uint32),
        domain_lower,
        domain_upper,
        np.uint32(1),
        np.uint32(field_count),
        forest,
        modes.astype(np.int32),
        normals.astype(np.int32),
    )
    mesh.load_interior_data(backing)
    mesh.apply_ghost_cells()
    current = np.empty_like(output)
    positions = np.arange(field_count, dtype=np.uint32)

    def current_sample() -> None:
        mesh.uniform_grid_linear(
            current,
            np.asarray(output_shape, dtype=np.uint32),
            sample_lower,
            sample_upper,
            positions,
        )

    current_sample()
    difference = np.abs(output - current)
    scale = np.maximum(np.abs(current), 1.0)
    values = output.size
    return {
        "field_count": field_count,
        "output_shape": output_shape,
        "seconds": seconds,
        "million_samples_per_second": values / seconds / 1e6,
        "current_seconds": median_seconds(current_sample, repeats),
        "current_max_absolute_error": float(np.max(difference)),
        "current_max_relative_error": float(np.max(difference / scale)),
        "current_max_ulp_error": maximum_ulp_distance(output, current),
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
    }


def bounded_composition(
    root_shape: tuple[int, int, int],
    block_shape: tuple[int, int, int],
    field_count: int,
    capacity: int,
    repeats: int,
) -> dict:
    (
        root,
        block,
        domain,
        domain_lower,
        domain_upper,
        coord_to_rank,
        rank_to_coord,
        faces,
        backing,
        completed,
        all_ids,
        lower,
        upper,
        modes,
        normals,
    ) = completed_full_payload(root_shape, block_shape, field_count)
    sample_lower = np.array([-1.5, 1.5, 10.25])
    sample_upper = np.array([5.5, 6.5, 13.75])
    output_shape = (96, 72, 48)
    output = np.empty((field_count, *output_shape), dtype=np.float64)
    expected = np.empty_like(output)
    valid_lower = i3(0, 0, 0)
    spatial_shape = tuple(int(value) for value in upper + 1)
    valid_upper = np.asarray(spatial_shape, dtype=np.int64)
    sample_level1_trilinear(
        completed,
        valid_lower,
        valid_upper,
        lower,
        upper,
        all_ids,
        domain_lower,
        domain_upper,
        domain,
        block,
        coord_to_rank,
        rank_to_coord,
        sample_lower,
        sample_upper,
        expected,
    )

    ids = np.empty(capacity, dtype=np.int64)
    payload = np.empty(
        (capacity, field_count, *spatial_shape),
        dtype=np.float64,
    )
    fields = np.arange(field_count, dtype=np.int64)
    zero = i3(0, 0, 0)

    def traverse() -> int:
        first = 0
        chunks = 0
        while first < faces.shape[0]:
            primary_count, selected_count = plan_level1_halo_chunk(
                first,
                faces,
                ids,
            )
            gather_blocks_into(
                backing,
                zero,
                block,
                ids[:selected_count],
                fields,
                payload[:selected_count],
                lower,
            )
            fill_physical_halos(
                payload[:selected_count],
                lower,
                upper,
                ids[:selected_count],
                faces,
                modes,
                normals,
            )
            fill_same_level_halos(
                payload[:selected_count],
                lower,
                upper,
                ids[:selected_count],
                primary_count,
                faces,
                modes,
                normals,
            )
            sample_level1_trilinear(
                payload[:primary_count],
                valid_lower,
                valid_upper,
                lower,
                upper,
                ids[:primary_count],
                domain_lower,
                domain_upper,
                domain,
                block,
                coord_to_rank,
                rank_to_coord,
                sample_lower,
                sample_upper,
                output,
            )
            first += primary_count
            chunks += 1
        return chunks

    chunks = traverse()
    seconds = median_seconds(traverse, repeats)
    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    traverse()
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    difference = np.abs(output - expected)
    return {
        "capacity": capacity,
        "chunks": chunks,
        "seconds": seconds,
        "million_samples_per_second": output.size / seconds / 1e6,
        "managed_workspace_bytes": workspace_nbytes(
            capacity,
            field_count,
            valid_upper,
        ),
        "max_absolute_error_vs_full": float(np.max(difference)),
        "max_ulp_error_vs_full": maximum_ulp_distance(output, expected),
        "bitwise_equal_to_full": bool(
            np.array_equal(output.view(np.uint64), expected.view(np.uint64))
        ),
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
    }


def reference_comparison(repeats: int) -> dict:
    root_shape = (2, 2, 2)
    block_shape = (4, 4, 4)
    (
        _,
        block,
        domain,
        domain_lower,
        domain_upper,
        coord_to_rank,
        rank_to_coord,
        _,
        _,
        payload,
        ids,
        lower,
        upper,
        _,
        _,
    ) = completed_full_payload(root_shape, block_shape, 2)
    sample_lower = np.array([-1.5, 1.5, 10.25])
    sample_upper = np.array([5.5, 6.5, 13.75])
    expected = np.empty((2, 7, 5, 3), dtype=np.float64)
    actual = np.empty_like(expected)
    started = time.perf_counter()
    sample_level1_trilinear_reference(
        payload,
        lower,
        ids,
        domain_lower,
        domain_upper,
        domain,
        block,
        coord_to_rank,
        rank_to_coord,
        sample_lower,
        sample_upper,
        expected,
    )
    reference_seconds = time.perf_counter() - started
    valid_lower = i3(0, 0, 0)
    valid_upper = np.asarray(payload.shape[2:], dtype=np.int64)

    def compiled() -> None:
        sample_level1_trilinear(
            payload,
            valid_lower,
            valid_upper,
            lower,
            upper,
            ids,
            domain_lower,
            domain_upper,
            domain,
            block,
            coord_to_rank,
            rank_to_coord,
            sample_lower,
            sample_upper,
            actual,
        )

    compiled_seconds = median_seconds(compiled, repeats)
    difference = np.abs(actual - expected)
    return {
        "reference_seconds": reference_seconds,
        "compiled_seconds": compiled_seconds,
        "speedup": reference_seconds / compiled_seconds,
        "max_absolute_error": float(np.max(difference)),
        "max_ulp_error": maximum_ulp_distance(actual, expected),
        "bitwise_equal": bool(
            np.array_equal(actual.view(np.uint64), expected.view(np.uint64))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=11)
    args = parser.parse_args()
    root_shape = (8, 6, 4)
    block_shape = (8, 8, 8)
    domain_lower = np.array([-2.0, 1.0, 10.0])
    domain_upper = np.array([6.0, 7.0, 14.0])
    scenarios = [
        ("coarse", (32, 24, 16), domain_lower, domain_upper),
        ("native", (64, 48, 32), domain_lower, domain_upper),
        ("fine", (96, 72, 48), domain_lower, domain_upper),
        (
            "subdomain",
            (40, 30, 20),
            np.array([-1.5, 1.5, 10.25]),
            np.array([5.5, 6.5, 13.75]),
        ),
    ]
    cases = []
    for field_count in (1, 4):
        for name, output_shape, sample_lower, sample_upper in scenarios:
            result = sampling_case(
                root_shape,
                block_shape,
                field_count,
                output_shape,
                sample_lower,
                sample_upper,
                args.repeats,
            )
            result["scenario"] = name
            cases.append(result)

    print(
        json.dumps(
            {
                "capability": "SAM-003",
                "root_shape": root_shape,
                "block_shape": block_shape,
                "sampling_cases": cases,
                "bounded_composition": [
                    bounded_composition(
                        root_shape,
                        block_shape,
                        4,
                        capacity,
                        args.repeats,
                    )
                    for capacity in (64, 256)
                ],
                "reference": reference_comparison(args.repeats),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
