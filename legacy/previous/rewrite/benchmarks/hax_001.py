"""Explicit-plan versus fused level-1 halo application benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc

import numpy as np

from simesh_rewrite.chunking import plan_level1_halo_chunk
from simesh_rewrite.halo_apply import apply_level1_same_level_halo_plan
from simesh_rewrite.halo_plans import fill_level1_halo_relation_plan
from simesh_rewrite.halos import fill_physical_halos, fill_same_level_halos
from simesh_rewrite.morton import level1_morton
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


def written_cell_count(
    block_ids: np.ndarray,
    primary_count: int,
    faces: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    spatial_shape: tuple[int, int, int],
) -> int:
    count = 0
    for primary in range(primary_count):
        block_id = int(block_ids[primary])
        for target in np.ndindex(spatial_shape):
            for axis in range(3):
                if target[axis] < lower[axis]:
                    face = 2 * axis
                elif target[axis] >= upper[axis]:
                    face = 2 * axis + 1
                else:
                    continue
                if faces[block_id, face] >= 0:
                    count += 1
                    break
    return count


def benchmark_case(
    root_shape: tuple[int, int, int],
    block_shape: tuple[int, int, int],
    halo: int,
    field_count: int,
    capacity: int,
    repeats: int,
) -> dict:
    root = i3(*root_shape)
    block = i3(*block_shape)
    coord_to_rank, rank_to_coord = level1_morton(root)
    faces = level1_face_neighbors(root, coord_to_rank, rank_to_coord)
    first = int(coord_to_rank[tuple(int(value // 2) for value in root)])
    ids = np.empty(capacity, dtype=np.int64)
    primary_count, selected_count = plan_level1_halo_chunk(first, faces, ids)
    block_ids = ids[:selected_count]
    source_slots = np.empty((primary_count, 27), dtype=np.int64)
    masks = np.empty((primary_count, 27), dtype=np.uint8)
    fill_level1_halo_relation_plan(
        block_ids,
        primary_count,
        faces,
        source_slots,
        masks,
    )

    backing = np.empty(
        (faces.shape[0], field_count, *block_shape),
        dtype=np.float64,
    )
    x, y, z = np.indices(block_shape, dtype=np.float64)
    values = 100.0 * x + 10.0 * y + z + 1.0
    for block_id in range(backing.shape[0]):
        for field in range(field_count):
            backing[block_id, field] = (
                100000.0 * block_id + 10000.0 * field + values
            )

    lower = i3(halo, halo, halo)
    upper = lower + block
    spatial_array = upper + halo
    spatial_shape = tuple(int(value) for value in spatial_array)
    payload = np.full(
        (selected_count, field_count, *spatial_shape),
        np.nan,
        dtype=np.float64,
    )
    fields = np.arange(field_count, dtype=np.int64)
    modes = np.empty((field_count, 6), dtype=np.uint8)
    for field in range(field_count):
        for face in range(6):
            modes[field, face] = (field + face) % 3
    normals = i3(-1, -1, -1)
    zero = i3(0, 0, 0)

    def gather() -> None:
        gather_blocks_into(
            backing,
            zero,
            block,
            block_ids,
            fields,
            payload,
            lower,
        )

    def physical() -> None:
        fill_physical_halos(
            payload,
            lower,
            upper,
            block_ids,
            faces,
            modes,
            normals,
        )

    def plan() -> None:
        fill_level1_halo_relation_plan(
            block_ids,
            primary_count,
            faces,
            source_slots,
            masks,
        )

    def explicit_apply() -> None:
        apply_level1_same_level_halo_plan(
            payload,
            lower,
            upper,
            source_slots,
            masks,
            modes,
            normals,
        )

    def fused_apply() -> None:
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

    gather()
    physical()
    base = payload.copy()
    explicit_result = base.copy()
    payload[...] = explicit_result
    explicit_apply()
    explicit_result[...] = payload
    payload[...] = base
    fused_apply()
    exact = np.array_equal(
        explicit_result.view(np.uint64), payload.view(np.uint64)
    )

    payload[...] = base
    explicit_seconds = median_seconds(explicit_apply, repeats)
    payload[...] = base
    fused_seconds = median_seconds(fused_apply, repeats)

    def plan_and_apply() -> None:
        plan()
        explicit_apply()

    payload[...] = base
    plan_apply_seconds = median_seconds(plan_and_apply, repeats)

    def explicit_composed() -> None:
        gather()
        physical()
        plan()
        explicit_apply()

    def fused_composed() -> None:
        gather()
        physical()
        fused_apply()

    explicit_composed_seconds = median_seconds(explicit_composed, repeats)
    fused_composed_seconds = median_seconds(fused_composed, repeats)

    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    explicit_apply()
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    written_values = written_cell_count(
        block_ids,
        primary_count,
        faces,
        lower,
        upper,
        spatial_shape,
    ) * field_count
    return {
        "capacity": capacity,
        "field_count": field_count,
        "halo": halo,
        "primary_count": primary_count,
        "selected_count": selected_count,
        "written_values": written_values,
        "plan_bytes": source_slots.nbytes + masks.nbytes,
        "explicit_apply_seconds": explicit_seconds,
        "explicit_million_values_per_second": (
            written_values / explicit_seconds / 1.0e6
        ),
        "fused_seconds": fused_seconds,
        "plan_apply_seconds": plan_apply_seconds,
        "plan_apply_over_fused": plan_apply_seconds / fused_seconds,
        "explicit_composed_seconds": explicit_composed_seconds,
        "fused_composed_seconds": fused_composed_seconds,
        "explicit_composed_over_fused": (
            explicit_composed_seconds / fused_composed_seconds
        ),
        "exact_fused": bool(exact),
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-shape", default="12,10,8")
    parser.add_argument("--block-shape", default="8,8,8")
    parser.add_argument("--halos", default="1,2,4")
    parser.add_argument("--fields", default="1,4")
    parser.add_argument("--capacities", default="64,256")
    parser.add_argument("--repeats", type=int, default=21)
    args = parser.parse_args()
    root_shape = tuple(int(value) for value in args.root_shape.split(","))
    block_shape = tuple(int(value) for value in args.block_shape.split(","))
    report = {
        "capability": "HAX-001",
        "root_shape": root_shape,
        "block_shape": block_shape,
        "repeats": args.repeats,
        "cases": [
            benchmark_case(
                root_shape,
                block_shape,
                int(halo),
                int(fields),
                int(capacity),
                args.repeats,
            )
            for capacity in args.capacities.split(",")
            for fields in args.fields.split(",")
            for halo in args.halos.split(",")
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
