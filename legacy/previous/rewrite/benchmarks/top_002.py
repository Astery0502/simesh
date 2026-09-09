"""On-demand refined contact throughput and lifecycle composition probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
import tracemalloc

import numpy as np

from simesh_rewrite.contacts import fill_refined_contact_targets
from simesh_rewrite.contacts_reference import refined_contact_targets_reference
from simesh_rewrite.forest import refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.morton import level1_morton


ALL_DIRECTIONS = np.asarray(
    [
        (dx, dy, dz)
        for dz in range(-1, 2)
        for dy in range(-1, 2)
        for dx in range(-1, 2)
        if (dx, dy, dz) != (0, 0, 0)
    ],
    dtype=np.int64,
)
FACE_DIRECTIONS = np.asarray(
    [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]],
    dtype=np.int64,
)


def full_tree_flags(root_count: int, internal_levels: int) -> np.ndarray:
    tree: list[bool] = []

    def visit(remaining: int) -> None:
        tree.append(remaining == 0)
        if remaining:
            for _ in range(8):
                visit(remaining - 1)

    visit(internal_levels)
    return np.tile(np.asarray(tree, dtype=bool), root_count)


def lookup_args(root, coord_to_rank, forest):
    return (
        root,
        coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
    )


def conformance_args(root, coord_to_rank, rank_to_coord, forest):
    return (
        root,
        coord_to_rank,
        rank_to_coord,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.parent_node_ids,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
    )


def median_seconds(operation, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def kind_counts(forest, source_ids, targets) -> dict:
    counts = {"physical": 0, "coarser": 0, "same": 0, "subdivided": 0}
    for source_leaf, target in zip(source_ids, targets, strict=True):
        if target < 0:
            counts["physical"] += 1
            continue
        source_node = int(forest.leaf_node_ids[int(source_leaf)])
        if forest.node_leaf_ids[target] < 0:
            counts["subdivided"] += 1
        elif forest.node_levels[target] < forest.node_levels[source_node]:
            counts["coarser"] += 1
        else:
            counts["same"] += 1
    return counts


def measure_depth(
    internal_levels: int,
    repeats: int,
    reference_limit: int,
) -> dict:
    root = np.asarray([1, 1, 1], dtype=np.int64)
    coord_to_rank, rank_to_coord = level1_morton(root)
    flags = full_tree_flags(1, internal_levels)
    forest = refined_forest(root, coord_to_rank, rank_to_coord, flags)
    source_ids = np.repeat(
        np.arange(forest.leaf_node_ids.size, dtype=np.int64), 26
    )
    directions = np.tile(ALL_DIRECTIONS, (forest.leaf_node_ids.size, 1))
    targets = np.empty(source_ids.size, dtype=np.int64)
    call_args = (*lookup_args(root, coord_to_rank, forest), source_ids, directions, targets)
    lifecycle_args = conformance_args(root, coord_to_rank, rank_to_coord, forest)
    validate_refined_forest_arrays(*lifecycle_args)

    def lookup() -> None:
        fill_refined_contact_targets(*call_args)

    lookup()
    seconds = median_seconds(lookup, repeats)
    fst_seconds = median_seconds(
        lambda: validate_refined_forest_arrays(*lifecycle_args),
        repeats,
    )

    def lifecycle() -> None:
        validate_refined_forest_arrays(*lifecycle_args)
        lookup()

    lifecycle_seconds = median_seconds(lifecycle, repeats)

    reference_seconds = None
    reference_exact = None
    if source_ids.size <= reference_limit:
        started = time.perf_counter()
        expected = refined_contact_targets_reference(
            root,
            forest.node_levels,
            forest.node_coords,
            forest.node_leaf_ids,
            forest.leaf_node_ids,
            source_ids,
            directions,
        )
        reference_seconds = time.perf_counter() - started
        reference_exact = bool(np.array_equal(targets, expected))

    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    lookup()
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "max_level": forest.max_level,
        "nodes": int(forest.node_levels.size),
        "leaves": int(forest.leaf_node_ids.size),
        "queries": int(source_ids.size),
        "lookup_seconds": seconds,
        "million_contacts_per_second": source_ids.size / seconds / 1.0e6,
        "fst_conformance_seconds": fst_seconds,
        "lifecycle_total_seconds": lifecycle_seconds,
        "output_bytes": targets.nbytes,
        "kind_counts": kind_counts(forest, source_ids, targets),
        "reference_seconds": reference_seconds,
        "reference_exact": reference_exact,
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
    }


def measure_dat(path: Path, repeats: int) -> dict:
    from simesh.amrvac.datio import get_metadata
    from simesh.utils.lib.amr.forest import AMRForest

    header, flags_input, _ = get_metadata(str(path))
    root = np.ascontiguousarray(
        header["domain_nx"] // header["block_nx"], dtype=np.int64
    )
    coord_to_rank, rank_to_coord = level1_morton(root)
    flags = np.ascontiguousarray(flags_input, dtype=bool)
    forest = refined_forest(root, coord_to_rank, rank_to_coord, flags)
    source_ids = np.repeat(
        np.arange(forest.leaf_node_ids.size, dtype=np.int64), 6
    )
    directions = np.tile(FACE_DIRECTIONS, (forest.leaf_node_ids.size, 1))
    targets = np.empty(source_ids.size, dtype=np.int64)
    call_args = (*lookup_args(root, coord_to_rank, forest), source_ids, directions, targets)
    lifecycle_args = conformance_args(root, coord_to_rank, rank_to_coord, forest)
    validate_refined_forest_arrays(*lifecycle_args)
    seconds = median_seconds(
        lambda: fill_refined_contact_targets(*call_args),
        repeats,
    )
    fst_seconds = median_seconds(
        lambda: validate_refined_forest_arrays(*lifecycle_args),
        repeats,
    )

    def lifecycle() -> None:
        validate_refined_forest_arrays(*lifecycle_args)
        fill_refined_contact_targets(*call_args)

    lifecycle_seconds = median_seconds(lifecycle, repeats)
    fill_refined_contact_targets(*call_args)

    started = time.perf_counter()
    current = AMRForest(
        3,
        *tuple(int(value) for value in root),
        flags.astype(np.int32),
    )
    current_constructor_seconds = time.perf_counter() - started
    current_columns = np.asarray([12, 14, 10, 16, 4, 22], dtype=np.int64)

    def current_table_copy() -> None:
        np.asarray(current.neighbor_type)[:, current_columns]
        np.asarray(current.neighbor_index)[:, current_columns]

    current_table_seconds = median_seconds(current_table_copy, repeats)
    current_face_bytes_copied = (
        forest.leaf_node_ids.size * 6 * np.dtype(np.uint32).itemsize * 2
    )
    current_topology_bytes = (
        np.asarray(current.neighbor_type).nbytes
        + np.asarray(current.neighbor_index).nbytes
        + np.asarray(current.neighbor_children).nbytes
    )
    return {
        "path": str(path),
        "staggered": bool(header["staggered"]),
        "nodes": int(forest.node_levels.size),
        "leaves": int(forest.leaf_node_ids.size),
        "max_level": forest.max_level,
        "queries": int(source_ids.size),
        "lookup_seconds": seconds,
        "fst_conformance_seconds": fst_seconds,
        "lifecycle_total_seconds": lifecycle_seconds,
        "million_contacts_per_second": source_ids.size / seconds / 1.0e6,
        "output_bytes": targets.nbytes,
        "kind_counts": kind_counts(forest, source_ids, targets),
        "current_constructor_seconds": current_constructor_seconds,
        "current_face_table_copy_seconds": current_table_seconds,
        "current_face_bytes_copied": current_face_bytes_copied,
        "current_topology_bytes": current_topology_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-level", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--reference-limit", type=int, default=20_000)
    parser.add_argument("--dat", type=Path)
    args = parser.parse_args()
    report = {
        "capability": "TOP-002",
        "repeats": args.repeats,
        "synthetic": [
            measure_depth(level - 1, args.repeats, args.reference_limit)
            for level in range(1, args.max_level + 1)
        ],
        "real_dat": measure_dat(args.dat, args.repeats) if args.dat else None,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
