"""BAL-001 all-touch policy runtime, scaling, and real-tree probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
import tracemalloc

import numpy as np

from simesh_rewrite.balance import validate_refined_all_touch_2to1
from simesh_rewrite.balance_reference import is_refined_all_touch_2to1_reference
from simesh_rewrite.forest import refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.morton import level1_morton


def full_tree_flags(root_count: int, internal_levels: int) -> np.ndarray:
    tree: list[bool] = []

    def visit(remaining: int) -> None:
        tree.append(remaining == 0)
        if remaining:
            for _ in range(8):
                visit(remaining - 1)

    visit(internal_levels)
    return np.tile(np.asarray(tree, dtype=bool), root_count)


def adaptive_flags(root, refine) -> np.ndarray:
    _, roots = level1_morton(root)
    flags: list[bool] = []

    def visit(level, coord) -> None:
        split = bool(refine(level, coord))
        flags.append(not split)
        if split:
            for child in range(8):
                bits = (child & 1, (child >> 1) & 1, (child >> 2) & 1)
                visit(
                    level + 1,
                    tuple(2 * coord[axis] + bits[axis] for axis in range(3)),
                )

    for coord in roots:
        visit(1, tuple(int(value) for value in coord))
    return np.asarray(flags, dtype=bool)


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


def balance_args(root, coord_to_rank, forest):
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


def median_seconds(operation, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def measure_uniform(
    max_level: int,
    repeats: int,
    reference_leaf_limit: int,
) -> dict:
    root = np.asarray([1, 1, 1], dtype=np.int64)
    coord_to_rank, rank_to_coord = level1_morton(root)
    forest = refined_forest(
        root,
        coord_to_rank,
        rank_to_coord,
        full_tree_flags(1, max_level - 1),
    )
    fst_args = conformance_args(root, coord_to_rank, rank_to_coord, forest)
    policy_args = balance_args(root, coord_to_rank, forest)
    validate_refined_forest_arrays(*fst_args)
    validate_refined_all_touch_2to1(*policy_args)
    seconds = median_seconds(
        lambda: validate_refined_all_touch_2to1(*policy_args),
        repeats,
    )

    def lifecycle() -> None:
        validate_refined_forest_arrays(*fst_args)
        validate_refined_all_touch_2to1(*policy_args)

    lifecycle_seconds = median_seconds(lifecycle, repeats)
    reference_seconds = None
    reference_balanced = None
    if forest.leaf_node_ids.size <= reference_leaf_limit:
        started = time.perf_counter()
        reference_balanced = is_refined_all_touch_2to1_reference(
            forest.node_levels,
            forest.node_coords,
            forest.leaf_node_ids,
        )
        reference_seconds = time.perf_counter() - started

    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    validate_refined_all_touch_2to1(*policy_args)
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    directions = forest.leaf_node_ids.size * 26
    return {
        "max_level": max_level,
        "nodes": int(forest.node_levels.size),
        "leaves": int(forest.leaf_node_ids.size),
        "directions": int(directions),
        "balance_seconds": seconds,
        "million_leaf_directions_per_second": directions / seconds / 1.0e6,
        "lifecycle_seconds": lifecycle_seconds,
        "reference_seconds": reference_seconds,
        "reference_balanced": reference_balanced,
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
    }


def measure_unbalanced(repeats: int) -> dict:
    root = np.asarray([2, 2, 1], dtype=np.int64)
    coord_to_rank, rank_to_coord = level1_morton(root)
    flags = adaptive_flags(
        root,
        lambda level, coord: (
            level == 1 and coord in {(0, 0, 0), (1, 0, 0), (0, 1, 0)}
        )
        or (level == 2 and coord == (1, 1, 0)),
    )
    forest = refined_forest(root, coord_to_rank, rank_to_coord, flags)
    fst_args = conformance_args(root, coord_to_rank, rank_to_coord, forest)
    policy_args = balance_args(root, coord_to_rank, forest)
    validate_refined_forest_arrays(*fst_args)

    def reject() -> None:
        try:
            validate_refined_all_touch_2to1(*policy_args)
        except ValueError:
            return
        raise AssertionError("expected balance violation")

    seconds = median_seconds(reject, repeats)
    return {
        "nodes": int(forest.node_levels.size),
        "leaves": int(forest.leaf_node_ids.size),
        "reject_seconds": seconds,
        "pairwise_balanced": is_refined_all_touch_2to1_reference(
            forest.node_levels,
            forest.node_coords,
            forest.leaf_node_ids,
        ),
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
    fst_args = conformance_args(root, coord_to_rank, rank_to_coord, forest)
    policy_args = balance_args(root, coord_to_rank, forest)
    validate_refined_forest_arrays(*fst_args)
    validate_refined_all_touch_2to1(*policy_args)
    seconds = median_seconds(
        lambda: validate_refined_all_touch_2to1(*policy_args),
        repeats,
    )

    def lifecycle() -> None:
        validate_refined_forest_arrays(*fst_args)
        validate_refined_all_touch_2to1(*policy_args)

    lifecycle_seconds = median_seconds(lifecycle, repeats)
    current_flags = flags.astype(np.int32)

    def current_constructor() -> None:
        AMRForest(3, *tuple(int(value) for value in root), current_flags)

    current_seconds = median_seconds(current_constructor, repeats)
    return {
        "path": str(path),
        "staggered": bool(header["staggered"]),
        "nodes": int(forest.node_levels.size),
        "leaves": int(forest.leaf_node_ids.size),
        "max_level": forest.max_level,
        "directions": int(forest.leaf_node_ids.size * 26),
        "balance_seconds": seconds,
        "million_leaf_directions_per_second": (
            forest.leaf_node_ids.size * 26 / seconds / 1.0e6
        ),
        "lifecycle_seconds": lifecycle_seconds,
        "current_constructor_seconds": current_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-level", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--reference-leaf-limit", type=int, default=600)
    parser.add_argument("--dat", type=Path)
    args = parser.parse_args()
    report = {
        "capability": "BAL-001",
        "repeats": args.repeats,
        "uniform": [
            measure_uniform(level, args.repeats, args.reference_leaf_limit)
            for level in range(1, args.max_level + 1)
        ],
        "face_balanced_diagonal_violation": measure_unbalanced(args.repeats),
        "real_dat": measure_dat(args.dat, args.repeats) if args.dat else None,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
