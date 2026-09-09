"""Runtime, allocation, and lifecycle overhead probe for FST-002."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
import tracemalloc

import numpy as np

from simesh_rewrite.forest import refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.forest_conformance_reference import (
    validate_refined_forest_arrays_reference,
)
from simesh_rewrite.morton import level1_morton


def parse_profile(value: str) -> tuple[tuple[int, int, int], int]:
    shape_text, depth_text = value.split(":", maxsplit=1)
    shape = tuple(int(part) for part in shape_text.lower().split("x"))
    if len(shape) != 3:
        raise ValueError(f"invalid profile {value!r}")
    return shape, int(depth_text)


def full_tree_flags(root_count: int, internal_levels: int) -> np.ndarray:
    tree: list[bool] = []

    def visit(remaining: int) -> None:
        tree.append(remaining == 0)
        if remaining:
            for _ in range(8):
                visit(remaining - 1)

    visit(internal_levels)
    return np.tile(np.asarray(tree, dtype=bool), root_count)


def arguments(root_shape, coord_to_rank, rank_to_coord, forest):
    return (
        root_shape,
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


def measure(
    shape: tuple[int, int, int],
    internal_levels: int,
    repeats: int,
    reference_limit: int,
) -> dict:
    root_shape = np.asarray(shape, dtype=np.int64)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    flags = full_tree_flags(rank_to_coord.shape[0], internal_levels)
    forest = refined_forest(root_shape, coord_to_rank, rank_to_coord, flags)
    call_args = arguments(root_shape, coord_to_rank, rank_to_coord, forest)
    assert validate_refined_forest_arrays(*call_args) == forest.max_level

    validate_seconds = median_seconds(
        lambda: validate_refined_forest_arrays(*call_args),
        repeats,
    )
    construct_seconds = median_seconds(
        lambda: refined_forest(
            root_shape,
            coord_to_rank,
            rank_to_coord,
            flags,
        ),
        repeats,
    )

    reference_seconds = None
    if forest.node_levels.size <= reference_limit:
        started = time.perf_counter()
        result = validate_refined_forest_arrays_reference(
            root_shape,
            rank_to_coord,
            forest.root_node_ids,
            forest.node_levels,
            forest.node_coords,
            forest.parent_node_ids,
            forest.child_node_ids,
            forest.node_leaf_ids,
            forest.leaf_node_ids,
        )
        reference_seconds = time.perf_counter() - started
        assert result == forest.max_level

    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    validate_refined_forest_arrays(*call_args)
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "shape": shape,
        "max_level": forest.max_level,
        "nodes": int(forest.node_levels.size),
        "leaves": int(forest.leaf_node_ids.size),
        "validate_seconds": validate_seconds,
        "million_nodes_per_second": (
            forest.node_levels.size / validate_seconds / 1.0e6
        ),
        "construct_seconds": construct_seconds,
        "validate_over_construct": validate_seconds / construct_seconds,
        "reference_seconds": reference_seconds,
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
    }


def measure_dat(path: Path, repeats: int) -> dict:
    from simesh.amrvac.datio import get_metadata

    header, flags_input, _ = get_metadata(str(path))
    root_shape = np.ascontiguousarray(
        header["domain_nx"] // header["block_nx"], dtype=np.int64
    )
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    flags = np.ascontiguousarray(flags_input, dtype=bool)
    forest = refined_forest(root_shape, coord_to_rank, rank_to_coord, flags)
    call_args = arguments(root_shape, coord_to_rank, rank_to_coord, forest)
    seconds = median_seconds(
        lambda: validate_refined_forest_arrays(*call_args),
        repeats,
    )
    return {
        "path": str(path),
        "staggered": bool(header["staggered"]),
        "nodes": int(forest.node_levels.size),
        "leaves": int(forest.leaf_node_ids.size),
        "max_level": forest.max_level,
        "validate_seconds": seconds,
        "million_nodes_per_second": forest.node_levels.size / seconds / 1.0e6,
        "exact_max_level": validate_refined_forest_arrays(*call_args)
        == forest.max_level,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profiles",
        default="4x4x2:1,4x4x2:2,4x4x2:3,4x4x2:4",
    )
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--reference-limit", type=int, default=50_000)
    parser.add_argument("--dat", type=Path)
    args = parser.parse_args()
    report = {
        "capability": "FST-002",
        "repeats": args.repeats,
        "profiles": [
            measure(shape, depth, args.repeats, args.reference_limit)
            for shape, depth in (
                parse_profile(value) for value in args.profiles.split(",")
            )
        ],
        "real_dat": measure_dat(args.dat, args.repeats) if args.dat else None,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
