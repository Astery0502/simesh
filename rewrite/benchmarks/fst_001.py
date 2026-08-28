"""Retained runtime, throughput, allocation, and scaling probe for FST-001."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
import tracemalloc

import numpy as np

from simesh_rewrite.forest import fill_refined_forest, refined_forest
from simesh_rewrite.forest_reference import refined_forest_reference
from simesh_rewrite.morton import level1_morton


def parse_profile(value: str) -> tuple[tuple[int, int, int], int]:
    shape_text, depth_text = value.split(":", maxsplit=1)
    shape = tuple(int(part) for part in shape_text.lower().split("x"))
    if len(shape) != 3:
        raise ValueError(f"invalid shape in profile {value!r}")
    return shape, int(depth_text)


def full_tree_flags(root_count: int, internal_levels: int) -> np.ndarray:
    tree: list[bool] = []

    def emit(remaining: int) -> None:
        tree.append(remaining == 0)
        if remaining:
            for _ in range(8):
                emit(remaining - 1)

    emit(internal_levels)
    return np.tile(np.asarray(tree, dtype=np.bool_), root_count)


def allocate_outputs(node_count: int, leaf_count: int, root_count: int):
    return (
        np.empty(node_count, dtype=np.int64),
        np.empty((node_count, 3), dtype=np.int64),
        np.empty(node_count, dtype=np.int64),
        np.empty((node_count, 8), dtype=np.int64),
        np.empty(node_count, dtype=np.int64),
        np.empty(leaf_count, dtype=np.int64),
        np.empty(root_count, dtype=np.int64),
    )


def measure(
    shape: tuple[int, int, int],
    internal_levels: int,
    repeats: int,
    reference_limit: int,
) -> dict:
    root_shape = np.asarray(shape, dtype=np.int64)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    root_count = rank_to_coord.shape[0]
    flags = full_tree_flags(root_count, internal_levels)
    node_count = flags.size
    leaf_count = int(np.count_nonzero(flags))
    outputs = allocate_outputs(node_count, leaf_count, root_count)
    fill_refined_forest(
        root_shape,
        coord_to_rank,
        rank_to_coord,
        flags,
        *outputs,
    )

    fill_samples = []
    allocating_samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fill_refined_forest(
            root_shape,
            coord_to_rank,
            rank_to_coord,
            flags,
            *outputs,
        )
        fill_samples.append(time.perf_counter() - started)

        started = time.perf_counter()
        refined_forest(root_shape, coord_to_rank, rank_to_coord, flags)
        allocating_samples.append(time.perf_counter() - started)

    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    fill_refined_forest(
        root_shape,
        coord_to_rank,
        rank_to_coord,
        flags,
        *outputs,
    )
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    reference_seconds = None
    if node_count <= reference_limit:
        started = time.perf_counter()
        refined_forest_reference(root_shape, rank_to_coord, flags)
        reference_seconds = time.perf_counter() - started

    fill_median = statistics.median(fill_samples)
    output_bytes = sum(output.nbytes for output in outputs)
    return {
        "shape": shape,
        "internal_levels": internal_levels,
        "max_level": internal_levels + 1,
        "roots": root_count,
        "nodes": node_count,
        "leaves": leaf_count,
        "fill_median_seconds": fill_median,
        "allocating_median_seconds": statistics.median(allocating_samples),
        "million_nodes_per_second": node_count / fill_median / 1.0e6,
        "output_bytes": output_bytes,
        "formula_bytes": 112 * node_count + 8 * leaf_count + 8 * root_count,
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
        "reference_seconds": reference_seconds,
    }


def measure_dat(path: Path, repeats: int, current_repeats: int) -> dict:
    from simesh.amrvac.datio import get_metadata
    from simesh.utils.lib.amr.forest import AMRForest

    header, flags_input, tree = get_metadata(str(path))
    flags = np.ascontiguousarray(flags_input, dtype=np.bool_)
    root_shape = np.ascontiguousarray(
        header["domain_nx"] // header["block_nx"], dtype=np.int64
    )
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    samples = []
    forest = None
    for _ in range(repeats):
        started = time.perf_counter()
        forest = refined_forest(
            root_shape,
            coord_to_rank,
            rank_to_coord,
            flags,
        )
        samples.append(time.perf_counter() - started)
    assert forest is not None
    leaf_nodes = forest.leaf_node_ids
    levels_exact = np.array_equal(
        forest.node_levels[leaf_nodes], np.asarray(tree[0], dtype=np.int64)
    )
    coords_exact = np.array_equal(
        forest.node_coords[leaf_nodes] + 1,
        np.asarray(tree[1], dtype=np.int64),
    )

    current_samples = []
    current_roundtrip_exact = True
    flags_i32 = np.ascontiguousarray(flags, dtype=np.int32)
    shape = tuple(int(value) for value in root_shape)
    for _ in range(current_repeats):
        started = time.perf_counter()
        current = AMRForest(3, *shape, flags_i32)
        current_samples.append(time.perf_counter() - started)
        current_roundtrip_exact = current_roundtrip_exact and np.array_equal(
            np.asarray(current.write_forest(), dtype=bool), flags
        )

    return {
        "path": str(path),
        "staggered": bool(header["staggered"]),
        "root_shape": shape,
        "nodes": int(flags.size),
        "leaves": int(leaf_nodes.size),
        "parents": int(np.count_nonzero(~flags)),
        "max_level": forest.max_level,
        "root_node_ids": forest.root_node_ids.tolist(),
        "output_bytes": sum(array.nbytes for array in forest[:-1]),
        "rewrite_median_seconds": statistics.median(samples),
        "leaf_levels_exact": bool(levels_exact),
        "leaf_coordinates_plus_one_exact": bool(coords_exact),
        "current_constructor_median_seconds": statistics.median(current_samples),
        "current_constructor_includes_connectivity": True,
        "current_forest_roundtrip_exact": bool(current_roundtrip_exact),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profiles",
        default="4x4x2:1,4x4x2:2,4x4x2:3,4x4x2:4",
    )
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--reference-limit", type=int, default=50_000)
    parser.add_argument("--dat", type=Path)
    parser.add_argument("--dat-repeats", type=int, default=31)
    parser.add_argument("--current-repeats", type=int, default=5)
    args = parser.parse_args()

    report = {
        "capability": "FST-001",
        "repeats": args.repeats,
        "profiles": [
            measure(shape, depth, args.repeats, args.reference_limit)
            for shape, depth in (
                parse_profile(value) for value in args.profiles.split(",")
            )
        ],
        "real_dat": (
            measure_dat(args.dat, args.dat_repeats, args.current_repeats)
            if args.dat is not None
            else None
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
