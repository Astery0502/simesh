"""GEO-002 selected refined geometry runtime, memory, and real-tree probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
import tracemalloc

import numpy as np

from simesh_rewrite._geometry import (
    fill_refined_leaf_geometry_unchecked,
    validate_selected_refined_geometry_unchecked,
)
from simesh_rewrite.balance import validate_refined_all_touch_2to1
from simesh_rewrite.contacts import refined_contact_targets
from simesh_rewrite.forest import refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.refined_geometry import (
    fill_refined_leaf_geometry,
    refined_leaf_geometry,
)
from simesh_rewrite.refined_geometry_reference import (
    refined_leaf_geometry_reference,
)


FACE_DIRECTIONS = np.asarray(
    [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)],
    dtype=np.int64,
)


def full_tree_flags(root_count: int, internal_levels: int) -> np.ndarray:
    values: list[bool] = []

    def visit(remaining: int) -> None:
        values.append(remaining == 0)
        if remaining:
            for _ in range(8):
                visit(remaining - 1)

    visit(internal_levels)
    return np.tile(np.asarray(values, dtype=np.bool_), root_count)


def conformance_args(root_shape, coord_to_rank, rank_to_coord, forest):
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


def geometry_args(
    domain_lower,
    domain_upper,
    root_shape,
    domain_cell_counts,
    block_cell_counts,
    forest,
    leaf_ids,
):
    return (
        domain_lower,
        domain_upper,
        root_shape,
        domain_cell_counts,
        block_cell_counts,
        forest.node_levels,
        forest.node_coords,
        forest.leaf_node_ids,
        leaf_ids,
    )


def median_seconds(operation, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def measure_uniform_level(
    max_level: int,
    repeats: int,
    reference_leaf_limit: int,
) -> dict:
    root_shape = np.asarray([1, 1, 1], dtype=np.int64)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    forest = refined_forest(
        root_shape,
        coord_to_rank,
        rank_to_coord,
        full_tree_flags(1, max_level - 1),
    )
    fst_args = conformance_args(root_shape, coord_to_rank, rank_to_coord, forest)
    validate_refined_forest_arrays(*fst_args)
    block_cells = np.asarray([8, 6, 10], dtype=np.int64)
    domain_cells = root_shape * block_cells
    domain_lower = np.asarray([-0.7, 1.1, -2.3], dtype=np.float64)
    domain_upper = np.asarray([1.9, 4.7, 7.2], dtype=np.float64)
    leaf_ids = np.arange(forest.leaf_node_ids.size, dtype=np.int64)
    if leaf_ids.size > 1:
        leaf_ids = np.ascontiguousarray((leaf_ids * 104729) % leaf_ids.size)
    args = geometry_args(
        domain_lower,
        domain_upper,
        root_shape,
        domain_cells,
        block_cells,
        forest,
        leaf_ids,
    )
    bounds = np.empty((leaf_ids.size, 2, 3), dtype=np.float64)
    spacing = np.empty((leaf_ids.size, 3), dtype=np.float64)
    checked_args = (*args, bounds, spacing)
    fill_refined_leaf_geometry(*checked_args)
    base_spacing = (domain_upper - domain_lower) / domain_cells
    validation_args = (
        domain_lower,
        domain_upper,
        root_shape,
        domain_cells,
        block_cells,
        forest.node_levels,
        forest.node_coords,
        forest.leaf_node_ids,
        leaf_ids,
        base_spacing,
    )
    fill_args = (
        domain_lower,
        domain_upper,
        domain_cells,
        block_cells,
        forest.node_levels,
        forest.node_coords,
        forest.leaf_node_ids,
        leaf_ids,
        base_spacing,
        bounds,
        spacing,
    )
    checked_seconds = median_seconds(
        lambda: fill_refined_leaf_geometry(*checked_args), repeats
    )
    validation_seconds = median_seconds(
        lambda: validate_selected_refined_geometry_unchecked(*validation_args),
        repeats,
    )
    fill_seconds = median_seconds(
        lambda: fill_refined_leaf_geometry_unchecked(*fill_args), repeats
    )
    allocating_seconds = median_seconds(
        lambda: refined_leaf_geometry(*args), repeats
    )

    def lifecycle() -> None:
        validate_refined_forest_arrays(*fst_args)
        fill_refined_leaf_geometry(*checked_args)

    lifecycle_seconds = median_seconds(lifecycle, repeats)
    reference_seconds = None
    if leaf_ids.size <= reference_leaf_limit:
        reference_seconds = median_seconds(
            lambda: refined_leaf_geometry_reference(*args), 1
        )
        expected_bounds, expected_spacing = refined_leaf_geometry_reference(*args)
        tolerance = 16.0 * np.finfo(np.float64).eps * np.maximum.reduce(
            (np.abs(domain_lower), np.abs(domain_upper), domain_upper - domain_lower)
        )
        if not np.all(np.abs(bounds - expected_bounds) <= tolerance):
            raise AssertionError("production bounds disagree with reference")
        np.testing.assert_allclose(
            spacing,
            expected_spacing,
            rtol=8.0 * np.finfo(np.float64).eps,
            atol=0.0,
        )

    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    fill_refined_leaf_geometry(*checked_args)
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "max_level": max_level,
        "nodes": int(forest.node_levels.size),
        "leaves": int(leaf_ids.size),
        "checked_seconds": checked_seconds,
        "validation_seconds": validation_seconds,
        "fill_seconds": fill_seconds,
        "allocating_seconds": allocating_seconds,
        "lifecycle_seconds": lifecycle_seconds,
        "million_leaves_per_second": leaf_ids.size / checked_seconds / 1.0e6,
        "output_bytes": int(bounds.nbytes + spacing.nbytes),
        "reference_seconds": reference_seconds,
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
    }


def expanded_face_evidence(root_shape, coord_to_rank, forest, bounds, current):
    leaf_count = forest.leaf_node_ids.size
    source_leaf_ids = np.repeat(
        np.arange(leaf_count, dtype=np.int64), FACE_DIRECTIONS.shape[0]
    )
    directions = np.tile(FACE_DIRECTIONS, (leaf_count, 1))
    targets = refined_contact_targets(
        root_shape,
        coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
        source_leaf_ids,
        directions,
    )
    counts = {"same": 0, "fine_to_coarse": 0, "coarse_to_fine": 0}
    canonical_mismatches = 0
    current_mismatches = 0
    current_max_gap = 0.0
    for query, target_node_value in enumerate(targets):
        target_node = int(target_node_value)
        if target_node < 0:
            continue
        source_leaf = int(source_leaf_ids[query])
        direction = directions[query]
        axis = int(np.flatnonzero(direction)[0])
        source_side = 1 if direction[axis] > 0 else 0
        target_side = 1 - source_side
        target_leaf = int(forest.node_leaf_ids[target_node])
        if target_leaf >= 0:
            target_leaves = (target_leaf,)
            source_node = int(forest.leaf_node_ids[source_leaf])
            relation = (
                "same"
                if forest.node_levels[source_node] == forest.node_levels[target_node]
                else "fine_to_coarse"
            )
        else:
            touching_bit = 0 if direction[axis] > 0 else 1
            target_leaves = tuple(
                int(forest.node_leaf_ids[child_node])
                for child, child_node in enumerate(forest.child_node_ids[target_node])
                if ((child >> axis) & 1) == touching_bit
            )
            relation = "coarse_to_fine"
        for adjacent_leaf in target_leaves:
            counts[relation] += 1
            canonical_left = bounds[source_leaf, source_side, axis]
            canonical_right = bounds[adjacent_leaf, target_side, axis]
            canonical_mismatches += int(
                canonical_left.tobytes() != canonical_right.tobytes()
            )
            current_left = current[source_leaf, source_side, axis]
            current_right = current[adjacent_leaf, target_side, axis]
            current_mismatches += int(
                current_left.tobytes() != current_right.tobytes()
            )
            current_max_gap = max(
                current_max_gap,
                abs(float(current_left) - float(current_right)),
            )
    return {
        "relation_counts": counts,
        "canonical_bit_mismatches": canonical_mismatches,
        "current_bit_mismatches": current_mismatches,
        "current_max_abs_gap": current_max_gap,
    }


def measure_dat(
    path: Path,
    repeats: int,
    reference_leaf_limit: int,
    chunk_leaves: int,
) -> dict:
    from simesh.amrvac.datio import get_metadata
    from simesh.utils.lib.amr.forest import AMRForest
    from simesh.utils.lib.amr.mesh import AMRMesh

    header, flags_input, tree = get_metadata(str(path))
    root_shape = np.ascontiguousarray(
        header["domain_nx"] // header["block_nx"], dtype=np.int64
    )
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    flags = np.ascontiguousarray(flags_input, dtype=np.bool_)
    forest = refined_forest(
        root_shape, coord_to_rank, rank_to_coord, flags
    )
    fst_args = conformance_args(root_shape, coord_to_rank, rank_to_coord, forest)
    validate_refined_forest_arrays(*fst_args)
    if not np.array_equal(
        forest.node_levels[forest.leaf_node_ids], np.asarray(tree[0], dtype=np.int64)
    ) or not np.array_equal(
        forest.node_coords[forest.leaf_node_ids] + 1,
        np.asarray(tree[1], dtype=np.int64),
    ):
        raise AssertionError("reconstructed geometry metadata disagrees with file")

    domain_lower = np.ascontiguousarray(header["xmin"], dtype=np.float64)
    domain_upper = np.ascontiguousarray(header["xmax"], dtype=np.float64)
    domain_cells = np.ascontiguousarray(header["domain_nx"], dtype=np.int64)
    block_cells = np.ascontiguousarray(header["block_nx"], dtype=np.int64)
    leaf_ids = np.arange(forest.leaf_node_ids.size, dtype=np.int64)
    args = geometry_args(
        domain_lower,
        domain_upper,
        root_shape,
        domain_cells,
        block_cells,
        forest,
        leaf_ids,
    )
    bounds = np.empty((leaf_ids.size, 2, 3), dtype=np.float64)
    spacing = np.empty((leaf_ids.size, 3), dtype=np.float64)
    checked_args = (*args, bounds, spacing)
    fill_refined_leaf_geometry(*checked_args)
    base_spacing = (domain_upper - domain_lower) / domain_cells
    validation_args = (
        domain_lower,
        domain_upper,
        root_shape,
        domain_cells,
        block_cells,
        forest.node_levels,
        forest.node_coords,
        forest.leaf_node_ids,
        leaf_ids,
        base_spacing,
    )
    fill_args = (
        domain_lower,
        domain_upper,
        domain_cells,
        block_cells,
        forest.node_levels,
        forest.node_coords,
        forest.leaf_node_ids,
        leaf_ids,
        base_spacing,
        bounds,
        spacing,
    )
    geometry_seconds = median_seconds(
        lambda: fill_refined_leaf_geometry(*checked_args), repeats
    )
    validation_seconds = median_seconds(
        lambda: validate_selected_refined_geometry_unchecked(*validation_args),
        repeats,
    )
    fill_seconds = median_seconds(
        lambda: fill_refined_leaf_geometry_unchecked(*fill_args), repeats
    )
    allocating_seconds = median_seconds(
        lambda: refined_leaf_geometry(*args), repeats
    )
    bounded_capacity = min(chunk_leaves, leaf_ids.size)
    bounded_bounds = np.empty((bounded_capacity, 2, 3), dtype=np.float64)
    bounded_spacing = np.empty((bounded_capacity, 3), dtype=np.float64)

    def bounded_traversal() -> None:
        for start in range(0, leaf_ids.size, bounded_capacity):
            stop = min(start + bounded_capacity, leaf_ids.size)
            count = stop - start
            fill_refined_leaf_geometry(
                domain_lower,
                domain_upper,
                root_shape,
                domain_cells,
                block_cells,
                forest.node_levels,
                forest.node_coords,
                forest.leaf_node_ids,
                leaf_ids[start:stop],
                bounded_bounds[:count],
                bounded_spacing[:count],
            )

    bounded_traversal()
    bounded_seconds = median_seconds(bounded_traversal, repeats)

    def lifecycle() -> None:
        validate_refined_forest_arrays(*fst_args)
        fill_refined_leaf_geometry(*checked_args)

    lifecycle_seconds = median_seconds(lifecycle, repeats)
    current_flags = flags.astype(np.int32)
    block_cells_current = block_cells.astype(np.uint32)
    domain_cells_current = domain_cells.astype(np.uint32)
    current_forest = AMRForest(
        3, *tuple(int(value) for value in root_shape), current_flags
    )

    def current_mesh_constructor() -> None:
        AMRMesh(
            3,
            block_cells_current,
            domain_cells_current,
            domain_lower,
            domain_upper,
            np.uint32(0),
            np.uint32(1),
            current_forest,
        )

    current_seconds = median_seconds(current_mesh_constructor, repeats)
    current_mesh = AMRMesh(
        3,
        block_cells_current,
        domain_cells_current,
        domain_lower,
        domain_upper,
        np.uint32(0),
        np.uint32(1),
        current_forest,
    )
    current_rnode = np.asarray(current_mesh.rnode)
    current_bounds = current_rnode[:, :6].reshape(-1, 2, 3)
    current_copy_seconds = median_seconds(lambda: current_rnode.copy(), repeats)
    validate_refined_all_touch_2to1(
        root_shape,
        coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
    )
    face_evidence = expanded_face_evidence(
        root_shape,
        coord_to_rank,
        forest,
        bounds,
        current_bounds,
    )
    reference_ids = leaf_ids[: min(reference_leaf_limit, leaf_ids.size)]
    reference_args = geometry_args(
        domain_lower,
        domain_upper,
        root_shape,
        domain_cells,
        block_cells,
        forest,
        reference_ids,
    )
    expected_bounds, expected_spacing = refined_leaf_geometry_reference(
        *reference_args
    )
    tolerance = 16.0 * np.finfo(np.float64).eps * np.maximum.reduce(
        (np.abs(domain_lower), np.abs(domain_upper), domain_upper - domain_lower)
    )
    if not np.all(np.abs(bounds[: reference_ids.size] - expected_bounds) <= tolerance):
        raise AssertionError("real bounds disagree with exact reference")
    np.testing.assert_allclose(
        spacing[: reference_ids.size],
        expected_spacing,
        rtol=8.0 * np.finfo(np.float64).eps,
        atol=0.0,
    )
    return {
        "path": str(path),
        "staggered": bool(header["staggered"]),
        "nodes": int(forest.node_levels.size),
        "leaves": int(leaf_ids.size),
        "max_level": int(forest.max_level),
        "level_counts": {
            str(level): int(
                np.count_nonzero(
                    forest.node_levels[forest.leaf_node_ids] == level
                )
            )
            for level in range(1, int(forest.max_level) + 1)
        },
        "geometry_seconds": geometry_seconds,
        "validation_seconds": validation_seconds,
        "fill_seconds": fill_seconds,
        "allocating_seconds": allocating_seconds,
        "bounded_seconds": bounded_seconds,
        "bounded_capacity": bounded_capacity,
        "bounded_workspace_bytes": int(
            bounded_bounds.nbytes + bounded_spacing.nbytes
        ),
        "million_leaves_per_second": leaf_ids.size / geometry_seconds / 1.0e6,
        "lifecycle_seconds": lifecycle_seconds,
        "current_mesh_constructor_seconds": current_seconds,
        "current_cached_copy_seconds": current_copy_seconds,
        "output_bytes": int(bounds.nbytes + spacing.nbytes),
        "current_bounds_max_abs_by_axis": np.max(
            np.abs(bounds - current_bounds), axis=(0, 1)
        ).tolist(),
        "current_spacing_max_abs": float(
            np.max(np.abs(spacing - current_rnode[:, 6:9]))
        ),
        "face_evidence": face_evidence,
        "face_evidence_balance_precondition": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-level", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--reference-leaf-limit", type=int, default=600)
    parser.add_argument("--chunk-leaves", type=int, default=4096)
    parser.add_argument("--dat", type=Path)
    args = parser.parse_args()
    report = {
        "capability": "GEO-002",
        "repeats": args.repeats,
        "uniform": [
            measure_uniform_level(
                level, args.repeats, args.reference_leaf_limit
            )
            for level in range(1, args.max_level + 1)
        ],
        "real_dat": (
            measure_dat(
                args.dat,
                args.repeats,
                args.reference_leaf_limit,
                args.chunk_leaves,
            )
            if args.dat
            else None
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
