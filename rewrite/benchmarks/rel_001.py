"""REL-001 runtime, memory, scaling, and real-tree cache decision probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
import tracemalloc

import numpy as np

from simesh_rewrite._relations import fill_balanced_refined_relations_unchecked
from simesh_rewrite.balance import validate_refined_all_touch_2to1
from simesh_rewrite.contacts import refined_contact_targets
from simesh_rewrite.forest import refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.relations import (
    RELATION_COARSER,
    RELATION_FINER,
    RELATION_PHYSICAL,
    RELATION_SAME,
    balanced_refined_relations,
    fill_balanced_refined_relations,
)
from simesh_rewrite.relations_reference import (
    balanced_refined_relations_reference,
)


ALL_DIRECTIONS = np.asarray(
    [
        (dx, dy, dz)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ],
    dtype=np.int64,
)
FACE_DIRECTIONS = np.asarray(
    [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)],
    dtype=np.int64,
)
EDGE_CORNER_DIRECTIONS = np.ascontiguousarray(
    ALL_DIRECTIONS[np.count_nonzero(ALL_DIRECTIONS, axis=1) >= 2]
)
NONCENTER_COLUMNS = np.asarray(
    [column for column in range(27) if column != 13], dtype=np.int64
)
BIT_COUNTS = np.asarray([value.bit_count() for value in range(8)], dtype=np.uint8)


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


def relation_forest_args(root_shape, coord_to_rank, forest):
    return (
        root_shape,
        coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
    )


def output_arrays(primary_count: int, direction_count: int):
    shape = (primary_count, direction_count)
    return (
        np.empty(shape, dtype=np.uint8),
        np.empty(shape, dtype=np.uint8),
        np.empty(shape, dtype=np.uint8),
        np.empty((*shape, 4), dtype=np.int64),
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
    root_shape = np.asarray([1, 1, 1], dtype=np.int64)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    forest = refined_forest(
        root_shape,
        coord_to_rank,
        rank_to_coord,
        full_tree_flags(1, max_level - 1),
    )
    fst_args = conformance_args(root_shape, coord_to_rank, rank_to_coord, forest)
    rel_forest_args = relation_forest_args(root_shape, coord_to_rank, forest)
    validate_refined_forest_arrays(*fst_args)
    validate_refined_all_touch_2to1(*rel_forest_args)
    leaf_ids = np.arange(forest.leaf_node_ids.size, dtype=np.int64)
    if leaf_ids.size > 1:
        leaf_ids = np.ascontiguousarray((leaf_ids * 104729) % leaf_ids.size)
    outputs = output_arrays(leaf_ids.size, ALL_DIRECTIONS.shape[0])
    checked_args = (*rel_forest_args, leaf_ids, ALL_DIRECTIONS, *outputs)
    unchecked_args = checked_args
    fill_balanced_refined_relations(*checked_args)
    checked_seconds = median_seconds(
        lambda: fill_balanced_refined_relations(*checked_args), repeats
    )
    fill_seconds = median_seconds(
        lambda: fill_balanced_refined_relations_unchecked(*unchecked_args),
        repeats,
    )
    allocating_seconds = median_seconds(
        lambda: balanced_refined_relations(
            *rel_forest_args, leaf_ids, ALL_DIRECTIONS
        ),
        repeats,
    )

    def lifecycle() -> None:
        validate_refined_forest_arrays(*fst_args)
        validate_refined_all_touch_2to1(*rel_forest_args)
        fill_balanced_refined_relations(*checked_args)

    lifecycle_seconds = median_seconds(lifecycle, repeats)
    reference_seconds = None
    if leaf_ids.size <= reference_leaf_limit:
        reference_args = (
            root_shape,
            forest.node_levels,
            forest.node_coords,
            forest.leaf_node_ids,
            leaf_ids,
            ALL_DIRECTIONS,
        )
        expected = balanced_refined_relations_reference(*reference_args)
        if not all(
            np.array_equal(actual, wanted)
            for actual, wanted in zip(outputs, expected, strict=True)
        ):
            raise AssertionError("production relations disagree with reference")
        reference_seconds = median_seconds(
            lambda: balanced_refined_relations_reference(*reference_args), 1
        )

    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    fill_balanced_refined_relations(*checked_args)
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    record_count = leaf_ids.size * ALL_DIRECTIONS.shape[0]
    source_occurrences = int(np.sum(outputs[2], dtype=np.int64))
    return {
        "max_level": max_level,
        "nodes": int(forest.node_levels.size),
        "leaves": int(leaf_ids.size),
        "records": int(record_count),
        "checked_seconds": checked_seconds,
        "fill_seconds": fill_seconds,
        "allocating_seconds": allocating_seconds,
        "lifecycle_seconds": lifecycle_seconds,
        "million_records_per_second": record_count / checked_seconds / 1.0e6,
        "million_sources_per_second": source_occurrences / checked_seconds / 1.0e6,
        "source_occurrences": source_occurrences,
        "output_bytes": int(sum(output.nbytes for output in outputs)),
        "reference_seconds": reference_seconds,
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
    }


def measure_dat(
    path: Path,
    repeats: int,
    chunk_leaves: int,
) -> dict:
    from simesh.amrvac.datio import get_metadata
    from simesh.utils.lib.amr.forest import AMRForest

    header, flags_input, _ = get_metadata(str(path))
    root_shape = np.ascontiguousarray(
        header["domain_nx"] // header["block_nx"], dtype=np.int64
    )
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    flags = np.ascontiguousarray(flags_input, dtype=np.bool_)
    forest = refined_forest(
        root_shape, coord_to_rank, rank_to_coord, flags
    )
    fst_args = conformance_args(root_shape, coord_to_rank, rank_to_coord, forest)
    rel_forest_args = relation_forest_args(root_shape, coord_to_rank, forest)
    validate_refined_forest_arrays(*fst_args)
    validate_refined_all_touch_2to1(*rel_forest_args)
    leaf_ids = np.arange(forest.leaf_node_ids.size, dtype=np.int64)

    all_outputs = output_arrays(leaf_ids.size, ALL_DIRECTIONS.shape[0])
    all_args = (*rel_forest_args, leaf_ids, ALL_DIRECTIONS, *all_outputs)
    fill_balanced_refined_relations(*all_args)
    all_seconds = median_seconds(
        lambda: fill_balanced_refined_relations(*all_args), repeats
    )
    all_fill_seconds = median_seconds(
        lambda: fill_balanced_refined_relations_unchecked(*all_args), repeats
    )
    all_allocating_seconds = median_seconds(
        lambda: balanced_refined_relations(
            *rel_forest_args, leaf_ids, ALL_DIRECTIONS
        ),
        repeats,
    )

    subset_measurements = {}
    for name, directions in (
        ("faces", FACE_DIRECTIONS),
        ("edges_corners", EDGE_CORNER_DIRECTIONS),
    ):
        outputs = output_arrays(leaf_ids.size, directions.shape[0])
        arguments = (*rel_forest_args, leaf_ids, directions, *outputs)
        fill_balanced_refined_relations(*arguments)
        seconds = median_seconds(
            lambda arguments=arguments: fill_balanced_refined_relations(*arguments),
            repeats,
        )
        subset_measurements[name] = {
            "directions": int(directions.shape[0]),
            "records": int(leaf_ids.size * directions.shape[0]),
            "seconds": seconds,
            "output_bytes": int(sum(output.nbytes for output in outputs)),
        }

    bounded_capacity = min(chunk_leaves, leaf_ids.size)
    bounded_outputs = output_arrays(
        bounded_capacity, ALL_DIRECTIONS.shape[0]
    )

    def bounded_traversal() -> None:
        for start in range(0, leaf_ids.size, bounded_capacity):
            stop = min(start + bounded_capacity, leaf_ids.size)
            count = stop - start
            fill_balanced_refined_relations(
                *rel_forest_args,
                leaf_ids[start:stop],
                ALL_DIRECTIONS,
                *(output[:count] for output in bounded_outputs),
            )

    bounded_traversal()
    bounded_seconds = median_seconds(bounded_traversal, repeats)

    def lifecycle() -> None:
        validate_refined_forest_arrays(*fst_args)
        validate_refined_all_touch_2to1(*rel_forest_args)
        fill_balanced_refined_relations(*all_args)

    lifecycle_seconds = median_seconds(lifecycle, repeats)

    current_flags = flags.astype(np.int32)

    def current_constructor() -> None:
        AMRForest(3, *tuple(int(value) for value in root_shape), current_flags)

    current_seconds = median_seconds(current_constructor, repeats)
    current = AMRForest(
        3, *tuple(int(value) for value in root_shape), current_flags
    )
    current_arrays = (
        np.asarray(current.neighbor_type),
        np.asarray(current.neighbor_index),
        np.asarray(current.neighbor_children),
    )
    current_copy_seconds = median_seconds(
        lambda: tuple(value.copy() for value in current_arrays), repeats
    )
    current_kinds = current_arrays[0][:, NONCENTER_COLUMNS]
    kinds, masks, counts, _ = all_outputs
    supported = (masks == 0) | (kinds == RELATION_PHYSICAL)
    mixed = (masks != 0) & (kinds != RELATION_PHYSICAL)
    if not np.array_equal(kinds[supported], current_kinds[supported]):
        raise AssertionError("supported relation kinds disagree with current")
    current_mixed_physical = int(
        np.count_nonzero(current_kinds[mixed] == RELATION_PHYSICAL)
    )
    if current_mixed_physical != int(np.count_nonzero(mixed)):
        raise AssertionError("current mixed records are not uniformly physical")

    face_sources = np.repeat(leaf_ids, FACE_DIRECTIONS.shape[0])
    face_directions = np.tile(FACE_DIRECTIONS, (leaf_ids.size, 1))
    face_contact_seconds = median_seconds(
        lambda: refined_contact_targets(
            *rel_forest_args,
            face_sources,
            face_directions,
        ),
        repeats,
    )
    original_nonzero = np.count_nonzero(ALL_DIRECTIONS, axis=1)[None, :]
    reduced_nonzero = original_nonzero - BIT_COUNTS[masks]
    reduced_face_records = int(np.count_nonzero(reduced_nonzero == 1))

    record_count = leaf_ids.size * ALL_DIRECTIONS.shape[0]
    kind_counts = {
        str(kind): int(np.count_nonzero(kinds == kind))
        for kind in (
            RELATION_PHYSICAL,
            RELATION_COARSER,
            RELATION_SAME,
            RELATION_FINER,
        )
    }
    mixed_kind_counts = {
        str(kind): int(np.count_nonzero(mixed & (kinds == kind)))
        for kind in (RELATION_COARSER, RELATION_SAME, RELATION_FINER)
    }
    return {
        "path": str(path),
        "staggered": bool(header["staggered"]),
        "nodes": int(forest.node_levels.size),
        "leaves": int(leaf_ids.size),
        "max_level": int(forest.max_level),
        "records": int(record_count),
        "all_seconds": all_seconds,
        "all_fill_seconds": all_fill_seconds,
        "all_allocating_seconds": all_allocating_seconds,
        "million_records_per_second": record_count / all_seconds / 1.0e6,
        "source_occurrences": int(np.sum(counts, dtype=np.int64)),
        "output_bytes": int(sum(output.nbytes for output in all_outputs)),
        "kind_counts": kind_counts,
        "nonzero_physical_masks": int(np.count_nonzero(masks)),
        "pure_physical_records": int(np.count_nonzero(kinds == RELATION_PHYSICAL)),
        "mixed_records": int(np.count_nonzero(mixed)),
        "current_mixed_reported_physical": current_mixed_physical,
        "mixed_kind_counts": mixed_kind_counts,
        "subset_measurements": subset_measurements,
        "bounded_capacity": bounded_capacity,
        "bounded_seconds": bounded_seconds,
        "bounded_workspace_bytes": int(
            sum(output.nbytes for output in bounded_outputs)
        ),
        "lifecycle_seconds": lifecycle_seconds,
        "current_constructor_seconds": current_seconds,
        "current_cached_copy_seconds": current_copy_seconds,
        "current_retained_bytes": int(sum(value.nbytes for value in current_arrays)),
        "top003_candidate_bytes": int(54 * leaf_ids.size),
        "face_contact_build_seconds": face_contact_seconds,
        "reduced_face_records": reduced_face_records,
        "reduced_face_fraction": reduced_face_records / record_count,
        "idealized_earliest_cache_break_even_pass": 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-level", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--reference-leaf-limit", type=int, default=100)
    parser.add_argument("--chunk-leaves", type=int, default=1024)
    parser.add_argument("--dat", type=Path)
    args = parser.parse_args()
    report = {
        "capability": "REL-001",
        "repeats": args.repeats,
        "uniform": [
            measure_uniform(
                level, args.repeats, args.reference_leaf_limit
            )
            for level in range(1, args.max_level + 1)
        ],
        "real_dat": (
            measure_dat(args.dat, args.repeats, args.chunk_leaves)
            if args.dat
            else None
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
