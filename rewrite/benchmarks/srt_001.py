"""Selected Refined Transfer Planning group composition benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import sysconfig
import time

import Cython
import numpy as np

from simesh_rewrite._refined_support import (
    plan_balanced_refined_support_prefix_unchecked,
    plan_selected_refined_support_prefix_unchecked,
)
from simesh_rewrite.balance import validate_refined_all_touch_2to1
from simesh_rewrite.forest import refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.relations import (
    RELATION_COARSER,
    RELATION_FINER,
    balanced_refined_relations,
)
from simesh_rewrite.selected_refined_support import (
    maximum_selected_refined_support_slots,
)
from simesh_rewrite.storage import gather_blocks_into


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


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def command_output(*args: str) -> str:
    """Return one short environment command result without shell dispatch."""
    completed = subprocess.run(
        args,
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def environment_record() -> dict:
    """Capture the same-runner metadata required by PERFORMANCE.md."""
    return {
        "git_commit": command_output("git", "rev-parse", "HEAD"),
        "git_dirty": bool(command_output("git", "status", "--porcelain")),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "numpy": np.__version__,
        "cython": Cython.__version__,
        "compiler": sysconfig.get_config_var("CC"),
        "compiler_flags": sysconfig.get_config_var("CFLAGS"),
    }


def repetition_summary(samples: list[dict]) -> dict:
    """Summarize raw composed repetitions without discarding them."""
    return {
        name: {
            "median": statistics.median(sample[name] for sample in samples),
            "pstdev": statistics.pstdev(sample[name] for sample in samples),
            "minimum": min(sample[name] for sample in samples),
            "maximum": max(sample[name] for sample in samples),
        }
        for name in ("wall_seconds", "planning_seconds", "gather_seconds")
    }


def forest_arguments(root_shape, coord_to_rank, rank_to_coord, forest):
    conformance = (
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
    relation = (
        root_shape,
        coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
    )
    return conformance, relation


def selected_ids(forest, root_shape: np.ndarray, count: int) -> np.ndarray:
    leaf_count = int(forest.leaf_node_ids.size)
    if count == leaf_count:
        return np.arange(leaf_count, dtype=np.int64)
    node_ids = forest.leaf_node_ids
    levels = forest.node_levels[node_ids]
    coords = forest.node_coords[node_ids]
    scale = np.exp2(levels - 1).astype(np.float64)
    centers = (coords + 0.5) / (root_shape[None, :] * scale[:, None])
    distance2 = np.sum((centers - 0.5) ** 2, axis=1)
    nearest = np.argpartition(distance2, count - 1)[:count]
    return np.sort(nearest.astype(np.int64))


def traverse(
    primary_ids: np.ndarray,
    source_counts: np.ndarray,
    source_leaf_ids: np.ndarray,
    backing: np.ndarray,
    capacity: int,
    dense: bool = False,
) -> dict:
    selected = np.empty(capacity, dtype=np.int64)
    payload = np.empty((capacity, *backing.shape[1:]), dtype=np.float64)
    lower = np.zeros(3, dtype=np.int64)
    upper = np.asarray(backing.shape[2:], dtype=np.int64)
    fields = np.arange(backing.shape[1], dtype=np.int64)
    position = 0
    chunks = 0
    selected_total = 0
    planning_seconds = 0.0
    gather_seconds = 0.0
    total = np.float64(0.0)
    started_wall = time.perf_counter()
    while position < primary_ids.size:
        row_count = min(capacity, primary_ids.size - position)
        started = time.perf_counter()
        if dense:
            primary_count, selected_count = plan_balanced_refined_support_prefix_unchecked(
                position,
                source_counts[position : position + row_count],
                source_leaf_ids[position : position + row_count],
                selected,
            )
        else:
            primary_count, selected_count = plan_selected_refined_support_prefix_unchecked(
                primary_ids[position : position + row_count],
                source_counts[position : position + row_count],
                source_leaf_ids[position : position + row_count],
                selected,
            )
        planning_seconds += time.perf_counter() - started
        started = time.perf_counter()
        gather_blocks_into(
            backing,
            lower,
            upper,
            selected[:selected_count],
            fields,
            payload[:selected_count],
            lower,
        )
        gather_seconds += time.perf_counter() - started
        total += np.sum(payload[:primary_count], dtype=np.float64)
        position += primary_count
        chunks += 1
        selected_total += selected_count
    expected = np.sum(backing[primary_ids], dtype=np.float64)
    if total != expected:
        raise AssertionError("selected traversal did not consume each primary once")
    return {
        "wall_seconds": time.perf_counter() - started_wall,
        "planning_seconds": planning_seconds,
        "gather_seconds": gather_seconds,
        "chunks": chunks,
        "reader_calls": chunks,
        "selected_total": selected_total,
        "workspace_bytes": int(selected.nbytes + payload.nbytes),
    }


def measure_selection(
    relation_base,
    primary_ids: np.ndarray,
    query_shape: str,
    leaf_count: int,
    backing: np.ndarray,
    capacity: int,
    warmups: int,
    repeats: int,
) -> dict:
    started = time.perf_counter()
    relations = balanced_refined_relations(
        *relation_base, primary_ids, ALL_DIRECTIONS
    )
    relation_seconds = time.perf_counter() - started
    maximum = maximum_selected_refined_support_slots(
        primary_ids,
        leaf_count,
        relations[2],
        relations[3],
    )
    if maximum > capacity:
        raise ValueError(
            f"capacity {capacity} is below selected progress requirement {maximum}"
        )
    for _ in range(warmups):
        traverse(
            primary_ids,
            relations[2],
            relations[3],
            backing,
            capacity,
        )
    samples = [
        traverse(
            primary_ids,
            relations[2],
            relations[3],
            backing,
            capacity,
        )
        for _ in range(repeats)
    ]
    median_wall = statistics.median(sample["wall_seconds"] for sample in samples)
    result = dict(
        min(samples, key=lambda sample: abs(sample["wall_seconds"] - median_wall))
    )
    result["selected_repetitions"] = samples
    result["selected_repetition_summary"] = repetition_summary(samples)
    if query_shape == "full_domain":
        for _ in range(warmups):
            traverse(
                primary_ids,
                relations[2],
                relations[3],
                backing,
                capacity,
                dense=True,
            )
        dense_samples = [
            traverse(
                primary_ids,
                relations[2],
                relations[3],
                backing,
                capacity,
                dense=True,
            )
            for _ in range(repeats)
        ]
        dense_wall = statistics.median(
            sample["wall_seconds"] for sample in dense_samples
        )
        dense_result = min(
            dense_samples,
            key=lambda sample: abs(sample["wall_seconds"] - dense_wall),
        )
        if any(
            dense_result[name] != result[name]
            for name in ("chunks", "selected_total")
        ):
            raise AssertionError("dense and selected full traversals diverged")
        result["preserved_dense_wall_seconds"] = dense_result["wall_seconds"]
        result["selected_vs_dense_wall_ratio"] = (
            result["wall_seconds"] / dense_result["wall_seconds"]
        )
        result["dense_repetitions"] = dense_samples
        result["dense_repetition_summary"] = repetition_summary(dense_samples)
    block_bytes = int(np.prod(backing.shape[1:]) * backing.itemsize)
    requested_bytes = int(primary_ids.size * block_bytes)
    read_bytes = int(result["selected_total"] * block_bytes)
    dense_count = (
        int(primary_ids[-1] - primary_ids[0] + 1) if primary_ids.size else 0
    )
    finer_sources = int(
        np.sum(relations[2][relations[0] == RELATION_FINER], dtype=np.int64)
    )
    coarser_rows = int(np.count_nonzero(relations[0] == RELATION_COARSER))
    return result | {
        "primary_count": int(primary_ids.size),
        "query_shape": query_shape,
        "capacity": capacity,
        "maximum_progress_slots": maximum,
        "support_amplification": result["selected_total"] / primary_ids.size,
        "relation_seconds": relation_seconds,
        "relation_bytes": int(sum(value.nbytes for value in relations)),
        "requested_payload_bytes": requested_bytes,
        "read_payload_bytes": read_bytes,
        "dense_envelope_primary_count": dense_count,
        "dense_envelope_payload_bytes": dense_count * block_bytes,
        "read_vs_dense_envelope": read_bytes / (dense_count * block_bytes),
        "frp_rows": finer_sources,
        "frp_output_bytes": finer_sources * 96,
        "cwp_rows": coarser_rows,
        "cwp_output_bytes": coarser_rows * 168,
        "worst_correctness_discrepancy": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dat", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--small", type=int, default=32)
    parser.add_argument("--medium", type=int, default=512)
    parser.add_argument("--fields", type=int, default=3)
    parser.add_argument("--block", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from simesh.amrvac.datio import get_metadata

    header, flags_input, _ = get_metadata(str(args.dat))
    root_shape = np.ascontiguousarray(
        header["domain_nx"] // header["block_nx"], dtype=np.int64
    )
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    forest = refined_forest(
        root_shape,
        coord_to_rank,
        rank_to_coord,
        np.ascontiguousarray(flags_input, dtype=np.bool_),
    )
    conformance, relation_base = forest_arguments(
        root_shape, coord_to_rank, rank_to_coord, forest
    )
    validate_refined_forest_arrays(*conformance)
    validate_refined_all_touch_2to1(*relation_base)
    leaf_count = int(forest.leaf_node_ids.size)
    values = np.arange(
        leaf_count * args.fields * args.block**3, dtype=np.float64
    )
    backing = values.reshape(
        leaf_count, args.fields, args.block, args.block, args.block
    )
    counts = sorted({min(args.small, leaf_count), min(args.medium, leaf_count), leaf_count})
    report = {
        "group": "Selected Refined Transfer Planning",
        "dat": str(args.dat),
        "staggered_metadata_only": bool(header["staggered"]),
        "leaf_count": leaf_count,
        "directions": int(ALL_DIRECTIONS.shape[0]),
        "block_shape": [args.block] * 3,
        "fields": args.fields,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "environment": environment_record(),
        "selections": [
            measure_selection(
                relation_base,
                selected_ids(forest, root_shape, count),
                "full_domain" if count == leaf_count else "coherent_center_roi",
                leaf_count,
                backing,
                args.capacity,
                args.warmups,
                args.repeats,
            )
            for count in counts
        ],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
