"""Standard LOC/SAM/RPS refined repeated-point benchmark."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import sysconfig
import time
import tracemalloc

import Cython
import numpy as np

import simesh_rewrite.repeated_sampling as repeated_module
from simesh_rewrite.balance import validate_refined_all_touch_2to1
from simesh_rewrite.blockio import (
    array_block_reader,
    array_block_writer,
    make_block_reader,
)
from simesh_rewrite.forest import refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.point_location import fill_refined_point_leaf_ids
from simesh_rewrite.refined_geometry import refined_leaf_geometry
from simesh_rewrite.refined_halo import execute_selected_refined_halos_from_blocks
from simesh_rewrite.refined_sampling import (
    sample_refined_trilinear_point_groups,
    sample_refined_zero_order_point_groups,
)
from simesh_rewrite.repeated_sampling import (
    _make_point_plan,
    execute_refined_trilinear_points_from_blocks,
    execute_refined_zero_order_points_from_blocks,
)
from simesh_rewrite.storage import gather_blocks_into


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def command_output(*args: str) -> str:
    completed = subprocess.run(
        args,
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def current_rss_bytes() -> int | None:
    try:
        completed = subprocess.run(
            ("ps", "-o", "rss=", "-p", str(os.getpid())),
            check=True,
            capture_output=True,
            text=True,
        )
        return int(completed.stdout.strip()) * 1024
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def environment_record() -> dict:
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


def summary(samples: list[float]) -> dict:
    return {
        "raw_repetitions": samples,
        "median": statistics.median(samples),
        "pstdev": statistics.pstdev(samples),
        "minimum": min(samples),
        "maximum": max(samples),
    }


def timed(operation, repeats: int, warmups: int = 1) -> dict:
    for _ in range(warmups):
        operation()
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started)
    return summary(samples)


@dataclass
class CountingReaderState:
    backing: np.ndarray
    calls: list[np.ndarray] = field(default_factory=list)
    bytes_read: int = 0
    callback_seconds: float = 0.0

    def reset(self) -> None:
        self.calls.clear()
        self.bytes_read = 0
        self.callback_seconds = 0.0


def counting_read(
    state: CountingReaderState,
    source_lower: np.ndarray,
    source_upper: np.ndarray,
    block_ids: np.ndarray,
    field_ids: np.ndarray,
    destination: np.ndarray,
    destination_lower: np.ndarray,
) -> None:
    started = time.perf_counter()
    state.calls.append(block_ids.copy())
    gather_blocks_into(
        state.backing,
        source_lower,
        source_upper,
        block_ids,
        field_ids,
        destination,
        destination_lower,
    )
    extent = source_upper - source_lower
    state.bytes_read += (
        int(block_ids.size)
        * int(field_ids.size)
        * int(np.prod(extent, dtype=np.int64))
        * 8
    )
    state.callback_seconds += time.perf_counter() - started


def make_counting_reader(state: CountingReaderState):
    return make_block_reader(
        state,
        state.backing.shape,
        counting_read,
        memory_arrays=(state.backing,),
    )


def make_case(point_count: int) -> dict:
    root_shape = i3(4, 4, 4)
    block_counts = i3(4, 4, 4)
    domain_counts = root_shape * block_counts
    domain_lower = np.asarray((-2.0, 1.0, 10.0), dtype=np.float64)
    domain_upper = np.asarray((6.0, 9.0, 18.0), dtype=np.float64)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)

    flags: list[bool] = []
    refined_root = (1, 1, 1)
    for coordinate in rank_to_coord:
        refined = tuple(int(value) for value in coordinate) == refined_root
        flags.append(not refined)
        if refined:
            flags.extend([True] * 8)
    forest = refined_forest(
        root_shape,
        coord_to_rank,
        rank_to_coord,
        np.asarray(flags, dtype=np.bool_),
    )
    assert validate_refined_forest_arrays(
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
    ) == forest.max_level
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

    leaf_count = int(forest.leaf_node_ids.size)
    all_leaf_ids = np.arange(leaf_count, dtype=np.int64)
    bounds, spacing = refined_leaf_geometry(
        domain_lower,
        domain_upper,
        root_shape,
        domain_counts,
        block_counts,
        forest.node_levels,
        forest.node_coords,
        forest.leaf_node_ids,
        all_leaf_ids,
    )

    source_fields = 3
    x, y, z = np.indices(tuple(int(v) for v in block_counts), dtype=np.float64)
    backing = np.empty(
        (leaf_count, source_fields, *tuple(int(v) for v in block_counts)),
        dtype=np.float64,
    )
    for leaf in range(leaf_count):
        for field_index in range(source_fields):
            backing[leaf, field_index] = (
                leaf * 100000.0
                + field_index * 10000.0
                + x * 100.0
                + y * 10.0
                + z
            )

    clustered_owners = np.asarray((0, 1, 2, leaf_count - 1), dtype=np.int64)
    scattered_owners = all_leaf_ids
    clustered_sequence = np.resize(clustered_owners, point_count)
    fractions = np.asarray(
        ((0.25, 0.75, 3.75), (3.75, 0.25, 0.75), (0.75, 3.75, 0.25)),
        dtype=np.float64,
    )

    def points_for(owner_sequence: np.ndarray) -> np.ndarray:
        points = np.empty((owner_sequence.size, 3), dtype=np.float64)
        for row, leaf in enumerate(owner_sequence):
            points[row] = bounds[leaf, 0] + fractions[row % 3] * spacing[leaf]
        return np.ascontiguousarray(points)

    return {
        "root_shape": root_shape,
        "block_counts": block_counts,
        "domain_counts": domain_counts,
        "domain_lower": domain_lower,
        "domain_upper": domain_upper,
        "coord_to_rank": coord_to_rank,
        "forest": forest,
        "leaf_count": leaf_count,
        "all_leaf_ids": all_leaf_ids,
        "backing": backing,
        "queries": {
            "clustered": points_for(clustered_sequence),
            "scattered": points_for(scattered_owners),
        },
    }


def locator_arguments(case: dict, points: np.ndarray, output: np.ndarray) -> tuple:
    forest = case["forest"]
    return (
        case["domain_lower"],
        case["domain_upper"],
        case["root_shape"],
        case["domain_counts"],
        case["block_counts"],
        forest.max_level,
        case["coord_to_rank"],
        forest.root_node_ids,
        forest.child_node_ids,
        forest.node_leaf_ids,
        points,
        output,
    )


def prepare_resident(case: dict, points: np.ndarray, field_ids: np.ndarray) -> dict:
    forest = case["forest"]
    owners = np.empty(points.shape[0], dtype=np.int64)
    fill_refined_point_leaf_ids(*locator_arguments(case, points, owners))
    plan = _make_point_plan(owners)
    zero_payload = np.ascontiguousarray(
        case["backing"][plan.owner_leaf_ids][:, field_ids]
    )
    padded_shape = tuple(int(value) + 2 for value in case["block_counts"])
    completed = np.empty(
        (case["leaf_count"], field_ids.size, *padded_shape), dtype=np.float64
    )
    modes = np.zeros((field_ids.size, 6), dtype=np.uint8)
    normals = i3(-1, -1, -1)
    one = i3(1, 1, 1)
    execute_selected_refined_halos_from_blocks(
        array_block_reader(case["backing"]),
        array_block_writer(completed),
        case["all_leaf_ids"],
        field_ids,
        case["root_shape"],
        case["coord_to_rank"],
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
        one,
        one,
        modes,
        normals,
        case["leaf_count"],
    )
    tri_payload = np.ascontiguousarray(completed[plan.owner_leaf_ids])
    return {
        "owners": owners,
        "plan": plan,
        "zero_payload": zero_payload,
        "tri_payload": tri_payload,
        "padded_shape": padded_shape,
        "modes": modes,
        "normals": normals,
    }


def sampler_measurements(
    case: dict,
    points: np.ndarray,
    field_ids: np.ndarray,
    resident: dict,
    repeats: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    forest = case["forest"]
    plan = resident["plan"]
    zero = i3(0, 0, 0)
    one = i3(1, 1, 1)
    block = case["block_counts"]
    zero_values = np.empty((points.shape[0], field_ids.size), dtype=np.float64)
    tri_values = np.empty_like(zero_values)

    zero_args = (
        resident["zero_payload"],
        zero,
        block,
        zero,
        block,
        plan.owner_leaf_ids,
        case["domain_lower"],
        case["domain_upper"],
        case["domain_counts"],
        block,
        forest.node_levels,
        forest.node_coords,
        forest.leaf_node_ids,
        points,
        plan.grouped_point_indices,
        plan.owner_offsets,
        zero_values,
    )
    tri_args = (
        resident["tri_payload"],
        zero,
        i3(*resident["padded_shape"]),
        one,
        one + block,
        plan.owner_leaf_ids,
        case["domain_lower"],
        case["domain_upper"],
        case["domain_counts"],
        block,
        forest.node_levels,
        forest.node_coords,
        forest.leaf_node_ids,
        points,
        plan.grouped_point_indices,
        plan.owner_offsets,
        tri_values,
    )
    zero_timing = timed(
        lambda: sample_refined_zero_order_point_groups(*zero_args), repeats
    )
    tri_timing = timed(
        lambda: sample_refined_trilinear_point_groups(*tri_args), repeats
    )
    values = int(points.shape[0] * field_ids.size)
    return (
        {
            "zero": {
                **zero_timing,
                "million_field_values_per_second": values
                / zero_timing["median"]
                / 1e6,
            },
            "trilinear": {
                **tri_timing,
                "million_field_values_per_second": values
                / tri_timing["median"]
                / 1e6,
            },
        },
        zero_values.copy(),
        tri_values.copy(),
    )


def rps_arguments(
    case: dict,
    reader,
    points: np.ndarray,
    field_ids: np.ndarray,
    capacity: int,
    output: np.ndarray,
    *,
    trilinear: bool,
    resident: dict,
) -> tuple:
    forest = case["forest"]
    common = (
        reader,
        points,
        field_ids,
        case["domain_lower"],
        case["domain_upper"],
        case["root_shape"],
        case["domain_counts"],
        case["block_counts"],
        forest.max_level,
        case["coord_to_rank"],
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
    )
    if trilinear:
        return (
            *common,
            resident["modes"],
            resident["normals"],
            capacity,
            output,
        )
    return (*common, capacity, output)


def measure_time_to_first_sample(operation_name: str, operation) -> float:
    original = getattr(repeated_module, operation_name)
    first: list[float | None] = [None]
    started = time.perf_counter()

    def wrapped(*args):
        if first[0] is None:
            first[0] = time.perf_counter() - started
        return original(*args)

    setattr(repeated_module, operation_name, wrapped)
    try:
        operation()
    finally:
        setattr(repeated_module, operation_name, original)
    if first[0] is None:
        raise AssertionError("nonempty benchmark did not call the sampler")
    return float(first[0])


def composed_measurement(
    case: dict,
    points: np.ndarray,
    field_ids: np.ndarray,
    resident: dict,
    expected: np.ndarray,
    capacity: int,
    repeats: int,
    *,
    trilinear: bool,
) -> dict:
    state = CountingReaderState(case["backing"])
    reader = make_counting_reader(state)
    output = np.empty_like(expected)
    execute = (
        execute_refined_trilinear_points_from_blocks
        if trilinear
        else execute_refined_zero_order_points_from_blocks
    )
    args = rps_arguments(
        case,
        reader,
        points,
        field_ids,
        capacity,
        output,
        trilinear=trilinear,
        resident=resident,
    )

    samples: list[float] = []
    callback_samples: list[float] = []
    stats = None
    for _ in range(1 + repeats):
        state.reset()
        output.fill(np.nan)
        started = time.perf_counter()
        current_stats = execute(*args)
        elapsed = time.perf_counter() - started
        if stats is None:
            stats = current_stats
        elif current_stats != stats:
            raise AssertionError("execution stats changed across repetitions")
        if len(samples) < repeats and _ > 0:
            samples.append(elapsed)
            callback_samples.append(state.callback_seconds)
    if not np.array_equal(output.view(np.uint64), expected.view(np.uint64)):
        raise AssertionError("bounded and resident point values differ")

    operation_name = (
        "sample_refined_trilinear_point_groups_unchecked"
        if trilinear
        else "sample_refined_zero_order_point_groups_unchecked"
    )
    state.reset()
    ttf = measure_time_to_first_sample(operation_name, lambda: execute(*args))
    nonempty_calls = [value for value in state.calls if value.size]
    timing = summary(samples)
    callback_timing = summary(callback_samples)
    values = int(points.shape[0] * field_ids.size)
    owner_count = int(stats.owner_count)
    block_bytes = int(field_ids.size * np.prod(case["block_counts"]) * 8)
    return {
        "mode": "trilinear" if trilinear else "zero",
        "capacity": capacity,
        "stats": {name: int(getattr(stats, name)) for name in stats._fields},
        "timing": timing,
        "reader_callback_timing": callback_timing,
        "time_to_first_sample_seconds": ttf,
        "million_field_values_per_second": values / timing["median"] / 1e6,
        "owner_reuse_count": int(stats.inside_point_count - owner_count),
        "requested_owner_bytes": owner_count * block_bytes,
        "read_payload_bytes": state.bytes_read,
        "support_load_amplification": (
            stats.selected_load_count / owner_count if owner_count else 0.0
        ),
        "nonempty_reader_ids": [value.tolist() for value in nonempty_calls],
        "bitwise_resident_equal": True,
    }


def query_measurement(case: dict, name: str, points: np.ndarray, config: dict) -> dict:
    field_ids = np.asarray((2, 0, 2), dtype=np.int64)
    repeats = config["repeats"]
    owners = np.empty(points.shape[0], dtype=np.int64)
    locate = lambda: fill_refined_point_leaf_ids(
        *locator_arguments(case, points, owners)
    )
    locator_timing = timed(locate, repeats)
    single_owner = np.empty(1, dtype=np.int64)
    single_args = locator_arguments(case, points[:1], single_owner)
    single_timing = timed(
        lambda: fill_refined_point_leaf_ids(*single_args),
        config["single_repeats"],
        warmups=3,
    )
    locate()
    plan_timing = timed(lambda: _make_point_plan(owners), repeats)
    resident = prepare_resident(case, points, field_ids)
    sampler, expected_zero, expected_tri = sampler_measurements(
        case, points, field_ids, resident, repeats
    )

    zero_capacities = (config["zero_capacity"], case["leaf_count"])
    tri_capacities = (57, case["leaf_count"])
    composed = []
    for capacity in zero_capacities:
        composed.append(
            composed_measurement(
                case,
                points,
                field_ids,
                resident,
                expected_zero,
                capacity,
                repeats,
                trilinear=False,
            )
        )
    for capacity in tri_capacities:
        composed.append(
            composed_measurement(
                case,
                points,
                field_ids,
                resident,
                expected_tri,
                capacity,
                repeats,
                trilinear=True,
            )
        )
    owner_levels = case["forest"].node_levels[
        case["forest"].leaf_node_ids[np.unique(owners)]
    ]
    return {
        "query": name,
        "point_count": int(points.shape[0]),
        "owner_count": int(np.unique(owners).size),
        "owner_level_min": int(owner_levels.min()),
        "owner_level_max": int(owner_levels.max()),
        "locator": {
            **locator_timing,
            "million_points_per_second": points.shape[0]
            / locator_timing["median"]
            / 1e6,
            "single_point_latency": single_timing,
        },
        "group_plan": plan_timing,
        "resident_samplers": sampler,
        "bounded_compositions": composed,
    }


def traced_representative_execution(case: dict) -> dict:
    points = case["queries"]["clustered"]
    field_ids = np.asarray((2, 0, 2), dtype=np.int64)
    output = np.empty((points.shape[0], field_ids.size), dtype=np.float64)
    modes = np.zeros((field_ids.size, 6), dtype=np.uint8)
    normals = i3(-1, -1, -1)
    forest = case["forest"]
    arguments = (
        array_block_reader(case["backing"]),
        points,
        field_ids,
        case["domain_lower"],
        case["domain_upper"],
        case["root_shape"],
        case["domain_counts"],
        case["block_counts"],
        forest.max_level,
        case["coord_to_rank"],
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
        modes,
        normals,
        57,
        output,
    )
    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    stats = execute_refined_trilinear_points_from_blocks(*arguments)
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
        "contracted_managed_array_bytes": int(stats.managed_array_bytes),
    }


def comparison_metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    difference = np.abs(actual - expected)
    relative = difference / np.maximum(np.abs(expected), 1.0)
    actual_bits = actual.view(np.uint64)
    expected_bits = expected.view(np.uint64)
    sign = np.uint64(1 << 63)
    actual_ordered = np.where(
        actual_bits & sign, ~actual_bits, actual_bits | sign
    )
    expected_ordered = np.where(
        expected_bits & sign, ~expected_bits, expected_bits | sign
    )
    ulp = np.where(
        actual_ordered >= expected_ordered,
        actual_ordered - expected_ordered,
        expected_ordered - actual_ordered,
    )
    return {
        "bit_mismatch_count": int(np.count_nonzero(actual_bits != expected_bits)),
        "max_absolute_error": float(np.max(difference, initial=0.0)),
        "max_relative_error": float(np.max(relative, initial=0.0)),
        "max_ulp_error": int(np.max(ulp, initial=np.uint64(0))),
    }


def safe_current_refined_comparison() -> dict:
    from simesh.utils.lib.amr.forest import AMRForest
    from simesh.utils.lib.amr.mesh import AMRMesh

    root_shape = i3(2, 2, 2)
    block = i3(4, 4, 4)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    flags: list[bool] = []
    for coordinate in rank_to_coord:
        refined = tuple(int(value) for value in coordinate) == (0, 0, 0)
        flags.append(not refined)
        if refined:
            flags.extend([True] * 8)
    forest = refined_forest(
        root_shape,
        coord_to_rank,
        rank_to_coord,
        np.asarray(flags, dtype=np.bool_),
    )
    leaf_count = int(forest.leaf_node_ids.size)
    backing = np.empty((leaf_count, 2, 4, 4, 4), dtype=np.float64)
    x, y, z = np.indices((4, 4, 4), dtype=np.float64)
    for leaf in range(leaf_count):
        for field_index in range(2):
            backing[leaf, field_index] = (
                leaf * 100000.0
                + field_index * 10000.0
                + x * 100.0
                + y * 10.0
                + z
                + 1.0
            )

    lower = np.zeros(3, dtype=np.float64)
    upper = np.full(3, 2.0, dtype=np.float64)
    current_forest = AMRForest(
        3, 2, 2, 2, np.asarray(flags, dtype=np.int32)
    )
    current_mesh = AMRMesh(
        3,
        block.astype(np.uint32),
        (root_shape * block).astype(np.uint32),
        lower,
        upper,
        np.uint32(2),
        np.uint32(2),
        current_forest,
    )
    current_mesh.load_interior_data(backing)
    current_mesh.apply_ghost_cells()
    output_counts = np.asarray((12, 10, 8), dtype=np.uint32)
    output_shape = tuple(int(value) for value in output_counts)
    current_zero = np.empty((2, *output_shape), dtype=np.float64)
    current_tri = np.empty_like(current_zero)
    current_mesh.uniform_grid_zero_order(
        backing, current_zero, output_counts, lower, upper
    )
    current_mesh.uniform_grid_linear(
        current_tri,
        output_counts,
        lower,
        upper,
        np.asarray((0, 1), dtype=np.uint32),
    )

    spacing = (upper - lower) / output_counts
    points = np.empty((int(np.prod(output_counts)), 3), dtype=np.float64)
    for row, index in enumerate(np.ndindex(output_shape)):
        for axis in range(3):
            factor = np.float64(index[axis]) + np.float64(0.5)
            points[row, axis] = np.float64(lower[axis]) + factor * np.float64(
                spacing[axis]
            )
    fields = np.asarray((0, 1), dtype=np.int64)
    common = (
        array_block_reader(backing),
        points,
        fields,
        lower,
        upper,
        root_shape,
        root_shape * block,
        block,
        forest.max_level,
        coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
    )
    rewrite_zero = np.empty((points.shape[0], 2), dtype=np.float64)
    rewrite_tri = np.empty_like(rewrite_zero)
    execute_refined_zero_order_points_from_blocks(
        *common, leaf_count, rewrite_zero
    )
    execute_refined_trilinear_points_from_blocks(
        *common,
        np.zeros((2, 6), dtype=np.uint8),
        i3(-1, -1, -1),
        leaf_count,
        rewrite_tri,
    )
    rewrite_zero_grid = rewrite_zero.reshape((*output_shape, 2)).transpose(
        3, 0, 1, 2
    )
    rewrite_tri_grid = rewrite_tri.reshape((*output_shape, 2)).transpose(
        3, 0, 1, 2
    )
    return {
        "scope": "dyadic refined, continuous boundaries, non-tie centers",
        "zero_order": comparison_metrics(rewrite_zero_grid, current_zero),
        "trilinear": comparison_metrics(rewrite_tri_grid, current_tri),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "standard"), default="standard")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    profiles = {
        "smoke": {
            "point_count": 384,
            "repeats": 2,
            "single_repeats": 31,
            "zero_capacity": 8,
        },
        "standard": {
            "point_count": 4096,
            "repeats": 7,
            "single_repeats": 301,
            "zero_capacity": 8,
        },
    }
    config = profiles[args.profile]
    rss_before = current_rss_bytes()
    case = make_case(config["point_count"])
    queries = [
        query_measurement(case, name, points, config)
        for name, points in case["queries"].items()
    ]
    traced = traced_representative_execution(case)
    rss_after = current_rss_bytes()
    record = {
        "capability_group": "Bounded Refined Repeated Point Sampling",
        "capabilities": ["LOC-001", "SAM-004", "SAM-005", "RPS-001"],
        "profile": args.profile,
        "configuration": config,
        "environment": environment_record(),
        "forest": {
            "root_shape": case["root_shape"].tolist(),
            "block_shape": case["block_counts"].tolist(),
            "leaf_count": case["leaf_count"],
            "max_level": case["forest"].max_level,
            "field_ids": [2, 0, 2],
        },
        "queries": queries,
        "safe_current_refined_comparison": safe_current_refined_comparison(),
        "process": {
            "rss_before_bytes": rss_before,
            "rss_after_bytes": rss_after,
            "peak_rss_bytes": peak_rss_bytes(),
            **traced,
        },
    }
    encoded = json.dumps(record, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
