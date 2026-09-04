"""Standard HLO/CHS cached refined-vector sampling benchmark."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import sysconfig
import tempfile
import time
import tracemalloc

import Cython
import numpy as np

import simesh_rewrite.amrvac_dat_reader as dat_reader_module
from simesh.amrvac.datio import header_template, write_datfile_from_sfc
from simesh_rewrite._point_location import (
    fill_refined_point_leaf_ids_with_hints_unchecked,
)
from simesh_rewrite.amrvac_dat import (
    bind_amrvac_v5_forest,
    read_amrvac_v5_index,
)
from simesh_rewrite.amrvac_dat_reader import make_amrvac_v5_block_reader
from simesh_rewrite.balance import validate_refined_all_touch_2to1
from simesh_rewrite.blockio import array_block_reader
from simesh_rewrite.completed_halo_sampling import (
    clear_completed_halo_sampling_session,
    make_completed_halo_sampling_session,
    sample_refined_trilinear_vectors_cached,
)
from simesh_rewrite.forest import RefinedForest, refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.hinted_location import (
    fill_refined_point_leaf_ids_with_hints,
)
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.point_location import refined_point_leaf_ids
from simesh_rewrite.refined_geometry import refined_leaf_geometry
from simesh_rewrite.repeated_sampling import (
    execute_refined_trilinear_points_from_blocks,
)


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
        "byteorder": sys.byteorder,
    }


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


def scalar_summary(values: list[float]) -> dict:
    return {
        "raw_repetitions": values,
        "median": statistics.median(values),
        "pstdev": statistics.pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def timed(operation, repeats: int, *, warmups: int = 1) -> dict:
    for _ in range(warmups):
        operation()
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started)
    return scalar_summary(samples)


class PreadCounter:
    def __init__(self) -> None:
        self.calls = 0
        self.bytes = 0
        self.header_calls = 0
        self.header_bytes = 0
        self.payload_calls = 0
        self.payload_bytes = 0

    def record(self, byte_count: int, section: str) -> None:
        self.calls += 1
        self.bytes += byte_count
        if section == "record header":
            self.header_calls += 1
            self.header_bytes += byte_count
        else:
            self.payload_calls += 1
            self.payload_bytes += byte_count

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "bytes": self.bytes,
            "header_calls": self.header_calls,
            "header_bytes": self.header_bytes,
            "payload_calls": self.payload_calls,
            "payload_bytes": self.payload_bytes,
        }


@contextmanager
def count_native_preads(counter: PreadCounter, enabled: bool):
    if not enabled:
        yield
        return
    original = dat_reader_module._pread_exact

    def wrapped(file_descriptor, byte_count, offset, *, section):
        result = original(
            file_descriptor,
            byte_count,
            offset,
            section=section,
        )
        counter.record(byte_count, section)
        return result

    dat_reader_module._pread_exact = wrapped
    try:
        yield
    finally:
        dat_reader_module._pread_exact = original


@dataclass(frozen=True)
class Case:
    name: str
    root_shape: np.ndarray
    coord_to_rank: np.ndarray
    rank_to_coord: np.ndarray
    forest: RefinedForest
    domain_lower: np.ndarray
    domain_upper: np.ndarray
    domain_counts: np.ndarray
    block_counts: np.ndarray
    field_ids: np.ndarray
    boundary_modes: np.ndarray
    normal_field_slots: np.ndarray
    backing: np.ndarray | None
    forest_flags: np.ndarray | None
    leaf_bounds: np.ndarray
    leaf_spacing: np.ndarray

    @property
    def leaf_count(self) -> int:
        return int(self.forest.leaf_node_ids.size)

    @property
    def block_shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.block_counts)


@dataclass(frozen=True)
class Query:
    name: str
    points: np.ndarray
    hints: np.ndarray
    owners: np.ndarray
    batches: tuple[slice, ...]


def synthetic_case(block: int = 8) -> Case:
    root_shape = i3(4, 4, 4)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    flags: list[bool] = []
    for coordinate in rank_to_coord:
        split = tuple(int(value) for value in coordinate) == (1, 1, 1)
        flags.append(not split)
        if split:
            flags.extend([True] * 8)
    forest_flags = np.asarray(flags, dtype=np.bool_)
    forest = refined_forest(
        root_shape, coord_to_rank, rank_to_coord, forest_flags
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
    block_counts = i3(block, block, block)
    domain_counts = np.ascontiguousarray(root_shape * block_counts, dtype=np.int64)
    domain_lower = np.zeros(3, dtype=np.float64)
    domain_upper = root_shape.astype(np.float64)
    leaf_ids = np.arange(forest.leaf_node_ids.size, dtype=np.int64)
    bounds, spacing = refined_leaf_geometry(
        domain_lower,
        domain_upper,
        root_shape,
        domain_counts,
        block_counts,
        forest.node_levels,
        forest.node_coords,
        forest.leaf_node_ids,
        leaf_ids,
    )

    x, y, z = np.indices((block, block, block), dtype=np.float64)
    backing = np.empty(
        (forest.leaf_node_ids.size, 3, block, block, block), dtype=np.float64
    )
    for leaf in range(backing.shape[0]):
        for field in range(backing.shape[1]):
            backing[leaf, field] = (
                100000.0 * leaf
                + 10000.0 * field
                + 100.0 * x
                + 10.0 * y
                + z
                + 1.0
            )
    modes = np.zeros((3, 6), dtype=np.uint8)
    modes[0, 0] = 2
    modes[1, 2] = 3
    modes[2, 4] = 1
    return Case(
        "synthetic-balanced-refined-v5",
        root_shape,
        coord_to_rank,
        rank_to_coord,
        forest,
        domain_lower,
        domain_upper,
        domain_counts,
        block_counts,
        i3(2, 0, 2),
        modes,
        i3(-1, 1, -1),
        backing,
        forest_flags,
        bounds,
        spacing,
    )


def write_synthetic_v5(path: Path, case: Case) -> None:
    if case.backing is None or case.forest_flags is None:
        raise ValueError("synthetic fixture requires resident data and forest flags")
    header = header_template.copy()
    header.update(
        {
            "nw": int(case.backing.shape[1]),
            "w_names": ["b1", "b2", "b3"],
            "ndir": 3,
            "ndim": 3,
            "levmax": case.forest.max_level,
            "nleafs": case.leaf_count,
            "nparents": int(np.count_nonzero(~case.forest_flags)),
            "xmin": case.domain_lower.copy(),
            "xmax": case.domain_upper.copy(),
            "domain_nx": case.domain_counts.astype(np.int32),
            "block_nx": case.block_counts.astype(np.int32),
            "periodic": np.zeros(3, dtype=np.bool_),
            "geometry": "Cartesian_3D",
            "staggered": False,
        }
    )
    leaf_nodes = case.forest.leaf_node_ids
    tree = (
        case.forest.node_levels[leaf_nodes].astype(np.int32),
        (case.forest.node_coords[leaf_nodes] + 1).astype(np.int32),
        np.zeros(case.leaf_count, dtype=np.int64),
    )
    write_datfile_from_sfc(
        str(path),
        case.backing,
        header,
        case.forest_flags.astype(np.int32),
        tree,
        overwrite=True,
    )


def case_from_v5(index, binding, field_ids: np.ndarray, name: str) -> Case:
    forest = binding.forest
    leaf_ids = np.arange(index.leaf_count, dtype=np.int64)
    bounds, spacing = refined_leaf_geometry(
        index.domain_lower,
        index.domain_upper,
        binding.root_shape,
        index.domain_cell_counts,
        index.block_cell_counts,
        forest.node_levels,
        forest.node_coords,
        forest.leaf_node_ids,
        leaf_ids,
    )
    return Case(
        name,
        binding.root_shape,
        binding.coord_to_rank,
        binding.rank_to_coord,
        forest,
        index.domain_lower,
        index.domain_upper,
        index.domain_cell_counts,
        index.block_cell_counts,
        field_ids,
        np.zeros((3, 6), dtype=np.uint8),
        i3(-1, -1, -1),
        None,
        None,
        bounds,
        spacing,
    )


def locate(case: Case, points: np.ndarray) -> np.ndarray:
    return refined_point_leaf_ids(
        case.domain_lower,
        case.domain_upper,
        case.root_shape,
        case.domain_counts,
        case.block_counts,
        case.forest.max_level,
        case.coord_to_rank,
        case.forest.root_node_ids,
        case.forest.child_node_ids,
        case.forest.node_leaf_ids,
        points,
    )


def batch_slices(point_count: int, batch_size: int) -> tuple[slice, ...]:
    return tuple(
        slice(first, min(first + batch_size, point_count))
        for first in range(0, point_count, batch_size)
    )


def last_owner_hints(owners: np.ndarray) -> np.ndarray:
    hints = np.empty_like(owners)
    if owners.size:
        hints[0] = -1
        hints[1:] = owners[:-1]
    return hints


def synthetic_queries(case: Case, point_count: int, batch_size: int) -> tuple[Query, Query]:
    coherent = np.empty((point_count, 3), dtype=np.float64)
    coherent[:, 0] = np.linspace(0.02, 3.98, point_count)
    coherent[:, 1] = 1.25
    coherent[:, 2] = 1.25
    coherent_owners = locate(case, coherent)
    if np.unique(coherent_owners).size != 5:
        raise AssertionError("coherent fixture no longer has five owners")

    eligible = [
        leaf
        for leaf in range(case.leaf_count)
        if np.all(case.leaf_bounds[leaf, 0] > case.domain_lower)
        and np.all(case.leaf_bounds[leaf, 1] < case.domain_upper)
    ]
    chosen = [eligible[0]]
    while len(chosen) < 4:
        chosen.append(
            max(
                (leaf for leaf in eligible if leaf not in chosen),
                key=lambda leaf: min(
                    float(
                        np.linalg.norm(
                            case.leaf_bounds[leaf].mean(axis=0)
                            - case.leaf_bounds[other].mean(axis=0)
                        )
                    )
                    for other in chosen
                ),
            )
        )
    divergent = np.empty((point_count, 3), dtype=np.float64)
    for row in range(point_count):
        leaf = chosen[row % 4]
        divergent[row] = (
            case.leaf_bounds[leaf, 0] + 0.1 * case.leaf_spacing[leaf]
        )
    divergent_owners = locate(case, divergent)
    if set(map(int, divergent_owners)) != set(chosen):
        raise AssertionError("divergent fixture owner set changed")
    batches = batch_slices(point_count, batch_size)
    return (
        Query(
            "coherent-5-owner",
            coherent,
            last_owner_hints(coherent_owners),
            coherent_owners,
            batches,
        ),
        Query(
            "divergent-4-owner-revisited",
            divergent,
            last_owner_hints(divergent_owners),
            divergent_owners,
            batches,
        ),
    )


def tdm_query(case: Case, point_count: int, batch_size: int) -> Query:
    chosen = np.linspace(0, case.leaf_count - 1, 3, dtype=np.int64)
    points = np.empty((point_count, 3), dtype=np.float64)
    run = math.ceil(point_count / 3)
    center_factor = 0.5 * (case.block_counts.astype(np.float64) - 1.0)
    for row in range(point_count):
        leaf = int(chosen[min(row // run, 2)])
        points[row] = (
            case.leaf_bounds[leaf, 0]
            + center_factor * case.leaf_spacing[leaf]
        )
    owners = locate(case, points)
    return Query(
        "real-tdm-coherent-3-owner",
        points,
        last_owner_hints(owners),
        owners,
        batch_slices(point_count, batch_size),
    )


def locality_record(query: Query) -> dict:
    owners = query.owners
    transitions = int(np.count_nonzero(owners[1:] != owners[:-1])) if owners.size else 0
    runs: list[int] = []
    if owners.size:
        first = 0
        for row in range(1, owners.size):
            if owners[row] != owners[row - 1]:
                runs.append(row - first)
                first = row
        runs.append(owners.size - first)
    per_batch_owners = [
        int(np.unique(owners[batch]).size) for batch in query.batches
    ]
    return {
        "point_count": int(owners.size),
        "unique_owner_count": int(np.unique(owners).size),
        "owner_ids": np.unique(owners).tolist(),
        "owner_transition_count": transitions,
        "last_owner_hit_rate": (
            1.0 - transitions / max(1, owners.size - 1)
        ),
        "run_length_median": float(statistics.median(runs)) if runs else 0.0,
        "run_length_maximum": max(runs, default=0),
        "batch_count": len(query.batches),
        "batch_sizes": [batch.stop - batch.start for batch in query.batches],
        "owner_groups_per_batch": per_batch_owners,
        "maximum_batch_owner_working_set": max(per_batch_owners, default=0),
    }


def hlo_arguments(
    case: Case,
    points: np.ndarray,
    hints: np.ndarray,
    output: np.ndarray,
) -> tuple:
    forest = case.forest
    return (
        case.domain_lower,
        case.domain_upper,
        case.root_shape,
        case.domain_counts,
        case.block_counts,
        forest.max_level,
        case.coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
        points,
        hints,
        output,
    )


def hlo_measurement(case: Case, query: Query, config: dict) -> dict:
    checked_output = np.empty(query.points.shape[0], dtype=np.int64)
    checked_stats = None

    def checked_batch() -> None:
        nonlocal checked_stats
        current = fill_refined_point_leaf_ids_with_hints(
            *hlo_arguments(case, query.points, query.hints, checked_output)
        )
        if checked_stats is None:
            checked_stats = current
        elif current != checked_stats:
            raise AssertionError("checked HLO statistics changed")
        if not np.array_equal(checked_output, query.owners):
            raise AssertionError("checked HLO owners differ from LOC")

    base_spacing = np.empty(3, dtype=np.float64)
    for axis in range(3):
        extent = float(case.domain_upper[axis]) - float(case.domain_lower[axis])
        base_spacing[axis] = extent / float(case.domain_counts[axis])
    unchecked_output = np.empty_like(checked_output)

    def unchecked_batch() -> None:
        fill_refined_point_leaf_ids_with_hints_unchecked(
            case.domain_lower,
            case.domain_upper,
            case.domain_counts,
            case.block_counts,
            case.coord_to_rank,
            case.forest.root_node_ids,
            case.forest.node_levels,
            case.forest.node_coords,
            case.forest.child_node_ids,
            case.forest.node_leaf_ids,
            case.forest.leaf_node_ids,
            base_spacing,
            query.points,
            query.hints,
            unchecked_output,
        )
        if not np.array_equal(unchecked_output, query.owners):
            raise AssertionError("unchecked HLO owners differ from LOC")

    single_checked_output = np.empty(1, dtype=np.int64)
    single_unchecked_output = np.empty(1, dtype=np.int64)

    def checked_single() -> None:
        fill_refined_point_leaf_ids_with_hints(
            *hlo_arguments(
                case,
                query.points[:1],
                query.hints[:1],
                single_checked_output,
            )
        )

    def unchecked_single() -> None:
        fill_refined_point_leaf_ids_with_hints_unchecked(
            case.domain_lower,
            case.domain_upper,
            case.domain_counts,
            case.block_counts,
            case.coord_to_rank,
            case.forest.root_node_ids,
            case.forest.node_levels,
            case.forest.node_coords,
            case.forest.child_node_ids,
            case.forest.node_leaf_ids,
            case.forest.leaf_node_ids,
            base_spacing,
            query.points[:1],
            query.hints[:1],
            single_unchecked_output,
        )

    checked_batch_timing = timed(checked_batch, config["repeats"])
    unchecked_batch_timing = timed(unchecked_batch, config["repeats"])
    checked_single_timing = timed(
        checked_single, config["single_repeats"], warmups=3
    )
    unchecked_single_timing = timed(
        unchecked_single, config["single_repeats"], warmups=3
    )
    if not np.array_equal(checked_output, unchecked_output):
        raise AssertionError("checked and unchecked HLO owners differ")
    point_count = int(query.points.shape[0])
    return {
        "stats": {
            name: int(getattr(checked_stats, name))
            for name in checked_stats._fields
        },
        "checked_batch": {
            **checked_batch_timing,
            "million_points_per_second": point_count
            / checked_batch_timing["median"]
            / 1.0e6,
        },
        "unchecked_batch": {
            **unchecked_batch_timing,
            "million_points_per_second": point_count
            / unchecked_batch_timing["median"]
            / 1.0e6,
        },
        "checked_singleton": checked_single_timing,
        "unchecked_singleton": unchecked_single_timing,
        "checked_singleton_microseconds": 1.0e6
        * checked_single_timing["median"],
        "unchecked_singleton_microseconds": 1.0e6
        * unchecked_single_timing["median"],
        "bitwise_loc_equal": True,
    }


def rps_arguments(case: Case, reader, query: Query, output: np.ndarray) -> tuple:
    forest = case.forest
    return (
        reader,
        query.points,
        case.field_ids,
        case.domain_lower,
        case.domain_upper,
        case.root_shape,
        case.domain_counts,
        case.block_counts,
        forest.max_level,
        case.coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
        case.boundary_modes,
        case.normal_field_slots,
        min(57, case.leaf_count),
        output,
    )


def rps_measurement(
    case: Case,
    reader,
    query: Query,
    repeats: int,
    *,
    native: bool,
) -> tuple[dict, np.ndarray]:
    output = np.empty((query.points.shape[0], 3), dtype=np.float64)
    samples: list[float] = []
    preads: list[dict] = []
    stats = None
    for repetition in range(repeats + 1):
        counter = PreadCounter()
        started = time.perf_counter()
        with count_native_preads(counter, native):
            current_stats = execute_refined_trilinear_points_from_blocks(
                *rps_arguments(case, reader, query, output)
            )
        elapsed = time.perf_counter() - started
        if stats is None:
            stats = current_stats
        elif current_stats != stats:
            raise AssertionError("RPS statistics changed across repetitions")
        if repetition:
            samples.append(elapsed)
            preads.append(counter.as_dict())
    if not np.array_equal(locate(case, query.points), query.owners):
        raise AssertionError("RPS benchmark owner fixture changed")
    if any(value != preads[0] for value in preads[1:]):
        raise AssertionError("RPS pread counts changed across repetitions")
    point_count = int(query.points.shape[0])
    return (
        {
            "batching": "one public RPS call containing the complete query",
            "timing": scalar_summary(samples),
            "million_field_values_per_second": 3 * point_count
            / statistics.median(samples)
            / 1.0e6,
            "stats": {
                name: int(getattr(stats, name)) for name in stats._fields
            },
            "pread": preads[0],
            "logical_bytes_per_point": stats.selected_load_count
            * 3
            * int(np.prod(case.block_counts, dtype=np.int64))
            * 8
            / point_count,
        },
        output.copy(),
    )


def session_arguments(
    case: Case,
    reader,
    rhe_capacity: int,
    cache_budget: int,
) -> tuple:
    forest = case.forest
    return (
        reader,
        case.field_ids,
        case.domain_lower,
        case.domain_upper,
        case.domain_counts,
        case.block_counts,
        forest.max_level,
        case.root_shape,
        case.coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
        case.boundary_modes,
        case.normal_field_slots,
        rhe_capacity,
        cache_budget,
    )


_SUM_STATS = (
    "point_count",
    "inside_point_count",
    "owner_count",
    "hint_candidate_count",
    "hint_hit_count",
    "hierarchy_fallback_count",
    "cache_lookup_count",
    "cache_hit_count",
    "cache_miss_count",
    "cache_eviction_count",
    "halo_fill_count",
    "reader_call_count",
    "selected_load_count",
    "support_load_count",
    "logical_reader_bytes",
)


def aggregate_chs_stats(stats_rows: list) -> dict:
    if not stats_rows:
        raise ValueError("benchmark requires at least one batch")
    result = {
        name: sum(int(getattr(value, name)) for value in stats_rows)
        for name in _SUM_STATS
    }
    result.update(
        {
            "maximum_selected_slots": max(
                int(value.maximum_selected_slots) for value in stats_rows
            ),
            "call_managed_array_peak_bytes": max(
                int(value.call_managed_array_bytes) for value in stats_rows
            ),
            "call_managed_array_sum_bytes": sum(
                int(value.call_managed_array_bytes) for value in stats_rows
            ),
            "session_managed_array_bytes": int(
                stats_rows[0].session_managed_array_bytes
            ),
            "batch_count": len(stats_rows),
        }
    )
    if any(
        int(value.session_managed_array_bytes)
        != result["session_managed_array_bytes"]
        for value in stats_rows
    ):
        raise AssertionError("session managed bytes changed between batches")
    return result


def run_chs_sequence(session, query: Query) -> tuple[np.ndarray, np.ndarray, dict]:
    values = np.full((query.points.shape[0], 3), np.nan, dtype=np.float64)
    owners = np.full(query.points.shape[0], -9, dtype=np.int64)
    rows = []
    for batch in query.batches:
        rows.append(
            sample_refined_trilinear_vectors_cached(
                session,
                query.points[batch],
                query.hints[batch],
                values[batch],
                owners[batch],
            )
        )
    return values, owners, aggregate_chs_stats(rows)


def memory_breakdown(session) -> dict:
    state = session._state
    rhe_arrays = (
        *state.workspace_arrays,
        state.zero,
        state.one,
        state.block_shape_array,
        state.padded_shape_array,
        state.interior_upper,
        state.miss_primary_ids,
    )
    cache_metadata = state.cache_leaf_ids.nbytes + state.cache_recency.nbytes
    cache_accounted = state.cache_payload.nbytes + cache_metadata
    return {
        "cache_byte_budget": int(state.cache_byte_budget),
        "cache_capacity": int(state.cache_capacity),
        "cache_entry_bytes": int(state.cache_entry_bytes),
        "cache_payload_bytes": int(state.cache_payload.nbytes),
        "cache_metadata_bytes": int(cache_metadata),
        "cache_accounted_entry_bytes": int(cache_accounted),
        "rhe_workspace_and_fixed_execution_bytes": int(
            sum(value.nbytes for value in rhe_arrays)
        ),
        "call_independent_session_managed_bytes": int(
            state.session_managed_array_bytes
        ),
    }


def history_measurement(
    session,
    query: Query,
    expected_values: np.ndarray,
    repeats: int,
    *,
    native: bool,
    warm: bool,
) -> tuple[dict, np.ndarray, np.ndarray]:
    timing: list[float] = []
    pread_rows: list[dict] = []
    retained_stats = None
    retained_values = None
    retained_owners = None
    for _ in range(repeats):
        clear_completed_halo_sampling_session(session)
        if warm:
            run_chs_sequence(session, query)
        counter = PreadCounter()
        started = time.perf_counter()
        with count_native_preads(counter, native):
            values, owners, stats = run_chs_sequence(session, query)
        timing.append(time.perf_counter() - started)
        pread_rows.append(counter.as_dict())
        if retained_stats is None:
            retained_stats = stats
        elif stats != retained_stats:
            raise AssertionError("CHS aggregate statistics changed")
        if not np.array_equal(owners, query.owners):
            raise AssertionError("CHS owners differ from LOC/HLO")
        if not np.array_equal(
            values.view(np.uint64), expected_values.view(np.uint64)
        ):
            raise AssertionError("CHS values differ from public RPS")
        retained_values = values
        retained_owners = owners
    if any(value != pread_rows[0] for value in pread_rows[1:]):
        raise AssertionError("CHS pread counts changed across repetitions")
    point_count = int(query.points.shape[0])
    return (
        {
            "history": "warm" if warm else "cache-cleared",
            "batching": (
                f"{len(query.batches)} ordered CHS calls over stage-size batches"
            ),
            "timing": scalar_summary(timing),
            "million_field_values_per_second": 3 * point_count
            / statistics.median(timing)
            / 1.0e6,
            "aggregate_stats": retained_stats,
            "pread": pread_rows[0],
            "logical_bytes_per_point": retained_stats["logical_reader_bytes"]
            / point_count,
            "native_header_bytes_per_point": pread_rows[0]["header_bytes"]
            / point_count,
            "native_payload_bytes_per_point": pread_rows[0]["payload_bytes"]
            / point_count,
        },
        retained_values,
        retained_owners,
    )


def traced_warm_call(session, query: Query, *, native: bool) -> dict:
    clear_completed_halo_sampling_session(session)
    run_chs_sequence(session, query)
    counter = PreadCounter()
    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    with count_native_preads(counter, native):
        _, _, stats = run_chs_sequence(session, query)
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
        "call_managed_array_peak_bytes": stats["call_managed_array_peak_bytes"],
        "native_pread": counter.as_dict(),
    }


def session_measurement(
    case: Case,
    reader,
    query: Query,
    expected_values: np.ndarray,
    capacity: int,
    config: dict,
    *,
    native: bool,
) -> dict:
    entry_bytes = 3 * int(np.prod(case.block_counts + 2, dtype=np.int64)) * 8 + 16
    budget = capacity * entry_bytes
    rhe_capacity = min(57, case.leaf_count)
    create_counter = PreadCounter()
    rss_before = current_rss_bytes()
    started = time.perf_counter()
    with count_native_preads(create_counter, native):
        session = make_completed_halo_sampling_session(
            *session_arguments(case, reader, rhe_capacity, budget)
        )
    create_seconds = time.perf_counter() - started
    rss_after_create = current_rss_bytes()
    if session.cache_capacity != capacity:
        raise AssertionError("requested cache capacity was not produced")
    cleared, _, _ = history_measurement(
        session,
        query,
        expected_values,
        config["repeats"],
        native=native,
        warm=False,
    )
    warm, _, _ = history_measurement(
        session,
        query,
        expected_values,
        config["repeats"],
        native=native,
        warm=True,
    )
    trace = traced_warm_call(session, query, native=native) if native else None
    return {
        "capacity_label": "resident" if capacity == case.leaf_count else str(capacity),
        "capacity": capacity,
        "session_creation_seconds": create_seconds,
        "session_creation_pread": create_counter.as_dict(),
        "memory": memory_breakdown(session),
        "cache_cleared": cleared,
        "cache_warm_same_workload": warm,
        "traced_warm_call": trace,
        "rss_before_session_bytes": rss_before,
        "rss_after_session_create_bytes": rss_after_create,
        "bitwise_rps_equal": True,
    }


def capacity_knee(records: list[dict], query: Query) -> dict:
    zero_miss = next(
        (
            value["capacity"]
            for value in records
            if value["cache_warm_same_workload"]["aggregate_stats"][
                "cache_miss_count"
            ]
            == 0
        ),
        None,
    )
    resident = records[-1]["cache_warm_same_workload"]["timing"]["median"]
    near_resident = next(
        (
            value["capacity"]
            for value in records
            if value["cache_warm_same_workload"]["timing"]["median"]
            <= 1.1 * resident
        ),
        None,
    )
    return {
        "owner_working_set": int(np.unique(query.owners).size),
        "lowest_tested_zero_warm_miss_capacity": zero_miss,
        "lowest_tested_within_10_percent_of_resident_warm_runtime": near_resident,
    }


def query_workflow(
    case: Case,
    query: Query,
    native_reader,
    config: dict,
) -> dict:
    hlo = hlo_measurement(case, query, config)
    array_rps_report, expected_values = rps_measurement(
        case,
        array_block_reader(case.backing),
        query,
        config["repeats"],
        native=False,
    )
    native_rps_report, native_rps_values = rps_measurement(
        case,
        native_reader,
        query,
        config["repeats"],
        native=True,
    )
    if not np.array_equal(
        native_rps_values.view(np.uint64), expected_values.view(np.uint64)
    ):
        raise AssertionError("native and array public RPS values differ")

    owner_working_set = int(np.unique(query.owners).size)
    capacities = tuple(
        sorted({0, 1, 4, owner_working_set, case.leaf_count})
    )
    array_cases = [
        session_measurement(
            case,
            array_block_reader(case.backing),
            query,
            expected_values,
            capacity,
            config,
            native=False,
        )
        for capacity in capacities
    ]
    native_cases = [
        session_measurement(
            case,
            native_reader,
            query,
            expected_values,
            capacity,
            config,
            native=True,
        )
        for capacity in capacities
    ]
    for array_case, native_case in zip(array_cases, native_cases, strict=True):
        for history in ("cache_cleared", "cache_warm_same_workload"):
            if (
                array_case[history]["aggregate_stats"]
                != native_case[history]["aggregate_stats"]
            ):
                raise AssertionError("native and array CHS stats differ")
    return {
        "query": query.name,
        "execution_shapes": {
            "public_rps": "single full-query batch",
            "chs": (
                f"ordered sequence of {len(query.batches)} stage-size batches; "
                f"batch size at most {max(batch.stop - batch.start for batch in query.batches)}"
            ),
            "comparability_note": (
                "Times compare these declared batch shapes and are not the old "
                "per-point RPS probe."
            ),
        },
        "locality": locality_record(query),
        "hlo": hlo,
        "public_rps": {
            "array": array_rps_report,
            "native": native_rps_report,
            "bitwise_backend_equal": True,
        },
        "completed_halo_sessions": {
            "array": array_cases,
            "native": native_cases,
        },
        "capacity_knee": capacity_knee(native_cases, query),
    }


def tdm_workflow(path: Path, config: dict) -> dict:
    file_descriptor = os.open(path, os.O_RDONLY)
    try:
        index = read_amrvac_v5_index(file_descriptor)
        binding = bind_amrvac_v5_forest(index)
        if (
            index.staggered
            or index.geometry != "Cartesian_3D"
            or np.any(index.periodic)
        ):
            raise ValueError("tdm fixture is outside the CHS Cartesian scope")
        case = case_from_v5(index, binding, i3(4, 5, 6), "real-tdm-v5")
        query = tdm_query(case, config["tdm_points"], config["batch_size"])
        reader = make_amrvac_v5_block_reader(file_descriptor, index, binding)
        expected_report, expected_values = rps_measurement(
            case, reader, query, config["repeats"], native=True
        )
        owner_working_set = int(np.unique(query.owners).size)
        capacities = tuple(
            sorted(
                {
                    1,
                    min(4, case.leaf_count),
                    owner_working_set,
                    case.leaf_count,
                }
            )
        )
        sessions = [
            session_measurement(
                case,
                reader,
                query,
                expected_values,
                capacity,
                config,
                native=True,
            )
            for capacity in capacities
        ]
        return {
            "path": str(path),
            "file_bytes": path.stat().st_size,
            "leaf_count": case.leaf_count,
            "block_shape": list(case.block_shape),
            "field_ids": case.field_ids.tolist(),
            "query": locality_record(query),
            "public_rps": expected_report,
            "completed_halo_sessions": sessions,
            "capacity_knee": capacity_knee(sessions, query),
            "bitwise_rps_equal": True,
        }
    finally:
        os.close(file_descriptor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "standard"), default="standard")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--tdm", type=Path, default=REPOSITORY_ROOT / "data/tdm.dat"
    )
    parser.add_argument("--skip-tdm", action="store_true")
    args = parser.parse_args()
    profiles = {
        "smoke": {
            "point_count": 64,
            "batch_size": 4,
            "repeats": 2,
            "single_repeats": 31,
            "tdm": False,
            "tdm_points": 48,
        },
        "standard": {
            "point_count": 192,
            "batch_size": 4,
            "repeats": 5,
            "single_repeats": 301,
            "tdm": True,
            "tdm_points": 96,
        },
    }
    config = profiles[args.profile]
    rss_before = current_rss_bytes()
    started = time.perf_counter()
    case = synthetic_case()
    queries = synthetic_queries(
        case, config["point_count"], config["batch_size"]
    )
    with tempfile.TemporaryDirectory(prefix="simesh-chs-benchmark-") as directory:
        fixture_path = Path(directory) / "balanced-refined-vectors.dat"
        write_started = time.perf_counter()
        write_synthetic_v5(fixture_path, case)
        fixture_write_seconds = time.perf_counter() - write_started
        file_descriptor = os.open(fixture_path, os.O_RDONLY)
        try:
            metadata_started = time.perf_counter()
            index = read_amrvac_v5_index(file_descriptor)
            binding = bind_amrvac_v5_forest(index)
            metadata_seconds = time.perf_counter() - metadata_started
            native_reader = make_amrvac_v5_block_reader(
                file_descriptor, index, binding
            )
            synthetic_results = [
                query_workflow(case, query, native_reader, config)
                for query in queries
            ]
            synthetic_fixture = {
                "file_bytes": fixture_path.stat().st_size,
                "write_seconds": fixture_write_seconds,
                "metadata_bind_seconds": metadata_seconds,
                "metadata_bytes": int(index.offset_blocks),
                "reader_retained_offset_bytes": int(index.block_offsets.nbytes),
                "leaf_count": case.leaf_count,
                "max_level": case.forest.max_level,
                "root_shape": case.root_shape.tolist(),
                "block_shape": case.block_counts.tolist(),
                "field_ids": case.field_ids.tolist(),
                "cache_entry_bytes": 3
                * int(np.prod(case.block_counts + 2, dtype=np.int64))
                * 8
                + 16,
                "resident_payload_bytes": int(case.backing.nbytes),
                "native_array_query_bits_equal": True,
            }
        finally:
            os.close(file_descriptor)

    tdm = None
    tdm_skip_reason = None
    if config["tdm"] and not args.skip_tdm:
        if args.tdm.exists():
            tdm = tdm_workflow(args.tdm, config)
        else:
            tdm_skip_reason = "configured tdm fixture is unavailable"
    else:
        tdm_skip_reason = "disabled by profile or --skip-tdm"
    rss_after = current_rss_bytes()
    elapsed = time.perf_counter() - started
    record = {
        "capability_group": "Cached Refined Vector Sampling",
        "capabilities": ["HLO-001", "CHS-001"],
        "profile": args.profile,
        "configuration": config,
        "environment": environment_record(),
        "synthetic_fixture": synthetic_fixture,
        "synthetic_queries": synthetic_results,
        "real_tdm_native": tdm,
        "tdm_skip_reason": tdm_skip_reason,
        "process": {
            "wall_seconds": elapsed,
            "rss_before_bytes": rss_before,
            "rss_after_bytes": rss_after,
            "peak_rss_bytes": peak_rss_bytes(),
        },
    }
    encoded = json.dumps(record, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
