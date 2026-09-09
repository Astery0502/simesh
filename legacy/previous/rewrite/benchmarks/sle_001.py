"""Native Refined Field Lines group kernel and workflow benchmark."""

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
from simesh_rewrite._field_line_rhs import field_line_rhs_unchecked
from simesh_rewrite._rk4 import (
    field_line_rk4_finish_unchecked,
    field_line_rk4_stage_unchecked,
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
)
from simesh_rewrite.field_line_rhs import field_line_rhs_into
from simesh_rewrite.field_lines import execute_refined_field_lines
from simesh_rewrite.forest import RefinedForest, refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.point_location import refined_point_leaf_ids
from simesh_rewrite.refined_geometry import refined_leaf_geometry
from simesh_rewrite.rk4 import (
    field_line_rk4_finish_into,
    field_line_rk4_stage_into,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SENTINEL_BITS = np.uint64(0x7FF8000000005E01)
SENTINEL = np.asarray([SENTINEL_BITS], dtype=np.uint64).view(np.float64)[0]


def i3(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def command_output(*args: str) -> str:
    return subprocess.run(
        args,
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


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
        result = subprocess.run(
            ("ps", "-o", "rss=", "-p", str(os.getpid())),
            check=True,
            capture_output=True,
            text=True,
        )
        return int(result.stdout.strip()) * 1024
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def summary(values: list[float]) -> dict:
    return {
        "raw_repetitions": values,
        "median": statistics.median(values),
        "pstdev": statistics.pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
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

    def record_dict(self) -> dict:
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
            file_descriptor, byte_count, offset, section=section
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
    flags: np.ndarray | None
    leaf_bounds: np.ndarray
    leaf_spacing: np.ndarray

    @property
    def leaf_count(self) -> int:
        return int(self.forest.leaf_node_ids.size)


def synthetic_case(field_kind: str) -> Case:
    root_shape = i3(2, 1, 1)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    flags = np.asarray([False, *([True] * 8), True], dtype=np.bool_)
    forest = refined_forest(root_shape, coord_to_rank, rank_to_coord, flags)
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
    block_counts = i3(4, 4, 4)
    domain_counts = root_shape * block_counts
    lower = np.zeros(3, dtype=np.float64)
    upper = root_shape.astype(np.float64)
    leaf_ids = np.arange(forest.leaf_node_ids.size, dtype=np.int64)
    bounds, spacing = refined_leaf_geometry(
        lower,
        upper,
        root_shape,
        domain_counts,
        block_counts,
        forest.node_levels,
        forest.node_coords,
        forest.leaf_node_ids,
        leaf_ids,
    )
    backing = np.empty((leaf_ids.size, 3, 4, 4, 4), dtype=np.float64)
    grid = np.indices((4, 4, 4), dtype=np.float64)
    for leaf in range(leaf_ids.size):
        coordinates = [
            bounds[leaf, 0, axis]
            + (grid[axis] + np.float64(0.5)) * spacing[leaf, axis]
            for axis in range(3)
        ]
        if field_kind == "constant":
            backing[leaf, 0].fill(2.0)
            backing[leaf, 1].fill(0.0)
            backing[leaf, 2].fill(0.0)
        elif field_kind == "rotation":
            backing[leaf, 0] = -(coordinates[1] - 0.5)
            backing[leaf, 1] = coordinates[0] - 1.5
            backing[leaf, 2].fill(0.0)
        else:
            raise ValueError(f"unknown synthetic field kind {field_kind}")
    return Case(
        f"synthetic-{field_kind}",
        root_shape,
        coord_to_rank,
        rank_to_coord,
        forest,
        lower,
        upper,
        domain_counts,
        block_counts,
        i3(0, 1, 2),
        np.zeros((3, 6), dtype=np.uint8),
        i3(-1, -1, -1),
        backing,
        flags,
        bounds,
        spacing,
    )


def case_from_index(
    index,
    binding,
    name: str,
    field_ids: np.ndarray,
) -> Case:
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


def write_synthetic_v5(path: Path, case: Case) -> None:
    if case.backing is None or case.flags is None:
        raise ValueError("synthetic file case requires resident data and flags")
    header = header_template.copy()
    header.update(
        datfile_version=5,
        nw=3,
        ndir=3,
        ndim=3,
        levmax=case.forest.max_level,
        nleafs=case.leaf_count,
        nparents=int(np.count_nonzero(~case.flags)),
        xmin=case.domain_lower.copy(),
        xmax=case.domain_upper.copy(),
        domain_nx=case.domain_counts.astype(np.int32),
        block_nx=case.block_counts.astype(np.int32),
        periodic=np.zeros(3, dtype=np.bool_),
        geometry="Cartesian_3D",
        staggered=False,
        w_names=["b1", "b2", "b3"],
    )
    nodes = case.forest.leaf_node_ids
    tree = (
        case.forest.node_levels[nodes].astype(np.int32),
        (case.forest.node_coords[nodes] + 1).astype(np.int32),
        np.zeros(case.leaf_count, dtype=np.int64),
    )
    write_datfile_from_sfc(
        str(path),
        case.backing,
        header,
        case.flags.astype(np.int32),
        tree,
        overwrite=True,
    )


def session_arguments(case: Case, reader, cache_capacity: int) -> tuple:
    forest = case.forest
    padded_volume = int(np.prod(case.block_counts + 2, dtype=np.int64))
    entry_bytes = 3 * padded_volume * 8 + 16
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
        min(57, case.leaf_count),
        cache_capacity * entry_bytes,
    )


def locate(case: Case, points: np.ndarray) -> np.ndarray:
    forest = case.forest
    return refined_point_leaf_ids(
        case.domain_lower,
        case.domain_upper,
        case.root_shape,
        case.domain_counts,
        case.block_counts,
        forest.max_level,
        case.coord_to_rank,
        forest.root_node_ids,
        forest.child_node_ids,
        forest.node_leaf_ids,
        points,
    )


def fln_kernel(rows: int, repeats: int) -> dict:
    pattern = np.asarray(
        (
            (3.0, 4.0, 0.0),
            (0.0, -0.0, 0.0),
            (np.nan, 1.0, 0.0),
            (np.finfo(np.float64).max,) * 3,
        ),
        dtype=np.float64,
    )
    vectors = np.ascontiguousarray(np.resize(pattern, (rows, 3)))
    signs = np.where(np.arange(rows) % 2, -1, 1).astype(np.int8)
    checked_rhs = np.full((rows, 4), SENTINEL)
    checked_status = np.full(rows, 255, dtype=np.uint8)
    unchecked_rhs = checked_rhs.copy()
    unchecked_status = checked_status.copy()

    def checked() -> None:
        field_line_rhs_into(vectors, signs, checked_rhs, checked_status)

    def unchecked() -> None:
        field_line_rhs_unchecked(
            vectors, signs, unchecked_rhs, unchecked_status
        )

    with np.errstate(all="ignore"):
        checked()
        unchecked()
        checked_timing = timed(checked, repeats)
        unchecked_timing = timed(unchecked, repeats)
    if not np.array_equal(checked_status, unchecked_status):
        raise AssertionError("FLN checked/unchecked statuses differ")
    ok = checked_status == 0
    if not np.array_equal(
        checked_rhs[ok].view(np.uint64), unchecked_rhs[ok].view(np.uint64)
    ):
        raise AssertionError("FLN checked/unchecked successful RHS bits differ")
    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    checked()
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "rows": rows,
        "status_counts": {
            name: int(np.count_nonzero(checked_status == code))
            for code, name in enumerate(
                ("ok", "zero_field", "nonfinite_field", "unrepresentable_norm")
            )
        },
        "checked": {
            **checked_timing,
            "million_rows_per_second": rows / checked_timing["median"] / 1e6,
        },
        "unchecked": {
            **unchecked_timing,
            "million_rows_per_second": rows / unchecked_timing["median"] / 1e6,
        },
        "checked_unchecked_exact": True,
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
    }


def rks_kernel(rows: int, repeats: int) -> dict:
    rng = np.random.default_rng(20260904 + rows)
    base, k1, k2, k3, k4 = [
        rng.normal(size=(rows, 4)).astype(np.float64) for _ in range(5)
    ]
    checked = np.empty_like(base)
    unchecked = np.empty_like(base)
    operations = {
        "stage_half_checked": lambda: field_line_rk4_stage_into(
            base, k1, 0.125, True, checked
        ),
        "stage_half_unchecked": lambda: field_line_rk4_stage_unchecked(
            base, k1, 0.125, True, unchecked
        ),
        "stage_full_checked": lambda: field_line_rk4_stage_into(
            base, k3, 0.125, False, checked
        ),
        "stage_full_unchecked": lambda: field_line_rk4_stage_unchecked(
            base, k3, 0.125, False, unchecked
        ),
        "finish_checked": lambda: field_line_rk4_finish_into(
            base, k1, k2, k3, k4, 0.125, checked
        ),
        "finish_unchecked": lambda: field_line_rk4_finish_unchecked(
            base, k1, k2, k3, k4, 0.125, unchecked
        ),
    }
    result = {"rows": rows, "operations": {}}
    for name, operation in operations.items():
        timing = timed(operation, repeats)
        result["operations"][name] = {
            **timing,
            "million_rows_per_second": rows / timing["median"] / 1e6,
        }
    field_line_rk4_stage_into(base, k1, 0.125, True, checked)
    field_line_rk4_stage_unchecked(base, k1, 0.125, True, unchecked)
    if not np.array_equal(checked.view(np.uint64), unchecked.view(np.uint64)):
        raise AssertionError("RKS half-stage checked/unchecked bits differ")
    field_line_rk4_stage_into(base, k3, 0.125, False, checked)
    field_line_rk4_stage_unchecked(base, k3, 0.125, False, unchecked)
    if not np.array_equal(checked.view(np.uint64), unchecked.view(np.uint64)):
        raise AssertionError("RKS full-stage checked/unchecked bits differ")
    field_line_rk4_finish_into(base, k1, k2, k3, k4, 0.125, checked)
    field_line_rk4_finish_unchecked(
        base, k1, k2, k3, k4, 0.125, unchecked
    )
    if not np.array_equal(checked.view(np.uint64), unchecked.view(np.uint64)):
        raise AssertionError("RKS finish checked/unchecked bits differ")
    result["checked_unchecked_exact"] = True
    return result


@dataclass(frozen=True)
class Trajectory:
    name: str
    seeds: np.ndarray
    signs: np.ndarray
    step_size: float
    max_steps: int
    long: bool
    initial_owner_working_set: int


def point_in_leaf(case: Case, leaf: int, fraction: float = 0.37) -> np.ndarray:
    return np.ascontiguousarray(
        case.leaf_bounds[leaf, 0]
        + fraction * (case.leaf_bounds[leaf, 1] - case.leaf_bounds[leaf, 0])
    )


def trajectory_specs(case: Case, short_steps: int, long_steps: int) -> list[Trajectory]:
    single = point_in_leaf(case, 0)
    colocated = np.ascontiguousarray(np.repeat(single[None, :], 8, axis=0))
    chosen = (0, 1, 4, 8)
    divergent = np.ascontiguousarray(
        [point_in_leaf(case, chosen[row % len(chosen)]) for row in range(32)]
    )
    divergent_signs = np.asarray(
        [1 if divergent[row, 0] < 1.0 else -1 for row in range(32)],
        dtype=np.int8,
    )
    definitions = (
        ("single-short", single[None, :], np.ones(1, dtype=np.int8), 0.03125, short_steps, False),
        ("co-located-long", colocated, np.ones(8, dtype=np.int8), 0.0078125, long_steps, True),
        ("divergent-short", divergent, divergent_signs, 0.015625, short_steps, False),
    )
    result = []
    for name, seeds, signs, step, steps, long in definitions:
        owner_count = int(np.unique(locate(case, seeds)).size)
        result.append(
            Trajectory(name, seeds, signs, step, steps, long, owner_count)
        )
    return result


def allocate_outputs(spec: Trajectory) -> tuple[np.ndarray, ...]:
    count = int(spec.seeds.shape[0])
    return (
        np.full((count, spec.max_steps + 1, 3), SENTINEL),
        np.full((count, spec.max_steps + 1), SENTINEL),
        np.full(count, -1, dtype=np.int64),
        np.full(count, 255, dtype=np.uint8),
        np.full(count, 255, dtype=np.uint8),
    )


def reset_outputs(outputs: tuple[np.ndarray, ...]) -> None:
    outputs[0].view(np.uint64).fill(SENTINEL_BITS)
    outputs[1].view(np.uint64).fill(SENTINEL_BITS)
    outputs[2].fill(-1)
    outputs[3].fill(255)
    outputs[4].fill(255)


def copy_outputs(outputs: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    return tuple(value.copy() for value in outputs)


def outputs_equal(left: tuple[np.ndarray, ...], right: tuple[np.ndarray, ...]) -> bool:
    return all(
        np.array_equal(a.view(np.uint8), b.view(np.uint8))
        for a, b in zip(left, right, strict=True)
    )


def stats_record(stats) -> dict:
    return {name: int(getattr(stats, name)) for name in stats._fields}


def termination_histogram(stats) -> dict:
    names = (
        "seed_outside_count",
        "max_steps_count",
        "domain_exit_count",
        "zero_field_count",
        "nonfinite_field_count",
        "unrepresentable_norm_count",
        "unrepresentable_sample_count",
        "nonfinite_state_count",
        "no_progress_count",
    )
    return {name: int(getattr(stats, name)) for name in names}


def result_digest(outputs: tuple[np.ndarray, ...]) -> dict:
    positions, integrals, counts, codes, stages = outputs
    final_positions = []
    final_integrals = []
    for seed, count_value in enumerate(counts):
        count = int(count_value)
        if count:
            final_positions.append(positions[seed, count - 1])
            final_integrals.append(integrals[seed, count - 1])
    joined = (
        np.asarray(final_positions, dtype=np.float64)
        if final_positions
        else np.empty((0, 3), dtype=np.float64)
    )
    return {
        "point_counts": counts.tolist(),
        "termination_codes": codes.tolist(),
        "termination_stages": stages.tolist(),
        "final_position_sum": np.sum(joined, axis=0).tolist(),
        "final_integral_sum": float(np.sum(final_integrals, dtype=np.float64)),
    }


def run_trajectory(session, spec: Trajectory, outputs: tuple[np.ndarray, ...]):
    return execute_refined_field_lines(
        session,
        spec.seeds,
        spec.signs,
        spec.step_size,
        spec.max_steps,
        *outputs,
    )


def semantic_baseline(case: Case, spec: Trajectory) -> tuple[np.ndarray, ...]:
    if case.backing is None:
        raise ValueError("array baseline requires resident backing")
    session = make_completed_halo_sampling_session(
        *session_arguments(case, array_block_reader(case.backing), case.leaf_count)
    )
    outputs = allocate_outputs(spec)
    run_trajectory(session, spec, outputs)
    return copy_outputs(outputs)


def traced_execution(session, spec: Trajectory, warm: bool) -> dict:
    clear_completed_halo_sampling_session(session)
    if warm:
        run_trajectory(session, spec, allocate_outputs(spec))
    outputs = allocate_outputs(spec)
    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    stats = run_trajectory(session, spec, outputs)
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
        "executor_managed_array_bytes": int(stats.executor_managed_array_bytes),
        "sampler_call_managed_peak_bytes": int(
            stats.sampler_call_managed_peak_bytes
        ),
    }


def measure_history(
    session,
    spec: Trajectory,
    expected: tuple[np.ndarray, ...],
    repeats: int,
    *,
    native: bool,
    warm: bool,
    trace: bool,
) -> dict:
    outputs = allocate_outputs(spec)
    samples: list[dict] = []
    retained_stats = None
    for _ in range(repeats):
        clear_completed_halo_sampling_session(session)
        if warm:
            run_trajectory(session, spec, allocate_outputs(spec))
        reset_outputs(outputs)
        counter = PreadCounter()
        rss_before = current_rss_bytes()
        started = time.perf_counter()
        with count_native_preads(counter, native):
            stats = run_trajectory(session, spec, outputs)
        elapsed = time.perf_counter() - started
        rss_after = current_rss_bytes()
        if retained_stats is None:
            retained_stats = stats
        elif stats != retained_stats:
            raise AssertionError("SLE stats changed across equivalent repetitions")
        if not outputs_equal(outputs, expected):
            raise AssertionError("SLE trajectory changed by cache/backend/history")
        samples.append(
            {
                "seconds": elapsed,
                "native_pread": counter.record_dict(),
                "rss_before_bytes": rss_before,
                "rss_after_bytes": rss_after,
                "rss_delta_bytes": (
                    None
                    if rss_before is None or rss_after is None
                    else rss_after - rss_before
                ),
            }
        )
    assert retained_stats is not None
    seconds = [float(value["seconds"]) for value in samples]
    median = statistics.median(seconds)
    accepted_points = int(retained_stats.accepted_point_count)
    accepted_steps = int(retained_stats.accepted_step_count)
    hint_candidates = int(retained_stats.hint_candidate_count)
    hint_hits = int(retained_stats.hint_hit_count)
    output_bytes = sum(int(value.nbytes) for value in outputs)
    native_median_bytes = statistics.median(
        float(value["native_pread"]["bytes"]) for value in samples
    )
    native_median_headers = statistics.median(
        float(value["native_pread"]["header_bytes"]) for value in samples
    )
    native_median_payload = statistics.median(
        float(value["native_pread"]["payload_bytes"]) for value in samples
    )
    return {
        "history": "warm" if warm else "cache-cleared",
        "timing": summary(seconds),
        "raw_repetitions": samples,
        "stats": stats_record(retained_stats),
        "termination_histogram": termination_histogram(retained_stats),
        "accepted_points_per_second": accepted_points / median,
        "accepted_steps_per_second": accepted_steps / median,
        "seconds_per_accepted_point": median / max(1, accepted_points),
        "seconds_per_accepted_step": median / max(1, accepted_steps),
        "hint_hit_rate": hint_hits / max(1, hint_candidates),
        "fallbacks_after_initial_seed_samples": max(
            0,
            int(retained_stats.hierarchy_fallback_count)
            - int(retained_stats.interior_seed_count),
        ),
        "derived_owner_transitions": max(
            0,
            int(retained_stats.hierarchy_fallback_count)
            - int(retained_stats.interior_seed_count),
        ),
        "cache_lookups_per_accepted_point": int(retained_stats.cache_lookup_count)
        / max(1, accepted_points),
        "cache_hits_per_accepted_point": int(retained_stats.cache_hit_count)
        / max(1, accepted_points),
        "cache_misses_per_accepted_point": int(retained_stats.cache_miss_count)
        / max(1, accepted_points),
        "halo_fills_per_accepted_point": int(retained_stats.halo_fill_count)
        / max(1, accepted_points),
        "reader_calls_per_accepted_point": int(retained_stats.reader_call_count)
        / max(1, accepted_points),
        "selected_loads_per_accepted_point": int(retained_stats.selected_load_count)
        / max(1, accepted_points),
        "support_loads_per_accepted_point": int(retained_stats.support_load_count)
        / max(1, accepted_points),
        "logical_reader_bytes_per_accepted_point": int(
            retained_stats.logical_reader_bytes
        )
        / max(1, accepted_points),
        "native_bytes_per_accepted_point": native_median_bytes
        / max(1, accepted_points),
        "native_header_bytes_per_accepted_point": native_median_headers
        / max(1, accepted_points),
        "native_payload_bytes_per_accepted_point": native_median_payload
        / max(1, accepted_points),
        "executor_scratch_bytes": int(retained_stats.executor_managed_array_bytes),
        "sampler_call_managed_peak_bytes": int(
            retained_stats.sampler_call_managed_peak_bytes
        ),
        "session_managed_array_bytes": int(retained_stats.session_managed_array_bytes),
        "caller_output_bytes": output_bytes,
        "result_digest": result_digest(outputs),
        "traced": traced_execution(session, spec, warm) if trace else None,
        "exact_semantic_baseline": True,
    }


def measure_session(
    case: Case,
    reader,
    spec: Trajectory,
    expected: tuple[np.ndarray, ...],
    capacity: int,
    repeats: int,
    *,
    native: bool,
    trace: bool,
) -> dict:
    counter = PreadCounter()
    rss_before = current_rss_bytes()
    started = time.perf_counter()
    with count_native_preads(counter, native):
        session = make_completed_halo_sampling_session(
            *session_arguments(case, reader, capacity)
        )
    creation_seconds = time.perf_counter() - started
    rss_after = current_rss_bytes()
    if session.cache_capacity != capacity:
        raise AssertionError("requested cache capacity was not constructed")
    return {
        "capacity": capacity,
        "capacity_label": (
            "resident"
            if capacity == case.leaf_count
            else (
                "concurrent-owner-working-set"
                if capacity == spec.initial_owner_working_set
                else str(capacity)
            )
        ),
        "session_creation_seconds": creation_seconds,
        "session_creation_native_pread": counter.record_dict(),
        "rss_before_session_bytes": rss_before,
        "rss_after_session_bytes": rss_after,
        "cache_entry_bytes": int(session.cache_entry_bytes),
        "cache_payload_bytes": int(session.cache_payload_bytes),
        "session_managed_array_bytes": int(session.session_managed_array_bytes),
        "cache_cleared": measure_history(
            session,
            spec,
            expected,
            repeats,
            native=native,
            warm=False,
            trace=False,
        ),
        "cache_warm": measure_history(
            session,
            spec,
            expected,
            repeats,
            native=native,
            warm=True,
            trace=trace,
        ),
    }


def synthetic_workflow_matrix(
    array_case: Case,
    native_case: Case,
    native_reader,
    config: dict,
) -> list[dict]:
    if array_case.backing is None:
        raise ValueError("synthetic array workflow requires resident backing")
    reports = []
    for spec in trajectory_specs(
        array_case, config["short_steps"], config["long_steps"]
    ):
        expected = semantic_baseline(array_case, spec)
        capacities = {1, spec.initial_owner_working_set, array_case.leaf_count}
        if not spec.long:
            capacities.add(0)
        ordered_capacities = sorted(capacities)
        array_records = [
            measure_session(
                array_case,
                array_block_reader(array_case.backing),
                spec,
                expected,
                capacity,
                config["workflow_repeats"],
                native=False,
                trace=capacity == array_case.leaf_count,
            )
            for capacity in ordered_capacities
        ]
        native_records = [
            measure_session(
                native_case,
                native_reader,
                spec,
                expected,
                capacity,
                config["workflow_repeats"],
                native=True,
                trace=capacity == native_case.leaf_count,
            )
            for capacity in ordered_capacities
        ]
        for array_record, native_record in zip(
            array_records, native_records, strict=True
        ):
            for history in ("cache_cleared", "cache_warm"):
                if (
                    array_record[history]["stats"]
                    != native_record[history]["stats"]
                ):
                    raise AssertionError("array/native SLE statistics differ")
                if (
                    array_record[history]["result_digest"]
                    != native_record[history]["result_digest"]
                ):
                    raise AssertionError("array/native SLE result digests differ")
        reports.append(
            {
                "name": spec.name,
                "seed_count": int(spec.seeds.shape[0]),
                "step_size": spec.step_size,
                "max_steps": spec.max_steps,
                "trajectory_class": "long" if spec.long else "short",
                "initial_owner_working_set": spec.initial_owner_working_set,
                "initial_owner_ids": np.unique(locate(array_case, spec.seeds)).tolist(),
                "tested_capacities": ordered_capacities,
                "capacity_zero_restricted_to_short": True,
                "array": array_records,
                "native": native_records,
                "array_native_trajectory_and_stats_exact": True,
            }
        )
    return reports


def constant_line_evidence(case: Case) -> list[dict]:
    if case.backing is None:
        raise ValueError("constant evidence requires resident backing")
    records = []
    step = 0.03125
    step_count = 6
    for sign, seed_x in ((1, 0.25), (-1, 0.75)):
        seed = np.asarray(((seed_x, 0.25, 0.25),), dtype=np.float64)
        spec = Trajectory(
            "constant-forward" if sign > 0 else "constant-backward",
            seed,
            np.asarray((sign,), dtype=np.int8),
            step,
            step_count,
            False,
            1,
        )
        session = make_completed_halo_sampling_session(
            *session_arguments(case, array_block_reader(case.backing), case.leaf_count)
        )
        outputs = allocate_outputs(spec)
        started = time.perf_counter()
        stats = run_trajectory(session, spec, outputs)
        elapsed = time.perf_counter() - started
        expected_positions = np.repeat(seed[:, None, :], step_count + 1, axis=1)
        expected_positions[0, :, 0] += (
            sign * step * np.arange(step_count + 1, dtype=np.float64)
        )
        expected_integrals = (
            sign * 2.0 * step * np.arange(step_count + 1, dtype=np.float64)
        ).reshape(1, step_count + 1)
        expected_integrals[0, 0] = np.float64(0.0)
        exact = bool(
            np.array_equal(
                outputs[0].view(np.uint64), expected_positions.view(np.uint64)
            )
            and np.array_equal(
                outputs[1].view(np.uint64), expected_integrals.view(np.uint64)
            )
        )
        if not exact:
            raise AssertionError("constant field-line analytic bits differ")
        records.append(
            {
                "direction_sign": sign,
                "step_size": step,
                "step_count": step_count,
                "seconds": elapsed,
                "bitwise_position_integral_exact": exact,
                "stats": stats_record(stats),
                "result_digest": result_digest(outputs),
            }
        )
    return records


def rotational_evidence(case: Case) -> dict:
    if case.backing is None:
        raise ValueError("rotation evidence requires resident backing")
    distance = 0.4
    radius = 0.25
    angle = distance / radius
    expected_position = np.asarray(
        (1.5 + radius * math.cos(angle), 0.5 + radius * math.sin(angle), 0.5)
    )
    records = []
    for step in (0.05, 0.025):
        step_count = int(round(distance / step))
        seed = np.asarray(((1.75, 0.5, 0.5),), dtype=np.float64)
        spec = Trajectory(
            f"rotation-h-{step}",
            seed,
            np.ones(1, dtype=np.int8),
            step,
            step_count,
            True,
            1,
        )
        session = make_completed_halo_sampling_session(
            *session_arguments(case, array_block_reader(case.backing), case.leaf_count)
        )
        outputs = allocate_outputs(spec)
        started = time.perf_counter()
        stats = run_trajectory(session, spec, outputs)
        elapsed = time.perf_counter() - started
        final = outputs[0][0, int(outputs[2][0]) - 1]
        final_integral = float(outputs[1][0, int(outputs[2][0]) - 1])
        records.append(
            {
                "step_size": step,
                "step_count": step_count,
                "seconds": elapsed,
                "final_position": final.tolist(),
                "position_error_l2": float(np.linalg.norm(final - expected_position)),
                "expected_integral": distance * radius,
                "integral_absolute_error": abs(final_integral - distance * radius),
                "stats": stats_record(stats),
            }
        )
    order = math.log(
        records[0]["position_error_l2"] / records[1]["position_error_l2"], 2.0
    )
    if order < 3.8:
        raise AssertionError("rotational trajectory did not retain fourth order")
    return {
        "distance": distance,
        "radius": radius,
        "expected_position": expected_position.tolist(),
        "runs": records,
        "observed_order": order,
    }


def native_tdm_workflow(path: Path, config: dict) -> dict:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        metadata_started = time.perf_counter()
        index = read_amrvac_v5_index(descriptor)
        binding = bind_amrvac_v5_forest(index)
        metadata_seconds = time.perf_counter() - metadata_started
        if index.staggered or index.geometry != "Cartesian_3D" or np.any(index.periodic):
            raise ValueError("tdm fixture is outside the SLE Cartesian scope")
        if index.field_count <= 6:
            raise ValueError("tdm fixture lacks magnetic field positions 4,5,6")
        case = case_from_index(index, binding, "real-tdm-v5", i3(4, 5, 6))
        reader = make_amrvac_v5_block_reader(descriptor, index, binding)
        chosen = np.linspace(0, case.leaf_count - 1, 3, dtype=np.int64)
        seeds = np.ascontiguousarray(
            [case.leaf_bounds[int(leaf)].mean(axis=0) for leaf in chosen]
        )
        spacing = case.leaf_spacing[chosen]
        step = 0.25 * float(np.min(spacing))
        spec = Trajectory(
            "real-tdm-native",
            seeds,
            np.ones(3, dtype=np.int8),
            step,
            config["tdm_steps"],
            False,
            int(np.unique(locate(case, seeds)).size),
        )
        baseline_session = make_completed_halo_sampling_session(
            *session_arguments(case, reader, case.leaf_count)
        )
        expected = allocate_outputs(spec)
        run_trajectory(baseline_session, spec, expected)
        capacities = sorted({1, spec.initial_owner_working_set, case.leaf_count})
        records = [
            measure_session(
                case,
                reader,
                spec,
                expected,
                capacity,
                config["workflow_repeats"],
                native=True,
                trace=capacity == case.leaf_count,
            )
            for capacity in capacities
        ]
        return {
            "path": str(path),
            "file_bytes": path.stat().st_size,
            "metadata_bind_seconds": metadata_seconds,
            "metadata_bytes": int(index.offset_blocks),
            "leaf_count": case.leaf_count,
            "block_shape": case.block_counts.tolist(),
            "field_ids": case.field_ids.tolist(),
            "seed_count": int(seeds.shape[0]),
            "initial_owner_ids": np.unique(locate(case, seeds)).tolist(),
            "step_size": step,
            "max_steps": spec.max_steps,
            "capacities": records,
            "capacity_trajectory_exact": True,
        }
    finally:
        os.close(descriptor)


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
            "kernel_rows": 256,
            "kernel_repeats": 3,
            "workflow_repeats": 1,
            "short_steps": 1,
            "long_steps": 4,
            "tdm": False,
            "tdm_steps": 2,
        },
        "standard": {
            "kernel_rows": 8192,
            "kernel_repeats": 7,
            "workflow_repeats": 3,
            "short_steps": 2,
            "long_steps": 32,
            "tdm": True,
            "tdm_steps": 8,
        },
    }
    config = profiles[args.profile]
    started = time.perf_counter()
    rss_before = current_rss_bytes()
    constant_case = synthetic_case("constant")
    rotation_case = synthetic_case("rotation")

    kernels = {
        "fln": [
            fln_kernel(1, config["kernel_repeats"]),
            fln_kernel(config["kernel_rows"], config["kernel_repeats"]),
        ],
        "rks": [
            rks_kernel(1, config["kernel_repeats"]),
            rks_kernel(config["kernel_rows"], config["kernel_repeats"]),
        ],
    }
    analytic = {
        "constant_forward_backward": constant_line_evidence(constant_case),
        "rotation": rotational_evidence(rotation_case),
    }

    with tempfile.TemporaryDirectory(prefix="simesh-sle-benchmark-") as directory:
        path = Path(directory) / "synthetic-field-lines.dat"
        write_started = time.perf_counter()
        write_synthetic_v5(path, constant_case)
        write_seconds = time.perf_counter() - write_started
        descriptor = os.open(path, os.O_RDONLY)
        try:
            metadata_started = time.perf_counter()
            index = read_amrvac_v5_index(descriptor)
            binding = bind_amrvac_v5_forest(index)
            metadata_seconds = time.perf_counter() - metadata_started
            native_case = case_from_index(
                index, binding, "synthetic-native", i3(0, 1, 2)
            )
            native_reader = make_amrvac_v5_block_reader(
                descriptor, index, binding
            )
            matrix = synthetic_workflow_matrix(
                constant_case, native_case, native_reader, config
            )
            synthetic_file = {
                "file_bytes": path.stat().st_size,
                "write_seconds": write_seconds,
                "metadata_bind_seconds": metadata_seconds,
                "metadata_bytes": int(index.offset_blocks),
                "leaf_count": constant_case.leaf_count,
                "max_level": constant_case.forest.max_level,
                "root_shape": constant_case.root_shape.tolist(),
                "block_shape": constant_case.block_counts.tolist(),
                "resident_payload_bytes": int(constant_case.backing.nbytes),
            }
        finally:
            os.close(descriptor)

    real_tdm = None
    tdm_skip_reason = None
    if config["tdm"] and not args.skip_tdm:
        if args.tdm.exists():
            real_tdm = native_tdm_workflow(args.tdm, config)
        else:
            tdm_skip_reason = "configured tdm fixture is unavailable"
    else:
        tdm_skip_reason = "disabled by smoke profile or --skip-tdm"

    rss_after = current_rss_bytes()
    record = {
        "capability_group": "Native Refined Field Lines",
        "capabilities": ["FLN-001", "RKS-001", "TRM-001", "SLE-001"],
        "profile": args.profile,
        "configuration": config,
        "environment": environment_record(),
        "kernels": kernels,
        "analytic_evidence": analytic,
        "synthetic_file": synthetic_file,
        "synthetic_stage_major_workflows": matrix,
        "real_tdm_native": real_tdm,
        "tdm_skip_reason": tdm_skip_reason,
        "process": {
            "wall_seconds": time.perf_counter() - started,
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
