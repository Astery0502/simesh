"""Selected Refined Halo Completion group workflow and PWA benchmark."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
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

import simesh_rewrite.coarser_workspace_application as cwa_module
import simesh_rewrite.refined_halo as refined_halo_module
from simesh_rewrite._halo_apply import (
    apply_level1_same_level_halo_plan_unchecked,
)
from simesh_rewrite._halos import fill_physical_halos_unchecked
from simesh_rewrite._physical_widening import (
    apply_cartesian_physical_widening_unchecked,
)
from simesh_rewrite.balance import validate_refined_all_touch_2to1
from simesh_rewrite.blockio import (
    array_block_reader,
    make_block_reader,
    make_block_writer,
)
from simesh_rewrite.coarser_support import CANONICAL_DIRECTIONS
from simesh_rewrite.forest import refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.halo_apply import apply_level1_same_level_halo_plan
from simesh_rewrite.halos import fill_physical_halos
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.physical_widening import (
    apply_cartesian_physical_widening,
)
from simesh_rewrite.physical_widening_reference import (
    apply_cartesian_physical_widening_reference,
)
from simesh_rewrite.refined_halo import (
    RefinedHaloExecutionStats,
    execute_selected_refined_halos_from_blocks,
)
from simesh_rewrite.storage import gather_blocks_into
from simesh_rewrite.target_boxes import fill_directed_halo_target_boxes


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SENTINEL_BITS = np.uint64(0x7FF800000000C701)
SENTINEL = np.asarray([SENTINEL_BITS], dtype=np.uint64).view(np.float64)[0]


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
    """Return current RSS without adding instrumentation to timed regions."""
    proc_statm = Path("/proc/self/statm")
    if proc_statm.exists():
        try:
            resident_pages = int(proc_statm.read_text().split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            pass
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
    result = {
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
    try:
        from simesh.utils import openmp_build_info

        result["current_openmp"] = openmp_build_info()
    except (ImportError, RuntimeError):
        result["current_openmp"] = None
    return result


def scalar_summary(values: list[float]) -> dict:
    return {
        "median": statistics.median(values),
        "pstdev": statistics.pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def sample_summary(samples: list[dict], names: tuple[str, ...]) -> dict:
    return {
        name: scalar_summary([float(sample[name]) for sample in samples])
        for name in names
    }


def selected_ids(forest, root_shape: np.ndarray, count: int) -> np.ndarray:
    leaf_count = int(forest.leaf_node_ids.size)
    if count >= leaf_count:
        return np.arange(leaf_count, dtype=np.int64)
    node_ids = forest.leaf_node_ids
    levels = forest.node_levels[node_ids]
    coords = forest.node_coords[node_ids]
    scale = np.exp2(levels - 1).astype(np.float64)
    centers = (coords + 0.5) / (root_shape[None, :] * scale[:, None])
    distance2 = np.sum((centers - 0.5) ** 2, axis=1)
    nearest = np.argpartition(distance2, count - 1)[:count]
    return np.sort(nearest.astype(np.int64))


@dataclass
class CompactSink:
    primary_ids: tuple[int, ...]
    local_fields: tuple[int, ...]
    payload: np.ndarray
    written_ids: np.ndarray
    next_position: int = 0
    total_calls: int = 0
    nonempty_calls: int = 0
    bytes_written: int = 0
    started_at: float = 0.0
    first_write_seconds: float | None = None
    callback_seconds: float = 0.0
    measure_callback: bool = False

    def reset(self, started_at: float, *, measure_callback: bool = False) -> None:
        self.next_position = 0
        self.total_calls = 0
        self.nonempty_calls = 0
        self.bytes_written = 0
        self.started_at = started_at
        self.first_write_seconds = None
        self.callback_seconds = 0.0
        self.measure_callback = measure_callback
        self.written_ids.fill(-1)


def compact_write(
    state: CompactSink,
    source: np.ndarray,
    source_lower: np.ndarray,
    source_upper: np.ndarray,
    block_ids: np.ndarray,
    field_ids: np.ndarray,
    destination_lower: np.ndarray,
) -> None:
    callback_started = time.perf_counter() if state.measure_callback else 0.0
    state.total_calls += 1
    count = int(block_ids.shape[0])
    if count == 0:
        if state.measure_callback:
            state.callback_seconds += time.perf_counter() - callback_started
        return
    stop = state.next_position + count
    if stop > len(state.primary_ids) or any(
        int(block_ids[index]) != state.primary_ids[state.next_position + index]
        for index in range(count)
    ):
        raise ValueError("compact sink received an unexpected primary sequence")
    if field_ids.shape[0] != len(state.local_fields) or any(
        int(field_ids[index]) != state.local_fields[index]
        for index in range(field_ids.shape[0])
    ):
        raise ValueError("compact sink requires local ascending output fields")

    source_box = tuple(
        slice(int(source_lower[axis]), int(source_upper[axis]))
        for axis in range(3)
    )
    extent = tuple(
        int(source_upper[axis]) - int(source_lower[axis]) for axis in range(3)
    )
    destination_box = tuple(
        slice(
            int(destination_lower[axis]),
            int(destination_lower[axis]) + extent[axis],
        )
        for axis in range(3)
    )
    state.payload[(slice(state.next_position, stop), slice(None), *destination_box)] = (
        source[(slice(None), slice(None), *source_box)]
    )
    state.written_ids[state.next_position : stop] = block_ids
    if state.first_write_seconds is None:
        state.first_write_seconds = time.perf_counter() - state.started_at
    state.next_position = stop
    state.nonempty_calls += 1
    state.bytes_written += count * len(field_ids) * int(np.prod(extent)) * 8
    if state.measure_callback:
        state.callback_seconds += time.perf_counter() - callback_started


@dataclass
class TimedReader:
    backing: np.ndarray
    callback_seconds: float = 0.0
    total_calls: int = 0
    nonempty_calls: int = 0
    bytes_read: int = 0


def timed_read(
    state: TimedReader,
    source_lower: np.ndarray,
    source_upper: np.ndarray,
    block_ids: np.ndarray,
    field_ids: np.ndarray,
    destination: np.ndarray,
    destination_lower: np.ndarray,
) -> None:
    started = time.perf_counter()
    gather_blocks_into(
        state.backing,
        source_lower,
        source_upper,
        block_ids,
        field_ids,
        destination,
        destination_lower,
    )
    state.callback_seconds += time.perf_counter() - started
    state.total_calls += 1
    if block_ids.size:
        state.nonempty_calls += 1
        extent = source_upper - source_lower
        state.bytes_read += (
            int(block_ids.size)
            * int(field_ids.size)
            * int(np.prod(extent, dtype=np.int64))
            * 8
        )


def make_compact_sink(
    leaf_count: int,
    primary_ids: np.ndarray,
    field_count: int,
    padded_shape: tuple[int, int, int],
) -> tuple[CompactSink, object]:
    payload = np.full(
        (primary_ids.size, field_count, *padded_shape),
        SENTINEL,
        dtype=np.float64,
    )
    written_ids = np.full(primary_ids.size, -1, dtype=np.int64)
    state = CompactSink(
        tuple(int(value) for value in primary_ids),
        tuple(range(field_count)),
        payload,
        written_ids,
    )
    writer = make_block_writer(
        state,
        (leaf_count, field_count, *padded_shape),
        compact_write,
        memory_arrays=(payload, written_ids),
    )
    return state, writer


def stats_record(stats: RefinedHaloExecutionStats) -> dict:
    return {name: int(getattr(stats, name)) for name in stats._fields}


def payload_contains_sentinel(payload: np.ndarray, chunk_size: int = 128) -> bool:
    """Check a large output without an output-sized temporary boolean array."""
    for first in range(0, payload.shape[0], chunk_size):
        if np.any(
            payload[first : first + chunk_size].view(np.uint64) == SENTINEL_BITS
        ):
            return True
    return False


def run_workflow_once(
    executor_args: tuple,
    sink: CompactSink,
    *,
    measure_callback: bool = False,
) -> tuple[dict, RefinedHaloExecutionStats]:
    sink.reset(0.0, measure_callback=measure_callback)
    rss_before = current_rss_bytes()
    peak_before = peak_rss_bytes()
    started = time.perf_counter()
    sink.started_at = started
    stats = execute_selected_refined_halos_from_blocks(*executor_args)
    wall_seconds = time.perf_counter() - started
    rss_after = current_rss_bytes()
    peak_after = peak_rss_bytes()
    if sink.next_position != len(sink.primary_ids) or any(
        int(sink.written_ids[index]) != sink.primary_ids[index]
        for index in range(len(sink.primary_ids))
    ):
        raise AssertionError("compact sink did not receive every primary exactly once")
    sample = {
        "wall_seconds": wall_seconds,
        "time_to_first_write_seconds": float(sink.first_write_seconds),
        "rss_before_bytes": rss_before,
        "rss_after_bytes": rss_after,
        "rss_after_minus_before_bytes": (
            None if rss_before is None or rss_after is None else rss_after - rss_before
        ),
        "process_peak_rss_before_bytes": peak_before,
        "process_peak_rss_after_bytes": peak_after,
        "process_peak_rss_growth_bytes": max(0, peak_after - peak_before),
        "sink_total_calls_including_empty": sink.total_calls,
        "sink_nonempty_calls": sink.nonempty_calls,
        "sink_bytes_written": sink.bytes_written,
        "stats": stats_record(stats),
    }
    if measure_callback:
        sample["sink_callback_seconds"] = sink.callback_seconds
    return sample, stats


def measure_selection(
    forest_args: tuple,
    backing: np.ndarray,
    primary_ids: np.ndarray,
    query_shape: str,
    capacity: int,
    halo: int,
    warmups: int,
    repeats: int,
) -> dict:
    leaf_count, field_count = backing.shape[:2]
    block_shape = tuple(int(value) for value in backing.shape[2:])
    padded_shape = tuple(value + 2 * halo for value in block_shape)
    halo_array = i3(halo, halo, halo)
    field_ids = np.arange(field_count, dtype=np.int64)
    modes = np.zeros((field_count, 6), dtype=np.uint8)
    normals = i3(-1, -1, -1)
    sink, writer = make_compact_sink(
        leaf_count, primary_ids, field_count, padded_shape
    )
    executor_args = (
        array_block_reader(backing),
        writer,
        primary_ids,
        field_ids,
        *forest_args,
        halo_array,
        halo_array,
        modes,
        normals,
        capacity,
    )

    warmup_samples = [
        run_workflow_once(executor_args, sink)[0] for _ in range(warmups)
    ]
    sentinel_remaining_after_warmup = (
        payload_contains_sentinel(sink.payload) if warmups else None
    )
    samples_and_stats = [
        run_workflow_once(executor_args, sink) for _ in range(repeats)
    ]
    samples = [value[0] for value in samples_and_stats]
    stats_values = [value[1] for value in samples_and_stats]
    if any(value != stats_values[0] for value in stats_values[1:]):
        raise AssertionError("workflow stats changed across repetitions")
    if warmups == 0:
        sentinel_remaining_after_warmup = payload_contains_sentinel(sink.payload)
    if sentinel_remaining_after_warmup:
        raise AssertionError("compact sink contains an unwritten output value")

    summary_names = (
        "wall_seconds",
        "time_to_first_write_seconds",
        "process_peak_rss_growth_bytes",
    )
    summary = sample_summary(samples, summary_names)
    median_wall = summary["wall_seconds"]["median"]
    representative = min(
        samples, key=lambda value: abs(value["wall_seconds"] - median_wall)
    )
    stats = stats_values[0]
    block_bytes = field_count * int(np.prod(block_shape)) * 8
    requested_bytes = int(primary_ids.size) * block_bytes
    read_bytes = int(stats.selected_load_count) * block_bytes
    output_bytes = int(primary_ids.size) * field_count * int(
        np.prod(padded_shape)
    ) * 8
    if any(
        sample["sink_nonempty_calls"] != stats.writer_calls
        or sample["sink_total_calls_including_empty"] != stats.writer_calls + 1
        or sample["sink_bytes_written"] != output_bytes
        for sample in (*warmup_samples, *samples)
    ):
        raise AssertionError("compact sink call or byte accounting is inconsistent")
    return {
        "query_shape": query_shape,
        "primary_count": int(primary_ids.size),
        "capacity": capacity,
        "stats": stats_record(stats),
        "requested_input_payload_bytes": requested_bytes,
        "read_input_payload_bytes": read_bytes,
        "output_payload_bytes": output_bytes,
        "output_id_bytes": int(sink.written_ids.nbytes),
        "compact_sink_storage_bytes": int(
            sink.payload.nbytes + sink.written_ids.nbytes
        ),
        "support_load_amplification": (
            stats.selected_load_count / primary_ids.size
        ),
        "read_over_requested_bytes": read_bytes / requested_bytes,
        "managed_array_bytes": int(stats.managed_array_bytes),
        "managed_plus_output_payload_bytes": int(
            stats.managed_array_bytes + output_bytes
        ),
        "reader_calls": int(stats.reader_calls),
        "writer_calls": int(stats.writer_calls),
        "exact_primary_id_order": bool(
            all(
                int(sink.written_ids[index]) == int(primary_ids[index])
                for index in range(primary_ids.size)
            )
        ),
        "complete_output_overwrite": not bool(sentinel_remaining_after_warmup),
        "cold_sample": warmup_samples[0] if warmup_samples else samples[0],
        "warmup_samples": warmup_samples,
        "raw_repetitions": samples,
        "repetition_summary": summary,
        "representative_repetition": representative,
    }


@dataclass
class StageRecorder:
    seconds: dict[str, float] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)

    def add(self, name: str, elapsed: float) -> None:
        self.seconds[name] = self.seconds.get(name, 0.0) + elapsed
        self.calls[name] = self.calls.get(name, 0) + 1


@contextmanager
def instrument_refined_halo_stages(recorder: StageRecorder):
    stage_names = (
        (refined_halo_module, "fill_balanced_refined_relations_unchecked", "relation_generation"),
        (
            refined_halo_module,
            "plan_selected_refined_support_prefix_unchecked",
            "support_planning",
        ),
        (
            refined_halo_module,
            "resolve_refined_relation_source_slots_unchecked",
            "slot_resolution",
        ),
        (refined_halo_module, "fill_refined_relation_phase_codes_unchecked", "phase_codes"),
        (refined_halo_module, "_preflight_chunk_actions", "action_preflight"),
        (refined_halo_module, "_apply_chunk_actions_unchecked", "action_apply_total"),
        (refined_halo_module, "fill_same_level_source_boxes_unchecked", "same_geometry"),
        (refined_halo_module, "copy_region_into_unchecked", "same_copy"),
        (refined_halo_module, "fill_finer_restriction_boxes_unchecked", "finer_geometry"),
        (refined_halo_module, "restrict_cartesian_2to1_into_unchecked", "primary_restriction"),
        (refined_halo_module, "fill_coarser_workspace_boxes_unchecked", "cwp_geometry"),
        (refined_halo_module, "fill_coarser_slope_support_plan_unchecked", "csp_geometry"),
        (refined_halo_module, "_apply_coarser_workspace_plan_unchecked", "cwa_application"),
        (refined_halo_module, "prolong_cartesian_2to1_into_unchecked", "prolongation"),
        (refined_halo_module, "apply_cartesian_physical_widening_unchecked", "primary_pwa"),
        (cwa_module, "copy_region_into_unchecked", "cwa_direct_copy"),
        (cwa_module, "restrict_cartesian_2to1_into_unchecked", "cwa_restriction"),
        (cwa_module, "apply_cartesian_physical_widening_unchecked", "cwa_pwa"),
    )
    originals = {}
    for module, attribute, stage in stage_names:
        original = getattr(module, attribute)
        originals[(module, attribute)] = original

        def timed(*args, _operation=original, _stage=stage, **kwargs):
            started = time.perf_counter()
            try:
                return _operation(*args, **kwargs)
            finally:
                recorder.add(_stage, time.perf_counter() - started)

        setattr(module, attribute, timed)
    try:
        yield
    finally:
        for (module, attribute), original in originals.items():
            setattr(module, attribute, original)


def instrumented_selection_profile(
    forest_args: tuple,
    backing: np.ndarray,
    primary_ids: np.ndarray,
    capacity: int,
    halo: int,
) -> dict:
    leaf_count, field_count = backing.shape[:2]
    block_shape = tuple(int(value) for value in backing.shape[2:])
    padded_shape = tuple(value + 2 * halo for value in block_shape)
    halo_array = i3(halo, halo, halo)
    fields = np.arange(field_count, dtype=np.int64)
    modes = np.zeros((field_count, 6), dtype=np.uint8)
    normals = i3(-1, -1, -1)
    reader_state = TimedReader(backing)
    reader = make_block_reader(
        reader_state,
        backing.shape,
        timed_read,
        memory_arrays=(backing,),
    )
    sink, writer = make_compact_sink(
        leaf_count, primary_ids, field_count, padded_shape
    )
    executor_args = (
        reader,
        writer,
        primary_ids,
        fields,
        *forest_args,
        halo_array,
        halo_array,
        modes,
        normals,
        capacity,
    )
    recorder = StageRecorder()
    with instrument_refined_halo_stages(recorder):
        sample, stats = run_workflow_once(
            executor_args, sink, measure_callback=True
        )
    return {
        "primary_count": int(primary_ids.size),
        "wall_seconds": sample["wall_seconds"],
        "time_to_first_write_seconds": sample["time_to_first_write_seconds"],
        "stats": stats_record(stats),
        "stage_seconds_nested_do_not_sum": recorder.seconds,
        "stage_calls": recorder.calls,
        "reader_callback_seconds": reader_state.callback_seconds,
        "reader_callback_calls_including_empty": reader_state.total_calls,
        "reader_callback_nonempty_calls": reader_state.nonempty_calls,
        "reader_callback_bytes": reader_state.bytes_read,
        "writer_callback_seconds": sink.callback_seconds,
        "writer_callback_calls_including_empty": sink.total_calls,
        "writer_callback_nonempty_calls": sink.nonempty_calls,
        "writer_callback_bytes": sink.bytes_written,
    }


def direction_mask(direction: np.ndarray) -> int:
    result = 0
    for axis in range(3):
        if int(direction[axis]) != 0:
            result |= 1 << axis
    return result


def pwa_modes(field_count: int) -> tuple[np.ndarray, np.ndarray]:
    modes = np.empty((field_count, 6), dtype=np.uint8)
    for field_index in range(field_count):
        for face in range(6):
            modes[field_index, face] = (field_index + face) % 4
    return modes, i3(0, 1, 2)


def interleaved_timings(operations: list[tuple[str, object]], repeats: int) -> dict:
    samples = {name: [] for name, _ in operations}
    orders: list[list[str]] = []
    for repeat in range(repeats):
        order = [
            operations[(repeat + offset) % len(operations)]
            for offset in range(len(operations))
        ]
        orders.append([name for name, _ in order])
        for name, operation in order:
            started = time.perf_counter()
            operation()
            samples[name].append(time.perf_counter() - started)
    return {
        "raw_seconds": samples,
        "summary": {name: scalar_summary(values) for name, values in samples.items()},
        "orders": orders,
    }


def trace_operation(operation) -> dict:
    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    operation()
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "traced_current_delta_bytes": after_current - before_current,
        "traced_peak_delta_bytes": peak - before_current,
    }


def pwa_pure_case(repeats: int, seed: int) -> dict:
    fields = 4
    block = i3(8, 10, 12)
    lower = i3(2, 2, 2)
    upper = lower + block
    spatial = tuple(int(value + 2) for value in upper)
    rng = np.random.default_rng(seed)
    seed_payload = np.ascontiguousarray(
        rng.normal(size=(1, fields, *spatial)), dtype=np.float64
    )
    target_lower = np.empty((26, 3), dtype=np.int64)
    target_upper = np.empty_like(target_lower)
    fill_directed_halo_target_boxes(
        lower,
        upper,
        i3(0, 0, 0),
        i3(*spatial),
        CANONICAL_DIRECTIONS,
        target_lower,
        target_upper,
    )
    logical_lower = np.repeat(lower[None, :], 26, axis=0)
    logical_upper = np.repeat(upper[None, :], 26, axis=0)
    offsets = np.zeros((26, 3), dtype=np.int64)
    masks = np.asarray(
        [direction_mask(direction) for direction in CANONICAL_DIRECTIONS],
        dtype=np.uint8,
    )
    base_lower = logical_lower.copy()
    base_upper = logical_upper.copy()
    modes, normals = pwa_modes(fields)

    checked_payload = seed_payload.copy()
    unchecked_payload = seed_payload.copy()
    reference_payload = seed_payload.copy()
    hal_payload = seed_payload.copy()
    pwa_args = (
        0,
        logical_lower,
        logical_upper,
        offsets,
        CANONICAL_DIRECTIONS,
        masks,
        base_lower,
        base_upper,
        target_lower,
        target_upper,
        modes,
        normals,
    )

    def checked() -> None:
        apply_cartesian_physical_widening(checked_payload, *pwa_args)

    def unchecked() -> None:
        apply_cartesian_physical_widening_unchecked(unchecked_payload, *pwa_args)

    def reference() -> None:
        apply_cartesian_physical_widening_reference(reference_payload, *pwa_args)

    block_ids = i3(0)
    face_neighbors = np.full((1, 6), -1, dtype=np.int64)

    def hal_checked() -> None:
        fill_physical_halos(
            hal_payload,
            lower,
            upper,
            block_ids,
            face_neighbors,
            modes,
            normals,
        )

    def hal_unchecked() -> None:
        fill_physical_halos_unchecked(
            hal_payload,
            lower,
            upper,
            block_ids,
            face_neighbors,
            modes,
            normals,
        )

    with np.errstate(all="ignore"):
        checked()
        unchecked()
        reference()
        hal_checked()
    exact = bool(
        np.array_equal(checked_payload.view(np.uint64), unchecked_payload.view(np.uint64))
        and np.array_equal(checked_payload.view(np.uint64), reference_payload.view(np.uint64))
        and np.array_equal(checked_payload.view(np.uint64), hal_payload.view(np.uint64))
    )
    if not exact:
        raise AssertionError("pure PWA/reference/HAL results disagree")
    with np.errstate(all="ignore"):
        timings = interleaved_timings(
            [
                ("pwa_checked", checked),
                ("pwa_unchecked", unchecked),
                ("pwa_reference", reference),
                ("hal_checked", hal_checked),
                ("hal_unchecked", hal_unchecked),
            ],
            repeats,
        )
        traced = trace_operation(checked)
    spatial_written = int(np.prod(spatial) - np.prod(block))
    written = fields * spatial_written
    pwa_seconds = timings["summary"]["pwa_unchecked"]["median"]
    hal_seconds = timings["summary"]["hal_unchecked"]["median"]
    return {
        "label": "pure_all_26",
        "fields": fields,
        "block_shape": block.tolist(),
        "halo": 2,
        "row_count": 26,
        "written_field_cells": written,
        "candidate_field_cells": written,
        "candidate_equals_written": True,
        "semantic_bytes_per_written_cell": 16,
        "semantic_payload_bytes": 16 * written,
        "exact_checked_unchecked_reference_hal": exact,
        "timings": timings,
        "unchecked_million_cells_per_second": written / pwa_seconds / 1.0e6,
        "unchecked_semantic_gb_per_second": 16 * written / pwa_seconds / 1.0e9,
        "pwa_over_hal_unchecked": pwa_seconds / hal_seconds,
        "within_hal_20_percent_gate": pwa_seconds <= 1.2 * hal_seconds,
        "checked_allocation": traced,
    }


def pwa_mixed_case(repeats: int, seed: int) -> dict:
    fields = 4
    block = i3(8, 10, 12)
    lower = i3(2, 2, 2)
    upper = lower + block
    spatial = tuple(int(value + 2) for value in upper)
    rng = np.random.default_rng(seed + 101)
    seed_payload = np.ascontiguousarray(
        rng.normal(size=(2, fields, *spatial)), dtype=np.float64
    )
    seed_payload[
        0,
        :,
        int(lower[0]) : int(upper[0]),
        0 : int(lower[1]),
        int(lower[2]) : int(upper[2]),
    ] = seed_payload[
        1,
        :,
        int(lower[0]) : int(upper[0]),
        int(upper[1] - lower[1]) : int(upper[1]),
        int(lower[2]) : int(upper[2]),
    ]

    mixed_directions = np.asarray(
        [(-1, -1, 0), (-1, -1, -1)], dtype=np.int64
    )
    mixed_rows = [
        next(
            row
            for row, direction in enumerate(CANONICAL_DIRECTIONS)
            if np.array_equal(direction, requested)
        )
        for requested in mixed_directions
    ]
    all_target_lower = np.empty((26, 3), dtype=np.int64)
    all_target_upper = np.empty_like(all_target_lower)
    fill_directed_halo_target_boxes(
        lower,
        upper,
        i3(0, 0, 0),
        i3(*spatial),
        CANONICAL_DIRECTIONS,
        all_target_lower,
        all_target_upper,
    )
    target_lower = np.ascontiguousarray(all_target_lower[mixed_rows])
    target_upper = np.ascontiguousarray(all_target_upper[mixed_rows])
    logical_lower = np.repeat(lower[None, :], 2, axis=0)
    logical_upper = np.repeat(upper[None, :], 2, axis=0)
    offsets = np.zeros((2, 3), dtype=np.int64)
    masks = np.asarray([1, 5], dtype=np.uint8)
    base_lower = np.repeat(lower[None, :], 2, axis=0)
    base_upper = np.repeat(upper[None, :], 2, axis=0)
    base_lower[:, 1] = 0
    base_upper[:, 1] = lower[1]
    modes, normals = pwa_modes(fields)

    checked_payload = seed_payload.copy()
    unchecked_payload = seed_payload.copy()
    reference_payload = seed_payload.copy()
    hax_payload = seed_payload.copy()
    pwa_args = (
        0,
        logical_lower,
        logical_upper,
        offsets,
        mixed_directions,
        masks,
        base_lower,
        base_upper,
        target_lower,
        target_upper,
        modes,
        normals,
    )

    def checked() -> None:
        apply_cartesian_physical_widening(checked_payload, *pwa_args)

    def unchecked() -> None:
        apply_cartesian_physical_widening_unchecked(unchecked_payload, *pwa_args)

    def reference() -> None:
        apply_cartesian_physical_widening_reference(reference_payload, *pwa_args)

    source_slots = np.full((1, 27), -1, dtype=np.int64)
    hax_masks = np.zeros((1, 27), dtype=np.uint8)
    for column in range(27):
        direction = np.asarray(
            (column % 3 - 1, (column // 3) % 3 - 1, column // 9 - 1),
            dtype=np.int64,
        )
        hax_masks[0, column] = direction_mask(direction)
    for direction, mask in zip(mixed_directions, masks, strict=True):
        column = (
            (int(direction[2]) + 1) * 9
            + (int(direction[1]) + 1) * 3
            + int(direction[0])
            + 1
        )
        source_slots[0, column] = 1
        hax_masks[0, column] = mask

    def hax_checked() -> None:
        apply_level1_same_level_halo_plan(
            hax_payload,
            lower,
            upper,
            source_slots,
            hax_masks,
            modes,
            normals,
        )

    def hax_unchecked() -> None:
        apply_level1_same_level_halo_plan_unchecked(
            hax_payload,
            lower,
            upper,
            source_slots,
            hax_masks,
            modes,
            normals,
        )

    with np.errstate(all="ignore"):
        checked()
        unchecked()
        reference()
        hax_checked()
    exact = bool(
        np.array_equal(checked_payload.view(np.uint64), unchecked_payload.view(np.uint64))
        and np.array_equal(checked_payload.view(np.uint64), reference_payload.view(np.uint64))
        and np.array_equal(checked_payload.view(np.uint64), hax_payload.view(np.uint64))
    )
    if not exact:
        raise AssertionError("mixed PWA/reference/HAX results disagree")
    with np.errstate(all="ignore"):
        timings = interleaved_timings(
            [
                ("pwa_checked", checked),
                ("pwa_unchecked", unchecked),
                ("pwa_reference", reference),
                ("hax_checked", hax_checked),
                ("hax_unchecked", hax_unchecked),
            ],
            repeats,
        )
        traced = trace_operation(checked)
    written = fields * sum(
        int(np.prod(target_upper[row] - target_lower[row])) for row in range(2)
    )
    pwa_seconds = timings["summary"]["pwa_unchecked"]["median"]
    hax_seconds = timings["summary"]["hax_unchecked"]["median"]
    return {
        "label": "mixed_edge_and_corner",
        "fields": fields,
        "block_shape": block.tolist(),
        "halo": 2,
        "row_count": 2,
        "written_field_cells": written,
        "candidate_field_cells": written,
        "candidate_equals_written": True,
        "semantic_bytes_per_written_cell": 16,
        "semantic_payload_bytes": 16 * written,
        "exact_checked_unchecked_reference_hax": exact,
        "timings": timings,
        "unchecked_million_cells_per_second": written / pwa_seconds / 1.0e6,
        "unchecked_semantic_gb_per_second": 16 * written / pwa_seconds / 1.0e9,
        "pwa_over_hax_unchecked": pwa_seconds / hax_seconds,
        "within_hax_20_percent_gate": pwa_seconds <= 1.2 * hax_seconds,
        "checked_allocation": traced,
    }


def pwa_standard(repeats: int, seed: int) -> dict:
    return {
        "profile": "standard",
        "repeats": repeats,
        "hypothesis": (
            "allocation-free unchecked PWA is no more than 20% slower per "
            "written field-cell than HAL on pure targets and HAX on mixed targets"
        ),
        "cases": [
            pwa_pure_case(repeats, seed),
            pwa_mixed_case(repeats, seed),
        ],
    }


def validate_cli(args: argparse.Namespace) -> None:
    if args.capacity <= 0:
        raise ValueError("capacity must be positive")
    if args.small <= 0 or args.medium <= 0:
        raise ValueError("small and medium selection sizes must be positive")
    if args.fields <= 0:
        raise ValueError("fields must be positive for the workflow benchmark")
    if args.block < 4 or args.block % 2:
        raise ValueError("block must be even and at least four")
    if args.halo < 0 or args.halo > args.block // 2:
        raise ValueError("halo must lie in [0, block/2]")
    if args.warmups < 0 or args.repeats <= 0 or args.pwa_repeats <= 0:
        raise ValueError("warmups must be nonnegative and repeats must be positive")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dat", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--small", type=int, default=32)
    parser.add_argument("--medium", type=int, default=512)
    parser.add_argument("--fields", type=int, default=3)
    parser.add_argument("--block", type=int, default=4)
    parser.add_argument("--halo", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--pwa-repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    validate_cli(args)

    from simesh.amrvac.datio import get_metadata

    metadata_started = time.perf_counter()
    header, flags_input, _ = get_metadata(str(args.dat))
    metadata_seconds = time.perf_counter() - metadata_started
    root_shape = np.ascontiguousarray(
        header["domain_nx"] // header["block_nx"], dtype=np.int64
    )
    forest_started = time.perf_counter()
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    forest = refined_forest(
        root_shape,
        coord_to_rank,
        rank_to_coord,
        np.ascontiguousarray(flags_input, dtype=np.bool_),
    )
    validate_refined_forest_arrays(
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
    forest_args = (
        root_shape,
        coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
    )
    validate_refined_all_touch_2to1(*forest_args)
    forest_seconds = time.perf_counter() - forest_started

    leaf_count = int(forest.leaf_node_ids.size)
    if args.capacity > leaf_count:
        raise ValueError("capacity may not exceed the metadata leaf count")
    if args.capacity < min(57, leaf_count):
        raise ValueError("capacity is below the RHE all-26 progress bound")
    backing_started = time.perf_counter()
    values = np.arange(
        leaf_count * args.fields * args.block**3, dtype=np.float64
    )
    backing = values.reshape(
        leaf_count,
        args.fields,
        args.block,
        args.block,
        args.block,
    )
    backing_seconds = time.perf_counter() - backing_started

    counts = sorted(
        {min(args.small, leaf_count), min(args.medium, leaf_count), leaf_count}
    )
    selections = [
        (
            count,
            selected_ids(forest, root_shape, count),
            "full_domain" if count == leaf_count else "coherent_center_roi",
        )
        for count in counts
    ]
    selection_reports = [
        measure_selection(
            forest_args,
            backing,
            primary_ids,
            query_shape,
            args.capacity,
            args.halo,
            args.warmups,
            args.repeats,
        )
        for _, primary_ids, query_shape in selections
    ]
    profile_index = min(
        range(len(selections)),
        key=lambda index: (
            abs(selections[index][0] - min(args.medium, leaf_count)),
            selections[index][0],
        ),
    )
    _, profile_ids, _ = selections[profile_index]
    instrumented = instrumented_selection_profile(
        forest_args,
        backing,
        profile_ids,
        args.capacity,
        args.halo,
    )

    report = {
        "group": "Selected Refined Halo Completion",
        "capabilities": ["CSP-001", "PWA-001", "CWA-001", "RHE-001"],
        "profile": "standard",
        "command_arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "dat": str(args.dat),
        "staggered_metadata_only": bool(header["staggered"]),
        "metadata_seconds": metadata_seconds,
        "forest_build_validate_seconds": forest_seconds,
        "synthetic_backing_seconds": backing_seconds,
        "synthetic_backing_bytes": int(backing.nbytes),
        "leaf_count": leaf_count,
        "root_shape": root_shape.tolist(),
        "block_shape": [args.block] * 3,
        "halo": args.halo,
        "padded_shape": [args.block + 2 * args.halo] * 3,
        "fields": args.fields,
        "capacity": args.capacity,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "environment": environment_record(),
        "selections": selection_reports,
        "instrumented_profile": instrumented,
        "pwa_standard": pwa_standard(args.pwa_repeats, args.seed),
        "current_correctness_comparator": (
            "focused test_rhe_001.py dyadic current AMRMesh comparison"
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
