"""Synthetic milestone-workflow benchmark for LFE-001."""

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
import tempfile
import time
import tracemalloc

import Cython
import numpy as np

import simesh_rewrite.local_field as local_field_module
import simesh_rewrite.refined_halo as refined_halo_module
import simesh_rewrite.amrvac_dat_reader as dat_reader_module
from simesh.amrvac.datio import (
    header_template,
    read_blocks_sequential,
    write_datfile_from_sfc,
)
from simesh.amrvac import open_dataset
from simesh_rewrite.amrvac_dat import (
    bind_amrvac_v5_forest,
    read_amrvac_v5_index,
)
from simesh_rewrite.amrvac_dat_reader import make_amrvac_v5_block_reader
from simesh_rewrite.balance import validate_refined_all_touch_2to1
from simesh_rewrite.blockio import array_block_reader
from simesh_rewrite.forest import refined_forest
from simesh_rewrite.forest_conformance import validate_refined_forest_arrays
from simesh_rewrite.local_field import execute_selected_refined_curl_from_blocks
from simesh_rewrite.morton import level1_morton
from simesh_rewrite.refined_geometry import refined_leaf_geometry
from simesh_rewrite.region_selection import refined_region_windows


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SENTINEL_BITS = np.uint64(0x7FF800000000C701)
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


def scalar_summary(values: list[float]) -> dict:
    return {
        "median": statistics.median(values),
        "pstdev": statistics.pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


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
def count_native_preads(counter: PreadCounter):
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
class Fixture:
    root_shape: np.ndarray
    coord_to_rank: np.ndarray
    root_node_ids: np.ndarray
    node_levels: np.ndarray
    node_coords: np.ndarray
    child_node_ids: np.ndarray
    node_leaf_ids: np.ndarray
    leaf_node_ids: np.ndarray
    max_level: int
    domain_lower: np.ndarray
    domain_upper: np.ndarray
    domain_counts: np.ndarray
    block_counts: np.ndarray
    backing: np.ndarray
    forest_flags: np.ndarray


def make_fixture() -> Fixture:
    root_shape = i3(6, 4, 2)
    block_counts = i3(4, 4, 4)
    coord_to_rank, rank_to_coord = level1_morton(root_shape)
    flags: list[bool] = []
    for coordinate in rank_to_coord:
        split = sum(int(value) for value in coordinate) % 2 == 0
        flags.append(not split)
        if split:
            flags.extend([True] * 8)
    forest = refined_forest(
        root_shape,
        coord_to_rank,
        rank_to_coord,
        np.asarray(flags, dtype=np.bool_),
    )
    max_level = validate_refined_forest_arrays(
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
    domain_lower = np.asarray((-3.0, 1.0, -2.0), dtype=np.float64)
    domain_upper = np.asarray((9.0, 9.0, 4.0), dtype=np.float64)
    domain_counts = np.ascontiguousarray(root_shape * block_counts)
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
    backing = np.empty((leaf_ids.size, 3, 4, 4, 4), dtype=np.float64)
    local = np.indices((4, 4, 4), dtype=np.float64)
    for leaf in range(leaf_ids.size):
        centers = [
            bounds[leaf, 0, axis]
            + (local[axis] + 0.5) * spacing[leaf, axis]
            for axis in range(3)
        ]
        x, y, z = centers
        backing[leaf, 0] = y + 2.0 * z
        backing[leaf, 1] = 3.0 * z + 4.0 * x
        backing[leaf, 2] = 5.0 * x + 6.0 * y
    return Fixture(
        root_shape,
        coord_to_rank,
        forest.root_node_ids,
        forest.node_levels,
        forest.node_coords,
        forest.child_node_ids,
        forest.node_leaf_ids,
        forest.leaf_node_ids,
        max_level,
        domain_lower,
        domain_upper,
        domain_counts,
        block_counts,
        backing,
        np.asarray(flags, dtype=np.int32),
    )


def write_fixture_dat(path: Path, fixture: Fixture) -> None:
    leaf_nodes = fixture.leaf_node_ids
    block_levels = fixture.node_levels[leaf_nodes].astype(np.int32)
    block_coordinates = (fixture.node_coords[leaf_nodes] + 1).astype(np.int32)
    header = header_template.copy()
    header.update(
        datfile_version=5,
        nw=3,
        ndir=3,
        ndim=3,
        levmax=fixture.max_level,
        nleafs=int(leaf_nodes.size),
        nparents=int(fixture.node_levels.size - leaf_nodes.size),
        xmin=fixture.domain_lower.copy(),
        xmax=fixture.domain_upper.copy(),
        domain_nx=fixture.domain_counts.astype(np.int32),
        block_nx=fixture.block_counts.astype(np.int32),
        periodic=np.zeros(3, dtype=np.bool_),
        geometry="Cartesian_3D",
        staggered=False,
        w_names=["b1", "b2", "b3"],
    )
    tree = (
        block_levels,
        block_coordinates,
        np.zeros(leaf_nodes.size, dtype=np.int64),
    )
    write_datfile_from_sfc(
        str(path),
        fixture.backing,
        header,
        fixture.forest_flags,
        tree,
        overwrite=True,
    )


def open_native_fixture(path: Path, fixture: Fixture):
    write_started = time.perf_counter()
    write_fixture_dat(path, fixture)
    write_seconds = time.perf_counter() - write_started

    current_started = time.perf_counter()
    current = read_blocks_sequential(str(path), [0, 1, 2])
    current_seconds = time.perf_counter() - current_started
    current_equal = bool(
        np.array_equal(current.view(np.uint64), fixture.backing.view(np.uint64))
    )
    if not current_equal:
        raise AssertionError("current eager read differs from synthetic source")

    file_descriptor = os.open(path, os.O_RDONLY)
    metadata_started = time.perf_counter()
    index = read_amrvac_v5_index(file_descriptor)
    indexed = time.perf_counter()
    binding = bind_amrvac_v5_forest(index)
    bound = time.perf_counter()
    if (
        index.leaf_count != fixture.leaf_node_ids.size
        or not np.array_equal(binding.root_shape, fixture.root_shape)
        or not np.array_equal(
            binding.forest.leaf_node_ids, fixture.leaf_node_ids
        )
        or not np.array_equal(index.block_cell_counts, fixture.block_counts)
    ):
        os.close(file_descriptor)
        raise AssertionError("native DAT/FST metadata differs from fixture")
    reader = make_amrvac_v5_block_reader(file_descriptor, index, binding)
    return file_descriptor, reader, {
        "path": str(path),
        "file_bytes": path.stat().st_size,
        "write_seconds": write_seconds,
        "metadata_index_seconds": indexed - metadata_started,
        "forest_bind_seconds": bound - indexed,
        "metadata_total_seconds": bound - metadata_started,
        "metadata_bytes_read": int(index.offset_blocks),
        "current_eager_seconds": current_seconds,
        "current_eager_output_bytes": int(current.nbytes),
        "current_eager_bitwise_source_equal": current_equal,
    }


def roi_bounds(fixture: Fixture) -> list[tuple[str, np.ndarray, np.ndarray]]:
    lower = fixture.domain_lower
    extent = fixture.domain_upper - lower
    return [
        ("thin", lower.copy() + np.asarray((0.0, 0.22, 0.22)) * extent,
         lower + np.asarray((0.06, 0.78, 0.78)) * extent),
        ("small", lower + 0.42 * extent, lower + 0.58 * extent),
        ("medium", lower + 0.22 * extent, lower + 0.78 * extent),
        ("full", lower.copy(), fixture.domain_upper.copy()),
    ]


def selection_args(fixture: Fixture, lower: np.ndarray, upper: np.ndarray) -> tuple:
    return (
        fixture.domain_lower,
        fixture.domain_upper,
        fixture.root_shape,
        fixture.domain_counts,
        fixture.block_counts,
        fixture.node_levels,
        fixture.node_coords,
        fixture.leaf_node_ids,
        np.ascontiguousarray(lower),
        np.ascontiguousarray(upper),
    )


def execution_args(
    fixture: Fixture,
    reader,
    magnetic_field_ids: np.ndarray,
    selection,
    capacity: int,
    output: np.ndarray,
    accumulator: np.ndarray,
) -> tuple:
    return (
        reader,
        selection.leaf_ids,
        selection.cell_lower,
        selection.cell_upper,
        magnetic_field_ids,
        fixture.domain_lower,
        fixture.domain_upper,
        fixture.domain_counts,
        fixture.block_counts,
        fixture.max_level,
        fixture.root_shape,
        fixture.coord_to_rank,
        fixture.root_node_ids,
        fixture.node_levels,
        fixture.node_coords,
        fixture.child_node_ids,
        fixture.node_leaf_ids,
        fixture.leaf_node_ids,
        np.zeros((3, 6), dtype=np.uint8),
        i3(-1, -1, -1),
        capacity,
        output,
        2,
        accumulator,
    )


def stats_record(stats) -> dict:
    return {name: int(getattr(stats, name)) for name in stats._fields}


def logical_digest(output: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict:
    values: list[np.ndarray] = []
    for row in range(lower.shape[0]):
        box = tuple(slice(int(lower[row, a]), int(upper[row, a])) for a in range(3))
        values.append(output[(row, slice(None), *box)].reshape(-1))
    joined = np.concatenate(values) if values else np.empty(0, dtype=np.float64)
    finite = joined[np.isfinite(joined)]
    return {
        "value_count": int(joined.size),
        "finite_count": int(finite.size),
        "nan_count": int(np.count_nonzero(np.isnan(joined))),
        "positive_infinity_count": int(np.count_nonzero(np.isposinf(joined))),
        "negative_infinity_count": int(np.count_nonzero(np.isneginf(joined))),
        "finite_sum": float(np.sum(finite, dtype=np.float64)),
        "finite_l1": float(np.sum(np.abs(finite), dtype=np.float64)),
        "finite_min": None if not finite.size else float(np.min(finite)),
        "finite_max": None if not finite.size else float(np.max(finite)),
    }


def manual_sum(
    output: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    component: int,
    initial: float,
) -> np.ndarray:
    result = np.asarray([initial], dtype=np.float64)
    for row in range(lower.shape[0]):
        for x in range(int(lower[row, 0]), int(upper[row, 0])):
            for y in range(int(lower[row, 1]), int(upper[row, 1])):
                for z in range(int(lower[row, 2]), int(upper[row, 2])):
                    result[0] += output[row, component, x, y, z]
    return result


def current_derived_curl(
    path: Path,
    magnetic_fields: np.ndarray,
) -> tuple[np.ndarray, dict]:
    started = time.perf_counter()
    dataset = open_dataset(str(path), ghost_width=2, boundary_conditions="cont")
    opened = time.perf_counter()
    dataset.load_data(field_indices=magnetic_fields.tolist())
    loaded = time.perf_counter()
    dataset.register_derivative(
        "j1", (("b3", "y", 1.0), ("b2", "z", -1.0))
    )
    dataset.register_derivative(
        "j2", (("b1", "z", 1.0), ("b3", "x", -1.0))
    )
    dataset.register_derivative(
        "j3", (("b2", "x", 1.0), ("b1", "y", -1.0))
    )
    dataset.materialize_fields(("j1", "j2", "j3"))
    materialized = time.perf_counter()
    result = np.ascontiguousarray(
        dataset.blocks(field_names=("j1", "j2", "j3")),
        dtype=np.float64,
    )
    return result, {
        "open_metadata_seconds": opened - started,
        "load_and_ghost_seconds": loaded - opened,
        "materialize_current_seconds": materialized - loaded,
        "total_seconds": materialized - started,
        "selected_current_output_bytes": int(result.nbytes),
        "dataset_interior_bytes": int(dataset.data.nbytes),
    }


def compare_safe_current_cells(
    output: np.ndarray,
    selection,
    current: np.ndarray,
    block_shape: np.ndarray,
) -> dict:
    actual_parts: list[np.ndarray] = []
    expected_parts: list[np.ndarray] = []
    for row, leaf_id in enumerate(selection.leaf_ids):
        lower = np.maximum(selection.cell_lower[row], 1)
        upper = np.minimum(selection.cell_upper[row], block_shape - 1)
        if np.any(lower >= upper):
            continue
        box = tuple(slice(int(lower[axis]), int(upper[axis])) for axis in range(3))
        actual_parts.append(output[(row, slice(None), *box)].reshape(-1))
        expected_parts.append(current[(int(leaf_id), slice(None), *box)].reshape(-1))
    actual = (
        np.concatenate(actual_parts)
        if actual_parts
        else np.empty(0, dtype=np.float64)
    )
    expected = (
        np.concatenate(expected_parts)
        if expected_parts
        else np.empty(0, dtype=np.float64)
    )
    finite = np.isfinite(actual) & np.isfinite(expected)
    classification_equal = bool(
        np.array_equal(np.isnan(actual), np.isnan(expected))
        and np.array_equal(np.isposinf(actual), np.isposinf(expected))
        and np.array_equal(np.isneginf(actual), np.isneginf(expected))
    )
    difference = np.abs(actual[finite] - expected[finite])
    scale = np.maximum(np.abs(expected[finite]), np.finfo(np.float64).tiny)
    close = bool(
        classification_equal
        and np.allclose(actual[finite], expected[finite], rtol=5e-13, atol=5e-13)
    )
    if not close:
        raise AssertionError("LFE differs from current derived curl on safe cells")
    return {
        "safe_value_count": int(actual.size),
        "finite_pair_count": int(np.count_nonzero(finite)),
        "classification_equal": classification_equal,
        "bitwise_equal_count": int(
            np.count_nonzero(actual.view(np.uint64) == expected.view(np.uint64))
        ),
        "maximum_absolute_difference": (
            None if not difference.size else float(np.max(difference))
        ),
        "maximum_relative_difference": (
            None if not difference.size else float(np.max(difference / scale))
        ),
        "rtol": 5e-13,
        "atol": 5e-13,
        "within_tolerance": close,
        "operation_tree_note": (
            "LFE uses six separate centered derivatives then three scale-one "
            "differences; current batches +0 plus signed derivative terms"
        ),
    }


@dataclass
class StageRecorder:
    started_at: float = 0.0
    seconds: dict[str, float] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)
    first_result_seconds: float | None = None

    def add(self, name: str, elapsed: float) -> None:
        self.seconds[name] = self.seconds.get(name, 0.0) + elapsed
        self.calls[name] = self.calls.get(name, 0) + 1


@contextmanager
def instrument_stages(recorder: StageRecorder):
    targets = [
        (local_field_module, "refined_leaf_geometry", "selected_geometry"),
        (
            local_field_module,
            "execute_selected_refined_halos_with_consumer",
            "halo_consumer_total",
        ),
        (local_field_module, "cartesian_curl_unchecked", "curl"),
        (local_field_module, "accumulate_field_sum_unchecked", "reduction"),
        (refined_halo_module, "read_blocks_into", "reader_boundary"),
        (refined_halo_module, "_prepare_refined_halo_chunk", "halo_prepare"),
        (refined_halo_module, "_preflight_chunk_actions", "action_preflight"),
        (refined_halo_module, "_apply_chunk_actions_unchecked", "halo_apply"),
    ]
    originals: list[tuple[object, str, object]] = []
    for module, attribute, stage in targets:
        original = getattr(module, attribute)
        originals.append((module, attribute, original))

        def timed(*args, _operation=original, _stage=stage, **kwargs):
            started = time.perf_counter()
            try:
                return _operation(*args, **kwargs)
            finally:
                recorder.add(_stage, time.perf_counter() - started)
                if _stage == "curl" and recorder.first_result_seconds is None:
                    recorder.first_result_seconds = time.perf_counter() - recorder.started_at

        setattr(module, attribute, timed)
    try:
        yield
    finally:
        for module, attribute, original in originals:
            setattr(module, attribute, original)


def measure_selection(arguments: tuple, repeats: int) -> tuple[object, dict]:
    raw: list[float] = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = refined_region_windows(*arguments)
        raw.append(time.perf_counter() - started)
    assert result is not None
    return result, {
        "raw_repetitions_seconds": raw,
        "summary_seconds": scalar_summary(raw),
        "plan_bytes": int(sum(value.nbytes for value in result)),
    }


def measure_execution(
    fixture: Fixture,
    reader,
    backend: str,
    magnetic_field_ids: np.ndarray,
    selection,
    capacity: int,
    warmups: int,
    repeats: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    block_shape = tuple(int(value) for value in fixture.block_counts)
    output = np.full((selection.leaf_ids.size, 3, *block_shape), SENTINEL)
    accumulator = np.asarray([1.25], dtype=np.float64)
    arguments = execution_args(
        fixture,
        reader,
        magnetic_field_ids,
        selection,
        capacity,
        output,
        accumulator,
    )

    def run_once() -> tuple[dict, object]:
        output.view(np.uint64).fill(SENTINEL_BITS)
        accumulator[0] = 1.25
        rss_before = current_rss_bytes()
        faults_before = resource.getrusage(resource.RUSAGE_SELF)
        started = time.perf_counter()
        preads = PreadCounter()
        if backend == "native":
            with count_native_preads(preads):
                stats = execute_selected_refined_curl_from_blocks(*arguments)
        else:
            stats = execute_selected_refined_curl_from_blocks(*arguments)
        wall = time.perf_counter() - started
        faults_after = resource.getrusage(resource.RUSAGE_SELF)
        rss_after = current_rss_bytes()
        return {
            "wall_seconds": wall,
            "rss_before_bytes": rss_before,
            "rss_after_bytes": rss_after,
            "minor_faults": int(faults_after.ru_minflt - faults_before.ru_minflt),
            "major_faults": int(faults_after.ru_majflt - faults_before.ru_majflt),
            "process_peak_rss_bytes": peak_rss_bytes(),
            "pread": preads.as_dict(),
            "stats": stats_record(stats),
        }, stats

    cold, expected_stats = run_once()
    warmup_samples = [run_once()[0] for _ in range(warmups)]
    raw_and_stats = [run_once() for _ in range(repeats)]
    raw = [item[0] for item in raw_and_stats]
    if any(item[1] != expected_stats for item in raw_and_stats):
        raise AssertionError("execution stats changed across repetitions")
    if backend == "native":
        expected_pread = cold["pread"]
        if any(item["pread"] != expected_pread for item in raw):
            raise AssertionError("native pread counts changed across repetitions")
        if expected_pread["payload_bytes"] != expected_stats.logical_reader_bytes:
            raise AssertionError("native payload bytes differ from logical bytes")
        if expected_pread["header_bytes"] != 24 * expected_stats.selected_load_count:
            raise AssertionError("native header bytes differ from selected loads")

    expected_sum = manual_sum(
        output, selection.cell_lower, selection.cell_upper, 2, 1.25
    )
    reduction_exact = bool(
        np.array_equal(accumulator.view(np.uint64), expected_sum.view(np.uint64))
    )
    if not reduction_exact:
        raise AssertionError("LFE reduction differs from manual serial order")

    recorder = StageRecorder()
    output.view(np.uint64).fill(SENTINEL_BITS)
    accumulator[0] = 1.25
    recorder.started_at = time.perf_counter()
    with instrument_stages(recorder):
        instrumented_stats = execute_selected_refined_curl_from_blocks(*arguments)
    instrumented_wall = time.perf_counter() - recorder.started_at

    output.view(np.uint64).fill(SENTINEL_BITS)
    accumulator[0] = 1.25
    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    execute_selected_refined_curl_from_blocks(*arguments)
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    stats = expected_stats
    block_volume = int(np.prod(fixture.block_counts, dtype=np.int64))
    primary_bytes = int(stats.primary_count) * 3 * block_volume * 8
    support_bytes = int(stats.support_load_count) * 3 * block_volume * 8
    useful_bytes = int(stats.output_cell_count) * 3 * 8
    output_bytes = int(stats.output_value_count) * 8
    median_wall = statistics.median(item["wall_seconds"] for item in raw)
    report = {
        "backend": backend,
        "capacity": capacity,
        "stats": stats_record(stats),
        "cold_sample": cold,
        "warmup_samples": warmup_samples,
        "raw_repetitions": raw,
        "wall_summary_seconds": scalar_summary(
            [float(item["wall_seconds"]) for item in raw]
        ),
        "million_output_values_per_second": stats.output_value_count / median_wall / 1e6,
        "primary_logical_payload_bytes": primary_bytes,
        "support_logical_payload_bytes": support_bytes,
        "logical_reader_bytes": int(stats.logical_reader_bytes),
        "physical_reader": cold["pread"],
        "useful_selected_value_bytes": useful_bytes,
        "logical_output_bytes": output_bytes,
        "allocated_output_bytes": int(output.nbytes + accumulator.nbytes),
        "support_load_amplification": stats.selected_load_count / stats.primary_count,
        "block_cover_amplification": (
            stats.primary_count * block_volume / stats.output_cell_count
        ),
        "managed_plus_output_bytes": int(stats.managed_array_bytes + output.nbytes),
        "reduction_bitwise_equal_to_manual": reduction_exact,
        "accumulator_value": float(accumulator[0]),
        "accumulator_bits": int(accumulator.view(np.uint64)[0]),
        "numerical_digest": logical_digest(
            output, selection.cell_lower, selection.cell_upper
        ),
        "instrumented": {
            "wall_seconds": instrumented_wall,
            "time_to_first_result_seconds": recorder.first_result_seconds,
            "stage_seconds_nested_do_not_sum": recorder.seconds,
            "stage_calls": recorder.calls,
            "stats": stats_record(instrumented_stats),
        },
        "tracemalloc": {
            "current_delta_bytes": after_current - before_current,
            "peak_delta_bytes": peak - before_current,
        },
    }
    return report, output.copy(), accumulator.copy()


def profile_parameters(profile: str) -> tuple[int, int]:
    return (0, 2) if profile == "smoke" else (1, 3)


def real_dat_workflow(path: Path, warmups: int, repeats: int) -> dict:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        metadata_started = time.perf_counter()
        index = read_amrvac_v5_index(descriptor)
        indexed = time.perf_counter()
        binding = bind_amrvac_v5_forest(index)
        bound = time.perf_counter()
        if index.staggered or index.geometry != "Cartesian_3D":
            raise ValueError("real LFE input must be non-staggered Cartesian_3D")
        names = {
            name.lower(): position
            for position, name in enumerate(index.field_names)
        }
        try:
            native_fields = np.asarray(
                [names["b1"], names["b2"], names["b3"]], dtype=np.int64
            )
        except KeyError as error:
            raise ValueError("real LFE input must contain b1, b2, and b3") from error

        eager_started = time.perf_counter()
        eager = read_blocks_sequential(str(path), native_fields.tolist())
        eager_seconds = time.perf_counter() - eager_started
        current_curl, current_curl_report = current_derived_curl(
            path, native_fields
        )
        forest = binding.forest
        validate_refined_all_touch_2to1(
            binding.root_shape,
            binding.coord_to_rank,
            forest.root_node_ids,
            forest.node_levels,
            forest.node_coords,
            forest.child_node_ids,
            forest.node_leaf_ids,
            forest.leaf_node_ids,
        )
        fixture = Fixture(
            binding.root_shape,
            binding.coord_to_rank,
            forest.root_node_ids,
            forest.node_levels,
            forest.node_coords,
            forest.child_node_ids,
            forest.node_leaf_ids,
            forest.leaf_node_ids,
            forest.max_level,
            index.domain_lower,
            index.domain_upper,
            index.domain_cell_counts,
            index.block_cell_counts,
            eager,
            index.forest_flags.astype(np.int32),
        )
        native_reader = make_amrvac_v5_block_reader(descriptor, index, binding)
        array_reader = array_block_reader(eager)
        leaf_count = int(index.leaf_count)
        capacity = min(57, leaf_count)
        regions = []
        for label, lower, upper in roi_bounds(fixture):
            selection, selection_report = measure_selection(
                selection_args(fixture, lower, upper), repeats
            )
            array_report, array_output, array_accumulator = measure_execution(
                fixture,
                array_reader,
                "array",
                i3(0, 1, 2),
                selection,
                capacity,
                warmups,
                repeats,
            )
            native_report, native_output, native_accumulator = measure_execution(
                fixture,
                native_reader,
                "native",
                native_fields,
                selection,
                capacity,
                warmups,
                repeats,
            )
            equal = bool(
                np.array_equal(
                    native_output.view(np.uint64), array_output.view(np.uint64)
                )
                and np.array_equal(
                    native_accumulator.view(np.uint64),
                    array_accumulator.view(np.uint64),
                )
            )
            if not equal:
                raise AssertionError("real native and eager-array LFE outputs differ")
            current_comparison = compare_safe_current_cells(
                native_output,
                selection,
                current_curl,
                index.block_cell_counts,
            )
            regions.append(
                {
                    "label": label,
                    "region_lower": lower.tolist(),
                    "region_upper": upper.tolist(),
                    "primary_count": int(selection.leaf_ids.size),
                    "selection": selection_report,
                    "capacity": capacity,
                    "native_array_bitwise_equal": equal,
                    "current_safe_cell_comparison": current_comparison,
                    "array": array_report,
                    "native": native_report,
                }
            )
        return {
            "status": "complete",
            "path": str(path),
            "file_bytes": path.stat().st_size,
            "leaf_count": leaf_count,
            "max_level": forest.max_level,
            "block_shape": index.block_cell_counts.tolist(),
            "magnetic_field_ids": native_fields.tolist(),
            "metadata_index_seconds": indexed - metadata_started,
            "forest_bind_seconds": bound - indexed,
            "metadata_total_seconds": bound - metadata_started,
            "metadata_bytes_read": int(index.offset_blocks),
            "current_eager_seconds": eager_seconds,
            "current_eager_output_bytes": int(eager.nbytes),
            "current_derived": current_curl_report,
            "regions": regions,
        }
    finally:
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "standard"), default="standard")
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--dat",
        type=Path,
        help="reserved for an optional real DAT-003/current bridge run",
    )
    args = parser.parse_args()
    default_warmups, default_repeats = profile_parameters(args.profile)
    warmups = default_warmups if args.warmups is None else args.warmups
    repeats = default_repeats if args.repeats is None else args.repeats
    if warmups < 0 or repeats <= 0:
        raise ValueError("warmups must be nonnegative and repeats positive")
    if args.dat is not None and not args.dat.is_file():
        raise FileNotFoundError(args.dat)

    metadata_started = time.perf_counter()
    fixture = make_fixture()
    metadata_seconds = time.perf_counter() - metadata_started
    leaf_count = int(fixture.leaf_node_ids.size)
    capacities = sorted({min(57, leaf_count), leaf_count})
    cases = []
    with tempfile.TemporaryDirectory(prefix="simesh-lfe-native-") as directory:
        dat_path = Path(directory) / "lfe-synthetic-refined.dat"
        file_descriptor, native_reader, native_record = open_native_fixture(
            dat_path, fixture
        )
        try:
            array_reader = array_block_reader(fixture.backing)
            for label, lower, upper in roi_bounds(fixture):
                selection, selection_report = measure_selection(
                    selection_args(fixture, lower, upper), repeats
                )
                capacity_reports = []
                baseline_output = None
                baseline_accumulator = None
                for capacity in capacities:
                    array_report, array_output, array_accumulator = (
                        measure_execution(
                            fixture,
                            array_reader,
                            "array",
                            i3(0, 1, 2),
                            selection,
                            capacity,
                            warmups,
                            repeats,
                        )
                    )
                    native_report, native_output, native_accumulator = (
                        measure_execution(
                            fixture,
                            native_reader,
                            "native",
                            i3(0, 1, 2),
                            selection,
                            capacity,
                            warmups,
                            repeats,
                        )
                    )
                    backend_equal = bool(
                        np.array_equal(
                            native_output.view(np.uint64),
                            array_output.view(np.uint64),
                        )
                        and np.array_equal(
                            native_accumulator.view(np.uint64),
                            array_accumulator.view(np.uint64),
                        )
                    )
                    if not backend_equal:
                        raise AssertionError("native and array LFE outputs differ")
                    if baseline_output is None:
                        baseline_output = native_output
                        baseline_accumulator = native_accumulator
                    capacity_equal = bool(
                        np.array_equal(
                            native_output.view(np.uint64),
                            baseline_output.view(np.uint64),
                        )
                        and np.array_equal(
                            native_accumulator.view(np.uint64),
                            baseline_accumulator.view(np.uint64),
                        )
                    )
                    if not capacity_equal:
                        raise AssertionError("LFE output changed with capacity")
                    capacity_reports.append(
                        {
                            "capacity": capacity,
                            "native_array_bitwise_equal": backend_equal,
                            "bitwise_equal_to_minimum_capacity": capacity_equal,
                            "array": array_report,
                            "native": native_report,
                        }
                    )
                cases.append(
                    {
                        "label": label,
                        "region_lower": lower.tolist(),
                        "region_upper": upper.tolist(),
                        "primary_count": int(selection.leaf_ids.size),
                        "selection": selection_report,
                        "capacities": capacity_reports,
                    }
                )
        finally:
            os.close(file_descriptor)

    report = {
        "capability": "LFE-001",
        "profile": args.profile,
        "environment": environment_record(),
        "parameters": {
            "warmups": warmups,
            "repeats": repeats,
            "root_shape": fixture.root_shape.tolist(),
            "block_shape": fixture.block_counts.tolist(),
            "leaf_count": leaf_count,
            "max_level": fixture.max_level,
            "capacities": capacities,
            "array_reader": True,
            "cold_warm_policy": "one cold sample, optional warmups, then raw repetitions",
        },
        "synthetic_metadata_build_validate_seconds": metadata_seconds,
        "synthetic_backing_bytes": int(fixture.backing.nbytes),
        "native_fixture": native_record,
        "regions": cases,
        "real_dat": (
            {"status": "not_requested"}
            if args.dat is None
            else real_dat_workflow(args.dat, warmups, repeats)
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
