"""Milestone workflow benchmark for native selective AMRVAC v5 reads."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
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

import simesh_rewrite.amrvac_dat as dat_module
import simesh_rewrite.amrvac_dat_reader as reader_module
from simesh.amrvac.datio import (
    get_metadata,
    read_blocks_sequential,
    update_header,
    write_forest_tree,
    write_header,
)
from simesh_rewrite.amrvac_dat import (
    bind_amrvac_v5_forest,
    read_amrvac_v5_index,
)
from simesh_rewrite.amrvac_dat_reader import make_amrvac_v5_block_reader
from simesh_rewrite.balance import validate_refined_all_touch_2to1
from simesh_rewrite.blockio import array_block_reader, read_blocks_into
from simesh_rewrite.refined_geometry import refined_leaf_geometry
from simesh_rewrite.repeated_sampling import (
    execute_refined_trilinear_points_from_blocks,
    execute_refined_zero_order_points_from_blocks,
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
def count_reader_preads(counter: PreadCounter):
    original = reader_module._pread_exact

    def wrapped(file_descriptor, byte_count, offset, *, section):
        result = original(
            file_descriptor,
            byte_count,
            offset,
            section=section,
        )
        counter.record(byte_count, section)
        return result

    reader_module._pread_exact = wrapped
    try:
        yield
    finally:
        reader_module._pread_exact = original


def metadata_lifecycle(path: Path, repeats: int) -> tuple[dict, object, object]:
    raw: list[dict] = []
    retained_index = None
    retained_binding = None
    for _ in range(repeats):
        started = time.perf_counter()
        file_descriptor = os.open(path, os.O_RDONLY)
        opened = time.perf_counter()
        try:
            index = read_amrvac_v5_index(file_descriptor)
            indexed = time.perf_counter()
            binding = bind_amrvac_v5_forest(index)
            bound = time.perf_counter()
        finally:
            os.close(file_descriptor)
        raw.append(
            {
                "open_seconds": opened - started,
                "index_seconds": indexed - opened,
                "bind_seconds": bound - indexed,
                "total_seconds": bound - started,
            }
        )
        retained_index = index
        retained_binding = binding
    fields = ("open_seconds", "index_seconds", "bind_seconds", "total_seconds")
    report = {
        "raw_repetitions": raw,
        "summary": {
            field: scalar_summary([sample[field] for sample in raw])
            for field in fields
        },
        "metadata_bytes_read": int(retained_index.offset_blocks),
        "index_array_bytes": int(
            retained_index.domain_lower.nbytes
            + retained_index.domain_upper.nbytes
            + retained_index.domain_cell_counts.nbytes
            + retained_index.block_cell_counts.nbytes
            + retained_index.periodic.nbytes
            + retained_index.parameter_values.nbytes
            + retained_index.forest_flags.nbytes
            + retained_index.block_levels.nbytes
            + retained_index.block_coordinates.nbytes
            + retained_index.block_offsets.nbytes
        ),
        "binding_array_bytes": int(
            retained_binding.root_shape.nbytes
            + retained_binding.coord_to_rank.nbytes
            + retained_binding.rank_to_coord.nbytes
            + sum(value.nbytes for value in retained_binding.forest[:-1])
        ),
    }
    return report, retained_index, retained_binding


def evenly_spaced_ids(count: int, total: int) -> np.ndarray:
    if count >= total:
        return np.arange(total, dtype=np.int64)
    return np.unique(np.linspace(0, total - 1, count, dtype=np.int64))


def timed_native_read(
    reader,
    block_ids: np.ndarray,
    field_ids: np.ndarray,
    block_shape: np.ndarray,
    repeats: int,
) -> tuple[dict, np.ndarray]:
    destination = np.empty(
        (
            block_ids.size,
            field_ids.size,
            *(int(value) for value in block_shape),
        ),
        dtype=np.float64,
    )
    zero = i3(0, 0, 0)
    raw: list[dict] = []
    for _ in range(repeats + 1):
        counter = PreadCounter()
        started = time.perf_counter()
        with count_reader_preads(counter):
            read_blocks_into(
                reader,
                zero,
                block_shape,
                block_ids,
                field_ids,
                destination,
                zero,
            )
        elapsed = time.perf_counter() - started
        record = {"seconds": elapsed, **counter.as_dict()}
        if _ == 0:
            first = record
        else:
            raw.append(record)
    timing = scalar_summary([sample["seconds"] for sample in raw])
    if any(
        sample[key] != raw[0][key]
        for sample in raw[1:]
        for key in ("calls", "bytes", "header_calls", "payload_calls")
    ):
        raise AssertionError("native read call/byte counts changed")
    report = {
        "first_after_open": first,
        "warm_timing": timing,
        "pread": {key: raw[0][key] for key in raw[0] if key != "seconds"},
    }
    return report, destination.copy()


def tdm_workflow(path: Path, repeats: int) -> dict:
    metadata, index, binding = metadata_lifecycle(path, repeats)
    field_ids = i3(4, 5, 6)
    current_samples: list[float] = []
    current_payload = None
    for _ in range(repeats):
        started = time.perf_counter()
        current_payload = read_blocks_sequential(str(path), field_ids.tolist())
        current_samples.append(time.perf_counter() - started)

    file_descriptor = os.open(path, os.O_RDONLY)
    try:
        reader = make_amrvac_v5_block_reader(
            file_descriptor, index, binding
        )
        selections = []
        for requested in (4, 16, index.leaf_count):
            block_ids = evenly_spaced_ids(requested, index.leaf_count)
            read_report, actual = timed_native_read(
                reader,
                block_ids,
                field_ids,
                index.block_cell_counts,
                repeats,
            )
            expected = np.ascontiguousarray(current_payload[block_ids])
            if not np.array_equal(actual.view(np.uint64), expected.view(np.uint64)):
                raise AssertionError("native tdm read differs from current eager read")
            useful_bytes = int(actual.nbytes)
            read_report.update(
                {
                    "requested_blocks": int(block_ids.size),
                    "field_ids": field_ids.tolist(),
                    "useful_output_bytes": useful_bytes,
                    "payload_read_over_useful": (
                        read_report["pread"]["payload_bytes"] / useful_bytes
                    ),
                    "reader_retained_offset_bytes": int(
                        reader.state.block_offsets.nbytes
                    ),
                    "bitwise_current_equal": True,
                }
            )
            selections.append(read_report)

        traced_ids = evenly_spaced_ids(16, index.leaf_count)
        traced_output = np.empty((traced_ids.size, 3, 10, 10, 10))
        zero = i3(0, 0, 0)
        tracemalloc.start()
        before_current, _ = tracemalloc.get_traced_memory()
        read_blocks_into(
            reader,
            zero,
            index.block_cell_counts,
            traced_ids,
            field_ids,
            traced_output,
            zero,
        )
        after_current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    finally:
        os.close(file_descriptor)

    return {
        "path": str(path),
        "file_bytes": path.stat().st_size,
        "leaf_count": index.leaf_count,
        "block_shape": index.block_cell_counts.tolist(),
        "metadata_lifecycle": metadata,
        "current_eager_all_blocks": {
            **scalar_summary(current_samples),
            "output_bytes": int(current_payload.nbytes),
        },
        "native_selections": selections,
        "representative_trace": {
            "traced_current_delta_bytes": after_current - before_current,
            "traced_peak_delta_bytes": peak - before_current,
        },
    }


def repack_weno_regular_fields(
    source_path: Path,
    target_path: Path,
    source_fields: tuple[int, int, int],
) -> dict:
    source_descriptor = os.open(source_path, os.O_RDONLY)
    try:
        source_index = read_amrvac_v5_index(source_descriptor)
        if not source_index.staggered:
            raise ValueError("WENO bridge source is expected to be staggered")
        if source_index.byte_order != ("<" if sys.byteorder == "little" else ">"):
            raise ValueError("WENO bridge setup requires host-endian source bytes")
        if any(value < 0 or value >= source_index.field_count for value in source_fields):
            raise ValueError("WENO bridge source field is out of range")

        current_header, _, _ = get_metadata(str(source_path))
        target_header = update_header(
            current_header,
            nw=len(source_fields),
            w_names=[source_index.field_names[field] for field in source_fields],
            staggered=False,
        )
        block_volume = int(np.prod(source_index.block_cell_counts))
        field_bytes = block_volume * 8
        record_bytes = 24 + len(source_fields) * field_bytes
        target_offsets = (
            int(target_header["offset_blocks"])
            + np.arange(source_index.leaf_count, dtype=np.int64) * record_bytes
        )
        setup_started = time.perf_counter()
        bytes_read = 0
        with target_path.open("wb") as target:
            write_header(target, target_header)
            write_forest_tree(
                target,
                target_header,
                source_index.forest_flags.astype(np.int32),
                (
                    source_index.block_levels.astype(np.int32),
                    (source_index.block_coordinates + 1).astype(np.int32),
                    target_offsets,
                ),
            )
            for block_offset in source_index.block_offsets:
                ghost = os.pread(source_descriptor, 24, int(block_offset))
                if len(ghost) != 24:
                    raise OSError("short WENO ghost-header read")
                if any(ghost):
                    raise ValueError("WENO bridge requires zero saved ghosts")
                target.write(ghost)
                bytes_read += 24
                for field in source_fields:
                    offset = int(block_offset) + 24 + field * field_bytes
                    raw = os.pread(source_descriptor, field_bytes, offset)
                    if len(raw) != field_bytes:
                        raise OSError("short WENO regular-field read")
                    target.write(raw)
                    bytes_read += field_bytes
        setup_seconds = time.perf_counter() - setup_started
    finally:
        os.close(source_descriptor)
    return {
        "seconds": setup_seconds,
        "source_bytes_read": bytes_read,
        "target_bytes": target_path.stat().st_size,
        "source_fields": list(source_fields),
        "target_field_names": target_header["w_names"],
    }


def selected_owner_points(index, binding, point_count: int) -> tuple[np.ndarray, np.ndarray]:
    forest = binding.forest
    leaf_ids = np.arange(index.leaf_count, dtype=np.int64)
    levels = forest.node_levels[forest.leaf_node_ids]
    coordinates = forest.node_coords[forest.leaf_node_ids]
    scales = np.exp2(levels - 1).astype(np.float64)
    centers = (coordinates + 0.5) / (
        binding.root_shape[None, :] * scales[:, None]
    )
    distance = np.sum((centers - 0.5) ** 2, axis=1)
    owners = np.sort(np.argpartition(distance, 3)[:4].astype(np.int64))
    bounds, spacing = refined_leaf_geometry(
        index.domain_lower,
        index.domain_upper,
        binding.root_shape,
        index.domain_cell_counts,
        index.block_cell_counts,
        forest.node_levels,
        forest.node_coords,
        forest.leaf_node_ids,
        owners,
    )
    fractions = np.asarray(
        ((0.25, 0.75, 7.75), (7.75, 0.25, 0.75), (0.75, 7.75, 0.25)),
        dtype=np.float64,
    )
    points = np.empty((point_count, 3), dtype=np.float64)
    for point in range(point_count):
        slot = point % owners.size
        points[point] = bounds[slot, 0] + fractions[point % 3] * spacing[slot]
    return owners, np.ascontiguousarray(points)


def rps_arguments(index, binding, reader, points, output, *, trilinear: bool):
    forest = binding.forest
    common = (
        reader,
        points,
        i3(0, 1, 2),
        index.domain_lower,
        index.domain_upper,
        binding.root_shape,
        index.domain_cell_counts,
        index.block_cell_counts,
        forest.max_level,
        binding.coord_to_rank,
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
            np.zeros((3, 6), dtype=np.uint8),
            i3(-1, -1, -1),
            128,
            output,
        )
    return (*common, 8, output)


def timed_rps(execute, arguments: tuple, repeats: int) -> tuple[dict, object]:
    samples: list[dict] = []
    stats = None
    for _ in range(repeats + 1):
        counter = PreadCounter()
        started = time.perf_counter()
        if isinstance(arguments[0].state, reader_module._AMRVACV5BlockReaderState):
            with count_reader_preads(counter):
                current_stats = execute(*arguments)
        else:
            current_stats = execute(*arguments)
        elapsed = time.perf_counter() - started
        if stats is None:
            stats = current_stats
        elif stats != current_stats:
            raise AssertionError("RPS stats changed across repetitions")
        record = {"seconds": elapsed, **counter.as_dict()}
        if _ == 0:
            first = record
        else:
            samples.append(record)
    return (
        {
            "first_after_open": first,
            "warm_timing": scalar_summary([value["seconds"] for value in samples]),
            "pread": {
                key: samples[0][key] for key in samples[0] if key != "seconds"
            },
            "stats": {name: int(getattr(stats, name)) for name in stats._fields},
        },
        stats,
    )


def weno_bridge_workflow(
    source_path: Path,
    repeats: int,
    point_count: int,
) -> dict:
    source_metadata, source_index, source_binding = metadata_lifecycle(
        source_path, repeats
    )
    if not source_index.staggered:
        raise ValueError("configured WENO source is not staggered")
    source_descriptor = os.open(source_path, os.O_RDONLY)
    try:
        try:
            make_amrvac_v5_block_reader(
                source_descriptor, source_index, source_binding
            )
        except ValueError as error:
            staggered_rejection = "staggered" in str(error)
        else:
            staggered_rejection = False
    finally:
        os.close(source_descriptor)
    if not staggered_rejection:
        raise AssertionError("staggered source was not rejected")

    with tempfile.TemporaryDirectory(prefix="simesh-weno-bridge-") as directory:
        target_path = Path(directory) / "weno-regular-b.dat"
        repack = repack_weno_regular_fields(
            source_path, target_path, (4, 5, 6)
        )
        target_metadata, index, binding = metadata_lifecycle(target_path, repeats)
        if index.staggered or index.geometry != "Cartesian_3D" or np.any(index.periodic):
            raise AssertionError("WENO bridge has unexpected numerical metadata")
        validate_refined_all_touch_2to1(
            binding.root_shape,
            binding.coord_to_rank,
            binding.forest.root_node_ids,
            binding.forest.node_levels,
            binding.forest.node_coords,
            binding.forest.child_node_ids,
            binding.forest.node_leaf_ids,
            binding.forest.leaf_node_ids,
        )
        owner_ids, points = selected_owner_points(index, binding, point_count)
        file_descriptor = os.open(target_path, os.O_RDONLY)
        try:
            native_reader = make_amrvac_v5_block_reader(
                file_descriptor, index, binding
            )
            native_zero = np.empty((point_count, 3), dtype=np.float64)
            native_tri = np.empty_like(native_zero)
            zero_report, _ = timed_rps(
                execute_refined_zero_order_points_from_blocks,
                rps_arguments(
                    index,
                    binding,
                    native_reader,
                    points,
                    native_zero,
                    trilinear=False,
                ),
                repeats,
            )
            tri_report, tri_stats = timed_rps(
                execute_refined_trilinear_points_from_blocks,
                rps_arguments(
                    index,
                    binding,
                    native_reader,
                    points,
                    native_tri,
                    trilinear=True,
                ),
                repeats,
            )
        finally:
            os.close(file_descriptor)

        eager_started = time.perf_counter()
        eager_backing = read_blocks_sequential(
            str(source_path), [4, 5, 6]
        )
        eager_seconds = time.perf_counter() - eager_started
        array_zero = np.empty_like(native_zero)
        array_tri = np.empty_like(native_tri)
        array_zero_report, _ = timed_rps(
            execute_refined_zero_order_points_from_blocks,
            rps_arguments(
                index,
                binding,
                array_block_reader(eager_backing),
                points,
                array_zero,
                trilinear=False,
            ),
            repeats,
        )
        array_tri_report, array_tri_stats = timed_rps(
            execute_refined_trilinear_points_from_blocks,
            rps_arguments(
                index,
                binding,
                array_block_reader(eager_backing),
                points,
                array_tri,
                trilinear=True,
            ),
            repeats,
        )
        if not np.array_equal(native_zero.view(np.uint64), array_zero.view(np.uint64)):
            raise AssertionError("native WENO zero-order values differ")
        if not np.array_equal(native_tri.view(np.uint64), array_tri.view(np.uint64)):
            raise AssertionError("native WENO trilinear values differ")
        if tri_stats != array_tri_stats:
            raise AssertionError("native/array RPS trilinear stats differ")

        block_bytes = 3 * int(np.prod(index.block_cell_counts)) * 8
        zero_report.update(
            {
                "bitwise_array_equal": True,
                "requested_owner_ids": owner_ids.tolist(),
                "read_over_requested_owner_bytes": (
                    zero_report["pread"]["payload_bytes"]
                    / (len(owner_ids) * block_bytes)
                ),
            }
        )
        tri_report.update(
            {
                "bitwise_array_equal": True,
                "support_load_amplification": (
                    tri_stats.selected_load_count / tri_stats.owner_count
                ),
                "read_payload_bytes": tri_report["pread"]["payload_bytes"],
            }
        )
        return {
            "source_path": str(source_path),
            "source_file_bytes": source_path.stat().st_size,
            "source_metadata_lifecycle": source_metadata,
            "staggered_reader_rejected": staggered_rejection,
            "bridge_setup": repack,
            "bridge_metadata_lifecycle": target_metadata,
            "point_count": point_count,
            "owner_count": int(owner_ids.size),
            "native_zero": zero_report,
            "native_trilinear": tri_report,
            "array_zero": array_zero_report,
            "array_trilinear": array_tri_report,
            "current_eager_regular_fields": {
                "seconds": eager_seconds,
                "payload_bytes": int(eager_backing.nbytes),
            },
            "native_reader_retained_offset_bytes": int(index.block_offsets.nbytes),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "standard"), default="standard")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tdm", type=Path, default=REPOSITORY_ROOT / "data/tdm.dat")
    parser.add_argument(
        "--weno",
        type=Path,
        default=REPOSITORY_ROOT / "data/weno509_sub_0000.dat",
    )
    parser.add_argument("--skip-weno", action="store_true")
    args = parser.parse_args()
    profiles = {
        "smoke": {"repeats": 2, "points": 256, "weno": False},
        "standard": {"repeats": 5, "points": 2048, "weno": True},
    }
    config = profiles[args.profile]
    if not args.tdm.exists():
        raise FileNotFoundError(args.tdm)

    rss_before = current_rss_bytes()
    started = time.perf_counter()
    tdm = tdm_workflow(args.tdm, config["repeats"])
    weno = None
    weno_skip_reason = None
    if config["weno"] and not args.skip_weno:
        if args.weno.exists():
            weno = weno_bridge_workflow(
                args.weno,
                config["repeats"],
                config["points"],
            )
        else:
            weno_skip_reason = "configured WENO fixture is unavailable"
    else:
        weno_skip_reason = "disabled by profile or --skip-weno"
    elapsed = time.perf_counter() - started
    rss_after = current_rss_bytes()

    report = {
        "group": "Native Selective AMRVAC Read",
        "capabilities": ["DAT-001", "DAT-002", "DAT-003"],
        "profile": args.profile,
        "configuration": config,
        "environment": environment_record(),
        "tdm": tdm,
        "weno_regular_field_bridge": weno,
        "weno_skip_reason": weno_skip_reason,
        "process": {
            "wall_seconds": elapsed,
            "rss_before_bytes": rss_before,
            "rss_after_bytes": rss_after,
            "peak_rss_bytes": peak_rss_bytes(),
        },
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
