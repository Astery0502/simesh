"""Generate AMRVAC scaling reports for canonical and legacy implementations."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Callable, Iterable

_IMPORT_ERROR = None
try:
    import numpy as np

    from simesh.amrvac.datio import (
        SIZE_DOUBLE,
        SIZE_INT,
        get_metadata,
        header_template,
        read_blocks_sequential,
        update_header,
        write_datfile_from_sfc,
    )
    from simesh.legacy.frontends.amrvac import datio as legacy_datio
    from simesh.legacy.geometry.amr.amr_forest import AMRForest as LegacyAMRForest
    from simesh.legacy.meshes.amr_mesh import AMRMesh as LegacyAMRMesh
    from simesh.utils.lib.amr.forest import AMRForest
    from simesh.utils.lib.amr.mesh import AMRMesh
    from simesh.utils.lib.amr.morton import fill_morton_mapping3D
except ModuleNotFoundError as exc:
    _IMPORT_ERROR = exc


PROFILE_SIZES = {
    "smoke": (50,),
    "standard": (50, 100, 200),
    "large": (50, 100, 200, 300, 400, 500),
}
DEFAULT_BLOCK_NX = (10, 10, 10)
DEFAULT_CACHE_DIR = Path(".benchmark-data")
DEFAULT_OUTPUT_ROOT = Path("benchmark-results")
OPERATION_LABELS = {
    "read": "Block Read",
    "write": "Block Write",
    "ghost_exchange": "Ghost-Cell Exchange",
    "workflow": "End-to-End Workflow",
}
OPERATION_ORDER = ("read", "write", "ghost_exchange", "workflow")
IMPLEMENTATION_STYLES = {
    "canonical": {"label": "Canonical Cython-backed", "color": "#0072B2", "marker": "o"},
    "legacy": {"label": "Legacy Python-first", "color": "#D55E00", "marker": "s"},
}


@dataclass(frozen=True)
class Case:
    domain_nx: tuple[int, int, int]
    block_nx: tuple[int, int, int]
    nw: int
    ghost_width: int
    seed: int

    @property
    def label(self) -> str:
        return (
            f"nx{self.domain_nx[0]}x{self.domain_nx[1]}x{self.domain_nx[2]}"
            f"_block{self.block_nx[0]}x{self.block_nx[1]}x{self.block_nx[2]}"
            f"_nw{self.nw}_g{self.ghost_width}_seed{self.seed}"
        )

    @property
    def total_cells(self) -> int:
        return int(np.prod(self.domain_nx))

    @property
    def root_grid(self) -> tuple[int, int, int]:
        return tuple(int(d // b) for d, b in zip(self.domain_nx, self.block_nx))

    @property
    def nleafs(self) -> int:
        return int(np.prod(self.root_grid))


def parse_triplet(values: Iterable[str], name: str) -> tuple[int, int, int]:
    parsed = tuple(int(value) for value in values)
    if len(parsed) != 3 or any(value <= 0 for value in parsed):
        raise argparse.ArgumentTypeError(f"{name} must contain three positive integers")
    return parsed


def parse_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("--sizes must be a comma-separated list of positive integers")
    return sizes


def build_tree(case: Case):
    root_grid = case.root_grid
    nleafs = case.nleafs
    ig2morton = np.zeros(root_grid, dtype=np.uint32)
    morton2ig = np.zeros((nleafs, 3), dtype=np.uint32)
    fill_morton_mapping3D(ig2morton, morton2ig, *root_grid)

    is_leaf = np.ones(nleafs, dtype=np.int32)
    block_lvls = np.ones(nleafs, dtype=np.int32)
    block_ixs = morton2ig.astype(np.int32) + 1
    block_offsets = np.zeros(nleafs, dtype=np.int64)
    return is_leaf, (block_lvls, block_ixs, block_offsets)


def build_header(case: Case) -> dict:
    header = header_template.copy()
    header["nw"] = case.nw
    header["w_names"] = [f"w{i}" for i in range(case.nw)]
    header["levmax"] = 1
    header["nleafs"] = case.nleafs
    header["nparents"] = 0
    header["xmin"] = np.array([0.0, 0.0, 0.0], dtype=np.double)
    header["xmax"] = np.array([1.0, 1.0, 1.0], dtype=np.double)
    header["domain_nx"] = np.array(case.domain_nx, dtype=np.int32)
    header["block_nx"] = np.array(case.block_nx, dtype=np.int32)
    header["periodic"] = np.array([False, False, False], dtype=bool)
    return update_header(header)


def block_offsets(header: dict) -> np.ndarray:
    block_bytes = (
        2 * int(header["ndim"]) * SIZE_INT
        + int(np.prod(header["block_nx"])) * int(header["nw"]) * SIZE_DOUBLE
    )
    return int(header["offset_blocks"]) + np.arange(int(header["nleafs"]), dtype=np.int64) * block_bytes


def synthetic_sfc_data(case: Case) -> np.ndarray:
    data = np.empty((case.nleafs, case.nw, *case.block_nx), dtype=np.float64)
    x = np.arange(case.block_nx[0], dtype=np.float64)[:, None, None]
    y = np.arange(case.block_nx[1], dtype=np.float64)[None, :, None]
    z = np.arange(case.block_nx[2], dtype=np.float64)[None, None, :]
    base = x + 0.01 * y + 0.0001 * z + case.seed * 1.0e-9

    for ileaf in range(case.nleafs):
        for ifield in range(case.nw):
            data[ileaf, ifield] = base + 1000.0 * ifield + float(ileaf)
    return data


def ensure_input_dat(case: Case, cache_dir: Path, force_generate: bool) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{case.label}.dat"
    if path.exists() and not force_generate:
        return path

    data = synthetic_sfc_data(case)
    header = build_header(case)
    is_leaf, tree_info = build_tree(case)
    write_datfile_from_sfc(str(path), data, header, is_leaf, tree_info, overwrite=True)
    return path


def read_legacy_blocks(path: Path, field_indices: list[int] | None = None) -> np.ndarray:
    with path.open("rb") as fb:
        header = legacy_datio.get_header(fb)
        _forest = legacy_datio.get_forest(fb)
        tree = legacy_datio.get_tree_info(fb)
        block_shape = tuple(int(value) for value in header["block_nx"])
        if field_indices is None:
            field_indices = list(range(int(header["nw"])))

        data = np.empty((int(header["nleafs"]), len(field_indices), *block_shape), dtype=np.float64)
        for ileaf, offset in enumerate(tree[2]):
            for iout, ifield in enumerate(field_indices):
                data[ileaf, iout] = legacy_datio.get_single_block_field_data(
                    fb,
                    int(offset),
                    block_shape,
                    int(ifield),
                    int(header["ndim"]),
                )
    return data


def write_legacy_datfile(path: Path, data: np.ndarray, header: dict, is_leaf: np.ndarray, tree_info) -> None:
    offsets = block_offsets(header)
    block_lvls, block_ixs, _old_offsets = tree_info
    legacy_layout = np.transpose(np.asarray(data, dtype=np.float64), (0, 2, 3, 4, 1))
    with path.open("wb") as fb:
        legacy_datio.write_header(fb, header)
        legacy_datio.write_forest_tree(fb, header, is_leaf, (block_lvls, block_ixs, offsets))
        legacy_datio.write_blocks(fb, legacy_layout, int(header["ndim"]), offsets)


def load_legacy_interior(mesh: LegacyAMRMesh, data: np.ndarray) -> None:
    mesh.data[...] = 0.0
    mesh.data[
        :,
        mesh.ixMmin[0] : mesh.ixMmax[0] + 1,
        mesh.ixMmin[1] : mesh.ixMmax[1] + 1,
        mesh.ixMmin[2] : mesh.ixMmax[2] + 1,
        :,
    ] = np.transpose(data, (0, 2, 3, 4, 1))


def build_mesh_pair(case: Case):
    if any(value % 2 != 0 for value in case.block_nx):
        raise ValueError(
            "legacy ghost-cell exchange requires even block_nx values; "
            f"got {case.block_nx}. Use --block-nx with even dimensions."
        )

    is_leaf, _tree_info = build_tree(case)
    ng1, ng2, ng3 = case.root_grid
    block_nx = np.array(case.block_nx, dtype=np.uint32)
    domain_nx = np.array(case.domain_nx, dtype=np.uint32)
    xmin = np.array([0.0, 0.0, 0.0], dtype=np.double)
    xmax = np.array([1.0, 1.0, 1.0], dtype=np.double)

    forest = AMRForest(3, ng1, ng2, ng3, is_leaf.astype(np.int32))
    mesh = AMRMesh(3, block_nx, domain_nx, xmin, xmax, np.uint32(case.ghost_width), np.uint32(case.nw), forest)

    legacy_forest = LegacyAMRForest(ng1, ng2, ng3, int(forest.nleafs))
    legacy_forest.read_forest(is_leaf.astype(bool))
    legacy_forest.build_connectivity()
    legacy_mesh = LegacyAMRMesh(
        (float(xmin[0]), float(xmax[0])),
        (float(xmin[1]), float(xmax[1])),
        (float(xmin[2]), float(xmax[2])),
        [f"w{i}" for i in range(case.nw)],
        block_nx.astype(int),
        domain_nx.astype(int),
        legacy_forest,
        nghostcells=case.ghost_width,
    )
    return mesh, legacy_mesh


def time_call(func: Callable[[], object], repetitions: int, warmups: int) -> tuple[list[float], object]:
    result = None
    for _ in range(warmups):
        result = func()

    durations = []
    for _ in range(repetitions):
        gc.collect()
        start = time.perf_counter()
        result = func()
        durations.append(time.perf_counter() - start)
    return durations, result


def summarize_durations(durations: list[float]) -> dict:
    return {
        "runs": durations,
        "median_seconds": median(durations),
        "min_seconds": min(durations),
        "max_seconds": max(durations),
        "mean_seconds": mean(durations),
        "stdev_seconds": stdev(durations) if len(durations) > 1 else 0.0,
    }


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def machine_metadata(profile: str, repetitions: int, warmups: int, cache_dir: Path) -> dict:
    return {
        "profile": profile,
        "repetitions": repetitions,
        "warmups": warmups,
        "cache_dir": str(cache_dir),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "simesh_commit": git_commit(),
    }


def estimate_case_bytes(case: Case) -> dict:
    data_bytes = case.total_cells * case.nw * SIZE_DOUBLE
    padded_cells_per_block = int(np.prod([value + 2 * case.ghost_width for value in case.block_nx]))
    ghost_mesh_bytes = case.nleafs * padded_cells_per_block * case.nw * SIZE_DOUBLE
    return {
        "interior_data_bytes": data_bytes,
        "dat_file_bytes": data_bytes + case.nleafs * 2 * 3 * SIZE_INT,
        "one_padded_mesh_bytes": ghost_mesh_bytes,
    }


def print_estimates(cases: list[Case]) -> None:
    print("Estimated per-case data sizes:")
    for case in cases:
        estimate = estimate_case_bytes(case)
        print(
            f"  {case.domain_nx}: input ~{estimate['dat_file_bytes'] / 1e9:.2f} GB, "
            f"one padded ghost mesh ~{estimate['one_padded_mesh_bytes'] / 1e9:.2f} GB"
        )


def validate_small_case(case: Case, path: Path, output_dir: Path) -> None:
    print(f"Validating small fixture for {case.label}...")
    canonical = read_blocks_sequential(str(path), None)
    legacy = read_legacy_blocks(path, None)
    np.testing.assert_allclose(canonical, legacy, rtol=1e-12, atol=1e-12)

    mesh, legacy_mesh = build_mesh_pair(case)
    mesh.load_interior_data(np.ascontiguousarray(canonical, dtype=np.double))
    load_legacy_interior(legacy_mesh, canonical)
    mesh.apply_ghost_cells()
    legacy_mesh.getbc()
    np.testing.assert_allclose(mesh.padded_view(), legacy_mesh.data, rtol=1e-12, atol=1e-12)

    header, is_leaf, tree_info = get_metadata(str(path))
    canonical_out = output_dir / "validation-canonical.dat"
    legacy_out = output_dir / "validation-legacy.dat"
    write_datfile_from_sfc(str(canonical_out), canonical, header, is_leaf, tree_info, overwrite=True)
    write_legacy_datfile(legacy_out, canonical, header, is_leaf, tree_info)
    np.testing.assert_allclose(read_blocks_sequential(str(canonical_out), None), canonical)
    np.testing.assert_allclose(read_blocks_sequential(str(legacy_out), None), canonical)


def make_record(
    case: Case,
    operation: str,
    implementation: str,
    durations: list[float],
    file_size_bytes: int,
    repetitions: int,
    warmups: int,
) -> dict:
    record = {
        "operation": operation,
        "implementation": implementation,
        "domain_nx": "x".join(str(value) for value in case.domain_nx),
        "block_nx": "x".join(str(value) for value in case.block_nx),
        "nw": case.nw,
        "ghost_width": case.ghost_width,
        "nleafs": case.nleafs,
        "total_cells": case.total_cells,
        "file_size_bytes": file_size_bytes,
        "repetitions": repetitions,
        "warmups": warmups,
        "speedup_ratio": "",
        "throughput_gb_s": "",
    }
    record.update(summarize_durations(durations))
    if operation in {"read", "write", "workflow"} and record["median_seconds"] > 0:
        record["throughput_gb_s"] = file_size_bytes / record["median_seconds"] / 1e9
    return record


def add_speedups(records: list[dict]) -> None:
    grouped: dict[tuple[str, str], dict[str, dict]] = {}
    for record in records:
        key = (record["operation"], record["domain_nx"])
        grouped.setdefault(key, {})[record["implementation"]] = record

    for implementations in grouped.values():
        canonical = implementations.get("canonical")
        legacy = implementations.get("legacy")
        if canonical is None or legacy is None:
            continue
        speedup = legacy["median_seconds"] / canonical["median_seconds"]
        canonical["speedup_ratio"] = speedup
        legacy["speedup_ratio"] = 1.0


def run_case(case: Case, input_path: Path, output_dir: Path, repetitions: int, warmups: int) -> list[dict]:
    header, is_leaf, tree_info = get_metadata(str(input_path))
    file_size = input_path.stat().st_size
    case_output = output_dir / case.label
    case_output.mkdir(parents=True, exist_ok=True)

    records = []
    print(f"Benchmarking {case.label}...")

    durations, read_data = time_call(
        lambda: read_blocks_sequential(str(input_path), None),
        repetitions,
        warmups,
    )
    records.append(make_record(case, "read", "canonical", durations, file_size, repetitions, warmups))

    durations, _legacy_read_data = time_call(
        lambda: read_legacy_blocks(input_path, None),
        repetitions,
        warmups,
    )
    records.append(make_record(case, "read", "legacy", durations, file_size, repetitions, warmups))

    data = np.ascontiguousarray(read_data, dtype=np.double)

    durations, _ = time_call(
        lambda: write_datfile_from_sfc(
            str(case_output / "canonical-write.dat"),
            data,
            header,
            is_leaf,
            tree_info,
            overwrite=True,
        ),
        repetitions,
        warmups,
    )
    records.append(make_record(case, "write", "canonical", durations, file_size, repetitions, warmups))

    durations, _ = time_call(
        lambda: write_legacy_datfile(case_output / "legacy-write.dat", data, header, is_leaf, tree_info),
        repetitions,
        warmups,
    )
    records.append(make_record(case, "write", "legacy", durations, file_size, repetitions, warmups))

    mesh, legacy_mesh = build_mesh_pair(case)
    durations, _ = time_call(
        lambda: (mesh.load_interior_data(data), mesh.apply_ghost_cells()),
        repetitions,
        warmups,
    )
    records.append(make_record(case, "ghost_exchange", "canonical", durations, file_size, repetitions, warmups))

    durations, _ = time_call(
        lambda: (load_legacy_interior(legacy_mesh, data), legacy_mesh.getbc()),
        repetitions,
        warmups,
    )
    records.append(make_record(case, "ghost_exchange", "legacy", durations, file_size, repetitions, warmups))

    def canonical_workflow():
        workflow_data = read_blocks_sequential(str(input_path), None)
        workflow_mesh, _ = build_mesh_pair(case)
        workflow_mesh.load_interior_data(np.ascontiguousarray(workflow_data, dtype=np.double))
        workflow_mesh.apply_ghost_cells()
        return write_datfile_from_sfc(
            str(case_output / "canonical-workflow.dat"),
            workflow_mesh.interior_view(),
            header,
            is_leaf,
            tree_info,
            overwrite=True,
        )

    durations, _ = time_call(canonical_workflow, repetitions, warmups)
    records.append(make_record(case, "workflow", "canonical", durations, file_size, repetitions, warmups))

    def legacy_workflow():
        workflow_data = read_legacy_blocks(input_path, None)
        _mesh, workflow_legacy_mesh = build_mesh_pair(case)
        load_legacy_interior(workflow_legacy_mesh, workflow_data)
        workflow_legacy_mesh.getbc()
        write_legacy_datfile(
            case_output / "legacy-workflow.dat",
            np.transpose(workflow_legacy_mesh.data[
                :,
                workflow_legacy_mesh.ixMmin[0] : workflow_legacy_mesh.ixMmax[0] + 1,
                workflow_legacy_mesh.ixMmin[1] : workflow_legacy_mesh.ixMmax[1] + 1,
                workflow_legacy_mesh.ixMmin[2] : workflow_legacy_mesh.ixMmax[2] + 1,
                :,
            ], (0, 4, 1, 2, 3)),
            header,
            is_leaf,
            tree_info,
        )

    durations, _ = time_call(legacy_workflow, repetitions, warmups)
    records.append(make_record(case, "workflow", "legacy", durations, file_size, repetitions, warmups))
    return records


def write_results(output_dir: Path, metadata: dict, records: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    add_speedups(records)

    with (output_dir / "results.json").open("w", encoding="utf-8") as fh:
        json.dump({"metadata": metadata, "results": records}, fh, indent=2)

    fieldnames = [
        "operation",
        "implementation",
        "domain_nx",
        "block_nx",
        "nw",
        "ghost_width",
        "nleafs",
        "total_cells",
        "file_size_bytes",
        "repetitions",
        "warmups",
        "median_seconds",
        "min_seconds",
        "max_seconds",
        "mean_seconds",
        "stdev_seconds",
        "speedup_ratio",
        "throughput_gb_s",
        "runs",
    ]
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = record.copy()
            row["runs"] = json.dumps(row["runs"])
            writer.writerow(row)

    write_markdown_summary(output_dir, metadata, records)


def write_markdown_summary(output_dir: Path, metadata: dict, records: list[dict]) -> None:
    lines = [
        "# AMRVAC Scaling Benchmark Report",
        "",
        "This report is the integrated entry point for the benchmark run. It combines",
        "the run parameters, summarized timings, raw-result artifact links, and curve",
        "figures generated from the same result set.",
        "",
        "## Parameters",
        "",
        f"- profile: `{metadata['profile']}`",
        f"- repetitions: `{metadata['repetitions']}`",
        f"- warmups: `{metadata['warmups']}`",
        f"- platform: `{metadata['platform']}`",
        f"- python: `{metadata['python'].split()[0]}`",
        f"- numpy: `{metadata['numpy']}`",
        f"- simesh commit: `{metadata.get('simesh_commit')}`",
        "",
        "## Artifacts",
        "",
        "- Raw JSON: [results.json](results.json)",
        "- Raw CSV: [results.csv](results.csv)",
        "- Figures: [figures/](figures/)",
        "",
        "## Results",
        "",
        "| operation | mesh | canonical median (s) | legacy median (s) | speedup |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    grouped: dict[tuple[str, str], dict[str, dict]] = {}
    for record in records:
        grouped.setdefault((record["operation"], record["domain_nx"]), {})[record["implementation"]] = record

    def sort_key(item):
        (operation, _mesh), implementations = item
        order = OPERATION_ORDER.index(operation) if operation in OPERATION_ORDER else len(OPERATION_ORDER)
        total_cells = min(record["total_cells"] for record in implementations.values())
        return order, total_cells

    for (operation, mesh), implementations in sorted(grouped.items(), key=sort_key):
        canonical = implementations.get("canonical")
        legacy = implementations.get("legacy")
        if canonical is None or legacy is None:
            continue
        lines.append(
            f"| {operation} | {mesh} | {canonical['median_seconds']:.6g} | "
            f"{legacy['median_seconds']:.6g} | {canonical['speedup_ratio']:.3g}x |"
        )
    lines.extend(
        [
            "",
            "## Curves",
            "",
            "### Runtime Scaling By Operation",
            "",
            "![Runtime scaling](figures/runtime_scaling.png)",
            "",
            "### Legacy / Canonical Ratio",
            "",
            "![Canonical speedup](figures/speedup.png)",
            "",
            "### Read/Write Throughput",
            "",
            "![Read/write throughput](figures/throughput.png)",
            "",
            "### Per-Operation Runtime And Ratio",
            "",
            "![Block read detail](figures/read.png)",
            "",
            "![Block write detail](figures/write.png)",
            "",
            "![Ghost-cell exchange detail](figures/ghost_exchange.png)",
            "",
            "![Workflow detail](figures/workflow.png)",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_figures(output_dir: Path, records: list[dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to generate benchmark figures. "
            "Install the benchmark extra or rerun with --no-figures."
        ) from exc

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    operations = [operation for operation in OPERATION_ORDER if any(r["operation"] == operation for r in records)]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    def operation_records(operation: str, implementation: str) -> list[dict]:
        subset = [
            record
            for record in records
            if record["operation"] == operation and record["implementation"] == implementation
        ]
        return sorted(subset, key=lambda record: record["total_cells"])

    def plot_runtime_axis(ax, operation: str, x_key: str = "total_cells") -> None:
        for implementation, style in IMPLEMENTATION_STYLES.items():
            subset = operation_records(operation, implementation)
            if not subset:
                continue
            ax.plot(
                [record[x_key] for record in subset],
                [record["median_seconds"] for record in subset],
                color=style["color"],
                marker=style["marker"],
                linewidth=2.0,
                markersize=5,
                label=style["label"],
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(OPERATION_LABELS.get(operation, operation))
        ax.set_ylabel("Median runtime (s)")

    def plot_ratio_axis(ax, operation: str, x_key: str = "total_cells") -> None:
        subset = [
            record
            for record in operation_records(operation, "canonical")
            if record["speedup_ratio"] != ""
        ]
        if subset:
            ax.plot(
                [record[x_key] for record in subset],
                [record["speedup_ratio"] for record in subset],
                color="#009E73",
                marker="D",
                linewidth=2.1,
                markersize=5,
                label="Legacy / canonical",
            )
        ax.axhline(1.0, color="#555555", linewidth=1.0, linestyle=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(OPERATION_LABELS.get(operation, operation))
        ax.set_ylabel("Legacy / canonical median runtime")
        ax.text(
            0.02,
            0.96,
            "above 1: canonical faster",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#444444",
        )

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), sharex=False)
    for ax, operation in zip(axes.flat, operations):
        plot_runtime_axis(ax, operation)
        ax.set_xlabel("Total cells")
    for ax in axes.flat[len(operations):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("AMRVAC Runtime Scaling: Canonical vs Legacy", y=1.04, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(figures_dir / "runtime_scaling.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), sharex=False)
    for ax, operation in zip(axes.flat, operations):
        plot_ratio_axis(ax, operation)
        ax.set_xlabel("Total cells")
    for ax in axes.flat[len(operations):]:
        ax.set_visible(False)
    fig.suptitle("Legacy / Canonical Runtime Ratio", y=1.04, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(figures_dir / "speedup.png", bbox_inches="tight")
    plt.close(fig)

    for operation in operations:
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
        plot_runtime_axis(axes[0], operation)
        axes[0].set_xlabel("Total cells")
        axes[0].legend(frameon=False)
        plot_ratio_axis(axes[1], operation)
        axes[1].set_xlabel("Total cells")
        axes[1].legend(frameon=False)
        fig.suptitle(f"{OPERATION_LABELS.get(operation, operation)} Scaling", fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(figures_dir / f"{operation}.png", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharex=False)
    for ax, operation in zip(axes, ("read", "write")):
        for implementation, style in IMPLEMENTATION_STYLES.items():
            subset = [
                record
                for record in operation_records(operation, implementation)
                if record["throughput_gb_s"] != ""
            ]
            if not subset:
                continue
            ax.plot(
                [record["total_cells"] for record in subset],
                [record["throughput_gb_s"] for record in subset],
                color=style["color"],
                marker=style["marker"],
                linewidth=2.0,
                markersize=5,
                label=style["label"],
            )
        ax.set_xscale("log")
        ax.set_xlabel("Total cells")
        ax.set_ylabel("Throughput (GB/s)")
        ax.set_title(OPERATION_LABELS[operation])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Read/Write Effective Throughput", y=1.08, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(figures_dir / "throughput.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    plot_runtime_axis(axes[0], "ghost_exchange", x_key="nleafs")
    axes[0].set_xlabel("Leaf blocks")
    axes[0].legend(frameon=False)
    plot_ratio_axis(axes[1], "ghost_exchange", x_key="nleafs")
    axes[1].set_xlabel("Leaf blocks")
    axes[1].legend(frameon=False)
    fig.suptitle("Ghost-Cell Exchange: Runtime and Ratio", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(figures_dir / "ghost_exchange.png", bbox_inches="tight")
    plt.close(fig)


def default_output_dir(profile: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return DEFAULT_OUTPUT_ROOT / f"{profile}-{stamp}"


def build_cases(args: argparse.Namespace) -> list[Case]:
    sizes = args.sizes if args.sizes is not None else PROFILE_SIZES[args.profile]
    block_nx = args.block_nx
    cases = []
    for size in sizes:
        domain_nx = (size, size, size)
        if any(domain % block != 0 for domain, block in zip(domain_nx, block_nx)):
            raise ValueError(f"domain_nx={domain_nx} must be divisible by block_nx={block_nx}")
        cases.append(
            Case(
                domain_nx=domain_nx,
                block_nx=block_nx,
                nw=args.nw,
                ghost_width=args.ghost_width,
                seed=args.seed,
            )
        )
    return cases


def clean_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILE_SIZES), default="standard")
    parser.add_argument("--sizes", type=parse_sizes, help="comma-separated cubic mesh sizes, e.g. 50,100,200")
    parser.add_argument("--block-nx", nargs=3, default=DEFAULT_BLOCK_NX, metavar=("BX", "BY", "BZ"))
    parser.add_argument("--nw", type=int, default=1)
    parser.add_argument("--ghost-width", type=int, default=2)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force-generate", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--clean-output", action="store_true")
    args = parser.parse_args(argv)

    args.block_nx = parse_triplet(args.block_nx, "--block-nx")
    if args.nw <= 0:
        parser.error("--nw must be positive")
    if args.ghost_width < 0:
        parser.error("--ghost-width must be non-negative")
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if _IMPORT_ERROR is not None:
        parser.error(
            "unable to import benchmark dependencies or compiled AMR modules: "
            f"{_IMPORT_ERROR}. Run with the project environment and build extensions first, "
            "for example `make build-amr` or `pip install -e .`."
        )

    output_dir = args.output if args.output is not None else default_output_dir(args.profile)
    if args.clean_output:
        clean_output_dir(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    cases = build_cases(args)
    print_estimates(cases)
    if args.profile == "large":
        print("Large profile is opt-in and may require multiple GB of memory and disk space.")

    input_paths = [ensure_input_dat(case, args.cache_dir, args.force_generate) for case in cases]
    if not args.skip_validation:
        validate_small_case(cases[0], input_paths[0], output_dir)

    all_records = []
    for case, input_path in zip(cases, input_paths):
        all_records.extend(run_case(case, input_path, output_dir, args.repetitions, args.warmups))

    metadata = machine_metadata(args.profile, args.repetitions, args.warmups, args.cache_dir)
    write_results(output_dir, metadata, all_records)
    if not args.no_figures:
        plot_figures(output_dir, all_records)

    print(f"Wrote benchmark report to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
