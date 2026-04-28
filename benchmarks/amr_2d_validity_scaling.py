"""Benchmark and validate the canonical 2D AMR mesh implementation.

The benchmark builds Cartesian 2D forests from a simple level-1 mesh up to
larger mixed coarse/fine AMR meshes. Each case validates an affine field, which
bilinear interpolation should reproduce exactly away from physical boundaries,
then times mesh construction, ghost-cell exchange, linear interpolation, and an
end-to-end workflow.
"""

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
from typing import Callable

_IMPORT_ERROR = None
try:
    import numpy as np

    from simesh.utils.lib.amr.forest import AMRForest
    from simesh.utils.lib.amr.mesh import AMRMesh
    from simesh.utils.lib.amr.morton import fill_morton_mapping3D
except ModuleNotFoundError as exc:
    _IMPORT_ERROR = exc


DEFAULT_BLOCK_NX = (8, 8)
DEFAULT_OUTPUT_ROOT = Path("benchmark-results")
PROFILE_SIZES = {
    "smoke": (4,),
    "standard": (4, 8, 16, 32),
    "large": (8, 16, 32, 64),
}
OPERATIONS = ("build_mesh", "ghost_exchange", "linear_interpolation", "workflow")
OPERATION_LABELS = {
    "build_mesh": "Forest + Mesh Build",
    "ghost_exchange": "Ghost-Cell Exchange",
    "linear_interpolation": "Bilinear Uniform Export",
    "workflow": "Build + Ghost + Export",
}
TOPOLOGIES = ("centered", "complex")


@dataclass(frozen=True)
class Case:
    root_blocks: int
    block_nx: tuple[int, int]
    ghost_width: int
    nfields: int
    interpolation_factor: int
    max_level: int
    topology: str

    @property
    def root_grid(self) -> tuple[int, int, int]:
        return (self.root_blocks, self.root_blocks, 1)

    @property
    def domain_nx(self) -> tuple[int, int]:
        return (self.root_blocks * self.block_nx[0], self.root_blocks * self.block_nx[1])

    @property
    def uniform_nx(self) -> tuple[int, int, int]:
        return (
            self.interpolation_factor * self.domain_nx[0],
            self.interpolation_factor * self.domain_nx[1],
            1,
        )

    @property
    def total_uniform_cells(self) -> int:
        return int(np.prod(self.uniform_nx))

    @property
    def label(self) -> str:
        return (
            f"roots{self.root_blocks}x{self.root_blocks}"
            f"_block{self.block_nx[0]}x{self.block_nx[1]}"
            f"_g{self.ghost_width}_fields{self.nfields}"
            f"_interp{self.interpolation_factor}_l{self.max_level}_{self.topology}"
        )


def parse_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("--sizes must be a comma-separated list of positive integers")
    return sizes


def parse_pair(values: list[str], name: str) -> tuple[int, int]:
    parsed = tuple(int(value) for value in values)
    if len(parsed) != 2 or any(value <= 0 for value in parsed):
        raise argparse.ArgumentTypeError(f"{name} must contain two positive integers")
    return parsed


def patch_centers(root_blocks: int, topology: str) -> list[tuple[int, int]]:
    center = (root_blocks // 2, root_blocks // 2)
    if topology == "centered":
        return [center]

    candidates = [
        center,
        (max(root_blocks // 3, 0), max(root_blocks // 3, 0)),
        (min((2 * root_blocks) // 3, root_blocks - 1), min((2 * root_blocks) // 3, root_blocks - 1)),
        (min(center[0] + 1, root_blocks - 1), max(center[1] - 1, 0)),
    ]
    unique = []
    for coord in candidates:
        if coord not in unique:
            unique.append(coord)
    return unique


def root_target_levels(case: Case) -> dict[int, int]:
    if case.max_level <= 1:
        return {isfc: 1 for isfc in range(case.root_blocks * case.root_blocks)}

    centers = patch_centers(case.root_blocks, case.topology)
    ig2morton = np.zeros(case.root_grid, dtype=np.uint32)
    morton2ig = np.zeros((case.root_blocks * case.root_blocks, 3), dtype=np.uint32)
    fill_morton_mapping3D(ig2morton, morton2ig, *case.root_grid)

    targets = {}
    for ix in range(case.root_blocks):
        for iy in range(case.root_blocks):
            distance = min(max(abs(ix - cx), abs(iy - cy)) for cx, cy in centers)
            target_level = max(1, case.max_level - distance)
            targets[int(ig2morton[ix, iy, 0])] = target_level
    return targets


def append_uniform_refinement_2d(flags: list[bool], level: int, target_level: int) -> None:
    if level >= target_level:
        flags.append(True)
        return

    flags.append(False)
    for _j in range(2):
        for _i in range(2):
            append_uniform_refinement_2d(flags, level + 1, target_level)


def forest_flags(case: Case) -> np.ndarray:
    targets = root_target_levels(case)
    flags: list[bool] = []
    for isfc in range(case.root_blocks * case.root_blocks):
        append_uniform_refinement_2d(flags, 1, targets[isfc])
    return np.array(flags, dtype=np.int32)


def build_mesh(case: Case):
    is_leaf = forest_flags(case)
    forest = AMRForest(2, case.root_blocks, case.root_blocks, 1, is_leaf)
    mesh = AMRMesh(
        2,
        np.array(case.block_nx, dtype=np.uint32),
        np.array(case.domain_nx, dtype=np.uint32),
        np.array([0.0, 0.0], dtype=np.double),
        np.array([1.0, 1.0], dtype=np.double),
        np.uint32(case.ghost_width),
        np.uint32(case.nfields),
        forest,
    )
    return forest, mesh


def affine_coefficients(ifield: int) -> tuple[float, float, float]:
    return float(ifield + 1), float(ifield + 2), 100.0 * float(ifield)


def affine_block_data(mesh, case: Case) -> np.ndarray:
    bx, by = case.block_nx
    nleafs = int(np.asarray(mesh.rnode).shape[0])
    data = np.zeros((nleafs, case.nfields, bx, by, 1), dtype=np.double)
    ix = np.arange(bx, dtype=np.double)[:, None]
    iy = np.arange(by, dtype=np.double)[None, :]
    rnode = np.asarray(mesh.rnode)

    for ileaf in range(nleafs):
        x = rnode[ileaf, 0] + (ix + 0.5) * rnode[ileaf, 4]
        y = rnode[ileaf, 1] + (iy + 0.5) * rnode[ileaf, 5]
        for ifield in range(case.nfields):
            ax, ay, offset = affine_coefficients(ifield)
            data[ileaf, ifield, :, :, 0] = ax * x + ay * y + offset
    return data


def expected_uniform(
    case: Case,
    nx: tuple[int, int, int],
    xmin: np.ndarray,
    xmax: np.ndarray,
    field_positions: np.ndarray,
) -> np.ndarray:
    dx = (xmax - xmin) / np.array(nx[:2], dtype=np.double)
    x = xmin[0] + (np.arange(nx[0], dtype=np.double)[:, None] + 0.5) * dx[0]
    y = xmin[1] + (np.arange(nx[1], dtype=np.double)[None, :] + 0.5) * dx[1]
    expected = np.zeros((field_positions.shape[0], nx[0], nx[1], 1), dtype=np.double)

    for iout, ifield in enumerate(field_positions):
        ax, ay, offset = affine_coefficients(int(ifield))
        expected[iout, :, :, 0] = ax * x + ay * y + offset
    return expected


def run_linear_interpolation(mesh, case: Case, bounds: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    uniform = np.zeros((case.nfields, *case.uniform_nx), dtype=np.double)
    mesh.uniform_grid_linear(
        uniform,
        np.array(case.uniform_nx, dtype=np.uint32),
        np.array([bounds[0], bounds[0]], dtype=np.double),
        np.array([bounds[1], bounds[1]], dtype=np.double),
        np.arange(case.nfields, dtype=np.uint32),
    )
    return uniform


def validate_case(case: Case, rtol: float, atol: float) -> dict:
    forest, mesh = build_mesh(case)
    data = affine_block_data(mesh, case)
    mesh.load_interior_data(data)
    np.testing.assert_array_equal(mesh.interior_view(), data)
    mesh.apply_ghost_cells()
    np.testing.assert_array_equal(mesh.interior_view(), data)

    padded = np.asarray(mesh.padded_view())
    if not np.all(np.isfinite(padded)):
        raise AssertionError("2D ghost-cell exchange produced non-finite values")

    validation_bounds = (0.125, 0.875)
    validation_nx = (
        max(4, int(round(case.uniform_nx[0] * (validation_bounds[1] - validation_bounds[0])))),
        max(4, int(round(case.uniform_nx[1] * (validation_bounds[1] - validation_bounds[0])))),
        1,
    )
    actual = np.zeros((case.nfields, *validation_nx), dtype=np.double)
    fields = np.arange(case.nfields, dtype=np.uint32)
    xmin = np.array([validation_bounds[0], validation_bounds[0]], dtype=np.double)
    xmax = np.array([validation_bounds[1], validation_bounds[1]], dtype=np.double)
    mesh.uniform_grid_linear(actual, np.array(validation_nx, dtype=np.uint32), xmin, xmax, fields)
    expected = expected_uniform(case, validation_nx, xmin, xmax, fields)
    max_abs_error = float(np.max(np.abs(actual - expected)))
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)

    neighbor_type = np.asarray(forest.neighbor_type)
    return {
        "nleafs": int(forest.nleafs),
        "nparents": int(forest.nparents),
        "max_level_observed": int(forest.max_level),
        "has_coarse_fine_neighbors": bool(np.any((neighbor_type == 2) | (neighbor_type == 4))),
        "max_abs_error": max_abs_error,
        "validation_nx": "x".join(str(value) for value in validation_nx),
        "validation_bounds": f"{validation_bounds[0]}:{validation_bounds[1]}",
    }


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


def summarize(durations: list[float]) -> dict:
    return {
        "runs": durations,
        "median_seconds": median(durations),
        "min_seconds": min(durations),
        "max_seconds": max(durations),
        "mean_seconds": mean(durations),
        "stdev_seconds": stdev(durations) if len(durations) > 1 else 0.0,
    }


def workload_units(case: Case, operation: str, validation: dict) -> int:
    total_leaf_cells = validation["nleafs"] * case.block_nx[0] * case.block_nx[1]
    if operation == "build_mesh":
        return validation["nleafs"] + validation["nparents"]
    if operation == "linear_interpolation":
        return case.total_uniform_cells
    if operation == "workflow":
        return total_leaf_cells + case.total_uniform_cells
    return total_leaf_cells


def make_record(case: Case, operation: str, durations: list[float], validation: dict) -> dict:
    work_units = workload_units(case, operation, validation)
    record = {
        "operation": operation,
        "root_blocks": case.root_blocks,
        "domain_nx": "x".join(str(value) for value in case.domain_nx),
        "uniform_nx": "x".join(str(value) for value in case.uniform_nx),
        "block_nx": "x".join(str(value) for value in case.block_nx),
        "ghost_width": case.ghost_width,
        "nfields": case.nfields,
        "interpolation_factor": case.interpolation_factor,
        "max_level_requested": case.max_level,
        "topology": case.topology,
        "nleafs": validation["nleafs"],
        "nparents": validation["nparents"],
        "max_level_observed": validation["max_level_observed"],
        "has_coarse_fine_neighbors": validation["has_coarse_fine_neighbors"],
        "total_leaf_cells": validation["nleafs"] * case.block_nx[0] * case.block_nx[1],
        "total_uniform_cells": case.total_uniform_cells,
        "workload_units": work_units,
        "throughput_units_s": "",
        "scaling_efficiency": "",
        "max_abs_error": validation["max_abs_error"],
        "validation_nx": validation["validation_nx"],
        "validation_bounds": validation["validation_bounds"],
    }
    record.update(summarize(durations))
    if record["median_seconds"] > 0:
        record["throughput_units_s"] = work_units / record["median_seconds"]
    return record


def add_scaling_efficiency(records: list[dict]) -> None:
    baseline = {}
    for record in sorted(records, key=lambda item: item["workload_units"]):
        if record["throughput_units_s"] == "":
            continue
        baseline.setdefault(record["operation"], record["throughput_units_s"])

    for record in records:
        base_throughput = baseline.get(record["operation"])
        if not base_throughput or record["throughput_units_s"] == "":
            continue
        record["scaling_efficiency"] = record["throughput_units_s"] / base_throughput


def run_case(case: Case, repetitions: int, warmups: int, rtol: float, atol: float) -> list[dict]:
    validation = validate_case(case, rtol, atol)
    print(
        f"{case.label}: nleafs={validation['nleafs']}, "
        f"parents={validation['nparents']}, max error={validation['max_abs_error']:.3e}"
    )

    build_durations, _ = time_call(lambda: build_mesh(case), repetitions, warmups)

    _forest, mesh = build_mesh(case)
    data = affine_block_data(mesh, case)
    ghost_durations, _ = time_call(
        lambda: (mesh.load_interior_data(data), mesh.apply_ghost_cells()),
        repetitions,
        warmups,
    )

    mesh.load_interior_data(data)
    mesh.apply_ghost_cells()
    interp_durations, _ = time_call(lambda: run_linear_interpolation(mesh, case), repetitions, warmups)

    def workflow():
        _forest_workflow, mesh_workflow = build_mesh(case)
        workflow_data = affine_block_data(mesh_workflow, case)
        mesh_workflow.load_interior_data(workflow_data)
        mesh_workflow.apply_ghost_cells()
        return run_linear_interpolation(mesh_workflow, case)

    workflow_durations, _ = time_call(workflow, repetitions, warmups)

    return [
        make_record(case, "build_mesh", build_durations, validation),
        make_record(case, "ghost_exchange", ghost_durations, validation),
        make_record(case, "linear_interpolation", interp_durations, validation),
        make_record(case, "workflow", workflow_durations, validation),
    ]


def git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True)
    except Exception:
        return None
    return result.stdout.strip() or None


def metadata(args: argparse.Namespace) -> dict:
    return {
        "profile": args.profile,
        "sizes": list(args.sizes if args.sizes is not None else PROFILE_SIZES[args.profile]),
        "block_nx": list(args.block_nx),
        "ghost_width": args.ghost_width,
        "nfields": args.nfields,
        "interpolation_factor": args.interpolation_factor,
        "max_level": args.max_level,
        "topology": args.topology,
        "include_uniform_baseline": not args.skip_uniform_baseline,
        "repetitions": args.repetitions,
        "warmups": args.warmups,
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "simesh_commit": git_commit(),
    }


def write_outputs(output_dir: Path, run_metadata: dict, records: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    add_scaling_efficiency(records)
    with (output_dir / "results.json").open("w", encoding="utf-8") as fh:
        json.dump({"metadata": run_metadata, "results": records}, fh, indent=2)

    fieldnames = [
        "operation",
        "root_blocks",
        "domain_nx",
        "uniform_nx",
        "block_nx",
        "ghost_width",
        "nfields",
        "interpolation_factor",
        "max_level_requested",
        "topology",
        "nleafs",
        "nparents",
        "max_level_observed",
        "has_coarse_fine_neighbors",
        "total_leaf_cells",
        "total_uniform_cells",
        "workload_units",
        "throughput_units_s",
        "scaling_efficiency",
        "max_abs_error",
        "validation_nx",
        "validation_bounds",
        "median_seconds",
        "min_seconds",
        "max_seconds",
        "mean_seconds",
        "stdev_seconds",
        "runs",
    ]
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = record.copy()
            row["runs"] = json.dumps(row["runs"])
            writer.writerow(row)

    write_report(output_dir, run_metadata, records)


def write_report(output_dir: Path, run_metadata: dict, records: list[dict]) -> None:
    lines = [
        "# 2D AMR Validity And Scaling Benchmark",
        "",
        "This report validates the canonical 2D AMR path with an affine field and",
        "times mesh construction, ghost-cell exchange, bilinear uniform export, and",
        "the combined workflow from a simple mesh through larger AMR meshes.",
        "",
        "## Parameters",
        "",
        f"- profile: `{run_metadata['profile']}`",
        f"- sizes: `{','.join(str(value) for value in run_metadata['sizes'])}` root blocks per side",
        f"- block_nx: `{run_metadata['block_nx'][0]}x{run_metadata['block_nx'][1]}`",
        f"- ghost_width: `{run_metadata['ghost_width']}`",
        f"- nfields: `{run_metadata['nfields']}`",
        f"- interpolation_factor: `{run_metadata['interpolation_factor']}`",
        f"- max_level: `{run_metadata['max_level']}`",
        f"- topology: `{run_metadata['topology']}`",
        f"- repetitions: `{run_metadata['repetitions']}`",
        f"- warmups: `{run_metadata['warmups']}`",
        f"- platform: `{run_metadata['platform']}`",
        f"- python: `{run_metadata['python'].split()[0]}`",
        f"- numpy: `{run_metadata['numpy']}`",
        f"- simesh commit: `{run_metadata.get('simesh_commit')}`",
        "",
        "## Results",
        "",
        "`efficiency` is normalized per-operation throughput versus the smallest",
        "case for that operation. Values near 100% mean the operation keeps the",
        "same work-per-second rate as the simple baseline mesh.",
        "",
        "| operation | root blocks | leaf blocks | max level | workload | median (s) | efficiency | max abs error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def sort_key(record: dict) -> tuple[int, int]:
        return OPERATIONS.index(record["operation"]), record["total_leaf_cells"]

    for record in sorted(records, key=sort_key):
        efficiency = record["scaling_efficiency"]
        efficiency_text = f"{efficiency * 100.0:.1f}%" if efficiency != "" else ""
        lines.append(
            f"| {record['operation']} | {record['root_blocks']} | {record['nleafs']} | "
            f"{record['max_level_observed']} | {record['workload_units']} | "
            f"{record['median_seconds']:.6g} | {efficiency_text} | {record['max_abs_error']:.3e} |"
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- Raw JSON: [results.json](results.json)",
            "- Raw CSV: [results.csv](results.csv)",
            "- Figures: [figures/](figures/)",
            "",
            "## Figures",
            "",
            "![Runtime scaling](figures/runtime_scaling.png)",
            "",
            "![Scaling efficiency](figures/efficiency.png)",
            "",
            "![Validation error](figures/validation_error.png)",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_figures(output_dir: Path, records: list[dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for figures; rerun with --no-figures to skip plotting.") from exc

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
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

    colors = {
        "build_mesh": "#0072B2",
        "ghost_exchange": "#D55E00",
        "linear_interpolation": "#009E73",
        "workflow": "#CC79A7",
    }
    markers = {
        "build_mesh": "o",
        "ghost_exchange": "s",
        "linear_interpolation": "^",
        "workflow": "D",
    }

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for operation in OPERATIONS:
        rows = sorted(
            [record for record in records if record["operation"] == operation],
            key=lambda record: record["workload_units"],
        )
        ax.plot(
            [row["workload_units"] for row in rows],
            [row["median_seconds"] for row in rows],
            color=colors[operation],
            marker=markers[operation],
            linewidth=2.0,
            label=OPERATION_LABELS[operation],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Leaf cells")
    ax.set_ylabel("Median runtime (s)")
    ax.set_title("2D AMR Runtime Scaling")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "runtime_scaling.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for operation in OPERATIONS:
        rows = sorted(
            [
                record
                for record in records
                if record["operation"] == operation and record["scaling_efficiency"] != ""
            ],
            key=lambda record: record["workload_units"],
        )
        if not rows:
            continue
        ax.plot(
            [row["workload_units"] for row in rows],
            [row["scaling_efficiency"] for row in rows],
            color=colors[operation],
            marker=markers[operation],
            linewidth=2.0,
            label=OPERATION_LABELS[operation],
        )
    ax.axhline(1.0, color="#555555", linewidth=1.0, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel("Operation workload units")
    ax.set_ylabel("Normalized throughput")
    ax.set_title("2D AMR Scaling Efficiency")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "efficiency.png", bbox_inches="tight")
    plt.close(fig)

    rows = sorted(
        [record for record in records if record["operation"] == "linear_interpolation"],
        key=lambda record: record["workload_units"],
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(
        [row["workload_units"] for row in rows],
        [max(row["max_abs_error"], 1.0e-16) for row in rows],
        color="#6A5ACD",
        marker="o",
        linewidth=2.0,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Leaf cells")
    ax.set_ylabel("Max absolute interpolation error")
    ax.set_title("Affine-Field Validation")
    fig.tight_layout()
    fig.savefig(figures_dir / "validation_error.png", bbox_inches="tight")
    plt.close(fig)


def default_output_dir(profile: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / f"2d-amr-{profile}-{time.strftime('%Y%m%d-%H%M%S')}"


def build_cases(args: argparse.Namespace) -> list[Case]:
    sizes = args.sizes if args.sizes is not None else PROFILE_SIZES[args.profile]
    cases = []
    if not args.skip_uniform_baseline:
        cases.append(
            Case(
                sizes[0],
                args.block_nx,
                args.ghost_width,
                args.nfields,
                args.interpolation_factor,
                1,
                args.topology,
            )
        )

    for size in sizes:
        cases.append(
            Case(
                size,
                args.block_nx,
                args.ghost_width,
                args.nfields,
                args.interpolation_factor,
                args.max_level,
                args.topology,
            )
        )
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILE_SIZES), default="standard")
    parser.add_argument("--sizes", type=parse_sizes, help="comma-separated root-block counts per side")
    parser.add_argument("--block-nx", nargs=2, default=DEFAULT_BLOCK_NX, metavar=("BX", "BY"))
    parser.add_argument("--ghost-width", type=int, default=2)
    parser.add_argument("--nfields", type=int, default=2)
    parser.add_argument("--interpolation-factor", type=int, default=1)
    parser.add_argument("--max-level", type=int, default=3)
    parser.add_argument("--topology", choices=TOPOLOGIES, default="centered")
    parser.add_argument("--skip-uniform-baseline", action="store_true")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--atol", type=float, default=1.0e-10)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    args.block_nx = parse_pair(args.block_nx, "--block-nx")
    if any(value % 2 != 0 for value in args.block_nx):
        parser.error("--block-nx values must be even")
    if args.ghost_width <= 0:
        parser.error("--ghost-width must be positive for 2D ghost-cell and linear interpolation validation")
    if args.nfields <= 0:
        parser.error("--nfields must be positive")
    if args.interpolation_factor <= 0:
        parser.error("--interpolation-factor must be positive")
    if args.max_level < 1:
        parser.error("--max-level must be positive")
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if _IMPORT_ERROR is not None:
        parser.error(
            "unable to import benchmark dependencies or compiled AMR modules: "
            f"{_IMPORT_ERROR}. Run `make build-amr` or `pip install -e .` first."
        )

    output_dir = args.output if args.output is not None else default_output_dir(args.profile)
    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for case in build_cases(args):
        records.extend(run_case(case, args.repetitions, args.warmups, args.rtol, args.atol))

    run_metadata = metadata(args)
    write_outputs(output_dir, run_metadata, records)
    if not args.no_figures:
        plot_figures(output_dir, records)

    print(f"Wrote benchmark report to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
