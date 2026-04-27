"""Benchmark AMR ghost-cell exchange followed by linear interpolation.

The fixture uses the inexpensive bipolar magnetic-field configuration from
``simesh.utils.configurations`` and a mixed coarse/fine forest. It compares the
canonical Cython-backed AMR mesh with the legacy Python-first mesh, verifies
that their ghosted data and interpolated uniform grids agree, and writes timing
figures for several resolutions.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev

_IMPORT_ERROR = None
try:
    import numpy as np

    from simesh.legacy.geometry.amr.amr_forest import AMRForest as LegacyAMRForest
    from simesh.legacy.meshes.amr_mesh import AMRMesh as LegacyAMRMesh
    from simesh.utils.configurations import bipolar_Bvec
    from simesh.utils.lib.amr.forest import AMRForest
    from simesh.utils.lib.amr.mesh import AMRMesh
    from simesh.utils.lib.amr.morton import fill_morton_mapping3D
except ModuleNotFoundError as exc:
    _IMPORT_ERROR = exc


DEFAULT_BLOCK_NX = (8, 8, 8)
DEFAULT_DOMAIN = ((-1.0, -1.0, 0.0), (1.0, 1.0, 2.0))
DEFAULT_OUTPUT_ROOT = Path("benchmark-results")
PROFILE_SIZES = {
    "smoke": (16,),
    "standard": (16, 24, 32, 48),
}
TOPOLOGIES = ("centered", "complex")
IMPLEMENTATIONS = {
    "canonical": {"label": "Canonical Cython-backed", "color": "#0072B2", "marker": "o"},
    "legacy": {"label": "Legacy Python-first", "color": "#D55E00", "marker": "s"},
}


@dataclass(frozen=True)
class Case:
    base_resolution: int
    block_nx: tuple[int, int, int]
    ghost_width: int
    interpolation_factor: int
    max_level: int
    topology: str

    @property
    def root_grid(self) -> tuple[int, int, int]:
        return tuple(self.base_resolution // block for block in self.block_nx)

    @property
    def domain_nx(self) -> tuple[int, int, int]:
        return (self.base_resolution, self.base_resolution, self.base_resolution)

    @property
    def uniform_nx(self) -> tuple[int, int, int]:
        return tuple(self.interpolation_factor * value for value in self.domain_nx)

    @property
    def total_uniform_cells(self) -> int:
        return int(np.prod(self.uniform_nx))

    @property
    def label(self) -> str:
        return (
            f"base{self.base_resolution}"
            f"_block{self.block_nx[0]}x{self.block_nx[1]}x{self.block_nx[2]}"
            f"_g{self.ghost_width}_interp{self.interpolation_factor}_l{self.max_level}_{self.topology}"
        )


def parse_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("--sizes must be a comma-separated list of positive integers")
    return sizes


def parse_triplet(values: list[str], name: str) -> tuple[int, int, int]:
    parsed = tuple(int(value) for value in values)
    if len(parsed) != 3 or any(value <= 0 for value in parsed):
        raise argparse.ArgumentTypeError(f"{name} must contain three positive integers")
    return parsed


def level1_sfc_index(root_grid: tuple[int, int, int], coord: tuple[int, int, int]) -> int:
    ig2morton = np.zeros(root_grid, dtype=np.uint32)
    morton2ig = np.zeros((int(np.prod(root_grid)), 3), dtype=np.uint32)
    fill_morton_mapping3D(ig2morton, morton2ig, *root_grid)
    return int(ig2morton[coord])


def patch_centers(root_grid: tuple[int, int, int], topology: str) -> list[tuple[int, int, int]]:
    center = tuple(value // 2 for value in root_grid)
    if topology == "centered":
        return [center]

    candidates = [
        center,
        tuple(max(value // 3, 0) for value in root_grid),
        tuple(min((2 * value) // 3, value - 1) for value in root_grid),
        (min(root_grid[0] - 1, center[0] + 1), max(center[1] - 1, 0), center[2]),
        (max(center[0] - 1, 0), center[1], min(root_grid[2] - 1, center[2] + 1)),
    ]
    unique = []
    for coord in candidates:
        if coord not in unique:
            unique.append(coord)
    return unique


def root_target_levels(root_grid: tuple[int, int, int], max_level: int, topology: str) -> dict[int, int]:
    centers = patch_centers(root_grid, topology)
    targets = {}
    for coord in np.ndindex(root_grid):
        distance = min(max(abs(coord[idim] - center[idim]) for idim in range(3)) for center in centers)
        target_level = max(1, max_level - distance)
        targets[level1_sfc_index(root_grid, coord)] = target_level
    return targets


def append_uniform_refinement(flags: list[bool], level: int, target_level: int) -> None:
    if level >= target_level:
        flags.append(True)
        return

    flags.append(False)
    for _k in range(2):
        for _j in range(2):
            for _i in range(2):
                append_uniform_refinement(flags, level + 1, target_level)


def forest_flags(root_grid: tuple[int, int, int], max_level: int, topology: str) -> np.ndarray:
    target_levels = root_target_levels(root_grid, max_level, topology)
    flags = []
    for isfc in range(int(np.prod(root_grid))):
        append_uniform_refinement(flags, 1, target_levels[isfc])
    return np.array(flags, dtype=np.int32)


def build_mesh_pair(case: Case):
    is_leaf = forest_flags(case.root_grid, case.max_level, case.topology)
    ng1, ng2, ng3 = case.root_grid
    block_nx = np.array(case.block_nx, dtype=np.uint32)
    domain_nx = np.array(case.domain_nx, dtype=np.uint32)
    xmin = np.array(DEFAULT_DOMAIN[0], dtype=np.double)
    xmax = np.array(DEFAULT_DOMAIN[1], dtype=np.double)

    forest = AMRForest(3, ng1, ng2, ng3, is_leaf)
    mesh = AMRMesh(3, block_nx, domain_nx, xmin, xmax, np.uint32(case.ghost_width), np.uint32(3), forest)

    legacy_forest = LegacyAMRForest(ng1, ng2, ng3, int(forest.nleafs))
    legacy_forest.read_forest(is_leaf.astype(bool))
    legacy_forest.build_connectivity()
    legacy_mesh = LegacyAMRMesh(
        (float(xmin[0]), float(xmax[0])),
        (float(xmin[1]), float(xmax[1])),
        (float(xmin[2]), float(xmax[2])),
        ["b1", "b2", "b3"],
        block_nx.astype(int),
        domain_nx.astype(int),
        legacy_forest,
        nghostcells=case.ghost_width,
    )
    return mesh, legacy_mesh


def mesh_nleafs(mesh) -> int:
    if hasattr(mesh, "nleafs"):
        return int(mesh.nleafs)
    return int(np.asarray(mesh.rnode).shape[0])


def bipolar_block_data(mesh, block_nx: tuple[int, int, int]) -> np.ndarray:
    nleafs = mesh_nleafs(mesh)
    data = np.empty((nleafs, 3, *block_nx), dtype=np.double)
    ix = np.arange(block_nx[0], dtype=np.double)[:, None, None]
    iy = np.arange(block_nx[1], dtype=np.double)[None, :, None]
    iz = np.arange(block_nx[2], dtype=np.double)[None, None, :]
    zeros = np.zeros(block_nx, dtype=np.double)
    rnode = np.asarray(mesh.rnode)

    for ileaf in range(nleafs):
        x = zeros + rnode[ileaf, 0] + (ix + 0.5) * rnode[ileaf, 6]
        y = zeros + rnode[ileaf, 1] + (iy + 0.5) * rnode[ileaf, 7]
        z = zeros + rnode[ileaf, 2] + (iz + 0.5) * rnode[ileaf, 8]
        data[ileaf] = bipolar_Bvec(np.stack([x, y, z]), q_para=1.0, L_para=0.35, d_para=0.25)
    return data


def load_legacy_interior(mesh: LegacyAMRMesh, data: np.ndarray) -> None:
    mesh.data[...] = 0.0
    mesh.data[
        :,
        mesh.ixMmin[0] : mesh.ixMmax[0] + 1,
        mesh.ixMmin[1] : mesh.ixMmax[1] + 1,
        mesh.ixMmin[2] : mesh.ixMmax[2] + 1,
        :,
    ] = np.transpose(data, (0, 2, 3, 4, 1))


def canonical_workflow(mesh, case: Case, data: np.ndarray) -> np.ndarray:
    uniform = np.zeros((3, *case.uniform_nx), dtype=np.double)
    mesh.load_interior_data(data)
    mesh.apply_ghost_cells()
    mesh.uniform_grid_linear(
        uniform,
        np.array(case.uniform_nx, dtype=np.uint32),
        np.array(DEFAULT_DOMAIN[0], dtype=np.double),
        np.array(DEFAULT_DOMAIN[1], dtype=np.double),
        np.array([0, 1, 2], dtype=np.uint32),
    )
    return uniform


def legacy_workflow(mesh: LegacyAMRMesh, case: Case, data: np.ndarray) -> np.ndarray:
    load_legacy_interior(mesh, data)
    mesh.getbc()
    uniform = mesh.export_uniform(
        mesh.data,
        np.array(DEFAULT_DOMAIN[0], dtype=np.double),
        np.array(DEFAULT_DOMAIN[1], dtype=np.double),
        *case.uniform_nx,
    )
    return np.transpose(uniform, (3, 0, 1, 2))


def time_call(func, repetitions: int, warmups: int) -> tuple[list[float], object]:
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


def git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True)
    except Exception:
        return None
    return result.stdout.strip() or None


def make_record(case: Case, implementation: str, durations: list[float], mesh, max_abs_error: float) -> dict:
    record = {
        "operation": "ghost_exchange_then_linear_interpolation",
        "implementation": implementation,
        "base_resolution": case.base_resolution,
        "domain_nx": "x".join(str(value) for value in case.domain_nx),
        "uniform_nx": "x".join(str(value) for value in case.uniform_nx),
        "block_nx": "x".join(str(value) for value in case.block_nx),
        "ghost_width": case.ghost_width,
        "max_level": case.max_level,
        "topology": case.topology,
        "nleafs": mesh_nleafs(mesh),
        "total_uniform_cells": case.total_uniform_cells,
        "max_abs_error_vs_legacy": max_abs_error if implementation == "canonical" else 0.0,
        "speedup_ratio": "",
    }
    record.update(summarize(durations))
    return record


def add_speedups(records: list[dict]) -> None:
    grouped: dict[int, dict[str, dict]] = {}
    for record in records:
        grouped.setdefault(record["base_resolution"], {})[record["implementation"]] = record
    for implementations in grouped.values():
        canonical = implementations.get("canonical")
        legacy = implementations.get("legacy")
        if canonical is None or legacy is None:
            continue
        canonical["speedup_ratio"] = legacy["median_seconds"] / canonical["median_seconds"]
        legacy["speedup_ratio"] = 1.0


def run_case(case: Case, repetitions: int, warmups: int, rtol: float, atol: float) -> list[dict]:
    canonical_mesh, legacy_mesh = build_mesh_pair(case)
    data = bipolar_block_data(canonical_mesh, case.block_nx)

    canonical_grid = canonical_workflow(canonical_mesh, case, data)
    legacy_grid = legacy_workflow(legacy_mesh, case, data)
    ghost_error = float(np.max(np.abs(canonical_mesh.padded_view() - legacy_mesh.data)))
    grid_error = float(np.max(np.abs(canonical_grid - legacy_grid)))
    np.testing.assert_allclose(canonical_mesh.padded_view(), legacy_mesh.data, rtol=rtol, atol=atol)
    np.testing.assert_allclose(canonical_grid, legacy_grid, rtol=rtol, atol=atol)

    print(
        f"{case.label}: nleafs={mesh_nleafs(canonical_mesh)}, "
        f"uniform={case.uniform_nx}, max ghost diff={ghost_error:.3e}, "
        f"max grid diff={grid_error:.3e}"
    )

    canonical_durations, _ = time_call(
        lambda: canonical_workflow(canonical_mesh, case, data),
        repetitions,
        warmups,
    )
    legacy_durations, _ = time_call(
        lambda: legacy_workflow(legacy_mesh, case, data),
        repetitions,
        warmups,
    )

    return [
        make_record(case, "canonical", canonical_durations, canonical_mesh, grid_error),
        make_record(case, "legacy", legacy_durations, legacy_mesh, grid_error),
    ]


def write_outputs(output_dir: Path, metadata: dict, records: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    add_speedups(records)
    with (output_dir / "results.json").open("w", encoding="utf-8") as fh:
        json.dump({"metadata": metadata, "results": records}, fh, indent=2)

    fieldnames = [
        "operation",
        "implementation",
        "base_resolution",
        "domain_nx",
        "uniform_nx",
        "block_nx",
        "ghost_width",
        "max_level",
        "topology",
        "nleafs",
        "total_uniform_cells",
        "repetitions",
        "warmups",
        "median_seconds",
        "min_seconds",
        "max_seconds",
        "mean_seconds",
        "stdev_seconds",
        "speedup_ratio",
        "max_abs_error_vs_legacy",
        "runs",
    ]
    for record in records:
        record["repetitions"] = metadata["repetitions"]
        record["warmups"] = metadata["warmups"]
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = record.copy()
            row["runs"] = json.dumps(row["runs"])
            writer.writerow(row)
    write_report(output_dir, metadata, records)


def write_report(output_dir: Path, metadata: dict, records: list[dict]) -> None:
    lines = [
        "# AMR Ghost Exchange + Interpolation Benchmark",
        "",
        "This report times a mixed coarse/fine AMR ghost-cell exchange followed by",
        "linear interpolation to a uniform grid. The synthetic field is generated",
        "with `simesh.utils.configurations.bipolar_Bvec`, avoiding the more costly",
        "RBSL configuration.",
        "",
        "## Parameters",
        "",
        f"- profile: `{metadata['profile']}`",
        f"- repetitions: `{metadata['repetitions']}`",
        f"- warmups: `{metadata['warmups']}`",
        f"- interpolation factor: `{metadata['interpolation_factor']}`",
        f"- max AMR level: `{metadata['max_level']}`",
        f"- topology: `{metadata['topology']}`",
        f"- platform: `{metadata['platform']}`",
        f"- python: `{metadata['python'].split()[0]}`",
        f"- numpy: `{metadata['numpy']}`",
        f"- simesh commit: `{metadata.get('simesh_commit')}`",
        "",
        "## Results",
        "",
        "| base resolution | leaf blocks | uniform grid | canonical median (s) | legacy median (s) | speedup | max abs grid diff |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    grouped: dict[int, dict[str, dict]] = {}
    for record in records:
        grouped.setdefault(record["base_resolution"], {})[record["implementation"]] = record
    for base_resolution in sorted(grouped):
        canonical = grouped[base_resolution]["canonical"]
        legacy = grouped[base_resolution]["legacy"]
        lines.append(
            f"| {base_resolution} | {canonical['nleafs']} | {canonical['uniform_nx']} | "
            f"{canonical['median_seconds']:.6g} | {legacy['median_seconds']:.6g} | "
            f"{canonical['speedup_ratio']:.3g}x | {canonical['max_abs_error_vs_legacy']:.3e} |"
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "![Runtime scaling](figures/runtime_scaling.png)",
            "",
            "![Legacy/canonical speedup](figures/speedup.png)",
            "",
            "![Output agreement](figures/output_error.png)",
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

    def subset(implementation: str) -> list[dict]:
        return sorted(
            [record for record in records if record["implementation"] == implementation],
            key=lambda record: record["total_uniform_cells"],
        )

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for implementation, style in IMPLEMENTATIONS.items():
        rows = subset(implementation)
        ax.plot(
            [row["total_uniform_cells"] for row in rows],
            [row["median_seconds"] for row in rows],
            color=style["color"],
            marker=style["marker"],
            linewidth=2.0,
            label=style["label"],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Uniform-grid cells")
    ax.set_ylabel("Median runtime (s)")
    ax.set_title("Ghost Exchange Then Linear Interpolation")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "runtime_scaling.png", bbox_inches="tight")
    plt.close(fig)

    rows = subset("canonical")
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(
        [row["total_uniform_cells"] for row in rows],
        [row["speedup_ratio"] for row in rows],
        color="#009E73",
        marker="D",
        linewidth=2.0,
    )
    ax.axhline(1.0, color="#555555", linewidth=1.0, linestyle=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Uniform-grid cells")
    ax.set_ylabel("Legacy / canonical median runtime")
    ax.set_title("Canonical Speedup")
    fig.tight_layout()
    fig.savefig(figures_dir / "speedup.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(
        [row["total_uniform_cells"] for row in rows],
        [max(row["max_abs_error_vs_legacy"], 1.0e-16) for row in rows],
        color="#CC79A7",
        marker="^",
        linewidth=2.0,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Uniform-grid cells")
    ax.set_ylabel("Max absolute grid difference")
    ax.set_title("Canonical vs Legacy Output Agreement")
    fig.tight_layout()
    fig.savefig(figures_dir / "output_error.png", bbox_inches="tight")
    plt.close(fig)


def metadata(args: argparse.Namespace) -> dict:
    return {
        "profile": args.profile,
        "repetitions": args.repetitions,
        "warmups": args.warmups,
        "interpolation_factor": args.interpolation_factor,
        "max_level": args.max_level,
        "topology": args.topology,
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "simesh_commit": git_commit(),
    }


def default_output_dir(profile: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / f"ghost-interpolation-{profile}-{time.strftime('%Y%m%d-%H%M%S')}"


def build_cases(args: argparse.Namespace) -> list[Case]:
    sizes = args.sizes if args.sizes is not None else PROFILE_SIZES[args.profile]
    cases = []
    for size in sizes:
        if any(size % block != 0 for block in args.block_nx):
            raise ValueError(f"base resolution {size} must be divisible by block_nx={args.block_nx}")
        cases.append(
            Case(
                size,
                args.block_nx,
                args.ghost_width,
                args.interpolation_factor,
                args.max_level,
                args.topology,
            )
        )
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILE_SIZES), default="standard")
    parser.add_argument("--sizes", type=parse_sizes, help="comma-separated cubic base resolutions")
    parser.add_argument("--block-nx", nargs=3, default=DEFAULT_BLOCK_NX, metavar=("BX", "BY", "BZ"))
    parser.add_argument("--ghost-width", type=int, default=2)
    parser.add_argument("--interpolation-factor", type=int, default=1)
    parser.add_argument("--max-level", type=int, default=4)
    parser.add_argument("--topology", choices=TOPOLOGIES, default="complex")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--atol", type=float, default=1.0e-10)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    args.block_nx = parse_triplet(args.block_nx, "--block-nx")
    if any(value % 2 != 0 for value in args.block_nx):
        parser.error("--block-nx values must be even for the legacy AMR mesh")
    if args.ghost_width <= 0:
        parser.error("--ghost-width must be positive for linear interpolation")
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
