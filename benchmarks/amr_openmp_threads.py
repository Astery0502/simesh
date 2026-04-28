"""Benchmark OpenMP thread scaling for AMR ghost-cell interpolation."""

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

try:
    import numpy as np

    from benchmarks.amr_ghost_interpolation import (
        DEFAULT_DOMAIN,
        DEFAULT_OUTPUT_ROOT,
        PROFILE_SIZES,
        TOPOLOGIES,
        Case,
        bipolar_block_data,
        build_mesh_pair,
        git_commit,
        mesh_nleafs,
        parse_sizes,
        parse_triplet,
    )
    from simesh.utils import openmp_build_info
except ModuleNotFoundError as exc:
    np = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


DEFAULT_THREADS = (1, 2, 4, 8)
OPERATIONS = ("ghost_exchange", "linear_interp", "workflow")


@dataclass(frozen=True)
class WorkerResult:
    records: list[dict]
    checks: list[dict]


def parse_csv_ints(value: str, name: str) -> tuple[int, ...]:
    parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError(f"{name} must be a comma-separated list of positive integers")
    return parsed


def parse_topologies(value: str) -> tuple[str, ...]:
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    invalid = [item for item in parsed if item not in TOPOLOGIES]
    if not parsed or invalid:
        choices = ", ".join(TOPOLOGIES)
        raise argparse.ArgumentTypeError(f"--topologies entries must be one of: {choices}")
    return parsed


def build_cases(args: argparse.Namespace) -> list[Case]:
    sizes = args.sizes if args.sizes is not None else PROFILE_SIZES[args.profile]
    cases = []
    for topology in args.topologies:
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
                    topology,
                )
            )
    return cases


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


def make_record(case: Case, operation: str, threads: int, durations: list[float], nleafs: int) -> dict:
    record = {
        "operation": operation,
        "threads": threads,
        "base_resolution": case.base_resolution,
        "domain_nx": "x".join(str(value) for value in case.domain_nx),
        "uniform_nx": "x".join(str(value) for value in case.uniform_nx),
        "block_nx": "x".join(str(value) for value in case.block_nx),
        "ghost_width": case.ghost_width,
        "max_level": case.max_level,
        "topology": case.topology,
        "nleafs": nleafs,
        "total_uniform_cells": case.total_uniform_cells,
        "speedup_vs_1_thread": "",
        "parallel_efficiency": "",
        "ratio_mark": "",
    }
    record.update(summarize(durations))
    return record


def run_case(case: Case, threads: int, repetitions: int, warmups: int) -> tuple[list[dict], dict]:
    mesh, _legacy_mesh = build_mesh_pair(case)
    data = bipolar_block_data(mesh, case.block_nx)
    fields = np.array([0, 1, 2], dtype=np.uint32)
    nx = np.array(case.uniform_nx, dtype=np.uint32)
    xmin = np.array(DEFAULT_DOMAIN[0], dtype=np.double)
    xmax = np.array(DEFAULT_DOMAIN[1], dtype=np.double)
    nleafs = mesh_nleafs(mesh)

    def do_ghost():
        mesh.load_interior_data(data)
        mesh.apply_ghost_cells()

    ghost_durations, _ = time_call(do_ghost, repetitions, warmups)

    do_ghost()
    uniform = np.zeros((3, *case.uniform_nx), dtype=np.double)

    def do_interp():
        uniform.fill(0.0)
        mesh.uniform_grid_linear(uniform, nx, xmin, xmax, fields)

    interp_durations, _ = time_call(do_interp, repetitions, warmups)

    def do_workflow():
        mesh.load_interior_data(data)
        mesh.apply_ghost_cells()
        uniform.fill(0.0)
        mesh.uniform_grid_linear(uniform, nx, xmin, xmax, fields)

    workflow_durations, _ = time_call(do_workflow, repetitions, warmups)

    check = {
        "base_resolution": case.base_resolution,
        "topology": case.topology,
        "nleafs": nleafs,
        "uniform_nx": "x".join(str(value) for value in case.uniform_nx),
        "threads": threads,
        "uniform_checksum": float(np.sum(uniform)),
    }
    records = [
        make_record(case, "ghost_exchange", threads, ghost_durations, nleafs),
        make_record(case, "linear_interp", threads, interp_durations, nleafs),
        make_record(case, "workflow", threads, workflow_durations, nleafs),
    ]
    return records, check


def run_worker(args: argparse.Namespace) -> WorkerResult:
    records = []
    checks = []
    for case in build_cases(args):
        case_records, check = run_case(case, args.worker_threads, args.repetitions, args.warmups)
        records.extend(case_records)
        checks.append(check)
    return WorkerResult(records, checks)


def worker_command(args: argparse.Namespace, threads: int) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.amr_openmp_threads",
        "--worker",
        "--worker-threads",
        str(threads),
        "--profile",
        args.profile,
        "--block-nx",
        *(str(value) for value in args.block_nx),
        "--ghost-width",
        str(args.ghost_width),
        "--interpolation-factor",
        str(args.interpolation_factor),
        "--max-level",
        str(args.max_level),
        "--topologies",
        ",".join(args.topologies),
        "--repetitions",
        str(args.repetitions),
        "--warmups",
        str(args.warmups),
    ]
    if args.sizes is not None:
        cmd.extend(["--sizes", ",".join(str(value) for value in args.sizes)])
    return cmd


def run_thread_worker(args: argparse.Namespace, threads: int) -> WorkerResult:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env.setdefault("OMP_DYNAMIC", "FALSE")
    env.setdefault("PYTHONPATH", "src")
    if "src" not in env["PYTHONPATH"].split(os.pathsep):
        env["PYTHONPATH"] = os.pathsep.join(["src", env["PYTHONPATH"]])

    result = subprocess.run(
        worker_command(args, threads),
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    return WorkerResult(payload["records"], payload["checks"])


def add_ratios(records: list[dict]) -> None:
    baseline = {}
    for record in records:
        if record["threads"] == 1:
            key = (record["topology"], record["base_resolution"], record["operation"])
            baseline[key] = record["median_seconds"]

    for record in records:
        key = (record["topology"], record["base_resolution"], record["operation"])
        base_seconds = baseline.get(key)
        if not base_seconds or record["median_seconds"] <= 0:
            continue
        speedup = base_seconds / record["median_seconds"]
        efficiency = speedup / record["threads"]
        record["speedup_vs_1_thread"] = speedup
        record["parallel_efficiency"] = efficiency
        record["ratio_mark"] = f"{speedup:.2f}x / {efficiency * 100.0:.0f}%"


def metadata(args: argparse.Namespace) -> dict:
    return {
        "profile": args.profile,
        "sizes": list(args.sizes if args.sizes is not None else PROFILE_SIZES[args.profile]),
        "topologies": list(args.topologies),
        "threads": list(args.threads),
        "repetitions": args.repetitions,
        "warmups": args.warmups,
        "interpolation_factor": args.interpolation_factor,
        "max_level": args.max_level,
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "simesh_commit": git_commit(),
        "openmp_build": openmp_build_info(),
        "omp_dynamic": os.environ.get("OMP_DYNAMIC"),
    }


def default_output_dir(profile: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / f"openmp-threads-{profile}-{time.strftime('%Y%m%d-%H%M%S')}"


def write_outputs(output_dir: Path, run_metadata: dict, records: list[dict], checks: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    add_ratios(records)
    with (output_dir / "results.json").open("w", encoding="utf-8") as fh:
        json.dump({"metadata": run_metadata, "checks": checks, "results": records}, fh, indent=2)

    fieldnames = [
        "operation",
        "threads",
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
        "speedup_vs_1_thread",
        "parallel_efficiency",
        "ratio_mark",
        "runs",
    ]
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = record.copy()
            row["repetitions"] = run_metadata["repetitions"]
            row["warmups"] = run_metadata["warmups"]
            row["runs"] = json.dumps(row["runs"])
            writer.writerow(row)
    write_report(output_dir, run_metadata, records)


def write_report(output_dir: Path, run_metadata: dict, records: list[dict]) -> None:
    lines = [
        "# AMR OpenMP Thread Benchmark",
        "",
        "This report times the canonical AMR ghost-cell path over several mesh sizes",
        "and OpenMP thread counts. Ratios are measured against the 1-thread median",
        "for the same topology, mesh size, and operation.",
        "",
        "## Parameters",
        "",
        f"- profile: `{run_metadata['profile']}`",
        f"- sizes: `{run_metadata['sizes']}`",
        f"- topologies: `{run_metadata['topologies']}`",
        f"- threads: `{run_metadata['threads']}`",
        f"- repetitions: `{run_metadata['repetitions']}`",
        f"- warmups: `{run_metadata['warmups']}`",
        f"- interpolation factor: `{run_metadata['interpolation_factor']}`",
        f"- max AMR level: `{run_metadata['max_level']}`",
        f"- platform: `{run_metadata['platform']}`",
        f"- python: `{run_metadata['python'].split()[0]}`",
        f"- numpy: `{run_metadata['numpy']}`",
        f"- simesh commit: `{run_metadata.get('simesh_commit')}`",
        "",
        "## Results",
        "",
        "| topology | base | leaf blocks | operation | threads | median (s) | speedup | efficiency | ratio mark |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    sorted_records = sorted(
        records,
        key=lambda item: (
            item["topology"],
            item["base_resolution"],
            OPERATIONS.index(item["operation"]),
            item["threads"],
        ),
    )
    for record in sorted_records:
        speedup = record["speedup_vs_1_thread"]
        efficiency = record["parallel_efficiency"]
        speedup_text = f"{speedup:.3g}x" if speedup != "" else ""
        efficiency_text = f"{efficiency * 100.0:.1f}%" if efficiency != "" else ""
        lines.append(
            f"| {record['topology']} | {record['base_resolution']} | {record['nleafs']} | "
            f"{record['operation']} | {record['threads']} | {record['median_seconds']:.6g} | "
            f"{speedup_text} | {efficiency_text} | {record['ratio_mark']} |"
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
            "![Thread speedup](figures/thread_speedup.png)",
            "",
            "![Parallel efficiency](figures/parallel_efficiency.png)",
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
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    for metric, ylabel, filename in (
        ("speedup_vs_1_thread", "Speedup vs 1 thread", "thread_speedup.png"),
        ("parallel_efficiency", "Parallel efficiency", "parallel_efficiency.png"),
    ):
        fig, axes = plt.subplots(1, len(OPERATIONS), figsize=(12.0, 3.8), sharex=False)
        for ax, operation in zip(axes, OPERATIONS):
            subset = [record for record in records if record["operation"] == operation and record[metric] != ""]
            for topology in sorted({record["topology"] for record in subset}):
                for base in sorted({record["base_resolution"] for record in subset if record["topology"] == topology}):
                    rows = sorted(
                        [
                            record
                            for record in subset
                            if record["topology"] == topology and record["base_resolution"] == base
                        ],
                        key=lambda item: item["threads"],
                    )
                    ax.plot(
                        [row["threads"] for row in rows],
                        [row[metric] for row in rows],
                        marker="o",
                        linewidth=1.6,
                        label=f"{topology} {base}",
                    )
            ax.axhline(1.0, color="#555555", linewidth=1.0, linestyle=":")
            ax.set_title(operation)
            ax.set_xlabel("OpenMP threads")
            ax.set_ylabel(ylabel)
        axes[-1].legend(frameon=False, fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(figures_dir / filename, bbox_inches="tight", dpi=220)
        plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILE_SIZES), default="standard")
    parser.add_argument("--sizes", type=parse_sizes, help="comma-separated cubic base resolutions")
    parser.add_argument("--block-nx", nargs=3, default=(8, 8, 8), metavar=("BX", "BY", "BZ"))
    parser.add_argument("--ghost-width", type=int, default=2)
    parser.add_argument("--interpolation-factor", type=int, default=1)
    parser.add_argument("--max-level", type=int, default=4)
    parser.add_argument("--topologies", type=parse_topologies, default=("complex",))
    parser.add_argument("--threads", type=lambda value: parse_csv_ints(value, "--threads"), default=DEFAULT_THREADS)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-threads", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    args.block_nx = parse_triplet(args.block_nx, "--block-nx")
    if any(value % 2 != 0 for value in args.block_nx):
        parser.error("--block-nx values must be even for the AMR mesh fixture")
    if args.ghost_width <= 0:
        parser.error("--ghost-width must be positive")
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
            f"{_IMPORT_ERROR}. Run `make build-amr-openmp` first."
        )
    if not openmp_build_info()["enabled"]:
        parser.error("compiled AMR modules do not have OpenMP enabled. Run `make build-amr-openmp` first.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        worker_result = run_worker(args)
        json.dump({"records": worker_result.records, "checks": worker_result.checks}, sys.stdout)
        return 0

    output_dir = args.output if args.output is not None else default_output_dir(args.profile)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    checks = []
    for threads in args.threads:
        print(f"Benchmarking OpenMP threads={threads}...")
        worker_result = run_thread_worker(args, threads)
        records.extend(worker_result.records)
        checks.extend(worker_result.checks)

    run_metadata = metadata(args)
    write_outputs(output_dir, run_metadata, records, checks)
    if not args.no_figures:
        plot_figures(output_dir, records)

    print(f"Wrote OpenMP thread benchmark report to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
