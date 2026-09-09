# Performance Benchmarks

## Purpose

The AMRVAC scaling benchmark produces reproducible timing data and figures for
the canonical Cython-backed path versus the legacy Python-first path. It is a
reporting tool, not part of the default correctness test suite.

## Quick Smoke Run

```bash
make benchmark-smoke
```

This builds the AMR extensions, generates a `50^3` synthetic level-1 AMR input,
runs one timing repetition, writes JSON/CSV results, and creates figures under:

```bash
benchmark-results/smoke/
```

Generated input `.dat` files are cached under `.benchmark-data/`.

## Full Report

```bash
PYTHONPATH=src python3 -m benchmarks.amrvac_scaling \
  --profile standard \
  --repetitions 3 \
  --warmups 1
```

The standard profile runs `50^3`, `100^3`, and `200^3` meshes. The large profile
adds `300^3`, `400^3`, and `500^3` and prints estimated disk and memory needs
before running:

```bash
PYTHONPATH=src python3 -m benchmarks.amrvac_scaling --profile large
```

## OpenMP Thread Scaling

OpenMP is not part of the default build. After building the AMR extensions with
OpenMP, the thread benchmark compares the canonical ghost-cell/interpolation
workflow over multiple mesh sizes and thread counts:

```bash
make build-amr-openmp
PYTHONPATH=src OMP_DYNAMIC=FALSE python3 -m benchmarks.amr_openmp_threads \
  --profile standard \
  --threads 1,2,4,8 \
  --repetitions 3 \
  --warmups 1
```

If the build machine has no OpenMP implementation, keep using `make build-amr`
and skip this benchmark. You can confirm the compiled status with:

```bash
PYTHONPATH=src python3 -c "from simesh.utils import openmp_build_info; print(openmp_build_info())"
```

For a quick local check:

```bash
make benchmark-openmp-threads
```

The report marks each thread result with speedup versus the matching 1-thread
median and parallel efficiency, for example `1.40x / 70%`.

## Parameters

The default block size is `(10, 10, 10)`. This differs from the earlier planning
example of `(25, 25, 25)` because the legacy ghost-cell mesh requires even block
dimensions. Use `--block-nx BX BY BZ` to override the default for experiments
that keep this constraint in mind.

Useful options:

- `--sizes 50,100,200` overrides the profile mesh sizes.
- `--nw 1` controls the number of fields.
- `--ghost-width 2` controls ghost-cell padding.
- `--force-generate` rebuilds cached input files.
- `--skip-validation` skips small canonical/legacy sanity checks.
- `--no-figures` writes JSON/CSV/Markdown without requiring matplotlib.

## Outputs

Each run writes:

- `report.md`: integrated benchmark report with parameters, summary table, raw
  artifact links, and embedded curve figures.
- `results.json`: machine metadata and raw per-repetition timings.
- `results.csv`: tabular timing data for plotting or external analysis.
- `figures/runtime_scaling.png`: runtime versus total cells.
- `figures/speedup.png`: legacy/canonical runtime ratio by operation.
- `figures/throughput.png`: read/write throughput in GB/s.
- `figures/ghost_exchange.png`: ghost-cell exchange time versus leaf blocks.
- `figures/read.png`, `figures/write.png`, `figures/workflow.png`: focused
  per-operation runtime and legacy/canonical ratio figures.

Use `report.md` as the entry point when reviewing or sharing a benchmark run;
the JSON, CSV, and PNG files are supporting artifacts for that report.
