# Heavy AMRVAC Current Validation Design

## Context

AMRVAC dataset tests currently cover `.dat` reading, ghost-cell-backed mesh
storage, uniform-grid extraction, and writing with small generated fixtures.
They do not exercise the heavier real-data path that reads an existing
adaptive `.dat` snapshot, performs ghost-cell exchange, samples magnetic fields
onto a uniform grid, computes current density, and checks whether the result is
numerically reasonable on a representative slice.

The validation fixture is `data/weno509_sub_0000.dat`. Its relevant metadata is
Cartesian 3D geometry, fields `rho`, `m1`, `m2`, `m3`, `b1`, `b2`, `b3`,
`domain_nx=[16, 16, 8]`, `block_nx=[8, 8, 8]`, and `levmax=6`. The file is
about 1 GB, so this validation must stay disabled by default.

## Decision

Add an optional heavy validation to the existing script-based AMRVAC dataset
test file, `tests/amrvac/test_amrvac_dataset.py`. The validation runs only when
`SIMESH_RUN_HEAVY_TESTS=1` is set. Without that environment variable, the test
returns cleanly without reading the large fixture. If
`data/weno509_sub_0000.dat` is absent, the validation also returns cleanly with
a clear message.

Generated diagnostics are written only during enabled heavy runs. The artifact
directory is `report/amrvac-current/`, and `report/` should be ignored by git.

## Data Flow

The enabled validation pipeline is:

1. Open `data/weno509_sub_0000.dat` with `ghost_width=2`, because the refined
   fixture has coarse/fine interfaces and limited prolongation needs two ghost
   cells.
2. Load original field indices `[4, 5, 6]`, corresponding to `b1`, `b2`, and
   `b3`.
3. Call `exchange_ghost_cells()` to refresh ghost-cell storage explicitly.
4. Produce a uniform magnetic field with linear interpolation through
   `uniform_grid(..., interpolation="linear")`.
5. Compute current density as `J = curl(B)` using central differences on the
   uniform Cartesian magnetic field.
6. Select the center `z` slice of `|J|`.
7. Validate the slice with a robust smoothness metric and write diagnostics.

The uniform-grid resolution should use the effective full AMR resolution:

```text
domain_nx * 2 ** (levmax - 1)
```

For the selected fixture this is `[512, 512, 256]`. This resolution checks the
refined data path instead of only the root-level domain.

## Numerical Check

Current density is the derived vector field:

```text
J = curl(B)
```

where `B` is the uniform Cartesian magnetic field with components `b1`, `b2`,
and `b3`. Finite differences use physical spacing derived from the fixture
bounds and the selected uniform resolution.

The center-slice smoothness metric is:

```text
p99(|grad |J||) / (median(|grad |J||) + epsilon)
```

The high-percentile numerator avoids making the test fail on a single legitimate
sharp physical feature. The median denominator gives a robust scale for the
slice. The threshold should be measured from `data/weno509_sub_0000.dat` during
implementation and then set conservatively above the measured value. The test
should fail when the ratio indicates broad numerical ripples or block-boundary
artifacts inconsistent with a smooth current-density slice.

## Diagnostic Report

When the heavy validation runs, it writes:

- `report/amrvac-current/report.md`
- `report/amrvac-current/current_center_z.png`

The PNG shows the center `z` slice of `|J|` and labels the fixture, slice index,
metric value, and threshold. The Markdown report includes:

- fixture path
- enable gate
- validation pipeline
- uniform-grid resolution
- selected slice
- metric formula
- measured metric value
- threshold
- pass/fail result
- embedded PNG path

The report and PNG are generated artifacts, not source files.

## Documentation

Update the development test documentation, primarily `docs/installation.md`, to
include an optional heavy AMRVAC validation section. The documented command is:

```bash
SIMESH_RUN_HEAVY_TESTS=1 PYTHONPATH=src .venv/bin/python tests/amrvac/test_amrvac_dataset.py
```

The docs should state that `make test` remains default-safe because the heavy
validation returns unless the environment gate is set, and that the fixture must
exist at `data/weno509_sub_0000.dat`.

## Scope

This design intentionally avoids adding production API changes. Helper functions
for current-density calculation, smoothness measurement, PNG creation, and
Markdown report writing can live near the heavy test unless implementation shows
that a reusable module is warranted.

No ADR is needed for this decision. It is a reversible validation-test policy
and generated-artifact convention, not a hard-to-reverse architectural choice.
