# AMRVAC Current-Density Diagnostic Design

## Context

The AMRVAC legacy comparison diagnostic has moved the investigation past
ghost-cell exchange and selected-slice interpolation:

- refined datasets with coarse/fine interfaces now reject `ghost_width=1`
- current and legacy paths both use `ghost_width=2` for the refined WENO
  fixture
- ghost-cell slice comparison passes with near-roundoff error
- selected uniform interpolation slice comparison passes with near-roundoff
  error

The remaining failing validation is the heavy AMRVAC current-density check on
`data/weno509_sub_0000.dat`. It opens the fixture with `ghost_width=2`, samples
magnetic fields `[4, 5, 6]` onto the full-domain uniform grid, computes
`J = curl(B)`, and evaluates a smoothness ratio on the center `z` slice. The
latest report gives a smoothness metric of `1260.9280421` against the current
threshold `300`.

Because the canonical path now matches legacy through the evaluated
interpolation slice, the next diagnostic should validate the current-density
math and metric independently before changing production AMR behavior.

## Goal

Add a rerunnable, gated diagnostic that proves or falsifies the current-density
validation path and writes a report under:

`report/amrvac-current-diagnostic/report.md`

The diagnostic should identify whether the current failure is caused by:

- incorrect curl math or axis/spacing conventions
- sensitivity to spacing convention or axis interpretation
- sensitivity to smoothing or the smoothness metric
- a likely real field/sampling artifact downstream of the already validated
  interpolation slice

## Non-Goals

- Do not modify production AMR, Cython, or interpolation behavior.
- Do not change the existing heavy current validation threshold.
- Do not make the large WENO fixture part of default test execution.
- Do not claim a production fix from visual improvement alone.
- Do not require the WENO fixture for synthetic current-density checks.

## Proposed Test Shape

Add one focused gated test file:

`tests/amrvac/test_amrvac_current_diagnostic.py`

Run it with:

```bash
SIMESH_RUN_HEAVY_TESTS=1 PYTHONPATH=src .venv/bin/python tests/amrvac/test_amrvac_current_diagnostic.py
```

If the gate is not enabled, the test should skip with a clear message. If the
WENO fixture is missing, synthetic controls should still run and the WENO
section should be reported as `not evaluated`.

## Diagnostic Layers

### Layer A: Synthetic Curl Controls

Create compact magnetic-field arrays in the same layout used by current
validation:

`(3, nx, ny, nz)` for `(Bx, By, Bz)`.

Use analytic fields with known curl, including at least:

- `B = (0, 0, x)`, which gives `J = curl(B) = (0, -1, 0)`
- `B = (-y / 2, x / 2, 0)`, which gives `J = (0, 0, 1)`

Compute current using the same helper shape as the existing validation. Compare
the computed components and magnitude against the analytic expectation with
tight numerical tolerances. Report max and mean absolute component errors for
each synthetic case.

If synthetic controls fail, the report should be written before assertion and
the diagnosis should be that the current-density helper or its axis/spacing
convention is wrong before the WENO fixture enters the pipeline.

### Layer B: WENO Fixture Variants

When `data/weno509_sub_0000.dat` is available, load the fixture through the
canonical AMRVAC API:

1. open with `ghost_width=2`
2. load magnetic fields `[4, 5, 6]`
3. sample the full-domain uniform grid with linear interpolation
4. compute the center `z` current-density slice under controlled variants

Use the existing full-domain resolution:

`ds.domain_nx * 2 ** (ds.levmax - 1)`

Evaluate at least these variants:

- spacing convention `(xmax - xmin) / (resolution - 1)`
- spacing convention `(xmax - xmin) / resolution`
- no smoothing
- existing two-pass smoothing
- optional axis-swapped interpretation as a diagnostic only, not as a
  production assumption

For each variant, report:

- spacing convention
- smoothing passes
- axis interpretation
- `|J|` finite-value count
- `|J|` percentiles
- smoothness ratio
- result classification

The WENO variant section should not fail simply because the current metric is
high. Its purpose is to classify the failure source, not enforce a fix.

## Diagnosis Rules

- If synthetic curl controls fail, diagnose a current math or convention bug.
- If synthetic controls pass and WENO metrics change drastically under spacing
  or axis variants, diagnose a convention-sensitive failure.
- If synthetic controls pass and smoothing changes pass/fail behavior,
  diagnose a smoothing or metric-sensitive failure.
- If synthetic controls pass and WENO metrics remain high across controlled
  variants, diagnose either a real field/sampling artifact or a metric that is
  not aligned with intended visual quality.
- If the WENO fixture is missing, report a partial diagnosis based on synthetic
  controls only.

## Report Output

The Markdown report should include:

- fixture path and availability
- command gate
- synthetic field definitions
- expected curl for each synthetic field
- synthetic max and mean component errors
- WENO setup: field indices, `ghost_width=2`, resolution, and physical domain
- WENO variant table with spacing convention, smoothing passes, axis
  interpretation, current percentiles, smoothness ratio, and classification
- final diagnosis

The report should be written before any failing assertion so diagnostics remain
available after failure.

## Test Behavior

- The test is gated by `SIMESH_RUN_HEAVY_TESTS=1`.
- Synthetic controls run whenever the gated command is run.
- The WENO section is optional when the fixture is absent.
- Synthetic controls use strict assertions.
- WENO variants fail only for invalid setup or report-generation errors, not
  for a high smoothness metric.

## Success Criteria

- One explicit command writes `report/amrvac-current-diagnostic/report.md`.
- Synthetic curl correctness is established or falsified.
- The report classifies the current failure as math/convention-sensitive,
  smoothing/metric-sensitive, or likely downstream of interpolation.
- No production code changes are made.
