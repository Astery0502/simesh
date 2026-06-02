# AMRVAC Legacy Comparison Diagnostic Design

## Context

The current heavy AMRVAC current validation uses
`data/weno509_sub_0000.dat` and intentionally fails because the current-density
slice still shows visible striping. The previous production experiment changed
linear uniform interpolation to avoid exchanged ghost cells, reduced the heavy
metric, and then had to be reverted because it broke the existing affine
interpolation contract in `tests/utils/lib/test_amr.py`.

The next diagnostic should locate the first divergence between the canonical
AMRVAC implementation and the preserved legacy implementation. The user
expects the legacy path to act as ground truth, with special attention on
ghost-cell exchange and uniform interpolation. The diagnostic must stay light:
compare selected slices rather than the full WENO volume.

## Goal

Add a rerunnable, gated test that compares the current implementation against
the legacy implementation on `data/weno509_sub_0000.dat`, writes a report under
`report/`, and diagnoses the earliest evaluated divergence point:

- ghost-cell exchange
- uniform interpolation
- later current-density or visualization processing

The diagnostic is not expected to fix production AMR behavior.

## Non-Goals

- Do not modify production AMR, Cython, or legacy behavior.
- Do not compare the full WENO volume.
- Do not add a new public API.
- Do not replace the existing heavy current validation.
- Do not make the 997 MB fixture part of normal default test execution.

## Proposed Test Shape

Add one focused test file:

`tests/amrvac/test_amrvac_legacy_comparison.py`

The test should be gated by the existing heavy-test convention:

```bash
SIMESH_RUN_HEAVY_TESTS=1 PYTHONPATH=src .venv/bin/python tests/amrvac/test_amrvac_legacy_comparison.py
```

If the gate is not enabled or the fixture is missing, the test should skip with
a clear message.

## Current Path

The current path should:

1. Open `data/weno509_sub_0000.dat` with `ghost_width=1`.
2. Load magnetic fields `[4, 5, 6]`.
3. Call `exchange_ghost_cells()`.
4. Expose `blocks(include_ghosts=True)` for ghost-padded comparison.
5. Compute a bounded linear `uniform_grid(...)` slice for interpolation
   comparison.

## Legacy Path

The legacy path should:

1. Load the same fixture through
   `simesh.legacy.frontends.amrvac.io.amr_loader`.
2. Enable legacy ghost-cell handling.
3. Treat legacy `getbc()` output as ground truth.
4. Expose `mesh.data` for ghost-padded block comparison.
5. Use a legacy uniform interpolation path if one is available and compatible.

If legacy uniform interpolation cannot be used safely, Stage B should be
reported as `not evaluated`; Stage A should still run and report ghost-cell
comparison results.

## Diagnostic Stages

### Stage A: Ghost-Cell Slice Comparison

Compare one or a few selected ghost-padded block/interface slices between
current and legacy. The selection should be deterministic and should prefer an
internal block boundary or coarse/fine interface because those regions are the
most relevant to striping artifacts.

If metadata or connectivity alignment between current and legacy block order
cannot be proven, the test should fail as `comparison setup invalid` rather
than reporting a false ghost-cell divergence.

### Stage B: Uniform Interpolation Slice Comparison

Compare a small uniform-grid slice over the same physical region selected in
Stage A. Use linear interpolation in the current path and the closest matching
legacy interpolation path.

This stage should compute numeric differences directly, not image smoothness
or current-density metrics.

## Diagnosis Rules

- If Stage A fails, diagnose the first divergence as ghost-cell exchange.
- If Stage A passes and Stage B fails, diagnose the first divergence as
  uniform interpolation.
- If Stage A passes and Stage B passes, diagnose the currently evaluated
  pipeline as matching legacy through interpolation; the remaining artifact is
  likely later than interpolation.
- If Stage B is not evaluated, report a partial diagnosis based on Stage A
  only.

## Metrics

For each evaluated slice comparison, report:

- selected block or physical region
- field names and field indices
- slice orientation and index
- finite-value counts
- maximum absolute error
- mean absolute error
- relative error against a legacy scale
- pass/fail status

Ghost-cell slices should use a tight near-exact tolerance because both paths
read the same input data and should fill deterministic ghost values.

Uniform interpolation slices should use a small absolute and relative tolerance
to allow harmless floating-point differences while still catching visible
artifacts.

Initial tolerances:

- ghost-cell slices: `rtol=1e-12`, `atol=1e-12`
- uniform interpolation slices: `rtol=1e-10`, `atol=1e-10`

If the comparison setup itself introduces a known layout conversion or
coordinate alignment roundoff larger than these values, the implementation
should keep the strict ghost-cell tolerance and document any uniform tolerance
change in the report.

## Report Output

Write report artifacts under:

`report/amrvac-legacy-comparison/`

The Markdown report should include:

- fixture path
- command gate
- current implementation setup
- legacy implementation setup
- selected diagnostic region
- Stage A metrics and result
- Stage B metrics and result, or why it was not evaluated
- final diagnosis

The report should be written before any failing assertion so diagnostics remain
available after a failed run.

Optional small PNG or NumPy text artifacts may be added for the selected slice
only, but the core requirement is the Markdown report.

## Success Criteria

- The gated test can be rerun with one explicit command.
- Running the gated test writes `report/amrvac-legacy-comparison/report.md`.
- The report identifies the earliest evaluated divergence point.
- The comparison is slice-level and does not load or compare the full uniform
  WENO volume.
- Existing production AMR behavior and legacy behavior are unchanged.
