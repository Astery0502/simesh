# AMRVAC Derived Current Validation Design

## Context

The AMRVAC derived-variable design is implemented in the canonical dataset
layer. Existing local reports already document a manual current-density
diagnostic over `data/weno509_sub_0000.dat`, including the command shape,
heavy-test gate, slice layout, visual style, and report format.

This validation should not replace those reports. It should add a comparable
derived-variable path that proves `J = curl(B)` can be registered,
materialized, sampled, visualized, and reported through the new
`AMRVACDataSet` derived-field API.

## Goal

Add an optional heavy verifier that mimics the existing current-density
diagnostic while computing current density through materialized derived fields.

The verifier should:

- use `data/weno509_sub_0000.dat`
- follow the existing optional heavy-test gate
- load magnetic fields `[4, 5, 6]` as `b1`, `b2`, and `b3`
- open the dataset with `ghost_width=2`
- register ghost-required derived current components `j1`, `j2`, and `j3`
- materialize those components with `AMRVACDataSet.materialize_fields(...)`
- sample or select the materialized current components by field name
- compute `|J|` from the materialized components
- write the same style of slice images and markdown report as the former
  optional current diagnostic

## Non-Goals

- Do not change the derived-variable public API.
- Do not change the old manual current-density reports.
- Do not refactor unrelated AMRVAC loading, mesh, interpolation, or plotting
  code.
- Do not make the heavy verifier part of normal test execution.
- Do not add new scientific acceptance thresholds beyond the existing
  diagnostic comparison style.

## Recommended Implementation Shape

Add a new optional pytest-style verifier that mirrors the existing repository
pattern for local heavy AMRVAC diagnostics. The expected command should follow
the same shape as the existing report documents:

```bash
SIMESH_RUN_HEAVY_TESTS=1 PYTHONPATH=src .venv/bin/python tests/amrvac/test_amrvac_current_derived_optional.py
```

The exact test filename may be adjusted to match the existing local optional
test naming convention if a tracked or untracked predecessor is available, but
the verifier should remain a gated test rather than a one-off script.

Use a separate report directory, for example:

```text
report/amrvac-derived-current-diagnostic/
```

This keeps the existing manual-current artifacts intact and makes the derived
path easy to compare against `report/amrvac-current-diagnostic/`.

## Data Flow

1. Skip unless `SIMESH_RUN_HEAVY_TESTS=1`.
2. Skip clearly if `data/weno509_sub_0000.dat` is unavailable.
3. Open the AMRVAC dataset with `ghost_width=2`.
4. Load magnetic field indices `[4, 5, 6]`, which are expected to resolve to
   `b1`, `b2`, and `b3`.
5. Register `j1`, `j2`, and `j3` as ghost-required derived fields.
6. In each derived function, read padded magnetic fields from the derived
   context and compute the corresponding Cartesian curl component on the
   interior cells.
7. Materialize all three current components.
8. Use name-based field selection for `j1`, `j2`, and `j3` when moving through
   the block or uniform-grid path.
9. Compute `|J| = sqrt(j1**2 + j2**2 + j3**2)`.
10. Generate the same mid-x, mid-y, and mid-z baseline slice images as the
    former diagnostic, using the same display transform, colormap, clipping
    convention, and table style where practical.
11. Write `report.md` summarizing the command gate, fixture, loaded fields,
    derived fields, grid resolution, slice indices, numerical statistics, and
    image paths.

## Error Handling

Missing fixture data should be a skip, not a failure. A missing optional gate
should also skip. Derived-field failures should fail the optional verifier with
the original exception visible, because the purpose of the verifier is to catch
registration, ghost exchange, materialization, and name-selection regressions.

If the loaded field names do not include `b1`, `b2`, and `b3`, the report or
failure message should include the available loaded names so the data/header
mismatch is obvious.

## Output Artifacts

The report directory should contain:

- `report.md`
- `current_mid_x.png`
- `current_mid_y.png`
- `current_mid_z.png`

Additional diagnostic images may be added only if they directly mirror the
former optional diagnostic or clarify a failure.

## Success Criteria

The validation is successful when the optional command:

- runs with the repository venv Python
- loads `data/weno509_sub_0000.dat`
- registers and materializes `j1`, `j2`, and `j3`
- exercises name-based selection of materialized derived fields
- produces finite `|J|` values for the displayed slices
- writes the derived-current report and three baseline slice images
- preserves the old manual-current reports unchanged

## Scope Boundary

The implementation should be surgical. Expected touch points are limited to a
new optional test/verifier and its generated report artifacts. Any required
helper code should first try to live inside the optional verifier unless a
small shared helper already exists in the repository.
