# simesh Context

This context names the AMRVAC data and derived-field concepts used by `simesh`.
It is a glossary for project language, not an implementation specification.

## Language

**AMRVAC `.dat` Snapshot**:
A binary AMRVAC-style adaptive-mesh snapshot containing metadata, forest/tree structure, and block field payloads.
_Avoid_: datfile, raw file

**Ghost-Cell Exchange**:
The operation that fills ghost-cell storage around AMR block interiors from neighboring blocks or physical boundary modes.
_Avoid_: ghostcells exchange, ghost update

**Uniform Magnetic Field**:
The cell-centered Cartesian magnetic field components `b1`, `b2`, and `b3` sampled from AMR block data onto a uniform grid.
_Avoid_: uniform b, uniformized b

**Current Density**:
The derived vector field `J = curl(B)` computed from a uniform Cartesian magnetic field.
_Avoid_: current condition

## Example Dialogue

Developer: "For the heavy validation test, should we compute Current Density directly from AMR block data?"

Domain expert: "No. First perform Ghost-Cell Exchange, sample `b1`, `b2`, and `b3` into a Uniform Magnetic Field, then compute Current Density as `J = curl(B)`."

Developer: "Which AMRVAC `.dat` Snapshot should the test use?"

Domain expert: "Use `data/weno509_sub_0000.dat`; skip the heavy validation when that snapshot is absent."

Developer: "When should that validation run?"

Domain expert: "Only when heavy tests are explicitly enabled with `SIMESH_RUN_HEAVY_TESTS=1`."
