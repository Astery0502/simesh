# Independent Potential-Field Tools Design

## Purpose

Add a standalone scientific helper for Cartesian potential-field extrapolation
from a uniform bottom `b3` field. The feature is independent of AMRVAC files,
AMR meshes, compiled AMR structures, and dataset objects.

The first tool takes a bottom-face normal magnetic field and returns a
cell-centered magnetic field box with components `(b1, b2, b3)`. It is intended
for direct Python use when a user wants a quick potential-field initialization
or analysis field without constructing a mesh or reading a file.

## Scope

This design covers a first implementation under a broad independent-tools
namespace:

```text
src/simesh/tools/
    __init__.py
    potential_field.py
```

Public import:

```python
from simesh.tools import potential_field_green
```

The tool supports only uniform Cartesian 3D boxes. The bottom field is a 2D
array on the lower `z = xmin[2]` face, below the first cell-centered layer.

In scope:

- direct Green-function potential-field extrapolation
- cell-centered magnetic field output
- automatic `nx`, `ny` inference from `b3_bottom.shape`
- optional bottom-flux balancing
- FFT-accelerated convolution when SciPy is available
- direct convolution backend for validation and fallback
- focused tests for shape, geometry, backend parity, and input validation

Out of scope:

- AMRVAC `.dat` reading or writing
- AMR block or mesh objects
- CT staggered face fields
- vector potential output
- scalar-potential output
- linear force-free fields with nonzero alpha
- spherical geometry
- file export helpers

## Background

AMRVAC's local implementation in
`/Users/astery/science/amrvac/src/physics/mod_lfff.t` has two relevant paths:

- `calc_lin_fff(...)` directly computes the magnetic field `Bf`
- `get_potential_field_potential(...)` computes a scalar potential

The practical AMRVAC examples checked during design call `calc_lin_fff(...)`
and assign its field output to magnetic variables. The scalar-potential route
exists, but it is not the primary initializer path.

The reference implementation in
`/Users/astery/science/project/relaxff-ct/potential_field/potential_ct_green.py`
is CT-oriented. It builds staggered face fields and reconstructs vertical faces
through a discrete CT divergence relation. That machinery is valuable when the
target is solver-compatible face data, but it is intentionally outside this
tool's first scope.

The same reference code uses `scipy.signal.fftconvolve` to accelerate
Green-kernel convolution. Direct and FFT convolution were checked during
design and matched to roundoff-level differences for representative small
Green-kernel cases. FFT was substantially faster for the tested kernel sizes.

## Public API

Primary function:

```python
potential_field_green(
    b3_bottom,
    xmin,
    xmax,
    nz,
    *,
    backend="auto",
    balance_flux=True,
)
```

Return value:

```python
bfield, geometry
```

`bfield` has shape:

```text
(3, nx, ny, nz)
```

where:

- `bfield[0]` is `b1`
- `bfield[1]` is `b2`
- `bfield[2]` is `b3`
- `nx, ny = b3_bottom.shape`

`geometry` is a frozen dataclass that records:

- `xmin`
- `xmax`
- `domain_nx = (nx, ny, nz)`
- `dx`
- `dy`
- `dz`
- enough information to reconstruct cell-center coordinates
- whether bottom-flux balancing was applied
- the mean value removed from `b3_bottom`

The function interprets `b3_bottom[i, j]` as the bottom-face normal field at:

```text
x_i = xmin[0] + (i + 0.5) * dx
y_j = xmin[1] + (j + 0.5) * dy
z   = xmin[2]
```

Output cell centers are at:

```text
x_i = xmin[0] + (i + 0.5) * dx
y_j = xmin[1] + (j + 0.5) * dy
z_k = xmin[2] + (k + 0.5) * dz
```

## Backend Options

The `backend` argument controls how 2D Green-kernel convolutions are evaluated.

`backend="auto"` is the default. It uses FFT convolution through SciPy when
SciPy is available. If SciPy is unavailable, it falls back to the NumPy direct
convolution implementation.

`backend="fft"` forces FFT convolution. It requires SciPy and raises
`ImportError` with a clear message when SciPy cannot be imported.

`backend="direct"` forces direct convolution through the NumPy fallback
implementation. This backend is slower for normal workloads, but it is useful
for small validation cases, debugging, and parity tests.

The FFT backend is an acceleration of the same Green-kernel convolution, not a
separate spectral potential-field solver. It must use the same kernel and crop
rule as the direct backend.

## Data Flow

1. Convert `b3_bottom`, `xmin`, and `xmax` to NumPy arrays.
2. Validate that `b3_bottom` is 2D and infer `nx, ny` from its shape.
3. Validate that `xmin` and `xmax` are length-3 arrays.
4. Validate that `nz` is positive and all domain lengths are positive.
5. Build uniform Cartesian geometry and spacing.
6. If `balance_flux=True`, subtract `mean(b3_bottom)` before extrapolation.
7. For each cell-centered height `z_k`, build Green kernels for `b1`, `b2`,
   and `b3`.
8. Convolve the prepared bottom field with each component kernel.
9. Fill `bfield` in component-first layout `(3, nx, ny, nz)`.
10. Return `bfield` and `geometry`.

## Green-Function Model

The first implementation uses the potential-field half-space kernel:

```text
B(x, y, z) = (1 / 2pi) integral B3(xi, eta, zmin)
             * (x - xi, y - eta, z - zmin) / r^3 dxi deta
```

with:

```text
r = sqrt((x - xi)^2 + (y - eta)^2 + (z - zmin)^2)
```

For the discrete bottom grid, the first implementation uses a midpoint-style
source-cell kernel compatible with uniform 2D convolution. A finite-cell
integrated kernel is left for future work if the midpoint approximation proves
insufficient.

The bottom zero mode is handled by optional flux balancing. With
`balance_flux=True`, the extrapolated field uses:

```text
b3_bottom - mean(b3_bottom)
```

This matches the decaying half-space convention for a finite computational
box. The removed mean is recorded in `geometry`.

## Discrete Correctness Checks

For this first tool, "correctly generated" means correct relative to the
documented discrete Green-function model, not exact compatibility with a CT
solver stencil. The returned field is cell-centered and sampled from a
midpoint-source Green kernel. Therefore, finite-difference divergence and curl
checks are diagnostic convergence checks, not machine-zero invariants.

Use these checks to judge the generated field:

- Direct-sum reference: for small grids, compare the output against an
  independently written explicit source-cell sum using the same midpoint
  kernel, source area factor, cell-center coordinates, and optional flux
  balancing. This is the strongest discrete implementation check because it
  verifies indexing, signs, spacing, kernel normalization, and crop rules
  without relying on the FFT backend.
- Backend parity: compare FFT and direct convolution outputs against the same
  discrete model. Parity only proves that both backends implement the same
  convolution; it does not by itself prove the physics.
- Interior divergence residual: compute a central-difference divergence on
  interior cells only, excluding at least one cell from each boundary. Normalize
  by a characteristic field-gradient scale, for example
  `max(abs(B)) / min(dx, dy, dz)`. The relative residual should be small for
  smooth balanced inputs and should decrease under grid refinement away from
  the bottom source plane.
- Interior curl residual: compute a central-difference curl with the same
  interior mask and normalization. A potential field should have small
  normalized curl, with refinement convergence away from the lower boundary.
- Refinement convergence: repeat a smooth balanced test case at multiple
  resolutions over the same physical box. Compare fields at common physical
  locations or compare normalized divergence/curl residuals. The residuals
  should decrease at roughly the order expected from the finite-difference
  diagnostic and midpoint source approximation, except near boundaries and
  sharp bottom-field features.
- Symmetry and sign checks: use simple balanced sources with known symmetry,
  such as a positive-negative pair. The generated field should preserve the
  expected mirror symmetries, component parities, and sign orientation.
- Height decay: for balanced compact sources, `b3` and horizontal field
  magnitudes should weaken with height in an aggregate norm. This is a useful
  smoke test, but it is weaker than the direct-sum and residual checks.

Do not require `bfield[2, :, :, 0]` to equal `b3_bottom`. The input is a
bottom-face normal field at `z = xmin[2]`, while the output is evaluated at the
first cell-centered height `z = xmin[2] + 0.5 * dz`. With the midpoint kernel,
the first layer is an extrapolated field above the boundary, not a copy of the
boundary data.

If a future tool needs a magnetic field that is exactly divergence-free under
the AMRVAC CT divergence operator, it should produce staggered face fields and
construct or correct them with the same discrete CT stencil. That is a
different acceptance criterion from this independent cell-centered helper.

## Error Handling

Error handling is narrow and explicit:

- raise `ValueError` when `b3_bottom` is not 2D
- raise `ValueError` when `xmin` or `xmax` is not length 3
- raise `ValueError` when `nz <= 0`
- raise `ValueError` when any domain length is non-positive
- raise `ValueError` for unknown backend names
- raise `ImportError` only when `backend="fft"` is requested and SciPy is not
  available
- for `backend="auto"` without SciPy, fall back to direct convolution

No file-system side effects happen inside the function.

## Testing

Add focused tests under a new tools test area, such as:

```text
tests/tools/test_potential_field.py
```

Required coverage:

- output shape is `(3, nx, ny, nz)`
- `nx` and `ny` are inferred from `b3_bottom.shape`
- geometry records `xmin`, `xmax`, `domain_nx`, and spacing correctly
- invalid `b3_bottom`, `xmin`, `xmax`, `nz`, and backend values raise clear
  errors
- `balance_flux=True` subtracts the bottom mean and records the removed value
- direct and FFT backends agree within a tight tolerance when SciPy is
  installed
- direct backend agrees with an explicit small-grid direct-sum reference
- `backend="auto"` returns finite output
- normalized interior divergence and curl residuals are small for a smooth
  balanced source and decrease under a simple refinement check
- symmetric balanced sources preserve expected component parity and sign
  orientation
- a compact balanced source produces finite fields and a weaker `b3` magnitude
  at higher layers than near the bottom

The direct backend is the reference path for backend parity tests.

## Documentation

Update project docs only where the public surface changes:

- `README.md`: mention `simesh.tools` as a home for independent scientific
  helpers if the new namespace is exported as public
- `docs/python-api-map.md`: document the new `simesh.tools` namespace
- `docs/api-reference.md`: add a compact reference entry for
  `potential_field_green(...)`
- `docs/user-guide.md`: add a short example showing bottom-field input and
  `(3, nx, ny, nz)` output

Do not expand AMRVAC file-format docs for this feature because the tool is not
an AMRVAC reader or writer.

## Implementation Phasing

Implement this feature in separated phases. Only one phase should be completed
in one agent turn. If a turn starts a phase, it should stop after that phase's
success criteria are met and report the next handoff checkpoint instead of
continuing into the next phase.

### Phase 1: Public API and Geometry Shell

Belongs in this phase:

- create `src/simesh/tools/`
- export `potential_field_green`
- define the frozen geometry dataclass
- implement input conversion and validation
- infer `nx` and `ny` from `b3_bottom.shape`
- compute `dx`, `dy`, and `dz`
- record flux-balance metadata

Explicitly not in this phase:

- Green kernels
- convolution
- FFT acceleration
- physical correctness tests

Dependencies: none.

Handoff checkpoint: the function signature, import path, validation behavior,
and geometry fields are stable.

Success criteria:

- `from simesh.tools import potential_field_green` works
- invalid inputs raise the specified errors
- geometry records `xmin`, `xmax`, `domain_nx`, and spacing correctly

Risk or ambiguity: geometry fields are described conceptually as enough
information to reconstruct coordinates, so the implementation should choose a
minimal clear field set.

### Phase 2: Canonical Discrete Green Model and Direct Backend

Belongs in this phase:

- implement midpoint-source Green kernels
- include the source area factor
- implement direct NumPy convolution
- fill component-first output `(3, nx, ny, nz)`
- apply optional bottom-flux balancing before extrapolation
- add small-grid direct-sum reference tests

Explicitly not in this phase:

- SciPy FFT backend
- documentation expansion
- AMRVAC, AMR mesh, or CT compatibility work

Dependencies: Phase 1.

Handoff checkpoint: the direct backend is the canonical numerical path and
matches an independent explicit source-cell sum.

Success criteria:

- `backend="direct"` returns finite fields with the correct shape
- signs, spacing, kernel normalization, crop rules, and mean-removal behavior
  are covered by tests

Risk or ambiguity: kernel normalization and crop/index conventions are the
highest-risk details and should be pinned before acceleration is added.

### Phase 3: Backend Dispatcher and FFT Acceleration

Belongs in this phase:

- add `backend="auto"`, `backend="fft"`, and `backend="direct"` dispatch
- use SciPy `fftconvolve` when available
- fall back to direct convolution for `backend="auto"` when SciPy is absent
- raise `ImportError` only for forced `backend="fft"` when SciPy is absent
- add backend parity tests

Explicitly not in this phase:

- a new physics model
- a spectral potential-field solver
- behavior differences between FFT and direct backends

Dependencies: Phase 2.

Handoff checkpoint: FFT and direct use the same kernels and crop rule.

Success criteria:

- direct and FFT outputs agree within a tight tolerance when SciPy is installed
- `backend="auto"` works without requiring SciPy

Risk or ambiguity: SciPy may be absent in some environments, so tests must skip
or branch cleanly.

### Phase 4: Scientific Diagnostic Tests

Belongs in this phase:

- add normalized interior divergence and curl residual tests
- add a simple refinement trend check
- add symmetry, component parity, and sign-orientation checks
- add height-decay checks
- add finite-output checks for compact balanced sources

Explicitly not in this phase:

- CT-stencil machine-zero divergence requirements
- requiring `bfield[2, :, :, 0]` to equal `b3_bottom`

Dependencies: Phases 2 and 3.

Handoff checkpoint: the numerical implementation has physics-oriented
regression coverage beyond shape checks and backend parity.

Success criteria:

- smooth balanced cases produce small normalized interior divergence and curl
  residuals
- residuals decrease under a simple refinement check
- symmetry, sign, and height-decay checks pass

Risk or ambiguity: the design gives qualitative thresholds, so exact tolerances
need conservative calibration from observed behavior.

### Phase 5: Public Documentation

Belongs in this phase:

- update `README.md`
- update `docs/python-api-map.md`
- update `docs/api-reference.md`
- update `docs/user-guide.md`
- include a compact example showing bottom-field input and `(3, nx, ny, nz)`
  output

Explicitly not in this phase:

- AMRVAC `.dat` documentation
- mesh or CT documentation
- file export claims

Dependencies: Phases 1 through 4 should be stable enough to document.

Handoff checkpoint: the API and behavior are stable enough to document without
expected churn.

Success criteria:

- docs show the import path, required arguments, output shape, and geometry
  return value
- docs state that the tool is independent of AMRVAC files and meshes

Risk or ambiguity: avoid overpromising solver-compatible CT fields or exact
boundary copying.

### Phase 6: Final Integration Check

Belongs in this phase:

- run the focused tools tests
- run the relevant existing test command
- verify no file-system side effects happen inside `potential_field_green`
- check the public import in an editable-install context

Explicitly not in this phase:

- feature expansion
- unrelated AMRVAC or legacy refactors

Dependencies: Phases 1 through 5.

Handoff checkpoint: the implementation is ready to hand off as complete.

Success criteria:

- users can import `potential_field_green` from `simesh.tools`
- the default backend works without requiring SciPy
- FFT acceleration works when SciPy is available
- focused tests and public docs are updated
- all implementation success criteria in this design are satisfied

Risk or ambiguity: the build or test environment may not include SciPy, so the
final report should state exactly what was and was not verified.

## Success Criteria

The feature is ready for implementation when the design is accepted. The
implementation is complete when:

- users can import `potential_field_green` from `simesh.tools`
- the function accepts `b3_bottom`, `xmin`, `xmax`, and `nz`
- output is component-first cell-centered data with shape `(3, nx, ny, nz)`
- `nx` and `ny` are inferred from `b3_bottom`
- the default backend works without requiring SciPy
- SciPy FFT acceleration is used when available through `backend="auto"` or
  explicitly through `backend="fft"`
- direct and FFT backends agree within test tolerance
- no AMRVAC file, mesh, or CT structures are required
- focused tests and public docs are updated
