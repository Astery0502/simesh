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
- `backend="auto"` returns finite output
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
