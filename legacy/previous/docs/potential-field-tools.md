# Potential-Field Extrapolation Tools

This page describes the Green-function potential-field model used by
`simesh.tools.potential_field_green`. The tool constructs a cell-centered
Cartesian magnetic field above a prescribed bottom normal field.

## Continuous Model

The method assumes a potential magnetic field in the half-space above the
lower boundary,

$$
z > z_{\min},
$$

with bottom normal field

$$
B_z(x, y, z_{\min}).
$$

For the decaying half-space convention, the Green-function extrapolation is

$$
\mathbf{B}(x, y, z)
= \frac{1}{2\pi}
\int B_z(\xi, \eta, z_{\min})
\frac{(x - \xi,\ y - \eta,\ z - z_{\min})}{r^3}
\, d\xi\, d\eta ,
$$

where

$$
r =
\sqrt{(x - \xi)^2 + (y - \eta)^2 + (z - z_{\min})^2}.
$$

Equivalently, each infinitesimal bottom surface element contributes a vector
parallel to the displacement from the source point $(\xi, \eta, z_{\min})$ to
the target point $(x, y, z)$. The contribution decays as $1 / r^2$ in magnitude
because the displacement vector is divided by $r^3$.

Component-wise, the model is

$$
B_x(x, y, z)
= \frac{1}{2\pi}
\int B_z(\xi, \eta, z_{\min})
\frac{x - \xi}{r^3}
\, d\xi\, d\eta ,
$$

$$
B_y(x, y, z)
= \frac{1}{2\pi}
\int B_z(\xi, \eta, z_{\min})
\frac{y - \eta}{r^3}
\, d\xi\, d\eta ,
$$

$$
B_z(x, y, z)
= \frac{1}{2\pi}
\int B_z(\xi, \eta, z_{\min})
\frac{z - z_{\min}}{r^3}
\, d\xi\, d\eta .
$$

These equations define the mathematical field being approximated. The
finite-array implementation samples this model over a finite rectangular
bottom patch rather than over an infinite plane.

## Discrete Midpoint-Source Approximation

The input bottom field is defined on a uniform grid with cell widths `dx` and
`dy`. The method approximates the surface integral by a midpoint quadrature
over bottom cells.

Let

$$
\xi_i = x_{\min} + \left(i + \frac{1}{2}\right)\Delta x,
$$

$$
\eta_j = y_{\min} + \left(j + \frac{1}{2}\right)\Delta y,
$$

$$
z_k = z_{\min} + \left(k + \frac{1}{2}\right)\Delta z
$$

be cell-center coordinates. For a target cell center $(x_m, y_n, z_k)$, each
bottom cell `(i, j)` contributes as if its flux were concentrated at the
bottom cell center $(\xi_i, \eta_j, z_{\min})$ and weighted by the source-cell
area $\Delta x\,\Delta y$.

The discrete field is therefore

$$
B_c[m,n,k]
= \sum_i \sum_j b^{\mathrm{bottom}}_3[i,j]\,
K_c[m-i,n-j,k],
$$

with component kernels

$$
K_x[p,q,k]
= \frac{1}{2\pi}\Delta x\,\Delta y
\frac{p\,\Delta x}{r_{pqk}^3},
$$

$$
K_y[p,q,k]
= \frac{1}{2\pi}\Delta x\,\Delta y
\frac{q\,\Delta y}{r_{pqk}^3},
$$

$$
K_z[p,q,k]
= \frac{1}{2\pi}\Delta x\,\Delta y
\frac{z_{\mathrm{offset},k}}{r_{pqk}^3},
$$

where

$$
z_{\mathrm{offset},k}
= z_k - z_{\min}
= \left(k + \frac{1}{2}\right)\Delta z,
$$

$$
r_{pqk}
= \sqrt{(p\,\Delta x)^2 + (q\,\Delta y)^2
+ z_{\mathrm{offset},k}^2}.
$$

The output field is sampled at cell centers above the lower boundary. In
particular, the first `z` layer is at `z_min + 0.5 dz`. It is an extrapolated
field value, not a copy of the bottom boundary field.

## Bottom-Flux Balancing

The half-space Green-function convention represents the decaying field
associated with the nonzero horizontal structure of the bottom flux. On a
finite box, a nonzero mean bottom field is a zero horizontal wavenumber mode.
That mode does not have a decaying half-space counterpart in the same sense as
the finite-wavenumber modes.

By default, the tool removes this finite-box zero mode before extrapolation by
subtracting the mean bottom field:

$$
\overline{b_3}
= \frac{1}{n_x n_y}
\sum_i \sum_j b^{\mathrm{bottom}}_3[i,j],
$$

$$
b^{\mathrm{balanced}}_3[i,j]
= b^{\mathrm{bottom}}_3[i,j] - \overline{b_3}.
$$

The Green-function sum is then applied to `b3_balanced`. This makes the
discrete bottom flux sum zero,

$$
\sum_i \sum_j b^{\mathrm{balanced}}_3[i,j] = 0,
$$

so the extrapolation is driven by finite-box variations about the mean rather
than by a uniform offset.

## Direct and FFT Algorithms

Both available algorithms evaluate the same discrete Green-kernel convolution.
They differ only in how the sums are computed.

The direct method evaluates the discrete convolution explicitly. For every
target cell `(m, n, k)` and every component `c`, it sums all source-cell
contributions:

$$
B_c[m,n,k]
= \sum_i \sum_j b^{\mathrm{bottom}}_3[i,j]\,
K_c[m-i,n-j,k].
$$

At a fixed height `k`, the same operation can be written as a 2D convolution
for each component:

$$
B_c(\cdot,\cdot,k)
= b^{\mathrm{bottom}}_3 * K_c(\cdot,\cdot,k),
$$

where `*` denotes convolution over the two horizontal directions.

FFT acceleration uses the convolution theorem:

$$
\mathcal{F}(f * g) = \mathcal{F}(f)\,\mathcal{F}(g).
$$

For each height and component, the bottom field and Green kernel are
transformed, multiplied in Fourier space, inverse-transformed, and cropped to
the same physical output region as the direct convolution. This changes the
algorithmic cost, not the mathematical model: the direct and FFT backends
compute the same discrete Green-kernel convolution.

## Inputs and Outputs

`b3_bottom` is a 2D array containing the bottom-face normal magnetic field on
the plane `z = xmin[2]`. Its shape defines the horizontal cell counts:

$$
(n_x, n_y) = \operatorname{shape}(b^{\mathrm{bottom}}_3).
$$

`xmin`, `xmax`, and `nz` define a uniform Cartesian box. The horizontal
spacing comes from `b3_bottom.shape` and the `x` and `y` extents; the vertical
spacing comes from `nz` and the `z` extent.

The returned `bfield` has shape

$$
(3, n_x, n_y, n_z)
$$

with components ordered as

$$
(B_x, B_y, B_z).
$$

The `backend` option controls how the convolution is evaluated:

| Backend | Meaning |
| --- | --- |
| `auto` | Use FFT acceleration when SciPy is available; otherwise use the direct method. |
| `fft` | Require SciPy-backed FFT convolution. |
| `direct` | Use explicit summation, mainly for validation or fallback. |
