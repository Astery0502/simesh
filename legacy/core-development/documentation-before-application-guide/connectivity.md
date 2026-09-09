# Magnetic connectivity on AMR fields

`qsl` computes squashing factors, localized magnetic footpoints, line length,
and optional twist from completed native `Fields`. It consumes Cartesian 3D AMR
directly, without resampling a uniform volume or invoking FastQSL. Existing
`trace` keeps its accepted-prefix behavior.

## A map from arbitrary seed points

```python
import numpy as np
import simesh as sm

limit = 2 * 1024**3
with sm.open_amrvac("snapshot.dat", memory_limit=limit) as source:
    magnetic = sm.prepare(source, ("b1", "b2", "b3"),
                          scheme="coordinate-phase", memory_limit=limit)

lo, hi = magnetic.mesh.lower, magnetic.mesh.upper
x = np.linspace(lo[0], hi[0], 66)[1:-1]
y = np.linspace(lo[1], hi[1], 66)[1:-1]
xx, yy = np.meshgrid(x, y, indexing="ij")
seeds = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, lo[2])))
result = sm.qsl(magnetic, seeds, step_fraction=.125, workers=4,
                memory_limit=limit)
log_q_map = np.where(result.valid, result.log10_q, np.nan).reshape(xx.shape)
twist_map = np.where(result.complete, result.twist, np.nan).reshape(xx.shape)
```

Seeds have shape `(n,3)` in physical mesh coordinates. Their order is preserved;
they may describe a surface, AMR cell centers, a volume, or unrelated points.
Seeds on either side of the closed physical box are accepted. The field group
contains three ordered magnetic components with common units. Labels do not
convert units or normalize current.

For twist without Q, `line_diagnostics(..., quantities=("twist",))` and its
iterator reuse the central integration with no Q neighbors or unit-vector
gradients. Q arrays are then None and `valid` means complete finite twist.
The `qsl` and `iter_qsl` interfaces still always compute Q. See
[standard applications](standard-applications.md) for surface and bottom helpers.

## Methods and normalization

The default `method="finite-difference"` traces the central line and four nearby
lines, then differentiates the two endpoint maps. Interior perturbations lie in
a plane perpendicular to B; boundary seeds use their box face's tangent plane.
`delta` is the physical perturbation distance. Its default is 0.001 times the
smallest cell spacing at the seed, reduced to remain inside the box and local
sphere. Unresolved perturbations or stencils reaching different boundary types
give `stencil_valid=False` and NaN Q. This is not an automatic separatrix classifier.

`method="variational"` transports two initially orthonormal transverse vectors:
`dU/ds = direction * (U · grad)b`, and likewise for V, with `b = B / |B|`.
The existing centered-derivative operator supplies gradients of normalized
prepared nodes. This approximates the derivative of the sampled direction,
especially near physical and refinement boundaries. The vectors are rescaled
together during integration, with logarithmic scale factors. Two valid input
halo layers are required. This method can miss separatrices; compare it with
endpoint differences when connectivity gradients are important.

Endpoint variations are projected along B onto their target surfaces. Q is the
squared Frobenius norm of the mapping Jacobian divided by its absolute determinant.
`normalization="mapping"` (default) obtains the determinant from transported
areas without assuming exactly solenoidal discrete B. `normalization="flux"`
uses the magnetic-flux identity employed by FastQSL. The latter can give Q below
two when the discrete field or transport violates that identity; values are not
silently clamped. Method and normalization are recorded in the result.

`q_perp` uses planes perpendicular to endpoint B. Its finite-difference version
requires delta and step convergence checks; the variational version is the closer
counterpart of FastQSL's Scott-method output. `log10_q` and `log10_q_perp` are
evaluated before exponentiation and can remain finite when Q overflows. Singular
or numerically collapsed maps are invalid rather than assigned an arbitrary high Q.

## Twist and surface events

Twist integrates `(curl(B) · B) / (4*pi*|B|^2)` over positive arc length on both
branches; contributions are added. `curl_field` accepts a retained native
`curl(magnetic)` with matching provenance. Otherwise one curl is reused across
batches. `twist=False` skips it. Finite differences without twist require only
one valid halo; automatically computing curl requires two.

RK4 steps are capped by `step_fraction` (default 0.25, at most 1) times the
smallest cell spacing encountered in a trial, plus optional physical cap `step`.
Local-sphere runs also cap steps at one quarter of the sphere radius.
`max_steps` (default 10000) and `max_length` apply separately to each branch.
This is spatially limited RK4, not FastQSL's RKF45 error controller.

The default targets are the six physical box faces. Explicit `bounds` chooses a
smaller closed box inside the mesh. This deliberately selects mapping surfaces;
preparation's requested region never becomes a new boundary. Missing field
coverage inside the box stops integration without implicit reads.

Crossings are localized by bracketing and reintegrating the final step, then
placing its endpoint on the surface. Box and local-sphere surface extensions
are used solely for event trials, so missing data beyond a target cannot preempt
that target's event. Default `boundary_tolerance` is the
larger of `1e-10 * min(box widths)` and 32 machine epsilons times the coordinate
scale. It controls localization, not total trajectory error. Grazing exits and
edge/corner targets may not define a regular map.

`local_radius=r` ends each branch at a sphere centered on the original seed,
or a nearer box face. Here `q` and its alias `q_local` describe the local map;
twist and length describe the shortened segment. Neighboring stencil lines share
the central seed's sphere. Use separate calls for global and local products.

| Result | Meaning |
| --- | --- |
| `q`, `log10_q`, `valid` | Squashing between localized surfaces; inspect validity |
| `q_perp`, `log10_q_perp` | Squashing between endpoint-normal planes; may describe a truncated segment |
| `twist`, `length` | Sum over both branches; twist is None when disabled |
| `footpoints`, `endpoint_fields` | `(n,2,3)`; branch 0 is against B, branch 1 along B |
| `termination`, `steps` | `(n,2)` status and accepted-step counts |
| `boundary` | ZMIN/ZMAX=1/2, YMIN/YMAX=3/4, XMIN/XMAX=5/6, EDGE=7, LOCAL_SPHERE=9, NONE=0 |
| `complete` | Both central branches reached targets; a corner can be complete with invalid Q |
| `stencil_valid` | Resolved, consistent neighboring maps; not needed by the variational method |

`ConnectivityTermination` distinguishes limits, missing coverage, null/nonfinite
fields, unrepresentable samples and event-step underflow. Q requires both target
surfaces. Q_perp and twist may remain finite after step/length limits, describing
the accepted segment. Sampling/diagnostic failures invalidate those values.
Incomplete endpoint arrays are last accepted positions, not localized footpoints.

## Batching and validation

`iter_qsl` accepts the same controls plus `seed_batch` and yields owned results.
Gradients and optional curl are built once per iterator. This bounds output
batches, not field loading; inputs remain resident Fields. `memory_limit` counts
inputs, generated diagnostics, outputs and estimated scratch, not process RSS.
Retained previous batches and unrelated products remain the caller's responsibility.

Refine integration step, delta, and input mesh separately for a new dataset.
Output seed refinement alone cannot remove field reconstruction errors. AMR
interfaces and piecewise interpolation can limit convergence. Neither method
promises exact FastQSL discretization equivalence or detection of every separatrix.
Spherical, periodic, CT and GPU analysis remain outside the native profile.

Tests cover analytic Q and signed twist, mixed AMR, endpoint differences,
local spheres, boundary seeds, failures and convergence. Optional upstream comparison:

```bash
.venv/bin/python scripts/compare_fastqsl2.py \
  --reference /path/to/FastQSL2 \
  --reference-python /path/to/reference/python \
  --cells 64 --output comparison.json
```

The external checkout must contain a compiled `fastqsl.x`; its Python environment
needs NumPy and Matplotlib. The script records its exact revision.

## Scientific provenance

Definitions follow public mathematical methods discussed in
[Chen et al. (2026), FastQSL 2](https://arxiv.org/abs/2604.16195), including Scott
et al. (2017), Pariat and Démoulin (2012), Titov (2007), and Berger and Prior (2006).
See also [Zhang et al. (2022), FastQSL](https://arxiv.org/abs/2208.12569).

This independently written implementation uses simesh geometry, interpolation,
derivatives and Cython. No upstream Fortran is translated or bundled.
The [upstream repository](https://github.com/el2718/FastQSL2) declares CC BY-NC-SA 4.0
and remains external; comparison used revision
`314bbf01ab72e43f82cb6b2e1c2a4d22d93aacdd`. simesh's code license is unchanged.
