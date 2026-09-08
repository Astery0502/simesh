# Thermal Ray Traversal And Parallelism

Status: implementation and measured comparison in progress, 2026-09-08.
The user explicitly reopened thermal ray optimization and parallel tests after
`a61a0ac`; the previous 500x500 estimates were extrapolations, not acceptance.
This record extends the existing [thermal evidence](p3-l-thermal.md), retaining
its physical model and all real-temperature/calibration limitations.

## Source Review And Candidate Selection

Sources were checked on 2026-09-08. External capabilities below are source facts;
our adoption decisions are inferences for the current double-precision AMR/EUV
contract, not benchmarks of those libraries.

| Library / algorithm | Relevant source facts | Decision for this round |
| --- | --- | --- |
| [Embree](https://www.embree.org/) | Optimized ray/geometry traversal, including macOS ARM CPUs; a building block for renderers. | Keep as a future locator/BVH comparator. The current source already has a validated octree and ray ownership; a second hierarchy/adapter is not required to remove Python overhead. No claim our locator beats Embree. |
| [OSPRay / Open VKL AMR](https://www.ospray.org/documentation.html#adaptive-mesh-refinement-amr-volume) | Scientific volume rendering; AMR block scalar input is documented as OSP_FLOAT, with CURRENT/FINEST/OCTANT reconstruction choices. OSPRay delegates volume access to Open VKL. | Defer a direct quantitative EUV replacement: representation, precision and interface interpolation need explicit conformance work. It remains a credible visualization/export route. |
| [Open VKL interval iterators](https://www.openvkl.org/) | Segmented traversal, optimized volume sampling and AMR support; intervals can support empty-space skipping and step selection. | Adopt the principle of traversing relevant ray intervals and sampling inside a known owner. Our immutable AMR hierarchy supplies the intervals without conversion. |
| [yt off-axis projection](https://yt-project.org/doc/reference/api/yt.visualization.volume_rendering.off_axis_projection.html) | AMRKDTree support, configurable OpenMP thread count, path-length integration, and separately documented interpolation/ghost options. | Useful functional reference. Its pre-interpolated field/weight composition does not directly define our nonlinear n,T-first response. No uncalibrated end-to-end timing comparison is made. |
| [ExaBricks](https://arxiv.org/abs/2009.03076) | Structured-AMR GPU volume/isosurface ray tracing using reorganized bricks and support regions. | Keep the earlier E1/E2/E4 dispositions: a new reconstruction/layout/device stack is a distinct investigation. No NVIDIA GPU path is available on this ARM Mac. |

Pretty volume rendering can include opacity, lighting and tone/color transfer
functions. Those operations do not establish DN/s/pixel optically thin emissivity
integration. This round preserves complete positive path contributions: no early
opacity termination, arbitrary intensity threshold or denoising is used to claim
speed. Conservatively proven zero-emission intervals could be a future response-
aware optimization, but no such skip or error relaxation is introduced here.

## Actual Hotspots Before Optimization

`profile_thermal.py` recorded the original Python implementation on a 32x32
full-domain oblique view with the same manufactured WENO temperature and /4
quadrature. Profiling took 4.027 s and 1,652,934 function calls. Nested cumulative
attribution: ray-node construction 1.303 s; response/emissivity 0.887 s; scanning
all leaves for intersections 0.567 s; prepared sampling 0.452 s; orchestration
itself 0.511 s. These are instrumented diagnostics, not additive disjoint stages
or baseline wall-time measurements.

There were 13,575 intersected-leaf visits. The old path allocated/sorted knot and
point arrays per visit, recomputed response-table logarithms per block, and located
points whose interval owner was already known. Threads around these Python loops
would leave substantial coordination and GIL work. A general rendering dependency
would not by itself fix the Python response/point-production boundary.

## Selected Implementation

A separate `thermal_rays.pyx` uses the existing native ray interval owner and
trilinear sampler through `native.pxd`. The unchanged interpolation body now
lives inline in that shared declaration file; interval ownership remains in
`native.pyx`. An initial C-API-per-sample version passed the first WENO check but
still passed three memoryview descriptors by value at every sample. Generated-C
inspection justified the inline refinement; partial first-candidate evidence is
preserved under `capi-probe/` and its long run was deliberately stopped.
Tree traversal avoids the full-leaf scan. Incremental cell-center-plane crossing
avoids per-leaf knot sorting/allocation, and each interval directly supplies its
owner for interpolation. Table logarithms/slopes are immutable constants;
response interpolation and n^2 R(T) remain double precision without a surrogate
fit, precision reduction or relaxed compiler math.

The compiled row loop releases the GIL and keeps ray/knot/response state on its
private stack. The Python wrapper dispatches bounded coherent row ranges to
1/2/4 thread-pool calls, shares readonly prepared data and writes disjoint pixels.
No preparation or numerical cache is mutated. No new OpenMP policy, process pool
or persistent worker runtime is introduced; main-session backend work remains
separate. Default worker count is one. `implementation="reference"` retains the
Python all-leaf path for comparison and requires one worker.

Both paths preserve the same thermodynamic reconstruction and composite Gauss2
subdivision count. Incremental knots and ordered scalar accumulation can differ
from NumPy's unique/dot in roundoff, so finite agreement is tested at rtol=atol=
1e-10 DN/s/pixel; per-ray serial/parallel output must agree exactly. ULP-sized
grazing intervals retain their weights and clamp rounded sample coordinates to
their selected half-open owner. Sample budgets count actually evaluated samples;
the compiled path checks each knot interval before evaluating it, whereas the
reference prechecks a whole leaf. Failed pixels remain invalid NaN and never
certify a shortened integral. No bitwise equality of failure sample counters or
roundoff-degenerate knot counts is claimed.

## Measurements And Outcome

Pending matched 64x64 comparisons and actual 500x500 one/two/four-worker images.
Raw attribution: ignored `thermal-profile.txt`, `thermal-profile-setup.json`,
`thermal-reference.prof`. New build/test/benchmark artifacts stay under the same
`benchmark-results/analysis-core/` directory. The original thermal.json remains
historical evidence and must not be overwritten by this follow-up.
