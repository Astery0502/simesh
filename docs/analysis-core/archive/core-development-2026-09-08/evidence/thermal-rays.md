# Thermal Ray Traversal And Parallelism

After worktree integration, raw artifacts referenced below are preserved under
`benchmark-results/analysis-core/euv-geometry/` in the main checkout.
The migration manifest records file sizes and SHA-256 checksums; measurements
below retain their original revision and scope.

Status: completed and retained, 2026-09-08. The native compiled implementation
is adopted; parallel execution is optional and defaults to one worker.
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

All 23 analysis checks passed after shared-inline relocation, including native/
reference thermal parity, manufactured continuum accuracy, negative and grazing
views, variable near bounds, missing/invalid inputs, explicit limits and exact
1/2/4-worker equality. Ordinary F/D/scalar LOS/twist/uniform behavior also passed.
The extra C-API candidate defect found by the new failure check was a nonfinite
sample classification (8 instead of the reference's 4), fixed before promotion.

The first C-API probe overlapped a main-session `probe_paired_backends.py` run.
Its oblique timings became unstable; those results are retained as preliminary,
not used to claim stable parallel performance. The final run waits for that
process's exit via a kernel process-exit event, without interrupting it. Desktop
load and OS page cache still remain uncontrolled. Available disk fell from about
3 GiB initially to about 1.95 GiB during host memory pressure; task-created
artifacts remained small. The initial 2 GiB disk reserve was therefore not
continuously maintained by the host. No unrelated files or processes were removed.

The final benchmark completed three actual 500x500 views after matched 64x64
acceptance. Results below use three recorded repeats after warmup, include thread
creation/coordination and full image delivery, and use the same /4 nonlinear
quadrature. They are desktop measurements with significant memory-pressure
variation, not isolated hardware performance guarantees.
Raw attribution: ignored `thermal-profile.txt`, `thermal-profile-setup.json`,
`thermal-reference.prof`. New build/test/benchmark artifacts stay under the same
`benchmark-results/analysis-core/` directory. The original thermal.json remains
historical evidence and must not be overwritten by this follow-up.


### Matched 64x64 Images

| View | Python reference median s | Native 1 worker s | Native 2 workers s | Native 4 workers s | Maximum image difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Axis z | 17.42714 | 0.40064 | 0.20783 | 0.12416 | 7.39e-13 |
| Oblique (0.3,0.2,1) | 15.37851 | 2.04094 | 0.23505 | 0.14313 | 9.38e-13 |
| Diagonal (1,-0.8,0.6) | 11.99355 | 0.27142 | 0.14246 | 0.16208 | 2.73e-12 |

All images/depths satisfy the frozen tolerance; the largest relative L2 difference
is 1.35e-15. Serial/parallel values, statuses and sample counts are exactly equal.
Native/reference sample totals differ by 0, -80 and +96 respectively from rounded
knot degeneracies; complete depths and image integrals agree. The reference wall
ranges were 17.14--28.37, 12.67--16.64 and 8.79--21.52 s. The native oblique serial
samples themselves ranged 0.371--2.098 s. Do not interpret the resulting median
ratios, especially an apparent >4x four-worker gain, as isolated parallel scaling.
Even accounting for those ranges, removal of the Python hot loop shows a clear
useful improvement in the matched consumer.

### Actual 500x500 Images

Each image launches 250,000 rays. Entries are **median [minimum, maximum] seconds**
of three repeats, not the earlier 64-ray linear extrapolation.

| View | 1 worker | 2 workers | 4 workers | Evaluated samples | Nonempty pixels |
| --- | ---: | ---: | ---: | ---: | ---: |
| Axis z | 23.00 [12.46, 81.04] | 8.40 [7.09, 9.15] | 18.27 [11.30, 21.16] | 209,354,760 | 250,000 |
| Oblique (0.3,0.2,1) | 29.41 [24.60, 30.88] | 15.59 [13.32, 15.60] | 9.22 [8.16, 12.32] | 237,965,104 | 236,958 |
| Diagonal (1,-0.8,0.6) | 17.86 [15.43, 21.71] | 12.09 [9.66, 13.71] | 6.41 [6.28, 6.47] | 240,795,928 | 179,818 |

All 500x500 images are complete, including explicit zero-valued empty rays, and
all one/two/four-worker images and counts agree exactly. A separate 500x500
Python-reference image was not run: smaller matched scientific checks establish
implementation conformance, while the actual large images establish scale and
parallel identity. These scopes remain distinct.

The axis single-worker range reaches 81.04 s; four workers lose to two in its
recorded median. Major-fault counts during warmup plus three repeats were
341/155/479 axis, 448/435/437 oblique and 520/247/0 diagonal for 1/2/4 workers.
CPU-time arrays and minor faults remain in JSON. This evidence supports the
compiled implementation and optional parallelism, but **does not select four
workers as universally optimal**, nor certify stable worker scaling on an idle
machine. Keep the default one worker and let the caller select two or four for
the actual workload/resource state. Another expensive desktop rerun is not needed
to claim the already demonstrated implementation improvement; an isolated scaling
study remains a concrete reopen condition.

### Setup, Alternative Reconstruction And Storage

This final run spent **19.72 s** from file to
prepared thermodynamics (10.43 s density;
9.28 s manufactured temperature and state).
That setup is shared by repeated views and is separate from the query timings.
It is slower than the earlier round under different desktop load. The physical
inputs are unchanged: actual WENO density with the declared manufactured
0.45--1.65 MK temperature and demonstration density/length scales. These are
not real-snapshot thermal validation.

The alternative prepared-node emissivity reconstruction remains useful and
explicitly different. This run's whole-field materialization took
**2.208 s** and retained **312,977,760 B**.
Four-worker 500x500 scalar LOS medians and relative L2 differences from the
nonlinear /4 images were:

| View | Retained-emissivity LOS s | Relative L2 difference |
| --- | ---: | ---: |
| Axis z | 0.818 | 0.2054% |
| Oblique (0.3,0.2,1) | 1.181 | 0.2535% |
| Diagonal (1,-0.8,0.6) | 0.977 | 0.3076% |

This is not a same-result speed comparison. Node emissivity has its own
reconstruction and scalar quadrature, and the differences are measured against
manufactured-temperature thermodynamics, not observations.

Thermal state occupies 625,593,696 B; initial coexistence admission remains
1,518,339,216 B. Each large LOS result has 12,000,024 B of direct arrays; origin,
near/far and conversion scratch are included in the wrapper's conservative 128
bytes/pixel allowance plus mesh, state and worker scratch. Shared fields are not
replicated per worker. This experiment also retains comparison results and three
2 MB image-value arrays. Peak observed RSS was
**858,161,152 B**; OS compression, swapping and page cache are not
controlled by that array budget. No large fixture or scratch cube was written.

### Disposition, Reproduction And Integration

**Adopt:** compiled AMR interval traversal, known-owner interpolation, fused
response/accumulation, precomputed table logarithms and optional GIL-free parallel
rays. Preserve Python reference, one-worker default and the scalar-emissivity
alternative. Reject malformed thermal component backing/slot directories before
unchecked sampling; initial geometry failure also yields NaN. The unchanged
interpolation body was verified directly after its relocation. All 23 analysis
checks passed, followed by the focused thermal checks after input hardening.

**Defer:** direct Embree/Open VKL dependency, GPU backend, automatic worker choice,
value-based skipping, adaptive response quadrature and additional reconstruction
changes. Reopen external acceleration if the remaining compiled traversal costs
justify its conversion and precision/semantics work. Reopen worker selection on
an adequately controlled host; do not silently turn this into main-session
numerical cache or general backend work.

```bash
.venv/bin/python scripts/build_ext.py --group analysis
PYTHONPATH=src:rewrite/src:scripts:tests/analysis:rewrite/benchmarks .venv/bin/python -m unittest discover -s tests/analysis -v
PYTHONPATH=src:rewrite/src:scripts .venv/bin/python -m analysis_core.profile_thermal
PYTHONPATH=src:rewrite/src:scripts .venv/bin/python -m analysis_core.benchmark_thermal_rays
PYTHONPATH=src:rewrite/src:scripts .venv/bin/python -m analysis_core.render_thermal_images
```

Measured implementation checkpoint: `9f423e9`; the subsequent final checkpoint
adds input/failure guards, evidence and the historical benchmark's explicit
reference selector without changing the measured kernel arithmetic. Raw results
are `thermal-rays.json` / `thermal-rays.log`. `thermal-500-images.npz` stores the
three quantitative arrays, units/model/normalization and manufactured-input label;
`thermal-500-projections.png` shows their explicitly labelled logarithmic color
mapping. No real-temperature, current-calibration or GPU result is claimed.

`native.pxd` now owns the unchanged inline interpolation definition; native.pyx
continues to own ray interval location. Both native and thermal extensions must
be rebuilt together. No build backend, OpenMP dispatch, slot cache or field
preparation provider was changed in this follow-up. The independent main-session
work remains unmerged and untouched.
