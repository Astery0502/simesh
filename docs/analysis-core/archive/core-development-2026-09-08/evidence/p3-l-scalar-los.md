# P3-L Scalar LOS Core And Remaining Physical Gate

2026-09-08, following `8a80a2f`. The geometric/scalar consumer is delivered;
P3-L's physical response/EOS/units acceptance is **not complete**.

## Delivered Geometry And Numerical Meaning

`integrate_los` returns pixel-owned line integrals, actual entry/exit/depth,
completion/status and sample/miss counts from a supplied prepared scalar.
`orthographic_plane` covers the projection of the entire physical domain.
Near/far may vary per pixel. Empty rays complete as zero; missing coverage,
nonfinite/unrepresentable values, geometry failure and sample limits produce
explicit invalid pixels. Ghost values supply reconstruction, never extra material.
Resident and bounded execution share immutable geometry and readonly preparation;
workers own disjoint ray continuations and ordered scalar sums.

The selected default is `gauss2`: split each leaf interval at native cell-center
planes and integrate the resulting cubic ray restriction with two Gaussian nodes.
This integrates the **declared trilinear scalar reconstruction** to roundoff; it
is not a physical truth claim about unresolved fields or nonlinear response.
Midpoint quadrature remains an explicit alternative with `step_fraction`.
No global uniform volume, unstructured dual mesh or materialized ray-cell action
list is needed. A nonlinear response evaluated after input interpolation has a
different meaning and is not silently provided by this scalar interface.

## Findings And Core Checks

Twelve combined native-core checks passed. Constant/affine manufactured fields,
positive/negative/oblique rays, variable per-pixel depth, physical empty rays,
mixed-level coverage, serial/parallel identity, bounded misses and incomplete
outputs are covered. An independent oracle intersects every leaf, sorts its
cell-center knots and uses scalar tensor interpolation; it shares no compiled
locator/marcher/sampler. On the manufactured multiaffine case, midpoint L2 error
against the reconstructed-field integral decreased from 0.005236 to 0.000630 when
its fraction changed .5 to .125. Gauss2 matched the independent reference within
the predeclared 1e-10 tolerance; serial and bounded schedules matched exactly.

The broader oblique view caught a real geometry issue: a coordinate probe could
cross the physical boundary while 1--2 ulps of positive ray parameter remained.
The implementation now locates the interval through ray/face intersection times
and directional ties, preserving that remaining length. No tolerance was widened
and no interval was dropped to make the case pass. Shared interpolation also
checks actual local stencil addressing before unsafe integer conversion/access;
unrepresentable samples have explicit F/LOS status.

## WENO Scalar Columns And Strategy Choice

The actual file has seven ordinary fields: rho, m1/m2/m3 and b1/b2/b3. It has no
energy or temperature field. This assessment integrates **stored rho in code
units** as a column diagnostic; it does not select or verify thermal emission.
Two 32x32 full-box orthographic views use directions (0,0,1) and (.3,.2,1).

| Resident single-worker strategy | Axis time | Oblique time | Maximum quadrature discrepancy from Gauss2, axis / oblique |
| --- | ---: | ---: | --- |
| Midpoint, fraction .5 | 4.907 ms | 7.231 ms | 5.760e-5 / 1.124e-4 |
| Midpoint, fraction .125 | 9.493 ms | 11.314 ms | 3.940e-6 / 4.076e-6 |
| Gauss2 | 5.621 ms | 8.696 ms | 0 / 0 |

The independent Gaussian oracle on eight spread nonempty pixels per view agrees
with max errors 8.882e-16 / 4.441e-16. Four-worker Gauss2 medians were 4.387 ms /
3.285 ms, with identical pixel values. Gauss2 uses 214,272 / 243,592 samples,
versus 767,135 / 623,843 for fraction-.125 midpoint. Retain Gauss2: it integrates
the fixed reconstruction accurately and costs less than the denser midpoint
control. It is somewhat slower than coarse midpoint at a different accuracy;
these are explicitly different quadrature strategies, not equal-work regressions.
No further quadrature variants are needed for this scope.

Matching 512-slot bounded Gauss2 passed all 1,024 pixels against the resident
provider within 2.220e-16 for both views. Its one/four-worker first passes took
8.090/8.195 s (axis) and 9.284/9.197 s (oblique), preparing 8,657/9,876 owners.
Preparation dominates; threading cannot remove RHE planning/missing-data costs.
Keep the resident path when affordable and the bounded path for memory limits.
The source's resident scalar bootstrap took 4.194 s and canonical bulk preparation
0.282 s. Those costs belong in first file-to-image latency.

## Resources And Artifacts

The bounded pool bound was 114,114,890 bytes, including the resident raw scalar,
metadata, prepared slots and RHE scratch. The simultaneous comparison additionally
held canonical padded/coarse backing, about 49 KB per retained image, other index
metadata and tile/worker arrays; it remained within the run's 2 GiB envelope.
The canonical preparation bound including raw input was 541,525,264 bytes.
Observed process high-water RSS was 579,256,320 bytes. No claim of bounded
original-file access follows from the resident input bootstrap.

```sh
.venv/bin/python build.py --inplace --group analysis
PYTHONPATH=src:rewrite/src:scripts:rewrite/benchmarks:tests/analysis .venv/bin/python -m unittest discover -s tests/analysis
PYTHONPATH=src:rewrite/src:scripts .venv/bin/python -m analysis_core.benchmark_los
```

Raw records: `benchmark-results/analysis-core/los.json`; scalar images:
`los-columns.npz` and `los-columns.png`. All are ignored outputs. Same arm64,
8 GiB, default non-OpenMP build as previous stages; one warmup/three resident
repetitions, descriptive matching bounded passes, uncontrolled OS page cache.

## Remaining Gate And Next Work

A real epsilon(rho,T), thermodynamic inputs, normalization/units and any instrument
response remain unspecified. The user question is pending; the WENO file also
lacks the energy/temperature needed to derive T. Do not mark physical P3-L complete
from column-density or constant-response evidence. Continue independent P4
native source/packaging/compatibility and feasible scale checks. Source inventory
found only this ~1 GB WENO fixture and the ~1.5 MB tdm fixture; no 10--20 GB or real
2D specimen is available in `data/`.
