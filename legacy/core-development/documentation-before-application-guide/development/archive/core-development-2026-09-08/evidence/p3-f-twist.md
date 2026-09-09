# P3-F Accepted-Segment Twist And Trajectories

2026-09-08, following `6558286`. The selected diagnostic is twist; endpoint-map
Q and auxiliary-vector Q are not implemented or claimed.

## Delivered Scope And Independent Evidence

Twist integrates `curl(B).B/(4*pi*|B|^2)` in positive arclength over accepted RK4
steps. B remains a two-halo primary group; curl remains an independent one-halo
group. `with_curl` retains resident groups and `CurlPool` prepares/cache-binds
only requested derivative owners beside the primary pool. No global D pass is
required for bounded diagnostic tracing. Workers see both completed groups under
one primary lease; partial/rejected stages contribute neither twist nor points.

Optional trajectories contain accepted prefixes, with stable seed IDs, counts
and NaN tails. Summary-only remains the default. `retrace` is an explicit call
that selects original seed IDs, admits the prior result and numeric selection
scratch, and runs supplied parameters with point retention. It is not automatic
replay or a diagnostic reconstructed from saved points.

Nine composed core checks passed after the final build/integration. For a
helical affine field with radius 0.7--1.3, accepted length 3 and step 0.01,
twist max error against the analytic constant-density integral was 1.10534e-10
(predeclared bound 2e-10). Halving the step reduced it to 6.87106e-12, 16.1x
smaller. Independent scalar augmented RK4 conformance passed 2e-12. Sampled curl
was independently checked as (0,0,2); separate trajectory errors remain in P2.
One/four workers, resident groups and a four-slot bounded B/curl pool agreed
exactly. Selected retracing reproduced the corresponding saved trajectories.

The composition also exposed a retained-selector ownership issue: pools now
copy/freeze selected field IDs so caller mutation cannot relabel cached values.
Closed pools drop borrowed source references without closing caller file
descriptors. Companion views expire with the primary lease. Zero-step/zero-length
traces do not create unnecessary curl groups. These changes preserve ordinary
trace and P1/D behavior under the existing composed checks.

## WENO Comparison

Same verified 1,045,232,320-byte WENO file, ordinary B fields 4/5/6, mixed-ROI
32 SFC-spread centers, 128 steps of 0.0013020833333333333, positive direction.
All 4,096 requested steps were accepted. The first B/curl query prepared 160
primary and 160 derived owners; all later warm repetitions prepared zero owners.

| Warm complete trace | One worker | Four workers | Output bytes |
| --- | ---: | ---: | ---: |
| Twist summaries | 2.3742 ms | 1.0025 ms | 3,328 |
| Twist with trajectories | 2.3648 ms | 1.0130 ms | 102,400 |

One warmup/three repetitions, application warm, OS cache uncontrolled. The tiny
summary/trajectory timing difference is noise, not a claim that points are free.
The same-run ordinary summary control took 2.2210 ms; twist added about 6.9% to
this one-worker warm request. First twist computation took 0.173911 s, following
5.108947 s of original-file resident source/index/adapter setup. Source startup
still dominates first use and justifies considering direct bounded original
ordinary-field access during integration.

An independent scalar augmented-stage calculation for eight WENO seeds agreed
to 1.11022e-16 maximum absolute error across final positions and twist. This
checks the composition on real AMR fields; it does not establish physical truth
of the underlying snapshot. Twist range for the selected accepted segments was
[-0.0878014, 0.2801291]. Explicitly selected IDs 24, 30 and 10 retraced identically,
retaining 387 points in 9,288 trajectory bytes. No existing canonical/rewrite
twist implementation supplies an equivalent performance comparator; ordinary F
is an incremental-cost control only.

The modified ordinary 2048-seed x 600-step one-worker control measured 0.430611 s
versus P2's 0.416888 s (about 3.3%, below the material-regression threshold).
The ordinary path does not prepare derivatives or allocate trajectory payloads.

## Resources And Reproduction

The B/curl pool's conservative array bound was 316,727,274 bytes, including its
resident source/provider/mesh, both groups, derivative batch and RHE scratch.
The fixed 32x128 driver additionally holds index/fixture metadata, seed/reference
arrays, prior summaries, trajectory outputs and selection scratch; allowing a
further 16 MiB conservatively bounds those controlled arrays below 334 MB.
Observed process high-water RSS was 239,960,064 bytes. Logical live storage and
RSS are different measures; this remains a ~1 GB source, not target-scale proof.

```sh
.venv/bin/python build.py --inplace --group analysis
PYTHONPATH=src:rewrite/src:scripts:rewrite/benchmarks:tests/analysis .venv/bin/python -m unittest discover -s tests/analysis
PYTHONPATH=src:rewrite/src:scripts:rewrite/benchmarks:tests/analysis .venv/bin/python -m analysis_core.benchmark_twist
```

Raw results: `benchmark-results/analysis-core/twist.json` and core-test/build
logs. Retain the separate derivative companion and RK-stage quadrature. No
additional reconstruction or Q method is needed to deliver this named diagnostic.
Exact boundary footpoints and complete boundary-reaching twist remain unverified;
`localized_endpoint=False` continues to identify the accepted-prefix result.
Next: ready LOS geometry/scalar integration and P4 source/workflow/scale gates.
P3-L's requested physical response/EOS/units still require the pending user choice.
