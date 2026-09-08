# P2 Independent-Seed Tracing

2026-09-08, following `ee96824`. Strategy and limitations are fixed in
[native-core-design](../native-core-design.md#p2-numerical-and-execution-contract).

## Delivered And Verified

Compiled arclength RK4, last-accepted positions, stable seed IDs, accepted length
and step counts, explicit terminal reasons, summary-only bounded delivery, private
persistent RK state on preparation misses and one-to-four independent CPU workers.
The default is one worker. Threaded kernels release the GIL; this is actual
single-node parallel execution without an OpenMP build dependency.

Seven composed native-core checks passed after the final kernel build. A helical
manufactured affine vector field crosses native AMR interfaces; at length 3,
step 0.02 endpoint max error was 1.03140e-9, decreasing to 6.55822e-11 at step
0.01 (15.7x). The same-step independent scalar RK4 reference passed 2e-12 finite
tolerance. One/four workers and a four-slot bounded pool produced identical
positions/steps/status; cache misses resume private stages without replaying
accepted steps. Constant/boundary launches, direction/length caps, outside seeds,
missing coverage, null/nonfinite fields and unrepresentable norms are explicit.
These results do not assert fourth-order convergence for arbitrary AMR fields.

## Measured Comparisons

```sh
.venv/bin/python build.py --inplace --group analysis
PYTHONPATH=src:rewrite/src:scripts:rewrite/benchmarks:tests/analysis .venv/bin/python -m unittest discover -s tests/analysis
PYTHONPATH=src:rewrite/src:scripts:rewrite/benchmarks:tests/analysis .venv/bin/python -m analysis_core.benchmark_traces
PYTHONPATH=src:rewrite/src:scripts .venv/bin/python -m analysis_core.probe_long_cache
```

Same 8 GiB arm64 host and default compiled build as P1. One warmup/three timed
repetitions; raw records are `benchmark-results/analysis-core/traces.json` and
`long-cache.json`. Application warm/cleared is explicit; OS cache uncontrolled.

| Request/state | One worker | Four workers | Interpretation |
| --- | ---: | ---: | --- |
| Manufactured, 2048 seeds x 600 steps, prepared fields | 0.416888 s | 0.112973 s | 3.69x parallel speedup, 92.3% efficiency |
| WENO 8 seeds x 8 steps, 16 slots, cleared | 11.822 ms | 11.851 ms | Preparation dominates; no useful thread gain |
| Same, warm | 0.1503 ms | 0.3829 ms | Thread startup is a material small-job overhead; default remains one |
| Same, 4 slots, warm | 12.154 ms | 12.595 ms | All eight owners reload; insufficient retention |
| WENO 32 seeds x 128 steps, 64 slots, cleared | 178.435 ms | 170.363 ms | Repeated preparation dominates |
| Same, warm | 136.540 ms | 134.754 ms | Still thrashing; threads cannot remove preparation costs |

WENO seeds are SFC-spread mixed-ROI centers, positive direction; step is one
quarter of the minimum selected spacing (long case 0.0013020833333333333).
All seeds reached the requested step count. All comparable new schedules had
identical positions/length/steps/status/IDs. Ordinary B bootstrap took 4.850410 s,
index 0.023371 s and adapter publication 0.016521 s. Fresh file-to-result must
include these costs; warm milliseconds are not an end-to-end file speedup.

Rewrite SLE's short-case last positions were equal with max error zero and
accepted counts agreed. Its first call took 24.642 ms and warm median 3.771 ms.
This is **different work**: CHS prepares one halo and SLE retains trajectories
and B-dot-dx integrals; the new path prepares two layers and returns arclength
summaries. No equal-work overall speedup is inferred. Canonical has no matching
tracer contract. Numerical semantics of old FLN/RKS/SLE entrypoints are unchanged.

## Bounded Cache Probe And Decision

The predeclared 64-versus-256-slot probe completed with identical trajectories:

| Pool | Cold time | Warm median | Cold / warm prepared owners | Controlled array bound |
| --- | ---: | ---: | --- | ---: |
| 64 slots | 180.023 ms | 133.113 ms | 163 / 124--130 | 296,456,682 bytes |
| 256 slots | 166.843 ms | 2.2778 ms | 160 / 0 | 304,433,130 bytes |

Retain 256 slots for repeated execution of this named long request: 7,976,448
extra controlled bytes avoid all warm preparation and improve warm time 58.4x.
Do not adopt 256 as a universal default. Cold preparation still costs about
0.167 s; the cache does not fix RHE planning for new coverage. The bounded probe
is closed, with no further variants needed. E5 fused/retained fill plans remain
unimplemented and can be reconsidered for dense D or repeatedly missing work.

The manufactured result used 196,608 output bytes and at most 1,048,576 admitted
private stage/query bytes, plus 2,657,664 prepared and 46,512 geometry bytes.
WENO pool bounds include resident source/provider/mesh, prepared values, RHE
scratch and directory/selector allowance; add actual seeds/IDs, bounded private
state and output under the trace admission. RSS is separately recorded (observed
process high-water 182,960,128 bytes), not substituted for logical live storage.

## Limits And Next Delivery

Endpoints are last accepted points, `localized_endpoint=False`. Domain-exiting
trial steps are rejected; exact boundary footpoints and complete boundary-reaching
lengths are not delivered. Optional saved trajectories, selected-seed retracing
and twist are P3-F work. Million-seed and 10--20 GB input acceptance remain open.
Proceed to P3-D global derivatives and retained slices, with resident/bounded
comparison and an explicit canonical bulk preparation alternative.
