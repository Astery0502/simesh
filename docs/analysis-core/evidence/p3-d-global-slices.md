# P3-D Whole-Domain Curl And Retained Slices

2026-09-08, following `f6dc48e`. See the selected
[contract](../native-core-design.md#p3-d-global-derivative-and-slice-contract).

## Delivery And Core Evidence

`global_curl` computes every physical leaf plus one remaining derived halo into
independently owned or explicitly supplied array/memmap backing. Source and
bounded primary slots can be released afterward. `Plane`/`sample_plane` provide
axis-aligned and oblique pixel-center slices from the retained native result,
without recomputing global derivatives or materializing a uniform volume.
The result is curl(B), not a physically normalized current.

A canonical bulk resident preparation adapter shares its component-adjacent C
buffer directly with analysis consumers. An owner-bearing NumPy base keeps C
allocations alive even when a sliced view outlives the product descriptor.
Its extra C-owned coarse/geometry storage is explicitly admitted. Canonical
coordinate-phase versus bounded rewrite exact-phase is a named strategy choice;
existing canonical and rewrite entrypoints are unchanged.

Eight composed core tests passed after the final compiled loop change. Global
bounded results match independently centered differences of complete extended
inputs, including physical extension; native file/memmap and source lifetimes,
all-leaf coverage, one remaining halo, repeated axis/oblique slices and the
resident adapter are covered. Manufactured affine curl samples are accurate away
from incompatible continuous physical extension. No global second-order claim
is inferred at arbitrary refinement or physical boundaries.

## WENO Complete Domain

Fixture and fields are the P1 profile: 22,614 leaves, three B fields, two primary
layers, continuous physical boundaries. A 542,736,000-byte native output contains
all three curl components over 10^3 valid cells per leaf. Compared **all
67,842,000 values** between bounded exact-phase and resident canonical-coordinate
preparation: max absolute discrepancy 8.10019e-13, within the predeclared 1e-10
finite comparison tolerance. Actual canonical derivative components also agree,
with respective max errors 2.84217e-14, 5.68434e-14 and 2.84217e-14.

A 128x128 axis plane and 96x80 oblique plane both had full validity and agreed
between retained products. Repeated axis/oblique queries on the resident derived
product took 1.350 ms / 0.512 ms median. These calls do not perform field I/O,
ghost preparation or derivative calculation.

| Complete operation before loop optimization | Measured cost |
| --- | ---: |
| First bounded global curl to memmap, then two slices | 21.949 s |
| Bounded repeat, then two slices | 23.710 s |
| Canonical resident construction/copy/ghost preparation | 1.228 s |
| Resident derived allocation/calculation and first two slices | 0.810 s |

The bounded numbers are two descriptive full passes, not a tight timing gate.
Their output-copy times were 0.277 s and 2.137 s, showing page/storage variability.
Both include full coverage; the resident path is much faster when its padded
fields fit. File index/read startup is separate in the raw record and must be
included in fresh file-to-result totals. The original ~543 MB bounded native
result remains in ignored `benchmark-results/analysis-core/global-curl.npy`.

## Feedback, Optimization And Retained Choice

The initial generic derivative repeated term metadata/axis/spacing work inside
every cell. At 0.517 s, it was slower than the sum (0.256 s) of canonical's three
separate component kernels, although their output-retention work differs. The
bounded two-variant investigation moved term invariants outside spatial loops.
It preserves the exact division, multiplication and per-output addition tree;
no reciprocal approximation, reassociation or looser tolerance was introduced.
The old cell-major kernel remains available as an explicit comparison function.

| Same-runner operation | Cell-major | Term-major |
| --- | ---: | ---: |
| Mixed 403-leaf derivative kernel | 8.661 ms | 3.129 ms |
| Full-domain derivative kernel | 0.648 s | 0.208 s |
| Full derived allocation + curl + both slices | 0.849 s | 0.400 s |

Mixed outputs were bitwise equal and core conformance passed. Full runs showed
substantial timing dispersion (recorded in raw samples), so do not interpret the
ratios as precise universal speedups. Both representative sizes and the complete
consumer improved; retain term-major execution and close this probe. The
post-change complete-consumer measurement is actual, not a sum of nested stages.
The earlier bounded full-pass measurements retain their pre-optimization scope;
preparation still dominates them, and no extra full sweep was run merely to
refresh a small derivative fraction.

Use affordable canonical bulk resident preparation for dense D. Keep bounded
RHE for explicit memory constraints and native file-backed delivery. This is a
material measured memory/time trade-off, not a claim that one path dominates.
E5's invariant/compiled-execution principle is adopted where demonstrated;
a new persistent compact fill-plan backend is deferred until repeated cold
coverage or an unaffordable resident case justifies it. A ~1% packing cost in P1
does not justify another payload-layout change. E1/E2/E3/E4 remain scoped decisions
in the design, not mandatory implementations.

## Resources, Commands And Limits

The bounded controlled upper bound was 859,582,506 bytes including the entire
logical output, source/provider/mesh, pool, scratch and derived batch. Canonical
resident input retention was bounded at 1,259,384,080 bytes, plus 543,097,824 for
the independent derived group. Canonical component comparison admitted
2,127,492,431 bytes including its extra output and comparison buffers, below the
2 GiB run allowance. Observed high-water RSS was 1,854,570,496 bytes. Closed
memmap reference backing and accessed comparison tiles are reported separately;
logical mapped bytes are not a resident-memory guarantee.

```sh
PYTHONPATH=src:rewrite/src:scripts .venv/bin/python -m analysis_core.benchmark_global
PYTHONPATH=src:rewrite/src:scripts .venv/bin/python -m analysis_core.probe_derivative_order
PYTHONPATH=src:rewrite/src:scripts .venv/bin/python -m analysis_core.probe_derivative_order --composition-only
```

The global driver preserves an existing output path; choose an explicit new
output or remove only that owned result before repeating. Raw JSON/logs live in
`benchmark-results/analysis-core/{global,derivative-order,derivative-composition}.*`.
The same arm64/8 GiB/default non-OpenMP build as P1/P2 was used. Pipeline scope is
Cartesian 3D and the declared transfer/current-like definition. This is actual
whole-domain WENO evidence, not a 10--20 GB or 1000^3 uniform-output claim.
Next: P3-F accepted-segment twist and explicit trajectory/retrace delivery.
