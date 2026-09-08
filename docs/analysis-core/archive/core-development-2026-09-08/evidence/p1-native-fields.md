# P0/P1 Native Fields: First Executable Evidence

2026-09-08, branch `codex/analysis-core-p0-p4`, based on `7b0c1b1` with new
analysis files. Canonical/rewrite numerical implementations were not modified.
The selected design is [native-core-design](../native-core-design.md).

## Delivered Scope

Explicit validated rewrite I/O/forest/RHC adapters, independent component-adjacent
field groups, exact selected/source-slot mapping, two-layer preparation, stable
scoped pool borrowing, bounded native-file-to-memmap delivery, detached results,
direct compiled location/sampling and centered derived groups with reduced
validity. P0 is assembled; P1's selected native product and bounded delivery are
usable. Full-domain D/F/L acceptance, prepared-window-only optimization, compact
support/fused fill plans, 2D/periodic adapter integration and target-scale claims
remain separate work. The bootstrap's existing padded RHE staging is explicit.

## Commands And Core Results

```sh
.venv/bin/python build.py --inplace --group analysis
PYTHONPATH=src:rewrite/src:scripts:rewrite/benchmarks .venv/bin/python -m unittest discover -s tests/analysis -p 'test_prepared.py'
PYTHONPATH=src:rewrite/src:scripts .venv/bin/python -m analysis_core.benchmark_prepared
```

Five focused composed checks passed: permuted fields/leaves and mixed-AMR
sampling against independent scalar interpolation and affine fields; nextafter
ownership; whole-parent/box coverage; derivative-interface sampling consuming
the outer primary layer; pool eviction/pinning/failure publication and detached
lifetime; actual selective native file input to bounded memmap output. A provider
boundary dtype mismatch was caught and fixed before numerical checks. Nested
borrows are rejected so one coordinator lease has bounded directory storage;
parallel workers share that lease. Nonpacked borrowed slots require explicit
per-leaf windows rather than an implicit gather/copy through `interior()`.

## WENO Profile And Comparisons

Exact fixture: `data/weno509_sub_0000.dat`, 1,045,232,320 bytes, 22,614 leaves,
8^3 cells, original ordinary fields 4/5/6 (b1/b2/b3), levels 3--6. Resident
canonical ordinary-field bootstrap from staggered-tail records; no bounded
original-file or CT claim. Continuous boundaries, three fields, two valid layers.
One warmup plus five repetitions; application warm, OS cache uncontrolled.
Finite acceptance was fixed at rtol=atol=1e-10; nonfinite classes also compared.

| Selected region | Leaves | Preparation median | Max abs error vs canonical full prepared data | Product bytes |
| --- | ---: | ---: | ---: | ---: |
| mixed | 403 | 0.408292 s | 1.22125e-15 | 16,897,352 |
| physical | 8 | 0.005583 s | 0 | 512,752 |
| small | 32 | 0.033812 s | 8.88178e-16 | 1,508,272 |
| thin | 728 | 0.311708 s | 0 | 30,378,352 |

All compared padded values passed, including outer layers, refinement and
physical boundaries. Canonical resident full-domain refresh median was 0.272841 s
(with noticeable dispersion, stdev 0.074235 s). Selected preparation versus
full-domain refresh is **different work**, not a speedup ratio. Preparation is
still the bootstrap's major limitation; no new fill-kernel superiority is claimed.

On identical 512 mixed-box target centers, direct general-point native sampling
median was 0.036583 ms versus canonical regular-grid linear sampling 0.085333 ms
(2.33x ratio for these ready-data interfaces). Maximum sample discrepancy was
1.38778e-15. The interfaces perform different location/control work on the same
points; this is not a file-to-result speedup. Source read alone took 4.927583 s,
index/binding 0.023366 s, new adapter publication 0.017175 s, canonical geometry/
allocation/copy/first preparation 2.191875 s. Repeated-ready gains do not amortize
those costs for a single tiny query.

Mixed-group curl consumed extended inputs in 9.327 ms median, produced 9,856,136
bytes with one valid halo. This establishes the local consumption boundary;
it does not certify global D or physical current normalization.

## Attribution And Resources

The mixed first preparation used eight RHC chunks, 1,882 support loads,
23,126,016 logical input value bytes, and 11,120,754 provider managed-array bytes.
Packing into final component-adjacent storage took 4.425 ms inside a 433.979 ms
first RHC composition. Packing is about 1% here; changing axis order again will
not remove the preparation bottleneck. Thin support read volume was 48,807,936
bytes at capacity 256. This is a width-two full-leaf request, not the historical
width-one/ROI-window LFE request.

New immutable mesh arrays occupy 4,910,480 bytes; source/provider resident arrays
281,886,919 bytes. The simultaneous comparison conservatively admitted
1,762,044,928 controlled bytes including canonical padded/coarse C allocations,
raw source and selected output/transients. Observed process high-water RSS was
1,284,145,152 bytes. Managed admission is not an RSS or page-cache guarantee.
Raw timing samples, versions and build information live in ignored
`benchmark-results/analysis-core/prepared-weno.json`; no large artifact was added
to Git. Host: arm64 macOS, Python 3.11.14, NumPy 2.4.4, Cython 3.2.4, default
non-OpenMP build, eight logical CPUs, 8 GiB RAM.

## Decision And Next Work

Retain the assembled provider and direct native consumer boundary. It delivers
verified semantics and usable bounded products while preserving comparators.
E5 preparation work remains warranted: inspect/profile structural planning in
the actual next consumer rather than optimize the small publication copy.
Advance immediately to independent-seed F with explicit terminal/accumulator
semantics and parallel private state. Global D and later WENO full-domain
composition will own dense preparation/storage decisions; no stage completion
ends the authorized P0--P4 run.
