# P4 Actual Seed And Output Scale

2026-09-08, `a5f1f88` source plus the scale-driver acceptance assertion.
Default non-OpenMP native build on macOS 26.5.2 arm64, 8 GiB RAM, one/four
Python workers around nogil kernels. The shared provider/public/helper suites
and the explicit OpenMP build check passed before restoring the default build.

```sh
PYTHONPATH=scripts .venv/bin/python -m analysis_core.benchmark_scale --section all
```

Raw record: `benchmark-results/analysis-core/scale-all.json`; log: `scale.log`.
This is one declared large validation run, not a stable multi-run timing SLA.

## Input And Preparation

The input is the actual `data/weno509_sub_0000.dat` (1,045,232,320 bytes), selecting
ordinary b1/b2/b3. No bridge is created. File/source preparation took 6.433 s
including 5.059 s of checked dense interior reading and 1.171 s of canonical
construction/copy/halo work. The retained prepared input bound was 1,259,384,080
bytes; its peak file/preparation workflow bound was 1,541,452,087 bytes.

## One Million Independent Seeds

A 1000x1000 uniform section is launched half a finest z cell above the lower
boundary. Step is 0.0013020833333333333, with a maximum of 32 steps and a 16,384
seed working batch. Points/length/steps/termination/sample counts/IDs are exactly
equal between one and four workers.

| Measure | Actual result |
| --- | ---: |
| Seeds | 1,000,000 |
| One-worker trace | 8.176259 s |
| Four-worker trace | 2.274146 s |
| Observed single-pair speedup | 3.60x |
| Accepted steps | 17,273,160 |
| B samples | 70,060,592 |
| Reached 32-step limit | 479,306 seeds |
| Domain-exit accepted prefixes | 520,694 seeds |
| Summary arrays | 96,000,000 bytes |
| Input seed array | 24,000,000 bytes |
| Simultaneous comparison controlled bound | 1,581,869,840 bytes |

No unexpected nonfinite, missing-coverage or unrepresentable-sample termination
occurred. The saved four-worker summary is `million-seeds.npz` in ignored results.
Many seeds exit quickly from the section; this is not one million 1000-step lines,
exact footpoints or a complete-boundary diagnostic proof. Endpoint semantics
remain the explicit last-accepted prefix contract.

## 1000 Cubed Uniform Output Stream

All 1,000,000,000 requested uniform cell-center positions were delivered with
true validity, in 1000 owned z slabs. The sink consumed each slab into a descriptive
checksum; it did not allocate/save a complete coordinate or value cube.

| Measure | Actual result |
| --- | ---: |
| Resolution | 1000 x 1000 x 1000 |
| Float64 value payload delivered cumulatively | 24,000,000,000 bytes |
| First slab | 0.087879 s |
| Complete stream | 81.283103 s |
| Maximum slab (values + validity + owners) | 33,000,000 bytes |
| Controlled bound including input, two slabs and tile scratch | 1,337,209,232 bytes |

The checksum is a descriptive ordered slab sum, not conservation or absolute
physical accuracy evidence. Smaller composed tests establish the sampling/
geometry contract against canonical uniform values; the large run establishes
actual coverage and bounded delivery at the declared output size.

Process high-water RSS across preparation, both seed runs, summary writing and
the uniform stream was 1,438,449,664 bytes. Logical live bounds and RSS remain
distinct. Task output/scratch stayed within the recorded disk allowance.

## What This Closes And What It Does Not

This is actual million-seed execution and larger-than-RAM **output** delivery.
The source is still ~1 GB and selected input fields fit resident preparation.
It does not certify 10--20 GB or larger-than-RAM **input** throughput, long-path
million-seed accuracy, Q, exact footpoints, new 2D/periodic native consumers or
thermal LOS physics. Those limitations are not hidden by the output/seed scale.
