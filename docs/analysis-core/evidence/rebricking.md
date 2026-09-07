# E1 Bounded Rebricking Feasibility: 2026-09-08

## Decision

**Defer default rebricking; retain the exact mapping/consumer prototype as evidence.**
There are abundant mergeable same-level regions, so the old uncertainty about
WENO fragmentation is resolved. Dense compact storage has material potential,
but sparse requests expand markedly, and this prototype still performs original
leaf ghost preparation before packing. The measured storage saving is not an
eliminated read/fill cost. Reopen dense opt-in adoption only with a direct brick
boundary preparation path and native mapped consumers that win or justify their
complete costs at the same numerical result. Preserve original leaf identity;
no new source format, execution backend or numerical cache is selected here.

Code: `scripts/analysis_core/rebricking.py`, `benchmark_organization.py`.
Core checks: `tests/analysis/test_rebricking.py`.
Raw evidence: ignored `benchmark-results/analysis-core/bricks.json`, `bricks.log`.
A first run exposed a JSON serializer issue after the sparse numerical checks;
that tooling issue was fixed and the entire recorded run then completed.

## Frozen Candidates And Actual Geometry

The complete actual `weno509_sub_0000.dat` geometry was scanned: 22,614 leaves,
levels 3--6, ordinary 8^3 cells. Candidates are the original leaves, aligned x
pairs (2x1x1 leaves), and aligned octets (2x2x2). Merge only when every original
same-level leaf in the bounding region is present. Other leaves stay individual.
No fine cells are averaged, no holes filled, no covered parents added. The summed
physical volume is exactly **11,578,368 cells** in every candidate. Mapping keeps
original leaf IDs and offsets separately from compute brick and storage offsets.

| Candidate | Compute bricks | Leaves in merged groups | Padded/interior | Geometric halo cells | Mapping bytes | Build seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native | 22,614 | 0 | 3.375000 | 27,498,624 | 1,809,136 | 0.2416 |
| x pairs | 11,683 | 21,862 (96.6746%) | 2.831205 | 21,202,368 | 1,371,896 | 0.1561 |
| Octets | 4,183 | 21,064 (93.1458%) | 2.050583 | 12,164,032 | 1,071,896 | 0.0871 |

Octets reduce whole-domain retained padded values by **39.24%**, and ghost values
by **55.77%**, from measured eligible geometry. These are exact representation
counts, not measured end-to-end speedups. The native mapping row is the prototype's
uniform mapping control; the existing native core does not need this extra map.

## Requests, Preparation And Consumption

Sparse F: eight SFC-spread seeds, fixed step one quarter of the minimum selected
spacing, 128 RK4 steps, accepted-prefix output. Existing native tracing discovered
**37 actually requested leaves**. Dense: all-cell curl on 768 central leaves plus
an 8x8 coherent central raster of width 1/8 of the domain, integrating its full
z depth. Union: **1,695 requested leaves**, **28,584 Gaussian sample points**.
This is a bounded dense request, not a whole-domain derivative benchmark.
Ray geometry took 0.05217 s and shared request arrays occupied 1,163,808 B.

Each candidate first prepares every original leaf belonging to a requested
brick with the unchanged two-halo RHE provider, then packs a disjoint partition
of interiors and exterior halos. Direct mapped sampling and derivative views
consume the compact storage without unpacking it. Thus native expanded padded
storage and compact storage coexist during conversion and both are reported.
Support-only original reads remain visible in provider statistics.

| Request / candidate | Expanded primaries | Expansion vs request | Support leaf loads | Prepare s | Pack s | Compact bytes / requested native bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sparse native | 37 | 1.000 | 435 | 0.02507 | 0.00060 | 1.000 |
| Sparse pairs | 61 | 1.649 | 559 | 0.06871 | 0.00091 | 1.288 |
| Sparse octets | 162 | 4.378 | 915 | 0.22805 | 0.00234 | 2.307 |
| Dense native | 1,695 | 1.000 | 6,074 | 2.13019 | 0.02810 | 1.000 |
| Dense pairs | 1,725 | 1.018 | 6,304 | 1.69816 | 0.03069 | 0.855 |
| Dense octets | 1,794 | 1.058 | 6,441 | 1.67442 | 0.02710 | 0.634 |

Original B reads transfer 12,288 bytes per support leaf load. Relative to useful
requested interiors, sparse read amplification is **11.757 / 15.108 / 24.730**;
dense is **3.583 / 3.719 / 3.800**. Preparation values still include each expanded
native 12^3 patch; original leaf fill has not been eliminated. Preparation times
are single descriptive measurements with uncontrolled OS cache and different
chunk/support order. The lower dense candidate times do not establish a speedup.
The common source open took 0.04165 s. The original bounded native sparse
file-to-trace comparison, including request discovery, took 0.18608 s.

Complete matched vectorized F consumption medians (one warmup, three repeats)
were **0.04335 / 0.04336 / 0.04317 s** for native-map/pairs/octets. Matched dense
curl-plus-LOS consumption was **0.03235 / 0.03218 / 0.03222 s**. Geometry/packing
therefore did not turn into a consumer throughput improvement in this prototype.
The native existing sampler in the same vectorized F harness took **0.00509 s**;
its dense counterpart took **0.02250 s**. This discrepancy is attributable to the
prototype's NumPy gather implementation versus the compiled native sampler, not
proof that merged-brick native code must be slower. The separate compiled native
trace, extended-halo curl and scalar LOS timings are in JSON; curl's extra retained
halo is explicitly different work from the prototype's interior-only result.

All compared two-halo windows agree exactly in the recorded WENO case. Mixed-AMR
core checks compare every leaf window, source-cell coverage, mapped samples and
curl. F positions/counts and dense derivative/LOS outputs also agree within the
predeclared 1e-10 finite tolerance. The compact organization therefore preserves
the demonstrated support/numerical meaning, but has not earned default adoption.
Peak run RSS was **495,599,616 B**. JSON retains disjoint packing/conversion storage,
source/mesh/plan storage and all provider scratch admission statistics. No source
fixtures or large value products were written.

## Reproduction And Integration Boundary

```bash
PYTHONPATH=src:rewrite/src:scripts .venv/bin/python -m analysis_core.benchmark_organization --stage bricks
PYTHONPATH=src:rewrite/src:scripts:tests/analysis:rewrite/benchmarks .venv/bin/python -m unittest discover -s tests/analysis -p test_rebricking.py -v
```

Future native consumers need `(source_leaf -> brick, local_cell_offset)` plus
brick shape/start tables and an explicit active-brick directory. Existing runtime
cache slots cannot be embedded in persistent geometry. Sparse source requests
must decide whether to expand to a whole brick or use the original leaf path.
The main session can evaluate that policy with its execution/cache work; this
round supplies measured coverage and cost evidence, not a mandatory migration.
