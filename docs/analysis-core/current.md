# Analysis-Core Checkpoint

Updated: 2026-09-08. Resume through this file and [development](development.md).

## Active Scope

The user activated autonomous **P0 through P4**, including P3-D, P3-L and P3-F.
P2 is an intermediate checkpoint. Earlier documentation-only and first-run-P2
recommendations are superseded. Continue the next ready work after checkpoints.

- Checkout: `/Users/astery/science/simesh`, branch `codex/analysis-core-p0-p4`,
  branched from `7b0c1b1e17e2a16311e2c5e2cfbe22d908c98283` with existing
  uncommitted documents preserved. Do not stage unrelated entry changes.
- Active delivery: P4 native source, installed workflow and feasible scale gates.
  P3-L scalar LOS is delivered; its physical response gate remains open. P0/P1, P2, P3-D and selected P3-F twist are delivered;
  see [P1](evidence/p1-native-fields.md), [P2](evidence/p2-field-lines.md),
  [P3-D](evidence/p3-d-global-slices.md), [P3-F](evidence/p3-f-twist.md) and
  [scalar LOS](evidence/p3-l-scalar-los.md).
- Selected design: [native-core-design.md](native-core-design.md).
- Executable location: additive experimental `src/simesh/analysis/`, compiled
  `src/simesh/utils/lib/analysis/`, development adapters `scripts/analysis_core/`.
  Canonical/rewrite implementations remain available.
- New executable checks: twelve core/F/D/twist/LOS composed cases pass. Full WENO
  curl coverage (67,842,000 retained values) and axis/oblique slices compare
  successfully. The term-major derivative preserves operation order and improves
  the measured complete resident consumer. Keep fast resident canonical and
  memory-bounded exact-phase preparation as explicit alternatives. Exact
  boundary endpoints, physical LOS response and target scale remain open.
  Twist/trajectory/retrace conformance and WENO costs are recorded in P3-F.
  Gauss2 scalar LOS is selected after independent references and WENO columns;
  ray-time ownership fixes an oblique boundary-rounding failure.

## Run Profile

The endpoint bounds the run; no user time/token limit was supplied. Host has
8 GiB RAM and 8 logical CPUs. Entry free disk was about 7.6 GiB. Initially use
at most 4 compute workers, 2 GiB controlled live arrays per measurement and
2 GiB task-created scratch/output; keep at least 2 GiB disk free. Do not delete
source fixtures or unrelated data. Count resident inputs, providers, copies,
scratch and outputs; managed arrays do not bound RSS or OS page cache.
One discretionary probe at a time, at most two variants/15 minutes each before
retaining or deferring; required debugging is separate.

Real specimen: `data/weno509_sub_0000.dat`, 1,045,232,320 bytes, 22,614 leaves,
levels 3--6, 8^3 cells. This is not a 10--20 GB or million-seed certificate.
Larger-than-RAM fixture acceptance is resource-unverified; continue feasible
integration/scaling without relabeling smaller evidence.

## Scientific Choices And Open Items

Initial preparation preserves rewrite RST/LIM/PRL exact-phase semantics through
its RHE provider: float64 cell averages, all-26 closure, two valid layers,
continuous physical boundaries. Canonical comparison uses its recorded numerical
tolerance, not presumed bitwise equality. New consumer choices are recorded
before implementation. D exposes curl(B) without inventing physical current
normalization; F starts with length and explicit terminal status.

P3-L response/EOS/units are unspecified. An asynchronous question asks the user
to specify them or delegate the first verifiable model. This does not block
P0/P1/F/D or LOS geometry work.

## Next Action

The original-record ordinary reader, installed source/resident entrypoints and
interior-only products are implemented. Reader conformance passes (21 cases),
and file lifetime/selection, 2D/periodic-metadata/VTK and uniform-stream checks pass.
Direct WENO short tracing is 0.129 s versus 5.118 s for the eager-source comparator,
with equal results. Zero-ghost facts reduce CPU cost; one-record private dense I/O
reduces B read wall time 7.495 -> 5.012 s without weakening callback preflight.

A clean-source wheel (~9.35 MB) built and passed isolated non-editable import and
file-to-result checks. The first harness copy included stale egg-info; excluding
that generated metadata fixed it. Build cleanup now preserves .venv/references/
results and make clean is clean-only. Next run the supported Make test/build,
affected provider/public/helper checks, OpenMP compatibility, and declared actual
million-seed/1000^3 stream cases. Refresh P4 evidence and final scope dispositions.


P3-L response/EOS/units delegation is still pending; WENO contains rho/m/B but
no energy/temperature. Only WENO (~1 GB) and tdm (~1.5 MB) exist in data/; actual
10--20 GB, real 2D and full physical thermal-image acceptance remain unavailable.
New 2D/periodic analysis semantics must be explicitly scoped against retained
canonical workflows. Preserve the ~543 MB owned global-curl result unless its
scratch space is needed; smaller LOS arrays/figure are also in ignored results.
