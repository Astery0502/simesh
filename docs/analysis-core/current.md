# Analysis-Core Checkpoint

Updated: 2026-09-08. Resume through this file and [development](development.md).

## Active Scope

The user activated autonomous **P0 through P4**, including P3-D, P3-L and P3-F.
P2 is an intermediate checkpoint. Earlier documentation-only and first-run-P2
recommendations are superseded. Continue the next ready work after checkpoints.

- Checkout: `/Users/astery/science/simesh`, branch `codex/analysis-core-p0-p4`,
  branched from `7b0c1b1e17e2a16311e2c5e2cfbe22d908c98283` with existing
  uncommitted documents preserved. Do not stage unrelated entry changes.
- Active delivery: P2 independent-seed F. P0 provider assembly and the P1
  selected native/bounded product are delivered; see [evidence](evidence/p1-native-fields.md).
- Selected design: [native-core-design.md](native-core-design.md).
- Executable location: additive experimental `src/simesh/analysis/`, compiled
  `src/simesh/utils/lib/analysis/`, development adapters `scripts/analysis_core/`.
  Canonical/rewrite implementations remain available.
- New executable checks: five P1 composed cases pass. Selected WENO two-layer
  comparisons pass; direct ready sampling is measured. Dense/global, parallel
  F/D/L and scale acceptance remain open.

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

Implement P2 with explicit fixed-RK4 tangent/length/terminal semantics and
independent worker state. Reuse the verified native groups and pool miss boundary;
compare serial/parallel and fitting/thrashing residency in actual tracing.
Continue to ready P3-D/L/F and P4 integration. Retained RHE preparation overhead,
2D/periodic/direct original-record bounded adapters and actual large-fixture
acceptance remain tracked; they do not block initial F or D development.
