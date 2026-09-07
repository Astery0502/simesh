# Analysis-Core Checkpoint

Updated: 2026-09-08. Resume through this file and [development](development.md).

## Active Scope

The user activated autonomous **P0 through P4**, including P3-D, P3-L and P3-F.
P2 is an intermediate checkpoint. Earlier documentation-only and first-run-P2
recommendations are superseded. Continue the next ready work after checkpoints.

- Checkout: `/Users/astery/science/simesh`, branch `codex/analysis-core-p0-p4`,
  branched from `7b0c1b1e17e2a16311e2c5e2cfbe22d908c98283` with existing
  uncommitted documents preserved. Do not stage unrelated entry changes.
- Active delivery: P3-D whole-domain curl(B) and retained slices. P0/P1 and
  the P2 accepted-prefix parallel tracer are delivered; see [P1](evidence/p1-native-fields.md)
  and [P2](evidence/p2-field-lines.md).
- Selected design: [native-core-design.md](native-core-design.md).
- Executable location: additive experimental `src/simesh/analysis/`, compiled
  `src/simesh/utils/lib/analysis/`, development adapters `scripts/analysis_core/`.
  Canonical/rewrite implementations remain available.
- New executable checks: seven core/F composed cases pass. WENO selected
  two-layer values and short/long F comparisons pass. A 2048-seed parallel
  case gives 3.69x speedup; WENO long-cache probe retains 256 slots for that
  repeated request (zero warm fills). Exact boundary endpoints, global D/L,
  twist/trajectory delivery and target scale remain open.

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

Implement P3-D global curl(B) with independently owned/optionally backed derived
fields and axis/oblique sampled slices. Compare the bounded exact-phase RHE
provider with an explicitly named canonical bulk resident preparation adapter;
keep transfer arithmetic distinctions visible. Preserve complete global coverage
and include retained output in admission. Then advance P3-F and ready LOS work.
P3-L physical response delegation is still pending. Direct original-record bounded
input, installed provider/2D/periodic integration and actual large fixtures remain
later gates; no automatic cutover of the canonical API is planned.
