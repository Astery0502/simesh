# Analysis-Core Checkpoint

Updated: 2026-09-08. Resume through this file and [development](development.md).

## Active Follow-up: Thermal Ray Traversal And Parallelism

The latest user instruction explicitly authorizes online ray-tracing research,
profiling, bounded optimization and tests, including parallel thermal LOS. This
supersedes the previous no-new-500x500-test request for this follow-up. Continue
in the same clean isolated worktree, based on `a61a0ac`, preserving the completed
round below. Scope is the resident thermal ray consumer; the main session still
owns broader backend, preparation and numerical cache work. Do not merge its
concurrent changes or duplicate its OpenMP/runtime policy work.

Endpoint: source-backed algorithm/library disposition, actual hotspot evidence,
a numerically conforming accelerated thermal consumer with 1/2/4 worker results,
and measured representative image scaling. Keep the Python implementation as
an explicit scientific/reference comparator. Limits remain four workers, 2 GiB
controlled arrays, 2 GiB scratch and at least 2 GiB free disk. One active probe:
existing all-leaf/Python path versus reused AMR-tree compiled ray traversal,
then identical row work shared across existing-style GIL-free thread calls.
Allowance: 30 minutes of comparative execution, no new external dependency or
hardware unless current evidence makes it necessary. Do not infer instrument
accuracy or generic library superiority from this implementation comparison.

Status: compiled tree traversal and shared inline interpolation implemented;
all 23 analysis checks passed after inline relocation. The first C-API
candidate's partial 64x64 comparisons are preserved in `capi-probe/`. Generated-C
inspection established avoidable per-point descriptor passing; shared inline
arithmetic is unchanged. Final timing is paused until main-session process
`probe_paired_backends.py` exits, using a kernel process-exit notification rather
than repeated polling. Do not interrupt or edit that independent work.

Next: verify the completed check log, then run benchmark_thermal_rays when the
main benchmark has exited; record 64x64 parity and true 500x500 1/2/4-worker timings,
render the labelled manufactured-temperature images, close evidence and commit.
Later checkpoints below remain historical until updated.

## Completed Round: Thermal Response, Organization And Geometry Plans

This explicitly authorized round is **complete in its declared feasible scope**.
The user delegated the first response/EOS/unit choices and routine implementation;
missing WENO temperature did not block manufactured/isothermal verification or
the organization/plan work. No renewed authorization is pending. Actual snapshot
thermal truth and current-instrument calibration remain explicitly unverified.
The earlier whole P0--P4 target-scale input gate is separate from this round.

| Direction | Delivered / measured disposition | Evidence |
| --- | --- | --- |
| AIA171 optically thin response | Adopt pinned historical table, explicit H/He EOS, electron vs upstream hydrogen-proxy normalization, external K/isothermal input and both LOS reconstruction orders. Thermodynamics-first with four subdivisions is the default; refine for the scientific request. | [Thermal](evidence/p3-l-thermal.md) |
| E1 rebricking | Bounded whole-WENO geometry plus mapped F/D/LOS prototype completed. Defer default adoption: octets merge 93.15% of leaves and save 39.24% total padding, but sparse primaries expand 4.38x. Retain exact mapping/probe outside production compute identity. | [Rebricking](evidence/rebricking.md) |
| E5 geometric plans | Adopt explicit optional selected fill plans. Exact prepared/consumer output and equal reads; repeated composed costs improve 2.56x sparse and 6.67x dense. Plan construction amortizes after two executions in this profile. | [Plans](evidence/geometry-plans.md) |

## Checkout And Recovery

- Isolated worktree: `/Users/astery/science/simesh-euv-geometry`.
- Branch: `codex/analysis-core-euv-geometry`; original base `bdfda30`.
- Round delivery checkpoint: `ab75a17` (implementation, tests, bounded comparisons
  and evidence); this record has a subsequent recovery-note commit.
- Inherited dirty/untracked documentation was preserved separately in `d1aaf66`.
  The original `/Users/astery/science/simesh` checkout was not edited by this round.
  Do not merge the inherited snapshot blindly over concurrently maintained docs;
  the subsequent round delivery is a separate checkpoint for review/integration.
- Ignored `.venv` and `data` symlinks borrow the existing runtime/fixtures. No pip
  installation, shared-environment mutation or source-fixture rewrite occurred.
  Existing compiled extensions were copied into this worktree; no Cython source
  or executor/scheduler was changed.
- Code: `src/simesh/analysis/thermal.py`, `_aia171_table.py`, `geometry_plans.py`;
  one optional plan-builder hook in fields/providers. The E1 code is evidence
  tooling under `scripts/analysis_core/rebricking.py`.
- Use [usage](usage.md) for concrete thermal/external-temperature and plan calls.
  The existing [selected design](native-core-design.md) owns all decisions.

## Verification And Resources

The complete analysis suite passed **22 tests**, including existing field/source,
F/D/scalar LOS/twist/uniform checks and the new thermal/geometry-plan/rebrick cases:

```bash
PYTHONPATH=src:rewrite/src:scripts:tests/analysis:rewrite/benchmarks .venv/bin/python -m unittest discover -s tests/analysis -v
PYTHONPATH=src:rewrite/src:scripts .venv/bin/python -m analysis_core.benchmark_thermal
PYTHONPATH=src:rewrite/src:scripts .venv/bin/python -m analysis_core.benchmark_organization --stage bricks
PYTHONPATH=src:rewrite/src:scripts .venv/bin/python -m analysis_core.benchmark_organization --stage plans
```

Raw JSON/logs are retained in ignored `benchmark-results/analysis-core/` (under
1 MiB before any later user output). No large numerical scratch was created.
Host: inherited 8 GiB RAM/eight logical CPUs; up to four existing workers allowed,
new probes used one worker. Bounds: 2 GiB controlled live arrays/case, 2 GiB new
scratch, at least 2 GiB free disk. Measured peaks: thermal RSS 997.9 MB, E1 495.6 MB,
E5 527.5 MB; thermal coexistence admission 1.518 GB. OS cache is uncontrolled.
Discretionary variants are closed; no time/token budget was supplied.

The table/source and full input/output unit derivation are recorded in thermal
evidence. Its manufacturing/reference errors are distinct from WENO /64 quadrature
comparisons. The first model's independent continuum maximum error reached
7.24e-7 at subdivision 16, versus 1.39e-3 for prepared-node emissivity on that test.
WENO manufactured-temperature /4 relative L2 differences from /64 are 3.19e-5
axis and 1.79e-5 oblique. None is a claimed real-snapshot physical accuracy.

## Main-Session Handoff And Remaining Gaps

The main session owns execution backends, parallel dispatch and runtime ghost
numerical caches. This work supplies optional immutable geometry plans with local
support ordinals, source-to-brick maps, and explicit thermal field/result meaning.
Do not put cache slots, value generations or minmod slopes into persistent plans.
Field values/count/order can rebind on the same immutable mesh; geometry, source
cell ordering, selection, partition, boundary or transfer-rule changes rebuild.

No further ready work is required for this round. The next integration action is
to review delivery commit `ab75a17` and use these interfaces in the main
session without automatically replacing native leaf organization or cache policy.
Reopen E1 only with a direct brick-boundary/native-consumer cost case; reopen E5
full-domain retention/fusion only when its measured workload and memory justify it.

Still unverified: actual WENO thermal state and physical normalization, the pinned
table's missing CHIANTI/abundance/calibration generation metadata, current-date
instrument effects, nonlinear parallel/bounded thermal execution, whole-domain
persistent plan feasibility, production direct-brick preparation, and the earlier
10--20 GB/larger-than-RAM Cartesian input acceptance. Missing inputs do not undo
the completed manufactured/actual-geometry evidence. Canonical public paths and
previous P0--P4 source checkpoints remain preserved.
