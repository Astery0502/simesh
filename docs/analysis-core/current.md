# Analysis-Core Checkpoint

Updated: 2026-09-08. Resume through this file and [development](development.md).

## Authorized Scope And Actual State

The user explicitly authorized autonomous **P0 through P4**, including P3-D,
P3-L and P3-F; P2 was an intermediate checkpoint. All currently ready work has
advanced through implementation, comparisons, integration and feasible scale.
**The whole scope is not marked complete:** physical P3-L and actual large-input
acceptance still require missing inputs. Do not restart completed stages or
reinterpret the old P2 recommendation as the endpoint.

Execution is **blocked on external inputs** after repeated revalidation: the
physical response/thermodynamic inputs and a suitable large Cartesian fixture
remain unavailable. Resume the corresponding gate when either input is supplied.

| Delivery | Actual status and evidence |
| --- | --- |
| P0/P1 | Complete selected design, provider assembly, native interior/two-halo products, stable bounded borrowing and direct consumption; [P1](evidence/p1-native-fields.md) |
| P2 | Complete declared accepted-prefix parallel F, terminal/length results and cache comparisons; [P2](evidence/p2-field-lines.md) |
| P3-D | Complete whole-domain curl(B), independent retained groups and axis/oblique slices; [D](evidence/p3-d-global-slices.md) |
| P3-F | Complete selected accepted-segment twist, optional trajectories and explicit selected-ID retracing; [F](evidence/p3-f-twist.md) |
| P3-L | Scalar/emissivity-field integration core and WENO rho columns complete; **physical response/EOS/units gate open**; [L](evidence/p3-l-scalar-los.md) |
| P4 | Additive installed source/workflow/build compatibility and actual million-seed/1000^3 output checks complete; **10--20 GB / larger-than-RAM input unverified**; [integration](evidence/p4-integration.md), [scale](evidence/p4-scale.md) |

## Checkout And Recoverable Work

- Checkout: `/Users/astery/science/simesh`, branch `codex/analysis-core-p0-p4`.
- Source checkpoints: `ee96824` (P1), `f6dc48e` (P2), `6558286` (D),
  `8a80a2f` (twist), `2c00792` (scalar LOS), `a5f1f88` (file/packaging integration).
  Final acceptance/navigation has its own subsequent checkpoint.
- Substantial pre-existing modified/untracked documentation remains preserved.
  Stage only owned work; no blanket reset, staging or source-fixture cleanup.
- Core: `src/simesh/analysis/`; compiled consumers: `src/simesh/utils/lib/analysis/`;
  file adapters: `src/simesh/amrvac/analysis*.py`. Existing rewrite is an explicit
  bundled provider with its original numerical directives and contracts.
- Design/route decisions: [native-core-design](native-core-design.md).
  Runnable examples and restrictions: [usage](usage.md).

## Latest Verification And Resource Profile

- Safe `make clean` and complete `make test PYTHON=.venv/bin/python` passed;
  virtualenv NumPy remained intact. Provider suite: 1232 passed. Public/AMR/helper
  checks: 92 passed, two opt-in heavy cases excluded. New-core composed checks:
  17 passed. OpenMP build plus 23 AMR checks passed; default non-OpenMP restored.
- Fresh-source wheel (~9.35 MB) built and passed isolated `python -S` imports and
  file-to-result execution without editable checkout hooks.
- Actual one million seeds (maximum 32 steps): one/four workers agree; trace
  8.176 / 2.274 s; 17,273,160 accepted steps; last-accepted endpoint semantics.
- Actual 1000^3 output stream: every billion point valid, 24 GB cumulative values,
  81.283 s, maximum 33 MB slab. Peak run RSS 1,438,449,664 bytes.
- Direct original-file WENO short trace: 0.129 s and ~3 MB reads versus 5.118 s
  eager-source comparison, identical results. Dense checked B reading improved
  to ~5.01 s through record-order I/O, retaining full callback checks.
- Host: 8 GiB RAM / eight logical CPUs. Run limits: <=4 compute workers,
  <=2 GiB controlled live arrays per case, <=2 GiB task-created disk scratch,
  >=2 GiB free disk. No time/token limit was supplied. Bounds do not promise
  process RSS or OS page-cache limits. Candidate probes are closed with outcomes.

## Inputs Still Required

1. **P3-L physical definition and data:** an asynchronous question remains pending
   for epsilon(rho,T), EOS/normalization/units/instrument response or explicit
   delegation of a first model. WENO and tdm have no energy/temperature. Do not
   relabel stored-rho columns as a thermal image. Additional temperature/physical
   inputs must match the selected response before its independent verification.
2. **Large Cartesian input fixture:** an asynchronous question requests a readable
   path to an existing 10--20 GB file, including an external drive if available.
   Repository inventory found WENO (~1 GB), tdm (~1.5 MB) and spherical bw
   (~3.6 MB). The spherical reference is outside the selected geometry. Neither
   million-seed output nor a sparse/fake large file substitutes for this input gate.

No ready authorized implementation or verification remains blocked merely by a
milestone report. On a supplied response/model or fixture, resume that specific
gate using the existing core and record, not an unrestricted new exploration.

## Explicit Limits And Dispositions

Canonical APIs remain the default/rollback path. Their generated 2D read/write,
bilinear/coarse-fine, Dataset, VTK and helper workflows are verified. New native
consumers are nonperiodic Cartesian 3D; periodic metadata roundtrips are not
periodic halo computation (canonical forest currently disables periodic links).
No new 2D extrusion, periodic tracing, CT or non-Cartesian computation is claimed.
Exact boundary footpoints, Q and GPU remain outside the selected delivered
profiles, with reopen conditions in the design. E1/E2/E3/E4 and new compact
persistent fill plans are evidence-backed deferred routes, not unfinished
mandatory prototypes. Retain fast resident and memory-bounded preparations.

## Retained Outputs

Ignored `benchmark-results/analysis-core/` contains all raw JSON/logs, the ~543 MB
global-curl native product, ~96 MB million-seed summary, LOS scalar arrays/figure,
and clean-source build/wheel artifacts. The 24 GB uniform value stream was
consumed, not saved as a cube. Preserve these useful results; reclaim only owned
scratch if needed under the recorded resource limit.
