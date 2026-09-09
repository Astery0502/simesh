# WENO Feature-Completion Reference

## Role And Trigger

`data/weno509_sub_0000.dat` is the classic complex real-data example for
functional development and comparative feasibility/performance assessment.
It is an explicit feature-completion profile, never default pytest, `make test`,
per-edit, per-commit or automatic CI work. WORKFLOW owns the completion trigger;
PERFORMANCE owns comparison and regression rules. Select affected sections
after focused correctness passes, and record inapplicable sections with reasons.
Synthetic cases still own independent accuracy, boundary/failure and fast checks.

## Fixture Contract

Record the source path, file size, metadata, source-field positions, working
revision/diff and build. The recorded fixture is 1,045,232,320 bytes, Cartesian
3D, nonperiodic, root (2,2,1), B=(8,8,8), 22,614 leaves, levels 3/4/5/6 with
88/646/3,256/18,624 leaves. It is block-structured AMR, not an arbitrary
unstructured mesh. Never infer 2D from the singleton root z extent.

Original regular magnetic fields are positions (4,5,6), names (b1,b2,b3).
The original file is staggered; DAT-003 must continue to reject it. Reuse
`dat_003.repack_weno_regular_fields` to copy regular-field bits and preserve the
real forest into a temporary non-staggered fixture. Validate forest identity,
balance and selected payload bits. Count bridge time, read/write volume and disk
space separately and include setup in fresh one-shot feasibility. No direct
staggered/CT support or full-field physical-accuracy claim follows from a bridge.

## Stable Cases

Coordinates below are fractions of the physical domain. Keep these existing
local-field cases; additions get a named request definition, not silent changes.

| Case | Fractional lower -> upper | Purpose |
| --- | --- | --- |
| small | (.48,.48,.48) -> (.52,.52,.52) | 32 primaries; coarser contacts |
| mixed | (.42,.42,.42) -> (.58,.58,.58) | 403 primaries across levels 4--6 |
| thin | (.495,.3,.3) -> (.505,.7,.7) | 728 same-level primaries; support amplification |
| physical | (0,.3,.3) -> (.02,.7,.7) | 8 primaries; physical and FINER support |

LFE: three-field curl, width one, continuous physical modes, exact serial
component sum; capacities 57/256. Report selected windows, untouched output,
all relation kinds, reads/support, first result, planning/check/application and
kernel costs. Existing results are in [M1-WENO-ASSESSMENT](evidence/M1-WENO-ASSESSMENT.md).

Full-halo control: all 22,614 leaves, three fields, width two on both canonical
AMRMesh and RHC, continuous modes, RHC capacity 256. Reuse the canonical output
allocation as the RHC sink to stay within 2 GiB; account for RHC's full-payload
output copy. One complete numerical check warms the path, followed by three
timed repetitions and one separate attribution pass. This explicitly expensive
control is selected for whole-halo/dense regressions, not every local feature.

Sampling: a 192-point coherent x-line from fractions (.42,.5,.5) to
(.58,.5,.5), batches of four; and 36 SFC-spread owners from the mixed ROI,
four repetitions at lower bound +0.1 cell spacing, as a single 144-point batch.
Use exact LOC ownership and RPS as semantic reference. Cache capacities zero,
below the observed working set, at the working set, and twice it; application
cache cleared/warm. Do not allocate a whole-forest cache by default.

A common canonical interpolation case is a (16,8,4) grid of target cell centers
in the mixed box. Compare zero-order and linear values with exact ownership;
classify safe interiors separately from halo/interface-dependent samples.
Include a CHS session sized to the observed owner working set, with construction,
first query and subsequent warm query measured separately on those same points.
Raw-value and same-strategy comparisons are exact; canonical cross-strategy
comparisons record actual discrepancies, not automatic tolerance expansion.

Trajectories: eight SFC-spread mixed-ROI leaf centers, positive direction,
eight fixed RK4 steps, step length one quarter of the minimum selected spacing.
Compare original/retained M1 and cache histories, terminations, positions and
integrals. Original simesh has no equivalent contracted tracer: report that
gap instead of inventing legacy parity. This is not an accuracy/footpoint study.

## Component And Comparator Matrix

| Component | Original canonical simesh | Rewrite comparison |
| --- | --- | --- |
| Header/tree | get_metadata + AMRForest | DAT index/binding/balance; distinguish eager connectivity |
| Read/copy | selected-field eager read + interior copy | selective DAT bridge reads; matched selected bits, different whole-file work |
| Halo | original AMRMesh full-domain width two | selected RHE width one; compare common values, never call work volume equivalent |
| Operator | original first_derivative_fields | selected LFE/curl; safe interior and interface results separately |
| Sampling | uniform zero/linear grid methods | LOC/RPS/CHS on identical target centers; include preparation separately |
| Cache/trace | no equivalent native session/tracer contract | original M1, previous retained, candidate and uncached references |
| Output | padded/full-domain/grid allocations | selected output, caller buffers and optional future sinks |

Original M1 baseline: `397aaffc61be68d1141c8f30c0427dd37a4c8bc8`; first retained
optimization: `7b0c1b1`. Freeze the actual canonical source revision too. Current
canonical source at that revision is the original implementation comparator;
`simesh.legacy` is optional historical evidence, not required for every feature.
Classification per comparison: equivalent, restricted-common-domain,
different-work, unsupported, or resource-deferred. Report unsuccessful
equivalence checks as findings, with location/error magnitude and contract scope.

## Evaluation Budget And Outputs

Default one warmup and five repetitions, paired where variants share a process.
Reuse prior unchanged LFE evidence; run canonical, sampling and trajectories
sequentially to avoid resource contention. For this 8 GiB runner, use at most
2 GiB controlled live buffers in the canonical core stage and 512 MiB in bounded
query stages, with <=512 MiB temporary bridge disk. Inventory C allocations too.
Do not construct a maximum-resolution whole-domain uniform grid: each three-field
grid alone is approximately 1.61 GB. Admission failures must be recorded rather
than silently selecting smaller scientific requests. A later larger profile
requires its own explicit budget and does not follow from this fixture's size.

Separate startup, persistent bridge, cold/cleared/warm queries, output, whole
workflow footprint, RSS and page cache. Use disjoint/exclusive attribution or
label nesting; keep instrumentation out of headline samples. Preserve raw wall
and CPU timing, dispersion, exact counts and correctness results. A case without
an independent oracle says so, even if variants agree.

Close with one compact evidence record: feature/revisions, chosen cases and
omissions, semantic differences, costs/resources, feasibility and ranked next
actions. Assessment alone does not reopen M1 or start M2/optimization software.

## Explicit Commands

```text
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/m1_weno_assessment.py --output rewrite/benchmark-results/m1-weno-assessment.json
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/weno_reference.py --section canonical --output rewrite/benchmark-results/weno-canonical.json
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/weno_reference.py --section sampling --output rewrite/benchmark-results/weno-sampling.json
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/weno_reference.py --section trajectory --output rewrite/benchmark-results/weno-trajectory.json
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/weno_reference.py --section cache-grid --output rewrite/benchmark-results/weno-cache-grid.json
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/weno_reference.py --section attribution --output rewrite/benchmark-results/weno-attribution.json
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/weno_reference.py --section halo-full --repeats 3 --output rewrite/benchmark-results/weno-halo-full.json
```

The new component commands are selected explicitly; no command runs the entire
matrix implicitly. No source fixture or existing canonical report is overwritten.
