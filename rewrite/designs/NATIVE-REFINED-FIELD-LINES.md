# Native Refined Field-Line Design

## Problem And Outcome

The M1 core now supplies exact refined ownership, trilinear vector sampling,
native selective reads, and a completed-owner sampling session. It still has no
field-following workflow. An exhaustive current/history audit found no repo-
owned tracer, stepper, normalization, termination, line-integral, or trajectory
output contract. An old `yt.LinePlot` uses a predetermined straight segment and
is not field following; archived point samplers have ownership behavior already
rejected by LOC-001. This group therefore introduces explicit new semantics
rather than claiming migration parity.

The composed result is:

```text
finite seeds + direction signs
  -> CHS-001 stage vector samples
  -> FLN-001 normalized field-line RHS and oriented integral RHS
  -> RKS-001 fixed classical RK4 stages/candidate
  -> TRM-001 explicit nonperiodic result/termination decisions
  -> SLE-001 serial stage-major accepted-prefix trajectories
```

The first representative reducer is cumulative oriented
`I = integral(B dot dx)`. The path uses physical arclength because
`dx/ds = sign * B/|B|`. The group does not add adaptive stepping, error control,
periodic wrapping, direction-projected support, raw source caching, prefetch,
parallel seed scheduling, dataset lifecycle, or public cutover.

## Numerical Choices And Alternatives

Tracing `dx/dt=B` makes step size depend on field magnitude and is appropriate
for a physical velocity streamline, but a magnetic field line has no natural
time scale. Unit tangent gives a physical-distance step and makes forward/
backward traces comparable. A robust scale-first norm avoids overflow and
underflow that a direct square sum would introduce. Exact zero is a classified
termination; no arbitrary epsilon changes the topology of weak nonzero fields.

Euler/midpoint are simpler but give poor curved-line accuracy at representative
steps. Adaptive RK would require error norms, accept/reject policy, minimum/
maximum steps, interpolation reuse, and more data-dependent sampling before the
cache/ownership path is proven. Fixed classical RK4 is deterministic, familiar,
has an independent operation-tree reference, and supplies a convergence
baseline. Its four stage samples remain explicit; there is no FSAL or hidden
adaptive reuse.

Clipping a failed step to the domain boundary needs a separate root solve and
continuous-segment contract. M1 instead rejects the whole attempted step when a
stage/candidate is exterior and retains the last accepted point. Exact global
upper remains exterior under LOC. Periodic metadata is not operational in the
current forest and cannot imply wrapping before M3.

A one-seed loop minimizes scratch but repeats Python/cache calls and prevents
independent seeds at the same RK stage from sharing owner groups. SLE uses
stage-major batches: active seeds stay in ascending input order for k1, k2, k3,
k4, then candidate commits occur in that order. Numerical state remains per seed
and is unchanged by batch composition. Parallel scheduling remains substitutable
later.

## Fixed Field-Line RHS

For finite sampled `(Bx,By,Bz)` and `sign` in `{-1,+1}`:

```text
ax = abs(Bx); ay = abs(By); az = abs(Bz)
m  = max(ax, ay, az)                 # ordered comparisons
u  = B / m
r  = sqrt(((ux*ux) + (uy*uy)) + (uz*uz))
mag = m * r
dx/ds = sign * (u / r)
dI/ds = sign * mag
```

Every multiply/add/divide is separately rounded binary64 and the production
implementation prevents contraction. Nonfinite source components, exact
`m==0`, and nonfinite `mag` receive distinct statuses and do not publish an RHS
row. The scale-first tree keeps representable very small/large vectors usable.

## Fixed RK4 And Termination

RKS operates on augmented state `(x,y,z,I)`. With positive normal `h`:

```text
s2 = base + (0.5*h) * k1
s3 = base + (0.5*h) * k2
s4 = base + h * k3
t2 = 2*k2; a = k1+t2
t3 = 2*k3; b = a+t3
c  = b+k4
h6 = h/6
candidate = base + h6*c
```

TRM fixes result rather than cache/I/O failure semantics. Finite exterior seeds
store zero points and `SEED_OUTSIDE`. An interior seed is point zero with
integral positive zero. At every RK stage, nonfinite state precedes domain exit;
after a successful sample, FLN status determines field termination. Candidate
precedence is nonfinite state, all-coordinate bitwise no-progress, then domain
exit. Only a fully accepted candidate appends position/integral atomically.
After exactly `max_steps` accepted steps the result is `MAX_STEPS`.

There is no continuous segment-boundary test. A rejected stage/candidate and its
sample/integral are not appended. A numerical termination affects only that
seed. Source/reader/resource failure remains an exception and preserves prior
accepted prefixes; it is not converted to a scientific status.

SAM-005 can reject a GEO-valid interior point when binary64 stencil arithmetic
is unrepresentable. CHS exposes that one case as a stable indexed `ValueError`
subclass. SLE maps it to `UNREPRESENTABLE_SAMPLE` at the current RK marker,
removes that seed, and retries the untouched compact batch; other CHS exceptions
retain prefix/propagation semantics.

## SLE Execution And Cache Composition

SLE accepts a caller-created CHS session, so cache provenance, capacity, clear,
and warm history stay explicit. It validates seeds, signs, scalar controls,
output shapes/writability/aliases, cache/session state, and checked byte
arithmetic before any output/cache mutation or I/O. It does not clear the
session.

For each accepted-step round, active seed IDs are ascending. Each stage builds a
private compact batch, classifies state/domain before sampling, calls CHS once,
maps FLN statuses, and retains returned owners as the next hints. Candidate
classification/commit is also ascending. Cache capacity/history changes only
reads/stats, never trajectory bits or termination.

The outputs are fixed-capacity caller arrays: positions
`float64[N,max_steps+1,3]`, cumulative integral
`float64[N,max_steps+1]`, counts `int64[N]`, termination code `uint8[N]`, and
termination stage `uint8[N]`. Only accepted prefixes and final control arrays
change; tails retain caller bits.

## Decomposition Records

### FLN-001

- Responsibility: map sampled magnetic vectors and orientation to a normalized
  field-line tangent plus oriented `B dot dx` integrand/status.
- Owned decisions: scale-first norm tree, exact-zero/nonfinite/unrepresentable
  categories, orientation, and published RHS layout.
- Non-owned: interpolation, RK, termination policy, cache, I/O, and scheduling.
- Inputs/outputs: borrowed vector/sign rows; caller RHS/status arrays.
- Mutation/ownership: full preflight before status/RHS writes; failed rows
  preserve RHS bits.
- Access/reach: pointwise rows, no halo.
- Reference: scalar operation tree, analytic magnitude/tangent invariants.
- Producer/consumer: CHS -> RKS/SLE.
- Performance: `hot-kernel`; rows/s, fixed call/allocation, SLE composition.
- Five questions: yes/yes/yes/yes/yes.

### RKS-001

- Responsibility: construct/finalize fixed classical RK4 augmented-state rows
  from explicit RHS stages.
- Owned decisions: half/full stage and final binary64 operation trees.
- Non-owned: RHS meaning, sampling, accept/reject, termination, cache, scheduling.
- Inputs/outputs: borrowed `(N,4)` states/RHS and positive normal step; caller
  stage/candidate arrays.
- Mutation/ownership: validated nonoverlapping output only; no retention.
- Access/reach: pointwise rows, no halo.
- Reference: scalar operation tree and analytic convergence compositions.
- Producer/consumer: FLN -> SLE/TRM.
- Performance: `hot-kernel`; rows/s and SLE composition.
- Five questions: yes/yes/yes/yes/yes.

### TRM-001

- Responsibility: classify nonperiodic seed, RK-stage/RHS, candidate, and budget
  outcomes into stable codes/stage markers and accepted-prefix rules.
- Owned decisions: code vocabulary, precedence, half-open domain, no-progress
  bits, whole-step rejection, and max-step outcome.
- Non-owned: RK arithmetic, field status production, sampling/cache/I/O, and
  seed scheduling.
- Inputs/outputs: scalar/row state and FLN status to exact enum values.
- Mutation/ownership: pure return values; SLE owns result arrays.
- Access/reach: coordinates/domain only.
- Reference: exhaustive truth table.
- Producer/consumer: FLN/RKS/LOC domain -> SLE.
- Performance: `cold/control`; fixed scalar work, no allocation.
- Five questions: yes/yes/yes/yes/yes.

### SLE-001

- Responsibility: execute one deterministic serial stage-major multi-seed
  field-line schedule into caller-owned accepted-prefix results.
- Owned decisions: active-seed order, CHS call schedule, scratch reuse, atomic
  candidate commit, statistics, and external-failure prefix behavior.
- Non-owned: FLN/RKS/TRM/LOC/SAM/RHE meanings, cache policy/provenance, adaptive
  control, periodicity, parallelism, and dataset lifecycle.
- Inputs/outputs: explicit CHS session, seeds/signs/step/budget, fixed-capacity
  result arrays, exact execution statistics.
- Mutation/ownership: complete ordinary preflight first; private stage arrays;
  accepted prefixes and prior cache admissions survive external later failure.
- Access/reach: four CHS point batches per attempted step, fewer after stops.
- Reference: scalar one-seed FLN/RK/TRM composition; cache/backend invariance;
  constant/rotational analytic fields and step-halving convergence.
- Producer/consumer: CHS/FLN/RKS/TRM -> M1 native streamline vertical slice.
- Performance: `milestone-workflow`; trajectory/cache/I/O/memory/latency scaling.
- Five questions: yes/yes/yes/yes/yes.

## Group Completion And Reopen Gates

Cover FLN extreme/IEEE/status rows, exact RK trees and convergence, exhaustive
TRM precedence, empty/exterior/zero/nonfinite/no-progress/domain/max-step paths,
forward/backward oriented integrals, single/multiple coherent/divergent seeds,
cache/backend/capacity invariance, mixed refined relation/PBC sampling, ordinary
atomicity, reader failure between stages, accepted-prefix preservation, exact
stats/memory, and synthetic plus real tdm native trajectories. Run one build,
focused dependency regressions, all rewrite tests, standard kernel/workflow
profile, evidence, staged review, and one cohesive checkpoint.

At group close, use actual native trajectory miss/support/owner traces to decide
direction-projected support, raw caching, exact neighbor transition, or reusable
RK stage buffers. Do not implement them from pre-step point schedules alone.
