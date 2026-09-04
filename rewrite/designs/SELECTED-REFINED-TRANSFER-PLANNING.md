# Selected Refined Transfer Planning Group

## Outcome

An explicit strictly increasing selected-leaf stream is closed over requested REL
directions and mapped to SAME, FINER, and COARSER source/workspace boxes without
scanning unselected primary leaves or reading payload.  The group members are
SPR-001, FRP-001, and CWP-001.  Existing STO-004 and numerical contracts remain
unchanged.

## Exploration Coverage

Current approach: STO-004 plans dense ascending primary IDs, SLB-001 translates
same-level boxes, and RST/PRL own ratio-two values without transfer placement.

Structurally different credible alternative: convert a sparse selection into
maximal dense ID runs and reuse STO-004 separately for every run, while deriving
all transfer geometry inside a later refined halo executor.

Most decision-changing uncertainty: whether sparse selection requires a new
support representation or makes source-slot and transfer artifacts incompatible.

Reopen evidence or consumer: a sorted sparse selection containing distant leaf IDs
must produce the same primary-prefix/support representation consumed by
RSL/TGT/RPH while omitting all intervening IDs.  The group composition measures
plan-to-gather bytes and support amplification; a representation change is
reopened only if that consumer cannot remain bounded or deterministic.

The dense-run alternative is deferred because fragmentation makes its number of
plans proportional to the number of runs and can still admit unwanted primary
IDs if runs are coalesced.  The later executor owns scheduling and may still use
STO-004 directly for a genuinely dense full-domain request.  Arbitrary caller
order is deferred to a higher selection/result adapter with an inverse
permutation; reopen only when a public consumer requires that ordering.

## Algorithm Selection Record

Primary analysis workflow and query shape: selected-region local fields over a
small or medium sparse leaf selection and a small field set.

Expected access density and locality: selected primaries may be noncontiguous;
support is locally clustered around each primary and overlaps within a region.

Reusable state and its lifetime: forest/REL semantics live for the unchanged
snapshot; selected relation rows and transfer boxes live for one compatible
operator batch.

Dominant expected cost: payload I/O and support amplification, followed by
relation/plan bytes.  Geometry arithmetic is secondary.

Current and structurally different candidate strategies: direct ordered
selected-prefix planning versus dense-run decomposition; explicit separate
transfer boxes versus executor-internal action dispatch.

Workload-specific metrics and representative consumer: exact selected/support
IDs, requested/read bytes, support amplification, reader calls, plan/workspace
bytes, and plan-to-gather time for small/medium/full selected sets.

Deferred alternatives and concrete reopen triggers: hashing or dense marker
tables remain deferred until a real selected composition shows linear bounded
membership dominates; fused transfer planning remains deferred until retained
box artifacts are measured as material overhead in refined halo application.

## Group Gate

SPR-001 is composition-only at selected REL/plan/gather.  FRP-001 and CWP-001
are cold/control `O(R)` integer geometry with exact caller-owned output budgets
and no internal size-dependent allocation.  Focused references freeze each
meaning.  The closing composition must prove exact order and preservation, no
unselected primary payload reads, bounded plan/workspace bytes, and SAME/FINER/
COARSER box coverage.  Run one rewrite build, all focused tests, the accumulated
rewrite suite, safe current comparisons, and one selected plan-to-gather
composition.  Do not rerun unaffected historical kernel benchmarks.
