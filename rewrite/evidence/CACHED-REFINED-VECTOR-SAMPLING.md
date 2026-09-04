# Cached Refined Vector Sampling Evidence

## Outcome And Review

HLO-001 and CHS-001 prepare the M1 field-line slice without changing point,
interpolation, or halo semantics. HLO tests one explicit prior-owner leaf with
the exact GEO half-open faces and otherwise calls the same compiled descent as
LOC-001. CHS validates one immutable reader/FST/GEO/ordered-field/PBC session,
reuses one RHE miss workspace, groups dynamic points by ascending owner, and
retains completed one-cell primary halos in a deterministic byte-bounded LRU.
Hits call unchanged SAM-005 without I/O or RHE planning. Capacity zero remains
the uncached numerical reference; public RPS and fused SAM-002/SAM-003 remain
available strategies.

Independent design review first proposed a raw-interior LRU, then reversed that
choice after the disposable probe showed 29--45x completed-halo gains at about
98% hits. A raw cache cannot remove the measured RHE preflight/application and
had no post-halo-cache native capacity evidence. Review approved the separate
HLO/CHS boundary and full all-26 miss behavior.

Implementation review found one material alias defect: dynamic points/hints
could overlap persistent workspace/cache arrays that mutate after stencil
preflight. Those aliases are now rejected before planning, output, cache, or
I/O. The same review required enforced non-reentrancy. Sampling now sets a busy
flag through a `try/finally`; nested sampling or clearing rejects before shared
state mutation, and every success/failure restores the idle state.

Review also corrected cache publication. A reader/RHE/SAM failure leaves the
old victim untouched. Immediately before copying a new completed halo, CHS
invalidates the old key; it publishes the new key/recency only after the full
copy. A recoverable copy failure therefore leaves an invalid slot rather than
an old key pointing at partially overwritten payload. Final independent audits
approved both contracts and implementations.

## Correctness, Failure, And Memory

The clean extension build, 152 focused HLO/CHS/LOC/SAM/RPS/RHC/RHE/DAT checks,
and all 964 rewrite tests pass. Coverage includes:

- exact HLO equality with LOC for absent/correct/stale/random hints, refined
  lower/upper faces, combined faces/edges/corners, every exterior side and exact
  global upper, mixed depth, invalid/nonfinite/corrupt hinted geometry, aliasing,
  and pre-mutation atomicity;
- empty/all-exterior/interior CHS calls; coherent and round-robin owners;
  cache capacities zero, one, exact working set, and resident; ascending-owner
  grouping; deterministic invalid/LRU/tie choice; recency and payload-preserving
  clear; and per-call rather than cumulative statistics;
- bitwise owner/value equality with LOC and public RPS through cache histories,
  capacities, array/native readers, repeated/reordered file fields, and mixed
  SAME/FINER/COARSER/PHYSICAL relations plus boundary modes 0--3;
- one factory empty reader conformance call, no nonempty I/O on ordinary error,
  first/later reader failure prefixes, old-victim preservation, invalid-key copy
  failure, nested sample/clear rejection, and idle-state recovery;
- output versus dynamic/reader/metadata/session aliasing, including points over
  RHE payload and hints over miss IDs/cache keys/recency/payload;
- exact independent session-byte formulas, call-plan bytes, logical reads,
  selected/support loads, maximum slots, cache budget/capacity, overflow, copied
  field/PBC provenance, and real `data/tdm.dat` native integration.

One cache entry uses exactly
`3*product(B+2)*8 + 16` bytes. For the synthetic `B=8` fixture that is 24,016
bytes: 24,000 payload plus two `int64` metadata values. The reusable RHE and
fixed session state is 1,502,984 bytes. Capacity 1/4/5/resident (71) therefore
uses 1,527,090/1,599,138/1,623,154/3,208,210 managed bytes. Dynamic four-point
call plans are about one hundred bytes; sorting transients and process RSS are
reported separately.

Commands:

```text
.venv/bin/python rewrite/build_ext.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_hlo_001.py rewrite/tests/test_chs_001.py rewrite/tests/test_loc_001.py rewrite/tests/test_sam_005.py rewrite/tests/test_rps_001.py rewrite/tests/test_rhc_001.py rewrite/tests/test_rhe_001.py rewrite/tests/test_dat_003.py rewrite/tests/test_dat_003_integration.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests
```

## Standard Synthetic Native Profile

The standard Apple arm64/Python 3.11.14 run uses five repetitions, 192 points,
and 48 ordered four-point batches to model sequential RK-stage availability.
The balanced synthetic native v5 fixture has 71 leaves at levels one/two,
`B=8`, and reordered/repeated fields `[2,0,2]`. All HLO owners and every native/
array/cache result are bitwise equal to LOC/public RPS. Native and array CHS
statistics are exact matches.

HLO's checked/unchecked singleton medians are 24.29/1.96 us for the coherent
order and 23.88/2.00 us for the divergent order. Checked/unchecked batch medians
are 0.034/0.006 ms coherent and 0.040/0.014 ms divergent. Coherent hints hit
187/192 points; divergent prior-owner hints correctly hit 0/192 and fall back
to the shared hierarchy path.

| Query/cache | Hits / misses | Selected loads | Logical/native payload | Cleared median | Warm median | Session managed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| coherent, cap 0 | 0 / 48 | 1,008 | 12,386,304 / 8,257,536 B | 156.046 ms | 154.243 ms | 1,503,074 B |
| coherent, cap 1 | 43 / 5 | 99 | 1,216,512 / 811,008 B | 25.430 ms | 25.156 ms | 1,527,090 B |
| coherent, cap 5 | 43 / 5 | 99 | 1,216,512 / 811,008 B | 25.353 ms | 3.531 ms, 0 B | 1,623,154 B |
| coherent, cap 71 | 43 / 5 | 99 | 1,216,512 / 811,008 B | 25.509 ms | 3.524 ms, 0 B | 3,208,210 B |
| divergent, cap 0 | 0 / 192 | 4,896 | 60,162,048 / 40,108,032 B | 591.455 ms | 590.147 ms | 1,503,074 B |
| divergent, cap 1 | 0 / 192 | 4,896 | 60,162,048 / 40,108,032 B | 591.105 ms | 590.506 ms | 1,527,090 B |
| divergent, cap 4 | 188 / 4 | 102 | 1,253,376 / 835,584 B | 16.161 ms | 3.909 ms, 0 B | 1,599,138 B |
| divergent, cap 71 | 188 / 4 | 102 | 1,253,376 / 835,584 B | 16.927 ms | 3.934 ms, 0 B | 3,208,210 B |

The exact warm capacity knees are the five-owner coherent and four-owner
divergent working sets. Resident capacity provides no material warm improvement
over those budgets but retains about 1.59--1.61 MB more. Compared with capacity
zero, working-set caches reduce the cleared logical reads 10.2x/48.0x and warm
reads to zero; warm runtime improves 43.7x/151x for coherent/divergent.

Public RPS over each complete, already-known 192-point query takes 11.58 ms
coherent and 7.15 ms divergent with 43/78 selected loads. That is an important
batch reference, not a realizable data-dependent-stage schedule: CHS receives
48 batches sequentially and cannot know later RK points. The earlier per-point
RPS probe is the matching uncached schedule and took 583/536 ms. CHS capacity
1 coherent and capacity 4 divergent therefore retain the measured roughly
23--45x cold scheduling benefit while exact working-set warm calls remove all
miss work.

## Real TDM Native Profile

The real tdm profile uses 96 points in 24 four-point batches, three owners with
32-point runs, native magnetic fields `[4,5,6]`, and `B=10`. Hints hit 93/96
points. Capacity 3 is the exact working-set knee.

| Capacity | Cleared hits/misses | Loads | Native header/payload | Cleared median | Warm median | Session managed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 21 / 3 | 28 | 672 / 672,000 B | 7.510 ms | 7.486 ms | 1,255,134 B |
| 3 | 21 / 3 | 28 | 672 / 672,000 B | 7.510 ms | 1.763 ms, 0 B | 1,338,110 B |
| 4 | 21 / 3 | 28 | 672 / 672,000 B | 7.518 ms | 1.763 ms, 0 B | 1,379,598 B |
| 27 | 21 / 3 | 28 | 672 / 672,000 B | 7.516 ms | 1.765 ms, 0 B | 2,333,822 B |

Public full-query RPS takes 3.226 ms and selects 20 loads/480,000 payload bytes,
again using future points unavailable to a field-line stepper. CHS cold misses
select 28 rows across three separately completed owners, while the one-batch RPS
union proves 20 unique support rows. A resident raw-block cache could therefore
save at most 192,000 of 672,000 payload bytes (28.6%) here, but it would remove
none of the three RHE preflight/application costs. The measured native-versus-
array miss overhead is not yet material enough to add a second cache policy
before the actual streamline workflow.

Benchmark command and ignored raw record:

```text
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/chs_001.py --profile standard --output rewrite/benchmark-results/chs-001-standard.json
```

## Retained Decisions And Reopen Triggers

Retain exact hint-or-LOC ownership, session-level prevalidation, persistent RHE
workspace, ascending owner grouping, completed-halo LRU, capacity zero, and full
all-26 miss semantics. Cache size should follow the measured active-owner working
set, not total leaf count. HLO's remaining hierarchy fallbacks are only owner
transitions and are negligible beside cold halo fills; exact neighbor lookup and
TOP-003 remain deferred.

Direction projection remains attractive on cold misses: the synthetic working-
set fills average 19.8--25.5 selected loads per owner. However COARSER slope
support and mixed physical widening require a new planner/executor proof. The
contracted trigger is an actual native SLE miss-heavy or cache-thrashing trace,
not this pre-step point schedule. Proceed to FLN/RKS/TRM/SLE with CHS as the
explicit sampler; evaluate projection from that complete trajectory evidence.
