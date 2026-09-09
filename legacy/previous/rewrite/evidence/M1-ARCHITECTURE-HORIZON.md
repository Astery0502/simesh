# M1 Cartesian 3D Refined AMR Architecture Horizon

## Verdict

M1 is complete with one qualified real-fixture limitation. The functional core
now reconstructs and validates Cartesian 3D refined forests; plans and applies
balanced SAME/FINER/COARSER/physical halos; performs exact ownership, zero/
trilinear sampling, selected native v5 reads, bounded local curl/reduction, and
cached fixed-step field-line analysis. Both analysis-priority vertical slices
run through the native reader with explicit memory, I/O, failure, and numerical
evidence.

No additional 3D numerical capability is required before M2. Direction-
projected support is the highest-priority measured reopen, but implementing it
against the hard-coded 26-direction representation now would entrench the
assumption that M2 must replace. It becomes an explicit input to the first
dimension-aware M2 halo design while all-26 RHE/CHS remains the 3D reference.

Independent architecture and performance audits reached the same conclusion.
They also identified and this checkpoint repairs one stale lifecycle statement:
RPS-001 now depends on and documents the public RHC-001 completed-primary
boundary rather than the private hook that existed before LFE triggered its
promotion.

## Deliverable Audit

| M1 / analysis-gate deliverable | Authoritative evidence | Disposition |
| --- | --- | --- |
| Functional reader/writer and resident/bounded substitution | STO-003, RHE/RHC, array/native comparisons | complete |
| Refined parent/child forest, contacts, balance, geometry | FST-001/002, TOP-002, BAL-001, GEO-002; real WENO metadata/tree | complete for Cartesian 3D |
| Sparse selected support and SAME/FINER/COARSER/physical application | SPR/SLB/FRP/CWP/CSP/PWA/CWA/RHE and all relation/mask references | complete for even `B>=4`, side reach `<=B/2` |
| Refined zero/trilinear point sampling | LOC, SAM-004/005, RPS; retained SAM-002/003 | complete |
| Native selective `.dat` source | DAT-001/002/003 endian, provenance, tdm, WENO bridge, failure evidence | complete for v5 spatially 3D non-staggered payload |
| Selected local diagnostic and reduction | ROI/OPR-003/RHC/LFE; synthetic refined native and real tdm/current | complete |
| Cached field-line/streamline slice | HLO/CHS/FLN/RKS/TRM/SLE; analytic, synthetic native, real tdm | complete; explicitly new behavior, not false tracer parity |
| Runtime, memory, I/O and scaling | standard DAT/LFE/CHS/SLE profiles and exact managed-byte formulas | complete |
| Architecture horizon | this review and independent audits | complete |

The latest clean extension build, 301 field-line focused/dependency checks, and
all 1,210 rewrite tests pass. Earlier group evidence supplies the complete
capability-specific checks. M1 closure is distinct from later migration:
derived-field breadth is M4; full parsing/writing/export is M5; dataset/public
API lifecycle is M6; packaging, parallel runtime, fallback, and cutover are M7.

## Qualified Real-Data Evidence

The repository has no file that is simultaneously genuinely refined,
non-staggered, native, Cartesian 3D. M1 is nevertheless not synthetic-only:

- real non-staggered tdm exercises native selected reads, local curl/current
  comparison, completed-halo caching, and field lines;
- the real refined WENO file exercises metadata, tree, topology, balance, and
  geometry directly;
- the WENO regular-field bridge streams real `b1,b2,b3` bits into a temporary
  non-staggered file while preserving the real forest/geometry and exercises
  native refined RHE/RPS;
- independently generated non-staggered refined v5 files exercise native
  refined halos, local fields, caches, and trajectories end to end.

This is a qualified pass, not a claim of direct real refined non-staggered
parity. Acquire or provenance-record such a fixture when available; its absence
does not identify a missing implementation or an external decision that blocks
M2.

## Architecture Disposition Before M2

| Boundary or finding | Decision | Reopen timing |
| --- | --- | --- |
| Canonical `float64[slot,field,x,y,z]`, signed int64 IDs/triplets, STO-003, WSP, selected plans, RED | retain; singleton-z stays the interoperability layout | M2 adds inactive-axis invariants |
| Active dimension disappears after DAT parsing and is not carried by forest/binding | reopen: explicit `active_ndim` in every M2 artifact lifecycle; never infer from `root_shape[2]==1`; keep `ndir` separate | first M2 group |
| MOR-001 with z extent one | retain/generalize with exact 2D Morton proof | M2 foundation |
| Octree `(node,8)` and all-axis refinement | retain 3D wrapper; define dimension-stamped quadtree semantics before choosing four columns or sentinel-padded eight | M2 forest group |
| GEO/LOC/SAM/ROI refine and interpolate z | reopen at one active-axis geometry seam; inactive z has scale one, coordinate zero, window `[0,1)` | M2 geometry/sampling |
| DAT-001/003 fixed 3D metadata/tree and six ghost integers | reopen variable-width v5 parsing and `2*active_ndim` record headers; normalize disk 2D to singleton-z block arrays | before real M2 slice |
| REL/RPH/RST/PRL/CSP/RHE use 8 children, 26 directions, fine cap four, universal bound 57, and z reach | retain exact 3D wrappers; add active-dimension transfer/halo path with 8 directions, 13-slot 2D bound, inactive-z reach zero/Bz one, four-value restriction, xy slopes | M2 halo group |
| SAM-005 eight-load trilinear tree | retain; add genuine four-load y/x bilinear, not trilinear with zero z weight, to preserve NaN/signed-zero/reach semantics | M2 sampling |
| OPR-003 3D curl | retain; specify an explicit 2D/2.5D operator rather than silently dropping terms | M2 operator |
| PBC scalar rules, TGT/SLB box algebra, LIM, storage adapters | retain; extend admission/enumeration only over active axes | M2 consumers |
| RHC uses private RHE execution; CHS reuses private RHE workspace; SLE holds private CHS busy state | retain as measured M1 internal fused seams; do not copy them into 2D; extract a dimension-aware executor/session seam when the M2 halo consumer needs it | M2 halo design or M4--M6 |
| RHE/CHS repeat per-miss planning and LFE computes bounds while consuming only spacing | defer: correct, explicit, and not the measured selection bottleneck | reconsider at shared M2 geometry/executor seam |
| LFE allocates full block rows for partial ROI output | defer compact ragged/external sink strategy | M4/M6 or output-memory trigger |
| HPL/HAX, STO-004, SAM-001/002/003, INT-001 | retain/deprioritize as compatibility and exact reference surfaces | expand only for a concrete M2 consumer |
| TOP-003 face materialization | defer; no consumer amortizes it | post-initial HLO fallback >=10% or locator >=10% warm time |

## Performance Reopen Decisions

### Direction-Projected Support

The signal is material: small tdm LFE reads 27 blocks for one primary and
648,000 payload bytes for 1,536 useful output bytes. Correctly sized tdm SLE
performs three cold owner fills, 28 loads/672,000 payload bytes, and takes
8.814 ms cold versus 2.957 ms warm. But CHS gives 93/96 hits and zero warm
reads, and projection must still prove COARSER slope closure plus mixed physical
widening. No M1 correctness or usability gate fails.

Do not add a 3D-only shortcut. The first M2 halo design must represent requested
target directions separately from additional internal slope/base support and
must not copy unconditional full-neighborhood closure into bilinear/face-only
consumers. Promote projection when an exact prototype yields at least 20%
composed improvement or 2x support/byte reduction on two representative refined
cases. Preserve all-26 execution as reference and fallback.

### Raw Cache, Headers, Mmap, And Planning

- A raw-interior LRU can reduce the tdm SLE miss union only from 28 to 20 loads,
  at most 192,000 bytes/28.6%, and removes no RHE planning/application. Reopen
  after direction planning only if repeated native transfers remain at least
  20% of transition-heavy wall time under a materially smaller cache budget.
- Eager record-header scanning is rejected for sparse cold work: the WENO scan
  reads only 542,736 header bytes but takes 3.297 seconds cold versus 26 ms warm.
  Tdm CHS headers are 672 bytes beside 672,000 payload bytes. Owned fd/mmap/
  eager-header lifecycle belongs in M5/M6 unless post-support-reduction syscall
  time reaches 20% while sparse first-result latency is protected.
- Exact neighbor transition remains closed: measured tdm HLO fallbacks are the
  three initial seeds, with no transition fallback. Reopen at 10% post-initial
  fallback rate or 10% warm trajectory time, including TOP-003 build/memory.
- ROI selection is below 0.5% of small first-result time, CHS plans are about
  100--203 bytes, and SLE scratch is `314*N`; caching current all-26 plans would
  entrench a representation immediately before dimensional generalization.

Adaptive RK and independent-seed parallelism now have fixed-step accuracy and
stage-major baselines, but remain later execution strategies rather than M2
foundation requirements.

## M2 Entry Gate

The first M2 group is **Active-Dimension Foundation**, initially a singleton
DIM-001 capability. It must freeze explicit `active_ndim in {2,3}`, distinguish
it from `ndir`, retain the 5D singleton-z canonical array layout, define active/
inactive axis normalization and provenance, and prove that legitimate 3D data
with a singleton root/block z extent is not misclassified as 2D.

Only after DIM-001 integrates may M2 specify quadtree/Morton, variable-width DAT,
geometry, halo, bilinear sampling, and operator groups. A representative
Cartesian 2D `.dat` file is not present in this repository; generate and
provenance-record one before M2 closes its required real-data vertical slice.
