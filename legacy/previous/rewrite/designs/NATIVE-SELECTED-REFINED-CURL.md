# Native Selected Refined Curl Design

## Problem And Outcome

The refined core can select file blocks, complete their halos, and sample
points, but it cannot yet answer the primary local-analysis query: compute a
scientific stencil only for cells in a physical region, return compact results,
and reduce those same cells without loading/materializing the full snapshot.
Existing OPR-002 assumes one spacing and one box for every slot, while refined
ROI leaves can differ in level and local cell window. Current derived-field code
loads all requested fields, retains a padded whole-mesh payload, computes a
conservative padded derivative array, and commonly materializes a uniform grid.

This group produces a bounded native vertical slice:

```text
physical ROI
  -> ROI-001 canonical leaf/local-cell windows
  -> DAT-003 reader + RHC-001 complete primary chunks
  -> OPR-003 Cartesian curl over equal-window runs
  -> compact selected curl blocks + RED-001 component sum
```

ROI-001, OPR-003, and RHC-001 are independent prerequisites. LFE-001 owns only
their scheduling and call-scoped workspace. Streamline stepping/cache and the
broader derived-field registry remain separate work.

## Alternatives And Boundary Choices

A leaf-volume intersection alone gives a conservative block cover, not the
cells in the requested region; summing complete selected blocks would silently
change the scientific query. ROI-001 therefore returns one nonempty local
cell-center box per selected leaf. A future weighted volume integral or
face/volume overlap selector is a different capability.

Materializing all leaf bounds and vectorizing a mask costs `72*L` geometry
bytes per query. Hierarchy pruning can reduce sparse traversal but adds a more
complex algorithm before selection is known to matter. The initial two-pass
compiled algorithm scans canonical leaves in ascending ID, binary-searches
their center sequences, and allocates only the `56*S` output. It is reopened if
the standard profile makes selection more than 10% of small-ROI first-result
time or repeated ROI queries justify retained metadata.

Six OPR-002 derivative outputs followed by three OPR-001 differences provide a
clear reference but create six payload-sized temporaries and nine passes.
Current generic term batching avoids some storage but owns a broad expression
policy and initializes/accumulates through `+0 + coefficient*term`, including a
known signed-zero difference. OPR-003 instead fixes the concrete Cartesian curl
and fuses the exact OPR-002/OPR-001 arithmetic into one pass with no derivative
temporaries. A generic recipe engine requires a second concrete consumer.

OPR-003 retains a common source/output box, which is a clean numerical kernel
and enables contiguous slots. LFE-001 partitions each completed primary prefix
into maximal consecutive runs with identical ROI boxes; spacing remains
per-slot. This avoids one Python call per leaf without putting grouping policy
inside the operator or forcing padding/copies for unlike windows.

The RPS private completed-primary hook has acquired a second independent
consumer, activating its explicit promotion trigger. RHC-001 exposes a frozen
coarse-grained consumer descriptor and stable executor. Public RHE writer
behavior remains unchanged; RPS migrates to the same RHC boundary and proves
bitwise/statistical equivalence.

LFE output is compact `float64[P,3,Bx,By,Bz]` aligned with the ROI selection,
not a logical `L`-row BlockWriter. Only each row's selected cell box changes;
this avoids full-leaf result materialization while preserving a direct mapping
to canonical leaf IDs. The reduction is explicitly an unweighted serial sum of
one curl component, not a cell-volume integral.

## Canonical ROI Centers

For leaf level `ell`, global cell index `g`, and axis `a`, reuse GEO/SAM
operation order:

```text
base_h = (domain_upper - domain_lower) / float(domain_cell_counts)
h      = ldexp(base_h, -(ell-1))
factor = float(g) + 0.5
offset = factor * h
center = domain_lower + offset
```

Every operation is separately rounded binary64. A cell is selected exactly
when `region_lower <= center < region_upper` on all axes. Lower equality is
included and upper equality excluded. A contained region with any equal bounds
is empty. Selected windows are lower-bound intervals over the monotone center
sequence; first/last centers must remain finite and inside canonical leaf faces.

## Fixed Curl Arithmetic

Input field positions name `(Bx,By,Bz)`. For every slot/cell, OPR-003 evaluates
six OPR-002 derivatives with that slot's spacing and three OPR-001 scale-one
differences:

```text
Jx = d(Bz)/dy - d(By)/dz
Jy = d(Bx)/dz - d(Bz)/dx
Jz = d(By)/dx - d(Bx)/dy
```

Each derivative is separately `difference = plus-minus`, then
`derivative = difference * (0.5/h)`. Each component finishes with
`product = 1.0 * negative_derivative`, then
`result = positive_derivative - product`. All twelve neighbors are loaded.
For mixed spacing, the separate reference invokes OPR-002 one slot at a time
(or only over exact-equal-spacing groups), stores six binary64 fields, and then
uses OPR-001. Finite/signed-zero bits are equal; nonfinite classification/sign
follows the lower contracts. The fused tree intentionally differs from current
`+0 + coefficient*term` where operation order or signed zero matters.

## Decomposition Records

### ROI-001

- Responsibility: map a contained half-open physical box to exact refined
  cell-center windows.
- Owned decisions: center arithmetic, lower/upper inclusion, empty region,
  leaf omission, local boxes, and ascending row order.
- Non-owned: payload, spacing output, leaf-volume overlap, support, operator,
  reduction, scheduling, cache, and clipping.
- Inputs/outputs: borrowed FST/GEO/domain metadata and region bounds; caller-
  owned/allocated `leaf_ids[S]`, `cell_lower[S,3]`, `cell_upper[S,3]`.
- Mutation/ownership: complete validation/count before output writes; no input
  mutation or retention.
- Access/reach: metadata-only full leaf scan; no payload/halo.
- Reference: exhaustive scalar leaf/cell center enumeration.
- Producer/consumer: FST/GEO -> LFE primary IDs and cell windows.
- Performance: `cold/control`; `O(L*sum(log B))`, fixed scratch, `56*S` output;
  timed in LFE.
- Five questions: yes/yes/yes/yes/yes.

### OPR-003

- Responsibility: compute the three-component Cartesian curl over one explicit
  common box with per-slot spacing.
- Owned decisions: field/component mapping, six centered derivatives, exact
  OPR-002/OPR-001 operation tree, aggregate one-cell reach, and output mapping.
- Non-owned: field names, ROI grouping, geometry construction, halo values,
  reader/cache, scheduling, reduction, and current generic accumulation.
- Inputs/outputs: borrowed completed payload, boxes, field positions and
  `float64[S,3]` spacing; caller-owned destination fields/region.
- Mutation/ownership: only the translated destination box/three distinct fields;
  validation precedes mutation and nothing is retained.
- Access/reach: aggregate `LOCAL_STENCIL` lower/upper `(1,1,1)`; component-wise
  reaches are the two curl axes.
- Reference: per-slot six-OPR-002 plus three OPR-001 composition and scalar tree.
- Producer/consumer: GEO/RHE -> LFE output.
- Performance: `hot-kernel`; values/s, effective traffic, slots/box/level
  scaling, reference/current and LFE composition.
- Five questions: yes/yes/yes/yes/yes.

### RHC-001

- Responsibility: synchronously expose each fully completed selected refined
  primary prefix to one explicit coarse-grained consumer before workspace reuse.
- Owned decisions: consumer descriptor/lifetime, full-request RHE preflight,
  chunk callback order, read-only views, and consumer statistics.
- Non-owned: halo/topology/numerical meanings, reader backend, consumer work,
  cache, output/reduction, and parallel scheduling.
- Inputs/outputs: RHE inputs plus frozen consumer `(state,callable,output_arrays)`;
  exact chunk stats.
- Mutation/ownership: RHC mutates only its private workspace; consumer may
  mutate declared state/output arrays and must not retain views.
- Access/reach: caller-supplied RHE halo reach; consumer sees complete primaries,
  never support-only rows.
- Reference: existing private RPS composition and public RHE writer path.
- Producer/consumer: RHE internals -> RPS and LFE.
- Performance: `composition-only`; callback overhead at RPS/LFE, no cell-loop
  callback.
- Five questions: yes/yes/yes/yes/yes.

### LFE-001

- Responsibility: execute one bounded selected refined curl plus component sum
  over explicit ROI windows.
- Owned decisions: one magnetic read/halo traversal, equal-window run grouping,
  compact output placement, persistent serial reduction order, workspace and
  stats.
- Non-owned: ROI/curl/halo/reduction meaning, reader backend, cache, physical
  integral weighting, dataset registration, and streamline behavior.
- Inputs/outputs: functional reader, ROI arrays, selected fields, DAT/GEO/FST/
  PBC metadata and capacity; caller-owned compact curl output and accumulator.
- Mutation/ownership: call-scoped spacing/bounds/workspace; ordinary errors
  preflight before I/O/output; external failures may leave completed prefixes.
- Access/reach: one cell all sides through RHC/RHE; only requested output cells
  and one reduction component consumed.
- Reference: resident full-halo OPR-003 plus scalar RED in exact leaf/window
  order, and native/array/current comparisons.
- Producer/consumer: ROI/DAT/RHC/OPR/RED -> M1 local-field vertical slice and
  later derived-field integration.
- Performance: `milestone-workflow`; region selection, I/O/support/output bytes,
  TTFW, stages, calls, capacity, managed/output/RSS, numerical/reduction results.
- Five questions: yes/yes/yes/yes/yes.

## Completion And Reopen Gates

Correctness covers exact ROI center windows and edges, mixed level/box groups,
all curl axes/operation-order/IEEE/convergence cases, every refined relation and
PBC kind, RHC descriptor/view/failure semantics, capacity-invariant compact
outputs and reduction, native/array substitution, and a safe current common
region. Standard native small/medium/full WENO-bridge and real tdm profiles
record every analysis-workload metric available at this layer.

Reopen hierarchy pruning only from measured ROI cost, direction-specific
support from halo amplification, a generic recipe engine from a second operator,
and payload/header caching from the following streamline workload. Do not add
streamline stepping to this group merely because it also consumes fields.
