# Native Selective AMRVAC Read Design

## Problem And Group Outcome

The current reader parses metadata, then walks every leaf and materializes all
requested field interiors before any numerical consumer can run. That prevents
sparse refined analysis from preserving RHE/RPS selection and bounded-memory
semantics. The mmap helper is not a stable alternative: its final exported
views keep the mapping alive and currently raise `BufferError` on close.

This group separates three independently variable meanings:

1. DAT-001 decodes and structurally validates an AMRVAC v5 3D header,
   forest stream, tree rows, and minimum record offsets in either byte order.
2. DAT-002 reconstructs the canonical FST artifact, proves stored leaf
   levels/coordinates use exactly that leaf order, and carries source-file
   identity as explicit provenance.
3. DAT-003 accepts that binding plus its index and reads explicit non-staggered
   block/field/interior selections into a canonical STO-003 destination through
   a caller-owned file descriptor.

The composed outcome feeds the existing RHE-001 and RPS-001 operations without
importing current `datio`, retaining payload between calls, or introducing
dataset/cache lifecycle. Exact v3/v4 branches, staggered payload conversion,
2D, writing, and public dataset ownership remain M2/M5/M6 work.

## Format Recovery And Intentional Current Differences

MPI-AMRVAC's v5 writer records unaligned native scalar representations in this
order: header, preorder forest logicals, leaf levels, one-based leaf coordinates,
leaf int64 block offsets, then block records. Each ordinary block record starts
with six int32 lower/upper ghost extents followed by field-major float64 values;
within one field x varies fastest, then y and z. Canonical rewrite payload uses
`(slot,field,x,y,z)` with z fastest, so transfer must explicitly transpose/copy.

The file has no endian marker. DAT-001 decodes the first four bytes as both
little- and big-endian signed int32 and accepts the unique interpretation equal
to version five. Every remaining integer/double uses that byte order; returned
arrays and DAT-003 output use native NumPy representations. Opposite-endian
payload conversion is representational only and must preserve signed zero,
infinities, and NaN payload bits.

Current `get_header` reads v5-only periodic/geometry/staggered fields for v4 and
reads the v4 tail for v3, while its size/write thresholds disagree. DAT-001
therefore accepts exactly v5 instead of claiming broken v3/v4 compatibility.
It also rejects malformed counts/offsets before allocation rather than relying
on `struct`/reshape failures. These are intentional internal-scope corrections;
complete v3/v4 public behavior is revisited at M5.

## Alternatives And Exploration Coverage

One combined parser/forest/reader would make format versioning, canonical AMR
identity, and storage strategy change together. A two-way split that puts FST
binding inside parsing similarly prevents either parser or forest validation
from being substituted independently. The selected three boundaries pass plain
arrays/records and use existing MOR/FST/STO contracts.

An eager record index could read every 24-byte ghost header once and retain
per-leaf stored shapes/strides. A disposable WENO probe showed why it is not the
initial analysis strategy: only 542,736 logical header bytes across 22,614
records took 3.297 seconds cold but 26 ms warm because the headers are separated
by block payload pages. DAT-001 therefore validates the contiguous metadata and
minimum record spans only. DAT-003 preflights exact ghost/layout records for the
unique blocks in each request before destination mutation. An eager index may
reopen only if warm repeated header I/O dominates a real native consumer while
cold sparse time to first result remains protected.

For payload transfer, per-cell reads minimize surplus bytes but multiply system
calls. Reading a complete field for every selector occurrence repeats duplicate
fields. A WENO feasibility probe for selected interiors and `[6,4,6]` found:

| Unique blocks | Per occurrence | Unique fields | Exact contiguous runs |
| ---: | ---: | ---: | ---: |
| 4 | 12 calls / 49,152 B / 0.036 ms | 8 / 32,768 B / 0.028 ms | 4 / 49,152 B / 0.025 ms |
| 112 | 336 / 1,376,256 B / 0.946 ms | 224 / 917,504 B / 0.775 ms | 112 / 1,376,256 B / 0.656 ms |
| 512 | 1,536 / 6,291,456 B / 4.87 ms | 1,024 / 4,194,304 B / 4.41 ms | 512 / 6,291,456 B / 3.87 ms |

DAT-003 deduplicates block and field selectors call-locally. For a full stored
interior it coalesces only consecutive selected field IDs, reading their exact
bytes. For partial boxes or saved ghosts it reads one minimal contiguous
x-fast envelope per unique block/field and copies a strided view. Envelope
overread and useful bytes are distinct benchmark metrics. A broader min/max
field span is rejected because it silently reads unrequested field bytes.

The caller owns and closes the file descriptor. Position-independent reads do
not change its cursor and permit concurrent callbacks over immutable state.
Path-open callbacks add repeated open/stat cost; an owned handle/context object
would introduce lifecycle before the dataset layer needs it. The borrowed-fd
boundary is both explicit and directly reusable by that later owner.

## Decomposition Records

### DAT-001

- Responsibility: decode one spatially 3D v5 snapshot's format metadata into a
  safe canonical file index.
- Owned decisions: byte-order detection, exact v5 header/tree layout, native
  scalar conversion, file identity, count/offset arithmetic, and minimum record
  bounds.
- Non-owned: FST meaning, geometry/periodic support policy, exact record ghosts,
  payload transfer, selection, cache, halo, numerical work, and fd ownership.
- Inputs/outputs: borrowed open fd; frozen scalar/tuple record plus owned native
  arrays for forest flags, levels, zero-based coordinates, and offsets.
- Mutation/ownership: fd cursor/input unchanged; output owns arrays; nothing is
  retained outside the returned index.
- Access/reach: position-independent metadata reads only; no block record page
  is touched.
- Reference: upstream v5 byte layout, independent byte-crafted LE/BE fixtures,
  current v5 metadata on tdm/WENO, and exact offsets/file sizes.
- Producer/consumer: caller-opened immutable file -> DAT-002/DAT-003.
- Performance: `cold/control`; `O(K+L)` decode/allocation and exact metadata
  bytes, with no payload read.
- Five questions: yes/yes/yes/yes/yes.

### DAT-002

- Responsibility: prove DAT leaf rows are exactly one canonical FST leaf order
  and return that forest binding.
- Owned decisions: root shape from counts, MOR/FST reconstruction, maximum-level
  admissibility, and exact stored level/coordinate equality.
- Non-owned: byte parsing, payload offsets/values, geometry/periodic support,
  reader/cache, topology, and numerical work.
- Inputs/outputs: borrowed unchanged DAT-001 index; copied source provenance,
  owned MOR maps, and `RefinedForest` arrays grouped in one immutable tuple.
- Mutation/ownership: input unchanged; outputs caller-owned; nothing retained.
- Access/reach: contiguous metadata only; no file I/O or payload.
- Reference: FST-001/002 plus stored leaf levels and zero-based coordinates.
- Producer/consumer: DAT-001 -> DAT-003 and RHE/RPS metadata composition.
- Performance: `cold/control`; existing MOR/FST complexity/allocation plus one
  `O(L)` equality scan.
- Five questions: yes/yes/yes/yes/yes.

### DAT-003

- Responsibility: implement STO-003 selected interior reads from one validated
  non-staggered v5 file lifecycle.
- Owned decisions: selected record-header preflight, block/field deduplication,
  exact byte envelopes/runs, saved-ghost translation, endian conversion, and
  x-fast-to-canonical copy.
- Non-owned: metadata/FST meaning, selection planning, persistent cache,
  fd/path ownership, geometry/PBC/topology, halos, sampling, and parallel policy.
- Inputs/outputs: borrowed fd plus unchanged matching DAT-001 index and
  provenance-bearing DAT-002 binding;
  returned frozen `BlockReader`; per-call STO metadata and caller destination.
- Mutation/ownership: only declared destination boxes; ordinary API/selected-
  record errors precede mutation; external read/change and later resource/
  allocation failures may leave earlier unique block/field writes.
- Access/reach: explicit interior boxes; no numerical halo reach. Call scratch
  is `O(S+F)` metadata plus at most one consecutive-field run or one field
  envelope, never a chunk/full-file payload.
- Reference: independent raw address formula, current eager v5 reader, and
  array-reader substitution through RHE/RPS.
- Producer/consumer: DAT-001/002 -> STO-003 -> RHE/RPS, then local-field and
  streamline vertical slices.
- Performance: `milestone-workflow`; cold/warm time, first result, pread calls/
  bytes, useful/read/support amplification, scratch, faults/RSS, and composed
  runtime/memory.
- Five questions: yes/yes/yes/yes/yes.

## Completion And Reopen Gates

The group requires byte-crafted little/big-endian and corruption cases; exact
tdm metadata/payload equality; exact WENO refined metadata with explicit
staggered-reader rejection; saved-ghost and partial-box addressing; repeated/
reordered IDs and IEEE bits; DAT/FST binding; synthetic refined non-staggered
RHE/RPS array/native equality; and a streamed temporary non-staggered WENO
regular-field bridge for real refined values when the fixture is available.

The bridge supplements but does not erase the repository's lack of a native
real refined non-staggered fixture. It repacks only actual regular WENO field
bytes, preserves forest/tree/geometry, and excludes its setup time from reader
timings. The M1 horizon review must keep that limitation explicit.

Reopen eager record metadata, mmap, owned file lifecycle, asynchronous I/O,
payload cache, broader field-span reads, or parallel preads only from measured
native local-field/streamline bottlenecks. Full v3/v4/2D/staggered parsing and
writing remain their roadmap milestones rather than modes in this reader.

Independent contract review approved the three boundaries and selected I/O
strategy after requiring explicit DAT-002 source provenance at DAT-003,
separation of request-derived and record-derived preflight claims, and raw-bit
endian evidence. A direct NumPy byte-order probe preserved `+0`, `-0`, both
infinities, signed quiet/signaling NaNs, and their payload bits; production
tests freeze those exact patterns and scratch accounting includes any
byteswap/conversion temporary.
