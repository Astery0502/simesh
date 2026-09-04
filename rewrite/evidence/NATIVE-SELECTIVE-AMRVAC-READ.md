# Native Selective AMRVAC Read Evidence

## Outcome And Boundary Review

DAT-001, DAT-002, and DAT-003 provide the first native file-to-core path. A
borrowed regular-file descriptor is decoded with position-independent reads,
the stored leaf rows are proven against a canonical FST artifact, and selected
ordinary block interiors flow through the existing STO-003 reader boundary.
RHE-001 and RPS-001 consume that reader without backend branches or retained
payload.

Independent review approved the three-way split after two material corrections.
First, DAT-002 now copies source-file identity into its binding and DAT-003
requires exact index/binding provenance, so an unrelated same-sized forest
cannot be paired with payload rows. Second, request-derived validation is
explicitly pre-I/O, while record-derived geometry/address validation occurs
after all selected 24-byte headers are read but before any payload mutation.
Review also required raw endian tests for signed zero, infinities, signed quiet
and signaling NaNs, explicit conversion scratch accounting, and external
resource-failure semantics; all are covered.

The production parser accepts exactly spatially 3D v5 metadata in little or big
endian. This intentionally does not copy the current reader's incorrect version
thresholds: current `get_header` reads v5-only grid fields for v4 and reads the
v4 tail unconditionally for v3. Correct v3/v4 branches remain M5 work rather
than a false compatibility claim.

## Correctness, Failure, And Integration

The final build, 125 focused DAT/STO/RHE/RPS tests, and all 837 accumulated
rewrite tests pass. The focused evidence includes:

- independently byte-crafted little- and big-endian headers/trees, exact
  unaligned cursors, short/corrupt sections, checked int64 counts/offsets,
  forest counts, coordinates, version/dimension failures, fd cursor preservation,
  and file-identity changes;
- raw-bit header time/parameter and payload patterns covering `+0`, `-0`, both
  infinities, signed qNaN/sNaN payloads, with native output bits identical;
- exact MOR/FST construction and stored level/coordinate equality, copied
  provenance, no input retention, and deliberate geometry/periodicity
  independence;
- lazy all-selected record-header preflight, exact record ends, asymmetric saved
  ghosts, translated partial boxes, minimal envelopes, complete consecutive
  field runs, repeated/reordered block/field selectors, untouched cells, and
  two-run scratch lifetime;
- no I/O for empty or ordinary-invalid requests; complete destination
  preservation for a malformed/short later selected header; documented partial
  writes for later payload failure or final identity change; closed/reused fd
  behavior and alias rejection;
- 120 additional randomized LE/BE raw-bit transfers against an independent
  address oracle;
- exact metadata/FST comparisons for every available tdm, WENO, and
  `reference/bw.dat` fixture; spherical/periodic bw is storage-only evidence;
- bitwise tdm selected payload equality with current eager reading; and
- a 71-leaf synthetic non-staggered refined file whose direct translated reads,
  RHE halos, and repeated zero/trilinear values are bitwise equal between native
  and array readers across bounded/full capacities.

Commands:

```text
.venv/bin/python rewrite/build_ext.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_dat_001.py rewrite/tests/test_dat_002.py rewrite/tests/test_dat_003.py rewrite/tests/test_dat_003_integration.py rewrite/tests/test_sto_003.py rewrite/tests/test_rhe_001.py rewrite/tests/test_rps_001.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests
```

## Real tdm Selective Read

The standard Apple arm64/Python 3.11.14 run uses five repetitions. Real
`tdm.dat` is v5 Cartesian 3D, nonperiodic/non-staggered, 27 level-one leaves,
seven fields, `10^3` interiors, and 1,513,724 bytes. DAT index plus forest bind
takes 0.249 ms median: 0.019 ms open, 0.142 ms parse, and 0.088 ms bind. It reads
exactly the 1,076-byte metadata prefix and retains 1,214 index-array plus 4,344
binding-array bytes; the native reader retains another 216-byte offset copy.

Fields `[4,5,6]` form one exact consecutive run per unique block. Every result
is bitwise equal to current `read_blocks_sequential`, which reads/materializes
all 648,000 selected-field bytes in 0.984 ms median.

| Requested leaves | First after open | Warm median | Preads | Header bytes | Payload bytes | Useful bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.241 ms | 0.120 ms | 8 | 96 | 96,000 | 96,000 |
| 16 | 0.402 ms | 0.334 ms | 32 | 384 | 384,000 | 384,000 |
| 27 | 0.633 ms | 0.541 ms | 54 | 648 | 648,000 | 648,000 |

The selected path therefore reduces payload bytes exactly with selection and is
also 1.82x faster than current eager reading at full coverage on this warm
fixture. A separately traced 16-leaf call retains 3,432 Python bytes and peaks
at 33,517 bytes; its largest payload buffer is one 240,000-byte three-field run,
which `tracemalloc` does not fully attribute to Python allocation internals.

## Real Refined WENO Regular-Field Bridge

The only real refined Cartesian fixture is the 1,045,232,320-byte staggered
WENO snapshot. DAT-001/DAT-002 parse and bind its 25,844 nodes/22,614 leaves at
levels three through six in 32.76 ms median and DAT-003 rejects its original
staggered records before reader creation.

For payload evidence, the standard benchmark streams only the real regular
`b1,b2,b3` bytes into a temporary non-staggered v5 file while preserving the
forest, tree, bounds, geometry, and field bits and rebuilding offsets. This
setup reads 278,423,568 bytes, takes 6.90 seconds, and creates a 279,069,936-byte
temporary file; setup is excluded from native reader timings. The target index
and binding take 32.96 ms. Current eager reading of the same three regular
fields from the source takes 5.49 seconds and materializes 277,880,832 bytes.

The representative query has 2,048 points clustered in four refined leaves,
field positions `[0,1,2]`, capacity eight for zero order and 128 for trilinear.
Native and resident-array values and RPS stats are bitwise/exactly identical.

| Path | First after open | Warm median | Header/payload preads | Payload bytes | Loads/amplification | RPS managed bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native zero | 2.835 ms | 0.643 ms | 4 / 4 | 49,152 | 4 / 1.00 | 82,040 |
| array zero | 0.971 ms | 0.573 ms | 0 / 0 | resident | 4 / 1.00 | 82,040 |
| native trilinear | 21.539 ms | 4.448 ms | 54 / 54 | 663,552 | 54 / 13.50 | 3,371,498 |
| array trilinear | 4.007 ms | 3.542 ms | 0 / 0 | resident | 54 / 13.50 | 3,371,498 |

Native warm overhead is 12.2% for zero order and 25.6% for trilinear versus an
already resident array. The latter exceeds the default 20% timing threshold but
is retained for the declared material trade-off: it avoids a 277,880,832-byte
resident field array and reads 5,653.5x/418.8x fewer payload bytes for zero/
trilinear. Index, binding, reader offsets, and trilinear managed workspace total
about 7.56 MB before caller points/results, over 36x smaller than that eager
payload alone. First-read latency exposes scattered page access honestly; warm
calls show the numerical/transfer steady state.

The process report records 173,260,800-byte peak RSS and 54,280,192-byte final
RSS on this run. Allocator/page-cache behavior makes those descriptive; exact
managed arrays and logical/physical bytes above are the architecture gates.

Benchmark command and ignored raw record:

```text
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/dat_003.py --profile standard --output rewrite/benchmark-results/dat-003-standard.json
```

## Retained Strategy And Reopen Triggers

DAT-001 stops at `offset_blocks`: it never scans separated record pages.
The rejected eager alternative read only 542,736 logical WENO ghost-header bytes
but took 3.297 seconds cold across 22,614 seeks versus 26 ms warm. DAT-003
instead validates only unique selected records, suppresses duplicate selectors,
coalesces exact consecutive full-field runs, and otherwise uses one minimal
per-field envelope. It retains no payload or cache and never changes fd position.

The borrowed-fd strategy and lazy headers are retained. The local-field and
streamline vertical slices are the reopen consumers for mmap/owned lifecycle,
eager warm record metadata, asynchronous/parallel preads, broader overread,
and persistent block cache. The measured 13.5x all-26 trilinear support
amplification reinforces the existing direction-projection/cache trigger.

This group does not close the M1 real-fixture limitation: the WENO bridge uses
real regular values but is derived from a staggered source, and no native real
refined non-staggered Cartesian file exists in the repository. The M1 horizon
review must retain that fact rather than relabel the bridge.
