# Selected Refined Halo Completion Group

## Outcome And Members

For Cartesian 3D block extents of at least four cells per axis and requested
side reach no larger than half a block, an explicit ascending selected-primary
stream is read through one bounded functional reader and emitted through one
functional writer with every requested SAME, FINER, COARSER, pure-physical,
and mixed-physical halo cell complete.  Support slots remain interior-only and
no unselected primary is invented.

The dependency order is:

1. `CSP-001`: plan the complete CWP coarse slope-support rectangle;
2. `PWA-001`: apply explicit Cartesian physical widening;
3. `CWA-001`: assemble one explicit CWP/PRL coarse workspace;
4. `RHE-001`: execute bounded selected refined halos.

CSP and PWA are independent ready members.  CWA consumes both after their
focused and immediate-composition checks pass.  RHE consumes integrated CWA
plus the completed selected transfer-planning group.

## Exploration Coverage

Current approach: the canonical mesh allocates a coarse array for every leaf,
coarsens globally useful fine interiors, fills coarse physical boundaries,
pushes restriction/sibling values, then prolongs and performs a final physical
sweep.

Structurally different credible alternative: recursively close every CWP
rectangular fringe through a second topology walk, or weaken PRL to an explicit
seven-point cross-valid contract and leave unread rectangle corners undefined.

Most decision-changing uncertainty: whether CWP's one-cell axial slope reach
can leave the primary's all-26 REL/SPR source union.  A direction-tile probe
found that it can for `B=2`, and for reach above `B/2`, but not when every
`B>=4` and each side reach is at most `B/2`.  Under the selected gate a face,
edge, or corner COARSER row needs at most 18, 12, or 8 raw tiles; no required
tile has a FINER owner relative to the primary.

Reopen evidence or consumer: support recursion or a cross-valid PRL variant is
reopened by a required `B=2` workflow, wider/higher-order reach, weaker balance,
or measured CWA rectangle-fill cost.  Direction-subset projection is reopened
when the selected local-field consumer shows material all-26 overread.  It may
not append support after a reader call.

The selected approach keeps CWP and checked PRL validity unchanged, partitions
the complete required rectangle into bounded plain records, fills all
nonphysical bases before physical widening, and reuses one coarse scratch.
STO-004 dense traversal and HAL-002/SAM fused compatibility paths remain
unchanged.

During RHE's first all-action composition, the group reopened CWP's checked
accepted domain: its phase/sign condition belonged to the opposite-direction
restriction send schedule and rejected legal COARSER prolongation edge/corner
rows.  An independent material review over broad balanced-forest scans, every
WENO COARSER metadata row, and a bit-exact current dyadic target approved
removing that condition from CWP/CSP.  No numerical formula, output
representation, all-26 closure, 18-record bound, or prior accepted result changed.

## Analysis Algorithm-Selection Record

Primary analysis workflow and query shape: a sparse or coherent selected leaf
set, a small field set, and one complete rectangular refined halo shared by a
later local-field or refined sampler consumer.

Expected access density and locality: selected primaries may be sparse;
support clusters around each primary and repeats across adjacent primaries.

Reusable state and lifetime: conformed/balanced forest arrays live for the
snapshot; target boxes live for a compatible reach; chunk REL/slot/phase and
CSP plans live only until that chunk is written; one CWA scratch is reused for
the executor call.

Dominant expected cost: selected payload bytes and support amplification,
followed by restriction/prolongation and physical value work.  Fixed direction
and box planning is secondary.

Current and structurally different candidates: bounded pull-based selected
assembly versus current global coarse arrays; full CWP rectangular validity
versus an exact PRL cross-valid specialization.

Workload-specific metrics and consumer: requested/read/output bytes, support
amplification, reader/writer calls, time to first completed chunk, kind/tile
counts, PWA written cells, stage times, exact output, managed bytes, and peak
RSS for small/medium/full selected refined halos.

Deferred alternatives and reopen triggers: cache/reuse, direction-subset
projection, merged coarse tiles, cross-only workspace validity, and fused
application remain deferred until the complete consumer attributes a material
cost or the later local-field profile demonstrates avoidable reads.

## Performance Classes And Group Gate

- CSP is `cold/control`: bounded at 18 records, exact `O(1)` per CWP row, and
  has no size-dependent internal allocation.
- PWA is `hot-kernel`: measure candidate and written field-cells, bytes, checked
  and unchecked runtime, and its CWA/RHE consumers.
- CWA and RHE are `composition-only`: measure them together in complete
  selected refined halo execution.

Closing requires one rewrite build; focused CSP/PWA/CWA/RHE plus producer and
consumer tests; the accumulated rewrite suite; safe current refined comparisons;
bounded/full-capacity equivalence; a standard PWA kernel and selected-halo
composition run; exact reader/writer and memory accounting; and one concise
group evidence report.  Raw runs remain under the ignored benchmark-results
tree.

Closing commands are:

```text
.venv/bin/python rewrite/build_ext.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_csp_001.py rewrite/tests/test_pwa_001.py rewrite/tests/test_cwa_001.py rewrite/tests/test_rhe_001.py rewrite/tests/test_cwp_001.py rewrite/tests/test_pbc_001.py rewrite/tests/test_spr_001.py rewrite/tests/test_rsl_001.py rewrite/tests/test_rph_001.py rewrite/tests/test_slb_001.py rewrite/tests/test_frp_001.py rewrite/tests/test_rst_001.py rewrite/tests/test_prl_001.py rewrite/tests/test_sto_003.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/rhe_001.py --dat data/weno509_sub_0000.dat --capacity 128 --small 32 --medium 512 --fields 3 --block 4 --halo 2 --warmups 1 --repeats 5 --pwa-repeats 9 --output rewrite/benchmark-results/rhe-001-standard-final.json
```
