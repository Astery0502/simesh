# TOP Boundary Re-Audit

## Result

The Phase A Red TOP-002 draft has been split before implementation.  Both
untracked drafts were preserved intentionally as
`designs/TOP-003-face-cache.md` and
`designs/TOP-003-contract-draft.md`; the latter is explicitly not a stable
contract.  No implementation or stable topology contract changed in this
checkpoint.

## Independent Decisions

| ID | Singular responsibility | Owns | Does not own | Five questions | Status |
| --- | --- | --- | --- | --- | --- |
| FST-002 | Validate caller-supplied FST-001 flat artifacts. | Preorder/root/child/level/coordinate/leaf-map conformance and exact max level. | Contact, balance, geometry, cache, storage. | yes/yes/yes/yes/yes | complete |
| TOP-002 | Return the raw adjacent covering node for explicit leaf/direction queries. | Physical detection and root/child descent to source level. | Conformance, allowed gap, cache/kinds, support/value policy. | yes/yes/yes/yes/yes | complete |
| BAL-001 | Enforce global all-touch Cartesian 3D two-to-one balance. | `abs(level_a-level_b)<=1` over face/edge/corner contacts. | Contact discovery, cache representation, halo operations. | yes/yes/yes/yes/yes | complete |
| TOP-003 | Optionally materialize a compact six-face cache for admitted forests. | Face order, kind codes, node-ID representation, retained-memory choice. | Raw lookup, balance, edges/corners, support/value policy. | yes/yes/yes/yes/yes if retention is benchmark-justified | deferred pending consumer evidence |

Later REL-001 owns directional operation records and mixed physical masks;
STO-004 owns support union/order/capacity/traversal.  They remain separate.

## Evidence Behind The Split

- Exact current recovery confirms physical/coarser/same/finer face meanings and
  fine-child order, but those are cache presentation rather than raw contact
  semantics.
- A synthetic forest can be face-balanced while an edge/corner gap is two;
  current connectivity can misclassify that deep diagonal as physical.  Global
  all-touch balance is therefore an explicit policy, not a topology side
  effect.
- Six `uint8` kinds plus six `int64` node IDs occupy `54*L` bytes.  The
  pre-audit `49*L` formula was incorrect.  No retained cache is essential:
  HPL/REL consumers can derive source sets from raw contacts and FST children.
- FST conformance has independent consumers in refined topology, geometry, and
  native I/O; duplicating recursive validation in each would create divergent
  safety boundaries.

## Dependency Order

```text
FST-001 -> FST-002 -> TOP-002 -> BAL-001
                              -> optional TOP-003 after BAL-001
TOP-002 + BAL-001 -> REL-001 (complete) -> STO-004
```

FST-002, TOP-002, and BAL-001 are complete with independent references,
lifecycle/corruption coverage, real-tree/current evidence, and no
size-dependent validation scratch.  BAL-001 rejects the known face-balanced
diagonal violation and owns only the global all-touch level-gap policy.
TOP-003 remains deferred because
no immediate consumer yet justifies `54*L` retained bytes over measured
on-demand contacts.  Refined geometry is complete independently; REL-001 is the
first concrete relation consumer and is now complete.  Its real workload finds
only 24.8% reduced-face records; a six-face cache costs 1,221,156 retained bytes
and cannot amortize its construction on one pass.  TOP-003 therefore remains
deferred until a repeated transfer/halo consumer supplies composed evidence.
