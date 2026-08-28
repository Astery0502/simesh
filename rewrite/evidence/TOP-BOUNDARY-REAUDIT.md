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
| FST-002 | Validate caller-supplied FST-001 flat artifacts. | Preorder/root/child/level/coordinate/leaf-map conformance and exact max level. | Contact, balance, geometry, cache, storage. | yes/yes/yes/yes/yes | next proposed capability |
| TOP-002 | Return the raw adjacent covering node for explicit leaf/direction queries. | Physical detection and root/child descent to source level. | Conformance, allowed gap, cache/kinds, support/value policy. | yes/yes/yes/yes/yes | proposed after FST-002 |
| BAL-001 | Enforce global all-touch Cartesian 3D two-to-one balance. | `abs(level_a-level_b)<=1` over face/edge/corner contacts. | Contact discovery, cache representation, halo operations. | yes/yes/yes/yes/yes | proposed after TOP-002 |
| TOP-003 | Optionally materialize a compact six-face cache for admitted forests. | Face order, kind codes, node-ID representation, retained-memory choice. | Raw lookup, balance, edges/corners, support/value policy. | yes/yes/yes/yes/yes if retention is benchmark-justified | optional after BAL-001 |

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
TOP-002 + BAL-001 -> later REL-001 -> STO-004
```

FST-002 and TOP-002 are now complete with independent references, lifecycle/
corruption coverage, real-tree/current evidence, and allocation-free scaling.
BAL-001 is the next executable capability and owns only the global all-touch
level-gap policy.
