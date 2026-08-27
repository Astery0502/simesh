# Current Rewrite Checkpoint

## State

- Active milestone: Foundation leading to M0.
- Last completed capability: none.
- Current capability: FND-001, not started.
- Rewrite implementation: not created.
- Stable contracts: none beyond `CHARTER.md`.
- Unresolved differences: none.

## Decisions Already Established

- Compatibility means supported numerical behavior, not API or byte identity.
- The order is 3D level-1, 3D refined, 2D, then periodic meshes.
- Staggered data is not currently supported.
- Kernels are added through normal Cython rebuilds; no dynamic kernel system.
- Data larger than available memory is a design target.
- Performance is multi-objective.
- Agents may revise contracts autonomously through independent sub-agent review.
- The workflow deliberately avoids hashes and heavy governance.
- Completed capabilities become cohesive Git checkpoints on the continuing rewrite branch.
- Non-trivial capabilities use a short design note before their stable contract; simple capabilities go directly to a concise contract.
- A capability starts only after its dependencies are complete unless the ledger records a narrower exception.
- Every capability checkpoint runs the accumulated rewrite regression suite and stages only capability-related files.
- Optimization stops after a correct integrated implementation reaches a useful trade-off and further credible variants no longer materially improve it.

## Next Work

Specify FND-001 without implementing later layers. Establish only the layout,
index, ownership, mutation, and valid-region conventions required by the first
Morton, topology, storage, and operator consumers.

Before code is added, create the minimal isolated build and test path needed by
that capability. Do not design the complete future framework during FND-001.
