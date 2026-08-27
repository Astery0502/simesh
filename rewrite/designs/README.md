# Design Notes

Create a design note only for a capability with meaningful alternatives or
trade-offs. A useful note records:

- the domain problem and why it belongs in the current layer;
- the few credible approaches;
- implications for composition, memory use, and performance;
- the selected direction and the evidence that favored it;
- questions intentionally deferred to a later capability.

Design notes are exploratory and may evolve. The stable behavior chosen for
implementation belongs in `../contracts/`. Simple capabilities should skip a
separate design note and use one concise contract.
