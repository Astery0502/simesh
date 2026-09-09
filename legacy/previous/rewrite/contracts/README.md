# Contracts

Create a contract only when its capability becomes active. A contract is the
stable specification selected after any needed design exploration. It is a
short statement of meaning and boundary, not a defensive specification.

A useful contract normally contains:

```text
Concept and responsibility
User workflow and immediate producer/consumer
Inputs and outputs
Domain semantics and concrete host/device buffer protocol
Layout, required input access, and produced valid region
Ownership and allowed mutation
State lifetime, aliasing, and supported failure behavior
Exact or numerical strategy, determinism scope, and comparison behavior
Adequate evidence: direct inspection, existing checks or focused core tests as needed
Relevant performance/resource evidence when the behavior or claim depends on it
```

Keep simple contracts proportional; link shared definitions rather than copying
them. Existing contracts retain their strict arithmetic, layout, and failure
guarantees. The distinction between semantics, buffer protocols, and strategy
guides new work and does not relax old contracts. New precision, arithmetic, or
device interfaces need explicit conformance before substitution. A contract is
not a requirement to test each metadata field, quantity formula or internal step.

Update an existing contract when the concept becomes clearer. Direct inspection
and relevant core evidence are the default; [independent review](../WORKFLOW.md#independent-review-and-autonomy)
is optional for a specific unresolved issue. Exploratory questions use WORKFLOW's single
design record rather than another contract template.
