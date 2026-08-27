# Contracts

Create a contract only when its capability becomes active. A contract is the
stable specification selected after any needed design exploration. It is a
short statement of meaning and boundary, not a defensive specification.

A useful contract normally contains:

```text
Concept and responsibility
Inputs and outputs
Layout and valid region
Ownership and allowed mutation
Required halo or access pattern
Exact or numerical comparison behavior
Immediate producer and consumer
Focused correctness evidence
```

Update an existing contract when the concept becomes clearer. Stable contract
changes follow the independent sub-agent review in `../WORKFLOW.md`.
