# Design Notes

Create a design note only for a capability with meaningful alternatives or
trade-offs. Use the single [WORKFLOW design record](../WORKFLOW.md#one-design-record)
for workload, decision ownership, alternatives, cost, evidence, and reopen
conditions. Shared group reasoning is recorded once and linked by members;
DECOMPOSITION's questions do not require a second form.

Design notes are exploratory and may evolve. The stable behavior chosen for
implementation belongs in `../contracts/`. Simple capabilities should skip a
separate design note and use one concise contract.

## User-Requested Cross-Workflow Exploration

[Analysis lifetimes and execution sketches](../../docs/analysis-core/lifetime-sketches.md) compares
ownership, retained state and optionality costs across three usage sequences.
It records exploratory combinations requested before choosing a capability;
it is not a contract or an active implementation group.

[Three pipeline result contracts](../../docs/analysis-core/pipeline-results.md) incorporates the
corrected F/D/L priorities and operating scale. It supersedes the earlier
repeated-sampling phase suggestion, with open numerical and delivery decisions.
