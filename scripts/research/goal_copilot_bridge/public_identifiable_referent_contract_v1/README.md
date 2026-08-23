# Public Identifiable Referent Contract V1

状态：`C0_C1_CONTRACT_MECHANICS_READY / REFERENCE_IMAGE_UNIQUE / PUBLIC_PRIVATE_FIREWALL / NO_COHORT / NO_BASELINE / NO_ALGORITHM`

This package freezes the user-visible goal before episode observations, candidates, provider output, or outcomes. It
then separates the provider-public contract from an evaluator-private physical-instance lock.

V1 supports:

- `REFERENCE_IMAGE_INSTANCE`: always `UNIQUE`; the public image must either isolate one instance or expose a public
  target region. Optional language is supplementary recognition evidence, never identity authority.
- `LANGUAGE_REFERRING_EXPRESSION`: may be `UNIQUE`, `SET_VALUED`, or `AMBIGUOUS`.
- private physical IDs, source-native or independently reviewed identity binding, one world anchor per legal instance,
  and later per-frame visibility/region truth that is hash-bound to the pre-observation lock.

Teacher/model consensus cannot create physical identity authority. `AMBIGUOUS` carries no legal target, and
`NOT_VISIBLE` carries no target region. Public receipts recursively reject evaluator-private field names and never
expose physical instance IDs or world anchors.

The implementation deliberately leaves `cohort_freeze_authorized`, `passive_baseline_authorized`, and
`algorithm_authorized` false. It does not implement IEVE, Active Referent Search, detector changes, identity matching,
control, or product integration.

```powershell
python -m scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1.contract `
  --freeze-bundle <new-freeze-bundle.json> `
  --public-output <new-provider-public-receipt.json> `
  --private-output <new-evaluator-private-identity-lock.json>

python -m unittest `
  scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1.test_contract
```
