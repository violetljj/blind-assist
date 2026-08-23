# Public Identifiable Referent Contract V1

状态：`C0_C1_CONTRACT_MECHANICS_READY / C2_SMALL_ROSTER_MATERIALIZABLE_7_OF_7 / VISIBLE_ONLY_PROBE_20_FOUND_16_SAME_INSTANCE_4_DISTRACTOR_1_ABSTAIN / PUBLIC_PRIVATE_FIREWALL / NO_NOT_VISIBLE_EVIDENCE / NO_ALGORITHM`

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

The C0/C1 contract deliberately leaves `cohort_freeze_authorized`, `passive_baseline_authorized`, and
`algorithm_authorized` false. The separately frozen C2 adapter materialized one 7-source roster without provider or
model calls; it does not implement a passive baseline, IEVE, Active Referent Search, detector changes, identity
matching, control, or product integration.

```powershell
python -m scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1.contract `
  --freeze-bundle <new-freeze-bundle.json> `
  --public-output <new-provider-public-receipt.json> `
  --private-output <new-evaluator-private-identity-lock.json>

python -m unittest `
  scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1.test_contract `
  scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1.test_c2_small_roster
```

C2's one formal freeze/materialization has already been consumed. Do not rerun or overwrite it. Protocol and result:

- [`C2 protocol`](../../../../docs/research/goal-copilot/BLINDASSIST_PUBLIC_IDENTIFIABLE_REFERENT_C2_SMALL_ROSTER_PROTOCOL_V1_2026-08-24.md)
- [`C2 result`](../../../../docs/research/goal-copilot/BLINDASSIST_PUBLIC_IDENTIFIABLE_REFERENT_C2_SMALL_ROSTER_RESULT_2026-08-24.md)

## Visible-only passive identity probe

The separately user-authorized `visible_identity_probe.py` consumes the immutable C2 images without changing C2.
Each isolated Codex CLI call receives only the clean reference, a public target-region overlay, and one later image.
The evaluator privately reconstructs every native SUN3D instance in the later frame from the C2-bound annotation SHA.
A committed region is assigned by center containment and then highest IoU, with no score or success threshold.

The single observed `GPT-5.6-Sol/high` run completed all 21 calls: `FOUND=20`, `ABSTAIN=1`, and the 20 commits split
into `SAME_INSTANCE=16`, `SAME_CLASS_DISTRACTOR=4`, `UNRELATED_OBJECT=0`, `BACKGROUND=0`. Three of seven episodes
were same-instance correct in all three views. This is consumed, visible-only Discovery failure anatomy; it does not
measure `NOT_VISIBLE`, abstention calibration, navigation, safety, or product behavior, and it does not authorize an
algorithm or rerun. Local report:
`artifacts.local/evidence/public-identifiable-referent-visible-identity-probe-v0/run-20260824T033033+0800/final-report.json`.

Focused mechanics test:

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest `
  scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1.test_visible_identity_probe
```
