# Public Identifiable Referent Contract V1

状态：`C0_C1_CONTRACT_MECHANICS_READY / C2_SMALL_ROSTER_MATERIALIZABLE_7_OF_7 / VISIBLE_ONLY_PROBE_20_FOUND_16_SAME_INSTANCE_4_DISTRACTOR_1_ABSTAIN / ORACLE_COMPETITION_ORDER_COUNTERBALANCED / DINOV2_LOCAL_APPEARANCE_TARGET_OUTRANKS_13_OF_17_HISTORICAL_WRONG_3_OF_4_CONTROLS_10_OF_13 / LOCAL_EVIDENCE_COMPLEMENTARY_NOT_SUFFICIENT / PUBLIC_PRIVATE_FIREWALL / NO_NOT_VISIBLE_EVIDENCE / RELIABLE_VERIFIER_NOT_ESTABLISHED`

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

## Oracle competing-identity diagnostic

`oracle_competing_identity_probe.py` is a separate, consumed Discovery diagnostic over the immutable visible-probe
inputs. It gives the fixed `GPT-5.6-Sol/high` provider two same-class 384x384 crops with equal 20% context and asks
only `A / B / CONTESTED / NEITHER`. The evaluator privately maps slots to the frozen target and distractor; native IDs,
truth labels, and correct position never enter the provider workspace. It is deliberately oracle-candidate and cannot
measure candidate generation or product performance.

The 17-pair run used all four historical wrong-instance cases plus 13 of 16 original correct cases with a real
same-class competitor. It observed `2/4` target selections in the historical wrong stratum and
`12/13 TARGET / 1/13 CONTESTED / 0/13 DISTRACTOR` in controls. The three controls without a same-class competitor
were excluded rather than given a fabricated negative.

Because all four historical results initially coincided with the target's A/B position, a separately marked post-hoc
counterbalance swapped only the two candidate images. Paired by physical instance, one case selected the target in
both orders, two selected the same distractor in both orders, and one selected slot B in both orders. Thus explicit
competition has partial signal and low observed collateral in the original order, but only `1/4` historical errors
has an order-robust rescue; a reliable verifier is not established. No belief, tracker, Active Search, candidate
generator, representation sweep, App integration, or `NOT_VISIBLE` claim follows from this diagnostic.

Local reports:

- `artifacts.local/evidence/public-identifiable-referent-oracle-competing-identity-v0/run-20260824T034449+0800/final-report.json`
- `artifacts.local/evidence/public-identifiable-referent-oracle-competing-identity-order-counterbalance-v0/run-20260824T034852+0800/final-report.json`

Focused mechanics tests:

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest `
  scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1.test_oracle_competing_identity_probe `
  scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1.test_visible_identity_probe
```

## Order-free DINOv2 local appearance probe

`dinov2_local_appearance_probe.py` consumes the same 17 oracle pairs without invoking a VLM. It reuses the exact
frozen `facebook/dinov2-small@ed25f3a` files and P1-A2 preprocessing identity: 224x224 ImageNet-normalized crops,
last-layer 16x16x384 patch tokens, and per-patch L2 normalization. Reference and candidate crops use the already fixed
20% square context, but scoring retains only patch centers inside the object region. No red or green annotations enter
the encoder.

Each candidate is scored independently. The score is the mean of reference-to-candidate and candidate-to-reference
mean nearest-patch cosine; only afterward does a strict greater-than comparison choose the higher raw score. There is
no threshold, training, augmentation, positional prompt, fusion, or model/crop/layer sweep. `run-config.json` contains
neither target position, physical/native IDs, nor baseline outcome. `raw-scores.json` is written before the private
evaluator maps A/B to target/distractor.

The one run observed `13/17 TARGET_OUTRANKS / 4/17 DISTRACTOR_OUTRANKS / 0 TIE`. Historical wrong-instance cases were
`3/4` target outrank: the robust-target and order-sensitive cases both ranked target first, while the two prior
stable-distractor cases split `1/2`. The rescued stable case had target margin `+0.01698`; the remaining stable error
had `-0.04007`. Original-correct controls were only `10/13`, with two near-tie negative margins and one `-0.06206`
error. Local evidence is therefore complementary but not sufficient as a standalone verifier. This run does not
authorize threshold/fusion search, belief, tracking, Active Search, candidate generation, App integration, or any
`NOT_VISIBLE`, safety, or product claim.

Local report:
`artifacts.local/evidence/public-identifiable-referent-dinov2-local-appearance-v0/run-20260824T042110+0800/final-report.json`.

Focused mechanics test:

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest `
  scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1.test_dinov2_local_appearance_probe
```
