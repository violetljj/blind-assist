# Public Identifiable Referent Contract V1

状态：`C0_C1_CONTRACT_MECHANICS_READY / C2_SMALL_ROSTER_MATERIALIZABLE_7_OF_7 / VISIBLE_ONLY_PROBE_20_FOUND_16_SAME_INSTANCE_4_DISTRACTOR_1_ABSTAIN / ORACLE_COMPETITION_ORDER_COUNTERBALANCED / DINOV2_LOCAL_APPEARANCE_TARGET_OUTRANKS_13_OF_17 / TWO_REFERENCE_ZERO_RESCUE_THREE_COLLATERAL / TLESS_DINO_BASELINE_27_OF_30 / PDM_RESCUE_1_COLLATERAL_4_CONTROL_4_OF_6 / PDM_UNARY_REJECTED / NEAR_IDENTITY_V0_RESCUE_4_COLLATERAL_17_CONTROL_1_OF_18_COVERAGE_5_OF_135 / SPATIAL_LAYOUT_BASELINE_702_OF_900_LAYOUT_558_OF_900_RESCUE_74_COLLATERAL_218 / PASSIVE_SINGLE_REFERENCE_RGB_EXACT_INSTANCE_MAINLINE_STOP / SEMANTIC_ANCHOR_V1_16_OF_16_ZERO_WRONG_LOCK_4_OF_4_REACQUISITION_CONTROLLED_DERIVED_DEMO / PUBLIC_PRIVATE_FIREWALL / RELIABLE_VERIFIER_NOT_ESTABLISHED / NO_NONE_AUTHORITY`

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

## Matched two-reference information-gain probe

`dinov2_two_reference_matched_probe.py` is an additive Development adapter; it does not alter the singular-reference
V1 schema or any consumed C2 artifact. Before any new RGB GET, it deterministically excluded every C2 reference/later
filename, the seven old target IDs, and every same-class distractor ID listed in the C2 roster. Five reused SUN3D
capture sources then supplied five new physical target instances, ten reference images, and fourteen new
same-class-competition frames. The target and frame units are disjoint from C2, but the capture sources are reused;
the evidence is therefore Development, not source-new Confirmation.

Both arms run in one matched DINOv2-S execution. `single_reference` is the exact existing R1 symmetric local score.
`two_reference` applies one predeclared exemplar-set rule independently to each candidate:
`max(score(R1, candidate), score(R2, candidate))`. There is no threshold, training, alternate aggregation, or
model/crop/layer/fusion sweep. Raw scores are written and hashed before the evaluator-private target slot is read.

The frozen run observed:

```text
single_reference       14 / 14 target outranks
two_reference          11 / 14 target outranks
paired transitions      0 rescue / 3 collateral
net target-rank delta  -3
target-margin delta     1 positive / 4 zero / 9 negative
median margin delta    -0.0380834
```

R2 supplied the maximum distractor score in `9/14` pairs but the maximum target score in only `2/14`. The naive
two-exemplar OR rule therefore enlarged same-class distractor support more often than target support. This rejects
that aggregation as a reliable verifier and does not establish multi-view information gain. It also does not prove
that a single reference is information-complete: the new cohort had no single-reference error denominator to rescue.
Do not tune another aggregation on these opened outcomes; a future experiment would need a materially different,
predeclared distinctive-anchor/correspondence representation or a new hard-error cohort.

Evidence:

- `artifacts.local/evidence/public-identifiable-referent-dinov2-two-reference-matched-v0/frozen-roster-private.json`
- `artifacts.local/evidence/public-identifiable-referent-dinov2-two-reference-matched-v0/materialized/materialization-report.json`
- `artifacts.local/evidence/public-identifiable-referent-dinov2-two-reference-matched-v0/run-20260824T045037+0800/final-report.json`

Focused mechanics test:

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest `
  scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1.test_dinov2_two_reference_matched_probe
```

## Fresh-source hard-error unary PDM probe

`pdm_hard_error_unary_probe.py` freezes a source-disjoint T-LESS/BOP19 Development cohort before RGB materialization;
its 1280x1024 camera-metadata area gate became a stricter >=3.37% gate on the materialized 720x540 raster. The unchanged
DINOv2-S baseline produced `27/30` target outranks and six frozen hard pairs (three wrong plus three
lowest-margin correct) with six matched controls. One published PDM PerMIR configuration then yielded `1` rescue,
`4` collateral errors, and control retention `4/6=66.7%`; it fails the frozen `rescue > collateral` and `>=80%` gate.
Eight target-absent scores remain `NOT_EVALUABLE_NO_FROZEN_NONE_THRESHOLD`. Evidence is under
`artifacts.local/evidence/public-identifiable-referent-pdm-hard-error-unary-v0/`; focused tests are `test_pdm_*.py`.

## Read-only residual identity failure-layer audit

The audit in
`docs/research/goal-copilot/PUBLIC_IDENTIFIABLE_REFERENT_IDENTITY_FAILURE_LAYER_AUDIT_V0_RESULT_2026-08-24.md`
reads only the frozen DINO local, two-reference, and PDM reports plus already-materialized images, bboxes, and native
instance truth. It invokes no model and creates no new crop, feature, score, or threshold. Across the 12 frozen
hard/control pairs it assigns `NEAR_IDENTITY_REPRESENTATION_COLLAPSE=4`, `LOCAL_LAYOUT_INFORMATION_LOST=3`, and
`UNKNOWN=5`; no unknown is converted into pixel insufficiency or a background/quality claim. The unique routing
terminal is `LEARNED_NEAR_IDENTITY_REPRESENTATION_JUSTIFIED`, while `NEARID_NOT_RUN`, `NO_P1`, and
`DEFAULT_APP_UNCHANGED` remain explicit.

## Near-identity hard-negative unary V0

The separately authorized protocol is frozen before CORe50 download or model execution in
`docs/research/goal-copilot/NEAR_IDENTITY_HARD_NEGATIVE_UNARY_V0_PROTOCOL_2026-08-24.md`. The experiment keeps the
same frozen DINOv2-S backbone in both arms and trains only one fixed 384→256→128 projection head on source-, category-,
and physical-object-disjoint data. Test pixels remain sealed until the head, both absence thresholds, and quality
cutoffs are hash-bound. PDM, layout, fusion, multiple references, threshold sweeps, P1, and App wiring are forbidden.

The consumed run produced `4 rescue / 17 collateral`, retained only `1/18` baseline-correct controls, and accepted
only `5/135` target-present pairs. Its zero test absence false accepts therefore fail the non-trivial coverage gate and
do not establish a `NONE` authority. Result:
`docs/research/goal-copilot/NEAR_IDENTITY_HARD_NEGATIVE_UNARY_V0_RESULT_2026-08-24.md`.

## Spatial-layout identity verification V0

The fresh Washington RGB-D Object Dataset run fixed all 300 physical instances, 51 categories, video-1 references,
video-4 candidates and 900 same-instance-vs-same-class-distractor pairs before pixel decode. Both arms shared frozen
DINOv2-S patch tokens. The analytic layout arm used only reciprocal matches, bilateral coverage/dispersion,
local-neighborhood preservation, Procrustes residual and conflict consistency; it had no training, threshold, `NONE`,
fusion or second reference.

Baseline target outrank was `702/900`; layout was `558/900`, with `74 rescue / 218 collateral` and `484/702=68.9%`
control retention. Stable-distractor recovery was `29/69=42.0%`; direction invariance was `1800/1800` and candidate
permutation invariance `900/900`. The valid negative closes this passive single-reference RGB exact-instance mainline;
open-set calibration, P1 and App remain unrun. Result:
`docs/research/goal-copilot/SPATIAL_LAYOUT_IDENTITY_VERIFICATION_V0_RESULT_2026-08-24.md`.

## Active distinctive evidence acquisition V0

`active_distinctive_evidence_acquisition_v0.py` changes enrollment to a three-view sweep and builds stable local
anchors. SIFT audits whether the sweep contains repeatable points; stable DINOv2 patches then vote only for a candidate
that explains an anchor with absolute support and a margin over every competitor. It never uses max-over-reference
scores, and the injected lost step requires a fresh anchor-gated relock rather than tracker identity.

The curated Development demo contains two Wikimedia storefront targets plus Washington cereal-box and keyboard targets:
4 targets, 16 target-present decisions, and 4 lost/reacquisition opportunities. The final active arm exactly matched
the passive single-reference DINO comparator: top-1 `11/16`, wrong lock `9/20`, reacquisition `3/4`, with zero delta in
every scenario. Appearance-derived anchors therefore stop; OCR/logo/marker evidence was not evaluated because no
executable OCR runtime currently exists. Result:
`docs/research/goal-copilot/ACTIVE_DISTINCTIVE_EVIDENCE_ACQUISITION_V0_RESULT_2026-08-24.md`.

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest `
  scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1.test_active_distinctive_evidence_acquisition_v0
```

## Semantic Distinctive Anchor V1

`semantic_distinctive_anchor_v1.py` reuses the consumed Active Distinctive V0 sequence and frozen passive receipt, then
adds independently decodable goal evidence. RapidOCR is isolated under `artifacts.local`; semantic decisions lock only
on exactly one matching OCR substring, ArUco ID, or homography-verified sign patch and otherwise abstain. The controlled
derived demo changes available information rather than tuning appearance and moves top-1 `11/16 -> 16/16`, wrong locks
`9 -> 0`, and reacquisition `3/4 -> 4/4`. Only the Starbucks OCR rows are naturally occurring; the sign/code/marker rows
are disclosed canaries. No Android/default-App/P1 or general identity claim follows.

```powershell
pwsh -NoProfile -File `
  scripts/research/goal_copilot_bridge/public_identifiable_referent_contract_v1/install_semantic_anchor_v1_runtime.ps1

E:\codex-tools\bin\blindassist-python.cmd -m unittest `
  scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1.test_semantic_distinctive_anchor_v1
```

Result: `docs/research/goal-copilot/SEMANTIC_DISTINCTIVE_ANCHOR_V1_RESULT_2026-08-24.md`.
