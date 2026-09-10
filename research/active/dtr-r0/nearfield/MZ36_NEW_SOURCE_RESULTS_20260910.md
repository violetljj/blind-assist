# MZ36: frozen correction transfer on new XY placements

2026-09-10 EXPLORE. Frozen new-XY comparison: 380/400 frames in 76/80 groups admitted. MZ5/MZ28/MZ30/MZ35 FP=6/6/4/3, FN=10/2/2/2. MZ30: PASS, removed2FP, lost0TP; MZ35: PASS, removed3FP, lost0TP. No refit or calibration.

[Protocol](MZ36_NEW_SOURCE_PROTOCOL_20260910.md), [source preparation](mz36_source_prepare.py),
[raw-frame inference](mz36_frozen_inference.py), [admission/scoring](mz36_evaluate.py), [audit](mz36_audit.py).

## Source and denominator

Both dense_candidate_05/06 regions were previously used for a size experiment. New camera XY are at least6m from every prior proposed/accepted XY and from each other. All30 empty views passed sparse floor and visual source review; the first10 ranked sites per region were frozen before400 full frames were captured. Four TRAIN fixture families use rank offset1000. This is new XY/fixture Development with shared CitySample assets and previously consumed regions, not independent-region or natural data. Static/hovering vehicle artifacts and road-surface camera sites remain disclosed source limitations.

Native admission preserves all400 attempted frames and80 site/family groups: 400 native-labeled, 380 admitted and 20 excluded frames, 76 accepted groups. UNKNOWN per query=[20, 20, 20, 20]. Group exclusions={'NATIVE_FIXTURE_INTENT_MISMATCH': 4}. Any source/floor/fixture-intent/observation failure excludes the entire group; failed cases are retained. Native MZ events use crop=False and actual range bins, not the fixture's intended range. Negative event bits are not safety CLEAR.

## Fixed paired effect

| Method | All-four exact frames | FP bits | FN bits | Exact complete groups |
|---|---:|---:|---:|---:|
| MZ5 | 365/380 | 6 | 10 | 64/76 |
| MZ28 | 372/380 | 6 | 2 | 69/76 |
| MZ30 | 374/380 | 4 | 2 | 71/76 |
| MZ35 | 375/380 | 3 | 2 | 71/76 |

MZ28 versus MZ5: 8 addedTP, 0 addedFP, 0 lostTP. These are actual admitted query bits, not all-obstacle recall.

| Correction versus MZ28 | FP removed | TP lost | Additions preserved | Fixed transfer criterion |
|---|---:|---:|---|---|
| MZ30 | 2 | 0 | True | PASS |
| MZ35 | 3 | 0 | True | PASS |

The criterion requires removing FP while losing zero admitted MZ28 TP and preserving additions. It does not require fitting or reaching an error count selected from these new outcomes. Raw paired changes and identities are retained in paired-changes.json. Both selectors start from MZ28 independently.

| Region | Admitted frames | MZ5 FP/FN | MZ28 FP/FN | MZ30 FP/FN | MZ35 FP/FN |
|---|---:|---|---|---|---|
| dense_candidate_05 | 180 | 5/2 | 5/1 | 4/1 | 3/1 |
| dense_candidate_06 | 200 | 1/8 | 1/1 | 0/1 | 0/1 |

## Interface and audit

The raw adapter initially had numerical differences on8 consumed DEV frames although128 task bits matched. Padding alone did not fix this. Restoring the original RGB CUDA settings and using its16-image batch shape yielded exact final logits and256 task bits on the same-source complete16-frame historical batch, without relaxing tolerance. Both earlier failures and runner snapshots remain. Across these checks20 unique old DEV frames were used; no new-source outcome chose backend settings.

Independent scalar audit passes 6080 scored task bits, all paired counts, UNKNOWN/complete-group denominators, selected XY separation and all input/output/pre-capture model hashes. Native labels also passed a4-frame historical MZ15 regression. No fit, cutoff calibration, bank rebuild or model replacement occurred. Raw inference total=19.260s on desktop CUDA for380 frames; this includes ideal packet synthesis and does not measure a physical sensor or phone latency.

## Remaining errors and next decision

Both remaining misses are far crossbars with a correct positive modality that
the mean ensemble suppresses. At dense05 site002 BOTH/BODY_FAR, RGB+2.313848
and ToF-2.764154 yield MZ5-0.225153; original geometric support exists but the
packet availability restriction removes it. At dense06 site004
HEAD_ONLY/HEAD_FAR, RGB-7.625022 and ToF+3.141601 yield MZ5-2.241711; restricted
support remains but the local rank margin is-0.354300. Neither baseline-negative
case is eligible for the current alarm-removal selector. Its saved confidence
in the incorrect negative branch is0.004874 and0.000002951 respectively, so its
raw responsibility preference is already correct on these two consumed cases.

This identifies a concrete next hypothesis: whether a separately defined
positive-branch restoration rule can recover baseline-negative disagreements
without introducing false alerts. It does not establish that extending the
selector is safe or effective across the other cases. No such rule, threshold,
new prediction or fit was added to MZ36. The three remaining FP are all dense05
site006, HEAD_FAR crossbar, HEAD_NEAR cabinet and HEAD_FAR oblique_rod; two have
original geometric support and lie outside the selector's correction scope.
All five residual identities and branch/support values are in residuals.json.

The excluded site003 has native BODY support even in intended CLEAR/HEAD_ONLY
variants. Its20 frames remain a source/fixture-intent mismatch, not evidence of
algorithm correctness or safety. The earlier MZ35 <=20FP gate on the consumed
3000-frame cohort still fails at26; passing this separately frozen new-XY
criterion does not erase that recorded tradeoff.

## Disposition

Retain these finite new-placement outcomes and the original MZ5/MZ28/MZ30/MZ35 comparators. A passing correction earns evidence only for this admitted controlled cohort; a lost true alert is a transfer failure under the fixed criterion. Preserve failures and remaining misses before choosing a separately scoped successor. No source expansion, threshold rescue, protected EVAL, App/default promotion or safety claim follows.

Artifacts: artifacts.local/work/mz36-new-source-20260910/ contains frozen models/specs, empty review, full returned raw inputs, admission-v1, inference-v1, score-v1, paired-changes.json and audit.json. Worker originals are retained under G:/DevWorkspace/BlindAssist/artifacts/work/mz36-new-source-20260910; owned capture resources are released separately from durable evidence.
