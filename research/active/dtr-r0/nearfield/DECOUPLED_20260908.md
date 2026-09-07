# NF-G13: deep-content support gating

2026-09-08 · EXPLORE · **Consumed G12 screen followed by one prospective G13 same-Willow Development cohort.**

The predeclared prospective joint criterion was met. Retain D as a stronger same-Willow Development challenger, with G10 still the complete-contract baseline. B and C remain comparisons. End after three fixed fits and this one new cohort; no further tuning, fits or capture. This does not establish a general mechanism or natural-scene generalization.

## Question and controlled change

Can support from deep plus shallow features guide deep-only feature contents and combine B decision quality with C query localization? Let F be deep features and E shallow detail: C pools `(F+E) * sigmoid(Support(F+E))`; D pools `F * sigmoid(Support(F+E))`. No detach: near loss still reaches E through the support head, and the backbone remains shared. This is a gate-content intervention, not complete task isolation.

D starts from the same official RepViT initialization as G12 C, not a fine-tuned C checkpoint. Exact full/common parameter initialization and the entire 600-step batch sequence matched the G12 C receipt for each seed. Parameter count is unchanged from C. For the consumed diagnostic, B/C predictions and maps were copied with hash checks, with zero refits and zero inference; all seed/ensemble primary, secondary and localization evaluations exactly reproduce G12.

The prior assertion that detail necessarily contaminates semantics was a hypothesis, not a diagnosis: G12 C single-seed joints did not decline, compatibility ensembles tied, and different calibrated thresholds explained some ensemble misses. This experiment preserves those limitations.

## Frozen training and scoring

800 TRAIN images, shared G10 VAL96; seeds 17/29/43, 600 steps each, batch32 balanced across four variants. AdamW lr/weight-decay 1e-4; near BCE +0.25 support BCE; first minimum VAL near BCE every50 steps selects the checkpoint. Same full RGB144×256 normalization, BN policy and native18×32 query supervision. Only TRAIN/VAL influence fitting and calibration.

Primary uses the unchanged common-VAL recall-first selection rule: each head overall positive recall≥95%, HEAD additionally bar-only recall≥95%, minimizing FP then maximizing feasible inclusive threshold. Secondary retains G8 compatibility thresholds. Localization includes fixed0.5 IoU, peak hit, and separate VAL-selected mask IoU. Closing remains UNKNOWN and unscored.

G12 g6000–6031 are consumed. D passed all predeclared diagnostic conditions (joint30/32, diverse14/16, bar/both HEAD32/32 each, BODY TP62/64, no near FP, BODY/HEAD IoU0.4068/0.1788, peaks61/45). Only then was G13 g7000–7031 seed13080908 collected: 32 groups, 16 narrow/16 diverse, four variants each, unchanged parameter ranges and Willow map. All B/C/D checkpoints stayed fixed. New results below are prospective controlled Development after a consumed-data screen, not natural-scene confirmation.

## Prospective G13 ensemble results

| Metric | B | C | D |
|---|---:|---:|---:|
| Primary joint /32 | 31 | 30 | 31 |
| Primary narrow joint /16 | 16 | 16 | 16 |
| Primary diverse joint /16 | 15 | 14 | 15 |
| Primary BODYnear TP / FP / FN | 64/0/0 | 64/0/0 | 64/0/0 |
| Primary HEADnear TP / FP / FN | 63/0/1 | 61/0/3 | 63/0/1 |
| G8 compatibility joint /32 | 31 | 30 | 31 |
| G8 compatibility narrow joint /16 | 16 | 16 | 16 |
| G8 compatibility diverse joint /16 | 15 | 14 | 15 |
| G8 compatibility BODYnear TP / FP / FN | 64/2/0 | 64/4/0 | 64/2/0 |
| G8 compatibility HEADnear TP / FP / FN | 64/0/0 | 64/0/0 | 64/0/0 |
| BODY fixed IoU | 0.3644 | 0.3930 | 0.3957 |
| BODY peak hits /64 | 64 | 63 | 63 |
| BODY VAL-mask IoU | 0.6376 | 0.6530 | 0.6571 |
| HEAD fixed IoU | 0.1676 | 0.1916 | 0.2017 |
| HEAD peak hits /64 | 38 | 57 | 54 |
| HEAD VAL-mask IoU | 0.4105 | 0.5106 | 0.5187 |

## Per-seed and per-variant disclosure

| Arm/seed | Primary joint /32 | Compatibility joint /32 | BODY fixed IoU | HEAD fixed IoU | BODY peak /64 | HEAD peak /64 |
|---|---:|---:|---:|---:|---:|---:|
| B RepViT/17 | 26 | 31 | 0.3873 | 0.1704 | 64 | 41 |
| B RepViT/29 | 29 | 30 | 0.3968 | 0.1896 | 61 | 40 |
| B RepViT/43 | 27 | 29 | 0.2853 | 0.0948 | 62 | 33 |
| B RepViT/ensemble | 31 | 31 | 0.3644 | 0.1676 | 64 | 38 |
| C +detail/17 | 28 | 31 | 0.4138 | 0.1980 | 60 | 49 |
| C +detail/29 | 30 | 30 | 0.4228 | 0.2089 | 63 | 46 |
| C +detail/43 | 30 | 31 | 0.3047 | 0.1224 | 59 | 42 |
| C +detail/ensemble | 30 | 30 | 0.3930 | 0.1916 | 63 | 57 |
| D deep-content gate/17 | 28 | 31 | 0.4142 | 0.2058 | 62 | 52 |
| D deep-content gate/29 | 30 | 30 | 0.4254 | 0.2188 | 61 | 48 |
| D deep-content gate/43 | 30 | 30 | 0.3141 | 0.1128 | 61 | 46 |
| D deep-content gate/ensemble | 31 | 31 | 0.3957 | 0.2017 | 63 | 54 |

TP/FP/FN below retain all four variants; each variant contains32 samples.

| Policy/arm | Variant | BODY TP/FP/FN | HEAD TP/FP/FN |
|---|---|---:|---:|
| primary/B RepViT | both | 32/0/0 | 32/0/0 |
| primary/B RepViT | bar_only | 0/0/0 | 31/0/1 |
| primary/B RepViT | box_only | 32/0/0 | 0/0/0 |
| primary/B RepViT | neither | 0/0/0 | 0/0/0 |
| primary/C +detail | both | 32/0/0 | 30/0/2 |
| primary/C +detail | bar_only | 0/0/0 | 31/0/1 |
| primary/C +detail | box_only | 32/0/0 | 0/0/0 |
| primary/C +detail | neither | 0/0/0 | 0/0/0 |
| primary/D deep-content gate | both | 32/0/0 | 32/0/0 |
| primary/D deep-content gate | bar_only | 0/0/0 | 31/0/1 |
| primary/D deep-content gate | box_only | 32/0/0 | 0/0/0 |
| primary/D deep-content gate | neither | 0/0/0 | 0/0/0 |
| secondary/B RepViT | both | 32/0/0 | 32/0/0 |
| secondary/B RepViT | bar_only | 0/1/0 | 32/0/0 |
| secondary/B RepViT | box_only | 32/0/0 | 0/0/0 |
| secondary/B RepViT | neither | 0/1/0 | 0/0/0 |
| secondary/C +detail | both | 32/0/0 | 32/0/0 |
| secondary/C +detail | bar_only | 0/2/0 | 32/0/0 |
| secondary/C +detail | box_only | 32/0/0 | 0/0/0 |
| secondary/C +detail | neither | 0/2/0 | 0/0/0 |
| secondary/D deep-content gate | both | 32/0/0 | 32/0/0 |
| secondary/D deep-content gate | bar_only | 0/1/0 | 32/0/0 |
| secondary/D deep-content gate | box_only | 32/0/0 | 0/0/0 |
| secondary/D deep-content gate | neither | 0/1/0 | 0/0/0 |

Full per-seed per-variant confusion counts, thresholds, selective-removal outcomes, negative mask activations, and unchanged G10 regression are retained in `fresh-summary.json` and `fresh/evaluation/`; the consumed diagnostic remains in `summary.json` and `main/evaluation/`.

## Predeclared prospective decision

| Diagnostic condition | Result |
|---|---|
| joint_nondecreasing | PASS |
| diverse_nondecreasing | PASS |
| HEAD_bar_nondecreasing | PASS |
| HEAD_both_nondecreasing | PASS |
| HEAD_FP_not_increasing | PASS |
| BODY_TP_nondecreasing | PASS |
| BODY_FP_not_increasing | PASS |
| HEAD_peak_gain8 | PASS |
| BODY_peak_preserved | PASS |
| BODY_IoU_nearC | PASS |
| HEAD_IoU_nearC | PASS |
| HEAD_IoU_gain01 | PASS |
| BODY_IoU_preserved | PASS |
| majority_seeds_nondecreasing | PASS |

The consumed screen required all twelve conditions in protocol.md and passed before fresh capture. The prospective rule above requires preserving B total/diverse joint, bar/both HEAD recall, BODY TP, both FP counts, at least2/3 paired-seed joints; gaining8 HEAD peak hits, preserving BODY peak within1, each fixed IoU within0.02 of contemporary C, HEAD IoU gain≥0.01 over B and BODY IoU within0.01 of B. Tolerances are descriptive, not statistical noninferiority. Both operating-point policies are disclosed, with no TEST-driven changes.

## Interpretation and residual errors

On the prospective cohort, D preserves B primary ensemble decisions while improving both fixed IoUs and HEAD peak localization. All three D seeds have higher primary joint counts than corresponding B seeds (28/30/30 versus26/29/27), and both fixed IoUs improve for each paired seed. This supports the tested gate-content design as a Development improvement.

The gain is not uniform relative to C: D HEAD peak54/64 is below C57/64, BODY peak63/64 is one below B64/64, and seed43 HEAD IoU is below C. D therefore combines much of B decision quality and C localization rather than dominating every metric. It does not prove that the prior failure was semantic contamination or establish a generally isolated semantic/spatial architecture.

B and D each still miss one bar-only sample at the primary VAL-selected working point. Compatibility thresholds recover HEAD recall but introduce two BODY false positives for each; the matching joint31/32 hides different error types. Do not reinterpret this as zero remaining misses.

Ensemble HEAD scores for every prospective group with an error in any arm are shown below; the complete paired scores for both phases and all seeds are retained in `paired-head-scores.json`.

| Arm | Group | Both score | Bar-only score | VAL threshold | Both / bar hit |
|---|---|---:|---:|---:|---|
| repvit | g7007 | 0.974236 | 0.773683 | 0.816796 | True / False |
| repvit_detail | g7007 | 0.925038 | 0.893047 | 0.964721 | False / False |
| repvit_detail | g7031 | 0.963137 | 0.972047 | 0.964721 | False / True |
| decoupled | g7007 | 0.975542 | 0.883399 | 0.885662 | True / False |

New native capture and verification passed all128 frames on the accepted worker source. Capture/verification/package wall time was96.60s; nine fixed-checkpoint inference runs took12.56s including local preparation. Worker process audit confirmed nine tracked process identities released and its scheduled job removed. Full native depth/logs remain on the worker; hashed RGB and evaluator receipts were returned for same-input B/C/D evaluation.

## Cost and verification

Three fits / 1800 optimizer steps; summed fitting time 374.76s, trainer wall time 398.15s. Actual backend cuda, device NVIDIA GeForce RTX 5060 Laptop GPU; D has 4,736,372 parameters. This is training cost, not mobile inference latency.

| Seed | Selected step | Fit seconds |
|---|---:|---:|
| 17 | 600 | 129.33 |
| 29 | 600 | 120.78 |
| 43 | 200 | 124.66 |

Two focused tests passed: exact C parameter initialization, and attached detail/support gradients with deep-only pooled contents. Runtime checks matched full initialization and batch hashes for all three fits. All eight B/C seed/ensemble evaluations reproduce G12 exactly under both policies and localization. Executed source, helper AST parity, checkpoint/input hashes and process receipts remain in the artifact root.

## Evidence

- Artifact root: `artifacts.local/nearfield/decoupled-20260908/` (canonical F-backed junction).
- Frozen `protocol.md`; `helper-reproduction.json`; `pipeline-receipt.json`; `evaluation-receipt.json`; `summary.json`; `main/learned/receipt.json`; `main/evaluation/result.json`; `fresh-summary.json`; `fresh/evaluation/result.json`; `fresh/learned/receipt.json`; worker capture/verification/transfer/process receipts.
- Registration input: `experiments/nearfield/nf-g13-decoupled-20260908-inputs.json`.
- Preceding comparison: [G12](REPRESENTATION_20260908.md).

After this fixed prospective check, no further training/capture, temporal replay, deployment or App promotion is included. Spatial changes do not inherit old approaching-motion performance without an actual replay.
