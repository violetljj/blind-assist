# City fine-tuning pilot

Prospectively fixed before this pilot's prediction outcomes; EXPLORE on
already inspected City acquisition data and consumed Willow Development data.
Question: does a small City-only update improve the retained G13 D single
checkpoint on a different region, without sacrificing Willow behaviour?

Baseline is the original G13 D seed17 checkpoint (not the three-seed ensemble).
One new fit warm-starts those exact weights, unchanged architecture and input
resolution. Use City street TRAIN750 only; plaza TEST750 has no role in fitting,
selection or calibration. The 96 existing G10 Willow VAL frames are disclosed
consumed regression data, not fresh test data and not used for this fit.

Fixed budget: one seed17 fit, exactly 200 AdamW steps, batch32, lr1e-5,
weight_decay1e-4. Uniform TRAIN sampling with replacement from a seed17 RNG.
All parameters trainable, BatchNorm running statistics frozen in eval mode.
Near BCE +0.25 unknown-aware class-balanced support BCE. No image augmentation,
replay, class reweighting, hyperparameter sweep, early stopping or best-checkpoint
selection. Use the final step200 checkpoint. Stop after this fit and paired
evaluation; numerical/infrastructure failure may be diagnosed, not silently
retried with a changed recipe. Keep original G13 files/checkpoints untouched.

Both arms use inclusive 0.5 near and support thresholds. Report each BODY/HEAD
TP/FP/FN/TN, recall, specificity and precision; macro balanced error is primary.
Also report positive-frame known-pixel support IoU, peak-hit counts, negative
map activation and City all-three-variants joint correctness. Preserve missing
denominators as null. Report fixed-threshold binary task performance only,
not WARNING/DANGER, approaching or whole-world/pixel-equivalent performance.

Useful pilot result requires lower City macro balanced error, neither City's
head recall declining, and neither Willow head's FP or FN increasing. Report
localization tradeoffs separately. If the joint condition fails, keep the old
checkpoint as baseline and record the new fit as a diagnostic candidate only;
do not rerun/tune to rescue it. Even a pass is single-seed same-map Development,
not natural-world validation or automatic app/model promotion.

The City adapter expands old controlled-object support to all visible scene
surfaces and ignores pixel UNKNOWN. The corpus has zero HEAD-only/HEAD DANGER
and all HEAD positives are scaffold frames; these coverage limitations remain.

## Observed result and disposition

The single 200-step run completed on the worker RTX 3060 Laptop GPU: 29.942 s
CUDA fitting, 35.445 s whole script. No additional fit or threshold selection
followed. The original checkpoint was unchanged. Metrics tests: 8/8 PASS.

| City plaza TEST750 | Original seed17 | City step200 |
| --- | ---: | ---: |
| BODY TP / FP / FN / TN | 4 / 3 / 131 / 612 | 128 / 561 / 7 / 54 |
| HEAD TP / FP / FN / TN | 0 / 4 / 15 / 731 | 9 / 348 / 6 / 387 |
| BODY recall / precision | 2.96% / 57.14% | 94.81% / 18.58% |
| HEAD recall / precision | 0% / 0% | 60% / 2.52% |
| Macro balanced error | 49.52% | 45.94% |
| All-three-variants and both-heads correct | 115/250 | 3/250 |
| BODY / HEAD positive support mean IoU | 0 / 0 | 0.1419 / 0.0495 |
| BODY / HEAD support peak hits | 18/135 / 1/15 | 39/135 / 0/15 |
| BODY / HEAD negative frames with false-positive mask | 1/615 / 2/735 | 442/615 / 150/735 |

Willow consumed VAL96 retains zero near FP/FN for both heads in both arms.
However, BODY support IoU falls 0.4260 to 0.2155, and HEAD 0.1873 to 0.1036.
Thus unchanged binary regression does not mean unchanged localization.

The prospectively declared joint criterion is **true**, as recorded without
alteration in result.json. It was too weak to constrain City false alarms:
BODY false-positive rate rises from 0.49% to 91.22%, HEAD from 0.54% to 47.35%.
Execution PASS and that formal criterion do not establish useful improvement.
**Do not replace the retained baseline.** Keep this checkpoint as a diagnostic
challenger demonstrating the failure of this scoped City-only recipe.

This does not establish the cause of the failure. Scene/label-domain differences,
support-loss interactions and limited coverage are hypotheses, not measured
attributions. Before another fit, inspect existing TRAIN predictions and label
overlays for negative variants and support semantics; any later comparison needs
a new protocol with explicit false-positive and localization constraints.
The inspected plaza remains consumed Development, not a fresh confirmation set.

Evidence under the canonical ignored artifact entry:
`artifacts.local/nearfield/city-pcg-20260908/worker-finetune-prep-v1/run-v1/`
contains result.json, receipt.json, protocol.json and four cached prediction
files. Worker weights remain at
`G:/DevWorkspace/BlindAssist/artifacts/work/city-finetune-pilot-20260908/run-v1`.
Task process and scheduled task were released; durable inputs/results retained.

Original checkpoint SHA256:
`0c14179486102993508dd2385a3b6f9e17d51a6d53e657c0240325e5f78d5c7b`.
New checkpoint SHA256:
`65a25a30ea0753e9692ad1766b3df72efa192bb0e0ad35b92328fbfcfd36d913`.
Result SHA256:
`a2426c632e0ec5020f2bd6dd032a59592f39c793e874c5ebbfe9fc8eb8006877`.
The worker's pre-outcome protocol snapshot has SHA256
`b6e05c75b98e7451cae62ef5d6e26d66b0e7def07d0e197d021971d4dae42854`;
the sections above this result appendix preserve its recipe and criteria.
