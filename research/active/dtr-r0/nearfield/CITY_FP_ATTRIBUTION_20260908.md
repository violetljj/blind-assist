# City false-positive diagnosis (zero training)

Scope: consumed Development diagnostics of the completed City seed17/step200
pilot. Reuse the four cached prediction files; do not search thresholds or run
an optimizer. Additional TRAIN750 inference, if executed, is a separate read-only
check of whether the failure occurs in the fitted street region too. The cached
four files contain plaza TEST750 and Willow96, not City TRAIN predictions.

Questions: did negative logits move upward; does that movement correlate with
support activation; where do selected false-positive maps respond; and can the
existing triplets supply a genuinely different counterfactual target?
Probability caches permit clipped inverse-sigmoid logits, not recovery of exact
pre-sigmoid values at float saturation. Correlation is not an intervention on
support, and cannot identify the responsible loss, feature, bias or normalization.

## Verified supervision and pairing contract

The actual native verifier constructs `corridor_masks` from reconstructed 3D
surface points. A positive pixel must lie inside the body's forward query volume.
It does **not** label every visible city surface positive. Invalid native depth
is UNKNOWN; known points outside the volume are zero. Thus a nearby wall can be
correct positive evidence, while distant facades and ordinary ground below the
BODY height band are not. `city_data.py` pools these existing labels rather than
turning visible background into positive support.

`city_cf_contract_audit.py` verifies the original captured specification against
the capture receipt and the 1,500 native truth rows. Each region has 250 groups
with identical camera/floor within each clear/center/right-clearance triplet.

| Each region, separately | BODY | HEAD |
| --- | ---: | ---: |
| Mixed-label groups (0,1,0) | 135 | 15 |
| All-negative groups (0,0,0) | 115 | 235 |
| Ordered positive-versus-negative comparisons available | 270 | 30 |
| Negative frames with any native positive support pixel | 0/615 | 0/735 |

Consequences:

- Rank only heads with different actual labels. An injected center object can
  remain negative because of distance or open geometry; variant names are not labels.
- On jointly known pixels, subtracting a negative binary support mask from a
  positive mask leaves the positive mask unchanged in this cohort. This proposed
  target alone adds no positive localization information. UNKNOWN remains separate.
- A loss on **predicted within-pair support differences** would be a different
  intervention from subtracting the ground-truth masks; its behavior is untested.
- Equal camera poses do not establish pixel-identical background: insertion can
  change shadows, occlusion and temporal rendering. RGB subtraction is not an
  obstacle-instance ground truth. This capture has native depth/support, not a
  verified per-object instance-mask export for that proposal.
- Group ranking is invariant to a shared additive logit offset. BCE or another
  absolute operating constraint is still needed to control false positives.

The model code does contain attached support gating of deep features before the
near head. This establishes an architectural path, not that support-area growth
caused the observed decision failure.

Audit artifact: `artifacts.local/nearfield/city-fp-attribution-20260908/contract-audit.json`.
Original results remain in [the fixed pilot report](CITY_FINETUNE_PILOT_20260908.md).

## Observations

Two additional eval/inference_mode passes on TRAIN750 took 5.269 s on the worker
RTX 3060 Laptop GPU, zero optimizer steps and zero repeated TEST inference.

| Step200, fixed >=0.5 | Street TRAIN750 | Plaza TEST750 |
| --- | ---: | ---: |
| BODY TP / FP / FN / TN | 131 / 2 / 4 / 613 | 128 / 561 / 7 / 54 |
| HEAD TP / FP / FN / TN | 11 / 2 / 4 / 733 | 9 / 348 / 6 / 387 |
| BODY recall / FPR | 97.04% / 0.33% | 94.81% / 91.22% |
| HEAD recall / FPR | 73.33% / 0.27% | 60% / 47.35% |
| BODY negative frames with positive predicted support | 615/615 | 442/615 |
| HEAD negative frames with positive predicted support | 732/735 | 150/735 |

The original TRAIN baseline had BODY TP1/FP0/FN134/TN615 and HEAD
TP0/FP22/FN15/TN713. The fit learned useful TRAIN discrimination; it did not
uniformly flip both City regions to positive. This is a major region-transfer
failure, without an identified causal feature. Training-set success is not
generalization evidence.

For plaza negatives, derived near logits increased on 594/615 BODY and 470/735
HEAD frames. Median increases were 0.21087 and 0.01506 respectively. Many original
probabilities lie near 0.5; changed decision counts are not a measured class prior.

| Plaza FP breakdown | BODY | HEAD |
| --- | ---: | ---: |
| clear | 230/250 | 115/250 |
| center (actually negative) | 107/115 | 92/235 |
| right_clearance | 224/250 | 141/250 |

By acquisition condition, BODY's 561 errors comprise 230 clear, 224 lateral,
94 centered at nominal 3.5/6 m anchors, and 13 other centered negatives. HEAD's
348 comprise 115 clear, 141 lateral, 66 centered at 3.5/6 m, and 26 other centered
negatives. These are metadata-based error categories, not inferred surface types.
Mean center-minus-clear logit contrast over all 250 plaza groups changes from
0.0040 to 1.0447 for BODY, and -0.0033 to -0.1865 for HEAD. The heads behave
differently; these averages mix positive and negative center conditions.

Pearson / Spearman correlation of step200 negative support statistics with the
derived near-logit increment:

| Support statistic | BODY (615 negatives) | HEAD (735 negatives) |
| --- | ---: | ---: |
| All-pixel max | 0.518 / 0.639 | -0.265 / 0.118 |
| All-pixel mean | 0.720 / 0.596 | -0.375 / -0.079 |
| All-pixel >=0.5 fraction | 0.816 / 0.622 | -0.376 / -0.029 |
| Known-pixel >=0.5 fraction | 0.811 / 0.628 | -0.459 / -0.049 |

The BODY association survives excluding UNKNOWN pixels, but does not explain
HEAD. TRAIN is another counterexample to a simple support-area explanation:
almost every negative frame has some predicted support activation while near
FP stays low. The architecture weights gated deep feature content, not just
the number of lit pixels. These observational correlations cannot attribute
the failure to the auxiliary loss.

All 1,000 center-versus-control/head GT comparisons in plaza have support delta
equal to the center positive mask on jointly known pixels. Negative controls
contain zero positive mask pixels. Across these comparisons, 82 pooled center
positive cells land on UNKNOWN in a control; these are not known-negative
counterfactual evidence and must not be silently converted to zero.

## Visual review

The primary agent inspected all ten contact sheets: top50 BODY FP by predicted
near probability (stable sample-ID tie break), plus seeded random50 of the 54 TN.
Each row shows RGB, original support, step200 support and encoded GT. Selection
was made before the visual review and is recorded in selection.json.

The top50 contains 29 center and 21 right-clearance frames, **no clear frames**.
It is a high-confidence error sample, not a random sample of the 561 errors.
Examples: 820/970 highlight bench vicinity; 1180/1105 table vicinity; 818/893
highlight laterally placed benches; 850/925/1000 highlight lower scaffold and
adjacent pavement. Activation spreads outside visible object outlines and
occasionally into road/shadow or distant background. TN examples include faint
or fragmented background responses as well as inserted objects. This qualitative
review does not supply semantic pixel labels, instance overlap scores, or
population-wide counts by surface type; the 230 clear FPs require separate
sampling before such a claim. The evidence does not justify saying the FP maps
mainly activate ordinary building facades rather than the inserted assets.

## Decision and evidence

No City-CF fit was launched. The proposed immediate rationale is not established:
City-wide near activation is contradicted by TRAIN; the subtraction target adds
no known positive support information; and the support-near association differs
between heads. Group ranking remains an eligible hypothesis, not a demonstrated
repair. The current recipe's TRAIN discrimination already succeeds, so adding
more TRAIN ranking pressure does not by itself establish a way to generalize.

Before any subsequent fit, define whether the intervention changes predicted
pair differences, negative localization suppression, or source diversity, and
freeze explicit FPR, recall, joint and localization comparisons. Preserve this
plaza as consumed Development. Do not package an untested combination of ranking
and unchanged masks as a demonstrated causal mechanism, or select a threshold
on these inspected outcomes. Original checkpoints and the failed fit are retained.

Artifacts under `artifacts.local/nearfield/city-fp-attribution-20260908/`:

- `evidence-v2/city-fp-attribution-20260908-v2/`: summary.json, per-frame.csv/json,
  paired-contrasts.json, matched-gt-support.json, selection.json, gallery.html,
  100 overlays, 10 contact sheets and fixed-threshold scatter plots. CPU cached
  analysis took 5.803 s; v2 adds the requested paired-logit plots to v1 without
  new model inference or changed selection.
- `evidence/city-fp-attribution-20260908/train-inference/`: both new TRAIN prediction
  caches and receipt.json. The executed inference helper remains in the artifact
  root as train_inference.py, with source hash in the receipt.
- `focused-checks.json`: inverse-sigmoid, constant/tied correlation, variant counts,
  confusion recomputation, deterministic unique selection and 260 output hashes PASS.
- `final-release.json`: both weight hashes unchanged, no task-owned processes left.

This is an explanatory follow-up to the existing pilot, not a new training arm
or a claim of identified causality. No UE scene was modified or recaptured.
