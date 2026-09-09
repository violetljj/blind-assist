# Frozen expanded B: strict forward-distance pairs

2026-09-09 EXPLORE. **Retain expanded B; range R0 remains a negative control.** This zero-training test finds some within-object distance response, but it is not stable across the tested regions. No new model, range display, threshold, or collection is promoted.

## Fixed comparison and source admission

The pre-inference [protocol](BODY_QUERY_DISTANCE_PAIRS_PROTOCOL_20260909.md) freezes 32 pairs / 64 frames. The existing 5000-frame source had 2000 exclusive HEAD-near/far frames but **zero strict matching pairs** after removing only common forward translation. The new batch therefore fixes camera, background, assembly dimensions/material/rotation/lateral position/height within each pair and translates the complete assembly, including supports. Near/far distances are 1.0/2.1 m at the first site and 1.3/2.6 m at the second. Four families at two sites in each of four regions give 32 pairs.

All 64 frames pass capture health, paired-spec equality, floor validation, native BODY=0 / HEAD=1 and exclusive intended HEAD range-event acceptance. All 64 were visually inspected before inference; no replacements or exclusions. Native UNKNOWN is preserved. Truth is used only for admission/scoring, never model input.

Expanded B checkpoint SHA256 `c7aef143bcc7ac464097e95219dc5239523df7ce5716fc95926f37592f94e776`; original DEV thresholds BODY `0.18024158477783203`, HEAD `0.5914403796195984`. Input RGB 144x256, unchanged B forward path. CUDA inference on RTX 5060 Laptop GPU, torch 2.11.0+cu130; **zero training steps**. `S_F` is P(sum of the three HEAD-far capped counts >= 3), computed from saved distributions; it is not a calibrated distance probability.

## Main observations

| Observation | Result |
| --- | --- |
| Far score increases, delta > 1e-6 | 25/32 (78.125%) |
| Reverse direction / ties | 7/32 / 0/32 |
| Both endpoints HEAD alert | 29/32 |
| HEAD near-end / far-end hits | 32/32 / 29/32 |
| HEAD hits, all frames | 61/64 |
| BODY false alerts, all frames | 5/64 |
| Mean / median far-score delta | -0.057944 / 0.003414 |

Raw sign and the predeclared 1e-6 tolerance agree. All 25 direction-correct pairs also retain both HEAD alerts. This source has no HEAD-negative frames, so **HEAD false-positive rate cannot be estimated**; BODY errors are a different denominator.

The positive direction fraction does not tell the whole story: many increments are small, while several reversals are large. Mean delta is negative despite a positive median. All near-frame far scores already exceed 0.87, although native far events are absent in those frames. Thus a small within-pair ranking response coexists with substantially incorrect absolute range attribution; it does not rescue the near-cell semantics.

## Region and family slices

| Region | Far higher | Both HEAD | Mean delta | Median delta |
| --- | --- | --- | --- | --- |
| big01 | 3/8 | 5/8 | -0.243169 | -0.077058 |
| big03 | 7/8 | 8/8 | -0.033566 | 0.005246 |
| big05 | 8/8 | 8/8 | 0.014659 | 0.002383 |
| big06 | 7/8 | 8/8 | 0.030300 | 0.021114 |

big01 accounts for five of seven reversals and all three missed HEAD frames, each at the far endpoint. The other regions do not make this failure disappear. Region is not proven causal: sites, background, illumination and appearance covary here.

| Family | Far higher | Both HEAD | Mean delta |
| --- | --- | --- | --- |
| crossbar | 6/8 | 7/8 | -0.053808 |
| cabinet | 7/8 | 7/8 | -0.103918 |
| oblique_rod | 6/8 | 7/8 | -0.035043 |
| hanging_sign | 6/8 | 8/8 | -0.039006 |

## Size and site detail

| Target dimensions x/y/z (m), family | Far higher | Both HEAD | Mean delta |
| --- | --- | --- | --- |
| crossbar:[[0.03159732148573795, 2.0, 0.03159732148573795]] | 3/4 | 3/4 | -0.118672 |
| cabinet:[[0.045, 1.1, 0.56]] | 7/8 | 7/8 | -0.103918 |
| oblique_rod:[[0.05418688374125799, 1.516859041186462, 0.05418688374125799]] | 3/4 | 4/4 | 0.003878 |
| hanging_sign:[[0.07, 0.95, 0.38]] | 6/8 | 8/8 | -0.039006 |
| crossbar:[[0.07280304348305072, 2.0, 0.07280304348305072]] | 3/4 | 4/4 | 0.011056 |
| oblique_rod:[[0.07111355025768072, 1.5214507444062646, 0.07111355025768072]] | 3/4 | 3/4 | -0.073965 |

These are six target-size strata, not 32 independent sizes. Cabinet and sign target dimensions do not vary; crossbar and oblique rod have two sizes each. Assembly/template ranks can still differ in placement and supports. Each size stratum reuses geometry across sites, and size is not crossed independently with every distance interval.

| Site | Far higher | Both HEAD | Mean delta |
| --- | --- | --- | --- |
| big01_site_003 | 2/4 | 3/4 | -0.144535 |
| big01_site_005 | 1/4 | 2/4 | -0.341803 |
| big03_site_002 | 4/4 | 4/4 | 0.024054 |
| big03_site_003 | 3/4 | 4/4 | -0.091186 |
| big05_site_003 | 4/4 | 4/4 | 0.009518 |
| big05_site_004 | 4/4 | 4/4 | 0.019800 |
| big06_site_002 | 4/4 | 4/4 | 0.036414 |
| big06_site_003 | 3/4 | 4/4 | 0.024186 |

## All reverse pairs

| Pair | S_F near | S_F far | Delta | HEAD alerts near/far |
| --- | --- | --- | --- | --- |
| bqdist-big01_site_003-crossbar | 0.933031 | 0.409432 | -0.523599 | [True, False] |
| bqdist-big01_site_003-oblique_rod | 0.990874 | 0.872805 | -0.118069 | [True, True] |
| bqdist-big01_site_005-crossbar | 0.997881 | 0.961834 | -0.036047 | [True, True] |
| bqdist-big01_site_005-cabinet | 0.956548 | 0.040041 | -0.916507 | [True, False] |
| bqdist-big01_site_005-oblique_rod | 0.997526 | 0.580137 | -0.417389 | [True, False] |
| bqdist-big03_site_003-hanging_sign | 0.997543 | 0.623468 | -0.374075 | [True, True] |
| bqdist-big06_site_003-hanging_sign | 0.991023 | 0.970536 | -0.020488 | [True, True] |

## Interpretation and stop

The frozen output responds in the intended direction for most of these controlled translations. It therefore contains more than a fixed query-position prior, but this experiment cannot distinguish angular-size cues, context, occlusion or shadows from explicit geometric inference. It does **not** establish robust distance estimation, correct HEAD-near occupancy, or transferable body-space semantics. No complement `1-S_F` is interpreted as a near probability.

Existing CitySample sites/assets/templates remain consumed shared-world Development. Pairs are correlated, and no independent-trial confidence claim is made. Moving rigid supports over curbs can leave their feet visually above lower roadway; near supports can be cropped. These were recorded during source review and retained, rather than fixed after seeing predictions. The assemblies are controlled fixtures, not natural-object trials.

The bounded experiment is complete. Keep expanded B for its previously established controlled alert behavior, keep range R0 negative, and retain these paired data as a diagnostic component. Do not infer a stable geometry capability from 25/32 or automatically open another loss/readout/depth sweep. Any future geometry change must preserve alerts and address the observed large reversals and near-frame far activation, not merely improve pooled AUC.

## Reproduction and evidence

Entry points: `body_query_distance_pairs_spec.py` and `body_query_distance_pairs.py prepare|infer`. Use the repository research GPU runtime; pass `--root artifacts.local/work/body-query-distance-pairs-20260909`. Fresh output directories prevent silently overwriting this run.

Durable artifacts under that root: `existing-pair-audit.json`, `specs-v1/manifest.json`, `capture-v1/*` source/native/RGB/health/release receipts, `visual-review-v1/source-visual-review.json`, `prepared-v1/acceptance.json`, and `inference-v1/{result.json,predictions.npz,receipt.json,validation.json}`. All 32 pair scores are saved, including failures. Source visual-review SHA256: `552538b4422f99c7db245a17ac96a1075c389ffd9b9eb9d5546e868fbe8c3504`.

Independent cached-count enumeration of all 64 three-cell count states agrees with the range operator to maximum absolute error `5.906733435701028e-16`; pair signs, alerts, eligibility and receipt hashes pass. Spec mutation checks reject changes to dimensions/material/rotation. Python AST validation covers both new source files. The first verification invocation encountered a task-local `inspect.py` shadowing Python stdlib; rerunning with Python `-P` fixed import resolution without altering predictions. No extra inference or training was performed.

Task-owned capture processes were released by each capture wrapper; final ownership checks and disposable-temp cleanup are recorded in `cleanup.json`. Raw evidence, logs and shared DDC are retained.
