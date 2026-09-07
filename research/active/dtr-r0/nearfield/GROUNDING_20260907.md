# NF-G9-A frozen bar/box intervention diagnostic

Phase: EXPLORE, controlled synthetic Development. Completed; selective object response insufficient on the new controlled source.

## Question and fixed scope

[G8](WHISKER_20260907.md) showed useful task learning and temporal closing, but auxiliary support did not establish decision attribution. G9-A asks whether each BODY/HEAD near alert responds selectively to its corresponding object. The nine G8 checkpoints and exact G8 validation-derived thresholds are frozen. No retraining, backbone replacement, mask-gated head, closing score or App promotion.

Fresh source:12 parameter groups, four variants (bar+box, bar only, box only, neither), three static frames each:144 images,48 windows. Groups g004..g015 are separate from two unscored preflight groups. Variant presentation order rotates acrossgroups to avoid always presenting a particular condition first. Retained object coordinates, sizes, materials and camera/body poses are exactly equal within each quartet. Camera is1.70m above floor, nominalpitch-5deg. BODY box and HEAD bar occupy disjoint vertical queries. Three static images fill the unchanged model interface; near heads still read currentRGB only.

Every present object is independently verified against native UE depth within3cm, with at least3visible pixels. Query masks union only these verified object surfaces; intended BODY/HEAD labels must agree with source expectations. Separate bar/box support rasters enable descriptive evidence accounting. Neither means no visible TASK object, not an all-scene free-space certificate. Evaluator-only native depth, poses, object geometry and masks never enter model inputs.

Adding a low box and compound scenes differs from G8's single-object source. Failure may reflect distribution shift, calibration or nonspecific false positives; it does not identify a particular shortcut. Deleting objects changes shadows/occlusion and temporal rendering state. Exact geometry of retained objects is not bit-identical background RGB. This is controlled intervention evidence, not a complete causal identification experiment.

## Operating points and metrics

Use exact per-arm/per-seed G8 normal thresholds; ensemble scores recomputed from allthree seeds. No new threshold selection. Report near TP/FP/FN and UNKNOWN, allfour variants, unconditional selective-removal correctness, conditional correctness given initially correct both-object predictions, all-four HEAD/BODY/joint correctness and probability deltas. No large probability delta alone counts as correct flipping.

Auxiliary-map metrics distinguish GT query coverage from the fraction of predicted probability mass inside the query, other-object-only and background fractions, plus IoU/pointing. Probability mass includes allpixels, including lowbackground probabilities; it is not a causal contribution or a saliency guarantee. Zero-mass cases are undefined, not successful localization.

## Acquisition preflight and UE adaptation

An independent48-frame preflight compared8 and2 settling updates on matching scenes. The shorter setting reduced median frame wall cost from406.19ms to273.86ms. Native labels and allquery masks agreed, but6/8 final-frame global RGB comparisons exceeded the predefined2/255 MAE limit. Local object crops passed5/255. Therefore main capture retained8; the experiment does not claim2settling is invalid universally, since scene/render evolution can contribute to differences.

Median stage wall costs at8: preparation+settling173.28ms, PNGexport42.19ms, raw native-depth readback2.80ms, Python float conversion+native write185.18ms. These are call-boundary wall measurements, not pure GPU kernel profiles. The first32update warmup frame is excluded. Main predicted steady capture58.49s plus39.24s capture-script startup/otheroverhead; launcher/engine boot outside script timing is not included.

The useful UE extensions delivered here are factorial multiobject generation, identity checks, independent native visibility perobject, sanitized RGB observations, stage timing and quality-constrained settling selection. Saved Willow map is unchanged; actors are temporary and task-owned process trees are released.

For this diagnostic the existing street is sufficient. Before large acquisition, optimize measured Python pixel conversion/write cost and validate any reduced settling against both taskregions and background. Before generalization claims, add independently heldout layouts/backgrounds, obstacle shapes/materials, camera mounting/turns, occlusion and lighting in small grouped batches. No new large map, photorealism work, human-mesh contacts or100k-frame capture is required for G9-A. Those capabilities should follow the question they resolve.

## Results and disposition

Native verification passed all144 frames and48 windows. All12 retained-geometry quartets verified; near truth/prediction UNKNOWN0. Closing is deliberately unscored (48 UNKNOWN perhead), not a failure.

| Ensemble | BODY TP/FP/FN (24positive,24negative) | HEAD TP/FP/FN | Both heads correct in allfour variants |
| --- | --- | --- | --- |
| single_frame | 19/3/5 | 19/7/5 | 0/12 |
| ordinary_video | 20/2/4 | 21/8/3 | 1/12 |
| bio_video | 9/3/15 | 22/6/2 | 0/12 |

| Ensemble | Initially both heads correct | Correct bar removal + BODY retention | Correct box removal + HEAD retention |
| --- | --- | --- | --- |
| single_frame | 8/12 | 1/8 conditional; 1/12 overall | 5/8 conditional; 5/12 overall |
| ordinary_video | 10/12 | 2/10 conditional; 2/12 overall | 6/10 conditional; 6/12 overall |
| bio_video | 3/12 | 0/3 conditional; 0/12 overall | 2/3 conditional; 2/12 overall |

Ordinary HEAD detections are12/12 withboth and9/12 withbaralone. Withboxalone, HEAD falsely alerts8/12; withneither it falsely alerts0/12. Removingbar lowers averageHEAD score by0.385, but only4/12 cross the frozen threshold correctly, and only2/12 also satisfy initially correct bothheads and retainedBODY correctness. A large scorechange is not sufficient evidence of reliable selective decisions. BODY boxdetections are10/12 both and10/12 boxalone.

Ordinary positive-support IoU BODY0.093/HEAD0.339; pointing5/24 and18/24. HEAD predictedprobabilitymass averages17.97% on the other-object-only region and57.26% on background. This includes diffuse lowprobabilities; it is descriptive map mass, not causal importance. G8 versus G9 denominators/stimuli differ, so these are not a matched architecture regression.

| Individual seed | BODY TP/FP | HEAD TP/FP | Joint allfour correct |
| --- | --- | --- | --- |
| single_frame 17 | 21/4 | 16/8 | 0/12 |
| single_frame 29 | 10/2 | 18/1 | 0/12 |
| single_frame 43 | 17/3 | 22/8 | 0/12 |
| ordinary_video 17 | 20/5 | 19/8 | 0/12 |
| ordinary_video 29 | 10/1 | 19/5 | 0/12 |
| ordinary_video 43 | 16/1 | 21/9 | 0/12 |
| bio_video 17 | 14/4 | 22/5 | 0/12 |
| bio_video 29 | 8/2 | 23/4 | 0/12 |
| bio_video 43 | 9/3 | 22/6 | 0/12 |

Main capture-script elapsed100.00s; frozen nine-model inference pipeline elapsed3.43s, CUDA on NVIDIA GeForce RTX 5060 Laptop GPU, torch2.11.0+cu130. Allcheckpoints hash-matched G8; fit_count0. Twelve source/inference/metric unit tests passed; actual native verification, inference and frozen evaluation ran.

**Disposition:** selective task-object response is inadequate in these compound/box Development scenes. Keep G8 as a scoped learning baseline, with G9 as a new intervention regression set. Do not promote current near predictions as well-grounded, and do not infer that a larger backbone or mask gating alone fixes this. The next learning candidate should address body-part/target separation with grouped object-presence data and an explicit same-data comparator; source variation and support constraints remain hypotheses. This turn stops after G9-A and necessary UE experiment capability delivery.

## Reproduction and lifecycle

Artifact entry: `artifacts.local/nearfield/grounding-20260907/`. `grounding_spec.py`, `grounding_launch.py`, `grounding_capture.py`, `grounding_verify.py`, `grounding_preflight.py`, `grounding_predict.py` and `grounding_evaluate.py` implement the stages. Freeze reference: `artifacts.local/nearfield/whisker-20260907-v2/learned` and its `evaluation/result.json`. Preflight and main separate outputs; failed source and receipts remain if any. Unit tests cover source factorial invariants, inference boundary and selective-removal/UNKNOWN metric cases. The main stop condition is one complete verified capture and frozen scoring; no automatic G9-B/C experiment follows.
