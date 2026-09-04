# L10 reference-conditioned commitment on fresh SEVN PAN episodes

Date: 2026-09-05

Decision: `L10_SEVN_REFERENCE_COMMITMENT_FRESH_DEVELOPMENT_GATE_NOT_MET`

The unchanged triggered observation policy retained all five fixed-sweep correct
bindings, avoided one wrong binding and used four fewer online observations.
The new geometric commitment check rejected every correct triggered binding.
Keep the action-policy result as a narrow fresh Development observation; do not
promote this verifier into the controller.

## Frozen intervention and source

Candidate generation, OCR, portal masks, action selection and confidence-based
final-candidate selection are reused from `l10_sevn_progressive_episode.py`.
The only scientific intervention is a separate verifier after final selection.
It can retain that candidate or return `UNKNOWN`; it cannot replace a candidate,
change an action or request another observation. The rejected ZuBuD CA1 threshold
is not reused.

The metadata-only source freezer selected four PAN_LEFT and four PAN_RIGHT
episodes, each with two different-frame human-labeled public reference crops
and a nonoverlapping differently addressed sibling door. All 24 query/reference
frames are distinct and excluded from the seven earlier panels (245 addresses,
260 frames). Selection used no RGB or model outcomes. Different addresses with
overlapping door annotations were recorded as aliases, never sibling negatives.

SEVN associates a door annotation with an address; it does not provide an
independent cross-frame physical-door ID. Reference crops are deliberately
supplied privileged inputs, not recovered by the algorithm. Only their pixels
enter the verifier; reference extraction metadata and query truth do not.
The provider's [data description](https://github.com/mweiss17/SEVN-data) documents
the address-associated door annotation.

The verifier uses SIFT matching and RANSAC to project the reference-door rectangle
into the query and requires that it agree with the unchanged selected mask.
The [OpenCV tutorial](https://docs.opencv.org/4.13.0/d1/de0/tutorial_py_feature_homography.html)
motivates the ratio-test/homography mechanism. The protocol freezes ratio 0.7,
at least 11 matches and inliers, 5-pixel RANSAC residual, inlier fraction 0.5 and
projected-mask IoU 0.5. These are Development choices, not calibrated confidence.
Either of the two presupplied references may support the candidate.

## Results

| Arm | Correct | Wrong | UNKNOWN | Commit precision | Extra online views |
|---|---:|---:|---:|---:|---:|
| PASSIVE | 0 | 0 | 8 | Not evaluable | 0 |
| FIXED_SWEEP | 5 | 1 | 2 | 83.3% | 24 |
| TRIGGERED_ACTIVE | 5 | 0 | 3 | 100% | 20 |
| TRIGGERED_VERIFIED | 0 | 0 | 8 | Not evaluable | 20 |

The action baseline's 16.7% online-view reduction is separate from verifier
efficiency. Reference setup costs another 16 supplied views. The verifier has
zero correct retention against the required 80%; error reduction against
TRIGGERED_ACTIVE is not evaluable because that baseline made no wrong commits.
Zero commits is not 100% precision. No old cohort or threshold was tuned.

Truth-proposed diagnostics, evaluated after runtime choices were sealed:

- Target boxes supported: 2/8 (SEVN002 and SEVN004).
- Nonoverlapping same-scene sibling boxes accepted: 0/8.
- Detector masks accepted when the provider target region was wholly outside
  the viewport: 0/61. These are clustered candidate diagnostics, not 61
  independent negative episodes or an open-world specificity estimate.
- The metadata-only cross-query label-absent controls were not executed; label
  absence alone would not establish physical absence.

Two correct runtime proposals failed even though the corresponding human target
box had geometric support: reference extent and the predicted mask need not
describe the same amount of door surface. The remaining unsupported target boxes
also show limited cross-view matching evidence. This separates the mask/extent
compatibility problem from the feature-availability problem; neither supports
lowering a threshold on these consumed images.

## Execution and validation

The user's existing CARLA process occupied GPU capacity (524 MiB free at freeze).
All four arms used the same CPU backend, weights, preprocessing and thresholds;
the first real portal inference took 0.503 seconds. The CPU adapter changes only
the inference device and device synchronization while executing the frozen mask
postprocessor. Torch/OpenCV versions differ from the earlier 24-episode run;
comparisons here are within the new shared runtime, not against old-run numbers.

A NumPy scalar serialization error interrupted episode 2 after episode 1 had
been checkpointed. A versioned implementation-only repair normalized NumPy
scalars with `item()`, retained episode 1, and repeated only incomplete episode 2
under the declared deterministic Development recovery contract. Both protocols
and the failed log remain available. No scientific rule changed. All eight
episodes completed, the process exited successfully and its owner lock was
removed.

Targeted checks cover known translated-target acceptance, rejection of a wrong
region and a target-absent image, NumPy result serialization, source disjointness,
and the final action/selection/denominator audit. No Android or full-repository
test was needed for this isolated research runner.

Reproduction uses the repaired protocol; a terminal output cannot be overwritten:

```powershell
$env:PYTHONPATH='E:\linnan\linnan\artifacts.local\runtime\l10-ppocrv6-medium-v1\.venv\Lib\site-packages;E:\linnan\linnan\artifacts.local\runtime\ocr-gpu-ort'
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True'
$env:HF_HUB_OFFLINE='1'
& E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe -u research/active/l10-r0/l10_sevn_reference_commitment_episode.py --protocol research/active/l10-r0/l10_sevn_reference_commitment_protocol_v1_serialization_fix.json --archive F:\ba-data\SEVN\high-res-panos.zip --work artifacts.local/work/l10-sevn-reference-commitment-v1 --output research/active/l10-r0/l10_sevn_reference_commitment_result_v1.json
```

The run retains logs, checkpoints and backend receipts under
`artifacts.local/work/l10-sevn-reference-commitment-v1/`. They are recovery and
diagnostic evidence, not disposable capacity. No task-owned process remains.

## Inheritance and next decision

`NEGATIVE_CONTROL`: presupplied-reference SIFT/homography plus whole-extent IoU
must not be an unconditional hard commit gate for these predicted portal masks.
Keep the source separation, episode comparison harness and frozen failure for
future falsification. Keep the unchanged triggered policy available as a
component; this eight-episode result does not override earlier failures.

The task-local structured disposition is in
`l10_sevn_reference_commitment_disposition_v1.json`. Central inheritance
assignment was attempted through `tools/knowledge.py`, but its prerequisite
validation rejected a pre-existing input-fingerprint mismatch at
`experiments/index.jsonl:252` (the 2026-09-01 complementary-confirmation freeze).
Its inputs were not changed by this task. Central registration therefore remains
pending; all existing shared-registry bytes were preserved, and this unrelated
failure did not prevent scoped experiment delivery.

The next mechanism needs evidence that distinguishes missing reference matches
from contradictory identity, while keeping local identity support separate from
complete endpoint extent. Do not rescue this cohort by adjusting ratios, inlier
counts, crop selection or overlap thresholds. No provider independence, physical
motion benefit, entrance ownership, arrival, handoff, device readiness, user
benefit or safety claim follows.
