# MZ6: smoothing loses detections; sparse correspondence never activates

2026-09-10 EXPLORE. **Retain current-frame MZ5. The completed200-frame pilot
does not establish a temporal upgrade.** Mean3 loses two far true positives;
the compatible-history recipe makes zero packet completions. More fundamentally,
the fixed readout misses all25 BODY_FAR and24 HEAD_FAR thin-pole opportunities
even with clean returns. Restoring the original clean packet alone would not
repair those misses. This does not reject all spatial or temporal fusion.

## Bounded source and fixed comparison

[Protocol](MZ6_SHORT_SEQUENCE_PROTOCOL_20260910.md) was fixed before new model
outputs. Four25-frame target clips each have a matched25-frame target-absent
control: head-bar approach, body obstacle entering laterally, thin pole against
a background wall, and body obstacle leaving. Frozen Willow, level HFoV100deg,
eye1.7m, nominal12Hz. All are settled scheduled samples, not continuous real-time
walking. Native depth supplies generic45deg8x8 two-return packets and independent
visible-pixel support labels. Exact poses/objects remain evaluator-only.

[Source admission](../../../../artifacts.local/work/mz6-short-sequence-20260910/capture-v3/source-admission.json)
accepts200/200 frames and8 clips: actual target-position/native support and file
checks pass, all200 native asset/shader/render readiness observations are READY,
40 sampled frames were visually inspected, map hash unchanged. First failure:
high Zen port became negative (zero frames). Second:200 raw cold-cache frames
had initially incomplete materials (zero admitted). Both are preserved. The same
spec was recaptured after receipt-bound warm-cache reuse and an existing hashed
native readiness gate; no model saw either failed attempt. No source redesign.

The unchanged compact MZ5 is CURRENT. MEAN3 averages current and up to two prior
logit vectors. COMPATIBLE tracks RGB corners with forward/backward LK flow and
tests background-range compatibility before completing a missing foreground
hypothesis from one of two prior raw packets. No recycled completions, future
data, UE pose or target identity enter it. The zero-motion correspondence
falsifier also runs once. Zero fitting, threshold/weight selection or rescue.

## Task effects

Clean and missing-return inputs produce identical flags in every arm (logits
are not identical). Thus this table applies separately to each condition.
Each recall cell gives TP/positive opportunities. Raw event truth is>=3 native
visible pixels in the BODY/HEAD range query; a false bit is not certified CLEAR.

| Method | Exact /200 | BODY_NEAR | BODY_FAR | HEAD_NEAR | HEAD_FAR | Wrong-far /18 | Cross-body /44 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CURRENT |143|2/9|6/36|9/9|0/38|0|3|
| MEAN3 |142|1/9|4/36|9/9|0/38|0|3|
| COMPATIBLE |143|2/9|6/36|9/9|0/38|0|3|
| Zero-motion falsifier |143|2/9|6/36|9/9|0/38|0|3|

MEAN3 gains2/loses3 exact frames. BODY_NEAR false positives fall5->2 and BODY_FAR
3->2, but it also loses one near and two far TPs. HEAD_NEAR FP remains19.
All100 target-absent control outputs match no positive visible support; however,
75/100 controls have no valid simulated return within4m. They must remain
UNKNOWN observations, not sensor-certified clear space. Across200 frames,
84 have no valid packet; all200 contain some invalid native pixels elsewhere.
The thin-wall control has valid returns and no positive BODY/HEAD support.

These new Willow fixtures are a different controlled distribution from the old
1500 EVAL frames. Do not compare143/200 with1378/1500 as a paired improvement.
The new source exposes limited fixed-readout transfer, not natural performance.

## First detection and release require the right interpretation

- **Approach:** HEAD_FAR is missed for all14 eligible samples, but HEAD_NEAR is
  asserted throughout all25 frames, including16 false near assertions. HEAD_ANY
  therefore hits all23 in-range positive samples. This is distance confusion,
  not evidence that no head warning occurred. The actual near interval is hit9/9.
- **Entering:** BODY_FAR starts at sample14 and both CURRENT and MEAN3 first hit
  there; subsequent correct coverage falls6/11->4/11. BODY_ANY retention falls
  10/11->6/11. Smoother output does not imply more persistent useful coverage.
- **Thin pole:** neither BODY nor HEAD has a hit, clean or stressed. The25 BODY
  and24 HEAD positives remain missed; no first-hit improvement is established.
- **Exiting:** the positive BODY_NEAR interval is samples0–8. CURRENT first hits6,
  MEAN3 first hits8: two nominal samples (0.167s) later, with retention2/9->1/9.
  This episode is left-censored at clip start. BODY_NEAR and BODY_ANY are two
  metric records for this same physical episode, not two delayed obstacle events.
  All methods are silent from sample9 and confirm two-sample silence at10.

Silence after an event never detected (such as HEAD_FAR on approach) is not a
successful release. BODY/HEAD unions and right-censored intervals are reported
alongside range bits. These are scheduled-sample response times, not measured
camera-to-alert latency or safety clearance. The limited exit opportunity does
not establish robust temporal deactivation in general.

## Why history had no effect

The declared low-area first-bin stress has85 eligible zone-frames, all in the
thin-pole clip, and deletes39 nearer returns (seed107), retaining background.
At30/39 deletions a similar same-zone near return exists in the preceding two
raw observations. That is an evaluator opportunity diagnostic, not proof of
correct return ownership or a privileged predictor input.

Yet **COMPATIBLE completes zero zones**, clean and stressed. A post-run gate
diagnosis finds2042 clean and1897 stressed range-compatible source/destination
pairs. None have the required three local correspondences; clean has none with
even one, and stressed has only four pairs with one match each. Thin straight
structures against this plain wall supply inadequate corner-based association
for this recipe. Thresholds are unchanged. The zero-motion falsifier also never
activates, so it cannot establish the value of motion correspondence here.

The pressure changes thin-pole logits by up to0.8923 but changes no flags.
The clean fixed-readout already misses every thin-pole positive; exact restoration
to its original clean packet would still leave those measured misses. This
pilot has real deleted-return and historical-return opportunities, but no
demonstrated usable correspondence and no clean-readout recovery response.
Do not describe an inactive transport module as validated robust temporal fusion.

## Decision, cost and verification

The candidate fails the required-gain criterion; keep MZ5 as the scoped controlled
baseline and retain this recipe/source as a diagnostic control. Before another
temporal fit, a materially different mechanism must establish usable local
support on thin surfaces and a readout that responds correctly to restored
evidence. This run stops without loosening thresholds, adding fits, enlarging
the source or promoting an App/alert/hardware change.

Actual capture script154.70s excludes editor startup. Frozen RGB extraction plus
sensor/evaluator preparation took19.65s on RTX5060 Laptop CUDA. All scoring and
four history/control passes took14.49s; one200-frame compatible pass took4.50–4.58s
on CPU, since this OpenCV build has no CUDA backend. These offline stage totals
are not per-frame deployment latency and do not recharacterize the132872 readout
parameters as whole-system size.

Five focused tests cover actual translated-image flow/textureless abstention,
range compatibility, raw-history expiry, clip reset, causality and episode
censoring. [Post-run analysis](mz6_sequence_analysis.py) independently reconstructs
all1600 saved prediction-row metrics and400 Mean3 rows; all agree. Original
alerts remain stored separately and unchanged. Source/checkpoint/packet/prediction
hashes and failure receipts are preserved. UE/Zen/worker processes and ports are
released; all task-owned cache/temp trees were removed and removal verified.

- [Results](../../../../artifacts.local/work/mz6-short-sequence-20260910/evaluation-v1/result.json),
  [audit and gate diagnosis](../../../../artifacts.local/work/mz6-short-sequence-20260910/evaluation-v1/analysis/audit-and-diagnosis.json),
  [event timeline](../../../../artifacts.local/work/mz6-short-sequence-20260910/evaluation-v1/analysis/event-timeline.png).
- [Preparation receipt](../../../../artifacts.local/work/mz6-short-sequence-20260910/evaluation-v1/prepare-receipt.json),
  [scoring receipt](../../../../artifacts.local/work/mz6-short-sequence-20260910/evaluation-v1/score-receipt.json),
  [resource cleanup](../../../../artifacts.local/work/mz6-short-sequence-20260910/cleanup-receipt.json).
- [Capture source](capture_mz6_short_sequence.py), [launcher](launch_mz6_short_sequence.py),
  [fixed spec](mz6_short_sequence_spec.py), [causal module](mz6_causal_packet.py),
  [sensor/extraction/inference/scoring](mz6_sequence_evaluate.py).

Reproduction: generate a new spec under the canonical MZ6 artifact root, launch
capture in a fresh child, and perform readiness/native/visual source admission.
Then run `mz6_sequence_evaluate.py --phase prepare --capture <admitted-capture>
--output <fresh-evaluation>`, followed by `--phase score` on that evaluation and
`mz6_sequence_analysis.py --run <evaluation>`. Local assets/checkpoints and the
native capture helper are required. Never overwrite completed or failed attempts.

Registration delivery validation uses a HEAD-based staged view to preserve the
unrelated worktree edits:10 knowledge unit tests and library validation pass,
with zero invalid experiment associations. The34-case decision check has an
existing history-retrieval recall0.70 below0.80; an unchanged-HEAD replay reproduces
the same failed cases and outputs (runtime aside). This unrelated pre-existing
gap is retained, not repaired or reported as passing. Logs are preserved in
`artifacts.local/work/mz6-delivery-20260910/`.
