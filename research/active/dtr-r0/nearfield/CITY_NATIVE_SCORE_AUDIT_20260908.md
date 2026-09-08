# Native bollard fit-versus-transfer diagnostic

2026-09-08. Engineering decomposition of the completed native-only and
[replay/thin fits](CITY_NATIVE_REPLAY_20260908.md). No new fitting, capture,
threshold selection, or model promotion. Existing consumed Development data
remain consumed. Original checkpoint and both completed final checkpoints are
the only compared weights; saved DEV operating points remain fixed.

## View and scale audit

The four independently reliable native TRAIN bollard BODY positives are frames
6,8,11,13. The consumed route has two such positives, frames6,7. Native target
masks retain UNKNOWN and label only visible surfaces inside the body-height
query, not the entire object silhouette. The withdrawn meter verification is
not reinstated or counted as a model failure.

| Source/frame | Camera yaw | Horizontal target distance | BODY positive pixels at640x360 | BODY support box width at256x144 |
| --- | --- | --- | --- | --- |
| TRAIN6/8 | 0 degrees | 1.502 m | 486 / 484 | 6.8 px |
| TRAIN11/13 | 0 degrees | 2.401 m | 184 / 183 | 4.4 px |
| Route6 | 90 degrees | 2.064 m | 232 | 4.8 px |
| Route7 | 90 degrees | 1.576 m | 430 | 6.4 px |

Both sources use the same native Metal_Bollard_04 mesh on different original
instances. Camera height2.43m and pitch-5 degrees match. Route distances and
positive-support box sizes lie within the TRAIN ranges. This does not support
a simple out-of-range distance/size explanation. Widths above are projected
bounding-box dimensions, not a measurement of preserved information after BOX
resampling.

Visual inspection of all six RGB views confirms a substantial composition
difference: TRAIN looks along a row of bollards and the street; the route looks
across the row toward a different background. Camera direction, target rotation,
instance and surroundings all differ. This is a coverage gap and a possible
transfer explanation, not an isolated causal test of camera yaw or background.

The scientific comparison figure and exact poses/mask hashes are in
`artifacts.local/nearfield/city-native-score-audit-20260908/view-comparison.png`
and `view-audit.json`; `view_audit.py` reproduces this read-only figure.

## Score audit: exposed positives fit, route transfer fails

New inference evaluated unchanged TRAIN45 once per original/native-only/replay
checkpoint. DEV45 and route40 probabilities were read from existing caches.
No thresholds were changed. The table shows BODY probability for the four
TRAIN positives and two route positives, in fixed sample order.

| Model | TRAIN6 | TRAIN8 | TRAIN11 | TRAIN13 | Route6 | Route7 | Fixed DEV cutoff |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original | 0.4920 | 0.4918 | 0.4926 | 0.4913 | 0.4922 | 0.4925 | 0.493135 |
| Native-only | 0.8336 | 0.8677 | 0.9775 | 0.9549 | 0.7487 | 0.5225 | 0.977686 |
| Replay + thin | **0.9693** | **0.9652** | **0.9903** | **0.9843** | **0.7727** | **0.6796** | **0.929635** |

At each model's already locked DEV cutoff, original and native-only alert on
0/4 TRAIN thin positives; replay alerts on **4/4**, with joint alert/support
overlap **4/4**. All three remain **0/2** route alerts and joint hits. At the
unchanged historical BODY cutoff 0.982388, replay alerts on 2/4 TRAIN positives
and 0/2 route positives. Thus replay's residual route miss is not simply a
failure to fit the four exposed positives.

Original support overlap is 0/4 TRAIN and 0/2 route. Both adapted models have
4/4 TRAIN and 2/2 route overlap at support threshold0.5, but only one support
peak hits the target in each source (1/4 TRAIN, 1/2 route). Some overlapping
support does not imply precise localization or successful near-alert transfer.

Whole-TRAIN45 at the same fixed DEV policies provides context; cells are
`TP/positive; FP/negative`:

| Model | BODY | HEAD |
| --- | --- | --- |
| Original | 5/31; 0/14 | 0/20; 0/25 |
| Native-only | 18/31; 0/14 | 20/20; 0/25 |
| Replay + thin | **26/31; 1/14** | **18/20; 0/25** |

All four replay TRAIN thin scores exceed every known TRAIN, DEV and route BODY
negative. In contrast, replay Route6's score0.772687 has **5/17** DEV negatives
at or above it; Route7's score0.679635 has **6/17**. Their route-negative ranks
are 1/22 and 2/22 respectively. These are diagnostic negative counts **at each
target's score**, not the actual FP count at the locked threshold. Actual
replay DEV-selected BODY FP remains 0/17 and route FP remains 0/22.

Recovering Route6 by only lowering a scalar BODY threshold to its score would
therefore admit at least5/17 DEV negatives (29.4%); recovering Route7 admits at
least6/17 (35.3%). Either exceeds the frozen DEV10% rule. No such alternative
operating point was adopted or evaluated as a new selected policy. The missing
near-alert separation, not an allowed threshold adjustment, is the observed gap.

## Evidence and limits

`tools/audit_city_native_scores.py` is reusable and performs no optimizer steps.
The audit froze `scores-v1/input-manifest.json` with intent, exact inputs,
checkpoint hashes and existing cutoffs before new TRAIN inference. New scores,
all per-target negative ranks, whole-TRAIN metrics and receipt are under
`artifacts.local/nearfield/city-native-score-audit-20260908/scores-v1/`.

Actual CUDA execution: RTX5060 Laptop GPU, Torch2.11.0+cu130. Three model-load/
TRAIN-inference durations were 6.586s original (including first-use loading),
0.360s native-only and 0.357s replay; total audit11.240s. These are not comparative
steady-state model-speed measurements. Process exited successfully; no new
fit or persistent model process remains.

Checkpoint hashes and exact shared-original cache parity were checked before
inference. A post-inference supplemental check verified historical model-source
hashes, label-manifest/array hashes, locked-DEV hashes and exact cached metrics
for all16 fit/arm/domain/policy comparisons; no inference was repeated. These
checks now run before inference in the reusable tool. Original fit receipts did
not store NPZ byte hashes: audit-time byte hashes plus exact metric parity are
the available cache evidence, not a claim of a previously recorded byte seal.
The supplemental-validation file preserves this timing distinction. An
UNKNOWN-aware negative-rank check and scoped whitespace check passed.

Input manifest SHA-256:
`e54d080cde8fa7e1ab7ebafb817536e8b65806bdd5725b09abe41d4d53afc217`.
Result SHA-256:
`8d5af5cd7c0b7fb85b6956d86baed27b399ccc419ed78f413e92144a9404dae1`.

Decision: keep **RETAIN_ORIGINAL** from the completed recipe evaluation. Replay
fits the repeatedly exposed thin positives, but support/near response fails to
transfer to the two consumed route views at the fixed false-positive budget.
Geometry/view evidence identifies a possible coverage gap without isolating
its cause. This diagnostic neither promotes the model nor authorizes another
fit, capture, threshold change or universal safety claim.
