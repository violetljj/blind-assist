# DTR X6-X9 static-world and RGB source authority

Date: 2026-08-29

## Decision

X3's full-replay attribution showed that static pseudo-motion, not missing
route forecasting, dominates its false alerts.  X6 therefore added one
independent source observation: a causal raw-LiDAR world-occupancy anchor.  It
did not alter X3 velocity, route geometry, lifecycle, thresholds, or scorer.

The 60-frame X6 falsifier passed.  This authorized exactly one full replay of
the frozen anchor.  X7 then reduced X3 false segments from `94` to `72` while
preserving all six CONTACT recoveries, `3.816 s` median first-alert lead, and
`8/18` dropout recovery.  The full gate nevertheless failed because 72 false
segments remain and Event F1 is only `14.29%`.

Accept both results without a parameter rescue:

- `DTR_X6_STATIC_WORLD_PERSISTENCE_FALSIFIER_GATE_MET`;
- `DTR_X7_FULL_STATIC_WORLD_ANCHOR_GATE_NOT_MET`.

The anchor has real source-level effect, but is not a sufficiently selective
dynamic-risk representation.  Do not tune its map age, occupancy radius,
motion bounds, route, horizon, lifecycle, or scorer on this opened cohort.

## What changed

X6/X7 use sealed X3 cells only as candidate locations and retain their velocity
unchanged.  For a candidate at frame `t`, raw LiDAR and native ego poses ask
whether the same world location was occupied:

1. recently, within the already frozen raw-flow history bound; and
2. across enough elapsed time for a cell moving at the frozen minimum dynamic
   speed to have crossed one frozen BEV-cell diagonal.

Only a location satisfying both causal past-occupancy checks is vetoed as a
static-world anchor.  Missing or unsupported history keeps the X3 candidate.
This changes the information source rather than deriving another confidence
rule from X3 flow.

## X6 decisive falsifier

The single opened 60-frame roster contained the localized positive frames and
one representative frame for each frozen X0 source-error unit.  X6 processed
`10,964` X3 cells, removed `4,836`, and retained `6,128`.

| check | frozen requirement | observed |
|---|---:|---:|
| correct positive frames | `>=2` | **3** |
| correct route-entry frames | `>=2` | **2** |
| source-error units suppressed | `>=24/34` | **`26/34` (76.47%)** |
| source compute p95 | `<=0.06961 s` | **`0.06435 s`** |

All four checks passed.  This was post-outcome Development diagnosis, not a
confirmation result, and frame-local suppression was not yet event scoring.
It authorized one frozen full replay only.

## X7 full-replay terminal

X7 materialized all 4,811 timeline frames across the six already opened
source-disjoint sequences.  The first four frames of each sequence are causal
five-scan warm-up and are fail-closed with zero X3 cells; the remaining 4,787
frames have X3 source support.  Across the full timeline the anchor removed
`340,734/642,217` candidate cells and retained `301,483`.

| arm | CONTACT recall | false segments | Event F1 | median first lead | dropout recovery |
|---|---:|---:|---:|---:|---:|
| frozen M1-PDC | `4/6` | **25** | **22.86%** | `2.291 s` | `5/18` |
| X3 lag-Floxel | **`6/6`** | 94 | 11.32% | **`3.816 s`** | **`8/18`** |
| X7 static-world anchor | **`6/6`** | 72 | 14.29% | **`3.816 s`** | **`8/18`** |

X7 preserves the information gain that X3 created and removes 22 false
segments, so a causal world anchor is a useful component.  It still misses all
three selectivity checks: false segments are above 16 and above PDC, while F1
is below 35%.  The result closes X7 as a standalone source rather than erasing
the demonstrated anchor effect.

Read-only overlap with the sealed X3 attribution assigns 42 of the 72 remaining
false segments to `STATIC_PSEUDO_MOTION`, 14 to bad magnitude, one to reversed
direction, five to real noncritical movers, four to wrong bindings, four to
fragmentation, one to route geometry, and one to a new or unmatched range.
The next source must therefore add an independent observation of scene motion;
it cannot be another same-flow authority or a retuned static-map radius.

## X8 independent RGB static veto

X8 supplied that second observation without changing X7 velocity or downstream
logic.  Sparse forward/backward LK tracks in the synchronized stitched RGB
panorama compare the observed image displacement with two ego-pose-conditioned
hypotheses: a rigid static-world reprojection and the X7 moving displacement.
A candidate is vetoed only when valid tracks support the static hypothesis,
place it closer than the moving hypothesis, and meet the unchanged C22 visual
confidence rule.  Missing images, invalid projection, or failed tracks retain
the X7 candidate.

The single X8 run reused the 60-frame X6 roster.  All requested stitched images
matched in all six bags; each bag also exposed all ten raw camera topics.  X8
processed 6,128 X7 cells, vetoed 1,725, and retained 4,403.

| check | frozen requirement | observed |
|---|---:|---:|
| correct positive frames | `>=2` | **3** |
| correct route-entry frames | `>=2` | **2** |
| source-error units suppressed | `>=24/34` | **`27/34` (79.41%)** |
| frame-local compute p95 | `<=0.06961 s` | **`0.03184 s`** |

Accept `DTR_X8_RGB_STATIC_VETO_FALSIFIER_GATE_MET`.  The observed suppression
is 13/18 static-pseudo units and 14/16 bad-flow units; the one real noncritical
mover is correctly outside the source-error gate and is retained.  Runtime
includes projection, LK, confidence, and veto only; bag scanning, image
matching, and decoding are excluded.  This pass authorizes one frozen full
replay, not a threshold sweep or an RGB performance claim.  The sealed hashes
are:

- X8 script: `f41eb6e6fb2cf46373cadf33ee0e27cbe72de8b8788a5f02dd7d5bd368cd9842`;
- materialization: `6f57d6f6154e7a5dadce9cd4bae31016933ee4ef8ebaa3a52f43b4456fe80bf6`;
- result: `30bc45bcc322b56f486eb96f45e9bdcc382e67567d603e0c6663c4916ab757a4`.

## X9 full RGB-veto replay terminal

The X8 pass opened exactly one frozen full replay.  X9 applied the unchanged
RGB static veto after every sealed X7 cell across all 4,811 timeline frames.
All requested stitched images matched; 4,803 frames had a previous consecutive
image and valid pose pair, while eight frames retained X7 candidates under the
frozen missing-evidence policy.  X9 processed 301,483 X7 cells, vetoed 62,757,
and retained 238,726.

| arm | CONTACT recall | false segments | Event F1 | median first lead | dropout recovery |
|---|---:|---:|---:|---:|---:|
| frozen M1-PDC | `4/6` | **25** | **22.86%** | `2.291 s` | `5/18` |
| X3 lag-Floxel | **`6/6`** | 94 | 11.32% | **`3.816 s`** | **`8/18`** |
| X7 static-world anchor | **`6/6`** | 72 | 14.29% | **`3.816 s`** | **`8/18`** |
| X9 X7 + RGB static veto | **`6/6`** | 64 | 15.79% | **`3.816 s`** | **`8/18`** |

The independent RGB source removes another eight false segments without losing
recall, aggregate median lead, or dropout recovery.  It still fails all three
selectivity checks: false segments remain above 16 and above PDC, and Event F1
remains below 35%.  Accept `DTR_X9_FULL_RGB_STATIC_VETO_GATE_NOT_MET`; close
this fixed visual veto without a threshold or camera-fusion rescue.

X9 used a corrected second freeze root before any worker started.  An initial
prepare-only freeze inherited X7's incorrect 4,787-frame assembler assertion;
it produced no ledger, lock, prediction, or score.  The executed v2 freeze
explicitly distinguishes 4,811 timeline frames, 4,787 X3-supported frames, and
24 fail-closed warm-up frames.  All six worker stderr logs are empty and all
processes and locks were released.

Key v2 SHA-256 values:

- X9 script: `be20007022cb3a9d561ce5dd470544dafbfd254b9134a693622a31ec858421fb`;
- executed freeze: `f253aee19d9f321e3f5f417233f28676fa6a64c32aa84df327b2b83b32061b35`;
- materialization: `09d2293c8fefe4a36e12fdac427cd25a8bcaa93c1251b90de6797dd336aa3826`;
- sealed predictions: `5e200b0e96fe7b9060580b7304e88e7fcdb15fdb4a7430b305cea10dc061e9e6`;
- result: `d8b4afcac35398132a53e98b03673e4ae0433c694451fab22d266e9091512eef`.

## Timeline-count amendment

The frozen X7 assembler initially compared 4,811 timeline rows with X3's 4,787
source-supported rows.  A separate amendment verified every sequence ledger,
the six `0..3` zero-cell warm-ups, the frozen script, and the freeze receipt,
then corrected only the materialization count.  It changed no candidate cell,
velocity, prediction, score, or gate.

Key SHA-256 values:

- frozen X7 script: `c9b5a58eafc732b692d2892adc5dbbb17a20e2b1240f86f114c48d63ec3e3f8c`;
- freeze: `57faf3768e95cf7b6f06357de89ef5e85249e54b36fde6d8388167dc0c429aa6`;
- corrected materialization: `8766f6d0e069cfba491c82aa01b970bb46a91ebb76d466aa7fcff0a203f934dc`;
- amendment receipt: `f2cd6f39b0616fb6b61886b24639f85986e573b0f75fbceaa097940c4f16642a`;
- sealed predictions: `420cc19a55070c7931d9d929e6cec3d7cb2870f82c712bb27fe05ec12dbd2843`;
- X7 result: `d35dabee7a5aabf08f75943b5f56d780ec50c2450589b1090d609480bd772ca3`.

## Claim boundary

X6--X9 are privileged Development evidence on the already opened C31/X0
six-sequence cohort. They establish that causal world persistence and
synchronized ego-rigid RGB residuals can each remove some static pseudo-motion
while retaining X3's recovered events. The RGB component is a sparse
frame-local source mechanism evaluated with public robot-camera data, not an
end-to-end RGB obstacle system. These results do not establish source-disjoint
confirmation, total decode/runtime latency, Android behavior, user benefit,
reliability, or safety.
