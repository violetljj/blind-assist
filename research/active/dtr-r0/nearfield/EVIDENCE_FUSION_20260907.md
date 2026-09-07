# NF-G4: cane-complementary forward awareness and optional ground evidence

Product/research scope update requested by the user: class-agnostic forward
obstacle awareness complementary to a cane. Prioritize elevated/suspended
obstacles,body/head protrusions,walls and large forward obstructions; then
multi-height poles/supports. Retain low/ultra-low observations as secondary
compatibility evidence. Dynamic obstacles and earlier warning are priorities
for future temporal validation, not demonstrated by the current static tests.

Do not claim measured cane coverage or label historical examples CANE_DOMINANT
from height alone. Old low(.065–.65),body(.65–1.40),head(1.40–1.85)m boundaries
remain frozen. Knee-height hazards are not discarded. New task-aligned source
labels/height subdivisions require an explicit future protocol; this change
does not rewrite past denominators or turn selected scores into confirmation.

## Fixed cache experiment

EXPLORE, consumed NF-G3 eighteen views and NF-G1 eleven frames,plus existing88
analytic negative/geometry controls. Compare Raw,Ground,union,and conditional
fit switch. No model,inference,UE,new capture,threshold fitting or temporal
smoothing. Cached branch observations are sufficient to test nine-cell fusion;
CPU boolean/JSON work is TASK_NOT_GPU_SUITABLE. No speedup of depth inference
is claimed. A separate receipt-bound suspended-bar diagnosis uses cached depth.

Branch availability is separate from obstacle evidence. Union preserves positive
Raw or Ground observations. An unavailable ground branch is UNKNOWN,not empty
space. Directional output can be NEAR while height is UNKNOWN. Raw still uses
known camera height/pitch; it is not a newly validated ground-free detector.

Expose RAW_ONLY,GROUND_ONLY,BOTH,NONE provenance. Different height sets are
branch disagreement,not physical conflict without same-surface association.
Ground height bands remain separate from unassigned raw height hypotheses.
Do not claim one alert per physical object or average incompatible heights.

Retain full historical9-cell reference comparison explicitly as diagnostic
height hypotheses. Also score three directional alerts against native positive
directions; do not count a direction-only alert as confirmed height recall.
Report body/head diagnostic cells as a subgroup under the new emphasis, not as
measured VISION_COMPLEMENTARY recall or a retroactively selected headline.

Mechanical existing-set calculation predicts52/87 union cells with0 FP on NF-G3;
this is consumed expected behavior,not independent evidence. Execute the actual
merge and verify provenance/UNKNOWN and unknown-height outputs. Check old11
frames and88 controls for regressions. Retain simple union only if it preserves
the observed complementarity without added reference FP; stop at this comparison.
Do not relax thresholds or automatically retrain after the suspended-bar audit.

## Results

Implemented `evidence_fusion.py`, verified through its real merge path, and
scored once in `artifacts.local/nearfield/fusion-20260907-v1/comparison`.

| NF-G3 arm | Legacy hypothesis TP/FP/FN | Direction TP/FP/FN | Body/head hypothesis TP/FP/FN |
| --- | --- | --- | --- |
| Raw |22/0/65 |10/0/34 |15/0/32 |
| Ground |37/0/50 |33/0/11 |4/0/43 |
| Union |52/0/35 |39/0/5 |16/0/31 |
| Conditional switch |52/0/35 |39/0/5 |16/0/31 |

Denominators remain87 positive cells/162 cells,44 positive directions/54
directions,and47 body/head positive hypotheses. Body/head16/47 is a diagnostic
subgroup, not verified-height recall or measured cane-complementary performance.
Direction collapse can conceal a missed high obstacle when another lower surface
already triggers that direction;39/44 must not replace suspended-hazard scoring.

Union provenance:15 RAW_ONLY,30 GROUND_ONLY,7 BOTH,110 NONE cells. Six positive
directions have UNKNOWN height; wall alerts are retained without claiming a
trusted body/head assignment. Available ground bands and unassigned raw height
hypotheses are exposed separately. No physical conflict is asserted without
surface association. The historical52 cell hypotheses do not become52 verified
height observations simply because the directional alert is useful.

Union and conditional switch have identical alert sets here, but cell UNKNOWN
counts differ37 vs28: conditional selection can omit uncertainty from the other
available branch. Product-direction state additionally preserves UNKNOWN when
ground is unavailable and no positive evidence exists. UNKNOWN is never FAR.
The sample does not establish that union has superior empirical recall to the
switch. Retain union as the minimal additive component matching the requested
evidence-preservation contract, not as a new demonstrated state-of-the-art method.

Old eleven frames remain21/26,0 FP with both fusion modes. All88 analytic cases
remain240 TP,1 FP,0 FN: the coherent3x3 artifact remains a false alert. Union
preserves erroneous evidence as well as correct evidence; it is not an automatic
error-rejection mechanism. No App/default model was changed.

## Suspended-bar diagnosis and priority

`bar-diagnosis.json` verifies the frozen NF-G3 receipts and reproduces both raw
and ground alert/state outputs before attribution. All1,054 native supported
center/head near pixels have finite predicted depth >=12 m, outside the fixed
validity range. Native forward P50 is1.960 m; predicted forward P05/P50/P95 is
13.770/14.288/14.848 m. Both branches have zero eligible center/head candidates,
including zero weak near candidates. Three-pixel support is not the bottleneck.

Ground changes median reconstructed height from0.983 to1.523 m; range still
invalidates every reference pixel. Raising the12 m validity limit alone would
not create a <=3 m alert. This single reference-cell diagnosis does not prove
instance identity or determine whether model/domain choice, local ambiguity or
other frontend behavior caused the depth error. Next prioritize a bounded
forward-geometry/model diagnostic for this suspended structure; do not return
to ultra-low boundary tuning. No model replacement was tested in this run.

## Delivery and limits

Six focused fusion tests passed: missing-ground positive retention with unknown
height, unavailable absence not FAR, complementary provenance/input immutability,
unassociated height disagreement, conditional/union distinction, and invalid
state/mode handling. The executed comparisons cover all cached cells and all
analytic controls. Branch-cache read plus fusion/scoring took0.057 s internally;
including CUDA analytic encoding it took1.011 s. Bar attribution took0.755 s
internally. These are cached research costs, not complete live perception latency.
No inference,ground refitting,UE or threshold sweep occurred; owned processes exited.

Project wording updated in root README,project state,current decisions,DTR current
and ledger entry,documentation map,and demo guide. The packaged detector/model
card remains unchanged. New priorities are explicit research/product intent;
no measured cane coverage,dynamic performance,95% target result or real-user
effect is inferred. Historical reports remain unchanged.

Central registration failed before mutation on the existing index line252
fingerprint mismatch. `registration.log` preserves it; local receipts and report
are complete, but no new registered terminal is claimed.

Comparison result SHA-256:
`6e0fe898b6edf79686a194084c2bc7555d1b406099d5d4c29d82ba0b10a298ea`.
Bar diagnosis SHA-256:
`aaf0078ff601d6317921e80c0422eb9ad94aa7ac6c1726efe1c50dddbec5b366`.
The comparison's `protocol.md` preserves pre-score criteria and receipts record
executed code/input hashes. The source expansion,body/head subgroup and fusion
scores are all explicitly consumed Development,not fresh confirmation.

Reproduction with existing CUDA Python and a fresh output directory:

```powershell
python -m unittest discover -s research/active/dtr-r0/nearfield -p test_evidence_fusion.py -v
python research/active/dtr-r0/nearfield/run_evidence_fusion.py --g3 artifacts.local/nearfield/distinct-views-20260907-v1/evaluation/result.json --g1 artifacts.local/nearfield/ground-anchor-20260907-v1/result.json --original artifacts.local/nearfield/representation-20260907-v1/result.json --output artifacts.local/nearfield/fusion-new-run/comparison
```
