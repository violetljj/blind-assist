# Multi-candidate local verification Development successor

2026-10-15. User explicitly authorizes this successor after the completed
280-image pilot. Reuse its exact observations, checkpoint, ree head, cal80/16
and eval160/24 partitions. All evaluation images have been consumed; this is
adaptive Development, never fresh confirmation. Do not modify the old plan.

## Decision and budget

Start15:22 HKT; finish by16:52 HKT (90min). New downloads0, no paid resources,
training or parameter search. New GPU process wall<=900s plus<=300s for observed
implementation repairs, CPU scoring<=600s and review<=300s. Existing local GPU
runtime; require>=3GB free VRAM before launch. Payload:
artifacts.local/work/functional-parts-multicandidate-20261015/.
Keep old results and any failures; release owned processes/locks on completion.

Old AGD point proxy .3620 and IIT3/24 leave headroom. One IIT image is4.17
percentage points, so report paired event counts and avoid certainty from
small differences. The question is whether retaining alternative parts and
verifying them beats BOTH equal-call fixed crops and TTA, and whether rejection
helps at matched coverage. Candidate-oracle improvements alone are not method
success. No positive result is required; stop after the fixed comparisons.

## Observation-only proposal and views

Use old original-image ree minmax map B. Pixel-center coordinates and half-open
half-width/half-height crop geometry follow run_baseline.py. Select three peaks
greedily, suppressing normalized elliptical distance<=0.2 in coordinates x/W,
y/H after each selected seed; skip any duplicate crop box. Ties row-major.
Candidate1 is the original peak. If fewer than3 remain, fill from deterministic
fixed crops with distinct boxes; record any residual shortage, never invent
evidence. Constant maps use fixed crops and flag fallback.

Three fixed crops are centered at .25,.5,.75 along the image's longer dimension,
and .5 on the shorter dimension; width>=height chooses horizontal. Fixed-crop
anchor c is the B peak INSIDE its box, not a possibly empty geometric center.
Candidate anchor c is its proposal seed. TTA uses base, inverse full-image
horizontal flip, center crop, inverse horizontal flip of the center crop.

Each matched system costs four224x224 calls: base plus three local views or
three TTA views. Actual shared/cached calls and wall time are reported separately
from deployment-equivalent cost. All new crops capture the existing classifier's
36x14x14 logits via a read-only forward hook; no model mathematics is changed.
Old cached crop maps lack these logits, so rerun needed crops, deduplicating
identical image/action/box/flip within this run. Cached base/flip stay usable.

## Fixed local verifier and configurations

For each crop, L is its ree resize/minmax map; q its local argmax mapped back to
the image. e=softmax(mean_hw(all36raw logits), temperature1)[query action]. This
uses the released classifier's GAP training convention. e is relative action
support, NOT calibrated presence probability; actions can co-occur.
B(q) and L(c) use pixel-center bilinear sampling of maps normalized to max1.
Spatial score a=sqrt(B(q)*L(c)); combined s=(B(q)*L(c)*e)^(1/3).
Zero/degenerate support yields0. Select largest score, ties proposal order,
and emit its q. No GT enters anchors, proposals, selection or confidence.

Fixed methods, all except original reference use the same four-call budget:

- B0: cached original argmax, one-call reference only.
- B1: equal-mass base/fullflip/center/flipcenter fusion argmax.
- B2: equal-mass base/three fixed-crop fusion argmax.
- B3: fixed crops selected by the SAME combined verifier s.
- C0: equal-mass base/three candidate-crop fusion argmax.
- C1: first candidate's local q, ignoring alternatives (compute-matched ablation).
- C2: candidate selected by spatial a.
- C3: candidate selected by semantic e only.
- C4: candidate selected by combined s (primary proposed method).

All fusion maps use old resize/minmax->L1 convention and actual crop projection,
with .25 mass per view. Selection methods return a point, not a fabricated
whole-image distribution; native heatmap metrics only for actual fusion maps.
Record all candidate q, anchors, scores and raw maps for error analysis.

## Evaluation and refusal

Reuse old AGD soft point quality/localGTmass/peakdistance and IIT5|9 native-hit
metrics, separate datasets and keep native-empty5/24. Report paired changes
against B1,B2,B3, not only the weak B0. For C0–C4 report candidate q oracle
best-quality/max-hit and selection gap as evaluator-only upper bound. Initial
seed oracle is separately diagnostic. Neither oracle is a deployable method.

For B0/B1/B2/C0 entropy confidence uses64x64 probability of their output maps.
C1 uses its crop's s; C2 uses selected a; C3 selected e; B3/C4 selected s.
For B0 also preserve old flip-consistency reference. Each dataset/method gate
is calibrated to retain>=80% of its original calibration split, same nth-score
rule/ties as before; seal thresholds before this successor's eval aggregation.
Report actual eval coverage, retained point quality, total useful outputs and
native-empty accepted. Also report complete risk-coverage/AURC and fixed
50/70/80/90/100% equal-count ranking diagnostics: sort confidence descending,
ties sample_id, retain ceil(f*N). This uses no GT to choose rankings, but these
eval-ranked counts are NOT deployable calibrated thresholds. No GT threshold
search or equal-coverage superiority claim from unmatched gate coverage.

Same-image paired bootstrap2000 draws, fixed seed20261015, for C4 point-quality
difference versus each B1/B2/B3; intervals are descriptive reused-Development
uncertainty, not independence or multiple-comparison proof. Reuse old size
strata cutoffs without refitting. Show rescue/harm cases chosen automatically.
Validate geometry and hook selected-class equality; independently recompute
aggregates and a few point samples. No new variants after seeing outcomes.
