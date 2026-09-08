# Region-separated DEV and matched update-policy baseline

Pre-outcome EXPLORE protocol. Keep G13-D, RGB144x256, two near heads and support
heads unchanged. No augmentation, replay, new support loss, counterfactual loss,
resolution change, temperature scaling, threshold search on plaza, or new backbone.

TRAIN remains the original street750. The nearby contextual1000 frontage at
x-76/y15 is not independent of that source; none of it is assigned to DEV by
random image splitting. New DEV uses the actual east-south building frontage,
target x54/y-15, with translated/mirrored supported fixtures. Four-family native
canary verifies placement before one128 capture. The minimum camera XY separation
from TRAIN must exceed100m, with disjoint group/site identity. This is another
region of the **same map**, not independent-world evidence.

DEV has32 matched quartets, eight per crossbar/cabinet/inclined-member/hanging-sign
family. The32 groups are the deterministic first32 contextual-generation groups;
camera, group factors and relation enumeration are fixed before any model output.
Intended CLEAR/BODY_ONLY/HEAD_ONLY/BOTH each32; accept actual native visible-support
labels, never replace them by intended relation. Both heads need at least48 known
positives and48 known negatives. Preserve any assembly/visibility disagreement and
UNKNOWN masks. Native visible query semantics match City TRAIN; no instance-name
filter or RGB difference mask. DEV entire site stays together and is evaluator-only.

One seed17 comparison, final step200 checkpoints only:

| Arm | Parameter updates | Additional fits |
| --- | --- | ---: |
| A | Original all-parameter City step200, reused unchanged | 0 |
| B | RepViT frozen; projection/detail/support/near trainable | 1 |
| C | B for100 steps, then RepViT lr1e-6 for100; remaining layers lr1e-5 throughout | 1 |

All arms start from original G13-D seed17, not sequential warm starts from A or B.
Reuse the exact original200x32 sampling schedule and TRAIN750 cache; AdamW
weight_decay1e-4, head lr1e-5, ordinary near BCE+.25 existing globally-balanced
unknown-aware support BCE. All BN running buffers remain frozen. B also freezes
backbone BN affine parameters; C unfreezes them at101 with the backbone. Keep
head optimizer state at that transition. Check B backbone unchanged, all running
buffers unchanged and B/C complete model tensor equality after100 steps. Use
the original worker Torch2.9/cu128 environment. No early stopping or budget rescue.

DEV selection per arm/head: empirical FPR<=10%, then maximize recall, ties lower
FPR, then higher threshold. Comparison is inclusive and float64, including an
all-negative sentinel. UNKNOWN excludes only that head's unavailable labels and
is counted. Choose one whole checkpoint by maximum minimum-head recall, then
macro recall, lower macro FPR, then fixed A/B/C order; never splice model heads.
This tie policy and the100/100 C schedule are routine choices fixed before scores.
Report absolute metrics even if the best available candidate remains weak.

Secondary: group joint correctness, positive known-pixel support IoU/peak and
negative support activation, using unchanged .5 support threshold. Save DEV
thresholds/checkpoint identity before opening old plaza for new-arm diagnostics.
Plaza stays consumed and cannot revise thresholds or winner. No fresh final TEST
has yet been captured or accessed: DEV performance is selection evidence, not
an unbiased generalization result or model-promotion claim. A future fresh-region
test must follow the frozen candidate and thresholds without further selection.

Stop after the two200-step fits, selection and stated diagnostics; if DEV admission
fails, fix the source evidence before training rather than relabeling/omitting
difficult samples. Preserve checkpoints, schedules, source/label hashes and receipts.

## Executed DEV admission

The new capture contains128 frames in32 quartets: BODY64positive/64negative and
HEAD64positive/64negative, with16 HEAD positives in each of the four families.
Native visible-support labels agree with the intended relations on128/128; no
visibility-gap sample was dropped. Minimum TRAIN-to-DEV camera XY distance is
133.4685m. Capture/job/cache validation completes in123.980s on the worker;
native editor capture accounts for86.844s and CUDA cache construction1.788s.
Main-machine thin evidence and the complete training-resolution DEV cache are
under `artifacts.local/work/city-dev-baseline-20260908/evidence/`; worker raw
capture remains at `work/city-dev-baseline-20260908/capture-v1`.

Canary visuals checked all four mounted families; the root also inspected the
captured crossbar quartet endpoints. This establishes bounded synthetic placement
and label coverage. The new region still shares map/assets/rendering with TRAIN,
and its obstacle-family distribution also changes: any outcome cannot isolate
region shift from shape shift or establish independent-world generalization.
Thirty-two matched groups are not128 independent scenes. Each head has64
negatives, so the empirical10% constraint permits at most6 false positives;
it is not a population-FPR guarantee.

## Result: functioning baseline, weak DEV-selected B; no promotion

The predeclared whole-checkpoint rule selects **B**. Its minimum-head recall is
25%, versus18.75% for A and23.4375% for C. This is a selected Development result,
not evidence that freezing RepViT fixes cross-region transfer.

Each DEV head has64 positives and64 negatives. Metrics below use that arm's
own DEV-selected thresholds; no head/checkpoint mixing occurred.

| Arm | BODY TP / FP | BODY recall / FPR | HEAD TP / FP | HEAD recall / FPR | All-correct quartets |
| --- | --- | --- | --- | --- | --- |
| A full FT | 13 / 2 | 20.31% / 3.13% | 12 / 6 | 18.75% / 9.38% | 0/32 |
| B frozen backbone | 26 / 6 | 40.63% / 9.38% | 16 / 5 | 25.00% / 7.81% | 2/32 |
| C freeze then unfreeze | 21 / 5 | 32.81% / 7.81% | 15 / 5 | 23.44% / 7.81% | 0/32 |

Thresholds BODY/HEAD: A `0.991744339466095 / 0.5547345876693726`;
B `0.4268244802951813 / 0.4970442056655884`;
C `0.576921820640564 / 0.4976978600025177`.
The B-versus-C HEAD difference is only one positive frame; one seed and32
groups cannot establish a stable strategy ordering.

At fixed0.5 the corresponding DEV BODY recall/FPR is A100%/98.44%,
B23.44%/3.13%, C46.88%/25%; HEAD is A48.44%/39.06%, B15.63%/1.56%,
C20.31%/4.69%. All three have0/32 correct quartets at that fixed point.
The selected thresholds make the operating-point tradeoff explicit; they do
not create missing positive/negative separation.

Support remains at0.5, independent of the near thresholds:

| Arm | DEV BODY / HEAD IoU | BODY / HEAD peak hits, each /64 | Negative BODY / HEAD known-pixel activation |
| --- | --- | --- | --- |
| A | 0.13714 / 0.04546 | 11 / 4 | 11.70% / 2.35% |
| B | 0.06448 / 0.00000 | 7 / 2 | 11.03% / 0.0034% |
| C | 0.07658 / 0.00000 | 7 / 2 | 13.72% / 0.0034% |

B's lower HEAD background activation accompanies zero positive support IoU;
it is not restored localization. Near labels are known for128/128 per head;
support retains19.40%/19.38% UNKNOWN pixels, excluded only from known-pixel
scoring, while peaks on UNKNOWN remain misses.

After `selection.json` was saved and hashed, the unchanged thresholds were
applied to the consumed plaza. Denominators are BODY135positive/615negative,
HEAD15positive/735negative. These diagnostics cannot revise the winner.

| Arm | BODY TP / FP | BODY recall / FPR | HEAD TP / FP | HEAD recall / FPR | All-correct triplets |
| --- | --- | --- | --- | --- | --- |
| A | 8 / 3 | 5.93% / 0.49% | 4 / 21 | 26.67% / 2.86% | 110/250 |
| B | 29 / 18 | 21.48% / 2.93% | 0 / 18 | 0.00% / 2.45% | 120/250 |
| C | 21 / 12 | 15.56% / 1.95% | 0 / 27 | 0.00% / 3.67% | 116/250 |

The B candidate still misses every plaza HEAD positive. Its plaza BODY/HEAD
support IoU is0.05068/0, versus A0.14189/0.04946. Improved group counts with
many all-negative triplets do not outweigh missing risk or broken localization.
No Willow regression was run for B/C in this bounded comparison; old A's Willow
results cannot be inherited by these weights.

Disposition: retain G13-D and this source-separated selection workflow; keep
B only as a weak diagnostic challenger, retaining A and C as matched controls.
Do not promote any new weights. Freezing changes both adaptation capacity and
optimization speed; fixed200 steps is a budget-matched comparison, not a
convergence-matched causal isolation of backbone drift. Final logged batch loss
is0.80955 for B and0.54515 for C. No extra steps, losses or threshold rescues
were added. A stronger next training corpus needs separate TRAIN regions and
HEAD-family coverage while this entire DEV site stays held out; do not move
its128 samples into training. Fresh TEST remains uncaptured/unaccessed and must
follow a fixed candidate if used for confirmation.

## Execution and verification

Only B and C were newly fitted, each200 steps. Worker RTX3060 Laptop CUDA,
Torch2.9.1+cu128: B10.060s, C19.382s, full fitting/evaluation36.847s.
B backbone tensor identity is unchanged; all BN buffers remain unchanged for
both; complete B/C state after100 steps matches SHA
`19efbd8e0a6bc56626ea8f1df792b61fdd25c56feb569813fcf08000fff4cf7b`.
Original TRAIN cache, initialization, exact200x32 schedule and common source
hashes are checked. No model accesses DEV during fitting. Immutable runtime
contains the pre-outcome protocol version; this report appends results afterward.

The selector's eight focused tests pass, including UNKNOWN coverage, exact10%
boundary, deterministic checkpoint ties, and a float64 all-negative sentinel.
All four new Python sources parse; worker import preflight and the full bounded
integration run pass. Model result and selection hashes are respectively
`c2830bc4459f2833abccf8265954594e7cff76fe491012f3cb992532766c8087`
and `bc3d3e209ac968de77eba81aa60686c6b2ef866b97bf8ea27c8b5fcc9d5a93b6`.

Main evidence: `artifacts.local/work/city-dev-baseline-20260908/model-run-v1/`
contains18 hash-verified result/log/prediction files. Raw capture and the B/C
checkpoints remain on the worker under the same work prefix. Selected B SHA:
`42cc4d1bbba632b81b8b391a644170361c2e701f879601a9646cd499974f892e`.
Task-owned capture/training processes are released; durable evidence and caches
are retained. No source-map or concurrent UE changes are part of this delivery.
