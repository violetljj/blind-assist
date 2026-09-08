# Matched 64-frame regional coverage pilot

Pre-outcome EXPLORE continuation after HEAD-X0. Hypothesis: limited coverage
of the same body-relative relation under different scene appearances contributes
to cross-region failure. HEAD-X0 showed matching old/plaza pooled positive
masks with score-order reversal, but did not isolate background causality.

Acquire64 additional TRAIN frames:16 fixed background/camera units, each with
CLEAR/BODY_ONLY/HEAD_ONLY/BOTH intended states. Existing attached fixture
families are retained; negative relations preserve structures via height or
position changes rather than universally deleting them. Use native visible
support for labels and retain UNKNOWN/disagreements; never force balance.
Distribute units across physically distinct background/pose locations. Record
same-unit camera/light equality and actual source/instance separation.

Acquire a separate64-frame EVAL set,16 complete quartets in a distinct region.
No same-unit or nearby duplicate views may cross TRAIN/EVAL. Freeze site and
pose identities, source roles and native admission before model evaluation;
check floor, visible supported assembly and geometry with source canaries first.
These are new same-world Development regions, not protected blind/fresh-world
confirmation. Existing consumed plaza remains excluded from added TRAIN and
from primary evaluation. Do not score captured EVAL to choose sites or repairs.

A is the existing full F2000 checkpoint and its stored DEV-selected thresholds;
do not refit A. B starts from original G13 seed17, not A: old750+relational384
+new64=1198 TRAIN, one2000-step batch32 run. Same full-parameter AdamW1e-5,
weight decay1e-4, near BCE plus0.25 unchanged global support BCE, frozen BN
running buffers, uniform replacement seed17 and original preprocessing. No
augmentation, new loss, changed head or sampling reweighting. Record actual
new-frame/group draw counts and per-frame minimum/maximum exposure.

Only B final2000 receives the unchanged original DEV128 per-head selector:
empirical FPR<=10%, maximum recall, then lower FPR, then higher threshold.
A retains its existing DEV threshold. Once selected, evaluate both on the
new64 EVAL; no EVAL-derived threshold or checkpoint choice. Report HEAD/BODY
AUC/AP, TP/FP denominators, recall/FPR, support IoU/peak/negative activation,
group correctness and old DEV behavior. Report same-group positive-versus-
negative score ordering, with strict wins, ties and losses separately, for
each head. It is a diagnostic metric, not an additional training objective.
Inspect B fitting on the added64 and original TRAIN partitions so lack of
exposure/fitting is not misclassified as generalization failure.

A useful data intervention should improve new-region HEAD ordering and recall
without concealing increased false positives or localization/BODY regressions.
If FP falls while all positives remain missed, do not call the task repaired.
If new TRAIN fits but EVAL does not, record failure to transfer this coverage;
one64-frame pilot cannot prove architecture failure. If added TRAIN is weak,
inspect exposure/labels rather than making a transfer claim. No automatic model
promotion, count scaling sweep, extra steps, LR rescue or second new fit.

Stop after the one fit and listed comparison. Mechanical capture/evaluation
failures retain original receipts and may recover missing outputs without
repeating completed training. Capture engineering acceptance and model effect
remain separate. Preserve maps, caches, checkpoints and provenance; release only
task-owned UE/worker processes and temporary transfers. Exact site/admission and
execution identities will be appended before outcomes are interpreted.

## Source admission before fitting

The initial4x4 grids crossed the real tree rows: canary-v1 stopped on an
undeployed support-board material, and v2 exposed foliage/bench interference
with4/8 intended/native disagreements. Both failed sources and receipts remain;
no model was trained or scored to choose the repaired cameras. A read-only
native instance probe located tree rows at y24/36/48/60m and benches at y18/66m.
The accepted source uses tree-row corridors, without editing the world:
TRAIN x32..46m at2m spacing, y29.5/30.5m, yaw0; EVAL x67..77.5m at1.5m spacing,
y53.5/54.5m, yaw180. Each role has16 camera units in an8x2 arrangement, not16
independent scenes. Nearby within-role views remain correlated. Cross-role
camera separation is at least31m; exact metadata distances are recorded below.

Root viewed all8 relation previews and2 empty-ground views from canary-v3:
no camera-through-foliage or bench interference was visible, attached fixtures
had visible grounded backing, and all8 relation labels matched native truth.
Two empty-ground depth checks reported median errors8.29/8.54mm, with4800
pixels each and100% within5cm; both empty views were natively BODY/HEAD clear.
These checks admit source acquisition, not model performance. Full128 frames
will be acquired in one UE session and split into role-specific64-frame caches.
The source rigidly transforms locally verified +X query geometry into world
coordinates; native depth independently supplies the world-visible labels.

## Completed acquisition

The single128-frame capture completed in90.906s (capture/cache job137.35s,
excluding earlier source repair and transfer). Each role contains64 frames and
16 quartets; each head has32 positives and32 negatives. All128 native labels
match intended relations, with no removed/replaced samples. Root viewed both
16-frame CLEAR contact sheets after capture. These preserve visible raised or
out-of-path structures rather than emptying every negative image.

Actual minimum camera separations: TRAIN/EVAL31.144823m; EVAL versus prior
TRAIN/DEV68.395180m. Distances to consumed plaza cameras are only13.729530m
(EVAL) and12.349089m (newTRAIN), so this is a new source partition of the same
plaza, with correlated world assets and within-role views. It is not an untouched
city/world test. No consumed plaza frames are included in the new caches.
The original source map/project hashes are unchanged. Two role caches retain
native UNKNOWN support pixels (roughly25% of pooled pixels); EVAL retains
source indices64..127 and its truth stays under evaluator/, not supervision/.
165 transferred files passed SHA verification; task-owned UE processes were
released before fitting.

Captured mixed source SHA256:
`d8689b45ca9d6a00599c7a5c13b75a3a01d9fc76f4d5f7759967849985645e43`.
Source/capture admission, caches, native evidence and transfer validation are
under `artifacts.local/work/city-coverage64-20260908/`. The two cache manifests
bind the same mixed source by hash and select disjoint roles/groups/cameras.

## Outcome: limited transfer, not a HEAD repair

Exactly one new fit completed. A reused its original checkpoint and thresholds
BODY0.6319661140/HEAD0.9686279893. B's unchanged original DEV selector chose
BODY0.6730185151/HEAD0.9902606606. Both DEV operating points have6/64 false
positives per head. New EVAL scores did not choose any threshold/checkpoint.

| New EVAL64 (32 positive/32 negative per head) | A: existing F2000 | B: +64 TRAIN |
| --- | ---: | ---: |
| BODY TP / FP | 15/32 / 3/32 | 16/32 / 1/32 |
| BODY recall / FPR | 46.875% / 9.375% | 50.000% / 3.125% |
| BODY AUC / AP | 0.782227 / 0.814642 | 0.753906 / 0.810054 |
| BODY positive support IoU | 0.168243 | 0.195746 |
| BODY peak hits | 13/32 | 13/32 |
| BODY within-group strict ordering | 63/64 | 62/64 |
| HEAD TP / FP | 8/32 / 3/32 | 12/32 / 1/32 |
| HEAD recall / FPR | 25.000% / 9.375% | 37.500% / 3.125% |
| HEAD AUC / AP | 0.655273 / 0.700175 | 0.708984 / 0.748386 |
| HEAD positive support IoU | 0.138472 | 0.185173 |
| HEAD peak hits | 8/32 | 12/32 |
| HEAD within-group strict ordering | 43/64 | 51/64 |
| Joint all-four/all-head correct groups | 0/16 | 3/16 |

There are no within-group ranking ties. Pair counts are4 label-discordant
positive/negative pairs per quartet/head, correlated within16 source groups,
not64 independent experiments. Each TP or FP is3.125 percentage points here.
HEAD negative known-pixel support activation nevertheless rises from1.869% to
4.114%; BODY rises3.202% to4.265%. Lower near FP therefore does not establish
suppression of all irrelevant spatial responses. At the unchanged0.5 diagnostic
cutoff, HEAD TP stays26/32 while FP falls18/32 to14/32; BODY TP19 to18 and FP7
to3. The table above uses the required DEV-selected operating points instead.

| Original DEV128 | A | B |
| --- | ---: | ---: |
| BODY TP / FP (64 each) | 52 / 6 | 58 / 6 |
| HEAD TP / FP (64 each) | 46 / 6 | 54 / 6 |
| BODY support IoU / peak hits | 0.207447 / 34 | 0.214311 / 31 |
| HEAD support IoU / peak hits | 0.203243 / 18 | 0.185211 / 22 |
| Joint correct groups | 6/32 | 12/32 |

DEV near recall improves, but HEAD IoU regresses and BODY peak hits fall; this
is not a uniform regression-free improvement. DEV negative known-pixel support
activation rises BODY1.301% to1.972%, HEAD5.375% to6.113%.

B fits all2396 TRAIN near bits at0.5. At the stricter DEV operating point it
misses one old750 HEAD positive; relational384 and added64 remain fully correct.
New64 has128/128 near bits and16/16 groups correct, BODY/HEAD support IoU
0.194993/0.205511 and peak hits17/32,15/32. Its exact full-partition near BCE
is0.0025182 and support BCE0.1154178, versus old750: 0.0005352/0.0302442 and
relational384: 0.0012115/0.0706133. Classification fitting does not imply precise
support fitting. Every added frame receives34..66 draws, total3364 across the
fixed64000 draws; the added data was actually exposed and learned.

Retain this data intervention and checkpoint as a Development challenger.
HEAD ordering, calibrated detection, positive support IoU and peak hits all
improve in the new region, so the result is more than simply suppressing every
prediction. However20/32 HEAD positives remain missed, BODY ranking slightly
falls, and spatial false activation increases. This small same-world pilot
supports limited useful coverage transfer, not a solved relation model, proved
background causality, or proof for/against architecture adequacy. Retain G13-D
and the existing baseline authority; do not automatically promote B or expand
the completed2000-step budget. Any next coverage/representation experiment must
be a new comparison with explicit old-DEV localization and negative activation.

## Execution and verification

Worker RTX3060 Laptop, Torch2.9.1+cu128: fit290.709s, script297.817s,
job301.06s. BN running buffers remain unchanged. One original-seed17 fit,
no rescue fit, no augmented input, no changed loss, no EVAL selection.
Uniform replacement is regenerated over1198 indices; it is the same seed and
sampling rule, not byte-identical1134-frame draws. A single seed/budget cannot
separate data effects from every stochastic optimization effect.

23 returned files/26,773,466bytes pass SHA verification, including final B
checkpoint and all saved predictions. Root independently recomputed EVAL
confusion/AUC/IoU/peak/within-group ordering/joint counts and sampling exposure
from saved predictions and labels, and viewed fixed-unit HEAD overlays for all
four families. Evidence is in `model-run-v1/root-independent-check.json` and
`head-contact-1.png`, `head-contact-2.png`. No EVAL frame is chosen by outcome
for these contacts. Earlier source canaries and their failures remain recorded.

B checkpoint SHA256:
`d60c871fd0a292bfd5b30c625e5a4189632a7c42576fd6d951c71e40e6d8b7bc`.
Result SHA256:
`f9617deeb58cdd83768eaf2c8e56ba029f5e4b4080b04a3c342cc7dd3985140c`.
TRAIN/EVAL manifest SHA256:
`343a92394ea57c44e41a1266555046dd7326ab3f53eac41a64def53479f230a0` /
`08c27dd45d4ef1b644dae3fbfacff10bc54fa61d0d57000c556d34055de783f6`.
Executed model runner SHA256:
`20b19c00e7f5dfe95687bcb6c5650d29aa7cd98b6db0c0c42494d750c46cfe0a`.
Task-owned UE/model processes and scheduled tasks are released; verified
transfer archives are removed. Durable native capture/cache/weights and worker
optimizer/RNG state remain under the task work prefix for reproducibility.
The generator's station-count comment was corrected after acquisition; its AST
is unchanged and the executed source snapshot is retained separately.
