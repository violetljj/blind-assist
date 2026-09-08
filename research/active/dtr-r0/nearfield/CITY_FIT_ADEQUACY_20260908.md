# Fixed-budget frozen-B fit adequacy

Pre-outcome EXPLORE protocol, authorized continuation after relational TRAIN384.
The prior frozen-B200 fit on old750+new384 had weak TRAIN separation: new-data
BODY/HEAD AUC0.644/0.616, HEAD IoU0, joint0/96. The question is whether the same
recipe learns with more updates, before adding representation/loss mechanisms.

One fresh initialization from original G13 seed17; same1134 TRAIN images,
same frozen RepViT/all BN buffers, head AdamW lr1e-5/weight_decay1e-4, near BCE
plus0.25 unchanged globally balanced support BCE. Extend the fixed budget from
200 to **2000 steps**, batch32, identical uniform-replacement seed17 procedure.
The first200 index arrays and all model tensors at200 must exactly match the
prior relational run. This is a new bounded budget experiment, not reopening
or extending the old run's terminal. No LR changes, extra augmentation, new
data, losses, architecture or additional arms.

Save step200/500/1000/2000 checkpoints. After all2000 training steps finish,
score old/new TRAIN separately at each saved point: fixed0.5 confusion,
support IoU/peak/negative activation, joint and tie-aware AUC/PR/descriptive
recall envelope. Reuse verified step200/final cached predictions where possible.
Intermediate checkpoints never see DEV and cannot be selected for deployment.
Only final2000 receives the unchanged DEV per-head threshold rule (empirical
FPR<=10%, maximize recall, lower FPR, higher threshold). Compare against prior
relational200, and only afterward apply unchanged thresholds to consumed plaza.
No fresh TEST or Willow regression. These remain consumed Development scores.

If TRAIN clearly improves but DEV does not, distinguish fitting capacity from
region transfer. If TRAIN remains weak, retain the failure and investigate
optimization/representation adequacy next; do not infer that data is useless.
If both improve, retain the recipe as a candidate pending further validation.
Report mixed classification/localization outcomes explicitly. No automatic
promotion or selection from intermediate TRAIN/DEV results.

Stop after this one2000-step fit and the listed diagnostics. No budget rescue.
Preserve model/optimizer resume state, schedule, hashes and execution receipts;
release task-owned worker processes. Mechanical evaluation failures can recover
from saved checkpoints without repeating completed training.

## Result: longer fitting helps some scores, but does not solve HEAD

The single2000-step run passes exact step200 tensor and schedule-prefix parity.
Backbone and BN buffers remain unchanged. No intermediate checkpoint saw DEV.

| Final checkpoint | DEV BODY recall / FPR | DEV HEAD recall / FPR | Joint | BODY / HEAD support IoU |
| --- | --- | --- | --- | --- |
| Prior relational200 | 37.50% / 7.8125% | 28.125% / 9.375% | 2/32 | 0.06366 / 0.00260 |
| Fixed2000 | 43.75% / 4.6875% | 31.25% / 6.25% | 1/32 | 0.05595 / 0.04870 |

Both DEV heads retain64positive/64negative. Final TP/FP are28/3 BODY and20/4
HEAD. Selected thresholds are0.6500800251960754/0.3487178087234497. The small
recall improvement accompanies fewer false positives, but does not establish
strong task performance. At0.5, final DEV recall is75%/7.8125%, FPR28.125%/0;
threshold calibration remains consequential.

The support increase is not a clean localization recovery. On negative DEV
known pixels, BODY activation rises8.42% to44.82%, and HEAD0.0403% to26.88%.
Positive peak hits only rise7/64 to8/64 BODY and2/64 to5/64 HEAD. Joint worsens
from2/32 to1/32; do not summarize this as an across-the-board gain.

TRAIN trajectories use fixed0.5 for confusion/IoU, plus threshold-independent
AUC. The descriptive recall envelopes are label-derived curves, not exported
operating thresholds or checkpoint-selection rules.

| Step | New TRAIN BODY / HEAD AUC | New TRAIN BODY / HEAD recall envelope at FPR<=10% | New TRAIN BODY / HEAD IoU |
| --- | --- | --- | --- |
| 200 | 0.64390 / 0.61575 | 30.89% / 17.71% | 0.08986 / 0 |
| 500 | 0.65448 / 0.63007 | 27.23% / 19.79% | 0.05103 / 0.04071 |
| 1000 | 0.67254 / 0.61344 | 29.32% / 19.79% | 0.05128 / 0.05351 |
| 2000 | 0.71280 / 0.61081 | 38.74% / 17.19% | 0.05842 / 0.05731 |

The new384 TRAIN HEAD ranking is essentially stagnant despite10x updates.
At2000, its fixed0.5 BODY recall is91/191 (47.64%), FP30/193 (15.54%);
HEAD recall8/192 (4.17%), FP2/192 (1.04%). Joint stays0/96 at every snapshot.
This is not simply a high-fitted TRAIN model failing only on DEV.

Old750 does show improved ranking: BODY AUC0.61407 to0.86219, HEAD0.20853 to
0.78231. Final descriptive recall envelopes at FPR<=10% are68.89%/26.67%.
Yet fixed0.5 recall is only4/135 BODY and0/15 HEAD, showing why fixed-point
accuracy alone would misdescribe the learned ordering. Final old support IoU
is0.04380/0.00926. The final logged batch loss is0.44872 versus0.97019 at200;
these are different batches, not a convergence certificate.

After final DEV thresholds were frozen, consumed plaza gives BODY18/135 TP,
7/615 FP (13.33% recall,1.14% FPR); HEAD0/15 TP,3/735 FP (0%,0.41%). Joint
127/250 still cannot compensate for zero HEAD recall. No fresh TEST or Willow
regression was run, and no checkpoint is promoted.

Disposition: retain the trajectory and candidate only for diagnosis. More
updates help old-domain ranking and some BODY scores, but merely extending
this recipe does not fix relational HEAD discrimination. This does not prove
that the architecture is defective, the data is useless or convergence has
been reached. The next bounded check should use a small balanced TRAIN subset
and inspect separate near/support losses and gradients to establish whether
the model can fit those relations, before adding a generalization mechanism.
No further fit or learning-rate search was performed in this run.

## Execution and persistence

Worker RTX3060 Laptop, CUDA/Torch2.9.1+cu128: fit91.853s, complete model
evaluation102.314s. One2000-step fit only. Source syntax/import preflight,
frozen dataset/common-module hashes, step200 schedule/model equality, final
backbone/BN checks and full bounded integration pass. The unchanged selector
retains its prior eight focused passing tests. The executed runner snapshot SHA
(raw runtime bytes, before any Git line-ending normalization) is:
`d042b7feb0983799f9fb1cc6b2995afd91c1ced85afb061c120c084c71628bd3`.

Main evidence: `artifacts.local/work/city-fit-adequacy-20260908/model-run-v1/`,
24 transferred result/prediction/log/source files (19,866,927bytes), each SHA
verified; transfer/release/compact-summary receipts are additional files.
Result SHA`5c85eba7958ad6489153e7fe4af87dd3ccb0f483a204c03ac53b2e3c3344e18a`;
selection SHA`c94327c7293602743766e9bcb496b3d934bed93c96651f6ca74e11c2113e9a84`.
Worker work prefix`city-fit-adequacy-20260908/model-run-v1` retains all four
checkpoints and`optimizer-resume-state.pt` (optimizer, CPU/CUDA RNG and step),
plus the deterministic schedule for a separately authorized future continuation.
Final checkpoint SHA`5a0ffce18cc18c38ee02f4c678b3cd1a2cc013f843db1b5443b4fccae4b40374`.
Task-owned processes and scheduled jobs are released; durable evidence remains.
