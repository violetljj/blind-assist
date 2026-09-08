# Full-parameter City fitting at matched 2000 steps

Pre-outcome EXPLORE protocol. The tiny32 TRAIN diagnosis showed that both
frozen and full updates fit all near bits, while full updates substantially
improved support IoU. It did not establish generalization or rule out every
implementation defect. This experiment asks whether that fitting advantage
transfers to the complete training distribution and consumed regional DEV.

One fresh run from original G13 seed17, not the tiny-fit checkpoint. Use the
same old750 plus relational384 TRAIN, identical full 2000x32 sampling array,
AdamW learning rate1e-5/weight decay1e-4, near BCE plus0.25 unchanged globally
balanced support BCE as frozen-B2000. The only training change is enabling all
parameter updates, including backbone BN affine parameters; BN running buffers
remain frozen. No new data, architecture, augmentation, loss or schedule.

Save200/500/1000/2000 checkpoints and optimizer/RNG resume state. After fitting
all2000 steps, score old/new TRAIN separately at these snapshots. Report near
and support losses, fixed0.5 confusion/IoU/peak/negative support activation,
joint correctness and descriptive ranking. Intermediate checkpoints never
receive DEV and cannot be selected. Batch-loss curves alone do not certify
convergence; compare fixed-data diagnostics when assessing fitting progress.

Only final2000 gets the unchanged per-head DEV threshold rule: empirical
FPR<=10%, maximize recall, then lower FPR, then higher threshold. Apply those
frozen thresholds to already-consumed plaza only after DEV selection. Report
BODY/HEAD TP and FP with denominators, recall/FPR, joint correctness and support
metrics beside frozen-B2000; show fixed0.5 results separately. Preserve UNKNOWN
and source boundaries. No fresh TEST, Willow regression or automatic promotion.

If localization and regional decisions both improve, retain full updates as a
candidate baseline. If TRAIN localization improves but DEV/plaza do not, retain
the fitting gain and investigate region transfer. If fitting remains weak,
report this without claiming convergence or intrinsic architecture failure.
An improved DEV IoU alone does not prove localization is solved. Plaza remains
diagnostic and cannot select thresholds, checkpoints or further run budgets.

Stop after this one2000-step fit and listed evaluation. No automatic extension
or rescue fit. Mechanical scoring failures may recover saved checkpoints
without repeating completed training. Preserve execution hashes, receipts and
checkpoints; release task-owned worker processes and transfer temporaries.

## Result: complete TRAIN fit and stronger DEV, persistent plaza HEAD failure

The single run completes2000 updates with exact full sampling-array equality
to frozen-B2000. All parameters are trainable, backbone tensors change and BN
running buffers remain unchanged. No intermediate DEV evaluation or extra fit.

Each checkpoint below uses its own DEV-selected thresholds under the identical
rule. Both DEV heads have64positive/64negative frames. Support uses fixed0.5,
positive-frame mean IoU with UNKNOWN excluded from the pixel union.

| Metric | Frozen B2000 | Full F2000 |
| --- | --- | --- |
| DEV BODY recall / FPR | 43.75% / 4.6875% | 81.25% / 9.375% |
| DEV HEAD recall / FPR | 31.25% / 6.25% | 71.875% / 9.375% |
| DEV BODY / HEAD TP | 28/64 / 20/64 | 52/64 / 46/64 |
| DEV BODY / HEAD FP | 3/64 / 4/64 | 6/64 / 6/64 |
| DEV BODY support IoU | 0.05595 | 0.20745 |
| DEV HEAD support IoU | 0.04870 | 0.20324 |
| DEV joint groups | 1/32 | 6/32 |
| New TRAIN BODY / HEAD IoU | 0.05842 / 0.05731 | 0.25527 / 0.23486 |
| Old TRAIN BODY / HEAD IoU | 0.04380 / 0.00926 | 0.24472 / 0.19911 |
| Plaza BODY TP / FP | 18/135 / 7/615 | 51/135 / 31/615 |
| Plaza HEAD TP / FP | 0/15 / 3/735 | 0/15 / 171/735 |
| Plaza BODY recall / FPR | 13.33% / 1.14% | 37.78% / 5.04% |
| Plaza HEAD recall / FPR | 0% / 0.41% | 0% / 23.27% |
| Plaza BODY / HEAD IoU | 0.04759 / 0.01416 | 0.14257 / 0.04635 |
| Plaza joint groups | 127/250 | 81/250 |

F thresholds are0.6319661140441895 BODY and0.9686279892921448 HEAD.
At fixed0.5, DEV instead gives BODY61TP/11FP and HEAD60TP/26FP;
plaza gives BODY61TP/74FP and HEAD7TP/659FP, joint2/250. Therefore calibration
does matter, but the frozen DEV HEAD threshold still transfers poorly. The
historical original750-only full200 result (BODY561FP/HEAD348FP at0.5) is not
the matched comparator: both data and update budget changed since that run.

DEV negative known-pixel activation falls44.82% to1.30% BODY and26.88% to5.37%
HEAD. Peak hits improve8/64 to34/64 BODY and5/64 to18/64 HEAD; F HEAD includes
7 UNKNOWN-peak misses, retained as misses. This is a material localization
gain, not proof localization is solved. On plaza HEAD, the peak misses all15
positives (3 UNKNOWN), despite its improved mean IoU; negative support covers
only1.79% of known background pixels while the near head still misfires.
Mask area and alert failure cannot be treated as interchangeable explanations.

## Fixed TRAIN fitting trajectory

Both old750 and new384 have perfect fixed0.5 near decisions and AUC1.0 for
both heads at1000 and2000. At final2000 there are2268/2268 correct near bits;
new384 contains191 BODY and192 HEAD positives, preserving the admitted native
label discrepancy rather than forcing intended quartet labels.

Losses below are evaluated over each entire fixed partition using logits,
with the unchanged global positive/negative support balance. They are not
the fluctuating training-batch losses or an average of per-head support losses.

| Step | Old near BCE / support BCE | New near BCE / support BCE | New BODY / HEAD IoU |
| --- | --- | --- | --- |
| 200 | 0.04492 / 0.23636 | 0.40514 / 0.25317 | 0.14622 / 0.10371 |
| 500 | 0.00635 / 0.10965 | 0.05739 / 0.22944 | 0.13166 / 0.11289 |
| 1000 | 0.00111 / 0.05460 | 0.00590 / 0.13099 | 0.16935 / 0.16737 |
| 2000 | 0.00035 / 0.02897 | 0.00144 / 0.06819 | 0.25527 / 0.23486 |

Support loss is still falling from1000 to2000; this is not a convergence
certificate. Nevertheless, the complete TRAIN classification fit versus
plaza HEAD failure now supplies direct evidence of a transfer gap. Merely
adding steps is not established as its remedy. No extension was run.

Disposition: retain full-parameter updates as the stronger Development
fitting/DEV candidate with G13-D unchanged. Do not promote these weights or
claim multi-region generalization, Willow retention, or natural-world benefit.
DEV and plaza are consumed Development, with only15 HEAD positives on plaza.
The next useful check is cached HEAD error attribution by region/family and
support-versus-near score separation before selecting one region-transfer
intervention. Scene shortcut remains a hypothesis, not an identified cause;
these aggregate results alone cannot choose a new loss or prove data adequacy.

## Verification and execution

AST/import/help preflight, original input/common-module identity, full schedule
equality, finite losses/gradients, changed backbone and unchanged BN checks
pass. Root independently recomputed DEV/plaza confusion, positive known-pixel
IoU and joint groups from cached predictions, and checked schedule/result SHA.
The unchanged threshold selector is reused; no threshold is fitted to plaza.

Worker RTX3060 Laptop, Torch2.9.1+cu128: fitting290.04s, script306.83s,
job310.03s. Exactly one new2000-step fit. Main evidence:
`artifacts.local/work/city-full-fit-20260908/model-run-v1/`,26 transferred
files/25,079,666bytes SHA-verified, plus later release/independent-check receipts.
Runtime source SHA (raw bytes, before Git newline normalization):
`1b0154a9f766eb1a2565009579e4c5428b9edaf20ce0949b268485caff95359d`.
Pre-outcome protocol SHA:
`d68846f4f6c395b233500b925dd948e82cb261fee5dd2e5073c8e9e7a4744be0`.
Result SHA:
`4c9c4df7ce9d6d2252f3443d12954e5d4e460122656e1a09a388a0fca63644d5`.
Final checkpoint SHA:
`d911a3a717efb5a93fadb3268ba962db94455565805effc7077f75215670cf46`.
The worker retains four checkpoints and final optimizer/RNG state under the
same work prefix. Task-owned processes and scheduled tasks are absent in the
release receipt; temporary transfer archives are removed, durable inputs retained.
