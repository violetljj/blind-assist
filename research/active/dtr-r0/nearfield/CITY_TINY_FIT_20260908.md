# Balanced TRAIN-only fitting and gradient diagnosis

Pre-outcome EXPLORE diagnostic. Prior frozen-B2000 improves some old-domain
ranking but leaves new relational TRAIN HEAD AUC0.611 and joint0/96. Test
whether the unchanged training/label/model pipeline can fit a tiny balanced
set, and whether backbone freezing materially restricts that ability.

Use only the admitted new TRAIN384 cache. From site`TRAIN-north-x-76`, take the
first two complete quartets of each of four families:8 groups/32 images,
16 positives and16 negatives per BODY/HEAD query. Selection is deterministic
from TRAIN metadata before inference. Require native balance and preserve
UNKNOWN support. No DEV, plaza, Willow, new source or test access in this run.

Two fixed arms, each original G13 seed17 initialization and2000 full-batch
updates over these same32 images. B freezes RepViT; F permits all parameters
to update. Both keep all BN running buffers frozen. AdamW lr1e-5,
weight_decay1e-4; near BCE plus0.25 unchanged globally balanced UNKNOWN-aware
support BCE. Same architecture, preprocessing and labels; no augmentation,
loss replacement, schedule search or early stopping. The full batch can stay
on CUDA to avoid repeated decoding/copying. Record exact subset and source
hashes before fitting. No additional fit if either arm fails to fit.

Log total loss and separate BODY/HEAD near BCE and support BCE at fixed steps
1/200/500/1000/2000. Per-head support losses are diagnostic decompositions;
their arithmetic mean need not equal the actual globally balanced support
loss, and they must not replace it. At1/200/1000/2000 measure separate near-loss
and weighted support-loss gradients for backbone, deep projection, detail,
support and near parameter groups. Use retained-graph autograd only for
measurement before the ordinary backward/update; report frozen parameters and
unused gradients separately rather than interpreting them as a wiring failure.
Loss/gradient records are pre-update at the named step; final metrics are after
all2000 updates. Check finite losses/gradients, frozen B backbone and BN buffers.

Final TRAIN-only metrics: fixed0.5 BODY/HEAD confusion, positive known-pixel
support IoU and peak, negative support activation, UNKNOWN,8-group joint, and
descriptive tie-aware score curves. No threshold selection. A strong tiny-fit
result means all64 known near decisions correct (8/8 groups) and mean positive
support IoU>=0.5 for each head. This diagnostic threshold is not a product or
generalization criterion. Report partial success even if that mark is missed.

If F fits while B does not, the frozen recipe is a material limitation for this
subset; this does not prove F will generalize. If both fail, inspect training
adequacy/target resolution/optimization before adding domain mechanisms. If
both fit, the broader mixture/optimization challenge remains. Gradient signs
and norms are descriptive, not a unique causal attribution. No new method,
automatic promotion or fresh-test claim follows from memorizing32 examples.

Stop after these two2000-step fits and diagnostics. Keep checkpoints,
optimizer/RNG state, input identity and receipts; release owned worker jobs.
Mechanical evaluation failures may recover from saved checkpoints without
rerunning completed fits. Report/commit only task-owned files; shared knowledge
WIP remains isolated.

## Result: classification fits in both; full updates improve localization

The selected indices are0–31: eight complete first-site groups, two of each
family. Actual native labels are16positive/16negative per head and match pooled
positive-support existence. No held-out data was read or evaluated.

| Arm after2000 updates | Correct near bits | Correct groups | BODY IoU | HEAD IoU | near BCE | global support BCE |
| --- | --- | --- | --- | --- | --- | --- |
| B frozen backbone | 64/64 | 8/8 | 0.10973 | 0.08902 | 0.168857 | 0.286822 |
| F full parameter updates | 64/64 | 8/8 | 0.55543 | 0.48797 | 0.000164 | 0.018637 |

Both heads in both arms have16TP/16TN/0FP/0FN at the unchanged0.5 near threshold.
Both arms **miss the predeclared strong-fit mark**: B has weak localization;
F's HEAD mean IoU remains below0.5. Do not round it up, change the threshold or
add steps to declare a pass. F nevertheless gives a large, useful localization
improvement under the matched budget. This is tiny TRAIN memorization evidence,
not evidence of held-out regional transfer.

BODY/HEAD positive peak hits improve from5/16 and5/16 in B to13/16 and10/16 in
F. F still has2/4 peaks respectively on UNKNOWN, counted as misses; they are
not moved onto known pixels. Negative known-pixel activation falls from
16.21%/12.42% to0.4845%/0.4252%. Even F retains at least one false support pixel
on16/16 BODY-negative and13/16 HEAD-negative images; low pixel rate is not
equivalent to perfectly empty negative maps. UNKNOWN fractions remain
10.742%/10.607% for BODY/HEAD.

To understand F's small support loss but sub0.5 HEAD IoU, a post-outcome
zero-training decomposition reused its cached masks at the unchanged0.5:
over the16 positive frames per head, BODY has212TP/154FP/0FN pixels; HEAD has
116TP/118FP/0FN. All known positive target pixels are covered, but extra known
background pixels reduce precision. BODY/HEAD micro precision is57.92%/49.57%,
with100% known-positive pixel recall. The median target area is only11/6 pooled
pixels. This explains why a few extra activated cells per image materially
affect IoU; it does not justify ignoring those errors. Results/inputs are saved
in `artifacts.local/work/city-tiny-fit-20260908/support-error-diagnostic.json`.
No threshold sweep, new prediction or training followed from this diagnostic.

## Loss and gradient findings

Across the four sampled diagnostic steps, both arms have finite, nonzero
module-level near gradients through deep projection, detail, support and near.
Weighted support gradients reach deep projection, detail and support; the two
near-classifier tensors are unused by that loss, as expected from the graph.
B's381 backbone tensors are explicitly frozen, while F's backbone receives
both gradient branches. No unexpected disconnected module was observed.
This is not a proof that every implementation or target semantic is correct.

The initial shared-head gradient norms are identical between arms before any
updates, consistent with the checked identical initialization and batch.
By final pre-update step2000, F's near BODY/HEAD BCE is0.00015595/0.00016499;
diagnostic per-head support BCE is0.0197888/0.0179808. The actual global support
BCE is0.0186574; per-head diagnostic values never replaced the training loss.
The final post-update losses in the table are separate evaluations. Raw L2
norms have different parameter counts and scales and do not establish which
branch causally dominates or whether two gradients oppose one another.

Disposition: keep G13-D and retain this fitting evidence. Frozen B can memorize
tiny classification, so the broader failure cannot be described as a wholly
broken classification gradient path. Allowing the backbone, including its BN
affine parameters, to update materially improves localization on this subset;
the running statistics remain frozen in both arms. This does not prove that
the frozen model cannot improve with another budget or that F will generalize.

The next clean comparison is full-parameter fitting on the same1134 TRAIN
frames for the same2000-step budget as the prior B run, from the original G13
initialization, with final-only unchanged DEV calibration. Do not warm-start
from these memorized tiny checkpoints or change the loss simultaneously.
That comparison has not been run here. No current weights are promoted, and
no DEV/plaza/Willow or fresh-test score is claimed.

## Verification and persistence

Source syntax/import preflight and a focused CPU autograd check pass: diagnostic
gradient calls leave`.grad` unpopulated, distinguish frozen/unused tensors and
permit the ordinary backward afterward. Native subset balance and known-positive
mask consistency pass. Actual CUDA runs check all losses/gradients finite,
unchanged B backbone, unchanged BN buffers in both arms and unchanged input hashes.
Source hash for the executed runtime bytes:
`910118a0bead5e52e44de8f7e38979976eb8fcb02d3d64465c392beb8761d006`.

Worker RTX3060 Laptop, Torch2.9.1+cu128: B71.669s, F264.392s, total script339.850s
(job343.07s). Exactly two2000-step fits; no retries or additional model inference
on other data. Main thin evidence at
`artifacts.local/work/city-tiny-fit-20260908/model-run-v1/` contains22 transferred
files/403,857bytes, each SHA verified, plus later release/transfer receipts.
Result SHA`d4e30d6a5010b34e6056bfe6ccaf0d41cfe702231ee2ff7eec242ba19fef98be`.
Worker retains both checkpoints and optimizer/RNG state at the same work prefix;
F checkpoint SHA`1d18bfba8c654d17fde423733882d06936717401de0a646e4e6ec3e3b32f7692`.
Processes and scheduled jobs are released; durable data/evidence remain and
temporary transport archives were removed.
