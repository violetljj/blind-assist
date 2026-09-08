# Small native-instance City adaptation

Pre-model-outcome EXPLORE protocol, 2026-09-08. Follow the native-route diagnosis:
check whether modest source adaptation improves original natural obstacle
responses without changing architecture or expanding into bulk acquisition.

One TRAIN segment around x=-350m and one DEV segment around x=-240m of unchanged
Small_City_LVL, excluding the original40-frame regression route around
x[-300,-289]. Discover original meshes only; no inserted visible fixtures.
Select a thin pole/bollard, protruding meter and supported raised sign per
segment from native inventory and visual/geometry inspection before inference.
No shared actor-component-instance identity across TRAIN, DEV and named
regression targets. Capture camera-to-camera separation must exceed30m between
TRAIN/DEV and10m from the regression route. Same map and repeated asset types
remain shared: this is source-separated Development, not independent-world TEST.

At most96 new labeled views total, with near approaches and lateral offsets
providing visible-support positives and negatives. Scout/geometry-only repairs
precede model execution. Retain source disagreements and UNKNOWN. Use the
existing native scene-depth/floor labels and exact-instance/raycheck target
diagnostics, with full nearby loading and distant HLOD appearance. Require
each TRAIN head at least8 known positives and8 known negatives before fitting.
DEV operating-point claims require the same coverage; otherwise NOT_EVALUABLE.
Never replace geometry labels by intended placement labels. No outcome-based
view dropping or source replacement.

One seed17 fit, initialized from original G13-D seed17. Final step300 only,
batch16 deterministic sampling with replacement, AdamW weight_decay1e-4,
head LR1e-4 and backbone LR1e-6. Freeze all batch-normalization running buffers;
ordinary masked near BCE +0.25 existing class-balanced UNKNOWN-aware support
BCE. All parameters otherwise trainable. No early stop, augmentation, replay,
alternate loss, architecture change, additional fits, or budget rescue.
This tests the combined native-source/adaptation recipe, not isolated causality.

Compare initial and final weights at historical thresholds and at DEV-calibrated
thresholds. Per head, inclusive empirical FPR<=10%, maximize recall, then lower
FPR then higher threshold, including an all-negative sentinel. Fix and save DEV
selection before opening original-route labels. No choosing a checkpoint or
threshold from original-route outcomes. The original40route is already-consumed
regression only. Also compare first16 original Willow VAL samples to catch
large forgetting; they never enter TRAIN. Report per-head recall/FPR, support
localization, target misses, UNKNOWN and denominators.

Practical retention requires both DEV heads recall>=50% at FPR<=10%, nonzero
positive support IoU for both, and no increase in original-route target misses
or more than2 additional false alerts per head versus the original seed17 at
the same chosen evaluation policy. Willow16 regression must lose no more than2
correct head decisions. These are small correlated Development criteria, not
safety evidence. Failure retains the original model and identifies the observed
gap; success retains only a Development challenger. No automatic further fit or
capture follows either outcome. Preserve complete evidence and release processes.
