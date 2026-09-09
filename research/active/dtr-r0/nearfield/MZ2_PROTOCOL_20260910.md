# MZ2: spatial resolution and fixed corruption comparisons

2026-09-10 EXPLORE. MZ1 improves exact accuracy but fails its full replacement
criterion on error counts. Preserve it as a challenger and retain MZ0. MZ2 tests
information needs; it does not extend the stopped MZ1 fit or promote that model.

Use the same5000 consumed frames and original roles. Freeze visual features,
targets, MZ1 initial weights,300-step TRAIN_ONLY schedule and all hyperparameters.
Generate clean1x1 and4x4 two-surface observations directly from native depth using
the same45degree crop and summarization as MZ0. Replicate each coarse zone to its
corresponding8x8 grid footprint before flattening range/valid features. Replication
adds no spatial information. Train one FUSION MLP per coarse resolution, exactly
300 steps. Compare with the already completed8x8 MZ1 FUSION and RGB_ONLY; no8x8
retraining or resolution-specific hyperparameter/threshold selection.

Separately stress the frozen8x8 FUSION checkpoint, EVAL_ONLY, with three fixed
one-factor interventions: Gaussian range noise sigma0.05m seed61;20% whole-zone
dropout seed67; and a1-column lateral packet shift with vacated zones invalid.
The latter is an explicit coarse correspondence stress (roughly one zone), not
a calibrated small sensor error. Keep validity explicit. Apply noise only to valid
ranges and invalidate perturbed values outside(0,4]m. No combined corruption,
noise-trained model or claim that fixed8x8 inference estimates optimal4x4 capacity.

Report exact four-event accuracy, wrong-far, cross-body errors, and original
alert parity for every fixed arm. Resolution arms are matched retraining;
corruptions are frozen-model stress, and must be plotted/labeled separately.
Retain all outcomes, no winner selection. No temporal stability or hardware
noise-fidelity conclusion follows. Stop after two coarse-resolution fits and
three fixed stress evaluations plus independent checks and delivery.
