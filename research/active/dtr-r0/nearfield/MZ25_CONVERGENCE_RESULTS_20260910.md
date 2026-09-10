# MZ25: cross-run prefix failed; deterministic sampling primitive validated

2026-09-10. The planned4800-step convergence comparison stopped at its declared
1200-step prefix gate. It did not produce a4800-step checkpoint, prediction set,
or convergence/task-quality result. No gate was loosened and no fit was restarted.
MZ5 baseline, MZ20 challenger and MZ24 objective component remain unchanged.

## What failed and what was checked

The [protocol](MZ25_CONVERGENCE_PROTOCOL_20260910.md) repeats the MZ24 batch stream
four times with the same initialization, head, objective and Adam configuration.
It requires the first1200-step checkpoint to match the separate MZ24 run at
atol=rtol=1e-4 before proceeding. All six learned tensors failed that comparison;
maximum absolute difference was0.314403. Fixed grid/relative tensors were exact.

Independent prefix audit confirms23 shared input hashes, unchanged availability
and loss module hashes, byte-identical initial weights/cutoffs and four exact
batch cycles. No pre1200 scientific update-path difference was identified. First
loss was exactly1.0424503088 in both runs, but subsequent losses differed. Neither
historical runner enabled deterministic algorithms or recorded full historical
backend versions/settings. Equal seeds did not establish cross-process trajectory
identity. This gate failure is not evidence that a longer optimization budget
helps or fails the obstacle task.

The process exited1 after1200 optimizer steps. `run-v1/receipt.json` was written
after exit with statusFAIL and reasonSTEP1200_PREFIX_PARITY_FAILED. Initial state,
4800 planned batch IDs, partial checkpoint, progress, start and prefix failure
are preserved. Final-only loss_samples/predictions were never written; they must
not be invented or treated as available. The complete-run auditor/extrema scripts
were prepared but not run, and are retained as ignored drafts.

## Fixed-input numerical diagnosis

`mz25_gradient_repeat.py` uses one saved MZ24 initial state and first-batch inputs,
without creating an optimizer. Four repeated forwards and losses are bit-identical.
However local convolution weight gradients differ in6367..6502 of9216 elements
between repetitions, with maximum absolute difference5.59e-9. Downstream MLP
gradients match. Enabling strict deterministic algorithms raises the runtime
error that `grid_sampler_2d_backward_cuda` has no deterministic implementation.

This directly demonstrates a numerical nondeterminism mechanism in the relevant
backward path. It does not uniquely attribute the entire0.314403 long-trajectory
drift to these small differences; historical runtime settings were not fully
recorded and hard-max/Adam trajectories can amplify perturbations. The appropriate
response is an explicit reproducible comparison design, not a larger parity
tolerance chosen after seeing failure.

## Implemented engineering component

`mz25_fixed_sampler.py` constructs a constant bilinear sampling matrix for the
fixed28x28 feature map and64x49 angular grid, using the same align_corners=False
coordinates and zero padding. Matrix multiplication supplies the forward and
adjoint without the grid sampler's scatter backward. `StableAngularAvailability`
keeps the same10577 learned parameters and checkpoint keys; its derived matrix
is not serialized. It adds9,834,496bytes of matrix storage (about9.8MB decimal).
This is a fixed-layout research primitive, not a general grid_sample replacement.

Generated-feature checks on the actual angular grid (including edge positions)
give forward maximum error1.97e-6 and adjoint1.91e-6 against the original sampler.
Initial-head normal/wrong-zone forward errors are below9e-8. Under strict CUDA
determinism, four backward calls have bit-identical gradients for all10577
parameters. No parameter was updated. The check records Torch2.11.0+cu130,
CUDA13.0 and cuDNN91900 for this diagnostic, not for the earlier failed trajectory.
Deterministic operation requires the tested backend configuration, including
CUBLAS_WORKSPACE_CONFIG and strict deterministic algorithms; a seed alone is
still insufficient. No full training trajectory has yet used this new primitive.

An independent real-feature check loads the unchanged final MZ24 weights and
covers all8 original added-false frames plus25pole frames clean and stress:
58frames,232 task bits. Availability differs by at most3.81e-6, with zero sign
changes; restricted raw scores, support and task decisions are exact. This
selected replay checks compatibility, not full-dataset or trained-model benefit.

## Disposition

Retain the fixed sampler as an engineering COMPONENT; preserve the failed MZ25
prefix as a methodological negative result. The next convergence experiment
should predeclare1200/4800 endpoints within one actual training trajectory using
the validated deterministic path and recorded runtime, so the shorter endpoint
shares a real prefix. This is a distinct future experiment, not retrospective
permission to continue the failed run or select a favorable checkpoint.

Durable outputs live under `artifacts.local/work/mz25-availability-convergence-20260910/`:
failed `run-v1`, `prefix-audit.json`, `gradient-v1`, `sampler-v1` and
`sampler-real-v1`. Failed-fit and
diagnostic processes exited; no capture/service or temporal state was created.
This turn supplies an implementation and numerical evidence, not better obstacle
accuracy, independent-source performance, device timing, App promotion or safety.
The full obstacle-improvement goal remains active.
