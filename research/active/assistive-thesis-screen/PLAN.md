# Assistive thesis direction screen

2026-10-15, EXPLORE. The user requested broader thinking and actual small-scale
validation after linking the ongoing assistive object-finding conversation.

## Outcome and scope

Compare feasible graduate-thesis directions tied to blind and low-vision
assistance, without requiring new hardware or manual dataset creation. Run small
real-data checks that distinguish available supervision, baseline failures,
mechanism headroom and novelty overlap. This is topic selection evidence, not
formal method confirmation or user-benefit evaluation.

The existing `assistive-object-finding` route remains owned by its original chat.
This screen owns only this directory and
`artifacts.local/work/assistive-thesis-screen-20261015/` (canonical F: junction).

## Parallel checks and budget

| Branch | Check | New download ceiling | Compute ceiling |
| --- | --- | --- | --- |
| thin_structure | Existing manual SANPO masks; pretrained segmentation and input-resolution/detail comparison where feasible | 300 MB | 600 CPU command-wall s; 300 GPU process-wall s |
| route_memory | Genuine separate traversals if available; retrieval and causal sequence/rejection comparison | 300 MB | 600 CPU command-wall s |
| accessible_qa | Public VizWiz-related data; answerability/quality/reading feasibility and an actual small baseline | 300 MB | 600 CPU command-wall s |
| integration / novelty | LFVQA response-compression pilot, source overlap, negative-evidence data contract, metric inspection and report | 100 MB | 600 CPU command-wall s |

Total new payload ceiling 1,000,000,000 bytes; no paid compute/API, credentials,
hardware purchases, full-dataset training or app changes. Downloads and metadata
time are recorded separately. Sample mix can change within these budgets.
CPU metadata, classical retrieval and sparse statistical models are
TASK_NOT_GPU_SUITABLE; tensor inference uses an available accelerator after a
representative placement check. Do not interfere with the other active chat.

## Comparisons and interpretation

- Reuse explicitly consumed/train/public-development inputs; do not open protected
  test or frozen validation sets or resume a stopped historical experiment.
- Keep truth out of inference features. Label oracle ceilings, synthetic temporal
  perturbations, label-only statistics and real model gains separately.
- Preserve source identity, sample selection, per-item outputs, failures, units
  and denominators. Paired comparisons share images, model and preprocessing
  except for the intervention under examination.
- A classical/accessible baseline is not the strongest current benchmark system.
  Record modern strong baselines as NOT_RUN when not actually executed.
- Continue a direction if its mechanism has observable headroom and accessible
  evaluation. Deprioritize if only a known recipe, missing supervision, or an
  unsuitable proxy remains. No statistical significance gate for these pilots.
- Stop each branch when its decision is resolved or its budget is reached;
  release task-owned processes, preserve evidence, integrate scoped files only.

## Deliverables

Runnable scripts, retained source/metric receipts, branch reports, one ranked
Chinese synthesis with the strongest next experiment, CURRENT and a RUNS entry.
Relevant checks: inspect outputs, recompute decision-critical counts where useful,
`git diff --check`, and project structure/docs checks for the new route.
