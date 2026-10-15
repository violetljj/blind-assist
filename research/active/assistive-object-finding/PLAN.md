# Assistive object finding: small Development pilot

Date: 2026-10-15. Status: EXPLORE completed; see README.md and RUNS.md.
Not a thesis-topic decision. The scope and stop conditions below are retained.

## Question and deliverables

Check actual public-data usability, residual errors of an accessible pretrained
language-grounding baseline, and measurable headroom for (1) distinguishing a
specified instance from similar candidates and (2) allocating re-localization
work after loss or ambiguity. Deliver runnable acquisition/inference/analysis,
input and model receipts, per-frame predictions, failure previews and a short
decision report. No app integration or protected/final evaluation is needed.

## Bounded scope and budget

- At most 24 video sequences / 600 labeled frames initially; adjust the mix
  within those limits to include visibility transitions and multiple candidates.
- New network payload at most 6 GB including models, data and dependencies;
  data acquisition subtask at most 1.5 GB. Reuse existing compatible runtimes.
- Local GPU process wall time at most 1,800 seconds; CPU-heavy processing at
  most 1,800 seconds; total pilot wall time at most 90 minutes from 13:00 HKT.
  Metadata and downloads are recorded separately. No new paid compute.
- Main hardware: local RTX 5060 Laptop, 8 GB VRAM; use actual CUDA inference.
  Start with existing weights/runtime; record limits if the strongest full
  systems cannot run under this pilot budget.
- All payload, cache and temporary files: artifacts.local/work/
  assistive-object-finding-pilot-20261015/ (canonical F: junction).

## Data and comparisons

Try updated official RefEgo annotations first. The user subsequently authorized
the official Ego4D access application and supplied its required fields; that
request was submitted. Agreement signing and approval are still pending. Do not
bypass authorization or treat submission as a granted video license. If pixels are unavailable,
record this explicitly and use a separately named public video Development
dataset (prefer MeViS train/public-label split) for an actual paired pilot.
No hidden test data is needed. Preserve official labels and annotation provenance.
Absence requires an explicit label or an annotated empty target mask.

Use one accessible modern pretrained grounding model and one independently
scored candidate/temporal comparison where feasible. Keep model, resolution and
candidate pool fixed for mechanism comparisons. Report candidate recall and
oracle selection only as diagnostic upper bounds, never as method improvements.
Fixed confidence threshold starts at 0.25; curves may describe sensitivity but
must not be reported as a newly confirmed chosen operating point.

## Decision and stop conditions

Continue exploratory implementation while it can resolve data readiness,
candidate omission vs ranking mistakes, absent-target errors, and the accuracy /
re-localization-call tradeoff. Stop the affected work at the applicable budget
or when these questions are answered. Retain failures. A failed source or model
startup is not a negative result for the algorithm hypothesis. Actual heuristic
gains, oracle headroom, cross-domain findings and NOT_RUN results stay distinct.

Human effort is limited to any necessary account authorization, reviewing the
task definition, and inspecting a small diagnostic montage. No new manual boxes
or user study is required for this pilot.
