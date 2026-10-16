# ORBIT independent worker validation and exploration

2026-10-16, EXPLORE / thesis line. Owner: this Codex chat. The user explicitly
authorized independent validation and exploration on the registered secondary
worker. Claude's active source and outputs remain separate inputs.

## Question and decision

Check which part of the reported gain survives independent scoring and simple
controls, then test whether competition between personal objects improves the
use of existing local evidence. Retain a mechanism only if it beats its simple
matched baseline; negative and inconclusive results remain useful outputs.

Inputs are frozen, hashed copies of existing consumed ORBIT caches. Existing
17-test-user prediction arrays may be rescored for statistical/implementation
audit only. No new test-user inference or method selection on those users.
Method exploration uses the 44 training users; no new external downloads,
paid resources, credentials, or annotation. All findings are Development.

## Work and resource envelope

- First, CPU-only independent scoring of cached test predictions: h0, half-life
  30-frame EMA, finite 30-frame history sensitivity, and cumulative history.
  Report official-style 50 task resampling separately from full cached-valid-frame
  descriptions. Paired uncertainty resamples users, not repeated tasks.
- CPU-only exploration on train cached patch-object scores: existing full-frame,
  top-8, and fusion baselines; random-8 (three seeds), maximum and top-16 pooling;
  two fixed query-competition strengths (0.5 and 1). Select strength only on
  three training-user folds, evaluate on the fourth. Never read mask annotations.
- If useful and GPU capacity is released, independently reconstruct support
  memory selection controls on the existing six validation users: random retained
  patches, same-object consistency only, negative separation only, combined
  selection, and unscreened memory with the same prototype budget. A mechanism
  claim must isolate selection from prototype count and query pooling.
- Budget for this independent task: at most 2 GPU-hours, 2 CPU worker-hours,
  4 GiB LAN input transfer, zero external download. First inspect capacity;
  overlapping work must preserve practical RAM headroom and existing ownership.
  CPU diagnostics may proceed while another owner holds the GPU. Do not wait
  indefinitely for GPU capacity if CPU evidence already determines the next step.
- Stop an affected mechanism on consistent matched-control failure; do not rescue
  it with an unrecorded sweep. Preserve errors, receipts and consumed-data status.

## Decision check and delivery

The primary comparison unit is the user, with both video-weighted and user-macro
accuracy, saved/harmed query counts, and explicit frame/video/user denominators.
The existing large local-memory gain has headroom; the small learned-adapter gain
must be compared with the stronger simple pooling control. Positive exploratory
scores alone are not fresh confirmation or proof of real user benefit.

Deliver independent scripts, results, configuration/source/input hashes,
resource costs and an actionable recommendation in this directory. Raw payloads,
worker job logs and receipts stay under artifacts.local/work/orbit-independent-20261016/.
Keep task output and worker jobs uniquely named; retrieve thin evidence and release
owned processes/jobs when complete. Integration uses exact task-owned paths.

Completed: independent CPU checks, eight-arm support controls on all six consumed
validation users, and standalone combined/negative-only construction timing with
three alternating repeats. Query-competition failure is retained without another
strength sweep. Actual GPU-job wall time is 243.807 seconds; no new test-user model
inference was run. Results and remaining claim limits are in [RESULTS](RESULTS.md).
