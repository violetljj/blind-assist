# TARO O0R candidate-scale canary

This runtime tests one falsifiable algorithmic step: whether registered
AppleDepth can recover the metric scale of an already sealed DepthART candidate
without reading FARO or changing the candidate.

The estimator is fixed before the full replay:

- sample the registered `1440x1920` candidate at frozen Apple pixel centers;
- retain `confidence == 2` pairs where both depths are in `[0.25, 6.0]` metres;
- require at least 256 pairs;
- estimate `median(log(AppleDepth_m / candidate_m))`.

Execution is deliberately two-phase. All 239 source-only records are written
and a completion seal is persisted before the existing FARO canary records are
opened. The second phase is retrospective scoring only. It applies no decision
threshold and cannot authorize a formal O0R, deployment, product, or safety
claim.

Focused tests:

```powershell
E:\codex-tools\venvs\riskseg-r0-py311\Scripts\python.exe `
  -m unittest scripts.research.taro_o0r_candidate_scale_runtime.test_apple_scale -v
```

One-shot local replay (the evidence root must be absent):

```powershell
E:\codex-tools\venvs\riskseg-r0-py311\Scripts\python.exe `
  scripts/research/taro_o0r_candidate_scale_runtime/run_candidate_scale_replay.py
```

## R1 source-anchored factor/query canary

`source_factor.py` applies the already sealed R0 Apple scale *before* the
candidate depth-range gate, support-plane fit, obstacle-boundary extraction and
query point-clearance calculation. It independently re-derives the scale from
the bound AppleDepth/confidence arrays and sealed candidate, so a caller cannot
submit an arbitrary scaled raster. Raw and anchored branches use the same FARO
query-local comparison surface.

R1 preserves failed extraction and failed knownness as unevaluable/`UNKNOWN`.
It compares deterministic point-clearance values only; it does not run the
formal uncertainty reducer, tune an abstention threshold, or authorize a
PASS/FAIL, deployment, product or safety claim. The reliability fields (MAD,
q95 absolute deviation and 4x4 tile-median IQR) are post-hoc diagnostics only.

Focused tests:

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
E:\codex-tools\venvs\riskseg-r0-py311\Scripts\python.exe `
  -m unittest `
  scripts.research.taro_o0r_candidate_scale_runtime.test_apple_scale `
  scripts.research.taro_o0r_candidate_scale_runtime.test_source_factor -v
```

Full 171-frame CPU replay (the R1 evidence root must be absent):

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
E:\codex-tools\venvs\riskseg-r0-py311\Scripts\python.exe `
  scripts/research/taro_o0r_candidate_scale_runtime/run_source_factor_canary.py
```

R1 completed on 171 frames / 1,539 queries. It improved metric support-height
and boundary errors on paired evaluable queries, but lost extraction on 112
queries across 14 frames and recovered none, so unconditional pre-scaling is
not adopted. The authoritative result is
`docs/research/taro/TARO_O0R_ARKITSCENES_SOURCE_ANCHORED_FACTOR_CANARY_R1_RESULT_2026-08-11.md`.

The first R1 derived summary exposed a 12-decimal JSON round-trip seam (query
records remained valid). `reconcile_source_factor_evidence.py` validates every
original manifest/query/frame seal and writes a small R1A reconciliation root;
it never reopens source arrays or recomputes geometry. `_seal` now canonicalizes
before returning, and the focused test covers summary round-trip stability.

## R2 Apple-seeded candidate refit

`apple_support_seed.py` tested whether a confidence-bound AppleDepth support
plane could seed a new plane fit on the source-scaled candidate. On the 14 R1
lost frames it recovered only one frame / two queries, with zero height-and-
normal no-regret queries. Candidate refit/veto is rejected. The result is
`docs/research/taro/TARO_O0R_ARKITSCENES_APPLE_SEEDED_SUPPORT_RECOVERY_R2_RESULT_2026-08-11.md`.

## R3 direct Apple SUPPORT

`direct_apple_support.py` gives SUPPORT directly to confidence-2 AppleDepth and
keeps the source-scaled candidate only for dense boundary/query geometry. The
Apple SUPPORT mask does not depend on candidate range; invalid final height or
slope remains `UNKNOWN`.

Phase A rebuilds K/pose from the raw `.pincam` and trajectory instead of
opening compact truth. R1 query records, FARO and compact truth are read only
after the source-phase completion seal.

Focused validation:

```powershell
E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe `
  -m unittest `
  scripts.research.taro_o0r_candidate_scale_runtime.test_direct_apple_support `
  scripts.research.taro_o0r_candidate_scale_runtime.test_run_direct_apple_support_canary -v
```

The consumed R3 execution accounted for 14 frames / 112 queries, retained
eight source SUPPORT frames, made 58 queries support-evaluable and found 20
height-and-normal no-regret queries. It remains descriptive partial headroom;
the authoritative result is
`docs/research/taro/TARO_O0R_ARKITSCENES_DIRECT_APPLE_SUPPORT_R3_RESULT_2026-08-11.md`.

## R4 full-cohort direct Apple SUPPORT

`direct_apple_full_cohort.py` replays the exact R3 method across all 171
existing eval frames / 1,539 queries. It retains the two-phase source firewall,
uses no threshold or training, and binds every comparison to the sealed R1
query row.

Focused validation:

```powershell
E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe `
  -m unittest `
  scripts.research.taro_o0r_candidate_scale_runtime.test_direct_apple_full_cohort `
  scripts.research.taro_o0r_candidate_scale_runtime.test_run_direct_apple_full_cohort_canary -v
```

The consumed replay found positive parent-macro height and normal reductions on
all 16 parents, but direct-only selection recovered 36 extraction-evaluable
queries and lost 108. It is therefore not adopted unconditionally. The
authoritative result is
`docs/research/taro/TARO_O0R_ARKITSCENES_DIRECT_APPLE_SUPPORT_R4_RESULT_2026-08-11.md`.

## R4A zero-parameter hybrid

`direct_apple_hybrid.py` applies the frozen policy
`DIRECT_WHEN_SOURCE_SUPPORT_AVAILABLE_ELSE_R1_BASELINE_V1`. Its only selection
input is the source-only Phase-A plane availability; truth-derived extraction,
error and knownness fields cannot influence selection. The public validator
requires the exact external R4 row.

Focused validation:

```powershell
E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe `
  -m unittest `
  scripts.research.taro_o0r_candidate_scale_runtime.test_direct_apple_hybrid `
  scripts.research.taro_o0r_candidate_scale_runtime.test_run_direct_apple_hybrid_replay -v
```

The consumed replay used direct SUPPORT for 1,422 queries and baseline fallback
for 117. It recovered 36 extraction-evaluable queries versus baseline and lost
none; both height and normal parent-macro errors improved on 16/16 parents.
Known point-clearance still lost two baseline-known queries, so this remains
retrospective factor/extraction headroom. The authoritative result is
`docs/research/taro/TARO_O0R_ARKITSCENES_DIRECT_APPLE_HYBRID_R4A_RESULT_2026-08-11.md`.

## R5 parent-disjoint task-metric confirmation amendment

The R5 amendment freezes the same zero-parameter policy on the eight former
`ADAPTER_FIT` parents / 211 exact frames / 1,899 query slots. It introduces an
independent `R5_TASK_METRIC_CONFIRMATION` role instead of weakening the old
ADAPTER_FIT validators. Phase A must seal all model/source decisions with zero
FARO/task/prior-eval reads before Phase B can score the same frames.

Validate the amendment and all predecessor/source bindings:

```powershell
E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe `
  -m scripts.research.taro_o0r_candidate_scale_runtime.validate_r5_amendment
```

The validator also requires the immutable pre-implementation transform-ID
repair. The original amendment bytes remain unchanged; only its descriptive
candidate transform labels are superseded by the already sealed DepthART
runtime IDs (`RGB_CUBIC_IMAGENET_V1` and torch bilinear
`align_corners=true`). The repair was frozen while the R5 evidence root was
absent and both R5 inference and task-metric counts were zero.

Focused mutation tests:

```powershell
E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe `
  -m unittest `
  scripts.research.taro_o0r_candidate_scale_runtime.test_validate_r5_amendment -v
```

The hash-bound R5 implementation lock is now frozen and validates the
independent role API, phase-scoped source reader, one-shot runner, effective
DepthART transform IDs, exact 211-frame cohort and future execution-lock
validator. Focused amendment/core/runner/lock tests pass 26/26. The
implementation lock remains `execution=false`: no R5 evidence root, inference,
source decision or truth score has been created. Its only successor is the
one-shot R5 execution lock carrying explicit user model/task execution
authority.

R5 was subsequently executed on its exact 8-parent / 211-frame cohort and
terminated with `TARO_O0R_DIRECT_APPLE_HYBRID_R5_TASK_METRIC_CONFIRMATION_FAIL`:
SUPPORT/BOUNDARY improved on all eight parents, but query knownness changed
from 7 to 5. The consumed R5 result must not be rerun or promoted.

## R6 factor-level ownership

`r6_factor_split.py` implements the frozen successor policy without another
selector: SUPPORT and BOUNDARY exact-copy the Phase-A selected component, while
QUERY_CLEARANCE always exact-copies R1 baseline. Factor blocks carry separate
depth hashes, and validators reject a wrong-owner block even if it is resealed.

`FORMATION_REPLAY` can only emit a non-promotable landscape. Formal
`UNTOUCHED_CONFIRMATION` rejects all 24 formation parents and fewer than eight
new parents. The 1,899-query formation replay passed implementation checks but
does not authorize untouched execution. The only successor is a hash-bound
untouched cohort and data-use lock.

Focused validation:

```powershell
E:\codex-tools\venvs\riskseg-r0-py311\Scripts\python.exe `
  -m unittest `
  scripts.research.taro_o0r_candidate_scale_runtime.test_r5_confirmation `
  scripts.research.taro_o0r_candidate_scale_runtime.test_factor_split_canary `
  scripts.research.taro_o0r_candidate_scale_runtime.test_r6_factor_split `
  scripts.research.taro_o0r_candidate_scale_runtime.test_validate_r6_implementation_lock
```
