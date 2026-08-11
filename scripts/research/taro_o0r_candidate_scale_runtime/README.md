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
