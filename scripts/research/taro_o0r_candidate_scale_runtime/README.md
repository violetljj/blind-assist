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
