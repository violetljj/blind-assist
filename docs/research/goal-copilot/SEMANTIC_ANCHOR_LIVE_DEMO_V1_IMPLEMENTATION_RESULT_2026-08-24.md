# Semantic Anchor Live Demo V1 — implementation result

Date: 2026-08-24

Status: `RUNTIME_SEAM_READY / POSITIVE_LIVE_ANCHOR_NOT_YET_OBSERVED`

App boundary: isolated research application; BlindAssist default App unchanged

## Outcome

The offline `SEMANTIC_DISTINCTIVE_ANCHOR_V1` result now has a runnable Android bridge:

`CameraX frame -> independent QR/OCR evidence -> SEARCH -> LOCKED -> LOST -> REACQUIRED`

This is not another appearance-identity matcher. No RGB appearance score, tracker continuity, pooling rule, or similarity threshold can grant identity. A state can enter `LOCKED` or `REACQUIRED` only after fresh matching semantic evidence.

The implementation lives in `apps/demos/semantic-anchor-demo-app` and installs as `com.linnan.blindassist.semanticanchor` (`Semantic Anchor Lab`).

## Narrow V1 sources

| Source | Default target | Identity authority |
|---|---|---|
| QR marker | `BLINDASSIST:ANCHOR:17` | normalized payload exact match |
| Natural text | `ROOM 302` | normalized OCR substring match |

Both ML Kit models are bundled in the APK, so a Play Services model download is not part of the demo path. Camera analysis uses `STRATEGY_KEEP_ONLY_LATEST`; an `ImageProxy` remains owned until its asynchronous recognition task completes and is then closed.

## State mechanics

- Two consecutive matching observations acquire the initial lock.
- Five consecutive observations without target evidence declare `LOST`.
- Two new matching observations after `LOST` produce `REACQUIRED`.
- A target change clears all previous lock and reacquisition state.
- Wrong marker values and non-target OCR text remain non-authoritative.

These constants debounce live observations; they do not infer identity from appearance.

## Verification

| Check | Result | Claim boundary |
|---|---:|---|
| `:semantic-anchor-demo-app:testDebugUnitTest` | PASS, 3 tests | state authority and reset semantics |
| `:semantic-anchor-demo-app:assembleDebug` | PASS | Android/CameraX/ML Kit integration compiles and packages |
| API 35 emulator QR replay | `REACQUIRED`, lock `1`, reacquire `1` | UI/state diagnostic only |
| API 35 emulator OCR replay | `REACQUIRED`, lock `1`, reacquire `1` | UI/state diagnostic only |
| API 35 emulator live OCR on blank virtual camera | `SEARCH`, `NO SEMANTIC EVIDENCE`; live frames processed; no crash | camera/runtime negative-control smoke only |

Replay is deliberately labelled `REPLAY CANARY` in the UI. It is not a live identity result and must not be counted as uplift. The emulator had no external QR or text scene, so a positive real-camera QR/OCR lock and reacquisition remains unobserved.

Diagnostic evidence is stored under ignored local storage:

- `artifacts.local/evidence/semantic-anchor-demo-app-v1/marker-replay/`
- `artifacts.local/evidence/semantic-anchor-demo-app-v1/ocr-replay/`
- `artifacts.local/evidence/semantic-anchor-demo-app-v1/live-ocr-blank-camera/`

Runnable APK:

- `artifacts.local/evidence/semantic-anchor-demo-app-v1/apk/semantic-anchor-demo-app-v1-debug.apk`
- SHA-256: `9fef0b8d5b8df5acd125547936b32c765ecfb282c35064b0c80c930dd568ef20`

## Run

```powershell
pwsh -NoProfile -File scripts/run_android_gradle.ps1 ':semantic-anchor-demo-app:assembleDebug'
```

Install the generated APK, select `QR Marker` or `Natural Text`, set the expected semantic value, and start a fresh camera session. Move the target into view until locked, remove it until `LOST`, then return the same target. A different QR payload or different text must not reacquire.

## Next mainline action

Put the exact QR payload and `ROOM 302` sign in front of a real device and capture one positive `LOCKED -> LOST -> REACQUIRED` episode for each source. If wrong locks remain zero, merge this semantic authority into the existing find/guide experience. Do not reopen the closed appearance-only exact-instance lane.
