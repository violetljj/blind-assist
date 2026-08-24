# Semantic Anchor Live Demo V1 — implementation result

Date: 2026-08-24

Status: `LIVE_POSITIVE_CANARY_OBSERVED / NARROW_WRONG_TARGET_CONTROLS_PASS / DEFAULT_APP_UNCHANGED`

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
| Samsung SM-S9280 live QR | `LOCKED -> LOST -> REACQUIRED` observed from `LIVE QR` | positive physical-device marker canary |
| Samsung SM-S9280 live OCR | `LOCKED -> LOST -> REACQUIRED` observed for `ROOM 302` | positive physical-device text canary |
| Samsung SM-S9280 wrong QR `BLINDASSIST:ANCHOR:99` | `SEARCH`, `NON-TARGET`, lock `0` | fresh-session wrong-marker control passes |
| Samsung SM-S9280 wrong text `ROOM 301` | `SEARCH`, `NON-TARGET`, lock `0` | fresh-session wrong-text control passes |

Replay is deliberately labelled `REPLAY CANARY` in the UI. It is not a live identity result and must not be counted as uplift. The emulator had no external QR or text scene; positive evidence comes only from the separately recorded physical-device run below.

## Physical-device result

Device: Samsung SM-S9280, Android 16 / API 36, serial `R5CX10M8Y8X`

APK SHA-256: `9fef0b8d5b8df5acd125547936b32c765ecfb282c35064b0c80c930dd568ef20`

### QR marker

- Initial `LOCKED` was observed at 2026-08-24 18:00:04 +08:00.
- `LOST` was observed at 18:00:39 and a fresh `REACQUIRED` at 18:01:05.
- A stable sealed hierarchy later recorded `REACQUIRED`, `source=LIVE QR`, `lock=1`, `reacquire=6`; the larger reacquisition count reflects repeated target crossings while arranging the phone, not six preregistered trials.
- A second stable sealed hierarchy recorded `LOST`, `source=LIVE QR`, with the same lock/reacquisition counters.

Stable evidence:

- `live-device/04-marker-reacquired-stable/`: screenshot SHA-256 `f524968a801b7e42e5bb3a25fd11a6cf68a6777b45eb5b55458c218dd95ebd3c`; hierarchy SHA-256 `6b20e23213751c1680e4e866a274576b824abcf7c312e2400d46f6f6396959a3`.
- `live-device/05-marker-lost-stable/`: screenshot SHA-256 `da5c8ff3a21801b3836da2614de7ebac47b99cdbc33856725f2b1134ced0dd52`; hierarchy SHA-256 `b9398bfaa1b88e441d0a72962888ba7bea61f1103e67a12b965318f1fd549719`.

### Natural text

- The live camera initially read unrelated desk text and correctly remained `SEARCH`.
- The `ROOM 302` card produced `LOCKED`; pointing the camera at a text-free wall produced `LOST`, `source=LIVE OCR`, `lock=1`, `reacquire=0`; returning the card produced `REACQUIRED` with a fresh `MATCH` containing `ROOM 302`.
- Samsung's accessibility exporter could not reach idle while the live OCR UI was changing. These three states are therefore screenshot diagnostics rather than sealed accessibility hierarchies.

Stable screenshots:

- `live-device/06-ocr-locked-stable/`: SHA-256 `151bd7b4db10ff823fa801732300d47c6a549e78abccfebd2b1b708e98d42043`.
- `live-device/07-ocr-lost-stable/`: SHA-256 `3a514e626396622a40a1c686a4d5eb3af18cb6b41a50b292e3440d6db85c9c55`.
- `live-device/08-ocr-reacquired-stable/`: SHA-256 `de10910c6f96b5a7dc777859a85d1272268dc147b8b23e81a7a32df8d51c598b`.

### Wrong-target controls

- With target `BLINDASSIST:ANCHOR:17`, the visually valid but wrong QR payload `BLINDASSIST:ANCHOR:99` produced `SEARCH`, `NON-TARGET`, `source=LIVE QR`, lock `0`, reacquire `0`. The screenshot and accessibility hierarchy agree.
- With target `ROOM 302`, a separately displayed `ROOM 301` card produced `SEARCH`, `NON-TARGET`, `source=LIVE OCR`, lock `0`, reacquire `0`, while the preview visibly contained `ROOM 301`.
- An earlier `ROOM 301` attempt was discarded because the camera could see unrelated on-screen text containing the target token. The clean run restarted the session and used an independent card viewer. Samsung's live-OCR accessibility exporter still could not reach idle, so the OCR control remains screenshot evidence rather than a sealed hierarchy.

Stable evidence:

- `live-device/09-qr-wrong99-search-stable/`: screenshot SHA-256 `1a43e6bcf4f4ff259c62356e678296595612b87da1f3bddb52e3734ceaaad1a1`; hierarchy SHA-256 `072960721ee703abe906be7adeb7bbdad193ec03a8f02a90cf275c42a21a0548`.
- `live-device/10-ocr-wrong301-search-stable/`: screenshot SHA-256 `b1dbb141e208c6e8020785cf759e88f75115a3dac287d6cc1bdbbc3dce57d3e5`.

This run establishes the positive live canary plus two narrow same-device wrong-target controls: independent semantic evidence can drive lock, loss, and fresh reacquisition, while nearby wrong marker/text values remain non-authoritative. It does not yet establish uplift over the original passive `11/16` denominator, a broader wrong-lock rate, environmental generalization, four independent reacquisition successes, or promotion into the default App.

Diagnostic evidence is stored under ignored local storage:

- `artifacts.local/evidence/semantic-anchor-demo-app-v1/marker-replay/`
- `artifacts.local/evidence/semantic-anchor-demo-app-v1/ocr-replay/`
- `artifacts.local/evidence/semantic-anchor-demo-app-v1/live-ocr-blank-camera/`
- `artifacts.local/evidence/semantic-anchor-demo-app-v1/live-device/`

Runnable APK:

- `artifacts.local/evidence/semantic-anchor-demo-app-v1/apk/semantic-anchor-demo-app-v1-debug.apk`
- SHA-256: `9fef0b8d5b8df5acd125547936b32c765ecfb282c35064b0c80c930dd568ef20`

## Run

```powershell
pwsh -NoProfile -File scripts/run_android_gradle.ps1 ':semantic-anchor-demo-app:assembleDebug'
```

Install the generated APK, select `QR Marker` or `Natural Text`, set the expected semantic value, and start a fresh camera session. Move the target into view until locked, remove it until `LOST`, then return the same target. A different QR payload or different text must not reacquire.

## Next mainline action

Merge this semantic authority into the existing find/guide experience behind the research-only path, then measure the original passive denominator plus four independent reacquisition episodes. Do not reopen the closed appearance-only exact-instance lane.
