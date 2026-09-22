# Development log

This file records only current milestones. Full earlier history is preserved at
`archive/pre-agent-surface-2026-08-26` and searchable through
`experiments/index.jsonl`.

## 2026-09-23 — v10.12.0 phone hotspot and presentation

- Camera and 8x8 ToF join the phone hotspot; UDP discovery, latest MJPEG, bounded
  ToF reads and conservative freshness run directly in core:device. No PC relay.
- Reworked the initially rejected dashboard into a graphite camera-first page;
  raw grid and connection controls are collapsed. Risk algorithm is unchanged.
- Both original 8 MB flashes backed up and verified; only app0 flashed. Fixed
  an observed ToF warm-reset I2C failure with bus release and init retry.
- Six wireless contract tests and four USB client device tests passed. App/test
  APK and lint passed (0 errors, 19 warnings, including intentional local cleartext).
  Package/version/signature/16 KB checks passed; installed upgrade preserves data.
- S24 Ultra hotspot, 45 seconds after warmup: 1295/1295 usable snapshots,
  919 distinct camera frames, 230 ToF frames. Camera age upper bound median/P95
  95/203 ms; ToF readout age bound 177/310 ms. Clock error bound 5 ms.
  These are client observation ages, not screen, speech or exposure-sync measurements.
- First next-day test failed because hotspot was off; preserved that result.
  After enabling it, discovery and direct transport passed. ToF reset clears old
  output and recovery after the firmware fix was observed on the phone.
- Evidence, private config, original firmware backups and archived APK remain in
  ignored hardware-bringup artifacts; no credentials committed. TalkBack and
  real-world accuracy not tested. User dashboard retained, no ADB reverse used.

## 2026-09-22 — v10.11.0 hardware obstacle showcase

- Added an opt-in page in the latest default App with a read-only localhost USB
  bridge, native RGB/8x8 ToF display, full-zone corridor baseline and throttled TTS.
  Geometry lives in core:assist, transport in core:device, App owns presentation.
- Chose the basic support-intersection readout rather than the stronger calibrated
  or learned research arms. Hardware FOV, uncertainty and coarse alignment remain
  engineering assumptions, with invalid/stale input and nonalerts kept UNKNOWN.
- User authorized switching XIAO back to 8x8; backed up the complete old 8 MB
  Flash, changed only app0, and verified 80 continuous 64-zone frames. Atom unchanged.
- Eight JVM tests and four on-device client tests passed. Debug and test APK builds
  plus lint passed (0 errors / 18 existing-surface warnings). Package/version/signature
  and 16 KB alignment verified; Samsung SM-S9280 upgraded in place to 39/10.11.0.
- Bounded live collection saved 4242 JPEG / 919 ToF with no transport gaps/errors.
  Screenshots verified advancing live data, disconnect UNKNOWN, reconnection and
  explicit silent replay. TTS engine ready; audibility and alert accuracy not tested.
  Dynamic accessibility dump failed twice; screenshot evidence remains diagnostic.
- Archived APK SHA256: `79DE5B2869D95FB9C0E8B08286F799A97DF1AA215B394CC8A05F729676C2C0E2`.
  [Usage and evidence](docs/HARDWARE_OBSTACLE_DEMO.md). Owned capture/forwarding and
  foreground App session released; existing dashboard retained for replay.

## 2026-08-28 — v10.10.0 default app visual promotion

- Promoted the refined Compose home and settings experience into the default
  `com.linnan.blindassist` application rather than a candidate package.
- Bumped the install identity from `versionCode=37 / versionName=10.9.0` to
  `versionCode=38 / versionName=10.10.0` so the new APK upgrades the previous
  default app in place when the signing identity matches.
- Kept camera, detector, risk, feedback, permissions, packaged model, and
  experimental build isolation unchanged; this is a UI and version promotion.
- Built the default debug APK and verified package `com.linnan.blindassist`,
  `versionCode=38`, `versionName=10.10.0`, debug signing metadata, and 16KB
  alignment; SHA256 is
  `0D12E61078246C10946EE8557BCCF8EB8A2DEE18A7EB68E9BAE78CBA0CE58309`.
- Archived the verified default APK as
  `E:\linnan\blind-assist-apk-archive\apks\BlindAssist-v10.10.0-debug-20260828-003935.apk`.

## 2026-08-27 — dynamic travel risk mainline

- Preserved GRAIL R1C-L/G0/G1 and the unseen-location router at their terminal
  commits and removed their closed runners from the tracked active surface.
- Promoted `research/active/dtr-r0/` as the sole tracked algorithm route.
- Kept the first DTR-R0 question narrow: shared lifecycle, three credible
  baselines, and one route-tube future-occupancy change.
- Implemented the shared 0.50-second clear grace before real-input capture and
  froze route intersection versus radial TTC as the primary comparison.
- Added a truth-blind RGB, causal-track, flat-ground, and pose observation
  materializer for the 24-event input canary.
- The dependency-free synthetic run is mechanics evidence only. A 24-event
  real-input materialization canary and then an exactly 120-event staged RGB
  Development cohort are still required before any scientific gate or claim.

## 2026-08-26 — agent surface reset

- Reduced the root agent map to stable policy and routed dynamic state through
  `docs/PROJECT_STATE.md` and `docs/CURRENT_DECISION.md`.
- Preserved the complete pre-cleanup tree with an annotated remote tag.
- Kept one active research route: `research/active/grail-r1cl/`.
- Added isolated workstation profiles and a Codex desktop environment.
- Externalized the generated full dataset ledger; retained a compact summary,
  hashes, row counts, and a reproducible generator.
- Removed closed runners, contracts, schemas, snapshots, and reports from the
  current branch. Their experimental terminals remain historically true.
- Verified the configured R1CL runtime with a two-sample DINOv2 CUDA
  forward/loss/backward smoke. This is mechanics evidence only.
