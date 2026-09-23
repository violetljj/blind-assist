# Development log

This file records only current milestones. Full earlier history is preserved at
`archive/pre-agent-surface-2026-08-26` and searchable through
`experiments/index.jsonl`.

## 2026-09-23 — v10.15.1 restore home and explicit Start assist

- Corrected the user's intended scope: MainActivity is launcher again; home stays
  idle. Its existing Start assist action selects Wi-Fi A+LOCAL and transfers speech/
  vibration toggles, instead of dispatching the old phone-camera algorithm.
- End assist returns to home and lifecycle cleanup stops acquisition/feedback.
  Hardware Activity is internal-only; frozen model and threshold bytes unchanged.
- Updated the affected entry, idle/start/end, settings handoff and detail-toggle
  regressions. Validation receipts live under artifacts.local/hardware-bringup/home-assist-20260923.
- Six device tests, debug/androidTest build, lint and APK checks passed. Installed
  v10.15.1 without clearing data, inspected the restored home, and left the session idle.

## 2026-09-23 — v10.15.0 frozen A+LOCAL phone experiment

- User requested the research combination on the hardware phone. Added the exact
  retained-A integral/threshold and frozen LOCAL HGB classifier, with source/model
  hashes and no fit or threshold change. Portable core tests cover 111 A oracle
  cases and 48 LOCAL feature/probability queries, including consumed observations.
- Hardware adaptation remains nominal: centre-cropped VGA, 100-degree camera,
  45-degree ToF, original coarse orientation, zone-centre radial-to-axial conversion.
  No physical calibration, motion synchrony or simulation-to-real accuracy claim.
- Switch defaults to experiment and persists; the original ToF mode is retained.
  LOCAL uses central queries only, runs off-main without queues, and cannot survive
  stale RGB/ToF or dispatcher delay. Camera loss retains A; LOCAL-only alerts keep
  UNKNOWN and no attributed distance. Saved evidence retains original verdict/status.
- Independent integration review found a post-dispatch RGB-expiry gap, fixed with
  a regression for both expiry and cached-verdict/new-preview cases. Seven focused
  JVM tests passed without skips. Device/release receipts are recorded in the guide.
- Eight S24 Ultra / Android 16 checks passed. Real-input 10 s window completed
  101/101 inferences, processing median/P95/max 53/61/82 ms; no physical event labels.
  Build/lint and APK identity/signature/16 KB checks passed; installed without data
  clearing. Final asset metadata newline normalization changes no code/model bytes.

## 2026-09-23 — v10.14.0 local evidence, replay, mount check and split inputs

- Added a bounded foreground ring (20 s / 256 frames), immutable saved decisions,
  optional asynchronous 2 Hz thumbnails, no-backup local records and explicit export/delete.
  Replay owns its timeline, disables output, stops live acquisition and restores paused
  after Activity recreation. Generation checks prevent late live/replay writes.
- Camera and ToF validity now gate their own output. ToF-only operation is explicit;
  stale ToF cannot keep a verdict alive. Full-pair liveUsable retains its former meaning.
  Firmware, 10 Hz sampling, 750 ms expiry and geometric parameters remain unchanged.
- Four guided captures check coarse center/left/right consistency against background;
  session-only result is not calibration and never changes geometry.
- 31 focused JVM tests and 17 S24 Ultra / Android 16 tests passed. Three 8 s real
  windows yielded 479/479 full-live samples; recorder offer P95 scalar/thumb was
  0.242/0.398 ms. Save/load preserved decisions; role-rejected camera and unavailable
  ToF checks passed. No optical, accuracy or user benefit claim.
- Visual smoke saved a 19.989 s, 234-frame, 2-event scalar clip on the phone and
  exercised its replay; screenshots cover live/records/replay/mount panels.
  System document export to Downloads matched the private record SHA256 byte-for-byte;
  both scalar copies are retained for the user. Test acquisition was stopped afterward.
- Debug + test APK builds, lint (0 errors / 19 warnings), signature/version/16 KB
  verification passed. v10.14.0 (43) installed in place and archived.
  Evidence: artifacts.local/hardware-bringup/app-evidence-20260923-*.
  Physical mount placement, manual TalkBack, human output perception and long-run
  behavior remain untested. All test-owned cache fixtures and debugger forwards released.

## 2026-09-23 — v10.13.0 hardware-first entry and event feedback

- User-authorized launcher promotion: HardwareDemoActivity is the main launcher;
  MainActivity remains available through More functions. Added direct offline retry.
- Added a pure Kotlin feedback policy: immediate first alert, 12-second reminder,
  300 mm approach escalation with 1-second pacing, 1-second non-alert release,
  single loss notification and recovery. UNKNOWN never announces clear space.
- Speech and vibration have independent persisted switches and accessibility labels.
  Replay is silent; leaving the page releases acquisition and stops feedback.
- 15 focused JVM tests and 7 on-device tests passed on S24 Ultra / Android 16;
  final accessibility-label revision reran its 2 affected Compose tests successfully.
  10-second wireless receipt: 287/287 usable snapshots, 203 camera / 100 ToF frames.
- Final debug APK 42 / 10.13.0 installed in place and archived; signature/version/
  16 KB static checks passed. No firmware, sampling, geometry or freshness change.
  Human speech/haptic experience, manual TalkBack and long-run behavior untested.
  Evidence: artifacts.local/hardware-bringup/app-entry-20260923-*.

## 2026-09-23 — bounded camera coalescing A/B/A, original retained

- Tested one camera-only mechanism: copy JPEG to a bounded PSRAM buffer, return
  camera framebuffer before TCP send, combine multipart header/JPEG into one chunk.
  VGA/Q12 and ToF 10 Hz were fixed; original timestamp semantics preserved.
- Three 45-second runs gave camera age-bound P95 200 / 161 / 175 ms (A/B/A), with
  all snapshots usable. First apparent 19.5% gain narrowed to 8% against restored
  baseline; a stable substantial improvement was not established. Original firmware
  and source restored, transport rechecked. No app version bump or new mechanism.
- See [trial receipt](research/active/hardware-bringup/CAMERA_LATENCY_TRIAL_20260923.md).
  Physical scene-to-screen optical latency remains unmeasured.

## 2026-09-23 — v10.12.1 latency reduction, fixed 10 Hz

- Initial 15 Hz / 5 ms HTTP poll pilot reduced ToF snapshot age bound from 177
  to 99 ms median, but had 17/1303 invalid snapshots in one transient burst.
  Preserved that receipt; user then explicitly fixed ToF at 10 Hz.
- Final firmware v3 sends new samples via leased UDP3335 (1 s renewal, 5 s lease).
  Phone rejects foreign, duplicate, regressing and expired packets with per-boot
  UDP3333 clock mapping. HTTP remains diagnostics-only. No stale sample refresh.
- Removed Activity's 33 ms plus IO-dispatch poll delay for wireless snapshots;
  lifecycle-owned Choreographer callback consumes latest state at display refresh.
- Final direct-hotspot 45 s transport test: 1300/1300 usable, 996 distinct camera
  frames, 445 ToF frames; ToF sequence advanced 455. Snapshot age upper bounds
  camera median/P95 86/162 ms, ToF 71/132 ms. ToF arrival bound median/P95 14/44 ms.
- Camera acquisition/JPEG-ready median 36.6 ms, transfer bound 14.9 ms, decode
  5.4 ms, copy 2.3 ms. Preserved VGA and safe bitmap ownership. Independent
  distributions must not be added as a single frame's end-to-end latency.
- Old HTTP ToF bound included full request RTT; UDP uses clock mapping instead.
  The smaller bound is partly tighter estimation, not a controlled claim of 60%
  physical response improvement. Readout excludes preceding sensing and I2C time;
  neither screen latency nor increased-frequency ranging accuracy is established.
- Evidence: ignored hardware-bringup/wifi-latency-20260923. First packaging
  attempt failed without a diagnostic cause; stacktrace retry succeeded.
- Final build/lint, six contract tests and APK verification passed; final
  Choreographer UI displayed live on-device and ToF disconnect/reboot recovery
  was checked by screenshots. Final APK SHA256
  437A26013D1E30CE2020976631196C02F05D86CB78EE4F6CC375726288D8DA64.

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

## Earlier milestones

The unchanged earlier entries are preserved at Git
`2893137734524a0035839c844b1d6cb00bc25187:DEVELOPMENT_LOG.md`.
