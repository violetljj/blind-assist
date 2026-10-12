# Development log

This file records only current milestones. Full earlier history is preserved at
`archive/pre-agent-surface-2026-08-26` and searchable through
`experiments/index.jsonl`.

## 2026-10-12 — v10.16.3 capture preview, transport and export

- Goal/scope: improve foreground phone collector stability, smoothness, frame handling and export speed; scoped code/build/device checks, no model or firmware work. Preserve original payloads, independent timestamps and per-record fsync. All evidence stays under `artifacts.local/evidence/phone-capture-opt-20261012/`.
- Read new preview frames in Canvas draw, with composition only observing availability. Fixed-rate latest-frame checks now run every16ms; duplicate receipts are not decoded. Batch CNH/IMU byte assembly, cache raw offsets, and disconnect HTTP before closing its buffered reader.
- Removed ZIP FilterOutputStream byte-at-a-time forwarding; use256KiB buffering, JPEG level0 and text BEST_SPEED. Progress/elapsed time and picker session-ID restoration added. Two same-phone30s captures,15.78/15.46MB original input: legacy29.091/36.513s vs optimized0.165/0.466s,176.5/78.4x; ZIP8.984→10.621MB and8.674→10.365MB. Cache-file benchmark, not a SAF/cloud speed guarantee. Actual system-picker local Downloads export of final5.05MB input completed in0.3s,3.392MB ZIP SHA equals cache export.
- Preserved first offline Atom failure and the two30s run's cleanup-only Compose hierarchy failure; both completed receipts/ZIP checks existed before cleanup. Replaced cleanup UI lookup with Activity close; corrected receipt-rate denominator to include samples sharing one TCP-chunk timestamp. Focused3-test rerun passed with two5s sessions; final16ms live10s test passed.40 capture JVM tests, final debug/androidTest/lint, APK identity/signature/16KB all passed.
- Two30s sessions saved794/708 RGB,155/155 CNH,3036/3024 IMU; stop completion329/88ms; independent host JPEG SHA and raw index/sequence checks passed. Final10s saved165/52/1014, stop185ms, preview156 unique images/10.021s, median19.874ms/P9539.254ms Android receipt→Canvas, compared to old5s median40.485/P9571.746ms. Frame rate is not claimed as an equal-source hardware gain: source RGB changed27.563→26.148→23.452→16.146Hz with larger JPEGs, while ToF~5.103Hz and IMU~100Hz. Atom synchronously sends JPEG before returning its framebuffer; sender sequence continuity does not prove no sensor-frame overwrites. Network vs receiver backpressure remains unresolved.
- Final version49/v10.16.3 APK260301878 bytes, SHA256 CF421F1CD82D9AA97ACBEBCB9336AA12078F80CB4D65F3F1A5E3600AAAF68D12; installed over existing data and archived. Earlier49 candidate and failed receipts retained. Actual raw/ZIP integrity does not establish clock synchronization or detector performance.
- Framework-free idle gfxinfo: old248/721 janky frames(34.40%), median7/P9536ms; final349/1241(28.12%), median5/P9532ms. Different24/38s windows and changing source JPEG load, descriptive only; tail jank remains. Ended on home with capture workers/connections released; final system-picker ZIP remains in Downloads.

## 2026-10-12 — v10.16.2 capture performance and lifecycle

- User requested speed/smoothness/stability of the three-stream page; scoped to this foreground collector, preserving all received raw records and per-record fsync.
- Isolated latest-only JPEG decode/Canvas transform and background HUD; moved controls to a serial worker with lifecycle generations. Reused append streams and SHA256; finalized counts use manifest with damaged/legacy fallback.
- Fixed old-now freshness race after waiting for storage and refused source registration during finalization; close/join stays outside the data lock. Preview grid/native-axis labels remain visible.
- 35 focused core tests passed, debug/androidTest/lint passed; final version48 APK260300372 bytes, SHA256 1B7AD5E47EC8527596A2442D1B2A086240CA72FCC59A144DDDC981FEB26334D3, identity/signature/16KB verified. Installed with existing records retained; first48 APK92DFD3B1... archived separately.
- First added draw-performance assertion failed with a sleep-only sample window; retained failed020734 receipt and advanced the Compose test clock during sampling. Independent screenshot then exposed clipped IMU footer, repaired explicit line heights and checked un-clipped measured bounds. Final system gfxinfo and live receipt results are in the [report](research/active/hardware-bringup/ATOM_IMU_PITCH_20261012.md).
- Actual5s saved138AtomRGB/26CNH/507IMU,26/26 causal,P9513.600ms,145 ZIP entries exact. External12s UI render10.145→29.728fps,jank94.31→32.60%,median27.990→6.354ms;tailP9538.203→39.078ms unimproved. Phone left ready preview without recording; return home releases its sensor connections.

## 2026-10-12 — v10.16.1 Atom RGB + CNH + IMU preview and capture

- Replaced this page's phone CameraX source with Atom original MJPEG JPEG/headers/device timestamps and independent Android receipt time; historical sessions remain exportable.
- Added simultaneous Atom picture, valid 8x8 ToF grid and native BMI270 attitude/six-axis previews with established orientation. Actual capture:138 RGB/26 CNH/508 IMU, 26/26 causal matches, P95 age12.339ms;145-entry ZIP exact, Activity recreation/reconnect passed.
- 33 focused JVM tests, debug/androidTest builds, lint and APK identity/signature/16KB verification passed. Version47 APK:260093247 bytes, SHA256 EB1E814630F1DF50D61FBEFD5B994B9614DCDF0043719F1C4D14CADFAC253670.

## 2026-10-12 — v10.16.0 phone RGB + CNH + Atom IMU capture

- Goal/scope: foreground phone RGB + H3 CNH + Atom IMU capture, durable raw records, independent clocks, interruptions and ZIP export. No experiment/training budget; scoped build/tests and live acquisition/export acceptance, fixing observed failures.
- Added home entry, nominal VGA10Hz CameraX, full raw64x16 CNH, notes/session list, raw issue/tail retention, atomic manifests and recovery. No clock/geometry or model promotion.
- Added UDP discovery and dual USB-provisioned Wi-Fi profiles, preserving legacy NVS and switching on loss.
- Preserved initial H3 first-line EOF; repaired whole-row30ms budget to400ms/200ms no-progress. Post-flash is_alive failed, user cold power-cycle restored CNH. Actual24-frame decoder,50RGB/27CNH and system-picker exact ZIP passed.
- Atom full8MB flash backed up first:SHA256 a46ed2a0b5ed73fec719523d9c493dadcdc20ac23e4f0020e0bf7468658cd68e. IMU phase only flashed Atom; failed I2C1 collision retained, BMI moved toI2C0 after camera init, MJPEG/STA identity restored.
- Added TCP3336 raw6axis/gravity/native pose, QPC nearest receipt alignment, per-frame pose and phone raw/index persistence. Desktop60s:6002IMU99.992Hz/306CNH5.10298Hz,no gaps;pitch SD.059765deg/drift-.188655deg. Old15.625ms clock retained;QPC102/102,absP958.7625ms.
- Historical phone-camera LAN/hotspot/restoredLAN counts50/25/507,51/27/509,50/25/506; causal25/25,26/27,25/25,P9510.438/10.465/11.196ms; ZIP57/58/57 entries exact. First hotspot CNH correctly unavailable; hotspot off restored Atom4.250s/ToF6.469s in same boots, then homepage/clients released.
- Operator reference return +3.856718deg, no optical accuracy claim/retuning; companion mounting record saved. Real12-frame CPU frozen replay:7warm/5READY,no extra reset/nonfinite,hashes/thresholds unchanged.
- 27 Android focused tests/build/lint/APK/recreation passed; version46 SHA256 C14E12186E2F42D991F462A3F74C5D03E243E8BA089D393A7708BF89C9A90714. Firmware14/host11 tests passed. Complete failures, receipts, rollback and limits: [Atom IMU report](research/active/hardware-bringup/ATOM_IMU_PITCH_20261012.md), ignored phone-capture-v1/phone-capture-imu-v1/atom-imu-v1 payloads.

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
