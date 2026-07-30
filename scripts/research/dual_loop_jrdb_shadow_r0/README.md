# JRDB dual-loop shadow replay R0

状态：`DIAGNOSTIC / ENGINEERING_SHADOW_CYCLE`

This module freezes the host-only source
`JRDB_ANNOTATION_CONDITIONED_LIDAR_CENTROID_REPLAY_V1`. It joins existing
immutable JRDB observation packets to their sensor-support ledgers, derives a
signed target-relative range rate from consecutive real LiDAR centroids, and
writes a deterministic TSV for the real `AssistDecisionKernel` shadow seam.

The 2D target identity and LiDAR association come from JRDB source annotations.
This is therefore not an independent perception source, runtime Android
evidence, an alert-effect evaluation, or a safety/product claim.

## 稳定 Interface

Producer:

```text
python produce.py \
  --canary-packet <observation-packet.json> \
  --canary-ledger <ledger.json> \
  --cross-manifest <input-manifest.json> \
  --cross-ledger <ledger.json> \
  --output <artifacts.local replay.tsv> \
  --receipt <artifacts.local producer_receipt.json>
```

Kernel replay:

```text
./gradlew :core:assist:runDualLoopJrdbShadowReplay \
  --args="<replay.tsv> <producer_receipt.json> <kernel_receipt.json>"
```

## 输出

All generated outputs belong under ignored `artifacts.local/`. Both stages
refuse to overwrite an existing namespace.

## 安全边界

- independence unit: one baseline-selected target in one current frame;
- clock: `REPLAY_TIMELINE`, with image capture time as `capturedAtNs`;
- availability: the maximum of current RGB, lower-LiDAR, and upper-LiDAR
  timestamps; decision occurs no earlier than that value;
- source target set: exact JRDB 3D-to-2D joined annotations only;
- 2D regions: source boxes are deterministically clamped to the 3760x480
  stitched-frame bounds, with clamp counts recorded;
- geometry: `(|p_previous| - |p_current|) / dt`, positive means approaching;
- support: both endpoint object rows and their motion pair must be
  `sensor-supported`, and both endpoints must have an exact 2D join;
- quality: `1.0` denotes that the already-frozen binary sensor-support contract
  passed; it is not a calibrated probability;
- TTL: 100 ms;
- behavior source: replay detections remain `DetectionSource.OBJECT_DETECTOR`
  so the production object-detector path is exercised; annotation conditioning
  is carried separately as `DualLoopTargetProvenance.REPLAY_ANNOTATION`;
- receipt binding: the JVM runner verifies the producer receipt identity,
  exact producer implementation hash, input hash, and unopened outcome flag,
  then recomputes the frozen per-sequence row/frame/eligible denominators from
  the actual TSV before any kernel call;
- allowlist: explicit replay harness only; production remains empty;
- success: at least one actual kernel admission and zero risk/event/feedback
  parity mismatches.

## 停止条件

Any hash/binding/count mismatch, non-monotonic time, missing exact join,
non-finite rate, adapter abstention, or decision parity mismatch makes the
cycle invalid. An engineering-valid terminal is limited to
`ENGINEERING_SHADOW_CYCLE_VALID / DIAGNOSTIC_ONLY / NO_EFFECT_CLAIM`.
