# Motion occupancy A0.1 Android probability-head canary readiness

Date: 2026-08-03

Status: `READY_BEFORE_ANDROID_DEVICE_EXECUTION`

The deterministic asset contains 1,716 feature rows from seven consumed A1
windows. It contains no outcome label.

- raw TSV SHA-256:
  `481987B60D80237B2E86C83C35CA05BA79C131C53265AD34F565F7E2512531A6`;
- compressed asset SHA-256:
  `6AC9FE2BBC7A6BCCB810B819C7B959C5902C5C2B215F9539026888005F76D595`;
- frozen model SHA-256:
  `624288C5231EF2A0881D44AC84B5ECDE553D9CAE38FBA7465351242A971EE3D2`;
- source report and RAFT identities match the frozen protocol.

The tracked asset is
`hftf-device-canary/src/main/assets/motion_occupancy_a0_1_android_head.tsv.gzbin`.
The instrumentation class is
`com.linnan.blindassist.hftf.MotionOccupancyA01DeviceHeadCanaryTest`.

Execution command:

```powershell
$env:JAVA_HOME = "E:\codex-tools\jdk-17"
$env:ANDROID_SERIAL = "R5CX10M8Y8X"
.\gradlew.bat :hftf-device-canary:connectedDualLoopShadowAndroidTest `
  -Pandroid.testInstrumentationRunnerArguments.class=`
com.linnan.blindassist.hftf.MotionOccupancyA01DeviceHeadCanaryTest `
  --max-workers=2 -Dorg.gradle.jvmargs=-Xmx2048m
```

Only the named class is admissible for this execution. A successful test is
device probability-head parity/runtime evidence, not full UniDepth/RAFT mobile
pipeline evidence.
