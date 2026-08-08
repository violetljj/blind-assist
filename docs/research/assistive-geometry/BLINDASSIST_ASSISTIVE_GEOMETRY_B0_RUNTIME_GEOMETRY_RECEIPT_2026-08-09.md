# BlindAssist Assistive Geometry B0 runtime geometry receipt

终态：`DEVICE_BENCHMARK_PASS / SM_S9280_OBSERVED / TENSOR_K_DERIVED / BENCHMARK_ONLY`

日期：`2026-08-09`

绑定合同：[B0 task contract](BLINDASSIST_ASSISTIVE_GEOMETRY_B0_TASK_CONTRACT_2026-08-09.md)

## 1. 设备与执行

- serial：`R5CX10M8Y8X`
- model/device：`SM-S9280 / e3q`
- Android：`16`，API `36`
- fingerprint：`samsung/e3qzcx/e3q:16/BP4A.251205.006/S9280ZCS6DZG1:user/release-keys`
- 隔离安装包：`com.linnan.blindassist.depthdemo`，`0.2-r0 (2)`
- Gradle task：`:device-benchmark:connectedDebugAndroidTest`
- tests：`CameraXAnalysisStreamGeometryAuditTest`、`CameraXSensorTransformAuditTest`
- 结果：`2/2 PASS`，`BUILD SUCCESSFUL`

旧的 depth demo 包因签名不一致阻止测试安装；在用户明确授权后卸载
`com.linnan.blindassist.depthdemo`（其包数据随之清除），再安装当前隔离 debug/test APK。
正式包 `com.linnan.blindassist` 未卸载、未覆盖、未清数据。

## 2. 实际 CameraX geometry

r832 在 30 帧上观测到单一几何：

```text
camera_id              0
requested              640 x 480, 4:3
actual buffer          640 x 480 RGBA_8888
crop                   [0, 0, 640, 480] (full buffer)
rotation               90 degrees
upright display        480 x 640
row/pixel stride       2560 / 4
backpressure           KEEP_ONLY_LATEST
timestamp monotonic    true
```

r833 观测到：

```text
active array           [0, 0, 4080, 3060]
sensor intrinsics      [2766.1165, 2771.1763, 2041.3307, 1530.0737, 0]
sensor -> buffer       diag(0.15686275, 0.15686275, 1)
principal buffer       [320.20874, 240.01158]
principal display      [238.98842, 320.20874]
```

这关闭了 B0 在当前 SM-S9280 canary 上“实际 buffer/crop/rotation/sensor-to-buffer 未观测”的
blocker，但不证明真实重投影、正式 App runtime 或其他设备具有相同几何。

## 3. Display K 与 tensor K

90 度旋转后，display 空间为 `480×640`。由 r833 的 sensor focal、sensor-to-buffer
scale 与旋转轴交换得到：

```text
K_display = [434.69433515,   0,           238.98842;
               0,           433.90064101, 320.20874;
               0,             0,             1]
```

B0 冻结的 DepthART resize 是 `480×640 → 448×608`，没有额外 crop/pad；因此
`sx=448/480`、`sy=608/640`：

```text
K_tensor = [405.71471281,   0,           223.05585867;
              0,           412.20560896, 304.19830300;
              0,             0,             1]
```

该 K 只绑定本次设备、camera id、actual crop/rotation 与 resize receipt。实现必须逐帧从
实际 CameraX geometry 派生；不得把这些常数硬编码成跨设备真值。任何 geometry 缺失或变化
必须 fail closed 为 `UNKNOWN_INPUT_GEOMETRY`。

## 4. 证据与 authority

- test result：`apps/benchmarks/device-benchmark/build/outputs/androidTest-results/connected/debug/SM-S9280 - 16/test-result.textproto`
- test result SHA-256：`57F31B105849CC90E323C0F35E1507373F632AC1BC1A2BE44D8E6B06A09AC669`
- HTML report SHA-256：`C6E79A01442AF542D11C3F7BD77CD4CD7BFD08067DBC2B93EABDC04307C70AAB`

本 receipt 只授权：

```text
SM_S9280_CURRENT_BUILD_CAMERAX_GEOMETRY_OBSERVED
DISPLAY_TO_448X608_TENSOR_K_DERIVATION_DEFINED
```

明确不授权：真实重投影、正式 App runtime、其他设备泛化、任务质量、student training、
QNN/HTP、latency、产品或 safety。
