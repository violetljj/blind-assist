# SAGE-LM V2-MARKER-POSE Live Seam Implementation

状态：`RESEARCH_ONLY_ANDROID_IMPLEMENTED / QR_EXACT_ID_PLUS_PLANAR_POSE / JVM_MECHANICS_8_OF_8_PASS / APK_BUILT / LIVE_DEVICE_NOT_RUN_NO_READY_DEVICE / DEFAULT_APP_UNCHANGED`

## 结论

独立的 `:semantic-anchor-demo-app` 已把原有实时
`SEARCH -> SEMANTIC LOCK -> LOST -> FRESH REACQUIRE` 接到 metric marker pose 与
target-front guidance。默认 `:app` 没有改变。

这条 seam 在每个 CameraX frame 上只接受一个可见的 exact QR payload。ML Kit 提供按左上角起、
顺时针排列的四个角点；已知 QR 物理边长固定为 `0.16 m`。运行时从实际后置 Camera2 的
sensor physical size、首帧 metadata focal fallback 与 capture-result 实际 focal length 构造当前 analysis image 的
近似 pinhole intrinsics，再用
square-planar homography decomposition（平面 PnP 特例）恢复 marker center、平面法向与距离。
固定 `0.65 m` standoff 由朝向相机的 marker normal 生成 target-front waypoint。

控制臂可在同一 UI 切换：

- `CENTER_BASELINE`：只按 marker image center 左右对准，并以 marker scale-derived range 完成；
- `PNP_POSE`：先收敛 plane yaw，再收敛 waypoint lateral，最后前进；连续两帧同时满足
  `waypoint range <= 0.22 m` 与 `|yaw| <= 12 deg` 才输出 `ARRIVE`。

重复 exact ID 不获得 physical-instance authority：同帧两个相同 payload 为 `AMBIGUOUS`。
`LOST` 立即输出 `STOP`，且只有同 ID 新鲜语义证据重新达到两帧门才进入 `REACQUIRED`；
appearance/tracker 不创建或恢复 identity。

## 验证

专项 JVM tests 共 `8/8` 通过，覆盖：

- frontal metric pose、lateral、range、normal-derived waypoint 与 reprojection；
- oblique marker yaw；
- PnP 的 LOST stop、双帧 arrival 与 center baseline；
- exact-ID lock/reacquisition、OCR normalization、reset 和 repeated-ID ambiguity。

```powershell
pwsh -NoProfile -File scripts/run_android_gradle.ps1 `
  :semantic-anchor-demo-app:testDebugUnitTest `
  :semantic-anchor-demo-app:assembleDebug
```

APK：`apps/demos/semantic-anchor-demo-app/build/outputs/apk/debug/semantic-anchor-demo-app-debug.apk`

- bytes: `75,547,568`
- SHA-256: `BBF2445728E49F8FE11B8A2CB32040D55BC44F9CB3C75091D2C35A279ED7BDBC`

Android health check 为 `degraded`：SDK、ADB、emulator、Java、Python 均可用，但
`ready_device_count=0`。因此本轮没有安装、没有读取真实相机、没有形成 18-run arrival/
lateral/yaw/premature-arrival/reacquisition 数字；`LIVE_DEVICE_NOT_RUN` 必须保留。

## 下一次设备执行

有唯一 ready physical device 后，使用同一 `0.16 m` printed QR 与固定 `0.65 m` standoff，
按 `3 initial headings x 3 start distances x {continuous, one LOST+reacquire}` 跑受控矩阵；
center baseline 与 PnP pose 应在匹配条件下比较。报告只写实际设备观测到的 arrival、终点
lateral/yaw、premature arrival、LOST stop 与同-ID reacquisition。手机内参是 Camera2 metadata
推导的 pinhole approximation，当前未校正 lens distortion；它足以进入 controlled canary，
不支持自然场景入口、通行性、导航安全、用户效果或默认 App 声明。

Claim ceiling：

`ANDROID_IMPLEMENTATION_AND_SYNTHETIC_POSE_MECHANICS_ONLY_NO_LIVE_DEVICE_EFFECT_NAVIGATION_SAFETY_OR_DEFAULT_APP_CLAIM`
