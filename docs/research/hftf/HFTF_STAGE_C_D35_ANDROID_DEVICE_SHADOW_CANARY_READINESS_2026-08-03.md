# HFTF Stage C D35：Android device shadow canary readiness

日期：2026-08-03

状态：`READY_FOR_DEVICE_EXECUTION`

科学终态：`NOT_EVALUATED`

## 已完成的执行层

- 新增独立 `com.android.test` 模块 `:hftf-device-canary`；
- test build type 与 target App `dualLoopShadow` 同名绑定，不修改默认 App 的
  `testBuildType`；
- instrumentation target package：
  `com.linnan.blindassist.dualloop.shadow`；
- 编译时 production `BuildConfig`：
  `DUAL_LOOP_SHADOW=true`、`DUAL_LOOP_ACTIVE=false`；
- 直接调用 production `CausalTrackTristateGeometryProducer` 与
  `AssistDecisionKernel`；
- 5,366-row source-only D34 corpus 以 gzip payload 写入 test APK；
- APK 内 payload SHA-256：
  `91039be8a9d6282d89a8a9dc3e6200a8e8e09cc6f4fc43aa80c9ae935aeecfec`；
- 解压后的冻结输入 SHA-256 由 instrumentation test 在设备上校验为：
  `d1f24dc7c61890e912d2a4a1cbca23e4b729dfceb1ef76b435cd573c97e6021e`；
- device report 使用 Android `AtomicFile` interruption-safe 写入。

## build receipt

命令：

```text
.\gradlew.bat :hftf-device-canary:assembleDualLoopShadow
  --max-workers=2 -Dorg.gradle.jvmargs=-Xmx2048m
```

结果：`BUILD SUCCESSFUL`

- target APK：
  `app/build/outputs/apk/dualLoopShadow/app-dualLoopShadow.apk`
  - bytes：`258358692`
  - SHA-256：
    `e28e5c996174adef706f43ad6267a44e1c2ab017261ad99643b4efd4016a9557`
- test APK：
  `apps/canaries/hftf-device-canary/build/outputs/apk/dualLoopShadow/`
  `hftf-device-canary-dualLoopShadow.apk`
  - bytes：`622321`
  - SHA-256：
    `adffd1be8c401a65070c25b2e51263394311951d1f9986ef1693f812d8e695c3`

`aapt` manifest inspection 确认 test APK 的 `targetPackage` 是
`com.linnan.blindassist.dualloop.shadow`。

## 当前可用性

`adb devices -l` 未发现连接设备；本机也没有已配置 AVD。因此未安装 APK、未运行
instrumentation、未产生 D35 parity/runtime/non-interference 结果。

这是 device availability 状态，不是算法、数据或实现负结果，不烧毁 frozen corpus，
也不产生 `NOT_SUPPORTED`。连接 Android 设备后，唯一剩余执行为：

```text
.\gradlew.bat :hftf-device-canary:connectedDualLoopShadowAndroidTest
  --max-workers=2 -Dorg.gradle.jvmargs=-Xmx2048m
```

执行后必须从 target App external files 拉取 `hftf-d35/report.json`，核验报告与
instrumentation task 一致后，才能生成 D35 result document。
