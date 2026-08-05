# AtomS3R-M12 + ToF4M Android 正式 QNN HTP R1

## 结论

正式 `com.linnan.blindassist` 的 AtomS3R-M12 实时链路已经在
`SM-S9280 / SM8650 / Android 16` 上真实进入 Qualcomm QNN HTP，而不是 CPU
回退。修复保留现代 native-library 打包：`useLegacyPackaging=false`、
`extractNativeLibs=false`；仅把 FastRPC 必须以真实文件访问的
`libQnnHtpV75Skel.so` 从 APK 校验后物化到 app-private `code_cache`。

一分钟正式短测处理 1457 帧，约 24.28 fps；0 sequence gap、0 latest-frame
覆盖、0 重连、0流错误、0记录错误。1457 次 `QnnGraph_execute` 全部返回
`status 0x0`，没有 `route=cpu_xnnpack`。

| 阶段 | P50 | P95 | P99 | 最大值 |
| --- | ---: | ---: | ---: | ---: |
| capture -> Android JPEG complete | 93.93 ms | 115.98 ms | 125.73 ms | 150.18 ms |
| JPEG decode | 4.46 ms | 9.38 ms | 12.75 ms | 14.03 ms |
| Bitmap -> RGBA | 0.00 ms | 0.00 ms | 0.00 ms | 0.00 ms |
| QNN HTP detector | 15.51 ms | 19.50 ms | 21.37 ms | 57.09 ms |
| capture -> risk complete | **115.13 ms** | **139.49 ms** | **148.41 ms** | **207.65 ms** |

这是 Development-only 的同设备部署与性能证据，不证明检测准确率、安全性、
真实用户效果、功耗优势或跨设备通用性。

## 失败根因

此前正式 App 的 capability probe 和 QNN backend 创建都成功，但 delegate 初始化
出现：

```text
qnn_open failed 0x80000406
Failed to load libQnnHtpV75Skel.so
Failed to create device_handle 14001
route=cpu_xnnpack
```

根因不是模型、算子或手机不支持 NPU，而是 HTP skeleton 的文件可见性：

- 正式 App 使用现代打包，native libraries 保留在 APK 内且不整体解压；
- `libQnnHtpV75Skel.so` 虽存在于 APK，但 FastRPC 需要可由文件路径读取的真实文件；
- 旧 provider 把 `applicationInfo.nativeLibraryDir` 传给 `setSkelLibraryDir`，该目录
  没有可读取的 V75 skeleton；
- 独立 `npu-candidate` 使用 `useLegacyPackaging=true`，skeleton 会真实落盘，因此
  对照包能成功进入 HTP。

该对照把问题定位为 packaging / FastRPC skeleton materialization，而非 NPU 能力。

## 正式修复

新增 `QnnHtpSkelInstaller`：

1. 只读取 APK 中 `lib/arm64-v8a/libQnnHtpV75Skel.so`；
2. 校验 ZIP entry 大小上限、解压后字节数与 CRC；
3. 先写 PID 隔离的临时文件并 `fsync`；
4. 验证后 rename 到
   `code_cache/qnn-2.47-sm8650-v75-skel/libQnnHtpV75Skel.so`；
5. 把该真实目录传给 `QnnDelegate.Options.setSkelLibraryDir()`。

正式 App 没有改成 legacy packaging。真机 app-private 文件为 17,265,488 bytes，
FastRPC 日志确认：

```text
Successfully opened file /data/user/0/com.linnan.blindassist/code_cache/
qnn-2.47-sm8650-v75-skel/./libQnnHtpV75Skel.so
Detector ready backend=qualcomm_qnn_htp
ProductionDetectorRoute: route=qualcomm_qnn_htp ... soc=SM8650
```

## 10 秒 smoke

- 244 帧；0 gap、0 overwrite、0重连、0流错误、0记录错误；
- 244 次 `QnnGraph_execute done. status 0x0`；
- skeleton 成功打开 1 次；QNN graph finalize `status 0x0`；
- `route=qualcomm_qnn_htp` 1 次，`route=cpu_xnnpack` 0 次；
- 结果目录：
  `artifacts.local/evidence/atoms3r-android-production-npu-smoke-20260806/`。

## 一分钟正式短测

- 请求 60 s，记录 60.161 s；1457 帧；
- 约 24.28 fps；0 gap、0 overwrite、0重连、0流错误、0记录错误；
- 对时成功 2 次、失败 0 次；ToF valid 1457/1457；
- Wi-Fi RSSI 最低和 P50 均为 -37 dBm；
- PSS 190,202 -> 196,231 KB，增加 6,029 KB；一分钟只作短期观测，
  不替代未来的长稳内存结论；
- `android_first_byte -> JPEG complete` P50/P95/P99 为
  32.62/43.84/49.73 ms；当前主要预算仍在设备采集、JPEG 准备与传输，而非 NPU；
- 实际 TTS/震动物理起点仍 `NOT_EVALUABLE`。

与先前 CPU 零复制一分钟基线相比，detector P50/P95 从约 53.30/53.61 ms
降至 15.51/19.50 ms，`capture -> risk` 从约 179.79/214.27 ms 降至
115.13/139.49 ms；CPU 基线处理 1030 帧并覆盖 324 帧，本次正式 NPU 处理
1457 帧且覆盖 0 帧。该比较支持同手机同输入链路上的性能改善，不授权算法质量结论。

## 构建与打包门禁

使用 Temurin JDK 17、Gradle `--max-workers=2`、2 GiB JVM 配置完成：

```text
:core:device:testDebugUnitTest
:app:testDebugUnitTest
:app:assembleDebug
:app:bundleDebug
BUILD SUCCESSFUL
```

仓库 `verify_apk_16kb.ps1` 严格门禁：

- APK：`PAGE_ALIGNMENT_16K`，37 个 native entries；所有 Android-loadable ELF
  的最小 PT_LOAD alignment 为 16384；
- AAB：`PAGE_ALIGNMENT_16K`，同样通过；
- HTP skeleton 是 Hexagon ELF machine 164，不是 Android loader 直接加载的 ELF，
  由门禁明确记录为 `NON_ANDROID_ELF_MACHINE`，不把其 4096 alignment 误判为
  Android 16 KB 页面失败。

因此本修复同时保留 FastRPC 可用性和现代 APK/AAB 16 KB 打包兼容性。

## 证据绑定

- 正式一分钟证据：
  `artifacts.local/evidence/atoms3r-android-production-npu-1min-20260806/`
- frames JSONL SHA-256：
  `ED6ED43EC9875BBD4038FD76314DC94FF822FD1D1A9B838880F20FE05E3F977A`
- summary SHA-256：
  `9859F00B45006789BA5006D691B1BB06FA15F854FCB76884F7D84E63D8EDF3F2`
- logcat SHA-256：
  `BB1FF66CEF8478AA2AD1068EE2272CA2609B68F63CE616AA8430E4C6AB84BF00`
- 重建 debug APK SHA-256：
  `9E7DD74901FA0D26720A9B2CD6A55642BA6563FE7F4E1BCA8EAB03453CEFB8F0`
- 真机实际运行的 `base.apk` 已拉取复核，与上述重建 APK SHA-256 完全一致；
- instrumentation test APK SHA-256：
  `885C039120A86CDD60A4F50FC3718BA81079838652FAE7C66164F72E02E0F78D`
- 重建 debug AAB SHA-256：
  `69862D9E5D7A52346660D949D1D81BB0029CC0A76B7A0BDE58C8B580B0D13CC1`

本地 evidence 位于 ignored `artifacts.local`；复现实验必须重新绑定 Git、APK、
设备、固件、endpoint 和 instrumentation 输出。

## 下一步

1. 保持日常顺序为 10 秒 smoke -> 1 分钟正式短测；大改或候选发布再做 5 分钟以上。
2. 优先调查偶发 NPU 推理最大 57.09 ms 和设备/Wi-Fi 长尾是否同帧相关；不先走固定曝光路线。
3. 手机在现场且有可控风险触发后，分别测 risk -> TTS request、audio physical onset、
   vibration call 和 physical onset；API 调用时间不替代物理反馈起点。
4. 相机与 ToF 外参标定继续暂停，ToF 仅作为带时间戳的诊断元数据。
