# EgoClear-v0 Phase 10：Android 端侧封装 + 骁龙 8 Gen 3 Hexagon v75 NPU 加速真机实测报告

- **日期**：2026-11-21（`EXPLORE` 通道，严格隔离于 `apps/canaries/egoclear-device-canary/`，零改动默认 `:app`、`dtr-r0` 与 `hardware-bringup`）
- **实测真机**：**Samsung Galaxy S24 Ultra（`SM-S9280`）** · Snapdragon 8 Gen 3 (`SM8650`, Hexagon v75 HTP NPU, `arm64-v8a`) · Android 16 (API 36) · 12 GB RAM
- **运行时栈**：Stage 1 编码器走 **Qualcomm QAIRT 2.47 SNPE HTP v75 FP16（离线图准备单一融合子图 `Total Subnets: 1`，NPU 缓存 `54.78 MB`）** + Stage 2 时序记忆与生存头走 **`onnxruntime-android:1.26.0` CPU 4T**
- **隔离模块文件**：
  - [build.gradle.kts](../../../apps/canaries/egoclear-device-canary/build.gradle.kts) · [CMakeLists.txt](../../../apps/canaries/egoclear-device-canary/src/main/cpp/CMakeLists.txt) · [egoclear_snpe_jni.cpp](../../../apps/canaries/egoclear-device-canary/src/main/cpp/egoclear_snpe_jni.cpp)
  - [EgoClearSnpeEncoder.kt](../../../apps/canaries/egoclear-device-canary/src/main/java/com/linnan/blindassist/egoclear/EgoClearSnpeEncoder.kt) · [EgoClearStreamingEngine.kt](../../../apps/canaries/egoclear-device-canary/src/main/java/com/linnan/blindassist/egoclear/EgoClearStreamingEngine.kt) · [EgoClearAndroidCanaryTest.kt](../../../apps/canaries/egoclear-device-canary/src/main/java/com/linnan/blindassist/egoclear/EgoClearAndroidCanaryTest.kt)
- **原始真机证据（`artifacts.local/evidence/egoclear-device-canary/`）**：
  - [HTP v75 NPU 60s 持续实测](../../../artifacts.local/evidence/egoclear-device-canary/egoclear_v66_htp_v75_npu_report.json) · [HTP v75 NPU 实景街景 5 帧精度实测](../../../artifacts.local/evidence/egoclear-device-canary/egoclear_v66_htp_v75_npu_real_report.json)
  - [CPU FP16 报告](../../../artifacts.local/evidence/egoclear-device-canary/egoclear_v66_fp16_android_report.json) · [CPU FP32 报告](../../../artifacts.local/evidence/egoclear-device-canary/egoclear_v66_fp32_clean_report.json) · [CPU INT8 报告](../../../artifacts.local/evidence/egoclear-device-canary/egoclear_v66_int8_arm64_report.json)

---

## 1. 真机四方案同台实测（`EgoClear-v6.6` 完整 24.53M+3.7M 双阶段流式，294×518）

| 推理方案 | 冷启动加载（enc + step） | 黄金 5 帧精度 MAE / MAX | 单调违背率 | `encoder` P50 | `step` P50 | **爆发整帧 P50 / 峰值帧率** | **60s 持续 5 Hz（200 ms 预算）** | 稳态 PSS / Native Heap | 温升 / 热降频 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Hexagon v75 NPU FP16 + CPU `step`（推荐）** | **`225.6 + 62.4 = 288.0 ms`** | 实景 **`MAE 0.00201`** / `max 0.0104`（白噪 `0.044`） | **`0 / 299`** | **`18.02 ms`**（**21.1× 提速！**） | **`20.06 ms`** | **`38.13 ms` / `25.98 FPS`**（**10.9× 提速！**） | **`0 / 299` 超时（100% 达标！P50 `56.76 ms`、Max `82.15 ms`）** | **`162.4 MB` / `59.1 MB`**（**PSS 降 3×，0 泄漏**） | **`27.7→28.1°C` (+0.4°C)** / **`thermal=0`** |
| CPU INT8 动态（`44.00 MB` 通用） | `283.0 + 72.5 = 355.5 ms` | 实景 MAE `0.0105`（白噪 max `0.149`） | `0 / 167` | `295.9 ms` | `30.4 ms` | `329.9 ms` / `3.01 Hz` | 持续 `364.2 ms` / `2.78 Hz` | `317.6 MB` / `349.1 MB` | `31.8→32.3°C` (+0.5°C) / `thermal=0` |
| CPU FP16 压缩（`57.22 MB`） | `317.2 + 69.4 = 386.6 ms` | **`9.54e-7`**（1 ULP） | `0 / 131` | `379.9 ms` | `26.1 ms` | `415.7 ms` / `2.35 Hz` | 持续 `475.0 ms` / `2.18 Hz` | `485.8 MB` / `408.5 MB` | `27.1→29.3°C` (+2.2°C) / `thermal=0` |
| NNAPI（Android 16 已废弃） | `552.2 + 85.6 ms` | `2.26e-6` | `0 / 25` | `5,770 ms` | `420 ms` | `6,172 ms` / `0.16 Hz` | — | `575.1 MB` | `thermal=0` |

---

## 2. 关键技术成果与诚实边界

1. **一次打通“零 CPU 回退”的 NPU 单一融合子图**：
   - 将 `EgoClear-v6.6` 完整编码器（`DAV2-Small + DPT Neck + feat_proj + disp_cues`，共 912 个算子、7.47 GMACs）经 `onnx_unique_constants` + QAIRT 2.47 `qairt-converter`（全程保持 NCHW IO，免转置）+ `snpe-dlc-graph-prepare --htp_socs sm8650 --optimization_level 3` 离线编译，生成 **`Total Subnets: 1`**（0 个 CPU 回退子图，NPU 缓存 `54.78 MB`，DDR 总读取仅 `69.3 MB`）的 Hexagon v75 FP16 离线缓存。
2. **全面超越 5 Hz 盲走实时指标**：
   - 爆发整帧仅 **`38.13 ms`（`25.98 FPS`）**，比纯 CPU FP16 快 **10.9×**（编码器单段暴降 **21.1×**：`379.9 ms → 18.02 ms`）；
   - 60 秒真实 5 Hz 节奏（299 帧 @ 200 ms 周期）实现 **`0 / 299` 超时**（P50 `56.76 ms`、Max `82.15 ms`，每帧留出 `> 117 ms` 空闲余量），内存从 `485.8 MB` 暴降至 **`162.4 MB`（Native 稳在 `59.1 MB` 零泄漏）**，电池整分钟仅升温 **`+0.4°C`**，全程 **`thermal_status = 0`**；
   - 在真实步行街景 5 帧序列上，NPU FP16 与主机端 FP16 ONNX 的走廊碰撞概率误差仅 **`MAE = 0.00201`**（`max = 0.0104`），`0` 次单调违背（白噪输入因 `q90-q10` 分母接近 0 放大了 FP16 舍入至 `0.044`，实景验证完全无损）。
3. **诚实范围声明（未证明什么）**：
   - 离线准备好的 DLC 当前打包了 FP32 原始权重 + 54.78 MB HTP 缓存（总计 `158.6 MB`），后续可通过 `--float_bitwidth 16` 或 QNN Context Binary 剥离原始权重压回 `~55 MB`；
   - 当前为真机离线张量 + 真实街景帧实测，**尚未**接入 CameraX + ARCore/IMU 实时位姿闭环与震动/语音告警 Demo，也**尚未**进行 TalkBack 无障碍手测或户外真实盲走测试（无障碍回归矩阵各项保持 `NOT_TESTED`）。
