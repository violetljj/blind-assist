# A568 异构平台准入预检 P0A R0

任务 ID：`HETEROGENEOUS_PLATFORM_P0A_R0`

日期：2026-07-31（Asia/Hong_Kong）

终点：`HOLD_NOT_EVALUABLE`

## 结论

当前主机的 Android 健康检查通过，ADB 可用且有一个 ready device；但该设备是
`Samsung SM-S9280 / SM8650`，不是目标 A568。没有发现 A568 的准确板卡、SoC、CPU/GPU/NPU、
SDK/推理运行时或可持续访问凭据。因此不能用 SM-S9280 的已有证据替代 A568，也不进入
YOLO11n 加载、固定帧 canary、调度器开发或性能比较。

## 已核验的主机与设备状态

### 主机工具

- Android automation health check：`ready`；ADB、emulator、Java、Python 均可发现。
- ADB：`E:\codex-tools\bin\adb.cmd`，Android Debug Bridge `1.0.41`。
- 当前 `adb devices -l` 只有：

  ```text
  R5CX10M8Y8X device product:e3qzcx model:SM_S9280 device:e3q transport_id:1
  ```

### 当前设备（仅用于证明目标不匹配，不作为 A568 证据）

| 属性 | 观测值 |
| --- | --- |
| model / product / device | `SM-S9280 / e3qzcx / e3q` |
| board / platform | `pineapple / pineapple` |
| SoC | `SM8650`, manufacturer `QTI` |
| CPU ABI | `arm64-v8a` |
| GPU hint | `ro.hardware.egl=adreno` |
| Android | `16`, API `36` |
| MemTotal | `11,350,704 kB` |
| serial | `R5CX10M8Y8X` |

这些字段明确说明当前可访问对象是既有 SM8650 手机；它们不回答 A568 的板卡身份，
也不构成 A568 的 NPU、温度、功耗或持续运行证据。

## A568 准入项

| 预检项 | 状态 | 证据/缺口 |
| --- | --- | --- |
| 准确板卡、SoC、CPU/GPU/NPU、内存 | `HOLD` | 没有 A568 设备或厂商硬件清单；不能由 SM-S9280 推断。 |
| 实际 SDK、推理运行时及版本 | `HOLD` | 没有 A568 侧 SDK/runtime 版本与可执行入口。 |
| 真实可持续访问 | `HOLD` | ADB 只发现 SM-S9280；没有 A568 serial/IP/连接凭据。 |
| YOLO11n 原始或等价导出模型加载 | `NOT_RUN` | 目标设备缺失；不在替代设备上代测。 |
| 固定帧逐帧结果、时延、错误导出 | `NOT_RUN` | 没有目标 runtime 和目标设备。 |
| 温度、功耗、系统日志能力 | `NOT_RUN` | 没有目标设备；当前手机能力不能代表 A568。 |

## 终止与重开条件

本轮在“缺少设备、运行时或可复算日志”处停止，终点保持：

```text
HOLD_NOT_EVALUABLE
```

只有在 A568 真实可访问后，才重开一个窄范围 P0A canary，至少补齐：准确硬件身份与
build fingerprint、SDK/runtime 版本、模型与数值格式、固定帧集、逐帧输出/时延/错误
receipt、温度与系统日志能力；功耗/能耗只有外部仪表可用时才记录。T4 固定双平台
可比性和 T5 离线 replay 在此之前均不启动。
