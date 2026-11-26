# EgoClear-v0（纯视觉自我中心碰撞感知路线）

所属：**避障线**（独立纯视觉 `EXPLORE` 研究路线；不改动默认 `:app` 基础 ToF 辅助）。

当前主力模型、S24 Ultra 全 NPU 实测与下一步决定以 [CURRENT.md](CURRENT.md) 为准。

## 核心里程碑入口

| 阶段 / 文件 | 内容 |
| --- | --- |
| [CURRENT.md](CURRENT.md) | 紧凑主线状态：`EgoClear-v14` Hyper-Omni 帕累托前沿与诚实局限 |
| [PHASE20_FRESH_UNSEEN_BLIND_PROBE_20261122.md](PHASE20_FRESH_UNSEEN_BLIND_PROBE_20261122.md) | 4 域 126 场景 / 11,569 帧全新 0 交集无损盲考（不触碰官方 `test` split） |
| [PHASE19_V14_HYPER_OMNI_20261122.md](PHASE19_V14_HYPER_OMNI_20261122.md) | `v14` Hyper-Omni（27.97M，6 通道 Omni，S24U 300 帧稳态 `37.69 ms` / Mixed `27.82 ms`） |
| [PHASE18_V13_APEX_OMNI_20261122.md](PHASE18_V13_APEX_OMNI_20261122.md) · [PHASE17_V12_TURBO_OMNI_20261122.md](PHASE17_V12_TURBO_OMNI_20261122.md) | 隔离头、光学 `UNKNOWN` 拒答与加速迭代 |
| [PHASE16_V11_OMNI_BLINDASSIST_20261122.md](PHASE16_V11_OMNI_BLINDASSIST_20261122.md) · [PHASE10_ANDROID_CANARY_20261121.md](PHASE10_ANDROID_CANARY_20261121.md) | `v11` 全能基线与首轮 Android 全 NPU canary 跑通 |
| [RUNS.md](RUNS.md) · [replay_handheld_audit.py](replay_handheld_audit.py) | 各阶段算力账本与真机手持录回放闭环评测脚本 |

## 对应独立 Android 模块

- 演示应用：[`:egoclear-demo-app`](../../../apps/demos/egoclear-demo-app/)（预装 `v14 + v6.6` 实时手持避障与录回放）
- 真机 Canary：[`:egoclear-device-canary`](../../../apps/canaries/egoclear-device-canary/)（双阶段全 NPU 流式引擎）

## 目录维护约定

Phase 0–20 既有文件保持路径稳定以保证复现命令有效；后续若新增多轮实验，须建独立子目录收纳，所有权重与大载荷统一存入忽略的 `artifacts.local/work/egoclear-v0*/`。
