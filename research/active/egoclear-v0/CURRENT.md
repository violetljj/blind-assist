# EgoClear-v0（纯视觉自我中心碰撞感知）：当前状态

所属：**避障线**（独立纯视觉研究路线；不改动 `hardware-bringup`、`dtr-r0` 及 App 默认辅助）。

更新：2026-11-22（**阶段 17–20 落地 + 四域 126 段全新 0 交集无损盲考完结**）：确立 **`EgoClear-v14` Hyper-Omni** 为默认全能主力，根治 QAIRT 2.47 `Clip(min='')` 缺陷，并在不碰官方 `test` split 下完成 126 段全新 0 交集盲考（`Fresh Real` 冻结 `thr=0.50` `v14 54.7%` 全场第一、`Fresh ARKit` 亮/暗全 `0.0%`、`Fresh Synth @120 2.3%` 反超 `v11`、4 域光学拒答 `12/12 = 100%`）；同步修复实拍暗室 `UNKNOWN=0.96` 全档门控并切默认 `1.0x` 主摄装机。所有临时 Colab 会话均已核验停机（余额 `156.43 CU`）。

## 当前帕累托前沿（全域 `mono = 0.000%`、SM8650 双单子图 `0` yield）

1. **新默认全能主力 — `EgoClear-v14` Hyper-Omni（`MoGe-Turbo 84×148 + v11 35-ep Step + v13 隔离头`, `27.97M`, 6 通道 Omni）**：
   - **S24 Ultra 全 NPU 实测（24-clip / 72 帧未见多域 + 300 帧稳态）**：
     - **`v14 FP16`（旗舰默认，`154 + 32.7 MB`）**：24/24 未见多域 **命中误差 `0.000 m`**、深度 **`z_MAE = 0.0135 m`**、`omni6_MAE = 0.0007`；爆发编码器 **`25.72 ms`**，**300 帧稳态 `37.69 ms` (`29.42+8.27 ms`, `26.5 FPS`, `+0.3°C`)**，比 `v11 ClipFix`（`72.00 ms / 13.9 FPS`）快 **`1.91×`** 且快于旧 `v6.6`（`38.1 ms`）；Demo App 双引擎冷启 **`383.2 ms (v14) + 151.8 ms (v6.6)`**。
     - **`v14 Mixed`（`59.3 + 32.7 MB`, 小 `4.1×`）**：爆发 Enc **`18.22 ms` (`54.9 FPS`)**，**300 帧稳态 `27.82 ms` (`35.9 FPS`, 比 `v11` 快 `2.59×`, `+0.5°C`)**，`omni6_MAE = 0.0181`、`hit_err = 0.217 m`。
   - **多域与 Omni 战绩**：**`Real 40 Dark @80/120 = 36.9% (41/111)`**（历史新低，`dyn 7.41%`）、**`Real 40 dyn_closing AUROC = 0.9728`**（历史新高，`F1 0.591, FPR 0.41%`）、`Real 40 @120 = 27.9%` / `thr=0.50 = 21.6% (FA 166.6)`（双胜 `v11`）、`Real 40 dyn@0.5 = 14.81%`（腰斩）、`AbsRel = 0.1672`、**4 域 `blk/finger/white` 全 `100.0%` 无记忆拒答**、`ARKit @120 = 3.8%` / `Pre-Hit = 0.0%` / `HEAD FPR 12.12%` / 绕行 `93.62%`、单模型 `Aria iso_grav_crop @120 = 60.6%` 与 `CODa iso_crop = 50.0%` 均胜 `v6.6`。
2. **备选与融合伙伴**：**`v11 ClipFix`**（`72.0 ms`, `z_MAE = 0.0154 m`, 保有最高 `Real 40 @120 = 23.4% / Synth = 15.3%`）；**`v6.6`**（全 NPU，`ENS(v6.6+v14)` 在 `Synth @120 = 5.1%`、`CODa = 50.0%`、`Aria iso_grav_crop @120 = 58.5%`）。

## 关键限制与下一步

- 仍为 `EXPLORE` 阶段（未读官方 `test` split）；诚实局限：`v14` 单模型 `Synth @120 = 21.9%`（双融合 `5.1%`）、`ARKit Dark @120 = 21.2%`（`@200 = 0.0%`）、`ARKit drop FPR = 12.24%`（端侧滞回兜底）、`CODa dyn_closing AUROC = 0.338`；实走与 TalkBack/大字无障碍回归标 **`NOT_TESTED`**，未改默认 `:app`。
- **下一步**：解锁 S24 Ultra 打开预装 `v14` 的 `:egoclear-demo-app` 实走并用 `录回放` 采集走廊/下台阶片段，跑 [replay_handheld_audit.py](replay_handheld_audit.py) 做真机闭环评测。

## 文件入口

- [Phase 20 全新 0 交集盲考](PHASE20_FRESH_UNSEEN_BLIND_PROBE_20261122.md) · [Phase 19 `v14`](PHASE19_V14_HYPER_OMNI_20261122.md) · [Phase 18 `v13`](PHASE18_V13_APEX_OMNI_20261122.md) · [Phase 17 `v12`](PHASE17_V12_TURBO_OMNI_20261122.md) · [运行记录](RUNS.md)
- [演示 App](../../../apps/demos/egoclear-demo-app/src/main/java/com/linnan/blindassist/egoclear/demo/EgoClearDemoActivity.kt) · [流式引擎](../../../apps/canaries/egoclear-device-canary/src/main/java/com/linnan/blindassist/egoclear/EgoClearStreamingEngine.kt) · [回放评测](replay_handheld_audit.py)
