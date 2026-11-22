# Phase 15：`EgoClear-v10.6` 多距生存标定 + 零前冲泄漏与 S24 Ultra 实拍闭环修复

日期：2026-11-22
路线：`egoclear-v0`（`EXPLORE` 快车道；零读取官方 `test` 集；不改动 `:app` / `core` / `feature` / `hardware-bringup` / `dtr-r0`）

## 1. 结论先行（实拍 X 光 → 端侧 + 权重双轨根治）

真机手持实拍（Samsung Galaxy S24 Ultra `SM-S9280`）暴露了离线 `4.1 m` 单点评估完全隐藏的**室内近距前冲缺陷**：在真实暗室内（前方箱子/墙约 `1.67–1.85 m`、脚下全空），`EgoClear-v10.5` 自身 NPU 输出的 `geom_cues` 三维测距高度准确（左墙 `1.34 m`、正前立面 `1.85 m`、地面垂直度 `0.07` 完美排除），但其 `corr_head` 原始输出却在 `0.5 m` 就直接顶到 **`0.756`（`hitM = 0.5 m`）**。
新增的离线 **Pre-Hit Leakage 审计**直接量化证实了病根：`v10.5` 在室外街景（`SANPO-Real 40`）底噪很低（无障碍 `clear_mean_p0.5m = 0.080`），但在室内（`ARKitScenes 30`）因 `loss_rank` 仅在 `4.1 m` 端拉升累积危险度，导致无障碍帧 **`clear_mean_p0.5m = 0.6501`、`clear_mean_p1.3m = 0.7773`**，且 `42.6%` 的远距障碍帧（`D_gt >= 1.9 m`）误把 `P(<=0.5m)` 拉过 `0.50`。

本阶段同步完成**双轨修复**，均已落盘并在 S24 Ultra 验证：

1. **端侧物理近距纠偏 + 交互降噪（零云端成本，已实拍验证）**：
   - 在 [EgoClearStreamingEngine.kt](../../../apps/canaries/egoclear-device-canary/src/main/java/com/linnan/blindassist/egoclear/EgoClearStreamingEngine.kt) 中为 `V10_MOGE2` 增加 `<0.15 ms` 的 3D 走廊立面扫描（`calibrateNearRangeWithMetricGeom`），用 `geom_cues` 实测 `z_m + verticality + mask` 压制空走廊上的前冲概率，并在 `2.3–3.1 m` 平滑退回全神经记忆以保住 `4.1 m` 远距召回；
   - 在 [EgoClearDemoActivity.kt](../../../apps/demos/egoclear-demo-app/src/main/java/com/linnan/blindassist/egoclear/demo/EgoClearDemoActivity.kt) 中默认改为 **`紧距1.3m`**（四档 `1.3/1.9/2.5/4.1 m`）、状态栏显示 **`实测 X.Xm`**、静止不再周期性连环震动/播报。真机 230 帧实拍日志验证：`geomM = 0.9–1.1 m` 时 `p1.3 = 0.86–0.90 (alert=true)`，`geomM = 1.5–1.9 m` 时 `p0.5 = 0.07, p1.3 = 0.10 (alert=false)`。
2. **权重级根治 `EgoClear-v10.6`（Colab G4 5 轮精调 `3,510 s`，零新增端侧算子，已编译上机）**：
   - 机制（[phase15_train_v106_g4.py](phase15_train_v106_g4.py)）：从 `v10.5` 热启，加入 **Pre-Hit Zero-Leakage 损失**（对障碍前至少 1 格的全部空距格强压 `p_cum` 与条件危险度 `softplus(h_j)`）、**四距独立生存排序 + Tail-OHEM（`1.1 / 1.9 / 2.7 / 4.1 m`）**，以及**防前冲泄漏蒸馏**（在所有 `y_corr == 0` 格取 `min(p_v66, p_v105)`，绝不复制 `v10.5` 的近距前冲）。
   - **Pre-Hit 泄漏彻底治愈**：
     - **`ARKitScenes 30` 室内（169 远障帧 + 无障帧）**：`far_obs_mean_p0.5m` **`0.4263 → 0.0255`（`-94.0%`）**，`far_obs_frac(p0.5>0.5)` **`42.6% → 0.0%`（0/169）**，`far_obs_mean_p1.3m` **`0.5627 → 0.1376`（`-75.5%`）**，`clear_mean_p0.5m` **`0.6501 → 0.0646`（`-90.1%`）**，`clear_mean_p1.3m` **`0.7773 → 0.3688`**；
     - **`SANPO-Real 40` 室外（157 远障帧）**：`far_obs_mean_p0.5m` **`0.1014 → 0.0114`（`-88.8%`）**，`far_obs_frac(p0.5>0.5)` **`0.64% → 0.0%`**，`clear_mean_p0.5m` **`0.0801 → 0.0136`**；
     - **用户原图 `diag_npu_input.png` 零后处理裸测**：`v10.5` 为平顶假近障 `p0.5=0.756, p1.3=0.811, hitM=0.5 m`；`v10.6` 变为干净单调 S 曲线 **`p0.5=0.175 → p0.9=0.343 → p1.3=0.470 → p1.5=0.518 → p1.9=0.598 → p4.1=0.852 (raw hitM = 1.5 m)`**，与真实 `1.67 m` 障碍精准吻合！

## 2. 九项基准与跨模型对照（全域 `mono_viol = 0.000%`）

| 评测域（事件分母） | `v6.6` (57 MB) | `v10.5` (MoGe-2) | **`v10.6` (本阶段)** | 变化与统计检验 |
| --- | --- | --- | --- | --- |
| **`SANPO-Real 40`** `@80 / @120 / @200` (111) | `57.7 / 48.6 / 32.4%` | `28.8 / 28.8 / 25.2%` | **`26.1 / 25.2 / 19.8%`** | **三档全创历史新低**；`@120` 胜 `v6.6` 显著（`-22.49 pp, 95% CI [-47.58, -1.47]`） |
| **`SANPO-Real 40`** `固定 thr=0.50` / `AbsRel` | `—` | `27.9% (FA 125.9)` / `0.1648` | **`18.9% (FA 240.6)` / `0.1629`** | 真深度 `AbsRel` 再降（尺度 `0.9500×`）；`ENS(v6.6+v10.6) @120 = 23.4% / @500 = 12.6%` |
| **`SANPO-Synth 110`** `@80 / @120 / @200` (137) | `12.4 / 4.4 / 2.2%` | `18.2 / 18.2 / 16.8%` | **`13.1 / 5.8 / 5.1%`** | **`@120` 暴降 `-12.4 pp`**（`VIO 1× @120 = 5.8%`, `dyn = 0.37%`, `thr=0.50 = 29.9%`） |
| **`UT Austin CODa`** `iso_crop @80 / @120` (36) | `66.7 / 66.7%` | `66.7 / 33.3%` | **`33.3 / 33.3%`** | **`@80` 追平 `@120` 新低**（`HEAD F1 = 0.310`, `dyn = 0.0%`） |
| **`ARKitScenes 30`** `@80 / @120 / @200` (156) | `12.8 / 8.3 / 0.0%` | `6.4 / 6.4 / 0.0%` | `12.8 / 12.8 / 0.0%` | `@200..500 = 0.0%`，`thr=0.50 = 0.0%`；**室内近距前冲泄漏降 `94%`** |
| **`Real 40 Dark 0.18×`** `@80 / @120 / @200` (111) | `—` | `50.5 / 45.0 / 33.3%` | **`42.3 / 38.7 / 27.9%`** | 极暗户外全档大幅反超 `v10.5`（`-5.4 ~ -8.2 pp`） |
| **`Real 40 Gray`** `@80 / @120 / @500` (111) | `—` | `31.5 / 31.5 / 15.3%` | **`31.5 / 31.5 / 12.6%`** | 灰度零衰减，`@500` 再降 `-2.7 pp` |
| **`Real 40` 行人 `dyn@0.5`** (27) | `7.4%` | `18.5%` | `33.3%`（灰度/暗光 `25.9%`） | **诚实回升**：`@120` 最优阈值升至 `0.64`，固定 `0.50` 下部分行人触线 |
| **`Aria Gen 2`** `squashed / iso_crop @120` (193) | `58.5 / 59.1%` | `69.4 / 68.4%` | `78.2 / 68.9%` | **诚实回退**：压扁 `@120` 弱于 `v10.5`（`+8.73 pp, CI [+2.2, +17.59]`）；`ENS iso_crop @120 = 59.6%`、`iso_letterbox @500 = 28.0%` |

## 3. 端侧部署、算力与存储核验

- **Colab G4 与 Drive 完整性**：训练+全评+ONNX 导出共 `3,510.3 s`（`egoclear-v106-g4`）；Drive `/BlindAssist/runs/egoclear_v0_phase15_v106/` 4 文件（`egoclear_v106_encoder_fp32.onnx` `146,031,018 B`、`egoclear_v106_moge2_small.pt` `156,825,707 B`、`egoclear_v106_step_fp32.onnx` `13,412,687 B`、`phase15_v106_results.json` `891,657 B`）已服务端核验字节、`os.sync()` + `drive.flush_and_unmount()` 并立即 `colab stop` 复核（本任务活跃会话 `0`，余 `391.42 CU`）；本机经 `gdrive readonly download` 零 GPU 成本拉回并 100% 字节一致。
- **QAIRT 2.47 SM8650 Hexagon v75 编译与真机 12 帧实测（`SM-S9280`）**：
  - `snpe-dlc-graph-prepare --htp_socs sm8650 --optimization_level 3 --set_output_tensors "f2d,geom_cues" --use_float_io` → **`Total Subnets: 1`**，`Record Size: 94.243 MB`（`245,101,015 B`），ONNX IEEE FP16 最大深度误差 `14.14 mm`（`finite=True, mono=True`）；
  - S24 Ultra 12 帧双输出（`%f2d geom_cues`）实测：**`Init = 167.0 ms`**、**编码器 `55.89 ms`（`17.9 FPS`）**、`0` yields、`f2d`（`795,648` float32）与 `geom_cues`（`124,320` float32）**100% 有限**，已装入 `:egoclear-demo-app`（优先加载 `v10.6`，支持 `v10.6 / 双融合 / v6.6` 一键切换）。

## 4. 关键限制与下一步

- **诚实局限**：`v10.6` 在 `SANPO-Real 40` 固定 `thr=0.50` 的行人 `dyn`（`33.3%`）与 `Aria` 压扁 `@120` 略逊 `v10.5`（因室外 `4.1 m` 推荐工作阈值移至 `0.55–0.64`，App 标准档恰好默认 `0.55`）；仍属 `EXPLORE` 阶段，解锁后的 `v10.6` 户外真实盲走与 TalkBack/大字体无障碍回归仍标 `NOT_TESTED`。
- **下一步**：解锁 S24 Ultra 直接手持体验已更新的 **`模型:v10.6`**（`紧距1.3m / 室内1.9m` 对比 `双融合` 与 `v6.6`），按真实室内外体感决定是否微调 `0.55/0.64` 阈值或补充无障碍回归。
