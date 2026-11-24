# EgoGeom-Omni（头戴眼镜有无 ToF 通吃基础模型）：当前状态

所属：**避障线**（独立 `EXPLORE`；不改 `egoclear-v0`、`hardware-bringup`、`dtr-r0` 与默认 App）。
更新于 2026-11-24。6 底座选型、`v4` 及 `v4.1` 逐轮明细见 [OVERNIGHT_LOG_20261122.md](OVERNIGHT_LOG_20261122.md)。

## 主线成品：`v4.1` 硅片精确 + 真实 H3 有符号 CNH 旗舰档（`51.18M / 50.75M`）

- **代码**：[egogeom_omni_v41_model.py](egogeom_omni_v41_model.py)、[train_egogeom_omni_v41_l4.py](train_egogeom_omni_v41_l4.py)、[eval_egogeom_v3_zeroshot.py](eval_egogeom_v3_zeroshot.py)、[CODEX_MATH_SPECS_V4.md](CODEX_MATH_SPECS_V4.md)。
- **相对 `v4` 的三大硬核升级（源自 [五题数学实测](math_theory/RESULTS.md) M1 & M5）**：
  1. **M5 芯片级双修原生入图**：有理泰勒 `stable_softplus_htp` + 四行解析 `host_focal`（`<3.3e-16` 误差），根治 Hexagon v75 FP16 `Softplus` 与图内 1 px 焦距差分截断。
  2. **M1 真实 H3 有符号相关 CNH 分支（`Physical1DCNHBranchV41`，`0.43M`）**：统一 20 通道契约 `[log1p(z)*d_m, d_m, sign(h)*log1p(10*|h|), c_m, has_tof]`，46 通道 NCHW（16 有符号 RMS 归一化全波 + 15 去 `bin0` 正尾 + 15 有符号 1D 距离差分）原生消化真实 VL53L8CH H3 `300.28 mm/bin`、13–48% 负值、`ρ=0.25` 跨 bin 相关性、82% `bin0` 串扰与 50–92% 距离丢帧。
  3. **室内/广角尺度精修**：由 `v4` 纯 RGB 温启动（4 Ep P1，`w_fscale=0.95`，50/50 室内外均衡）+ 锁死 `50.75M` 纯 RGB 主干训 5 Ep P2（474 键权重 `max_diff = 0.0`）。
- **S24 Ultra (`R5CX10M8Y8X`) Hexagon v75 FP16 实机验收（`N=50`，[s24u_v41_final_npu_report.json](../../artifacts.local/work/egogeom-omni-v0/s24u_v41_final_npu_report.json)）**：
  - `Pure-RGB`（`107.7 MB`）：HTP **`16.51 ms`（`60.6 FPS`）** / NetRun **`17.18 ms`**（较 `v4` 快 `1.27–1.33 ms`，突破 60 FPS）。
  - `Unified Omni + Signed-H3`（`109.1 MB`）：HTP **`19.11 ms`（`52.3 FPS`）** / NetRun `20.69 ms`（比 `v4` 快 `0.30 ms`）。
  - **实片保真（`0.15–25 m` vs PyTorch FP32）**：`Pure-RGB` **`0.788%`（`33.97 mm`，较 `v4 3.33%` 降 4.2×）**、`Omni+SignedH3` **`0.601%`（`27.67 mm`）**、`Omni+StdCNH` **`0.591%`（降 3.0×）**、`Omni+Depth` **`0.598%`**、`normal_cos=1.0000`；**芯片级零干扰 `max|Omni(0)-PureRGB| = 0.000000e+00 m`**。

## `v4.1` 严格零样本 & 1,209 帧真实 H3 硬件回放

- **`SANPO-Real ZS40` 真实户外头戴（[v41_history.json](../../artifacts.local/work/egogeom-omni-v0/v41_history.json)）**：
  - 纯 RGB：**`AbsRel=0.1385 / δ1=83.07% / RMSE=0.683 m / far=0.1810 / p95=18.13 m / med_scale=1.001`**（全线超越 `v4` `0.1514 / 80.74% / 0.748 m`；暗光 `0.18×` `0.1521 / 78.78%`）。
  - `+ Signed-H3`（300.28 mm/bin 负值 + 最高 92% 丢帧 + `-30°` 倾角）：Best `Ep 4` **`0.1148 / δ1=87.85% / RMSE=0.589 m / med_scale=0.996`**（`Ep 5` 达 `0.1113 / 88.42%`）；`+ CNH` **`0.1136 / 87.61%`**；`+ Depth` **`0.1151 / 87.77%`**。
- **`Hypersim Val` / `GlassRecon Val`**：纯 RGB 均显著优于 `v4`（`Hypersim` `δ1 63.36% → 66.85%`、`GlassRecon` `δ1 69.96% → 74.08% / glass 0.1916 → 0.1691`）；`+ Signed-H3` 分别达 **`0.1956 (77.62%)`** 与 **`0.1284 (86.78%, glass=0.1135)`**（`v4` 洁净 Std-CNH 在 GlassRecon 为 `0.0938 / 93.02%`，为 `v4` 与 `v4.1` 各自强项）。
- **4 套跨眼镜纯 RGB 盲测（全部超越 `v4`，[cross_glasses_zeroshot_v41_final.json](../../artifacts.local/work/egogeom-omni-v0/cross_glasses_zeroshot_v41_final.json)）**：
  - `Meta Aria Gen 2`：`1f MAE 0.934 → 0.910 m`，`+BEV v1 MAE 0.599 m / R@5% 35.0% → 37.6%`，`+BEV hyb AUROC 0.7060`。
  - `UT Austin CODa`：`1f MAE 0.928 → 0.915 m`，`+BEV hyb AUROC 0.7284`。
  - `ARKitScenes val30` 日/暗 `0.18×`：`1f AUROC 0.7160 / 0.7090`（`+0.5 / +1.0 pt`），`+BEV hyb 0.7368 / 0.7283`，`Late@5 7.5%`。
- **1,209 帧真实 ST VL53L8CH H3 实采录像回放（0 NaN，[real_h3_signed_replay_v41.json](../../artifacts.local/work/egogeom-omni-v0/real_h3_signed_replay_v41.json)）**：
  - 3 段真实极端丢帧走廊（`d_val=0.6–13.7%`、负值 `26.6–35.5%`、`bin0=81.5–86.4%`）：帧间抖动 `std` 较 `v4` 骤降 **6.0–7.9×**（`bright ±6.3 → ±0.8 cm`、`dark ±6.0 → ±1.0 cm`、`restart ±11.4 → ±1.9 cm`）。
  - 0.64 m 真实机柜放物/复原实验：`cabinet_init` 偏差减半（`0.761 → 0.703 m`），放物近移 `-5.7 cm`（抖动减半 `±6.5 → ±3.5 cm`），复原误差 `0.000 m`。

## 历史基线与交付状态

- **已验档位**：`v4`（`51.04M`，洁净 CNH 玻璃特化 `93.02%`，HTP `17.78/19.41 ms`）、`v3.2`/`v3.1`（`25.54M`）、`v2.1`（`14.46M`，NetRun `11.83 ms`）。
- **算力与清理**：`v4.1` 于 `violjjet` Pro+ L4 完成训练（实耗 **`7.82 CU`**，跨账号 `rclone` 零授权弹窗同步 `gdrive:BlindAssist`），会话已停机核销（`在线实例=0 / 0.00 CU/hr`），S24 Ultra `/data/local/tmp/egogeom_v41_final` 已清空。产物齐备于 Drive `/BlindAssist/runs/egogeom_omni_v41/` 与本地 `artifacts.local/work/egogeom-omni-v0/`（含 `qnn_v75_v41_final/*.bin`）。
- **下一步**：将 `egogeom_omni_v41_omni_v75_fp16.bin`（`19.1 ms`、`0.60%` 实片保真）+ `egogeom_temporal_bev.py` 接入实流，开展真实 Atom RGB + H3 有符号 CNH 现场联调与温升测试。
