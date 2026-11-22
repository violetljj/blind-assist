# EgoGeom-Omni（100% 自研头戴眼镜有无 ToF 通吃基础模型）：当前状态

所属：**避障线**（独立探索 `EXPLORE`；不改动 `egoclear-v0`、`hardware-bringup`、`dtr-r0` 与默认 App）。
更新：2026-11-22（完整迭代日志、根因解剖与逐轮明细见 [OVERNIGHT_LOG_20261122.md](OVERNIGHT_LOG_20261122.md)）。

## 三档成品定位（均完成 S24 Ultra `R5CX10M8Y8X` Hexagon v75 NPU 真机验通）

- **`v2.1` 极速档（`14.46M / 12.90M`）**：[egogeom_omni_model.py](egogeom_omni_model.py)，S24 Ultra HTP 纯算 `11.26 ms` / NetRun **`11.83 ms`（`84.5 FPS`）**，芯片零干扰 `0.00e+00`。
- **`v3.1` 近场专精档（`25.54M / 23.36M`）**：`SANPO-ZS40 + CNH` `AbsRel=0.1377 / δ1=84.96%`（暗光 `0.1260 / 85.48%`）、`GlassRecon + Depth` `glass_absrel=0.0864`，S24 Ultra HTP `16.51 ms`；边界：户外纯 RGB 远景坍塌（`CODa AUROC 0.6644`）。
- **`v3.2` 全景深旗舰档（`25.54M / 23.36M`，新主线）**：[egogeom_omni_v3_model.py](egogeom_omni_v3_model.py) + [egogeom_losses_and_physics.py](egogeom_losses_and_physics.py) + [train_egogeom_omni_v32_allrange.py](train_egogeom_omni_v32_allrange.py)。根治 `v3` 天空/远景坍塌（精准保留 `SANPO-Synth/TartanAir >=12 m` 远景真值 + `GlassRecon <10 m` 哨兵防护 + `SANPO-Real` 纯天空掩码单边 `8.5 m` 势垒 + 58% 户外平衡采样 + `GroupNorm(8)` ToF/CNH 重对齐）。
  - **S24 Ultra Hexagon v75 FP16 真机实测（`N=50` burst）**：`Pure-RGB`（`59.29 MB`）纯 HTP **`15.38 ms`（`65.0 FPS`）** / NetRun `16.02 ms`；`Unified + CNH`（`64.29 MB`）纯 HTP **`16.61 ms`（`60.2 FPS`）** / NetRun **`17.77 ms`（`56.3 FPS`）**；叠 `<0.85 ms` [egogeom_temporal_bev.py](egogeom_temporal_bev.py) 全链路 `<18.6 ms`。
  - **实片保真与零干扰（真实 Aria Gen 2 帧 vs PyTorch FP32）**：`Omni+CNH` 深度 MAE **`35.86 mm`（`2.25%`）**、`delta_h` MAE `4.65 mm`、`normal_cos = 0.999972`；**实片零干扰 `max|Omni(0)-PureRGB| = 0.000000e+00 m`**。

## `v3.2` 终版严格零样本核心突破

1. **4 套跨眼镜纯 RGB 零样本全线刷新全历史纪录（无 ToF，[cross_glasses_zeroshot_v32_final.json](../../artifacts.local/work/egogeom-omni-v0/cross_glasses_zeroshot_v32_final.json)）**：
   - **`UT Austin CODa` 户外校园（2,819/34 帧）**：单帧 **`AUROC = 0.7722`（较 `v3.1 0.6644` 暴涨 `+10.78 pt`，单帧直接反超 `v2.1 + BEV 0.7651`！）**，`MAE = 2.227 m`（比 `v2.1` 锐利 `32 cm`）；+BEV **`R@5%FPR = 67.7%`（+11.8 pt 超 `v2.1`）**。
   - **`Meta Aria Gen 2` 超广角灰度眼镜（1,794/840 帧）**：单帧 **`AUROC 0.7450 / AP 0.7368 / R@5% 31.4% / MAE 0.506 m`** 全项创历史新高且全面压过 `DAV2-Metric-S`（`0.7229 / 0.7023 / 24.4% / 1.271 m`）；+BEV **`MAE = 0.285 m`（`4.46×` 锐于 DAV2）**。
   - **`ARKit val30` 日间 / `0.18×` 极暗光（467/388 帧）**：单帧闭环漏晚报双双压至 **`3.2% / 3.2%`**（比 `DAV2-S 12.9%/10.8%` 降 `3.4–4.0×`），+BEV 锁死 **`1.1% / 1.1%`**（`R@5% = 64.4% / 60.3%`）。
2. **三套深度与 ToF/CNH 基准（`0.30–4.0 m` 近场 + `4–12 m` 远场，[v32_history.json](../../artifacts.local/work/egogeom-omni-v0/v32_history.json)）**：
   - `SANPO-ZS40` 纯 RGB `p95_m` 拉升至 **`6.37 m`**；+ CNH `AbsRel=0.1710 / δ1=80.8% / far_absrel=0.2269 / p95=8.68 m / sc=0.996`（Ep 3 `+Depth` `AbsRel=0.1589 / δ1=81.8%`）；`SANPO 暗光 + CNH` `AbsRel=0.1693 / δ1=80.84%`。
   - `GlassRecon Val` + CNH（Ep 3）：**`near_absrel=0.1176 / RMSE=0.614 m / δ1=87.74% / glass_absrel=0.1071`**（`+Depth` `glass_absrel=0.0909`）；`Hypersim Val` 纯 RGB 玻璃误差从 `v3.1 0.6388` 降至 **`0.4272`**（`-33.1%`），+ CNH 玻璃误差 **`0.2231`**（Ep 1）。

## 已落盘成品与资源释放

- **算力与手机清理**：`egogeom-omni-v32-l4` 已 `colab stop` 释放（`Active assignments: 0`, `Usage rate: 0.00/hr`, 余额 `125.79 CU`）；S24 Ultra `/data/local/tmp/egogeom_v32_final` 已清空。
- **主线产物**（Drive `/BlindAssist/runs/egogeom_omni_v32/` + 本地 `artifacts.local/work/egogeom-omni-v0/`）：`egogeom_omni_v32_unified_best.pt`、`egogeom_omni_v32_best_omni_fp32.onnx`、`egogeom_omni_v32_best_pure_rgb_fp32.onnx`、`v32_history.json`、`cross_glasses_zeroshot_v32_final.json`、`s24u_v32_final_npu_report.json`，及 S24 Ultra 开箱即用 QNN v75 二进制 `qnn_v75_v32_final/egogeom_omni_v32_{omni,pure_rgb}_v75_fp16.bin`。

## 下一步决策

- **直接上镜实流验证（推荐）**：将 `egogeom_omni_v32_omni_v75_fp16.bin`（`16.6 ms`）+ `egogeom_temporal_bev.py`（`<0.85 ms`）接到独立试验流，实测真实 Atom RGB + H3 `8×8×16` CNH 端到端告警与长期功耗温升。
