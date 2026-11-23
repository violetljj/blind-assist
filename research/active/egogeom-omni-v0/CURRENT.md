# EgoGeom-Omni（头戴眼镜有无 ToF 通吃基础模型）：当前状态

所属：**避障线**（独立 `EXPLORE`；不改 `egoclear-v0`、`hardware-bringup`、`dtr-r0` 与默认 App）。
更新于 2026-11-24。6 个底座的 NPU 选型与逐轮明细见 [OVERNIGHT_LOG_20261122.md](OVERNIGHT_LOG_20261122.md)。

## 主线成品：`v4` 旗舰档（`51.04M / 50.75M`，16.89 亿图 Meta DINOv3）

- **代码**：[egogeom_omni_v4_model.py](egogeom_omni_v4_model.py)、[egogeom_v4_loss.py](egogeom_v4_loss.py)、[train_egogeom_omni_v4_l4.py](train_egogeom_omni_v4_l4.py)。
- **核心设计**：`convnext_small.dinov3_lvd1689m` + 解析旋转不变 Canonical-Focal Disparity（`disp=(350/fx)·softplus≥0`，天空落在 `0` 彻底根治近远互斥）+ Stride-2 细杆干路 + 全 NCHW 零 VTCM 溢出 `Physical1DCNHBranchV4`（`0.28M` 去 `bin0` 尾波归一化 + 1D 距离差分峰检）+ 严格掩码 `MoGe-2-ViT-L` 法线蒸馏。
- **S24 Ultra (`R5CX10M8Y8X`) Hexagon v75 FP16 实机验收（`N=50`，[s24u_v4_final_npu_report.json](../../artifacts.local/work/egogeom-omni-v0/s24u_v4_final_npu_report.json)）**：
  - `Pure-RGB`（`107.72 MB`）：HTP **`17.78 ms`（`56.3 FPS`）** / NetRun **`18.51 ms`（`54.0 FPS`）**。
  - `Unified Omni + CNH`（`108.71 MB`）：HTP **`19.41 ms`（`51.5 FPS`，`<20 ms`）** / NetRun `20.91 ms`。
  - 实片保真（`0.15–25 m` vs PyTorch FP32）：`Omni+Depth 1.70%`、`Omni+CNH 1.78%`、`delta_h` MAE `4.80–5.27 mm`、`normal_cos=1.0000`；**实片零干扰 `max|Omni(0)-PureRGB| = 0.000000e+00 m`**。
- **其余三档已验基线**：`v3.2`（`25.54M`，HTP `15.38/16.61 ms`）、`v3.1`（`25.54M`，近场专精）、`v2.1`（`14.46M`，NetRun `11.83 ms / 84.5 FPS`）。

## `v4` 严格零样本核心指标（13/13 Epoch 连续破纪录）

- **`SANPO-Real ZS40` 真实户外头戴（[v4_history.json](../../artifacts.local/work/egogeom-omni-v0/v4_history.json)）**：
  - 纯 RGB（无 ToF）：**`AbsRel=0.1514 / δ1=80.74% / far=0.1965 / p95=17.27 m / med_scale=0.989`**（超 `v3.2` `+19.6 pt`、`v3.1` `+11.9 pt`；弱光 `0.18×` `δ1=78.38%`）。
  - `+ CNH / + Depth`：**`0.1084 (88.41%)` / `0.1075 (88.50%)`**，`RMSE=0.560 m`，`med_scale=1.001`；弱光 `0.18× + CNH` **`0.1130 (87.00%)`**。
  - **真实 H3 畸变（`-30°` 倾角 + 74% `bin0` + 50% 丢包）**：**`0.1122 / δ1=87.84% / med_scale=0.999`**（较 `v3.1 0.1862 / 80.81%` 提升 `-0.0740 / +7.03 pt`）。
- **`GlassRecon Val` 真实透明玻璃**：纯 RGB `δ1=69.96% / ssi=0.0531`；**`+ CNH` `δ1=93.02% / glass_absrel=0.0772`**；**`+ Depth` `91.56% / 0.0763`**；**真实 H3 畸变 `90.06% / 0.0888`**。
- **`Hypersim Val` 未见室内**：纯 RGB `δ1=63.36% / far=0.1492`；`+ CNH / + Depth` **`80.17% / 81.07%`**。
- **4 套跨眼镜纯 RGB 盲测（[cross_glasses_zeroshot_v4_final.json](../../artifacts.local/work/egogeom-omni-v0/cross_glasses_zeroshot_v4_final.json)）**：
  - `ARKitScenes` 日/夜：`1f AUROC 0.7106 / 0.6989`（超 `v3.2` `+12.3 / +13.6 pt`），`+BEV hyb 0.7360 / 0.7229`，`MAE 0.509 / 0.556 m`。
  - `UT Austin CODa`：`1f MAE 0.928 m`（`2.40×` 锐于 `v3.2`），配 `delta_h_m` 达 **`0.620 m`**，放宽 `4.5 m` 截断后 **`AUROC=0.9164 / R@5%=100.0%`**。
  - `Meta Aria Gen 2`：`1f R@5%=35.0%`、`+BEV hyb R@5%=32.7% / AP=0.7256`（边界：宽敞灰度室内单帧中位深度偏远，`MAE 0.934 m` vs `v3.2 0.506 m`）。

## 交付与下一步

- **2026-11-24　五组数学问题第二轮。** 原 S24 Ultra 上，零重训替换 Softplus 和焦距计算后，16 张跨条件开发图的纯 RGB 设备深度相对 FP32 误差从 **3.522% 降至 0.535%**。归档帧的 HTP 耗时从 17.86 降至 16.56 ms；Omni 加 CNH 同帧误差从 1.782% 降至 0.519%。双修正进入导出候选，默认系统沿用原 v4。Aria 的局部三角测量改善在 CODa 反转；清路计费和远距标签边界限制告警结论。[五题结果](math_theory/RESULTS.md) · [实机测量](math_theory/m5_info_geom_quant_reparam/SILICON_FOLLOWUP.md)

- **资源释放**：`egogeom-omni-v4-l4` 已 `flush_and_unmount` 并 `colab stop`（`4.70 hr ≈ 9.1 CU`）；S24 Ultra 临时目录已清空；成品齐备于 Drive `/BlindAssist/runs/egogeom_omni_v4/` 与本地 `artifacts.local/work/egogeom-omni-v0/`（含 `qnn_v75_v4_final/*.bin`）。
- **下一步**：将 `egogeom_omni_v4_omni_v75_fp16.bin`（`19.4 ms`）+ `egogeom_temporal_bev.py`（`<0.85 ms`）接入独立实流，验证真实 Atom RGB + H3 CNH 现场告警与温升。
