# EgoGeom-Omni（头戴眼镜有无 ToF 通吃基础模型）：当前状态

所属：**避障线**（独立 `EXPLORE`；不改 `egoclear-v0`、`hardware-bringup`、`dtr-r0` 与默认 App）。
更新于 2026-11-24。6 底座选型、`v4` 及 `v4.1` 逐轮明细见 [OVERNIGHT_LOG_20261122.md](OVERNIGHT_LOG_20261122.md)。

## 主线成品：`v4.1` 硅片精确 + 真实 H3 有符号 CNH 旗舰档（`51.18M / 50.75M`）

- **代码**：[egogeom_omni_v41_model.py](egogeom_omni_v41_model.py)、[train_egogeom_omni_v41_l4.py](train_egogeom_omni_v41_l4.py)、[eval_egogeom_v3_zeroshot.py](eval_egogeom_v3_zeroshot.py)、[CODEX_MATH_SPECS_V4.md](CODEX_MATH_SPECS_V4.md)。
- **相对 `v4` 的三大硬核升级（源自 [五题数学实测](math_theory/RESULTS.md) M1 & M5）**：
  1. **M5 芯片级双修原生入图**：有理泰勒 `stable_softplus_htp` + 四行解析 `host_focal`（`<3.3e-16` 误差），根治 Hexagon v75 FP16 `Softplus` 与图内 1 px 焦距差分截断。
  2. **M1 真实 H3 有符号相关 CNH 分支（`Physical1DCNHBranchV41`，`0.43M`）**：统一 20 通道契约 `[log1p(z)*d_m, d_m, sign(h)*log1p(10*|h|), c_m, has_tof]`，46 通道 NCHW（16 有符号 RMS 归一化全波 + 15 去 `bin0` 正尾 + 15 有符号 1D 距离差分）原生消化真实 VL53L8CH H3 `300.28 mm/bin`、13–48% 负值、`ρ=0.25` 跨 bin 相关性、82% `bin0` 串扰与 50–92% 距离丢帧。
  3. **室内/广角尺度精修**：由 `v4` 纯 RGB 温启动（4 Ep P1，`w_fscale=0.95`，50/50 室内外均衡）+ 锁死 `50.75M` 纯 RGB 主干训 5 Ep P2（474 键权重 `max_diff = 0.0`）。
- **S24 Ultra (`R5CX10M8Y8X`) Hexagon v75 FP16 实机验收（`N=50`，[s24u_v41_final_npu_report.json](../../../artifacts.local/work/egogeom-omni-v0/s24u_v41_final_npu_report.json)）**：
  - `Pure-RGB`（`107.7 MB`）：HTP **`16.51 ms`（`60.6 FPS`）** / NetRun **`17.18 ms`**（较 `v4` 快 `1.27–1.33 ms`，突破 60 FPS）。
  - `Unified Omni + Signed-H3`（`109.1 MB`）：HTP **`19.11 ms`（`52.3 FPS`）** / NetRun `20.69 ms`（比 `v4` 快 `0.30 ms`）。
  - **实片保真（`0.15–25 m` vs PyTorch FP32）**：`Pure-RGB` **`0.788%`（`33.97 mm`，较 `v4 3.33%` 降 4.2×）**、`Omni+SignedH3` **`0.601%`（`27.67 mm`）**、`Omni+StdCNH` **`0.591%`（降 3.0×）**、`Omni+Depth` **`0.598%`**、`normal_cos=1.0000`；**芯片级零干扰 `max|Omni(0)-PureRGB| = 0.000000e+00 m`**。

## `v4.1` 严格零样本 & 1,209 帧真实 H3 硬件回放

- **`SANPO-Real ZS40` 真实户外头戴（[v41_history.json](../../../artifacts.local/work/egogeom-omni-v0/v41_history.json)）**：
  - 纯 RGB：**`AbsRel=0.1385 / δ1=83.07% / RMSE=0.683 m / far=0.1810 / p95=18.13 m / med_scale=1.001`**（全线超越 `v4` `0.1514 / 80.74% / 0.748 m`；暗光 `0.18×` `0.1521 / 78.78%`）。
  - `+ Signed-H3`（300.28 mm/bin 负值 + 最高 92% 丢帧 + `-30°` 倾角）：Best `Ep 4` **`0.1148 / δ1=87.85% / RMSE=0.589 m / med_scale=0.996`**（`Ep 5` 达 `0.1113 / 88.42%`）；`+ CNH` **`0.1136 / 87.61%`**；`+ Depth` **`0.1151 / 87.77%`**。
- **`Hypersim Val` / `GlassRecon Val`**：纯 RGB 均显著优于 `v4`（`Hypersim` `δ1 63.36% → 66.85%`、`GlassRecon` `δ1 69.96% → 74.08% / glass 0.1916 → 0.1691`）；`+ Signed-H3` 分别达 **`0.1956 (77.62%)`** 与 **`0.1284 (86.78%, glass=0.1135)`**（`v4` 洁净 Std-CNH 在 GlassRecon 为 `0.0938 / 93.02%`，为 `v4` 与 `v4.1` 各自强项）。
- **4 套跨眼镜纯 RGB 盲测（全部超越 `v4`，[cross_glasses_zeroshot_v41_final.json](../../../artifacts.local/work/egogeom-omni-v0/cross_glasses_zeroshot_v41_final.json)）**：
  - `Meta Aria Gen 2`：`1f MAE 0.934 → 0.910 m`，`+BEV v1 MAE 0.599 m / R@5% 35.0% → 37.6%`，`+BEV hyb AUROC 0.7060`。
  - `UT Austin CODa`：`1f MAE 0.928 → 0.915 m`，`+BEV hyb AUROC 0.7284`。
  - `ARKitScenes val30` 日/暗 `0.18×`：`1f AUROC 0.7160 / 0.7090`（`+0.5 / +1.0 pt`），`+BEV hyb 0.7368 / 0.7283`，`Late@5 7.5%`。
- **1,209 帧真实 ST VL53L8CH H3 实采录像回放（0 NaN，[real_h3_signed_replay_v41.json](../../../artifacts.local/work/egogeom-omni-v0/real_h3_signed_replay_v41.json)）**：
  - 3 段真实极端丢帧走廊（`d_val=0.6–13.7%`、负值 `26.6–35.5%`、`bin0=81.5–86.4%`）：帧间抖动 `std` 较 `v4` 骤降 **6.0–7.9×**（`bright ±6.3 → ±0.8 cm`、`dark ±6.0 → ±1.0 cm`、`restart ±11.4 → ±1.9 cm`）。
  - 0.64 m 真实机柜放物/复原实验：`cabinet_init` 偏差减半（`0.761 → 0.703 m`），放物近移 `-5.7 cm`（抖动减半 `±6.5 → ±3.5 cm`），复原误差 `0.000 m`。

## 历史基线与交付状态

2026-11-24 [V5 九题实测与反例](math_theory/RESULTS_V5.md)支持保留 v4.1 FP16。S24 Ultra W8A8 虽达 4.51/4.96 ms，深度保真和零 ToF 一致性均未达标。[Oxford 跨录像复验](math_theory/m14_conformal_s2_atlas_equivariance/sequence_transfer/RESULTS.md)中，最近有效颜色填充在 9/9 段录像降低深度误差，四地点等权为 1.674→1.513 m；[已审定清路窗](math_theory/m14_conformal_s2_atlas_equivariance/alert_transfer/RESULTS.md)的误报时刻却由 6/22 增至 8/22。保留为深度开发候选。[Aria 逐事件诊断](math_theory/m9_pgl4_frobenius_integrability/aria_alert_support/RESULTS.md)发现 v4.1 在 96/129 个仍来得及提示的危险组合中已有接近真值的历史点簇，原 BEV 最终距离仅在 27/129 个组合内达到相同容差。点簇对应哪个物体尚待确认。

2026-11-25 [Aria 表面与告警复核](math_theory/RESULTS_V5.md#第三轮实测)将上述 96/129 个距离相近组合追溯至实际像素。v4.1 只有 12 个组合的源像素足以支持原最近障碍体素；另一台同步眼镜对同批体素给出支持的有 6 个，来自同一危险段。正确配准的图像运动使距离命中 19→30/129，排除同期训练后仍无真正危险及时告警。原 840 个危险帧用过去航向和过去点云重建，仅 381 个仍判危。后续解释须列出实际表面、参考覆盖和危险及时帧。

## 2026-11-25 三线实证推进结论（硅片 A / 模型 B / 算法 C）

- **Track A（S24U 混合精度突破，[SILICON_PERCHANNEL_PARETO.md](math_theory/m13_fisher_so_c_w8a8_silicon/SILICON_PERCHANNEL_PARETO.md)）**：定位 M13 W8A8 三大根因（逐张量权重与焦距端点缺失、暗光 `0.18×` 在 `stem+stages_0` 丢 2.5 bit、`smooth0` 量化致零 ToF 漂移），在 `v4.1` 零重训下于 S24 Ultra 跑通全新帕累托前沿：
  - **2× 压缩零干扰冠军 `pc_fp16dec_a16`**（W8A16 编码器 + FP16 解码/ToF）：**`57.5 / 59.1 MB`**（缩小 1.86×），HTP **`14.28 / 17.04 ms`**（比 FP16 快 2.2 ms），16 帧 4 域保真 **`1.229% / 1.011%`**（暗光 `0.96% / 0.60%`），**实片零干扰严格 `0.000000 m`**。
  - **超百帧速度档 `pc_stem16_tailfp16_a8`**（仅 `stem+stages_0` INT16 + 末端 FP16）：HTP **`9.16 ms`（109.1 FPS）/ `11.12 ms`（90.0 FPS）**，`55.8 / 57.5 MB`，保真 `2.481% / 2.182%`（纯 RGB 强烈推荐；Omni 零 ToF 漂移 `1.797 m` 需分图部署）。
- **Track B（`v4.2` 反幻觉与 H3 对齐实验，[train_egogeom_omni_v42_l4.py](train_egogeom_omni_v42_l4.py) / [eval_v42_full_audit.py](eval_v42_full_audit.py)）**：`violjjet` L4 跑完 P1（3 Ep 非对称自由空间/平坦地面/单色黑边）+ P2（3 Ep 锁死主干 Signed-H3 对齐，`comp=0.1281`），实机与全量法证给出明确分化：
  - **ToF 融合与 Aria BEV 增强**：`SANPO+CNH 0.1078/88.42%`、`Hypersim+H3 δ1 77.62%→79.42%`、`GlassRecon+H3 δ1 86.78%→88.78% (glass 0.1088)`、`Aria +BEV v1 MAE 0.599→0.487 m / hyb AUROC 0.706→0.728`；1,209 帧真实 H3 回放 `0 NaN`、零干扰 `0.000000 m`、机柜放物响应 `-8.23 cm`；S24U 实测 FP16 `16.06/18.70 ms`（`0.569%/0.345%`）、`pc_fp16dec_a16` `14.34/16.89 ms`（`1.727%/1.138%`，`0.000000 m`）。
  - **诚实负向证据（不晋升替换 `v4.1` 主线）**：Aria 清路 8 体素近距幻觉未解（`519/954 → 531/954 = 55.66%`）、M6 38 源交叉检验 `6/365`（略低于 `v4.1 8/365`）、CODa AUROC 退化（`0.728→0.656`）、Oxford 22 帧 MAE 升至 `2.079 m`（`v4.1 1.918 m`）。干预证实：仅靠针孔单目损失重加权无法突破未知大广角无纹理地面的单目尺度模糊，须依赖双目/多帧视差或主动 ToF。
- **Track C（M6 集成引擎，[egogeom_temporal_bev_v3.py](egogeom_temporal_bev_v3.py)）**：`EgoGeomTemporalBEVV3` 整合原 v2.3 与 M6 四头无量纲几何读出，25,744 帧全量对齐（`max_diff ≤ 1.11e-16`），单线程 CPU 耗时 P50/P95 **`1.02 / 1.69 ms`**（M6 仅 `0.33 / 0.54 ms`）。
- **算力与清理**：`egogeom-v42-l4` 已停机核销（全池 `在线实例=0 / 0.00 CU/hr`），S24U `/data/local/tmp` 已清空。

## 2026-11-25 实机温升、真实 Atom+H3 联调与因果两帧视差突破

- **S24 Ultra 持续满载温升对比（`6 波×300 次 = 1,800 次` 连续 HTP burst，[bench_s24u_thermal_soak.py](bench_s24u_thermal_soak.py) / [thermal_soak_fp16_vs_pc_fp16dec_a16.json](../../../artifacts.local/work/egogeom-omni-v0/s24u_thermal_soak/thermal_soak_fp16_vs_pc_fp16dec_a16.json)）**：
  - `v4.1 FP16`（`109.4 MB`）：冷态 `32.5 °C` 起跑 → 峰值 **`HMX 60.5 °C / HVX 62.8 °C / DDR 61.2 °C`**（温升 **`+28.0~+31.0 °C`**），HTP 均值 `19.11 ms`。
  - **`v4.1 pc_fp16dec_a16`（`59.1 MB`）**：热态 `42.2 °C` 起跑 → 峰值仅 **`HMX 55.0 °C / HVX 54.6 °C / DDR 51.9 °C`**（温升仅 **`+9.7~+12.8 °C`**，DDR 峰值低 **`9.3 °C`**、HVX 低 **`8.2 °C`**），HTP 全程稳在 **`16.82→16.86 ms`**（`+0.2%` 零降频，`59.4 FPS`）。
- **真实同步 Atom S3R RGB + VL53L8CH H3 端到端实机联调（129 帧配对实流，中位 `|Δt|=15.0 ms`，[replay_atom_h3_e2e_s24u.py](replay_atom_h3_e2e_s24u.py) / [atom_h3_e2e_s24u_report.json](../../../artifacts.local/work/egogeom-omni-v0/atom_h3_e2e_s24u/atom_h3_e2e_s24u_report.json)）**：
  - 在 0.64 m 真实机柜上，纯 RGB 误估 `1.835 ± 0.175 m` 且放物反向漂移 `+4.87 cm`；**S24U 59.1 MB `pc_fp16dec_a16` Omni+SignedH3** 立即收敛至 **`0.839 ± 0.071 m`**（抖动降 `2.45×`）、放物正确近移 **`-5.93 cm`**、移动追踪至 `0.634 m`、复原偏差 **`2.67 cm`**，全流 `0 NaN`、实片零干扰 **`0.000000 m`**。
  - 下游 Z-up `EgoGeomTemporalBEVV3` 的 M6 风险对真实放物实现 **4.3× 清晰跃升与归位**（背景 `0.200` → 放物 `0.860` → 移动 `0.552` → 复原 `0.213`），端到端延迟 **`22.2–25.4 ms`**（JPEG 预处理 `4.69 ms` + S24U HTP `16.72 ms` + BEV v3）。可部署产物已封装至 `artifacts.local/work/egogeom-omni-v0/deploy_v41_pc_fp16dec_a16/`。
- **零重训严格因果两帧极线视差集成（[egogeom_temporal_bev_v3.py](egogeom_temporal_bev_v3.py) / [bev_v3_fast_parallax_verification.json](../../../artifacts.local/work/egogeom-omni-v0/bev_v3_fast_parallax_verification.json)）**：
  - 将逐像素多深度极线证伪优化为 `geom_subsample=2` + `3×3` ZNCC + 中心差分梯度并正式并入 `EgoGeomTemporalBEVV3.step_with_parallax()`：单线程 CPU 仅增 **`1.065 ms`**（总耗时 P50 **`2.144 ms`**，提速 15×），在 12,872 帧 38 源留一 M6 交叉检验上保持 **`8 → 32/365`（4× 跃升，0.0 FA/km，4.14 clear/km）**，并将全局单阈值上限推至新高 **`59/365`**（Aria 上限 `0 → 6/143`）。
  - 对照实验（帧级标量尺度校准 `0.79 ms` CPU，[causal_epipolar_scale.py](causal_epipolar_scale.py)）虽将 Aria 清路幻觉由 `519 → 439/954`，但因 M6 读出天然尺度不变，38 源交叉检验停留在 `8/365`，实证只有**局部形变级的极线证伪**才能打穿无 ToF 告警上限。
- **Android 工程落地与 S24 Ultra 实机 App 进程验收（[egogeom_android_device_report.json](../../../artifacts.local/work/egogeom-omni-v0/deploy_v41_pc_fp16dec_a16/egogeom_android_device_report.json)）**：
  - `:core:vision` 落地纯 Kotlin `EgoGeomContracts`、`EgoGeomTemporalBevV3`（含 `stepWithParallax`）、`EgoGeomRuntime`，`:core:vision:testDebugUnitTest` 全部通过（0 失败）；
  - `:app` 落地 C++17 原生 QNN HTP 上下文二进制 JNI（`app/src/dav2Snpe/cpp/egogeom_qnn_jni.cpp`）与 `EgoGeomQnnRuntime.kt`，在真实 `com.linnan.blindassist` APK 进程内于 S24 Ultra 跑通 `EgoGeomQnnRuntimeDeviceTest`（`OK (1 test)`、`1.75 s`）：`59.1 MB` `v4.1 pc_fp16dec_a16` **App 内全往返中位 `24.03 ms`（`min 22.74 ms`）**、`0 NaN`、**严格 `0.000000 m` 零干扰**，真实 Atom+H3 放物使深度近移 **`-20.14 cm`**、Kotlin `EgoGeomTemporalBevV3` M6 风险由 **`0.210 → 0.610`（2.9× 跃升）**。
- **2026-11-25 晚：盲杖盲区 Camera+IMU × 可选 ToF 架构、`G2` 玻璃头合图实片与穿戴演示（[架构与端侧预算](cane_blind_band_probe/CAMERA_IMU_OPT_TOF_ARCHITECTURE.md) · [中高部探测](cane_blind_band_probe/RESULTS.md) · [三区消融](cane_blind_band_probe/STEP1_ABLATION_RESULTS.md) · [悬空与玻璃反事实](cane_blind_band_probe/OVERHANG_GLASS_COUNTERFACTUAL_RESULTS.md) · [无尺度 TTC 与玻璃头](cane_blind_band_probe/SCALEFREE_TTC_AND_GLASS_HEAD_RESULTS.md)）**：
  - **Option A（三通道融合，38 源 / 12,872 帧）**：`fuse_m6par_ang08_gyro`（CPU `3.08 ms`）将主基准推至 `full P5 = 90/365`、**`Shared CF = 63/365`（较基线 `45/365` `+40.0%`，同 `3.10 cw/km`）**、`Device CF = 109/365`、Aria 纯头部悬空 `ovh_head AUROC = 0.6855`（`+6.55 pt`）；100% 零 VIO/零 ToF `cam_imu_novio_fuse`（`1.49 ms`）以 **`0.00 cw/km`** 达成 `CF = 40/365`。
  - **Option B（`G2` 冻结玻璃语义头合图实片，S24U QAIRT 2.47 `N=50` + 16 帧面板，`0 NaN`）**：`G2`（`0.411M`、`810 KB FP16`，`GlassRecon` `[1.5, 4.0) m` 穿透玻璃救回 `82.91%`，未见街景 `670` 帧误报 `0.00–0.25%`）并入 `v4.1 pc_fp16dec_a16`：`Pure-RGB + G2`（`58.94 MB`）HTP **`14.475 ms`（仅 `+0.195 ms`）**、深度误差 `1.2277%`；`Omni + Signed-H3 + G2`（`60.56 MB`）HTP **`17.138 ms`（仅 `+0.098 ms`）**、深度误差 `1.0092%`；`glass_prob_s16` 均值误差 `0.00217`，**芯片级零干扰深度与玻璃双 `0.000000`**。
  - **Android 增补**：`:core:vision` 新增 `EgoGeomWearableRiskEngine.kt` 及 `EgoGeomWearableRiskEngineTest.kt`（Python↔Kotlin `< 2e-4`）；`:app` debug 源集新增独立 `EgoGeomWearableDemoActivity`（不改动默认 `MainActivity` 与 release 构建）。
- **当前路线决策**：**`v4.1 pc_fp16dec_a16 + G2`（`58.94 / 60.56 MB`）+ `EgoGeomQnnRuntime` + `EgoGeomTemporalBevV3` / `EgoGeomWearableRiskEngine` 已完成从数学验证到 Android App 原生实机跑通的全栈闭环**；下一步接实采眼镜或用 debug 演示页开展真实移动与纯悬空/玻璃场景实走验证。
