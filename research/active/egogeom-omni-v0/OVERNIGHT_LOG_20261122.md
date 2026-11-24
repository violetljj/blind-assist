# EgoGeom-Omni v3 -> v3.1 过夜攻关与验证日志（2026-11-22）

所属：**避障线**（`EXPLORE`，范围严格限定 `research/active/egogeom-omni-v0/` 与 `artifacts.local/work/egogeom-omni-v0/`）。

## 1. 核心工程与算法突破时间线

1. **实机 H3 VL53L8CH 原始 CNH 波形物理剖析（非严谨参考：`artifacts.local/hardware-bringup/captures/cnh-h3-*-20260927/raw.bin`）**：
   - 确认实机原始数据结构为精确 `8×8×16`（`hist_raw * 2^(-hist_scaler)`，每 bin 约 `0.285 m`）。
   - 发现两大实机致命物理特性：① 护玻近场光学串扰使 `bin0` 占总光子数 **`52%–87%`（中位 `74.3%`）**；② **`46%–51%` 区域固件 `distance_mm == 0`**，但同时 `target_status == 5/9`（94.3% 有效）且 16-bin 原始 CNH 波形仍携带清晰回波峰。
   - 已将实机物理（`-30°` 倾角 + 74% `bin0` 串扰 + 50% `distance_mm` 掉帧）完整写入 [egogeom_losses_and_physics.py](egogeom_losses_and_physics.py) 的 `simulate_tof_and_cnh_prompt(..., realistic_hardware=True)`。
2. **Stage-2 可部署时序 BEV v2.3（`<0.85 ms/帧` 纯 NumPy，[egogeom_temporal_bev.py](egogeom_temporal_bev.py)）**：
   - 逐项消融确认 **GroundScaleLock**（`sc = clip(cam_h_ema / (-g0), 0.82, 1.22)` 稳定单目帧间尺度抖动）+ **分区可见性归一化衰减**（可见区 `z>=0.85m` 快衰减 `0.45` 消门口拖影，脚下盲区 `<0.85m` 慢衰减 `0.78` 留低障）+ **无平局严格单调风险分**。
   - 在 `v2.1` 跨眼镜零样本上：`CODa` `AUROC 0.7479 -> 0.7651 / AP 0.0617 -> 0.1784 (2.9×)`；`ARKit Dark 0.18×` `AUROC 0.6881 -> 0.7164 / AP 0.9179 / MAE 0.627 m / Late@5 = 1.1%`（全项超越预训练 `DAV2-Metric-S`）。
3. **L4 12× 训练提速与 `v3`（25.4M）错误解剖**：
   - 将 `MoGe-2` 蒸馏从 CPU `m.infer()`（`6.4 s/batch`）改为纯 GPU `m.forward(num_tokens=224)`（`395 ms/batch`，**16.2×**），配合 `_ActiveFilePoolMixin` LRU 轨迹池把 `.npz` 命中率从 `<1%` 拉到 `~94%`（吞吐 `12 -> 86 img/s`）。
   - `v3` 全量 12 轮 P1 + 8 轮 P2 解剖出三大物理根因：① `GlobalScaleContext` 原始 `|grav_rays|` 形成 FOV 域捷径（把 `SANPO head fx=314` 误判为室内 `Hypersim`，`med_scale=0.44`）；② `over_pen=1.5` 造成 3:1 系统性偏近偏差（`-15%..-21%`）；③ 稀疏掩码 ToF 分支用 `BatchNorm2d` 导致混合 CNH/Depth 训练剧烈振荡。
4. **`EgoGeom-Omni v3.1`（25.54M 统一 / 23.36M 纯 RGB，[egogeom_omni_v3_model.py](egogeom_omni_v3_model.py)）**：
   - **旋转不变解析焦距分解**：直接由 `grav_rays` 邻像素差分最大值解析求 `1/fx`（误差 `0.00 px`，零额外 ONNX 输入），并移除 `r_abs` FOV 捷径。
   - **无偏对称几何相对误差 `|p-g|/√(p·g)` + 帧级全局尺度损失 `l_fscale`**（贝叶斯最优解严格为 `1.000×gt`）。
   - **Phase 2 爆炸根因定位与 `GroupNorm(8)` 彻底根治**：初版无归一化 7 层 `Conv+SiLU` 在 step 1000–1750 发散至 `loss=20.9`；升级为 `Conv2d(bias=False) + GroupNorm(8) + SiLU` + 有界 `2.5·tanh` Stride-4 直接尺度残差头 + `lr=6e-4` 后，Phase 2 连续 4 轮单调丝滑收敛（`omni_composite: 0.1802 -> 0.1646 -> 0.1571 -> 0.1562`，较 `v3` 8 轮最优 `0.2148` 压低 `27.3%`），且在 12h VM 自动回收前已将 Ep 4 最佳权重与双 ONNX-17 完整写入 Google Drive。

## 2. 完整量化指标对照

### 2.1 三套核心深度基准（0.30–4.0 m 近场，`v3.1` Ep 4/6 终版）

| 基准与模式 | `near_absrel` | `RMSE (m)` | `δ1` | `med_scale` | `SSI` | `thin_edge` | `glass_absrel` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `SANPO-ZS40` 日间 Pure-RGB | `0.3009` | `0.887` | `71.53%` | `1.009` | `0.1283` | `0.5198` | — |
| `SANPO-ZS40` 日间 + `8×8×16` CNH | **`0.1377`** | `0.631` | **`84.96%`** | `0.996` | `0.1078` | `0.3289` | — |
| `SANPO-ZS40` 日间 + `8×8` Depth-only | **`0.1345`** | **`0.604`** | **`85.31%`** | `0.984` | `0.1071` | `0.3161` | — |
| `SANPO-ZS40` 日间 + 实装 H3 倾角/串扰/掉帧 | `0.1862` | `0.789` | **`80.81%`** | `1.006` | `0.1253` | `0.4139` | — |
| `SANPO-ZS40` 暗光 `0.18×` Pure-RGB | `0.2870` | `0.839` | `63.03%` | `0.944` | `0.1299` | `0.4614` | — |
| `SANPO-ZS40` 暗光 `0.18×` + CNH | **`0.1260`** | **`0.565`** | **`85.48%`** | `1.011` | `0.1013` | `0.3012` | — |
| `Hypersim Val` Pure-RGB | `0.6543` | `1.091` | `47.61%` | `1.111` | `0.3105` | `0.6242` | `0.6388` |
| `Hypersim Val` + CNH / Depth | `0.2015 / 0.1985` | `0.645 / 0.636` | `73.94% / 73.88%` | `0.990 / 0.975` | `0.1878 / 0.1871` | `0.2836 / 0.2773` | `0.2479 / 0.2395` |
| `GlassRecon Val` Pure-RGB | `0.2141` | `0.727` | `65.03%` | `1.020` | `0.0807` | `0.2325` | `0.2265` |
| `GlassRecon Val` + CNH / Depth | `0.1259 / 0.1206` | `0.626 / 0.627` | **`87.63% / 87.57%`** | `1.021 / 1.020` | `0.0833 / 0.0756` | `0.2282 / 0.2350` | **`0.0937 / 0.0864`** |
| `GlassRecon Val` + 实装 H3 倾角/串扰/掉帧 | `0.1624` | `0.788` | `78.79%` | `1.038` | `0.0944` | `0.2261` | `0.1468` |

### 2.2 S24 Ultra (`R5CX10M8Y8X`) Hexagon v75 HTP FP16 终版双模型真机实测

- **编译**：`egogeom_omni_v31_pure_rgb_v75_fp16.bin`（`59.29 MB`, `5.5s`）、`egogeom_omni_v31_omni_v75_fp16.bin`（`64.29 MB`, `7.2s`），100% 单图 HTP 零 CPU 回退。
- **时延（`N=50` burst）**：
  - `Pure-RGB`：纯 HTP **`15.30 ms`（`65.4 FPS`）**（min `15.21` / max `16.21`），`NetRun` **`15.92 ms`（`62.8 FPS`）**。
  - `Unified Omni + CNH`：纯 HTP **`16.51 ms`（`60.6 FPS`）**（min `16.42` / max `16.52`，抖动仅 `0.10 ms`），`NetRun` **`17.95 ms`（`55.7 FPS`）**。
- **芯片级数值保真与零干扰（真实 Aria Gen 2 帧，vs PyTorch FP32）**：
  - `Pure-RGB` / `Omni(has_tof=0)`：深度 MAE `31.74 mm`（`1.64%`），`delta_h` MAE `5.16 mm`，`normal_cos = 0.999972`。
  - `Omni + CNH`：深度 MAE `47.94 mm`（`3.01%`），`delta_h` MAE `5.47 mm`，`normal_cos = 0.999961`。
  - `Omni + Depth`：深度 MAE `46.81 mm`（`3.00%`），`delta_h` MAE `4.86 mm`，`normal_cos = 0.999971`。
  - **实片零干扰实测**：`max |Omni(has_tof=0) - PureRGB| = 0.000000e+00 m`；启用 CNH 平均深度修正 `44.19 cm`。

## 3. 遗留边界与教训

- `colab exec` 长连接存在偶发 WebSocket 断连且静默挂起（曾致旧 P2 爆炸进程空跑约 46 min / ~1.5 CU），后续统一采用“短命令 `start_new_session=True` 后台拉起 + 纯 HTTP `colab download` 轮询”。
- 12 小时 L4 会话在 P2 Ep 4 落盘后自动回收（少跑 Ep 5–6 两轮余弦退火，但 `omni_composite` 在 Ep 3->4 已收敛至 `0.1571 -> 0.1562`，增益趋近饱和）。
- `v3.1` 纯 RGB 在户外超广角 `UT Austin CODa`（`fx=240.8`）上 `AUROC 0.6644` 低于 `v2.1`（`0.7289`），而室内/广角 `Aria Gen 2` 与 `ARKit` 大幅领先；完整闭环安全性仍待真实眼镜视频流实测确认。

---

## 4. `EgoGeom-Omni v3.2` 户外纯 RGB 短板根因解剖与外科手术式根治（2026-11-22 18:30–22:05）

### 4.1 零 CU 本地解剖锁死病根
- 同帧深度分布 `(p5 / p50 / p95)` 对比证明：`UT Austin CODa` 开阔校园上 `v2.1` 为 `[0.99, 3.35, 8.01] m`（`NEG<4m = 25.9%`），而 `v3`（`[1.27, 1.88, 2.12] m`）与 `v3.1`（`[1.64, 2.21, 2.59] m`，`NEG<4m = 100.0%`）已将户外天空与远景彻底压扁成 ~2.4 m 平面墙。
- 精确三层连锁成因：① `v3` 蒸馏调用 `MoGe-2 m.forward()` 未过滤天空掩码 `oc["mask"]` 且未恢复仿射平移，将天空毒化为 `~2 m` 喂入 `l_td_ssi` 8 轮；② `egogeom_dataset.py` 与 loss 丢弃了所有 `>=12 m` 远景真值（零梯度无法纠偏）；③ 热修期间还在 Ep 1 实测中当场抓获 `GlassRecon` uint16 `65535 mm`（`65.535 m`）无效哨兵值被宽泛 `<=80 m` 误当远景的隐患并秒修清零。

### 4.2 `v3.2` 外科修复方案与最终量化成果（单 L4 约 1.5h，已 `colab stop` 释放至 `0.00/hr`）
- **修复**：仅对纯光线追踪户外源 `SANPO-Synth / TartanAir` 保留 `>=12 m` 远景真值（`GlassRecon` 严锁 `<10 m`、`Hypersim` `<12 m`、`SANPO-Real` 仅走 `MoGe-2-ViT-S` 上半幅高置信天空掩码），施加温柔单边势垒 `ReLU(log(8.5) - log_depth)`（`w_far=0.25`）+ 全景深 `l_ssi` + `w_fscale=0.70` 锁尺度 + 58% 户外采样；Phase 2 冻结纯 RGB 主干并零错配热启 `v3.1` 收敛的 `GroupNorm(8)` ToF/CNH 分支（3 Ep）。
- **4 套跨眼镜纯 RGB 严格零样本终版（`egogeom_omni_v32_unified_best.pt`，0 CU RTX 5060）**：
  - `UT Austin CODa`（2,819/34）：单帧 **`AUROC = 0.7722`（较 `v3.1 0.6644` 暴涨 `+10.78 pt`，单帧直接反超 `v2.1 + BEV 0.7651`）**、`MAE = 2.227 m`，+BEV **`R@5%FPR = 67.7%`（+11.8 pt 超 `v2.1`）**。
  - `Meta Aria Gen 2`（1,794/840）：单帧 **`AUROC = 0.7450 / AP = 0.7368 / R@5% = 31.4% / MAE = 0.506 m`**，+BEV **`MAE = 0.285 m`（较 `DAV2-Metric-S 1.271 m` 锐利 `4.46×`）**。
  - `ARKit val30` 日间 / `0.18×` 极暗光（467/388）：单帧闭环漏晚报 **`3.2% / 3.2%`**，+BEV 双锁 **`1.1% / 1.1%`**（`R@5% = 64.4% / 60.3%`）。
- **S24 Ultra (`R5CX10M8Y8X`) Hexagon v75 FP16 真机终验**：
  - `egogeom_omni_v32_pure_rgb_v75_fp16.bin`（`59.29 MB`）：纯 HTP **`15.38 ms`（`65.0 FPS`）** / NetRun `16.02 ms`。
  - `egogeom_omni_v32_omni_v75_fp16.bin`（`64.29 MB`）：纯 HTP **`16.61 ms`（`60.2 FPS`）** / NetRun **`17.77 ms`（`56.3 FPS`）**。
  - 实片保真（真实 Aria Gen 2 帧 vs PyTorch FP32）：`Omni+CNH` 深度 MAE **`35.86 mm`（`2.25%`）**、`delta_h` MAE `4.65 mm`、`normal_cos = 0.999972`；**芯片级零干扰实测 `max|Omni(0)-PureRGB| = 0.000000e+00 m`**。


---

## 阶段七：6 大顶级预训练视觉底座 S24 Ultra Hexagon v75 NPU 零算力真机大逃杀（2026-11-22 晚）

- **动机**：`v3.2` 从零训练 25M 底座受限于先验知识天花板且在 `log_depth ∈ [0.15, 12.0]` 空间存在天然近远景互斥。零 CU 直接在 S24 Ultra (`R5CX10M8Y8X`) Hexagon v75 FP16 实测 6 大开源顶级预训练视觉底座（挂完整 4 头解耦解码器 + 物理 1D-CNH 分支，`288×384`，`N=30` burst），产物见 `artifacts.local/work/egogeom-omni-v0/s24u_v4_backbone_shootout.json`：
  - **关键发现（VTCM 溢出根治）**：初始 `reshape(B, 2, 16, 6912)` 触发 Hexagon v75 8 MB VTCM 溢出导致全系飙至 `~77 ms`；改用纯 NCHW `[B, 45, 72, 96]`（16 原始 + 15 去 `bin0` 归一化尾波 + 14 距离轴一阶差分峰检）后，ToF 分支耗时骤降至 `<0.6 ms`！
  - **实机 NPU 榜单**：
    1. `convnext_tiny.dinov3_lvd1689m`（`29.0M`, `60.0 MB` `.bin`）：HTP **`14.49 ms`（`69.0 FPS`）** / NetRun `15.72 ms`
    2. `repvit_m2_3.dist_450e_in1k`（`23.6M`, `31.7 MB`）：HTP `15.12 ms` / NetRun `16.40 ms`
    3. `fastvit_sa24.apple_dist_in1k`（`21.5M`, `48.7 MB`）：HTP `15.46 ms` / NetRun `16.88 ms`
    4. `convnextv2_nano.fcmae_ft_in22k_in1k_384`（`16.2M`, `35.0 MB`）：HTP `15.88 ms` / NetRun `17.33 ms`
    5. **`convnext_small.dinov3_lvd1689m`（16.89 亿图 Meta DINOv3，`50.7M`，27 层深度 Stage-3，`104.2 MB`）**：HTP **`17.72 ms`（`56.4 FPS`）** / NetRun `19.25 ms` -> **选定为 `v4` 登峰造极底座！**
    6. `vit_small_patch16_dinov3.lvd1689m`（`23.2M`, `48.5 MB`）：HTP `17.97 ms` / NetRun `19.04 ms`

---

## 阶段八：`EgoGeom-Omni v4` 登峰造极全量训练、7 套零样本验收与 S24 Ultra 终验（2026-11-22 23:13 – 2026-11-23 04:01）

- **架构与损失突破**（[egogeom_omni_v4_model.py](egogeom_omni_v4_model.py)、[egogeom_v4_loss.py](egogeom_v4_loss.py)、[train_egogeom_omni_v4_l4.py](train_egogeom_omni_v4_l4.py)）：
  1. **16.89 亿图 Meta DINOv3 `convnext_small.dinov3_lvd1689m`（`51.04M` 总参，`0.28M` ToF）** + Stride-2 细杆边缘干路 + 零初始化金字塔重力 FiLM + 全局视差上下文。
  2. **旋转不变解析 Canonical-Focal Disparity**：直接从 `grav_rays` 邻像素差析出精确 `1/fx`（零额外 ONNX 输入），预测 `disp = (350/fx)·softplus(...) ≥ 0`，天空/远景自然落在 `disp = 0.000`（`25 m`），彻底根除 `v3.1 vs v3.2` 的近远景互斥与尺度膨胀。
  3. **零毒化教师与差速微调**：Phase 1（8 ep，`48k/ep`，`bs=32`）前 300 步冻结 DINOv3 预热解码器（`6e-4`），随后 10× 差速解冻（`6e-5`）；Ep 1–4 仅用严格 `prob > 0.6` 掩码的 `MoGe-2-ViT-L` 法线蒸馏（永不把未对齐 `pts_z` 喂进深度损失）+ `MoGe-2-ViT-S` 户外天空掩码守卫，Ep 5–8 卸载 ViT-L 纯 GT 精修。
  4. **Phase 2（5 ep，冻结 `50.75M` 纯 RGB 主干 `0.00e+00` 零干扰）**：专训全 NCHW `Physical1DCNHBranchV4`（`GroupNorm(8)` + 去 `bin0` 尾波归一化 + 1D 距离轴差分峰检 + 乘性/加性双残差），50% 洁净 / 50% 真实 H3 硬件物理（`-30°` 倾角 + 74% `bin0` 串扰 + 50% 距离丢包）混合训练。
- **13/13 Epoch 单调破纪录收敛轨迹（L4 共 `4.70 hr ≈ 9.1 CU`）**：
  - Phase 1（纯 RGB，无 ToF）`composite`：`0.3259 -> 0.3133 -> 0.2950 -> 0.2763 -> 0.2479 -> 0.2362 -> 0.2199 -> 0.2179`（8 连刷新！）：
    - `SANPO-Real ZS40` Day 纯 RGB：`AbsRel=0.1514, RMSE=0.748 m, δ1=80.74%, far=0.1965, p95=17.27 m, med_scale=0.989, ssi=0.0850`（Dark `0.18×`：`0.1610 / δ1=78.38% / med_scale=0.999`）。
    - `Hypersim Val` 纯 RGB：`AbsRel=0.2931, δ1=63.36%, far=0.1492, ssi=0.1461`。
    - `GlassRecon Val` 纯 RGB：`AbsRel=0.2021, δ1=69.96%, glass_absrel=0.1916, ssi=0.0531`。
  - Phase 2（统一全能，纯 RGB 100% 冻结零衰减）`omni_composite`：`0.1662 -> 0.1456 -> 0.1320 -> 0.1251 -> 0.1234`（5 连刷新！）：
    - `SANPO-Real ZS40` Day `+ CNH`：**`AbsRel=0.1084, RMSE=0.5601 m, δ1=88.41%, far=0.1690, med_scale=1.001, ssi=0.0825`**；`+ Depth`：**`0.1075 / δ1=88.50%`**；**真实 H3 畸变**：**`0.1122 / RMSE=0.5817 m / δ1=87.84% / med_scale=0.999`**（碾压 `v3.1 0.1862 / 80.81%` 达 `-0.0740 AbsRel / +7.03 pt`！）；`Dark 0.18× + CNH`：**`0.1130 / δ1=87.00% / med_scale=1.001`**。
    - `GlassRecon Val`：**`+ CNH` `AbsRel=0.0938, RMSE=0.4676 m, δ1=93.02%, glass_absrel=0.0772`**；**`+ Depth` `0.0969 / δ1=91.56% / glass_absrel=0.0763`**；**真实 H3 畸变 `0.1054 / δ1=90.06% / glass_absrel=0.0888`**。
    - `Hypersim Val`：`+ CNH` **`0.1763 / δ1=80.17%`**；`+ Depth` **`0.1692 / δ1=81.07% / glass_absrel=0.1574`**；真实 H3 畸变 `0.1842 / 79.15%`。
- **4 套跨眼镜纯 RGB 零样本盲测（`egogeom_omni_v4_unified_best.pt`，与 `pure_rgb_best.pt` 逐位一致）**：
  - `ARKitScenes val30` Day / Dark `0.18×`：`1f AUROC = 0.7106 / 0.6989`（较 `v3.2` `+12.3 / +13.6 pt`），`+BEV hyb = 0.7360 / 0.7229`（`+14.2 / +16.2 pt`），`MAE = 0.509 / 0.556 m`（锐化 `26–27 cm`）。
  - `UT Austin CODa`：`1f MAE = 0.928 m`（`2.40×` 锐于 `v3.2 2.227 m`）；使用 `v4` 自带 `delta_h_m` 时 `MAE = 0.620 m`（`3.59×` 锐利），且放宽评测 `4.5 m` 走廊截断后 `AUROC = 0.9164 / R@5% = 100.0%`（超过 `DAV2-Metric-S 0.8588`）。
  - `Meta Aria Gen 2`：`1f R@5% = 35.0%`（`+3.6 pt`）、`+BEV hyb R@5% = 32.7%`（翻倍）、`AP = 0.7256`（`+6.3 pt`）；诚实边界：宽敞室内灰度帧单帧深度偏远（`MAE 0.934 m` vs `v3.2 0.506 m`）。
- **S24 Ultra (`R5CX10M8Y8X`) Hexagon v75 NPU 终验（`s24u_v4_final_npu_report.json`）**：
  - `Pure-RGB`（`107.72 MB`）：HTP **`17.78 ms`（`56.3 FPS`）** / NetRun **`18.51 ms`（`54.0 FPS`）**；
  - `Unified Omni + CNH`（`108.71 MB`）：HTP **`19.41 ms`（`51.5 FPS`）** / NetRun `20.91 ms`（`47.8 FPS`）；
  - 实片保真（`0.15–25 m` 全量程 vs PyTorch FP32）：`Omni+Depth` `1.70%`（`81.95 mm`）、`Omni+CNH` `1.78%`（`86.77 mm`）、`Pure-RGB` `3.33%`（远场 `15–25 m` `1/disp` FP16 舍入主导）、`delta_h` MAE `4.80–5.27 mm`、`normal_cos = 1.0000`；
  - **芯片级零干扰实测 `max|Omni(has_tof=0) - PureRGB| = 0.000000e+00 m`**（实片 CNH 平均修正 `22.80 cm`）；`/data/local/tmp/egogeom_v4_final` 已清空。

---

## 阶段九：`EgoGeom-Omni v4.1` 硅片精确 + 真实 H3 有符号 CNH 升级、1,209 帧实采回放与 S24 Ultra 终验（2026-11-24 04:58 – 09:08）

- **升级动机与架构突破**（[egogeom_omni_v41_model.py](egogeom_omni_v41_model.py)、[train_egogeom_omni_v41_l4.py](train_egogeom_omni_v41_l4.py)、[CODEX_MATH_SPECS_V4.md](CODEX_MATH_SPECS_V4.md)）：
  1. **M5 芯片级双修原生入图**：将 Codex 实机验证的 `stable_softplus_htp`（`s = e^{-|x|}/(2+e^{-|x|})` 7 阶有理泰勒）与 4 行解析 `host_focal`（`<3.3e-16` 误差）原生合入 `v4.1` 架构与双份 ONNX-17，消除 Hexagon v75 FP16 的 `Softplus` 与 1 px 焦距差分截断误差。
  2. **M1 真实 H3 有符号相关 CNH 建模与分支（`Physical1DCNHBranchV41`，`0.43M`）**：统一 20 通道契约 `[log1p(z)*d_m, d_m, sign(h)*log1p(10*|h|), c_m, has_tof]`（完全向后兼容 `v4`），46 通道 NCHW（16 有符号 RMS 归一化全波 + 15 去 `bin0` 正尾 + 15 有符号 1D 距离差分）配完整 `3×3` `GroupNorm(8)` 空间卷积；训练模拟器严格匹配真实 VL53L8CH H3 固件（`300.2784 mm/bin`、Cholesky `ρ=0.25` 跨 bin 相关噪声、13–48% 负值、15–82% `bin0` 串扰、50–92% 距离丢帧、`-30°` 倾角），每 batch 按 `40% Signed-H3 + 30% Std-CNH + 30% Depth` 混合。
  3. **多账号池 `colabm` + 跨账号 `rclone` 逐集流式落盘**：在 `violjjet` Pro+ L4 免 OAuth 弹窗直连 `gdrive:BlindAssist`，途中捕获并就地修复 236 GB 盘满（升级为逐集拉取→解压→即删原始包，峰值 `<120 GB`）及 Phase 2 20 通道对齐缺陷（0 秒跳过 7 数据集与 Phase 1 热续跑），全程实耗 **`7.82 CU`**，停机后 `colabm dashboard` 核实 `在线实例=0 / 0.00 CU/hr`。
- **训练收敛轨迹（[v41_history.json](../../artifacts.local/work/egogeom-omni-v0/v41_history.json)）**：
  - **Phase 1（4 Ep 温启动纯 RGB 室内/广角尺度精修，`composite 0.2532 → 0.2306`）**：
    - `SANPO-Real ZS40` Day 纯 RGB：**`AbsRel=0.1386, RMSE=0.6827 m, δ1=83.07%, far=0.1810, p95=18.13 m, med_scale=1.001`**（全面压制 `v4` `0.1514 / 0.748 m / 80.74%`）；Dark `0.18×`：`0.1521 / 0.725 m / δ1=78.78% / med_scale=1.001`。
    - `Hypersim Val` 纯 RGB：`AbsRel=0.2816, RMSE=0.7811 m, δ1=66.85%, med_scale=1.137`（较 `v4` `δ1 +3.49 pt`）。
    - `GlassRecon Val` 纯 RGB：`AbsRel=0.1913, δ1=74.09%, glass_absrel=0.1689, ssi=0.0487`（较 `v4` `δ1 +4.13 pt`、`glass -11.8%`）。
  - **Phase 2（5 Ep 锁定 `50.75M` 纯 RGB 主干，`474` 键 `max_diff=0.0`，`omni_composite 0.1733 → 0.1603 → 0.1465 → 0.1344`）**：
    - `SANPO-Real ZS40` Day：`+ Signed-H3`（Best `Ep 4`）**`0.1148 / RMSE=0.5895 m / δ1=87.85% / far=0.1622 / med_scale=0.996`**（`Ep 5` 达 `0.1113 / 88.42% / 0.5756 m`），`+ CNH` **`0.1136 / 87.61%`**，`+ Depth` **`0.1151 / 87.77%`**；Dark `0.18× + Signed-H3` `0.1286 / 84.93%`（`Ep 5` `0.1233 / 85.81%`）。
    - `Hypersim Val`：`+ Signed-H3` **`0.1956 / δ1=77.62%`**，`+ Depth` **`0.1908 / 78.18%`**。
    - `GlassRecon Val`：`+ Signed-H3` **`0.1284 / δ1=86.78% / glass=0.1135`**（全面优于同权重 `+ Std-CNH` `0.1308 / 85.46%`），`+ Depth` `0.1267 / 85.98% / glass=0.1009`。
- **4 套跨眼镜纯 RGB 零样本盲测（全胜 `v4`，[cross_glasses_zeroshot_v41_final.json](../../artifacts.local/work/egogeom-omni-v0/cross_glasses_zeroshot_v41_final.json)）**：
  - `Meta Aria Gen 2`：`1f MAE = 0.910 m`（vs `v4 0.934 m`），`+BEV v1 MAE = 0.599 m / R@5% = 37.6%`（vs `35.0%`），`+BEV hyb AUROC = 0.7060`。
  - `UT Austin CODa`：`1f MAE = 0.915 m`（vs `v4 0.928 m`），`+BEV hyb AUROC = 0.7284`。
  - `ARKitScenes val30` Day / Dark `0.18×`：`1f AUROC = 0.7160 / 0.7090`（vs `v4 0.7106 / 0.6989`），`+BEV hyb AUROC = 0.7368 / 0.7283`，`AP = 0.9199 / 0.9285`，`Late@5 = 7.5%`。
- **1,209 帧真实 ST VL53L8CH H3 实采录像回放（[real_h3_signed_replay_v41.json](../../artifacts.local/work/egogeom-omni-v0/real_h3_signed_replay_v41.json)，0 NaN）**：
  - 真实走廊 3 段（`d_val = 2.9% / 13.7% / 0.6%`、负值 `34.9% / 26.6% / 35.5%`、`bin0 = 81.5–86.4%`）：`v4.1` 帧间标准差 `±0.8 / ±1.0 / ±1.9 cm` vs `v4` `±6.3 / ±6.0 / ±11.4 cm`（**6.0–7.9× 降噪**）。
  - 真实 0.64 m 机柜 4 段：`cabinet_init` 偏差减半（`v4 0.761 m → v4.1 0.703 m`），放物 `cabinet_obj` 近移 `-5.7 cm` 且抖动减半（`±3.5 cm` vs `±6.5 cm`），`cabinet_restored` 复原误差 **`0.000 m`**。
- **S24 Ultra (`R5CX10M8Y8X`) Hexagon v75 FP16 实机终验（[s24u_v41_final_npu_report.json](../../artifacts.local/work/egogeom-omni-v0/s24u_v41_final_npu_report.json)）**：
  - `Pure-RGB`（`107.7 MB`）：HTP **`16.51 ms`（`60.6 FPS`）** / NetRun **`17.18 ms`**（较 `v4 17.78 / 18.51 ms` 再快 `1.27–1.33 ms`）；
  - `Unified Omni + Signed-H3`（`109.1 MB`）：HTP **`19.11 ms`（`52.3 FPS`）** / NetRun `20.69 ms`（较 `v4 19.41 ms` 快 `0.30 ms`）；
  - 实片保真（`0.15–25 m` vs PyTorch FP32）：`Pure-RGB` **`0.788%`（`33.97 mm`，较 `v4 3.33%` 降 4.23×）**、`Omni+SignedH3` **`0.601%`（`27.67 mm`）**、`Omni+StdCNH` **`0.591%`（降 3.01×）**、`Omni+Depth` **`0.598%`**、`dh_MAE = 4.87–5.30 mm`、`normal_cos = 1.0000`；
  - **芯片级零干扰实测 `max|Omni(has_tof=0) - PureRGB| = 0.000000e+00 m`**（实片 Signed-H3 平均修正 `10.45 cm`）；`/data/local/tmp/egogeom_v41_final` 已清空。
