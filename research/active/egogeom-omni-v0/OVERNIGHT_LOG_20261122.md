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
