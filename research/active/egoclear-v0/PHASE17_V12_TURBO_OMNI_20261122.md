# Phase 17 报告：EgoClear-v12 ("Turbo-Omni") 与 Hexagon v75 硅片级数值修复

- 日期：2026-11-22（`EXPLORE` 快车道；零触碰任何官方受保护 `test` 集）
- 训练算力：Colab G4（`NVIDIA RTX PRO 6000 Blackwell 96 GB`，总训练 `2,454.6 s`，含全量 9 基准 + 5 模态 OE 审计总时长 `4,234.2 s`）
- 真机验证：Samsung Galaxy S24 Ultra（`SM-S9280`, SM8650 Hexagon v75, QAIRT `2.47.0.260601`）
- 核心代码：[egoclear_v12_model.py](egoclear_v12_model.py)、[phase17_train_v12_turbo_omni_g4.py](phase17_train_v12_turbo_omni_g4.py)

## 1. 结论先行（收益与代价）

1. **架构提速 `2.19×` 且编码器减重 `1.6×`，不降级 ViT 主干**：
   - Phase 16 逐层 MAC 剖析发现，`v11` 编码器（`119.5 GMACs`）中有 `88.1 GMACs`（`73.7%`）耗费在 `MoGe-2` 的 4 路 `336×592` ResNet 头上，最终却被 `16×` 下采样至 `84×148`。
   - `v12` 保留 `100%` 预训练 `DINOv2 ViT-Small/14`（`22.5 GMACs`）与 `scale_head`，用单一 `~5 GMAC` `UnifiedTurboGeoDecoder` + 高分辨率 `RGB Edge & Optical-Quality Stem` 替代 `88.1 GMACs` 冗余卷积栈。
   - **Stage 1（3 ep，`204.9 s`）Real 40 深度门控即达 `AbsRel = 0.1633`、`scale = 0.9776×`**，反超 `v11` 的 `88.1 GMAC` 解码器（`0.1747 / 0.946×`）。
   - S24 Ultra 72 帧实测稳态推理：**FP16 编码器 `25.68 ms`（`Init 54.9 ms`, `0` yields，较 `v11` `56.30 ms` 快 `2.19×`）+ Step `8.77 ms` = `34.45 ms`（`29.0 FPS` Full-NPU）**；混合精度 **`Mixed (W8A8-ViT + FP16-GeoDec)` 编码器压至 `18.26 ms`（`59.3 MB` DLC，较 `v11` `245 MB` 小 `4.1×`，端到端 `27.03 ms` / `37.0 FPS`）**。
2. **重大真机硅片发现：修复 QAIRT 2.47 `Clip(min='')` 编译器缺陷（`74×` 真机深度误差下降）**：
   - 在 **24 条未见多域真机实测集（72 帧：`Real 40` 8 条 + `ARKit 30` 8 条 + `CODa` 8 条）** 上发现，原始 QAIRT 2.47 会将 ONNX 单边 `Clip(x, min='', max=M)`（来自 PyTorch `.clamp(max=M)`）在 Hexagon v75 HTP 上误编译为常量 `M`，导致编码器 `a2.clamp(max=1.0)` 全图恒为 `1.0`（偏移闭式 `shift` 求解 `-0.3..-2.2 m`）及 Step 中 `base.clamp(max=2.5)` / `age_w.clamp(max=10.0)` 失效。
   - 通过显式双边 `.clamp(0.0, M)` 与 ONNX 清洗脚本修复后（零重训、零延迟代价），S24 Ultra **24 条未见多域实测集** 上 FP16 真机与 CPU FP32 对齐误差实现数量级跃升：
     - 深度 `z_MAE`：**`0.9565 m → 0.0129 m`（`1.29 cm`，降低 `74.1×`）**；
     - 端到端 6 通道 `omni_probs MAE`：**`0.0337 → 0.0009`（降低 `37×`）**；
     - 首撞距离 `hit_dist_MAE`：**`0.250 m → 0.000 m`（24/24 条未见片段完全命中同一 20 cm 桶）**；
     - Full-NPU 端到端动态通道 `dyn_close MAE`（修复 Step 中 `base` 缺陷后）：**`0.10108 → 0.00042`（降低 `240×`）**。
3. **关键能力突破与诚实回退（直接驱动 Phase 18 `EgoClear-v13`）**：
   - **突破项**：
     - 室内 `ARKit`（`0/156 = 0.0%` @120 与固定 `thr=0.50`）及 `ARKit Dark`（`0/156 = 0.0%` @120、`4/156 = 2.6%` @80）显著优于 `v11`（配对 bootstrap 95% CI `[-21.25, -1.57] pp`）；
     - 室内平地 `drop_off` 误报率从 `v11` 的 `28.3%` 暴降 `7.1×` 至 **`3.98%`**（解锁 `22,631` 帧平地负样本），同时 `Real 40 drop_off` 保持 **`AUROC = 0.9411`（`F1 = 0.6971`, `FPR = 3.72%`）**，`CODa` 与 `Synth` `drop_off FPR = 0.00%`；
     - 5 模态光学 `UNKNOWN` 拒识：`Real 40` / `Synth` / `CODa` 上 `blackout / finger_red / whiteout` 均达 **`100.0%`（`mean_unc = 0.978–0.999`）**，`motion_blur 58–75%`、`defocus_blur 50–83%`，洁净风险-覆盖 `Real 40 err_cov100 9.55% → err_cov70 2.00%`；
     - 行人漏入静态通道 `Real 40 dyn@0.5` 由 `v11` `29.6% (8/27)` 降至 **`18.52% (5/27)`**（暗光 **`14.81% (4/27)`**）；
     - `Aria` 单模型 `iso_grav_crop @120 = 59.6%`（优于 `v6.6` `65.8%`）。
   - **明确回退与根因（留给 Phase 18 `v13` 逐一外科手术修复）**：
     - `CODa` 低机位（真值 `0.704 m`）上 `h_floor_cam` 因 `w_low` 向下偏置与尺度高估被拉至 `-1.836 m`（`MAE = 1.065 m`），把 9 层网格整体压入地下，导致 `CODa @120 = 66.7% (24/36)` 及 `dyn_closing AUROC = 0.306`；
     - Stage 2 初始阶段在冷启动 `step` 下直接解冻解码器 + ViT 且 `kd_w` 过低（`0.42 → 0.26`），导致深度从 Stage 1 的 `0.1633` 漂移至 `0.1859`，`Real 40 @120` 退至 `33.3% (37/111)`、`Synth @120` 退至 `21.2% (29/137)`（阈值分裂：`Synth thr=0.20` vs `Real thr=0.60`）；
     - `head_probs` 错误蒸馏了 `v11` 低召回输出（`10.8%`），导致室外 `Real 40 HEAD Rec@0.5` 仅 `12.1%`；
     - `unc_head` 仍拼接了记忆特征 `bev_ref`，导致 4/12 条 `ARKit` 近墙遮挡帧 `blackout` 拒识掉至 `66.7%`。

## 2. S24 Ultra 24 条未见多域真机精度与延迟对照（72 帧实测）

| 编码器编译方案（均配 Full-NPU FP16 Step `8.77 ms`） | DLC 大小 | 编码器延迟（72 帧） | 端到端 Full-NPU | 未见 24 片段 `z_MAE` | 未见 `E2E corr MAE` | 未见 `E2E omni6 MAE` | 未见 `hit_dist MAE` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v12` 原始 FP16（含 QAIRT `Clip(min='')` 缺陷） | `154.0 MB` | `25.70 ms` | `34.47 ms` (`29.0 FPS`) | `0.9565 m` | `0.0537` | `0.0337` | `0.250 m` |
| **`v12` ClipFix FP16（本阶段修复）** | `154.0 MB` | **`25.68 ms`** | **`34.45 ms` (`29.0 FPS`)** | **`0.0129 m`** | **`0.0014`** | **`0.0009`** | **`0.000 m`** |
| **`v12` ClipFix Mixed（`W8A8-ViT + FP16-GeoDec`）** | **`59.3 MB`** | **`18.26 ms`** | **`27.03 ms` (`37.0 FPS`)** | `0.6275 m` | `0.0284` | `0.0166` | `0.125 m` |
| `v12` 全量 W8A8（48 帧多域校准） | `55.1 MB` | `10.85 ms` | `19.62 ms` (`51.0 FPS`) | `1.5948 m` | `0.1423` | `0.0890` | `0.908 m` |
| 参考对照：`v11` FP16（Phase 16） | `245.1 MB` | `56.30 ms` | `62.42 ms` (`16.0 FPS`) | — | — | — | — |

> 注：以上全部 DLC 经 `snpe-diagview` 核验 `Num times yield occurred: 0`，`Total Subnets: 1`，`ALL6_FINITE = True`，`ALL6_MONO = True`。全量 `W8A8` 虽达 `10.85 ms`，但在未见室外大景深上 `hit_dist_MAE = 0.908 m`，因此否决全量 `W8A8`，确立 **`ClipFix FP16`（零误差旗舰）** 与 **`ClipFix Mixed`（59 MB / 37 FPS 轻量档）** 双轨方案。

## 3. 多域主线与 Omni 能力指标（EMA 胜出）

- **9 项主线基准（`@120 fr/km` 漏晚率，括号为 `late/avoidable`）**：
  - `synth_clean`: `21.2% (29/137)`（固定 `thr=0.50`: `38.0%`, `FA=31.3`；`ENS(v6.6+v12) @120 = 5.1%`；`dyn@0.5 = 0.47% (5/1067)`）
  - `synth_vio_1x`: `21.2% (29/137)`
  - `real_40_local`: `33.3% (37/111)`（固定 `thr=0.50`: `29.7%`, `FA=205.4`；`dyn@0.5 = 18.52% (5/27)`；`AbsRel = 0.1859`, `scale = 0.939×`）
  - `real_40_gray`: `33.3% (37/111)`（`dyn@0.5 = 18.52% (5/27)`）
  - `real_40_dark018`: `45.9% (51/111)`（`dyn@0.5 = 14.81% (4/27)`）
  - `arkit_indoor_val30`: **`0.0% (0/156)`**（固定 `thr=0.50`: **`0.0% (0/156)`**, `FA=151.7`）
  - `arkit_dark018`: **`0.0% (0/156)`**（`@80 = 2.6% (4/156)`）
  - `aria_g2_zeroshot`: squashed `75.1% (145/193)` / **`iso_grav_crop 59.6%`** / `ENS(v6.6+v12) squashed 54.9%`
  - `coda_campus_zeroshot`: `66.7% (24/36)`（`ENS(v6.6+v12) = 50.0%`, `dyn@0.5 = 0.00% (0/24)`）
- **Pre-Hit Zero-Leakage**：`ARKit` `n=169` `far_obs_mean_p0.5m = 0.0253`（`frac>0.5 = 0.0%`）；`Real 40` `n=157` `far_obs_mean_p0.5m = 0.0053`（`frac>0.5 = 0.0%`）。
- **Omni 能力审计**：
  - 3 通道绕行方向准确率：`Real 40 93.02% (n=43)`、`ARKit 93.00% (n=486)`、`Synth 90.77% (n=65)`、`CODa 100.0% (n=3)`；
  - `drop_off`：`Real 40 AUROC = 0.9411 / F1 = 0.6971 / FPR = 3.72% (n_pos=237, n_neg=2823)`，`ARKit FPR = 3.98% (n_neg=980)`，`CODa FPR = 0.00% (n_neg=2632)`，`Synth FPR = 0.00% (n_neg=4955)`；
  - `head_corridor`：`Synth F1 = 0.6235 (R=70.6%)`、`ARKit F1 = 0.8027 (R=92.9%, FPR=54.1%)`、`Real 40 F1 = 0.2144 (R=12.1%, Prec=92.3%)`；
  - `dyn_closing`：`Real 40 AUROC = 0.7882 (F1=0.3399, passing_ped_dcl_fpr=1.79%)`、`Synth AUROC = 0.7329 (F1=0.5944)`、`CODa AUROC = 0.3062`；
  - 5 模态 `UNKNOWN`（`>0.65` 拒识率）：`Real 40` `blackout 100% / finger_red 100% / whiteout 100% / motion_blur 75.0% / defocus_blur 83.3%`；`Synth` `100 / 100 / 100 / 58.3 / 75.0%`；`CODa` `100 / 100 / 100 / 75.0 / 50.0%`；`ARKit` `66.7 / 66.7 / 66.7 / 16.7 / 41.7%`。

## 4. 决策与下一步

- `v12` 已成功验证 **`MoGe-Turbo 84×148` 编码器（`25.68 ms` FP16 / `18.26 ms` Mixed）**、`ARKit 0.0%`、室内 `drop_off FPR 3.98%` 以及 `ClipFix` 真机 `0.000 m` 精度对齐，但上述 5 项算法回退使其尚不足以取代 `v11` 成为最终旗舰。
- 已基于本阶段根因审计立即启动 **Phase 18 `EgoClear-v13` ("Apex-Omni")**（热启自 `v12`，固定大覆盖 `Y_SLICES_V13` + 无偏软地面条件、全去记忆 `unc_head`、前 2 epoch 冻结编码器防止深度漂移、二次均衡全局阈值锚点 + 强双师 KD、去除缺陷 `v11` `head` 蒸馏、12% `Aria` 柱状信箱增强与末 3 epoch EMA Soup）。
- 局限说明：本阶段仍属 `EXPLORE` 离线基准 + 芯片实测，未包含真实户外盲人行走试验或 TalkBack 无障碍回归（标注 `NOT_TESTED`）。
