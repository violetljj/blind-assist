# Phase 18：`EgoClear-v13`（Apex-Omni）探索与诚实复盘

日期：2026-11-22（EXPLORE 快速通道；未读取任何官方受保护 `test` split）
算力：Colab G4 (`NVIDIA RTX PRO 6000 Blackwell 96 GB`)，训练总墙钟 `4757.8 s`（`train_sec = 2818.7 s`，EMA 胜出）。
代码与产物：
- [egoclear_v13_model.py](file:///e:/linnan/linnan/research/active/egoclear-v0/egoclear_v13_model.py)
- [phase18_train_v13_apex_omni_g4.py](file:///e:/linnan/linnan/research/active/egoclear-v0/phase18_train_v13_apex_omni_g4.py)
- Drive 归档：`/content/drive/MyDrive/BlindAssist/runs/egoclear_v0_phase18_v13/`（`egoclear_v13_apex_omni.pt`、`egoclear_v13_encoder_fp32.onnx`、`egoclear_v13_step_fp32.onnx`、`phase18_v13_results.json`）
- 本机结果副本：[phase18_v13_results.json](file:///e:/linnan/linnan/artifacts.local/work/egoclear-v0-phase18/phase18_v13_results.json)

---

## 1. 一句话结论（先看收益与代价）

- **结论：不直接晋级主线，但成功产出了 3 个关键架构突破并直接移交给 Phase 19 `EgoClear-v14`**。
- **核心收益（全部实测成立）**：
  1. **100% 无记忆光学 UNKNOWN 拒答门控 (`unc_head`)**：彻底切除 `bev_ref` 记忆泄露后，在 **全部 4 个域（Synth / Real 40 / ARKit 30 / CODa）上实现 `100.0%` 黑屏/红指遮挡/白屏过曝拒答（`mean_unc = 0.9988–0.9998`）**，完全修复 `v12` 在 ARKit 上的 `66.7%` 漏放；在 `Real 40` 上 **全部 5 种 OE 退化（含 `motion_blur 100.0%`、`defocus_blur 100.0%`）均达到 `100.0%`**！
  2. **多滞后径向接近率 TTC 动态逼近头 (`dyn_closing_head`)**：在 `Real 40` 上达到 **`AUROC = 0.9118`（比 `v12` 暴涨 `+12.4 pp`）、`F1@0.5 = 0.4835`（`+14.4 pp`、`FPR = 0.67%`）**；`Synth` 达到 **`AUROC = 0.8560`（`+12.3 pp`）、`F1@0.5 = 0.6335`（`Prec = 84.2%`）**；零样本 `CODa` 从 `~0.30` 提升到 `0.5865`。
  3. **头部障碍精度与深度稳定性**：`ARKit HEAD FPR@0.5` 大降 **`4.2×`（`54.1% → 12.97%`，`F1 = 0.7131`）**，`Real 40 HEAD AUROC` 由 `0.699` 升至 `0.7412`，`Synth HEAD AUROC = 0.9896`；深度尺度保持在 **`Real 40 AbsRel = 0.1753, scale = 0.9451×`**，原生内置 QAIRT 2.47 `Clip(0.0, M)` 免疫修复。
- **诚实代价（为何不直接发布 `v13`）**：
  - 主线 `corr_probs` 出现明显退化：`Real 40 @120 = 40.5% (45/111)`（vs `v12 33.3%`, `v11 21.6%`；相对 `v11` 配对 bootstrap 中位退化 `+16.92 pp`，95% CI `[+2.86, +33.98]`），`Synth @120 = 25.5% (35/137)`（vs `v11 15.3%`），`CODa @120 = 66.7% (24/36)`，`Real 40 dyn_only@0.5 = 33.3% (9/27)`，`Real 40` 固定阈值 `thr=0.50` 误报率暴涨至 `509.0 fr/km`。

---

## 2. 核心指标对比（全部带单位与分母）

### 2.1 主线静态走廊与鲁棒性

| 评估集 | 指标（分母） | `v11` (FP32) | `v12` (FP32) | `v13` (FP32) | 观察 |
| --- | --- | ---: | ---: | ---: | --- |
| `Synth clean` | `@120` 迟报率 (`n=137`) | **15.3%** (21/137) | 18.2% (25/137) | 25.5% (35/137) | 退化（`+10.2 pp` vs `v11`） |
| `Synth VIO 1x` | `@120` 迟报率 (`n=137`) | **15.3%** (21/137) | 19.0% (26/137) | 26.3% (36/137) | 同上 |
| `Real 40 local` | `@120` 迟报率 (`n=111`) | **21.6%** (24/111) | 33.3% (37/111) | 40.5% (45/111) | 退化（配对 CI `[+2.86, +33.98] pp`） |
| `Real 40 local` | `thr=0.50` 迟报 / FA | 25.2% / 201 fr/km | 33.3% / 141 fr/km | 23.4% / **509.0 fr/km** | 概率整体偏高导致 FA 暴涨 |
| `Real 40 local` | `dyn_only@0.5` (`n=27`) | 29.6% (8/27) | **18.5%** (5/27) | 33.3% (9/27) | 正样本二次锚点推高行人漏报 |
| `Real 40 gray` | `@120` 迟报率 (`n=111`) | — | — | 31.5% (35/111) | 灰度表现反而优于彩色 |
| `Real 40 dark 0.18` | `@120` 迟报率 (`n=111`) | — | — | 47.7% (53/111) | 暗光同步受影响 |
| `ARKit val30` | `@120` 迟报率 (`n=156`) | 10.3% (16/156) | **0.0%** (0/156) | 8.3% (13/156) | `thr=0.50` 仍 `0.0%`，FA `278.1 fr/km` |
| `ARKit dark 0.18` | `@120` 迟报率 (`n=156`) | — | **0.0%** (0/156) | 3.8% (6/156) | 仍强但不如 `v12` |
| `Aria G2` (零样本) | `iso_grav_crop @120` (`n=193`) | 69.9% | **59.6%** | 71.0% | 退化 |
| `CODa` (零样本) | `@120` 迟报率 (`n=36`) | **50.0%** (18/36) | 66.7% (24/36) | 66.7% (24/36) | 双融合 `ENS(v6.6+v13)` 回到 `50.0%` |

### 2.2 全能六通道能力突破（`v13` 的真正遗产）

| 能力 | 指标（分母） | `v12` | `v13` | 变化 |
| --- | --- | ---: | ---: | --- |
| **4 域光学 UNKNOWN** | `ARKit blackout / finger / whiteout` (`>0.65`) | 66.7% | **100.0% / 100.0% / 100.0%** (`0.999`) | **彻底修复记忆泄露** |
| **Real 40 5-模式 OE** | `blk / finger / white / mblur / dblur` | 100/100/100/—/— | **100% / 100% / 100% / 100% / 100%** | 5/5 满分 |
| **Risk-Coverage** | `Real 40` / `CODa` 错误率 (`100% → 70%`) | — | `18.55% → 3.11%` / `14.15% → 0.71%` | 风险下降 `6×–20×` |
| **动态逼近 TTC (`dcl`)** | `Real 40` (`n_pos=120, n_neg=2705`) | AUROC 0.788 / F1 0.340 | **AUROC 0.9118 / F1 0.4835** (FPR 0.67%) | **`+12.4 pp` AUROC** |
| **动态逼近 TTC (`dcl`)** | `Synth` (`n_pos=902, n_neg=4424`) | AUROC 0.733 / F1 0.48 | **AUROC 0.8560 / F1 0.6335** (Prec 84.2%) | **`+12.3 pp` AUROC** |
| **头部障碍 (`HEAD`)** | `ARKit` (`n_pos=821, n_neg=586`) | FPR 54.1% | **FPR 12.97%** (`F1=0.7131, AUROC=0.8257`) | **误报降 `4.2×`** |
| **头部障碍 (`HEAD`)** | `Real 40` (`n_pos=1377, n_neg=1448`) | AUROC 0.699 | **AUROC 0.7412** (`Prec=89.2%, FPR=1.17%`) | 稳步提升 |
| **跌落 / 台阶 (`DROP`)** | `Real 40` (`n_pos=237, n_neg=2823`) / `ARKit` (`n=980`) | AUC 0.941 / ARKit FPR 3.98% | **AUC 0.9383 (Rec 81.4%)** / ARKit FPR 4.90% | 保持低误报高召回 |
| **绕行方向选择** | `Real 40 (n=43)` / `ARKit (n=486)` / `Synth (n=65)` | — | **90.7% / 91.98% / 87.69%** | 三域均接近或超 90% |

---

## 3. 主线退化的精确根因剖析（直接驱动 Phase 19 `EgoClear-v14`）

对照 `v11 / v12 / v13` 的训练与权重初始化日志，定位到 3 个独立叠加的根因：
1. **重复冷启动核心 3D Step 主干**：`v12` 与 `v13` 都因改动 `Y` 切片或 `bev_enc` 通道数，丢弃了 `v11` 经 35 epoch 充分收敛的 `pt_reduce + bev_enc + mem_gru + bev_refine + corr_head` 权重，仅用 8–10 epoch 从头重建 3D 走廊判定边界。
2. **二次正样本锚点推高全局误报 (`FA@0.5 = 509 fr/km`)**：`v13` 在 `compute_loss_v13` 中引入了 `1.65 * ReLU(1.35 - cb)^2` 强二次正样本拉升项，在 Real 40 这种正负边界密集的真实道路上过度拉高累积风险，直接导致误报率飙升 `2.5–3×` 并把 `Real 40 dyn_only@0.5` 推回 `33.3%`。
3. **12% Aria 方形黑边增广污染了冻结教师输入 `rgb_s_clean`**：`V13ApexOmniDataset` 将方形 pillarbox 增广同时施加在教师和学生输入上，导致约 12% 训练片段的冻结 `v6.6 / v11` 软标签失真。

## 4. 向 Phase 19 `EgoClear-v14` 的精确交接

`EgoClear-v14` ("Hyper-Omni") 采用 **零冷启动三嫁接（Triple-Graft）** 直接收割三代最强部件：
- `v12` 的 `MoGe-Turbo 84×148` 编码器（S24 Ultra `25.68 ms` FP16 / `18.26 ms` Mixed `59 MB`，比 `v11` 快 `2.2×–3.1×`）+ 原生 QAIRT `Clip(0.0, M)` 免疫；
- `v11` 完整的 35-epoch 收敛 `NY=9` 核心 Step 主干（`100%` 原样加载，含 `corr_head` 与左右绕行头，预计 S24 Ultra Step 仅 `~6.2 ms`）；
- `v13` 的两个已验证隔离并行头（100% 无记忆 `unc_head` + `0.912 AUROC` 径向接近率 `dyn_closing_head`）；
- 取消二次正锚点与 Aria 教师污染，改用两阶段零冲击微调（Stage 1 紧对齐 `v11` 特征空间 → Stage 2 Ep 1 锁核心仅训隔离 Omni 头 → Ep 2–6 低 LR 联合精调，并在 `{EP1_LOCKED, RAW, EMA, SOUP}` 中客观择优）。
