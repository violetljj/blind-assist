# Phase 19：`EgoClear-v14`（Hyper-Omni）三嫁接旗舰全能避障 + S24 Ultra 全 NPU 实机落地

日期：2026-11-22（EXPLORE 快速通道；未读取任何官方受保护 `test` split）
算力与清理：Colab G4 (`NVIDIA RTX PRO 6000 Blackwell 96 GB`, `egoclear-v12-g4`)，全程 `3953.9 s`（`train_sec = 2059.1 s`，EMA 胜出）；全部 4 份产物已逐字节核验 Drive 一致性，并已执行 `os.sync()` + `drive.flush_and_unmount()` + `colab stop -s egoclear-v12-g4`（复核 `egoclear-v12-g4` 已释放，余额 `238.86 CU`，另 2 个并行会话未受触碰）。
代码与产物：
- [egoclear_v14_model.py](file:///e:/linnan/linnan/research/active/egoclear-v0/egoclear_v14_model.py) · [phase19_train_v14_hyper_omni_g4.py](file:///e:/linnan/linnan/research/active/egoclear-v0/phase19_train_v14_hyper_omni_g4.py)
- Drive 归档：`/content/drive/MyDrive/BlindAssist/runs/egoclear_v0_phase19_v14/`（`egoclear_v14_hyper_omni.pt` `114,506,823 B`、`egoclear_v14_encoder_fp32.onnx` `96,327,642 B`、`egoclear_v14_step_fp32.onnx` `18,676,501 B`、`phase19_v14_results.json` `897,837 B`）
- 本机结果与 DLC：[phase19_v14_results.json](file:///e:/linnan/linnan/artifacts.local/work/egoclear-v0-phase19/phase19_v14_results.json) · `egoclear_v14_encoder_htp_v75.dlc`（`154.0 MB`）· `egoclear_v14_encoder_mixed_htp_v75.dlc`（`59.3 MB`）· `egoclear_v14_step_htp_v75.dlc`（`32.7 MB`）

---

## 1. 核心结论（先看收益、速度与代价）

- **新确立的全能默认主力冠军：`EgoClear-v14` ("Hyper-Omni")**。
- **架构突破 — 零冷启动三嫁接（Triple-Graft, `v11_step_skipped = 0`）**：
  1. `v12` 的 `MoGe-Turbo 84×148` 编码器（`23.37M` 参数、`8.65 GMAC`，原样加载 `415` 个张量，原生内置 QAIRT 2.47 `Clip(0.0, M)` 免疫与 2-stage Mean FTZ-FP16 闭式 shift 求解）；
  2. `v11` 经 35 epoch 充分收敛的 `NY=9` 核心 3D Step 主干（含逐位对齐的 `ExportSafeV11ConvGRU` 与 `160→64→2` `corr_head`，原样加载 `145` 个张量、**跳过数为 `0`**，仅加柔和静态门控 `(1 - 0.72 * dyn_prob)` 抑制行人误报）；
  3. `v13` 的两个隔离并行全能头（100% 无记忆当前帧光学 `unc_head` + 多滞后径向接近率 `dyn_closing_head`，后者原样加载 `14` 个张量）。
  - **Stage 1 零样本验证（4 ep 编码器对齐，0 ep Step 训练）**：仅对齐 Turbo 编码器 4 epoch 后，直接接入冻结 `v11` Step 零样本即达到 **`Real 40 @120 = 23.4%`**（直接追平 `v11` 历史纪录，`AbsRel = 0.1687`），彻底验证零冷启动设计有效；随后 Stage 2 Ep 1 锁核心仅训隔离 Omni 头、Ep 2–6 低 LR 联合精调，EMA 以综合最优胜出（`0.4940` vs `EP1_LOCKED 0.5181 / RAW 0.5196 / SOUP 0.5490`）。
- **端侧实测飞跃（Samsung Galaxy S24 Ultra, SM8650 Hexagon v75 NPU, 双 `Total Subnets: 1`, `0` 次 yield）**：
  - **`v14 FP16`（旗舰无损默认）**：24-clip（72 帧）未见多域（Real 40 + ARKit 30 + CODa）实测 **命中距离误差 `0.000 m`（24/24 全精准命中）**、深度 **`z_MAE = 0.0135 m`**（ARKit `0.62 cm`）、六通道 **`omni6_MAE = 0.0007`**；爆发编码器 **`25.72 ms`**，**300 帧稳态全 NPU 仅 `37.69 ms`（`29.42 + 8.27 ms`，`26.5 FPS`，温升 `+0.3°C`）** —— 比 `v11 ClipFix`（`72.00 ms / 13.9 FPS`）快 **`1.91×`**，甚至超过旧轻量 `v6.6`（`38.1 ms`）！
  - **`v14 Mixed`（`W8A8-ViT + FP16-GeoDec`, `59.3 MB` 编码器，比 `v11` 小 `4.1×`）**：爆发编码器 **`18.22 ms`（`54.9 FPS`）**，**300 帧稳态全 NPU 仅 `27.82 ms`（`20.16 + 7.66 ms`，`35.9 FPS`，比 `v11` 快 `2.59×`，温升 `+0.5°C`）**；未见 24-clip `omni6_MAE = 0.0181, hit_dist_MAE = 0.217 m`（适合极速/低存储模式）。
- **核心算法破纪录项**：
  - **`Real 40 Dark 0.18× @80/120 = 36.9% (41/111)`**（**全系列 `v1–v14` 历史最低**）且暗光行人误报 **`dyn@0.5 = 7.41% (2/27)`**；
  - **`Real 40 dyn_closing AUROC = 0.9728`**（`n_pos=120, n_neg=2705`，`F1 = 0.5914`, `Prec = 83.3%`, `FPR = 0.41%`，**全系列历史最高**）；
  - **`Real 40` 单一阈值 `thr=0.50 = 21.6% (24/111, FA 166.6 fr/km)`**（迟报与误报同时优于 `v11` 的 `25.2% / 201 fr/km`），clean 行人误报 **`dyn@0.5 = 14.81% (4/27)`**（较 `v11` `29.6%` **腰斩**），真深度 **`AbsRel = 0.1672`**（优于 `v11` `0.1747`）；
  - **4 域 `blackout / finger_red / whiteout` 全 `100.0%` 无记忆光学拒答**（`mean_unc = 0.9946–0.9997`），且干净帧 `Real 40` 中位 `0.0456`（最大 `0.5282 < 0.65`）、`CODa` 中位 `0.0111`；
  - **`ARKit @120 = 3.8% (6/156)`**（较 `v11` `-5.92 pp`，`0` 条路线退化）、`thr=0.50 = 0.0%`、`Pre-Hit p0.5>0.5 = 0.0%`、`HEAD FPR = 12.12%`（较 `v12` 降 `4.5×`）、绕行准率 **`93.62% (n=486)`**；
  - **跨硬件零样本单模型反超 `v6.6`**：`Aria iso_grav_crop @120 = 60.6%`（vs `v6.6 65.8%`）、`Aria iso_crop @500 = 35.2%`（vs `v6.6 36.3%`）、`CODa iso_crop @80..500 = 50.0%`（vs `v6.6 66.7%`，`dyn = 0.00%`）。
- **诚实代价与局限**：
  - `Synth clean @120 = 21.9% (30/137)`（较 `v11` `15.3%` 差 `+6.6 pp`，配对 CI `[+2.14, +23.54]`，但双融合 `ENS(v6.6+v14) @120 = 5.1%` 仍平纪录）；
  - `ARKit Dark @120 = 21.2% (33/156)`（`@200 = 0.0%`，弱于 `v12` 的 `0.0%`）；
  - `ARKit` 平地 `drop FPR = 12.24%`（比 `v11` 的 `28.3%` 降 `2.3×`，高于 `v12` 的 `3.98%`，端侧 `0.62/0.48` 滞回护栏继续兜底）；
  - `CODa` 低机位（0.704 m）`dyn_closing AUROC = 0.3377` 仍偏弱；实走与无障碍（TalkBack/大字）回归仍标 **`NOT_TESTED`**。

---

## 2. 跨代全景对比（`v11` → `v12` → `v13` → `v14`，全带分母与单位）

### 2.1 主线静态走廊与跨硬件鲁棒性

| 基准与口径（分母） | `v11` (64ms Enc) | `v12` (30.6ms Enc) | `v13` (30ms Enc) | **`v14` (29.4ms Enc)** | `v14` 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| `Real 40 @80 / @120` (`n=111`) | **23.4% / 23.4%** | 33.3% / 33.3% | 40.5% / 40.5% | **28.8% / 27.9%** (`@200 21.6%`) | 显著胜 `v12/v13`；与 `v11` CI 跨零 `[-5.9,+16.8]` |
| `Real 40 thr=0.50` 迟报 / FA | 25.2% / 201 fr/km | 33.3% / 141 fr/km | 23.4% / 509 fr/km | **21.6% / 166.6 fr/km** | **固定阈值双项胜 `v11`！** |
| `Real 40 dyn@0.5` (`n=27`) | 29.6% (8/27) | 18.5% (5/27) | 33.3% (9/27) | **14.81% (4/27)** | **较 `v11` 腰斩！** |
| `Real 40 Dark @80/120` (`n=111`) | 38.7%–44.1% | 45.9% | 47.7% | **36.9% (41/111)** (`dyn 7.41%`) | **全系列历史新低！** |
| `Real 40 AbsRel / scale` | 0.1747 | **0.1633** | 0.1753 | **0.1672** (`0.944×`) | 优于 `v11/v13` |
| `Synth clean @120` (`n=137`) | **15.3%** (21/137) | 18.2% (25/137) | 25.5% (35/137) | 21.9% (30/137) (`ENS 5.1%`) | 单模型诚实落后 `v11`，双融合平纪录 |
| `ARKit val30 @120` (`n=156`) | 10.3% (16/156) | **0.0%** (0/156) | 8.3% (13/156) | **3.8%** (6/156, `@200 0.0%`) | 较 `v11` `-5.92 pp`（0 退化） |
| `Aria iso_grav_crop @120` (`n=193`) | **56.0%** | 59.6% | 71.0% | **60.6%** (`ENS 58.5%`) | 单模型胜 `v6.6 (65.8%)` `-5.2 pp` |
| `CODa iso_crop @80..500` (`n=36`) | **50.0%** | 66.7% | 66.7% | **50.0%** (`dyn 0.0%`) | 单模型胜 `v6.6 (66.7%)` `-16.7 pp` |

### 2.2 全能六通道能力对比

| Omni 能力与分母 | `v11` | `v12` | `v13` | **`v14`** |
| --- | ---: | ---: | ---: | ---: |
| 4 域 `blk / finger / white` (`>0.65`) | Real blk `31%` | ARKit `66.7%` | **4 域全 `100.0%`** | **4 域全 `100.0%`** (`0.995–1.000`) |
| `Real 40 dyn_closing` (`120 / 2705`) | AUC `0.912` | AUC `0.788` / F1 `0.340` | AUC `0.912` / F1 `0.484` | **AUC `0.9728` / F1 `0.5914`** (`FPR 0.41%`) |
| `Real 40 drop_off` (`237 / 2823`) | AUC `0.930` | AUC `0.941` / F1 `0.697` | AUC `0.938` / F1 `0.654` | **AUC `0.9367` / F1 `0.6936`** (`FPR 2.48%`) |
| `ARKit drop_off FPR` (`n=980`) | 28.3% | **3.98%** | 4.90% | 12.24%（`Synth/CODa = 0.00%`） |
| `HEAD` (`Synth F1` / `ARKit FPR` / `CODa AUC`) | — | — / 54.1% / — | 0.599 / 12.97% / 0.787 | **0.6534 / 12.12% / 0.8958** |
| 三车道绕行方向准率 (`Real43 / ARKit486`) | 90.7% / 96.9% | — | 90.7% / 92.0% | **90.7% / 93.62%**（`CODa 100%`） |

---

## 3. S24 Ultra Hexagon v75 NPU 硅上实测 + QAIRT 2.47 `Clip(min='')` 缺陷全景

在实机 `Samsung Galaxy S24 Ultra (SM8650, Hexagon v75, Android 16)` 上，用 **24 段未见多域视频（72 帧：Real 40 × 8 + ARKit 30 × 8 + CODa × 8）** + **300 帧 `sustained_high_performance` 稳态长测** 同口径实测：

| 引擎配置 | DLC 大小 (Enc+Step) | 爆发 Enc (72 fr) | 300 帧稳态 (Enc + Step) | 稳态温升 | 未见 24-clip `z_MAE` | `omni6_MAE` | 命中距离误差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v11 Raw`（修复前） | `245 + 29 MB` | `56.3 ms` | — | — | `0.1941 m` | `0.0070` | `0.058 m` |
| **`v11 ClipFix FP16`** | `245 + 32 MB` | `61.1 ms` | `72.00 ms` (`64.04+7.96`, `13.9 FPS`) | `+0.2°C` | **`0.0154 m`**（降 `12.6×`） | **`0.0007`** | `0.017 m` |
| `v12 Raw`（修复前） | `154 + 31 MB` | `25.7 ms` | — | — | `0.9565 m` | `0.0337` | `0.250 m` |
| `v12 ClipFix FP16` | `154 + 31 MB` | `25.7 ms` | `39.32 ms` (`30.60+8.72`, `25.4 FPS`) | **`+0.0°C`** | `0.0129 m`（降 `74.1×`） | `0.0009` | **`0.000 m`** |
| `v12 ClipFix Mixed` | `59.3 + 31 MB` | `18.3 ms` | `31.75 ms` (`23.09+8.66`, `31.5 FPS`) | **`+0.0°C`** | — | `0.0166` | `0.125 m` |
| **`v14 FP16`（旗舰默认）** | **`154 + 32.7 MB`** | **`25.72 ms`** | **`37.69 ms` (`29.42+8.27`, `26.5 FPS`)** | **`+0.3°C`** | **`0.0135 m`** | **`0.0007`** | **`0.000 m` (24/24!)** |
| **`v14 Mixed`（极速紧凑）** | **`59.3 + 32.7 MB`** | **`18.22 ms`** | **`27.82 ms` (`20.16+7.66`, `35.9 FPS`)** | **`+0.5°C`** | `0.5988 m` | `0.0181` | `0.217 m` |

- 全部上表配置均为 **双 `Total Subnets: 1`**、**`0` 次 HTP yield**；`v14` 源码原生无任何单边 `Clip(min='')` 节点（`patch_onnx_qairt_clip_bug.py` 复检报告 `fixed 0`）。
- 手持暗室 5 帧回放（`smoke_user_room.egoclear`, `thr=0.50`）：`v14` 与 `v11/v12` 一致在 `1.3 m` 零误报、`1.9 m` 100% 检出（`last_hit = 1.70 m`），且三项非目标通道误报全场最低（`max_drop=0.169`, `max_head=0.239`, `max_dyn_closing=0.098`）。

## 4. 真机 `:egoclear-demo-app` 实机部署验证

- `:egoclear-demo-app` 已升级 `v14 → v13 → v12 → v11 → v10.6 → v10.5` 自动加载优先级与 `lastModified` 缓存刷新校验，通过 `scripts/run_android_gradle.ps1 :egoclear-demo-app:installDebug :egoclear-device-canary:compileDebugKotlin` 编译安装至 S24 Ultra（`BUILD SUCCESSFUL in 34s`），推送 `v14 FP16` 全 NPU 双 DLC（`154.0 MB + 32.7 MB`）后冷启实测日志：
  - `CAMERA_BOUND minZoom=0.60 activeZoom=0.60`
  - `ENGINE_READY model=v14 backend=HTP_V75_NPU stepBackend=HTP_V75_NPU omni=true encLoadMs=346.0 stepLoadMs=37.1 totalLoadMs=383.2`
  - `ENGINE_READY model=v6.6 backend=HTP_V75_NPU stepBackend=HTP_V75_NPU encLoadMs=109.9 stepLoadMs=41.9 totalLoadMs=151.8`
- `v14` 与 `v6.6` 双全 NPU 引擎合计冷启 **`535.0 ms`** 零异常（支持 `v14 / 双融合 / v6.6` 一键切换）；解锁实走与 TalkBack/大字体无障碍回归仍诚实标记为 `NOT_TESTED`。
