# EgoGeom-Omni v4.1 / BlindAssist 究极数学理论与开放难题规格书 V5（M6–M14，可直接投喂 Codex）

> 适用仓库：`e:\linnan\linnan`（独立探索目录 `research/active/egogeom-omni-v0/math_theory/`）
> 承接基线：
> - 上轮五题实测与证伪结论：[math_theory/RESULTS.md](math_theory/RESULTS.md)（commit `6f704332`、`4e9df1ae`）
> - 主线成品：`EgoGeom-Omni v4.1`（commit `664bcd60`，51.18M / 50.75M DINOv3 `convnext_small.dinov3_lvd1689m`，S24 Ultra Hexagon v75 FP16 `16.51 ms` 纯 RGB / `19.11 ms` 统一 Omni + Signed-H3，实片保真 `0.788% / 0.601%`，零干扰 `0.000000 m`）
> 执行环境：
> - 主研究 Python：`E:\codex-tools\tools\venvs\blindassist-sam3-gpu\Scripts\python.exe`（PyTorch `2.11.0+cu130`、RTX 5060 8.5 GB、NumPy/SciPy/SymPy/OpenCV/Numba）
> - QNN 2.47 转换与实片验证环境：`E:\codex-tools\tools\venvs\qairt-2.47-py310\Scripts\python.exe`（目标真机 SM-S9280 / Hexagon v75 / SoC 57）

---

## 全局投喂约定（复制到任意单题 Prompt 顶部）

```text
[GLOBAL CODEX CONTRACT — V5]
1. 工作目录与严格隔离：
   - 所有新代码、定理推导与报告严格放入 `research/active/egogeom-omni-v0/math_theory/<task_id>/`；
   - 大体积缓存、模型二进制、逐帧数组与实验 JSON 严格放入 `artifacts.local/work/egogeom-omni-v0/math_theory/<task_id>/`；
   - 严禁修改 `egogeom_omni_v4_model.py`、`egogeom_omni_v41_model.py`、默认 Android `:app` 或其他并行研究目录的冻结文件。
2. 四位一体强制交付（缺一不可）：
   (A) `THEORY.md`：逐行完整数学推导、精确命题/定理陈述、全部前提假设、必要/充分条件，以及**独立的“失效边界与精确反例”小节**（严禁用“显然/易得/略去”跳过关键放缩、正定性、可观性秩或边界条件检查）。
   (B) `verify_theory.py`：使用 SymPy（符号验证微分/雅可比/旋度/李导数/欧拉-拉格朗日/变分条件）+ 蒙特卡洛/PDE/合成反例数值实验，检验每个定理界紧致度与假设破坏时的退化行为，输出 `theory_verification.json`。
   (C) `eval_empirical.py`：在下方冻结真实资产上执行严格因果、零未来泄露、零标签反馈的实证评测，输出 `v5_empirical_report.json`（含输入/代码 SHA-256、固定分母与共同支持双口径、原评分 FA/km 与真实清路重复提示/km 双口径、单线程 CPU 逐帧耗时 P50/P95）。
   (D) 诚实裁决节（写入 `THEORY.md` 末节）：给出明确可复现的“采用 / 保留诊断 / 物理否决”结论。**若某目标被严格数学界（CRLB/极值天花板/不可观反例）证伪，完整证明该不可能边界视同 100% 高质量完成，绝不刷数或事后挑子集掩盖失败。**
3. 反作弊与评估铁律：
   - 所有跨设备/跨地点评测须先按来源（source/site/walk/sequence）严格留出或双折交叉拟合，禁止在测试折选阈值；
   - 所有涉及预测距离 MAE 的对比，除常规有效输出 MAE 外，必须同步报告“缺测补固定值（如 6 m）”的固定分母 MAE 及共同支持子集 MAE，杜绝靠提高弃权率虚降 MAE；
   - 不得读取受保护的 `480/test` 集或任何未授权保留集；所有 Development 实测须明确标注证据边界（非真实临床/导航安全证明）。
```

---

## 冻结共享资产与基线清单（已核实全部存在于本地磁盘）

### A. `EgoGeom-Omni v4` & `v4.1` 模型与缓存流
- **`v4.1` 权重与双静态 ONNX-17**：
  - `artifacts.local/work/egogeom-omni-v0/egogeom_omni_v41_unified_best.pt`
  - `artifacts.local/work/egogeom-omni-v0/egogeom_omni_v41_pure_rgb_best.pt`
  - `artifacts.local/work/egogeom-omni-v0/egogeom_omni_v41_best_pure_rgb_fp32.onnx`
  - `artifacts.local/work/egogeom-omni-v0/egogeom_omni_v41_best_omni_fp32.onnx`
  - `artifacts.local/work/egogeom-omni-v0/qnn_v75_v41_final/`（含 `egogeom_omni_v41_pure_rgb_v75_fp16.bin` `107.72 MB / 16.51 ms / 0.7881%`、`egogeom_omni_v41_omni_v75_fp16.bin` `109.38 MB / 19.11 ms / 0.6008%`）
- **`v4` 冻结产物与四域因果特征流**（与 M1–M5 同口径直接纵比）：
  - `artifacts.local/work/egogeom-omni-v0/math_theory/v4_stream/`（含 `manifest.json` 与 `aria_gen2`、`coda_campus`、`arkit_val30_day`、`arkit_val30_dark018` 共 103 条流 / 12,872 帧 / 5,547 有效走廊帧，含 `depth_s4`, `dh_s4`, `nrm_s4`, `ls_s4`, `gray_s4`, `K`, `poses_R`, `poses_t`, `causal_fwd`, `strict_fwd`）
  - 构造脚本参考：`research/active/egogeom-omni-v0/math_theory/shared/build_v4_stream.py`（若需在 `v4.1` 上同步复验，可按同样无标签契约物化至 `artifacts.local/work/egogeom-omni-v0/math_theory/v41_stream/`）
- **四套跨眼镜原始打包数据**：
  - `artifacts.local/work/egogeom-omni-v0/zeroshot_eval/aria_g2_20clips/`（19 clips，1,794 valid / 840 pos `<=4m`）
  - `artifacts.local/work/egogeom-omni-v0/zeroshot_eval/coda_zeroshot_shard000/`（24 clips，2,819 valid / 34 pos `<=4m`）
  - `artifacts.local/work/egogeom-omni-v0/zeroshot_eval/arkit_val30_shard00/`（30 clips，467 valid / 388 pos `<=2m`，支持 `dark_scale=1.0` 与 `0.18`）
- **16 帧四域无标签硅片保真面板**：
  - `artifacts.local/work/egogeom-omni-v0/math_theory/m5_info_geom_quant_reparam/silicon_followup/panel_inputs/`

### B. 真实硬件、浅障、航向与 H3 实采资产
- **浅障（5–15 cm 挡轮块/矮台 + 同路空铺装对照）**：
  - `artifacts.local/work/shallow-alerts-20261118/`（含 `comparison-camera-valid.json`、`areas-camera-valid.json`、`cache/`、`controls/oxford_valid_camera.json`）
  - 源码参考：`research/active/hardware-bringup/host/shallow_*.py`、`research/active/hardware-bringup/SHALLOW_ALERTS_DEV_20261118.md`
- **头朝向 vs 行进方向（LaMAR + Nymeria 头部 IMU + 实机 18 段 BMI270/RGB/CNH）**：
  - `artifacts.local/work/lamar-independent-heading-dev-20261116-codex/`（`hololens`, `ios`, `navvis`, `runtime`）
  - `artifacts.local/work/lamar-miss-breakdown-dev-20261117/`（`breakdown.json`, `direction_offset_travel.json`）
  - `artifacts.local/work/cnh-nymeria-imu-causal-dev-20261008/`（`raw_right_60s.npz`, `cpf_attitude_60s.npz`）
  - `artifacts.local/work/phone-new-capture-20261012/`（18 段头戴实采：6,746 RGB + 1,818 H3 CNH + 36,114 BMI270 IMU）
- **真实 H3 有符号 8×8×16 CNH 录像（1,209 帧 + 十条件）**：
  - `artifacts.local/hardware-bringup/captures/simple-20260927T164234Z-cc4d0e/`（柜面初始/放物/移动/恢复四段）
  - `artifacts.local/work/real-pilot-calibration-replay-dev-20261011/relaxed-pilot/`（十条件）
  - 复现工具：`research/active/egogeom-omni-v0/math_theory/m1_poisson_cnh_deconv/signed_h3_replay.py`

### C. 上轮冻结核心基线（V5 必须逐项对照）
- **M3 清路预算告警天花板**（`eval_alert_operating_point.py`，38 来源 / 5,547 有效帧 / 365 可避让组合 / 5.799 walker-km）：
  - 四域单阈值事后最优上限（`<= 5 次清路重复提示/km`）：原始风险 **`6/365`**，三帧确认风险 **`13/365`**（其中 Aria 上限 **`0/143`**！）；
  - 双折交叉拟合设备族成本约束方案：**`70/365`**（原评分 `1.03 FA/km`，清路重复提示 `5.17 次/km`，Aria 两折关闭告警）；
  - 单帧 95% 距离下界 `<= 2 m`：`349/365`，但清路帧告警占比 **`82.42%`**（`400.40 次/km`）。
- **M4 远距动态可观性基线**（CODa 4.10–5.10 m 重叠带，1,030 动态专属格帧 / 5,027 静态专属格帧）：
  - 静态重投影残差区分 AUROC：KLT **`0.5068`**，DIS **`0.5252`**；
  - 强制精确静态对应仅替换两帧 `v4` 预测深度：KLT **`19,689 / 28,050`（70.2%）**、DIS **`107,603 / 153,952`（69.9%）** 越过 `0.75 m/s` 假速度阈值。
- **M2 跨眼镜标量尺度基线**（严格因果航向）：
  - `v4` 原版 BEV MAE / Hybrid AUROC：Aria **`0.607 m / 0.7138`**，CODa **`1.188 m / 0.7321`**，ARKit 日 **`0.509 m / 0.7360`**，ARKit 暗 **`0.556 m / 0.7229`**；
  - `v4.1` 纯 RGB BEV v2.3 MAE / Hybrid AUROC：Aria **`0.637 m / 0.7060`**（v1 BEV `0.599 m`），CODa **`1.003 m / 0.7284`**，ARKit 日 **`0.510 m / 0.7368`**，ARKit 暗 **`0.559 m / 0.7283`**；
  - Aria 840 帧标量尺度神谕诊断：即便每帧给最优标量 $s_t \in [0.4, 2.5]$，仍有 **`405/840`** 帧误差 `> 0.55 m`（且 `13/840` 帧数学上不可能命中）。
- **S24 Ultra `v4.1` 硅片基线**：
  - `Pure-RGB` FP16：`107.72 MB`，HTP **`16.51 ms`**（NetRun `17.18 ms`），相对 FP32 误差 **`0.7881%`（`33.97 mm`）**；
  - `Unified Omni + Signed-H3` FP16：`109.38 MB`，HTP **`19.11 ms`**（NetRun `20.69 ms`），相对 FP32 误差 **`0.6008%`（`27.67 mm`）**，零干扰 **`0.000000 m`**。

---

# 规格书 M6：重尾清路极值下的群不变序贯检验鞅（E-Process）与功能型 Neyman–Pearson 风险泛函

> 任务目录：`research/active/egogeom-omni-v0/math_theory/m6_evt_eprocess_risk/`

## 1. 精确数学问题定义

现行 [EgoGeomTemporalBEV](egogeom_temporal_bev.py) 仅把深度 `depth_s4` 与 `log_sigma_s4` 聚合成走廊距离箱内的标量占用密度 $s_t(z)$，然后用 $R_t = \max_z \frac{1 - e^{-s_t(z)/\tau}}{z + 0.15} + \dots$ 输出标量风险，**完全未利用已推理出的 `delta_h_s4` 与 `normal_grav_s4`**。在 [ALERT_OPERATING_POINT.md](math_theory/m3_online_conformal_safety/ALERT_OPERATING_POINT.md) 中已证明：受限于清路帧风险峰值次序统计量 $b_{(k+1)}$，任何固定单阈值在 `<= 5 次清路重复提示/km` 预算下的及时告警上限仅为 **`13/365`（Aria `0/143`）**。

将第 $t$ 帧走廊内的多通道观测建模为定义在三维体素/极坐标网格 $\Omega \subset \mathbb{R}^3$ 上的**时空标记随机场 / 标记点过程**：
$$\mathcal{Y}_t = \big\{(p_i, \hat Z_i, \widehat{\delta h}_i, \hat{\mathbf{n}}_i, \hat\sigma_i)\big\}_{i=1}^{N_t}$$
考虑复合零假设（清路流形）与备择假设（存在实体障碍）：
- **$H_0$（未知干扰群作用下的清路）**：存在未知干扰群元素 $g_t = (s_t, \delta\theta_t, \gamma_t) \in \mathcal{G} = \mathbb{R}_{>0} \times \mathrm{SO}(2)_{\text{pitch}} \times \mathbb{R}_{>0}$（设备尺度漂移、局部地面坡度/俯仰偏置、环境光噪比），使得观测点云来自光滑可通行地面 + 通道外侧墙的噪声投影，且残差场具有空间块相关性（correlation length $L_c$）与时间混合性（$\beta$-mixing）。
- **$H_1$（通道内实体障碍）**：存在紧支集实体表面 $\mathcal{K} \subset \mathcal{C}_{\text{body}}$ 满足：
  1. 局部高差与世界高度一致脱离地面流形；
  2. 局部表面法向呈现**非地面切空间正交分量**（$\|\hat{\mathbf{n}}_\perp\|^2 = 1 - (\hat{\mathbf{n}}\cdot \mathbf{e}_g)^2$ 显著聚集，区别于把地面翘起产生的假高度！）；
  3. 跨帧在世界系刚体位姿传播下保持空间相干性（而非单帧重尾噪声尖峰）。

清路误报代价按真实步距更新过程计费：每进入新清路段或距上次清路告警行走超过 $L_{\text{ref}} = 2 v_w\text{ m}$ 即记 1 次清路重复提示，要求每步行 1 km 的期望清路重复提示率 $\Lambda_{\text{clear}} \le \lambda_{\max} = 5\text{ km}^{-1}$。

## 2. 必须证明的核心定理与推导任务

1. **定理 M6.1（乘性尺度噪声下的极值尾指数相变：为何现行 BEV 必然卡死在重尾极值）**：
   - 令单帧深度受到乘性尺度扰动 $\hat Z = S \cdot Z$，其中 $S^{-1}$ 具有幂律右尾（或自由度 $\nu$ 的学生 $t$/对数正态厚尾）或走廊边界发生阶跃跳变；
   - 计算现行 BEV 极大池化统计量 $R_t^{\text{BEV}} = \max_z \psi(s_t(z), z)$ 在清路 $H_0$ 下的**广义极值分布（GEV）吸引域与 Pickands–Balkema–de Haan 尾指数 $\xi$**，并与走廊总长度 $K$（km）下的清路最大值 $M_K = \max_{t \in \text{clear}(K)} R_t$ 增长阶关联；
   - 精确刻画：为何当 $\xi > 0$（或清路存在异域双峰混合尖峰）时，即使样本量增大，任何单调标量阈值的清晰分离上限仍塌缩（直接给出 Aria `0/143` 现象的数学机理与定量诊断）。
2. **定理 M6.2（干扰群 $\mathcal{G}$ 下的 Hunt–Stein 极大不变风险泛函与轻尾保证）**：
   - 构造利用 `(depth_s4, dh_s4, nrm_s4, ls_s4)` 四通道的**尺度-俯仰不变局部几何检验统计量** $T_t(\mathcal{Y}_t)$（例如：将“竖直/非地面法向支持”、“局部地面自洽残差 $\widehat{\delta h}$ 与世界高度的互证”、以及“多帧世界系空间相干密度”结合成自归一化 / Studentized / 秩-对数似然比形式，显式压制单纯把地面算高的假高度和孤立通道边缘漏入）；
   - 证明在干扰群 $\mathcal{G}$ 的尺度/全局偏置变换下 $T_t$ 的不变性（或渐近不变性），并证明在有界/自归一化设计下其清路尾指数严格满足 **$\xi \le 0$（次高斯 / Gumbel / 有界 Weibull 尾）**，从而使 $K$ km 内的清路最大值增速从多项式 $K^{\xi}$ 降至至多 $O(\sqrt{\log K})$ 或 $O(1)$！
3. **定理 M6.3（面向按公里更新计费的 Anytime-Valid E-Process 序贯告警控制器）**：
   - 针对带步距不应期 $L_{\text{ref}}$ 的清路计费更新过程，构造非负超鞅（**E-process**）$\{E_t\}_{t \ge 0}$：在每次告警触发或清路分段重置后，令 $\mathbb{E}_{H_0}[E_t \mid \mathcal{F}_{t-1}] \le E_{t-1}$；
   - 利用 **Ville 不等式与更新回报定理（Renewal-Reward Theorem）**，严格证明：只要步进对数似然/自归一化 e-value 满足上界条件，无论轨迹多长、处于何种未知设备域，每公里期望清路重复提示率满足显式解析上界 $\mathbb{E}_{H_0}[\text{ClearWarn/km}] \le \Lambda(\alpha, L_{\text{ref}})$（且写清在序列存在时间自相关时的精确修正因子——如块独立/马尔可夫混合惩罚项，不允许偷换 i.i.d. 假设！）。
4. **算法 M6.4（`< 1.0 ms` 纯 NumPy 因果风险与 E-Process 告警引擎）**：
   - 严格因果、无需知道当前属于 Aria / CODa / ARKit 哪个设备族；
   - 完整实现 `InvariantEProcessBEV.step(depth_s4, dh_s4, nrm_s4, ls_s4, K, R_wc, t_w, strict_fwd)`。

## 3. 验证基线与验收门槛

- **符号与数值验证（`verify_theory.py`）**：
  - SymPy 验证不变性变换与超鞅条件 $\mathbb{E}_{H_0}[e_t \mid \mathcal{F}_{t-1}] \le 1$；
  - 合成空间+时间相关清路随机场（含重尾尺度突发 + 坡度突变）跑 `500` 条 10 km 轨迹：对比现行 `max_z` BEV 与新不变统计量的 Hill 尾指数估计 $\hat\xi$、以及 E-Process 实测清路重复提示/km 是否严格受控于理论界内（并展示时间相关性超出预设混合界时的反例失控边界）。
- **四域冻结实证（`eval_empirical.py`，完全复用 `eval_alert_operating_point.py` 的 38 来源双折交叉拟合框架与严格因果 `strict_fwd`）**：
  - 首先在冻结 `v4_stream` 上复现基线（单阈值上限 `13/365`、双折共用阈值 `6/365`、已知设备族 `70/365`）；
  - 评估新不变风险泛函 + E-Process 的**事后清路峰值上限**与**双折留出跨设备统一规则**成绩（同步在 `v4.1` 流上复验）；
  - **验收门槛**：
    1. 跨设备单阈值/统一规则下，四域事后清路峰值可达上限从 **`13/365` 提升至 `>= 90/365`**，且 **Aria 上限严格打破 `0/143`（`>= 35/143`）**；
    2. 严格双折来源留出（**不读取设备族标签**、全设备共用一套参数！）在清路重复提示 `<= 5.0 次/km`（或原评分 `<= 5.0 FA/km`）下，及时告警较基线共用阈值 `6/365` 至少翻倍以上提升（冲击超越已知设备族基线 `70/365`）；
    3. 单帧 CPU 耗时 `< 1.2 ms`（单线程 NumPy）。

---

# 规格书 M7：无深度作差的直接光度/特征扩张（Optical Looming）与法向耦合微分散度反演

> 任务目录：`research/active/egogeom-omni-v0/math_theory/m7_direct_looming_normal_ttc/`

## 1. 精确数学问题定义

[M4 第二轮](math_theory/m4_stochastic_cbf_ics/far_motion/RESULTS.md) 证明了两个核心物理事实：
1. 对连续两帧神经网络预测深度直接作差 $(D_t - D_{t-1})/\Delta t$，即便在**精确静态像素对应**上，仍有 **70.0%** 的静止点产生 $\ge 0.75\text{ m/s}$ 假速度；
2. 对于沿视线径向正迎面接近的高速目标（6 m/s 电动车/骑行者），其运动完全落在极线内部，纯极线重投影残差在 CODa 4.10–5.10 m 的动态/静态区分 AUROC 仅 **`0.5068 / 0.5252`**。

考虑图像归一化平面坐标 $\mathbf{x} = (x, y)^\top = (X/Z, Y/Z)^\top$ 上的一个局部刚性小面片，具有世界系绝对速度 $\mathbf{V}_{\text{obj}} \in \mathbb{R}^3$、当前深度 $Z > 0$ 与相机系单位表面法向 $\mathbf{n} = (n_x, n_y, n_z)^\top$（满足局部平面方程 $\frac{1}{Z(\mathbf{x})} = \frac{1}{Z_0} + \mathbf{p}^\top (\mathbf{x} - \mathbf{x}_0)$，其中 $\mathbf{p} = -\frac{(n_x, n_y)^\top}{n_z Z_0}$ 为表面倾斜梯度）。相机已知自身瞬时刚体运动 $(\mathbf{v}_{\text{ego}}, \boldsymbol{\omega}_{\text{ego}})$，相对平移速度为 $\mathbf{v}_{\text{rel}} = \mathbf{v}_{\text{ego}} - \mathbf{V}_{\text{obj}} = (v_x, v_y, v_z)^\top$。

连续运动场方程（Longuet-Higgins & Prazdny）：
$$\mathbf{u}(\mathbf{x}) = \mathbf{u}_{\text{rot}}(\mathbf{x}; \boldsymbol{\omega}_{\text{ego}}) + \frac{1}{Z(\mathbf{x})} \begin{pmatrix} -v_x + x v_z \\ -v_y + y v_z \end{pmatrix}$$
去旋后的纯平移运动场 $\tilde{\mathbf{u}}(\mathbf{x}) = \mathbf{u}(\mathbf{x}) - \mathbf{u}_{\text{rot}}(\mathbf{x})$ 的 $2\times 2$ 空间雅可比矩阵 $J_{\tilde{\mathbf{u}}}(\mathbf{x}_0) = \nabla_{\mathbf{x}}\tilde{\mathbf{u}}(\mathbf{x}_0)$ 可正交分解为四个微分不变量：
- **散度（Divergence / Looming）**：$D = \partial_x \tilde u_x + \partial_y \tilde u_y$；
- **旋度（Curl / Vorticity）**：$C = \partial_x \tilde u_y - \partial_y \tilde u_x$；
- **两个剪切分量（Deformation / Shear）**：$S_1 = \partial_x \tilde u_x - \partial_y \tilde u_y$，$S_2 = \partial_x \tilde u_y + \partial_y \tilde u_x$。

## 2. 必须证明的核心定理与推导任务

1. **定理 M7.1（法向耦合的微分运动场解耦定理：消除“静态斜坡 vs 正迎面接近”歧义）**：
   - 精确推导出 $(D, C, S_1, S_2)$ 关于相对逼近率 $\tau_{\text{rel}}^{-1} \equiv v_z / Z_0$、横向角速度 $\boldsymbol{\alpha}_\perp = (v_x/Z_0, v_y/Z_0)^\top$ 与表面斜率 $\mathbf{p} = -\frac{\mathbf{n}_\perp}{n_z Z_0}$ 的精确闭式表达式；
   - 严格证明：仅凭 2D 光流散度 $D$ 时，**静止斜墙**（$\mathbf{p} \neq \mathbf{0}, \mathbf{V}_{\text{obj}} = \mathbf{0}$）与**正迎面移动平面**（$\mathbf{p} = \mathbf{0}, V_{\text{obj},z} < 0$）存在不可消去的代数歧义；
   - **关键突破证明**：引入 `v4/v4.1` 输出的当前单帧表面法向 $\hat{\mathbf{n}}$（**完全不做跨帧深度相减！**）与已知自运动 $\mathbf{v}_{\text{ego}}$，证明可将斜率耦合项 $\mathbf{p}^\top \tilde{\mathbf{u}}_0$ 解析消去，得到**零深度作差**的正迎面目标相对逼近率估计量：
     $$\widehat{\Delta\kappa}(\mathbf{x}_0) \equiv \hat\kappa_{\text{obs}}(\mathbf{x}_0) - \hat\kappa_{\text{static}}(\mathbf{x}_0; \mathbf{v}_{\text{ego}}, \hat Z_t, \hat{\mathbf{n}}_t)$$
     并证明当单帧深度存在未知全局乘性尺度 $s$ 时，静态比值不变量 $\kappa_{\text{obs}} / \kappa_{\text{static}}$ 与全图多面片联立比值如何消去 $s$（同时给出自运动 $\mathbf{v}_{\text{ego}} = \mathbf{0}$ 原地站立时的精确退化特例——此时 $\kappa_{\text{static}} \equiv 0$，**任何显著正散度直接证明有迎面逼近目标！**）。
2. **定理 M7.2（直接光度尺度扩张 PDE 与正迎面高速探测的 Cramér–Rao 信息论相变极限）**：
   - 不依赖稠密光流二阶数值求导，直接在图像/特征金字塔亮度恒常方程 $\partial_t I + \tilde{\mathbf{u}}\cdot \nabla I = 0$ 中引入 6 参数局部仿射形变模型 $\tilde{\mathbf{u}}(\mathbf{x}) = \mathbf{t} + A(\mathbf{x} - \mathbf{x}_0)$（含尺度扩张参数 $\kappa$），推导在加性像素高斯噪声 $\sigma_I^2$、物体角宽 $\theta_{\text{obj}} \approx W_{\text{obj}}/Z$、采样间隔 $\Delta t$ 下的精确 **$6\times 6$ 光度 Fisher 信息矩阵与 $\mathrm{CRLB}(\kappa)$、$\mathrm{CRLB}(V_{\text{obj},z})$**；
   - **硬核物理边界**：将本项目真实参数代入（分辨率 `288×384`、水平 FOV $70^\circ$–$110^\circ$、典型骑行者宽 $W_{\text{obj}} = 0.6\text{ m}$、迎面速度 $V_{\text{obj},z} = -6.0\text{ m/s}$、真实图像梯度能量 $\|\nabla I\|^2$ 分布），绘制 $(Z \in [3, 12]\text{ m}, f_{\text{fps}} \in \{5, 10, 15, 30\}\text{ Hz})$ 的 **CRLB 探测极限相变图**：严格回答在当前 `5 Hz / 288×384` 条件下，8 m 外 6 m/s 正迎面目标在物理上究竟**可解还是不可解**，以及达到 95% 检出所需的最优帧率/多帧窗口长度 $K^*$（注意多帧累计下的线性速度 vs 透视非线性扩张权衡！）。
3. **算法 M7.3（多帧直接光度/仿射尺度扩张 GLRT 求解器）**：
   - 设计无需两帧深度相减的局部逆合成（Inverse Compositional）Gauss–Newton 仿射扩张与法向耦合 GLRT 算子，融合：
     (i) 横向运动项（传统极线/静态重投影残差，捕横穿目标）；
     (ii) 径向扩张项（法向去斜后的 Direct Looming GLRT，捕正迎面目标）；
   - 单帧 CPU 耗时目标 `< 3.0 ms`。

## 3. 验证基线与验收门槛

- **符号与合成刚体验证（`verify_theory.py`）**：
  - SymPy 严格验证 $J_{\tilde{\mathbf{u}}}$ 四分量解析解与法向去斜恒等式；
  - 合成 3D 渲染场景（静止斜墙 + 正迎面 6 m/s 纹理目标 + 侧穿目标，叠加 $\pm 15\%$ 帧间随机深度闪烁）：证明深度作差法假速度率 `> 60%`，而法向耦合 Direct Looming 法在静止斜墙上保持 `< 2%` 误报、对 6–9 m 正迎面 6 m/s 目标达到理论 CRLB 效率比 `<= 1.4`。
- **真实 CODa 4–10 m 实证（`eval_empirical.py`，完全复用 `far_motion/eval_far_motion.py` 的 1,030 动态 / 5,027 静态 4.10–5.10 m 重叠带与 7.86–10 m 诊断口径）**：
  - 同步报告带序列 13 与剔除序列 13 后的分序列统计；
  - **验收门槛**：
    1. 复现静态对应反例：在相同静态像素对应上，无深度作差的 Direct Looming 法静态误触发率较深度作差基线（`70.0%`）下降至少 **`5×`（`< 14%`）**；
    2. 在 CODa 4.10–5.10 m 重叠带，融合“极线残差 + 法向耦合 Looming GLRT”的连续分数区分动态/静态 AUROC 显著超越原 DIS 重投影基线 **`0.5252`**（或若 CRLB 定理证明 5 Hz / `96×72` 灰图信噪比低于相变界，则必须在全分辨率 `288×384` RGB / 更高特征尺度上对比验证并给出严格物理分辨率/帧率下界）。

---

# 规格书 M8：耦合人类反应延迟与 $Z^4$ 感知噪声的自由边界拟变分不等式（Free-Boundary QVI）最优告警停时

> 任务目录：`research/active/egogeom-omni-v0/math_theory/m8_free_boundary_qvi_stopping/`

## 1. 精确数学问题定义

[M3 第二轮](math_theory/m3_online_conformal_safety/ALERT_OPERATING_POINT.md) 揭示了一个深刻悖论：95% 共形距离下界虽然准确达成了 `95.06%` 覆盖率，但按“下界 $\le 2\text{ m}$ 即告警”却在 **`82.42%` 的清路帧触发告警**（`400.40 次/km`）。根本原因是静态决策忽视了**主动感知中的等待信息价值（Value of Information）**：当盲人处于较远距离且不确定度 $\sigma_D$ 较大时，继续行走 $\Delta t$ 会让单目测距噪声方差 $\sigma_Z^2(Z) \propto Z^4$ 剧烈收缩（或进入 ToF 4 m 量程），只要在此 $\Delta t$ 内跨入 M4 的不可避免碰撞集（ICS）的概率极小，**等待观察严格优于立即误报**。

建立连续时间部分可观信念空间最优停时 / 脉冲控制问题：
- **潜状态**：前方走廊最近障碍的真实距离 $D_t \in [0, \infty)$ 与相对逼近速度 $v_t \ge 0$（以及状态指示 $H \in \{0\text{ (清路)}, 1\text{ (障碍)}\}$），运动学 $dD_t = -v_t\, dt + \sigma_w\, dW_t$；
- **异方差距离相关观测**：
  $$dY_t = D_t\, dt + R^{1/2}(D_t)\, dB_t, \qquad R(D) = \left(c_{\text{mono}}^{-1} D^{-4} + \mathbf{1}_{\{D \le 4\text{ m}\}} c_{\text{tof}}^{-1}\right)^{-1}$$
- **信念充分统计量**：后验存在障碍概率 $p_t = \mathbb{P}(H=1 \mid \mathcal{F}_t)$、条件距离后验均值与方差 $(\hat D_t, \Sigma_t)$，由 Kushner–Stratonovich / 连续时间扩展 Riccati SDE 驱动（其中信息增益率 $\dot\Sigma_t = \sigma_w^2 - \Sigma_t^2 / R(\hat D_t)$）；
- **人机耦合脉冲控制代价**：
  - 立即告警（停时 $\tau_a$）：付出固定听觉/认知打扰成本 $c_{\text{beep}} > 0$（清路时全额计入误报代价；真障碍时进入延迟 $\tau_h = 0.65\text{ s}$ 后的残余碰撞期望损失 $L_{\text{coll}}(\hat D_t, \Sigma_t, v_t; \tau_h) = C_{\text{hit}}\,\mathbb{P}(D_{t+\tau_h} \le d_{\text{esc}}(v_t) \mid \mathcal{F}_t)$）；
  - 告警冷却状态（或重复提示不应期 $\gamma_t$）：刻画连续清路狂报导致的惩罚递增与盲人信任/反应延迟退化 $\tau_h(\beta_t)$。

值函数 $V(p, \hat D, \Sigma)$ 满足**自由边界拟变分不等式（Bensoussan–Lions QVI）**：
$$\max\Big\{\mathcal{L}_{\text{KS}} V(p, \hat D, \Sigma) - r V,\; \mathcal{M} V(p, \hat D, \Sigma) - V(p, \hat D, \Sigma)\Big\} = 0$$
其中 $\mathcal{L}_{\text{KS}}$ 为信念 SDE 的无穷小生成元，$\mathcal{M} V$ 为发出告警脉冲后的立即代价与重置后值函数。

## 2. 必须证明的核心定理与推导任务

1. **定理 M8.1（QVI 粘性解存在唯一性与“等待-观察 vs 立即告警”自由边界几何结构定理）**：
   - 写出 $(p, \hat D, \Sigma)$ 三维（及固定 $v$ 下二维 $(\hat D, \Sigma)$ 切片）的精确无穷小生成元 $\mathcal{L}_{\text{KS}}$ 与障碍函数 $g(p, \hat D, \Sigma)$；
   - 证明在单调损失假设下，连续区域 $\mathcal{C} = \{V > \mathcal{M}V\}$（继续等待）与触发区域 $\mathcal{S} = \{V = \mathcal{M}V\}$（立即告警）由一条光滑自由边界 $\hat D = \Gamma^*(p, \Sigma, v)$ 分隔；
   - **反直觉核心定理（证明为什么共形下界必然失败！）**：严格证明在 $R(D) \propto D^4$ 时，**自由边界 $\Gamma^*(p, \Sigma)$ 对不确定度 $\Sigma$ 呈现严格非单调的双区形态**：
     - **近场 ICS 保护区**（$\hat D \to D_{\text{ICS}}(v, \tau_h)$）：随着 $\Sigma$ 增大，$\Gamma^*$ 向外扩以守住 Smooth-Pasting 碰撞概率底线，推导边界层渐近展开（精确指数与前系数）；
     - **远场高不确定度抑制区**（高 $\Sigma$ 且 $p < 1$，恰好对应暗光/远场噪声清路）：由于等待 $\Delta t$ 的信息熵减率 $\propto \Sigma^2 / R(\hat D)$ 随接近而按 $D^{-4}$ 飙升，**继续等待的实物期权价值（Real-Option Value）严格压过立即告警收益**，导致临界置信概率要求 $p^*(\hat D, \Sigma)$ 急剧抬高，从而在理论上彻底消除共形下界的远场清路误报爆炸！
2. **定理 M8.2（脉冲冷却与告警疲劳耦合下的最优不应期策略）**：
   - 将清路重复计费周期 $L_{\text{ref}}$ 纳入脉冲算子 $\mathcal{M}$，证明在给定每公里最大清路告警预算 $\Lambda_{\max} = 5\text{ km}^{-1}$ 时，通过拉格朗日乘子 $c_{\text{beep}} = \lambda^*$ 得到的 QVI 策略是约束 POMDP 的全局 Pareto 最优解（Neyman–Pearson 脉冲控制推广）。
3. **算法 M8.3（2D/3D 迎风有限差分参考 QVI 求解器 + `< 0.1 ms` 闭式解析近似控制器 `QVIAlertPolicy`）**：
   - 先用政策迭代 / Howard 算法在 $(p, \hat D, \Sigma)$ 网格上数值求解精确 QVI 值函数与自由边界 $\Gamma^*$；
   - 拟合/推导无查表闭式解析边界，对接 `v4` / `v4.1`（或 M6 新风险）的逐帧预测均值与不确定度。

## 3. 验证基线与验收门槛

- **符号与 PDE/SDE 验证（`verify_theory.py`）**：
  - SymPy 验证生成元 $\mathcal{L}_{\text{KS}}$、Riccati 方程与 Smooth-Pasting 条件；
  - 在 `20,000` 条含真实 $Z^4$ 异方差噪声 + 突现障碍的合成行走轨迹上对比：① 固定距离阈值；② 95% 共形距离下界；③ 精确 QVI 自由边界策略；④ 解析近似 `QVIAlertPolicy`——验证在**相同碰撞率**下 QVI 将清路告警频率压降 **`>= 60%`**，且展示解析边界与数值 QVI 边界的最大间隙。
- **四域冻结流实证（`eval_empirical.py`，完全复用 M3 第二轮 38 来源双折留出框架）**：
  - 输入相同冻结 `v4`（及可选 M6）特征，对比原 `fixed_range_95_2m`（`82.42%` 清路告警）、`bev_distance_2m`（`135.88 次/km`）与 `confirmed_cost`（`6/365` 共用、`70/365` 设备族）；
  - **验收门槛**：
    1. 相比 `fixed_range_95_2m`，在接近或保持同等及时保护的同时，把清路告警占比从 **`82.42%`** 大幅压低并显式给出完整的（及时率 vs 清路重复提示/km）Pareto 曲线；
    2. 结合固定清路预算 `<= 5 次/km`，明确分离“纯策略停时增益（在旧特征上能涨多少）”与“特征排序天花板（M3 峰值界是否仍构成硬约束）”，做到零混肴。

---

# 规格书 M9：$\mathrm{PGL}(4)$ 射影规范在线校准与法向-深度 Frobenius 可积性旋度不变量

> 任务目录：`research/active/egogeom-omni-v0/math_theory/m9_pgl4_frobenius_integrability/`

## 1. 精确数学问题定义

[M2 第二轮](math_theory/m2_lie_group_scale_iekf/rgb_triangulation/RESULTS.md) 证明了一个关键限制：在 Aria Gen 2 上，即使给每帧看真值的**标量尺度神谕** $s_t \in [0.4, 2.5]$，仍有 **`405/840`** 帧走廊测距误差超过 `0.55 m`（且 `13/840` 帧在连续标量区间上被证明绝对不可达）。同时，仅用局部标量缩放虽将 Aria BEV MAE 降至 `0.492 m`，却使 CODa 2 m 误报暴增至 `79.2 次/km`。

根本几何原因在于：单目深度网络跨未知相机/俯仰/畸变时的系统误差不是单标量缩放，而是四维**射影基浮雕规范群（Projective Bas-Relief / Affine-in-Disparity Subgroup of $\mathrm{PGL}(4)$）**变换：
$$d^*(u, v) \equiv \frac{1}{Z^*(u, v)} = a \cdot \hat d(u, v) + b_0 + b_x \bar u + b_y \bar v, \qquad \boldsymbol{\psi} = (a, b_0, b_x, b_y)^\top \in \mathbb{R}_{>0}\times \mathbb{R}^3$$
其中 $(\bar u, \bar v) = ((u-c_x)/f_x, (v-c_y)/f_y)$ 为归一化成像平面坐标。注意：$b_0 \neq 0$ 造成远近非线性压缩，$(b_x, b_y) \neq \mathbf{0}$ 对应把平坦地面扭成上坡假墙（直接导致 CODa 误报爆炸！）。

与此同时，`EgoGeom-Omni v4/v4.1` 逐像素输出了**独立于全局尺度的稠密单位表面法向** $\hat{\mathbf{n}}(u, v)$。对于任意光滑三维曲面 $\mathbf{X}(\bar u, \bar v) = Z(\bar u, \bar v)\,\mathbf{r}(\bar u, \bar v)$（其中 $\mathbf{r} = (\bar u, \bar v, 1)^\top$），切向量 $\partial_{\bar u}\mathbf{X}, \partial_{\bar v}\mathbf{X}$ 必正交于 $\mathbf{n}$，从而深度与法向必须满足一阶偏微分系统（**Frobenius 可积性系统**）：
$$\nabla_{\bar u, \bar v} \log Z = \boldsymbol{\omega}(\mathbf{n}, \bar u, \bar v) \equiv -\frac{1}{\mathbf{n}\cdot \mathbf{r}}\begin{pmatrix} n_x \\ n_y \end{pmatrix}$$

## 2. 必须证明的核心定理与推导任务

1. **定理 M9.1（无尺度 Frobenius 旋度不变量与神经网络几何幻觉检测定理）**：
   - 推导二维外微分 $d\boldsymbol{\omega} = \left(\partial_{\bar u} \omega_y - \partial_{\bar v} \omega_x\right) d\bar u \wedge d\bar v$ 在像素离散网格上的精确环路积分形式（离散 Stokes 环量 $\oint_{\partial \square} \boldsymbol{\omega}\cdot d\mathbf{l}$）；
   - 严格证明：在任意无遮挡光滑 3D 曲面上，$d\boldsymbol{\omega} \equiv 0$ **严格独立于全局深度尺度 $s$ 与视差偏置 $b_0$**；反之，若预测深度与预测法向不一致（例如玻璃透射、暗光塌缩、地平线幻觉墙），则 **深度-法向兼容性残差**
     $$\mathbf{r}_{\text{FN}}(\boldsymbol{\psi}) = \nabla \log\!\big(a\hat d + b_0 + b_x \bar u + b_y \bar v\big) + \boldsymbol{\omega}(\hat{\mathbf{n}})$$
     与旋度 $|d\boldsymbol{\omega}|$ 严格为正！
   - 推导出**用 $\hat{\mathbf{n}}$ 直接解耦 $(b_x, b_y)$ 与 $b_0/a$ 的线性化闭式方程**：证明仅凭单帧 $\hat d$ 与 $\hat{\mathbf{n}}$（**无需任何外参平移！**）即可辨识射影畸变比 $(b_0/a, b_x/a, b_y/a)$ 的充要条件（需要场景内至少存在 2 个不同深度平面或非平行法向），从而把未知自由度从 4 维降回仅剩 1 维全局公制尺度 $a$！
2. **定理 M9.2（$\mathrm{PGL}(4)$ 4 参数神谕下界 vs M2 标量神谕下界的精确比较）**：
   - 在 Aria 840 帧上建立与 M2 `oracle.py` 完全可比的 4 参数射影规范神谕诊断（检查哪些原本在标量 $s$ 下绝对不可达的 13 帧与 405 帧超差帧是因为 $b_0$ 或 $(b_x, b_y)$ 射影漂移所致，哪些属于网络局部拓扑漏检，划清几何校准的真正天花板！）。
3. **算法 M9.3（`< 0.8 ms` 因果 Frobenius-$\mathrm{PGL}(4)$ 联合滤波器 + 幻觉掩码加权）**：
   - 结合：① 单帧 Frobenius 深度-法向约束（锁死地面平面倾斜 $(b_x, b_y)$ 与相对偏移 $b_0/a$，绝不把 CODa 路面扭翘！）；② 稀疏公制三角化或重力地面高度先验（仅更新全局公制尺度 $a_t$）；③ 基于 Frobenius 旋度 $|d\boldsymbol{\omega}|$ 与深度-法向不一致度的**反幻觉置信权重**，输入 `EgoGeomTemporalBEV`。
   - 严格因果、零重训、四域共享一套超参！

## 3. 验证基线与验收门槛

- **符号与合成验证（`verify_theory.py`）**：
  - SymPy 符号验证 Frobenius 偏微分方程、旋度在球面/平面/圆柱上的恒零性，以及射影变形 $d^* = a\hat d + b_0 + b_x\bar u + b_y\bar v$ 下由 $(\hat d, \hat{\mathbf{n}})$ 反演 $(b_0/a, b_x/a, b_y/a)$ 的雅可比满秩条件与退化反例（如全图仅单一平行平面时不可辨识）；
  - 合成射影畸变深度场（$a \in [0.5, 2.0], b_0 \in [-0.15, 0.15], |b_x|, |b_y| \le 0.12$）上验证 4 参数恢复精度 `< 2%`。
- **四域跨眼镜实证（`eval_empirical.py`，对比 M2 全部四臂与 `v4` / `v4.1`）**：
  - **验收门槛**：
    1. 报告 Aria 840 帧上 4 参数射影神谕 vs 标量神谕（`0.797 m` 固定分母 / `405` 超差帧）的严格定量拆解；
    2. 因果 Frobenius-$\mathrm{PGL}(4)$ 校准在四域全开同一超参下：**显著改善 Aria BEV 测距（固定分母 MAE 与共同支持 MAE 同步优于原 `v4` `1.574 / 0.609 m`），同时彻底杜绝 M2 局部臂在 CODa 上造成的误报爆炸（CODa 固定 2 m 误报不劣于原 `v4` `26.1 次/km`，BEV MAE 优于或持平原版）**；
    3. 滤波与 Frobenius 诊断核心耗时 `< 0.8 ms/帧`（不含光流，若复用灰度匹配则总耗时需单独拆分报告）。

---

# 规格书 M10：$SBV(\Omega)$ 自由不连续空间上的 $\mathrm{TGV}^2$ + 法向跳跃浅障分解（破解 5–15 cm 挡轮块死结）

> 任务目录：`research/active/egogeom-omni-v0/math_theory/m10_sbv_tgv2_shallow_manifold/`

## 1. 精确数学问题定义

[浅障逐次比较](../hardware-bringup/SHALLOW_ALERTS_DEV_20261118.md) 在真实停车场挡轮块与苏黎世/牛津铺装路上暴露了一个致命结构反转：在剔除鱼眼黑边后的有效视野内，空铺装路面（Christ Church 第 3400 帧）的局部高差支持面积高达 **`0.09846 m²`**，直接压过 5 次真实浅障接近中的最强目标（**`0.05818 m²`**），导致同误报配平下所有 0 阶高度法（降门槛、单一地面、局部高差）全部找回 **`0/5`**！

原因在于：真实户外路面是具有非零曲率（排水拱度、缓坡过渡 $\kappa \sim 0.02\text{–}0.05\text{ m}^{-1}$）且叠加单目低频深度弯曲的 **$C^{1,1}$ 光滑二维流形**；而 5–15 cm 挡轮块、路缘石与台阶下沿，是叠加在光滑曲面上的**紧支集 $SBV(\Omega)$ 跳跃不连续体**——其在横截面上同时表现为**高度阶跃（一维可求长冲击曲线 $\Gamma$）与表面法向偶极子（垂直端面法向翻转）**。

在重力对齐局部步行地面坐标 $(x, z) \in \Omega \subset [-1.5, 1.5] \times [0.3, 3.0]\text{ m}$ 上，给定观测高度场 $h_{\text{obs}}(x, z)$（带有效视野与置信权 $w(x, z)$）及表面法向投影坡度场 $\mathbf{g}_{\text{obs}}(x, z) = -(n_x, n_z)^\top / n_y$，建立**二阶总广义变分（$\mathrm{TGV}^2$）+ 稀疏跳跃分量凸变分分解**：
$$\min_{u, \mathbf{v}, j} \; \frac{1}{2}\big\|w^{1/2}(u + j - h_{\text{obs}})\big\|_2^2 + \alpha_1 \|\nabla u - \mathbf{v}\|_{2,1} + \alpha_0 \|\mathcal{E}\mathbf{v}\|_{F,1} + \beta \|j\|_1 + \gamma \|\nabla j\|_{2,1} + \eta \big\|\mathbf{v} - \mathbf{g}_{\text{obs}}\big\|_{\text{Huber}}$$
其中 $\mathcal{E}\mathbf{v} = \frac{1}{2}(\nabla \mathbf{v} + \nabla\mathbf{v}^\top)$ 为对称梯度（刻画地面曲率！），$u$ 吸收任意光滑坡面与缓弯路面，$j$ 专门提取紧支集凸起（挡轮块/路缘）与凹陷（台阶下沿/坑洞）。

## 2. 必须证明的核心定理与推导任务

1. **定理 M10.1（二次弯曲铺装零响应定理与“曲率-台阶”精确分离相变条件）**：
   - 利用对偶证书（Dual Calibration）与次微分最优性条件，严格证明：**对任意一阶斜坡 $h(x,z) = c_0 + c_1 x + c_2 z$，当 $\beta > 0$ 时 $j^* \equiv 0$ 严格零响应**；进一步，对任意光滑二次拱形路面 $h(x,z) = \frac{1}{2}\mathbf{x}^\top Q \mathbf{x}$（主曲率 $\|Q\|_F \le \kappa_{\max}$），推导保证 $j^* \equiv 0$ 的显式参数门槛 $\alpha_0(\kappa_{\max}), \beta(\kappa_{\max})$；
   - 反之，对宽度 $W \in [0.10, 0.30]\text{ m}$、高度 $H \in [0.05, 0.15]\text{ m}$ 的刚体挡轮块/单级路缘，推导严格恢复 $j^* > 0$ 所需的精确信噪比与尺度条件（并计算引入法向偶极子耦合项 $\eta$ 后，最小可分辨台阶高度 $H_{\min}$ 相比纯高度法的理论下降倍率！）。
2. **定理 M10.2（Chambolle–Pock 原始-对偶分裂收敛率与 `< 0.5 ms` 网格求解器）**：
   - 推导该复合非光滑凸优化的精确 Chambolle–Pock 原始-对偶迭代算子（闭式 prox 算子、算子范数上界 $\|K\|_2 \le \sqrt{12}$ 保证步长 $\tau\sigma \|K\|_2^2 < 1$ 严格收敛）；
   - 利用本项目有效视野的 $5\text{ cm}$ 地面栅格（仅约 $60 \times 54$ 网格！），写出无分支 NumPy/Numba 展开求解器，单帧耗时 `< 0.5 ms`。
3. **物理归因诊断（面向 Christ Church 第 3400 帧反例）**：
   - 直接在 `artifacts.local/work/shallow-alerts-20261118/` 缓存上拆解：Christ Church 第 3400 帧的 `0.09846 m²` 假高差在 $(u, \mathbf{v}, j)$ 分解下究竟落在平滑弯曲分量 $u$（曲率/单目弯曲）还是跳跃分量 $j$（纹理边缘深度伪影）——给出可复现的物理归因！

## 3. 验证基线与验收门槛

- **符号与合成验证（`verify_theory.py`）**：
  - SymPy 验证对称梯度伴随算子 $\mathcal{E}^* = -\mathrm{div}$ 与线性/二次曲面上的对偶证书解析解；
  - 合成“6° 斜坡 + 12 cm 拱度弯曲铺装 + 8 cm 挡轮块 + 10 cm 下台阶 + 单目低频深度弯曲”压力测试，证明 0 阶局部高差在拱形空路上假报 `100%`，而 $\mathrm{TGV}^2$ 分解在空拱路保持 `0%` 假报并精确还原 8 cm 挡轮块与 -10 cm 下台阶。
- **真实四地点浅障配对实证（`eval_empirical.py`，严格沿用 `SHALLOW_ALERTS_DEV_20261118.md` 的 `--camera-valid` 有效视野、空路配平与按地点留一协议）**：
  - 同步运行在原缓存深度上（公平对比 `0/5` 基线）以及用 `v4.1` 预测深度 + 法向重算的对照臂（明确区分“算法增益”与“换底座增益”）；
  - **验收门槛**：
    1. 精确复现原基线（修正后有效视野下直接降门槛/单一地面/局部高差在精确配平误报下均 `0/5`，Christ Church 3400 帧假支持 `0.09846 m²`）；
    2. 在严格保持原空路误报水平（`<= 1 窗、4 时刻`）且**按物理地点留一选阈值**下，检验 $\mathrm{TGV}^2$ 跳跃分量 $j$ 能否将真目标与空铺装排序反转回正，实现跨地点 `>= 2/5` 首次独立提醒（若 Christ Church 3400 帧实为单目深度本身的锐利纹理伪影而非光滑弯曲，须以定量残差谱严格证明该边界并指出需要法向/多帧证据）。

---

# 规格书 M11：$\mathrm{SE}(3)\times S^1$ 步态周期 Floquet 锁相可观性——纯头戴因果解耦“头朝向 vs 身体行进方向”

> 任务目录：`research/active/egogeom-omni-v0/math_theory/m11_floquet_gait_heading/`

## 1. 精确数学问题定义

[LaMAR 正前漏报拆解](../hardware-bringup/LAMAR_MISS_BREAKDOWN_DEV_20261117.md) 证实：**近一半（7/15）的正前漏报并非传感器没看到，而是头朝向与真实身体行进方向偏离（中位 18°、最高 58°）把警戒区画歪了！** 如果直接按行进方向画走廊，8/9 个含方向偏离的漏报片段当帧即达标。但盲人穿戴眼镜没有胸口 IMU，只能访问**头部单目图像 + 头部 6 轴 IMU** $(\tilde{\boldsymbol{\omega}}_t, \tilde{\mathbf{a}}_t)$。

定义头部与身体耦合运动学流形 $\mathcal{M} = \mathrm{SO}(3)_{\text{head}} \times S^1_{\text{body}} \times S^1_{\text{gait}} \times \mathbb{R}_{>0}$：
- 身体水平行进航向 $\psi_b(t) \in S^1$，头部世界偏航角 $\psi_h(t) = \psi_b(t) + \theta_{\text{neck}}(t)$（颈部扫视角 $\theta_{\text{neck}}$ 含快速随机转头，纯陀螺仪完全无法区分身体转弯与转头扫视！）；
- 双足步行周期相位 $\phi(t) \in S^1$（$\dot\phi = 2\pi f_{\text{stride}}$，典型步频 $f_{\text{stride}} \in [0.7, 1.3]\text{ Hz}$）；
- 在重力对齐世界系下，头部线性加速度 $\mathbf{a}_w(t) = R_w^h(t)\tilde{\mathbf{a}}_t - \mathbf{g}$ 的步态受迫谐波展开（Floquet 周期动力学）：
  $$\mathbf{a}_w(t) = \underbrace{A_{\text{ML}}\cos(\phi(t) + \varphi_{\text{ML}})\,\mathbf{e}_\perp(\psi_b)}_{\text{一倍频左右骨盆摆动 }\perp \psi_b} + \underbrace{A_{\text{AP}}\cos(2\phi(t) + \varphi_{\text{AP}})\,\mathbf{e}_\parallel(\psi_b)}_{\text{二倍频前后加减速 }\parallel \psi_b} + \underbrace{A_z\cos(2\phi(t) + \varphi_z)\,\mathbf{e}_z}_{\text{二倍频竖直落地冲击}} + \mathbf{a}_{\text{turn}} + \boldsymbol{\eta}_t$$
- 同时，视觉通道经陀螺仪去旋后的光流扩张焦点（FOE）给出头部瞬时平移方向观测 $\hat{\mathbf{v}}_h(t) / \|\hat{\mathbf{v}}_h(t)\|$，但受颈部转头杠杆臂 $\boldsymbol{\omega}_{\text{neck}} \times \mathbf{r}_{\text{neck}}$ 与低纹理退化污染。

## 2. 必须证明的核心定理与推导任务

1. **定理 M11.1（步态 Floquet 谐波互谱可观性与 180° 前后歧义打破定理）**：
   - 证明：仅用水平加速度协方差矩阵（经典 PCA 法）只能辨识行进轴线 $\psi_b \pmod{\pi}$，存在固有的 **180° 前后不可分辨歧义**，且当步态左右摆动能量与前后能量接近时发生特征值简并！
   - 推导基于**复数解析信号与一/二倍频互谱张量**的完整可观性定理：
     - 利用竖直冲击 $a_z(t)$ 锁相提取精确倍频相位 $2\phi(t)$，证明水平加速度在 $2f_{\text{stride}}$ 处与 $a_z$ 的**复互谱实部/虚部向量** $\mathbf{c}_{2} = \mathbb{E}[e^{-i2\phi}\mathbf{a}_{xy}]$ 同时锁定前后轴方向与唯一符号（由生物力学脚跟触地刹车的固定相位差 $\Delta\varphi = \varphi_{\text{AP}} - \varphi_z \in (-\pi, 0)$ 打破 180° 歧义！）；
     - 同时利用基频 $f_{\text{stride}}$ 能量主轴 $\mathbf{u}_1 \perp \mathbf{e}_\parallel(\psi_b)$ 正交约束，给出联合融合 IMU 步态谐波 + 去旋视觉 FOE 时的**精确 Cramér–Rao 下界 $\mathrm{CRLB}(\psi_b)$** 及全部退化流形（如静止站立 $A_z \to 0$、原地踏步、平滑轮椅/滑行无步态冲击）。
2. **定理 M11.2（不确定航向下“概率扫描走廊”最小贝叶斯风险几何）**：
   - 已知因果估计得到的是身体航向后验分布 $p(\psi_b \mid \mathcal{F}_t) = \mathrm{vonMises}(\hat\psi_b, \kappa_t)$ 而非完美真值；
   - 证明：固定角度硬走廊在航向方差 $\kappa_t^{-1}$ 大时必然同时制造漏报与侧墙误报；导出**基于 von Mises 后验积分的期望碰撞概率走廊算子**，给出最佳告警走廊半宽随航向集中度 $\kappa_t$ 变化的闭式律。
3. **算法 M11.3（`< 0.3 ms` 因果 $S^1$ 谐波锁相 + FOE 融合滤波器 `CausalBodyHeadingFilter`）**：
   - 严格因果（仅使用过去及当前 IMU/图像/位姿，严禁使用未来位移！）；
   - 自带运动/步态可观性自检门控（非步行或低集中度 $\kappa_t$ 时自动平滑退回并扩大不确定度，不乱猜方向）。

## 3. 验证基线与验收门槛

- **符号与合成验证（`verify_theory.py`）**：
  - SymPy 验证互谱旋转等变性与 180° 符号判别式；
  - 合成步行轨迹（含 $\pm 45^\circ$ 正弦/阶跃快速转头扫视 + 弯道转弯 + 停走切换）：验证纯头朝向误差中位 `> 20°` 时，因果滤波器将身体航向 MAE 压至 **`< 6°`**，且停步/原地转头时不发生虚假航向漂移。
- **真实数据实证（`eval_empirical.py`）**：
  - 在本地真实头部 IMU 轨迹（`artifacts.local/work/cnh-nymeria-imu-causal-dev-20261008/` 与 `artifacts.local/work/phone-new-capture-20261012/`）上验证实采步态倍频互谱相位锁定的存在性与稳定性；
  - 在 `artifacts.local/work/lamar-independent-heading-dev-20261116-codex/` 与 `lamar-miss-breakdown-dev-20261117/` 的真实头戴走廊对照上评测严格因果航向估计误差（对比 `head_yaw` 基线与历史运动门控基线）及对 15 段正前漏报 / 30 个误报时刻的净改善；
  - **验收门槛**：完整报告因果航向误差分位数（P50/P90）、180° 翻转率、以及对 LaMAR 方向偏离漏报段（原 7 段）的因果追回率与误报代价（若某套数据缺少高采样率原始 IMU 仅含低频位姿/视觉，须明确分通道检验“纯视觉 FOE 臂”与“头部 IMU 谐波臂”各自的实测边界）。

---

# 规格书 M12：真实 H3 有符号非单射 CNH 的射影白化不变量（$\mathbb{P}^{13}$）与多帧步行合成孔径超分辨

> 任务目录：`research/active/egogeom-omni-v0/math_theory/m12_h3_projective_whitening_sr/`

## 1. 精确数学问题定义

[M1 第二轮 SIGNED_H3.md](math_theory/m1_poisson_cnh_deconv/SIGNED_H3.md) 彻底查清了真实 ST VL53L8CH H3 固件的物理契约：
- 8×8 分区、16 粗箱（每 8 个原生 37.5348 mm 箱求和 $\Rightarrow \Delta z = 300.2784\text{ mm/bin}$），单帧非单射；
- 固件内隐含未知逐帧缩放/累积增益 $g_t > 0$ 与未知环境光/参考残差扣除 $\hat b_t$，导致 13%–48% 箱值为负且相邻箱呈显著正相关（$\bar\rho \approx 0.247$）。

将单个分区在时刻 $t$ 报出的有符号 16 维波形建模为**受未知射影-仿射干扰与相关噪声作用的物理观测**：
$$\mathbf{h}_t = g_t \Big(\mathbf{c}_{\text{xt},t} + \alpha_1 \boldsymbol{\phi}(z_1 - \delta_t) + \alpha_2 (1 - \tau_g)\boldsymbol{\phi}(z_2 - \delta_t) - \hat b_t \mathbf{1}\Big) + \boldsymbol{\varepsilon}_t, \qquad \boldsymbol{\varepsilon}_t \sim \mathcal{N}\big(\mathbf{0},\, \Sigma(A_t)\big)$$
其中：
- 未知干扰子空间 $U = [\mathbf{1},\, \mathbf{e}_0,\, \mathbf{e}_1] \in \mathbb{R}^{16\times 3}$（分别张成未知环境光扣除直流偏置与固定在 `bin0/bin1` 的护盖玻璃串扰！）；
- $\delta_t$ 为可观的传感器沿光轴步行位移（由 IMU/VO 或 `poses_t` 提供）；
- 同步 RGB 支路（`EgoGeom-Omni v4.1` Pure-RGB）在面对透明玻璃时穿透看到背后不透明面，提供背景距离先验 $\hat z_2 = \hat Z_{\text{RGB}}$。

## 2. 必须证明的核心定理与推导任务

1. **定理 M12.1（Mahalanobis 白化斜投影商空间 $\mathbb{P}^{13}$ 中的精确极大不变 GLRT）**：
   - 构造相对于相关噪声精度矩阵 $\Sigma(A_t)^{-1}$ 的斜投影算子：
     $$P_\perp = I - U (U^\top \Sigma^{-1} U)^{-1} U^\top \Sigma^{-1}, \qquad \tilde{\mathbf{h}}_t = \Sigma^{-1/2} P_\perp \mathbf{h}_t$$
     并做球面归一化 $\mathbf{q}_t = \tilde{\mathbf{h}}_t / \|\tilde{\mathbf{h}}_t\|_2 \in \mathbb{S}^{12} / \{\pm 1\}$（商掉未知逐帧增益 $g_t > 0$！）；
   - **严格证明**：$\mathbf{q}_t$ 是关于干扰群 $(\hat b_t, c_{\text{xt},0}, c_{\text{xt},1}, g_t)$ 的**精确极大不变量**——无论固件内部如何做环境光扣除、如何发生 `bin0/bin1` 串扰漂移或改变帧累积增益，$\mathbf{q}_t$ 的分布完全不变！
   - 在给定 RGB 背景先验 $z_2 = \hat Z_{\text{RGB}}$ 下，导出检验 $H_0$（仅背景回波）vs $H_1$（存在前景弱透明玻璃回波 $\alpha_1 > 0, z_1 < z_2$ 且引起背景衰减）的精确**配对双符号射影 GLRT**（将 M1 第二轮手工“近箱升且柜面箱降”规则升级为无背景标定也成立的精确解析统计量！）。
2. **定理 M12.2（多帧步行移位采样的 Papoulis 超分辨单射性恢复定理）**：
   - 单帧 300.28 mm 粗箱求和算子 $\mathcal{S}_8: \mathbb{R}^{128} \to \mathbb{R}^{16}$ 严格非单射（核空间维数 112）；
   - 考虑步行过程中的 $M$ 帧序列，已知相对径向位移 $0 = \delta_0 < \delta_1 < \dots < \delta_{M-1}$。在目标距离相对静态（世界系不动）而串扰保持在 $z=0$ 不动时，严格证明：
     - 当位移集合 $\{\delta_m \bmod \Delta z\}_{m=0}^{M-1}$ 在 $[0, \Delta z)$ 内满足非退化间隔条件且 $M \ge M_{\min}$（给出显式 $M_{\min}$ 与条件数界），复合观测算子对稀疏脉冲回波恢复**严格单射性**；
     - 推导多帧超分辨测距的精确 CRLB，定量给出 $M \in \{1, 3, 5, 8\}$ 帧下可从 `300.28 mm` 粗箱恢复出的最小理论测距误差（厘米级界）以及护盖串扰的完全分离条件。
3. **算法 M12.3（可直接替换 `Physical1DCNHBranchV41` 前端的 `< 0.2 ms` 可微白化射影物理层）**：
   - 将斜投影白化特征 $\tilde{\mathbf{h}}_t$（及多帧位移对齐特征）写成纯 PyTorch/NumPy 无分支张量层，满足 `v4.1` 零干扰契约（无 ToF 时严格输出 0）。

## 3. 验证基线与验收门槛

- **符号与合成验证（`verify_theory.py`）**：
  - SymPy 验证 $P_\perp U = 0$、Mahalanobis 正交性与对 $(g_t, \hat b_t, \mathbf{c}_{\text{xt}})$ 的严格代数不变性；
  - 模拟 8 合 1 粗箱 + 随机逐帧增益 $g_t \in [0.2, 5.0]$ + 随机直流扣除 + $\rho = 0.25$ 跨箱相关 + `82% bin0` 串扰：验证单帧 `argmax` 误差 `> 0.15 m`，而 $M=5$ 帧步行移位超分辨在无标定下实现 `< 0.04 m` MAE，且 RGB 背景先验条件下的弱玻璃前峰检测 AUROC 较未白化基线显著提升。
- **真实 1,209 帧 H3 + 18 段移动手机录像实证（`eval_empirical.py`，完全复用 `signed_h3_replay.py` 基线）**：
  - 复现原柜面恢复段基线（仅近箱升 `84/153`、双符号配对 `26/153`、六箱相关 95% 覆盖 `94.0%`）；
  - 测试射影白化不变量 GLRT 在**不依赖固定初始柜面均值**（零静态标定！）下的漂移抑制表现，并在 `phone-new-capture-20261012/` 移动头戴真实 H3 录像上检验多帧位移对齐是否有效抑制静止墙/空走廊的串扰误触发（诚实标注缺少玻璃真值的边界）。

---

# 规格书 M13：$\mathrm{SO}(C)$ 流形上的深度 Fisher 加权正交旋转均衡与 Hexagon v75 全网 W8A8/混合精度硅片压缩

> 任务目录：`research/active/egogeom-omni-v0/math_theory/m13_fisher_so_c_w8a8_silicon/`

## 1. 精确数学问题定义

`EgoGeom-Omni v4.1` 已在 Samsung Galaxy S24 Ultra（Hexagon v75 HTP）上实现 FP16 **`16.51 ms`（Pure-RGB, 60.6 FPS, `107.72 MB`）** 与 **`19.11 ms`（Unified Omni + Signed-H3, 52.3 FPS, `109.38 MB`）**，实片相对 FP32 误差仅 **`0.788% / 0.601%`**。要将全天候连续佩戴功耗与带宽压低 2–4× 并把延迟冲进 **`< 10 ms`（> 100 FPS）**，必须突破 DINOv3 ConvNeXt-Small + `GlobalDisparityContext` + `Physical1DCNHBranchV41` 的全网 **W8A8 / 混合 W8A16 量化**。

普通 LLM 量化方法（SmoothQuant / QuaRot）最小化的是各层无权激活 MSE，但对米制深度网络存在致命失配：
- `GlobalDisparityContext` 与 `disp_head` 的微小加性误差会经 $Z = 1/d$ 在全局尺度上产生**百倍各向异性放大**；
- `Physical1DCNHBranchV41` 的零初始化残差分支一旦被非对称零点量化污染，就会破坏 **`max|Omni(0) - PureRGB| = 0.000000 m`** 的硬零干扰契约！

定义每个可融合线性/卷积块第 $\ell$ 层激活 $\mathbf{h}_\ell \in \mathbb{R}^{C_\ell \times H_\ell \times W_\ell}$ 处的**任务黎曼-费雪敏感度矩阵**（相对深度对数误差度量）：
$$\mathcal{I}_{Z,\ell} = \mathbb{E}_{\mathbf{x}\sim \mathcal{D}_{\text{cal}}}\!\left[\frac{1}{|\Omega|}\sum_{p \in \Omega}\left(\frac{\partial \log Z(p)}{\partial \mathbf{h}_\ell}\right)\left(\frac{\partial \log Z(p)}{\partial \mathbf{h}_\ell}\right)^\top\right] \succeq 0$$
对网络中每对可等价吸收变换的相邻线性算子 $(W_{\ell+1}, W_\ell)$（如 ConvNeXt MLP、1×1 投影与 LayerNorm/GroupNorm 边界），寻找函数严格等价的可逆变换 $Q_\ell = R_\ell D_\ell \in \mathrm{SO}(C_\ell) \ltimes \mathrm{Diag}_+(C_\ell)$（正交旋转 + 通道对角均衡），最小化经 Fisher 度量传输到最终深度的总期望相对误差：
$$\min_{\{Q_\ell\},\,\{b_\ell \in \{8, 16\}\}} \sum_{\ell=1}^{L} \mathrm{Tr}\!\left(\mathcal{I}_{Z,\ell}\, Q_\ell^{-1}\, \mathrm{Cov}\!\big[\mathcal{Q}_{b_\ell}(Q_\ell \mathbf{h}_\ell) - Q_\ell \mathbf{h}_\ell\big]\, Q_\ell^{-\top}\right) \quad \text{s.t.} \quad \sum_\ell T_{\text{v75}}(\ell, b_\ell) \le T_{\max},\;\; \max_\ell M_{\text{VTCM}}(\ell, b_\ell) \le 8\text{ MB}$$

## 2. 必须证明的核心定理与推导任务

1. **定理 M13.1（$\mathrm{SO}(C)$ 上 Fisher 加权正交均衡的最优性条件与超额峰度压缩界）**：
   - 推导单个线性块在给定对角/低秩近似 Fisher 权重 $\mathcal{I}_{Z,\ell}$ 下，对角缩放 $D_\ell$ 的闭式最优解（证明其为权重量化方差与 Fisher 加权激活量化方差的精确几何平均，推广 SmoothQuant 的 $\alpha=0.5$ 启发式！）；
   - 证明在 $\mathrm{SO}(C)$ 上使用随机化/优化 Walsh–Hadamard 或 Cayley 正交变换 $R_\ell$ 将最大通道异常值压低 $\Theta(\sqrt{C/\log C})$ 时，Fisher 加权深度失真上界的精确衰减率，同时指出哪些算子（如 depthwise `7×7` conv、elementwise GELU/SiLU）**不可穿透正交旋转**并给出无额外运行时开销的精确图融合集合！
2. **定理 M13.2（零干扰对称零点量化保护定理）**：
   - 针对 `Physical1DCNHBranchV41` 输出的 `(dp1, s_res, a_res)` 与 `has_tof_mask` 门控，**严格证明**：当且仅当门控前后的定量编码满足对称零点约束 `zero_point == 0`（精确表示实数 `0.0`）且融合偏置零化时，定点图仍严格保持芯片级零干扰 **`max|Omni(0) - PureRGB| == 0.000000`**！
3. **定理 M13.3（混合精度位宽分配的拉格朗日凸松弛与对偶间隙界）**：
   - 对每层张量在 `{INT8, INT16}`（或 `FP16` 保护末层解码头）间的 0-1 背包分配问题，利用对角/Hutchinson 迹估计高效计算全网 200+ 层的 $\mathrm{Tr}(\mathcal{I}_{Z,\ell}\Delta^2_\ell)$，通过拉格朗日乘子凸松弛证明其解与全局整数最优解的对偶间隙至多不超过单层最大失真贡献，给出自动化生成 QNN 2.47 `--quantization_overrides` JSON 的精确算法。
4. **工程交付（零重训直接对接 QNN 2.47 + 可选 S24 Ultra 实片验收）**：
   - 直接加载 `egogeom_omni_v41_best_pure_rgb_fp32.onnx` 与 `egogeom_omni_v41_best_omni_fp32.onnx`，用无标签校准集（严禁使用测试标签！）生成等价均衡后 ONNX + 最优混合精度 overrides，调用 `qairt-converter` / `qnn-context-binary-generator` 编译 Hexagon v75 `.bin`；若 S24 Ultra 通过 `adb` 在线则直接跑 50 次真机测速与 16 帧面板保真验收！

## 3. 验证基线与验收门槛

- **符号与仿真验证（`verify_theory.py`）**：
  - SymPy 验证 $\mathrm{SO}(C)$ 旋转等价性、对角缩放几何平均最优解与对称零点零干扰充要条件；
  - 合成高异常值（kurtosis `> 80`）+ 强各向异性 Fisher 敏感度网络：验证 Fisher-$\mathrm{SO}(C)$ 混合量化较朴素 W8A8 PTQ 与无权 SmoothQuant 的最终相对误差降低 **`>= 3×`**。
- **真实 `v4.1` ONNX / QNN 2.47 / S24U 实证（`eval_empirical.py`，复用 16 帧无标签四域面板与 `v4.1` FP16 基线 `16.51 / 19.11 ms`, `0.788% / 0.601%`）**：
  - **验收门槛**：
    1. 成功生成通过 `onnx.checker` 且 100% 被 QNN 2.47 `qairt-converter` + `v75` 离线编译器接受的 W8A8/混合精度 `v4.1` Pure-RGB 与 Omni 图（零 unsupported op、零 VTCM spill）；
    2. 在 16 帧四域无标签面板上，相对 FP32 平均相对误差 **`<= 1.2%`**（理想冲击 `< 0.9%`），严格打败朴素全网 W8A8 PTQ（量化给出完整误差-耗时 Pareto 前沿），且 Omni 无 ToF 零干扰误差 **`<= 1e-6 m`（严格零点约束下 `0.000000 m`）**；
    3. 若连上 S24 Ultra 实测，HTP 执行时间相比 FP16 基线（`16.51 / 19.11 ms`）显著提速（目标 Pure-RGB **`< 10.5 ms`** / Omni **`< 12.5 ms`**，并报告真实 `.bin` 体积缩小倍数）。

---

# 规格书 M14（面向 `v5` 换代架构基础）：重力对齐球面 $\mathbb{S}^2$ 共形图册、Chebyshev–Milnor 最小畸变定理与精确 $\mathrm{SO}(2)$ 平移等变性

> 任务目录：`research/active/egogeom-omni-v0/math_theory/m14_conformal_s2_atlas_equivariance/`

## 1. 精确数学问题定义

现有单目网络（含 `v4.1`）在传统针孔平面像素网格 $(u, v)$ 上运行标准 2D 卷积。当跨相机部署到超广角鱼眼眼镜（如 Meta Aria Gen 2、Oxford 1440×1080 鱼眼）时出现两大根源性几何缺陷：
1. 针孔重投影在 $120^\circ$–$150^\circ$ 边缘产生 $3\text{–}6\times$ 各向异性面积拉伸与大块无效黑边（如 [SHALLOW_ALERTS_DEV_20261118.md](../hardware-bringup/SHALLOW_ALERTS_DEV_20261118.md) 暴露的 **40,736 / 307,200 无效黑边像素**引发最大假警）；
2. 标准 `Conv2d` 对平面平移等变，但对**头部水平偏航旋转 $R_z(\Delta\psi) \in \mathrm{SO}(2)$** 并不等变，导致物体移到画面边缘时几何形状畸变与尺度漂移！

考虑单位视球 $\mathbb{S}^2$ 上绕重力轴 $\mathbf{e}_g$ 的重力对齐球坐标 $(\varphi, \theta)$（方位角 $\varphi \in (-\pi, \pi]$、仰角 $\theta \in (-\pi/2, \pi/2)$）。定义重力对齐**球面共形（保角）图册** $\Phi: U \subset \mathbb{S}^2 \to \Omega \subset \mathbb{R}^2$，使得球面标准度量 $ds_{\mathbb{S}^2}^2 = d\theta^2 + \cos^2\theta\, d\varphi^2$ 拉回为各向同性共形度量：
$$ds_{\mathbb{S}^2}^2 = \lambda^2(\xi, \eta)\,\big(d\xi^2 + d\eta^2\big)$$
其中典型特例为**重力对齐墨卡托（Mercator）图** $(\xi, \eta) = (\varphi,\, \operatorname{gd}^{-1}(\theta) = \log\tan(\frac{\pi}{4} + \frac{\theta}{2}))$，共形因子 $\lambda(\eta) = \operatorname{sech}(\eta) = \cos\theta$；以及广义单叶全纯映射 $w = f(e^{\eta + i\varphi})$ 族。

## 2. 必须证明的核心定理与推导任务

1. **定理 M14.1（重力共形图上的精确 $\mathrm{SO}(2)$ 水平偏航平移等变性与各向同性卷积核定理）**：
   - 严格证明：在重力对齐墨卡托图 $(\xi, \eta)$ 上，任意头部水平偏航旋转 $R_z(\Delta\psi) \in \mathrm{SO}(2)$ 精确等价于 $\xi$ 轴的**刚性像素平移 $\xi \mapsto \xi + \Delta\psi$**，且共形因子 $\lambda(\eta)$ 完全独立于 $\xi$；因此任何标准 `Conv2d`（含任意空洞率与深度可分离卷积，零自定义算子、100% 原生 Hexagon HTP 兼容！）在水平偏航下满足**精确无近似平移等变性**，且局部各向异性剪切比 $\sigma_{\max}(J_\Phi)/\sigma_{\min}(J_\Phi) \equiv 1$（彻底消除边缘拉伸畸变与鱼眼黑边！）。
2. **定理 M14.2（可穿戴视场区域上的 Chebyshev–Milnor 最小尺度畸变极值问题）**：
   - 给定头戴眼镜的真实非对称视场球面区域 $\mathcal{F} \subset \mathbb{S}^2$（例如水平 $\pm 75^\circ$、垂直 $[-55^\circ, +35^\circ]$，或任意给定鱼眼视场圆盘），在所有共形图册中寻找使最大对数尺度畸变 $\mathcal{D}[\Phi] = \max_{\mathcal{F}} \log\lambda - \min_{\mathcal{F}} \log\lambda$ 最小的极值共形映射（经典 Chebyshev / Milnor 球面制图变分问题，对应求解 $\Delta_{\mathbb{S}^2} \log\lambda = 1$ 且边界 $\log\lambda|_{\partial \mathcal{F}} = \text{const}$ 的泊松 Dirichlet 边值问题！）；
   - 精确对比**针孔切平面、等距柱面（Equirectangular）、重力墨卡托（Mercator）、球极投影（Stereographic）与 Chebyshev 最优共形图**在 $60^\circ / 110^\circ / 150^\circ$ FOV 上的：① 最大各向异性剪切比；② 最大尺度动态范围 $\lambda_{\max}/\lambda_{\min}$；③ 有效像素利用率（对比 Oxford 针孔整流浪费 13.3% 黑边像素）。
3. **定理 M14.3（共形图上的精确深度-法向拉普拉斯/梯度算子与零重训重采样误差界）**：
   - 推导共形坐标系 $(\xi, \eta)$ 下新的深度-法向 Frobenius 可积性方程（证明 $\lambda(\eta)$ 仅作为已知闭式标量因子进入偏导，形式极其简洁！）；
   - 给出面向 `v5` 训练的**统一任意内参/鱼眼（Pinhole + Kannala–Brandt + MEI/Aria）重力共形重采样算子**（带精确边界有效光线掩码），并在现有冻结 `v4.1` 上做**零重训测试时共形重投影探测实验**（测清楚当前针孔训出来的 `v4.1` 在共形图上会有多少域漂移、以及单独消除黑边与畸变的上限收益）。

## 3. 验证基线与验收门槛

- **符号与微分几何验证（`verify_theory.py`）**：
  - SymPy 符号验证共形度量张量 $g_{ij} = \lambda^2 \delta_{ij}$、高斯曲率恒等式 $-\lambda^{-2}\Delta \log\lambda \equiv 1$、墨卡托图 $\mathrm{SO}(2)$ 精确平移等变性，以及共形深度-法向 PDE；
  - 用二维有限元/有限差分求解真实眼镜非对称 FOV 域 $\mathcal{F}$ 上的 Chebyshev–Milnor 泊松问题 $\Delta \psi = -1, \psi|_{\partial\mathcal{F}} = 0$，输出 5 种投影在 $60^\circ\text{–}150^\circ$ FOV 的定量畸变对照表。
- **真实鱼眼与广角眼镜实证（`eval_empirical.py`）**：
  - 在 `artifacts.local/work/shallow-alerts-20261118/controls/oxford_valid_camera.json`（1440×1080 鱼眼）与 `artifacts.local/work/egogeom-omni-v0/zeroshot_eval/aria_g2_20clips/`（Meta Aria Gen 2）上实测：有效像素覆盖率、各向异性畸变降幅、水平偏航旋转一致性（人工施加球面偏航重采样检验特征/深度等变误差）；
  - 输出一份可直接用于 `EgoGeom-Omni v5` 架构设计的精确工程契约（坐标映射、共形因子编码通道、梯度损失权重修正公式、以及 Hexagon v75 零自定义算子确认）。

---

## 推荐给 Codex 的并行派单波次

1. **第一波（立即可见系统级质变，直击最大瓶颈，零 GPU 重训）**：
   - **M6（极值尾指数 + 群不变 E-Process 风险泛函）** + **M8（自由边界 QVI 信念空间最优告警停时）**：双剑合璧，正面掀翻 M3 的 `13/365`（Aria `0/143`）清路峰值铁顶与 `82.42%` 清路告警爆炸！
   - **M7（法向耦合直接光度扩张 Looming + 正迎面 CRLB 相变定理）**：正面攻克 M4 的 `70%` 静止假速度反例与高速正迎面电动车盲区。
2. **第二波（跨眼镜几何精度 & 芯片翻倍提速）**：
   - **M9（$\mathrm{PGL}(4)$ 射影规范 + 法向 Frobenius 可积性旋度）**：打破 M2 标量神谕 `405/840` 天花板，并激活 `v4.1` 闲置的 `normal_grav` 出力。
   - **M13（$\mathrm{SO}(C)$ 深度 Fisher 正交旋转均衡 + Hexagon v75 W8A8/混合精度）**：冲击 S24 Ultra `< 10 ms`（> 100 FPS）与 2–4× 降功耗，直接实片验证。
3. **第三波（浅障、航向、真实 H3 硬件与 `v5` 底层几何换代）**：
   - **M10（$SBV / \mathrm{TGV}^2$ 曲面浅障分解）** + **M11（$\mathrm{SE}(3)\times S^1$ 步态 Floquet 锁相因果身体航向）** + **M12（H3 $\mathbb{P}^{13}$ 射影白化 + 多帧合成孔径超分辨）** + **M14（$\mathbb{S}^2$ 重力共形图册与 Chebyshev–Milnor 极值定理）**。
