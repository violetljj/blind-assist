# EgoGeom-Omni v4 究极数学理论与算法规格书（可直接投喂 Codex）

> 适用仓库：`e:\linnan\linnan`（独立探索目录 `research/active/egogeom-omni-v0/math_theory/`）
> 依赖主线基线：`EgoGeom-Omni v4`（commit `d78160a2`，51.04M DINOv3 `convnext_small.dinov3_lvd1689m`，S24 Ultra Hexagon v75 FP16 `17.78 ms` 纯 RGB / `19.41 ms` 统一 Omni + CNH）
> 执行环境：`E:\codex-tools\tools\venvs\blindassist-sam3-gpu\Scripts\python.exe`（PyTorch `2.11.0+cu130`、RTX 5060 8.5 GB、NumPy/SciPy/SymPy）；QNN 2.47 转换环境：`E:\codex-tools\tools\venvs\qairt-2.47-py310\Scripts\python.exe`

---

## 全局投喂约定（复制到任意单题 Prompt 顶部）

```text
[GLOBAL CODEX CONTRACT]
1. 工作目录与隔离：所有新代码与推导文档严格放入 `research/active/egogeom-omni-v0/math_theory/<task_id>/`，大体积中间结果放入 `artifacts.local/work/egogeom-omni-v0/math_theory/<task_id>/`。严禁改动 `egogeom_omni_v4_model.py`、默认 Android App 或其他并行研究目录。
2. 三位一体强制交付（缺一不可）：
   (A) `THEORY.md`：完整逐行数学推导、严格定理陈述、全部前提假设、必要/充分条件与精确反例边界（不允许用“显然/易得”跳过关键放缩步）。
   (B) `verify_theory.py`：使用 SymPy（符号验证微积分/雅可比/李导数/欧拉-拉格朗日方程）+ 蒙特卡洛数值实验（检验每个不等式、CRLB、覆盖率、保守间隙与渐近阶的紧致度，输出 `theory_verification.json`）。
   (C) `eval_on_v4.py`：直接加载本项目真实 `EgoGeom-Omni v4` 产物与零样本数据集跑实证对照，输出 `v4_empirical_report.json` 并与下方冻结基线逐项比对。
3. 诚信纪律：绝不伪造指标；若某定理条件在真实数据上不满足或仅局部成立，须明确标注失效区间与边界条件。
```

---

## 冻结共享资产与基线路径

- **模型权重**：`artifacts.local/work/egogeom-omni-v0/egogeom_omni_v4_unified_best.pt`（含 `cfg` 与 `ema`，模型定义 `research/active/egogeom-omni-v0/egogeom_omni_v4_model.py`）
- **双静态 ONNX-17**：
  - `artifacts.local/work/egogeom-omni-v0/egogeom_omni_v4_best_pure_rgb_fp32.onnx`
  - `artifacts.local/work/egogeom-omni-v0/egogeom_omni_v4_best_omni_fp32.onnx`
- **4 套零样本跨眼镜数据（`*.npz`，含 `rgb_offsets`, `rgb_blob`, `orig_hw`, `K_orig`, `poses_R`, `poses_t`, `causal_fwd`, `eval_rows`）**：
  - `artifacts.local/work/egogeom-omni-v0/zeroshot_eval/aria_g2_20clips/`（19 clips，1,794 valid / 840 pos `<=4m`）
  - `artifacts.local/work/egogeom-omni-v0/zeroshot_eval/coda_zeroshot_shard000/`（24 clips，2,819 valid / 34 pos `<=4m`）
  - `artifacts.local/work/egogeom-omni-v0/zeroshot_eval/arkit_val30_shard00/`（30 clips，467 valid / 388 pos `<=2m`，支持 `dark_scale=1.0` 与 `0.18`）
- **基线评测报告**：
  - `artifacts.local/work/egogeom-omni-v0/v4_history.json`
  - `artifacts.local/work/egogeom-omni-v0/cross_glasses_zeroshot_v4_final.json`
  - `artifacts.local/work/egogeom-omni-v0/s24u_v4_final_npu_report.json`
- **已有物理与时序参考实现**：
  - `research/active/egogeom-omni-v0/egogeom_losses_and_physics.py`（`compute_gravity_aligned_ray_map`, `simulate_tof_and_cnh_prompt`）
  - `research/active/egogeom-omni-v0/egogeom_temporal_bev.py`（`EgoGeomTemporalBEV`, `TemporalBEVConfig`）
  - `research/active/egogeom-omni-v0/eval_egogeom_v3_zeroshot.py`（含 `score_dataset`, `--arch v4`）

---

# 规格书 M1：16-bin 泊松光子直方图的强串扰双峰盲反卷积与信息论相变定理

> 任务目录：`research/active/egogeom-omni-v0/math_theory/m1_poisson_cnh_deconv/`

## 1. 精确数学问题定义

ST VL53L8CH 8×8 SPAD 传感器在单个分区输出 $K = 16$ 个时间分箱的离散光子计数向量 $\mathbf{y} = (y_0, \dots, y_{15})^\top \in \mathbb{Z}_{\ge 0}^{16}$，分箱宽度 $\Delta z \in [0.24, 0.29]\text{ m}$（标称 $0.25\text{ m}$，总有效量程 $4.0\text{ m}$）。当视锥同时包含**护盖玻璃内部串扰**（`bin0` 占总能量 $45\%\text{–}82\%$，真实 H3 典型值 $74\%$）、**前表面透明玻璃**（距离 $z_1$、弱菲涅尔回波强度 $\alpha_1 > 0$）、**背后不透明背景**（距离 $z_2 > z_1$、强回波强度 $\alpha_2 > \alpha_1$）与**太阳环境光直流底噪** $b > 0$ 时，观测服从独立泊松模型：
$$y_k \sim \mathrm{Poisson}(\lambda_k(\boldsymbol{\theta})), \qquad \lambda_k(\boldsymbol{\theta}) = b + \alpha_0 \psi_k^{\mathrm{xtalk}} + \alpha_1 \phi_k(z_1) + \alpha_2 \phi_k(z_2), \quad k = 0,\dots,15$$
其中：
- $\psi^{\mathrm{xtalk}} = (0.94, 0.06, 0, \dots, 0)^\top$（或未知双参数串扰核 $(\alpha_0, \alpha_{01})$）；
- 离散化仪器响应函数（IRF）为高斯积分核：
  $$\phi_k(z) = \int_{k\Delta z}^{(k+1)\Delta z} \frac{1}{\sqrt{2\pi}\sigma_{\mathrm{irf}}} \exp\!\left(-\frac{(u - z)^2}{2\sigma_{\mathrm{irf}}^2}\right) du, \qquad \sigma_{\mathrm{irf}} \in [0.06, 0.14]\text{ m}$$
- 待估参数向量 $\boldsymbol{\theta} = (z_1, z_2, \alpha_1, \alpha_2, \alpha_0, b)^\top$，总有效光子预算 $N_{\mathrm{sig}} = \alpha_1 + \alpha_2$，信背比 $\mathrm{SBR} = N_{\mathrm{sig}} / (16 b)$，串扰比 $\rho_{\mathrm{xt}} = \alpha_0 / (\alpha_0 + N_{\mathrm{sig}} + 16b)$。

## 2. 必须证明的核心定理与推导任务

1. **定理 M1.1（强串扰泊松双峰 Fisher 信息矩阵与 Cramér–Rao 下界）**：
   - 推导精确 $6\times 6$ Fisher 信息矩阵 $I(\boldsymbol{\theta})_{ij} = \sum_{k=0}^{15} \frac{1}{\lambda_k} \frac{\partial \lambda_k}{\partial \theta_i}\frac{\partial \lambda_k}{\partial \theta_j}$；
   - 利用 Schur 补消去干扰参数 $(\alpha_0, b, \alpha_1, \alpha_2)$，给出前表面透明玻璃测距方差下界 $\mathrm{CRLB}(z_1)$ 与双峰间距 $\Delta = z_2 - z_1$ 的**小间距渐近展开**（证明当 $\Delta \ll \sigma_{\mathrm{irf}}$ 时 $\mathrm{CRLB}(z_1)$ 按 $\Delta^{-2}$ 或 $\Delta^{-4}$ 发散的精确阶次，取决于反射率比 $r = \alpha_1 / \alpha_2$ 与串扰覆盖区间 $z_1 < 0.5\text{ m}$ 的耦合！）。
   - 显式导出**透明玻璃可辨识相变曲面** $\Delta_{\min}(\alpha_1, \alpha_2, \alpha_0, b, \sigma_{\mathrm{irf}})$：在给定误差容限 $\mathrm{RMSE}(z_1) \le 0.15\text{ m}$ 下，给出物理可解与物理不可解的精确分界。
2. **定理 M1.2（单峰 vs 双峰透明玻璃检测的广义似然比检验 GLRT 与渐近分布）**：
   - 针对嵌套假设 $H_0: \alpha_1 = 0$（无玻璃，单不透明面）vs $H_1: \alpha_1 > 0, z_1 < z_2$（存在前景透明玻璃），注意 $\alpha_1 = 0$ 位于参数空间边界且 $z_1$ 在 $H_0$ 下不可辨识（Davies 边界非正则问题！）；
   - 推导校正后的泊松偏差统计量 $T_{\mathrm{GLRT}}(\mathbf{y}) = 2\big(\ell(\hat{\boldsymbol{\theta}}_1) - \ell(\hat{\boldsymbol{\theta}}_0)\big)$ 的精确零分布上界（混合 $\chi^2$ / 高斯过程极值 Rice 公式上界），给出给定虚警率 $\alpha_{\mathrm{FA}}$ 下的理论检测门限。
3. **算法 M1.3（Beurling-LASSO / 连续域泊松滑动 Frank–Wolfe 超分辨率求解器 + 5 步展开可微物理层）**：
   - 在非负 Radon 测度空间 $\mathcal{M}_+([z_{\min}, 4.0])$ 上求解凸优化：
     $$\min_{\mu \in \mathcal{M}_+,\, b \ge 0,\, \alpha_0 \ge 0} \; D_{\mathrm{KL}}\!\left(\mathbf{y} \,\middle\|\, b\mathbf{1} + \alpha_0 \psi^{\mathrm{xtalk}} + \Phi \mu\right) + \lambda \|\mu\|_{\mathrm{TV}}$$
   - 推导对偶证书（Dual Certificate）$\eta(z) = \frac{1}{\lambda}\sum_{k=0}^{15}\left(\frac{y_k}{\lambda_k} - 1\right)\phi_k(z) \le 1$，证明其全局最优充要条件与 Sliding Frank–Wolfe（SFW）子线性/局部二次收敛性；
   - 构造纯张量化的 **5 步展开可微层**（无分支、支持 PyTorch 反向传播与 NCHW 推理），单帧处理 `64` 个分区耗时 `<0.30 ms`（CPU NumPy/C++ 等价算力）。

## 3. 验证基线与验收门槛

- **符号与蒙特卡洛验证（`verify_theory.py`）**：
  - 用 SymPy 验证 $\partial \lambda_k / \partial \theta_i$ 与 Schur 补表达式；
  - 在 $N_{\mathrm{sig}} \in \{50, 200, 1000, 5000\}$、$\rho_{\mathrm{xt}} \in \{0.05, 0.50, 0.74, 0.85\}$、$r = \alpha_1/\alpha_2 \in \{0.05, 0.15, 0.35\}$、$\Delta \in [0.15, 1.5]\text{ m}$ 网格上各跑 `2,000` 次蒙特卡洛泊松抽样：验证 SFW 估计器在相变界以上逼近 CRLB（效率比 $\mathrm{RMSE}/\sqrt{\mathrm{CRLB}} \le 1.35$），且 GLRT 实测虚警率不超过理论门限 $+1\%$。
- **对比基线（须严格打败的 3 个对照）**：
  1. 传统质心/最大值取 bin（`argmax(y[1:]) * 0.25`）——遇 `74% bin0` 串扰与透明玻璃严重偏后；
  2. 朴素双高斯最小二乘（L2 拟合，忽视泊松方差异质性）；
  3. 仅用现有 `Physical1DCNHBranchV4` 静态输入特征 `[c_raw, c_tail, c_diff]` 训练的轻量 MLP 玻璃分类器。
- **验收指标（`v4_empirical_report.json`）**：
  - 在 `realistic_hardware=True`（74% `bin0` 串扰 + 双峰玻璃，$\Delta \ge 0.45\text{ m}$）下，SFW 前景玻璃测距 `MAE` 较 `argmax` 降低 `>= 45%`、较 L2 拟合降低 `>= 20%`；
  - 玻璃存在检出 `AUROC >= 0.92`（在 $\mathrm{SBR} \ge 0.5, r \ge 0.15$ 区域）；
  - 5 步展开层单帧 64 分区推理 `<0.30 ms`（Python/PyTorch CPU 批测折算）。

---

# 规格书 M2：$\mathrm{SE}_2(3)\times\mathbb{R}_{>0}$ 李群上的无标定单目动态尺度可观性定理与不变滤波器

> 任务目录：`research/active/egogeom-omni-v0/math_theory/m2_lie_group_scale_iekf/`

## 1. 精确数学问题定义

对于任意未知焦距/未知安装高度的头戴眼镜，`EgoGeom-Omni v4` 在第 $t$ 帧纯 RGB 输出存在未知乘性全局尺度偏移 $s_t \in \mathbb{R}_{>0}$：
$$Z^{\mathrm{true}}_t(u, v) = s_t \cdot \hat{Z}_t(u, v), \qquad \delta h^{\mathrm{true}}_t(u,v) = s_t \cdot \widehat{\delta h}_t(u, v)$$
其中 $\hat{Z}_t, \widehat{\delta h}_t, \hat{\mathbf{n}}_t, \hat{\sigma}_t$ 为 `v4` 实际推理输出。设备同时提供：
- 连续帧相对位姿或消费级 6 轴 IMU 测量 $(\tilde{\boldsymbol{\omega}}_t, \tilde{\mathbf{a}}_t)$（带零偏 $\mathbf{b}_g, \mathbf{b}_a$）；
- 或视觉里程计/光流/已知相对帧间位移与旋转 $(R_t, \mathbf{v}_t, \mathbf{p}_t)$；
- 步行场景下的**局部近场地面点云流形观测**：由 `v4` 预测出近场地面掩码 $M_{\mathrm{gnd}} = \{p : |\widehat{\delta h}(p)| < 0.06\text{ m},\; \hat{\mathbf{n}}(p)\cdot \mathbf{e}_z > 0.92\}$，其缩放后的世界坐标高度满足局部缓变地面约束 $h_{\mathrm{cam}}(t) = s_t \cdot \hat{h}_{\mathrm{gnd}}(t)$，且帧间静物点满足刚体投影一致性：
  $$\mathbf{p}_{t+1} + R_{t+1} \big(s_{t+1} \hat{\mathbf{X}}_{t+1}\big) = \mathbf{p}_t + R_t \big(s_t \hat{\mathbf{X}}_t\big) + \boldsymbol{\epsilon}_t$$

定义扩增矩阵李群 $\mathcal{G} = \mathrm{SE}_2(3) \times \mathbb{R}_{>0}$（或 $\mathrm{Sim}_2(3)$ 嵌套表示），状态元素：
$$X_t = \big(R_t, \mathbf{v}_t, \mathbf{p}_t, s_t\big) \in \mathrm{SO}(3)\times\mathbb{R}^3\times\mathbb{R}^3\times\mathbb{R}_{>0}, \qquad \mathbf{b}_t = (\mathbf{b}_{g,t}, \mathbf{b}_{a,t}) \in \mathbb{R}^6$$

## 2. 必须证明的核心定理与推导任务

1. **定理 M2.1（Group-Affine 不变误差动力学与精确雅可比）**：
   - 在 $\mathcal{G}$ 上构造右不变/左不变误差 $\eta_t = \hat{X}_t^{-1} X_t$（其中尺度误差取对数 $\delta \ell_t = \log(s_t / \hat{s}_t)$），检验动力学 $f_u(X)$ 是否满足 **Barraud–Bonnabel Group-Affine 条件**：
     $$f_u(X_1 X_2) = f_u(X_1) X_2 + X_1 f_u(X_2) - X_1 f_u(I) X_2$$
   - 推导出李代数误差 $\boldsymbol{\xi}_t = (\boldsymbol{\phi}_t, \delta\mathbf{v}_t, \delta\mathbf{p}_t, \delta\ell_t)^\top \in \mathbb{R}^{10}$ 的精确线性化误差传播矩阵 $F_t$ 与观测雅可比 $H_t$，证明在无零偏时 $F_t$ 严格**独立于当前状态估计 $\hat{X}_t$**。
2. **定理 M2.2（Hermann–Krener 非线性局部弱可观性与退化运动完整分类）**：
   - 构造观测空间 $\mathcal{O}$ 由以下三类观测的李导数生成：
     (i) 单目帧间视差-位移一致性（`v4` 稠密深度 + 已知/IMU 积分平移）；
     (ii) 步态垂直加速度周期激励 + 局部地面高差 $\hat{h}_{\mathrm{gnd}}(t)$；
     (iii) 间歇稀疏 ToF 观测（存在率 $p_{\mathrm{tof}} \in [0, 1]$，含 $p_{\mathrm{tof}} = 0$ 纯视觉情形！）。
   - 用 SymPy 符号计算可观性矩阵 $\mathcal{O}(\mathbf{x}) = \nabla [\mathbf{h}, L_f \mathbf{h}, L_f^2 \mathbf{h}, \dots]$ 的秩，**严格证明**：
     - 情形 A（有 IMU + 无 ToF）：在存在非零水平加减速或步行垂直周期加速度 $\ddot{z}(t) \neq 0$ 时，全局尺度 $s$ 与重力方向、IMU 零偏联合可观；并给出精确退化流形（如纯原地旋转 $\mathbf{v}\equiv \mathbf{0}$ 或匀速无起伏平动）。
     - 情形 B（仅有相对位姿/已知步行姿态序列 + `v4` 深度，如 `Aria Gen 2` / `CODa` / `ARKit` 评测流）：推导仅用 `poses_R, poses_t`（因果单向！）+ `v4` `depth_m / normal_grav / log_sigma` 的**异方差重加权李群尺度观测方程**，并证明指数收敛速率 $\mathbb{E}[(\log s_t - \log s^*)^2] \le C e^{-\beta t} + \sigma_\infty^2$。
3. **算法 M2.3（`<0.3 ms` 因果在线李群尺度自校准滤波器，直接可插拔进 `EgoGeomTemporalBEV`）**：
   - 严格因果（第 $t$ 帧只允许使用 $0\dots t$ 帧的 `v4` 预测与因果位姿/IMU，**严禁偷看 `eval_rows` 或未来帧**！）；
   - 加入基于 `v4` `log_sigma` 与平面残差的 **Huber/Cauchy 鲁棒加权（M-estimator）** 以抵抗动态物体和玻璃反射离群点；
   - 实现纯 NumPy 因果接口 `LieScaleIEKF.step(depth_s4, dh_s4, nrm_s4, ls_s4, K, R_wc, t_wc)` -> 输出在线尺度因子 $\hat{s}_t$。

## 3. 验证基线与验收门槛

- **冻结 `v4` 实测基线（[cross_glasses_zeroshot_v4_final.json](cross_glasses_zeroshot_v4_final.json)）**：
  - `Meta Aria Gen 2`（1,794/840）：`1f_world_z` `AUROC=0.7018, AP=0.7204, R@5%=35.0%, MAE=0.934 m`；`stage2_bev_v23_hyb` `AUROC=0.7141, AP=0.7256, R@5%=32.7%, MAE=0.609 m`。
  - `UT Austin CODa`（2,819/34）：`1f_world_z` `AUROC=0.6718, MAE=0.928 m`；`stage2_bev_v23_hyb` `AUROC=0.7321, MAE=1.188 m`。
  - `ARKit val30 Day`（467/388）：`1f_world_z` `AUROC=0.7106, MAE=0.642 m, Late@5=10.8%`；`stage2_bev_v23_hyb` `AUROC=0.7360, MAE=0.509 m, Late@5=7.5%`。
  - `ARKit val30 Dark 0.18×`（467/388）：`1f_world_z` `AUROC=0.6989, MAE=0.751 m, Late@5=21.5%`；`stage2_bev_v23_hyb` `AUROC=0.7229, MAE=0.556 m, Late@5=7.5%`。
- **验收门槛（零重训、严格因果、同一套超参跨全部 4 套数据集！）**：
  1. `verify_theory.py`：SymPy 符号验证 Group-Affine 等式与李导数秩条件；合成轨迹上验证未知初始尺度偏差 $s_0 \in [0.4, 2.5]$ 在 `<15` 帧（3 秒 @ 5 Hz）内收敛至 $|\hat{s}_t / s^* - 1| < 5\%$，且在退化运动下协方差不发散/不误漂；
  2. `eval_on_v4.py`：把 `LieScaleIEKF` 因果输出的 $\hat{s}_t$ 乘到 `v4` 深度后跑完整 `eval_egogeom_v3_zeroshot.py` 协议：
     - **`Meta Aria Gen 2` `1f MAE` 从 `0.934 m` 降至 `<= 0.55 m`（且 `+BEV MAE <= 0.42 m`）**，同时 `AUROC` 不降反升（`>= 0.72`）；
     - **`ARKit Day / Dark` `Late@FA<=5/km` 从 `7.5% / 7.5%` 压降至 `<= 4.5% / 5.5%`**；
     - `UT Austin CODa` `AUROC` 与 `MAE` 同步改善或持平；
     - 单帧 CPU 滤波耗时 `<0.30 ms`。

---

# 规格书 M3：非平稳分布偏移与删失反馈下的有限样本在线共形避障安全包络

> 任务目录：`research/active/egogeom-omni-v0/math_theory/m3_online_conformal_safety/`

## 1. 精确数学问题定义

在时刻 $t = 1, \dots, T$，视觉感知模块输出场景特征 $x_t$ 与预测碰撞距离/不确定度 $(\hat{D}_t, \hat{\sigma}_t)$，真实前方最近障碍距离为 $D_t^* \in (0, \infty]$。决策器维护动态安全余量阈值 $\lambda_t \in \mathbb{R}$ 并输出告警决策：
$$a_t = \mathbb{I}\!\left\{S(x_t) \ge q_t\right\} \quad \text{或} \quad \mathcal{C}_t = \left[\hat{D}_t - \lambda_t \hat{\sigma}_t,\; \infty\right)$$
定义两类核心物理损失：
1. **安全漏报事件（Safety Violation / Miss）**：$M_t(\lambda_t) = \mathbb{I}\{D_t^* \le D_{\mathrm{safe}} \text{ 且 } a_t = 0\}$（或校准误差事件 $E_t(\lambda_t) = \mathbb{I}\{D_t^* < \hat{D}_t - \lambda_t \hat{\sigma}_t\}$）；
2. **误报成本（False Alarm）**：$F_t(\lambda_t) = \mathbb{I}\{D_t^* > D_{\mathrm{far}} \text{ 且 } a_t = 1\}$。

真实盲人行走的 **3 个非经典数学难点**：
- **(N1) 非平稳强分布突变**：环境序列 $\{P_t\}_{t=1}^T$ 在未知时刻发生剧烈切换（如室外强光 -> 漆黑走廊 `dark_scale=1.0 -> 0.18` -> 玻璃大厅），用总变差/阈值路径长度 $P_T = \sum_{t=2}^T |\lambda_t^* - \lambda_{t-1}^*|$ 刻画非平稳性；
- **(N2) 延迟反馈（Delayed Feedback）**：在时刻 $t$ 做出的决策直到 $t + d_t$（经过障碍物或到达临界制动点，$d_t \le \tau_{\max}$）才观测到真实误差；
- **(N3) 反事实干预删失（Intervention Censoring）**：当 $a_t = 1$（系统告警）且盲人因此停步/绕行时，未告警状态下的碰撞结果以未知倾向概率 $\pi_t \in [\epsilon_{\min}, 1]$ 被删失（缺失非随机，MNAR/MAR 边界）。

## 2. 必须证明的核心定理与推导任务

1. **定理 M3.1（延迟反馈 + 非平稳路径长度下的强自适应在线共形覆盖界）**：
   - 设计基于 **多尺度专家聚合（Strongly Adaptive Online Learning / Scale-Free Exponentiated Gradient over ACI step-sizes $\eta_k \in \{2^{-k}\}$）** 的在线阈值更新律 $\lambda_{t+1} = \mathcal{A}(\lambda_t, \{E_{s}\}_{s + d_s = t})$；
   - **严格证明**任意对抗性/非平稳序列下：
     1. 长期精确覆盖率保证（Long-run Validity）：$\left|\frac{1}{T}\sum_{t=1}^T E_t(\lambda_t) - \alpha\right| \le \frac{C_1(\tau_{\max})}{T} + O(T^{-1/2})$；
     2. **任意局部子区间 $I = [t_1, t_2]$（如突变进入暗光走廊的 30 秒内！）的强自适应动态后悔界（Strongly Adaptive Interval Regret）**：用 $P_I = \sum_{t \in I} |\lambda_t^* - \lambda_{t-1}^*|$ 显式刻画恢复速度，证明固定步长 ACI 在突变段存在 $\Omega(\sqrt{|I|})$ 滞后而多尺度 SA-ACI 可达 $O(\sqrt{P_I \log T})$ 最优恢复！
2. **定理 M3.2（删失反馈下的双重稳健逆倾向加权共形风险控制 DR-IPW-CRC）**：
   - 在反馈以观测概率 $e_t = P(O_t = 1 \mid \mathcal{F}_{t-1}) \ge \epsilon > 0$（如系统以极小探索率或靠后置 ToF/安全护栏回验真值）被删失时，构造无偏/双重稳健代理梯度：
     $$\hat{g}_t = \frac{O_t}{e_t}(E_t - \alpha) + \left(1 - \frac{O_t}{e_t}\right)(\hat{p}_t - \alpha)$$
   - 利用 **Freedman 鞅集中不等式（Martingale Concentration）** 给出有限样本高概率风险上界（精确列出 $\epsilon$ 与模型预测偏差 $|\hat{p}_t - \mathbb{E}[E_t|\mathcal{F}_{t-1}]|$ 对置信半径的显式依赖）。
3. **算法 M3.3（面向盲人闭环的 `RiskConformalController`）**：
   - 直接输入 `EgoGeom-Omni v4` 的单帧或 `EgoGeomTemporalBEV` 输出序列 $(\hat{D}_t, \hat{\sigma}_t, \text{score}_t)$，每帧计算 `<0.05 ms`；
   - 在不泄露未来真值的前提下输出带有限样本概率下界的保守安全距离 $\hat{D}_t^{\mathrm{safe}}(\alpha)$。

## 3. 验证基线与验收门槛

- **合成突变压力测试（`verify_theory.py`）**：
  - 构造包含 `5` 次阶跃分布漂移 + `d_t = 5` 帧延迟 + `30%` 删失率的长序列（$T = 10,000$ 步，重复 100 次），验证：
    - 目标漏报率 $\alpha \in \{0.02, 0.05, 0.10\}$ 下，全局实测违约率落入 $[\alpha - 0.008, \alpha + 0.008]$；
    - 突变后 `20` 步局部窗口内的最坏违约率较标准固定步长 ACI 下降 `>= 35%`。
- **真实 `v4` 跨眼镜拼接非平稳流实测（`eval_on_v4.py`）**：
  - 将真实 `v4` 在 `ARKit Day -> ARKit Dark 0.18× -> Aria Gen 2 -> CODa` 上生成的连续推理流拼接成**跨域突变长轨迹**；
  - 对比 3 个基线：① 固定距离阈值（如 `2.0 m`）；② 固定步长标准 ACI（Gibbs & Candès 2021）；③ 仅用未校准 `v4` `log_sigma` 高斯分位数；
  - **验收门槛**：在同等每公里误报数 `FA/km <= 5` 下，`SA-DR-CRC` 闭环漏晚报率 `Late@FA<=5/km` 较固定阈值与未校准 `log_sigma` 显著下降（且在 `Day -> Dark` 突变段保持严格覆盖率上界成立，绝不出现失控漏报雪崩）。

---

# 规格书 M4：耦合距离平方不确定度与人类反应延迟的随机控制障碍函数（SCBF）与不可避免碰撞集解析界

> 任务目录：`research/active/egogeom-omni-v0/math_theory/m4_stochastic_cbf_ics/`

## 1. 精确数学问题定义

考虑头戴眼镜行人与前方（可能高速迎面移动的）障碍物在地面水平面 2D 相对坐标系下的运动：
- 相对状态 $\mathbf{x}(t) = (p_x, p_y, v_x, v_y)^\top \in \mathbb{R}^4$，行人可行避让控制输入 $\mathbf{u}(t) = (a_{\mathrm{lon}}, a_{\mathrm{lat}})^\top \in \mathcal{U}$（制动减速 $a_{\mathrm{lon}} \in [-a_{\mathrm{brake}}, a_{\mathrm{acc}}]$、侧步横移 $|a_{\mathrm{lat}}| \le a_{\mathrm{side}}$，典型盲人步态 $a_{\mathrm{brake}} = 2.5\text{ m/s}^2, a_{\mathrm{side}} = 1.5\text{ m/s}^2$）；
- **系统 + 人类听觉反应总延迟** $\tau = \tau_{\mathrm{sys}} + \tau_{\mathrm{human}} \approx 0.05 + 0.60 = 0.65\text{ s}$：在 $[t, t+\tau]$ 内行人维持原速度，控制输入仅在 $t + \tau$ 后生效；
- 障碍物未知加速度扰动 $\mathbf{w}(t) \in \mathcal{W}$ 及**由 `v4` 异方差单目深度/场景流诱导的距离相关随机观测过程**：
  $$d\mathbf{x}_t = \big(A \mathbf{x}_t + B \mathbf{u}_{t-\tau}\big) dt + G(\mathbf{x}_t)\, d\mathbf{W}_t, \qquad \|G(\mathbf{x}_t)\| \propto c_0 + c_1 \|\mathbf{p}_t\|^2$$
  （真实反映单目视差噪声导致距离方差 $\sigma_Z^2(Z) \propto Z^4$，而当 ToF 有效时在 $Z \le 4\text{ m}$ 退化为常数 $\sigma_{\mathrm{tof}}^2$！）。
- 安全集 $\mathcal{S} = \{\mathbf{x} : \|\mathbf{p}\|_2 \ge r_{\mathrm{body}} + r_{\mathrm{obs}}\}$。

## 2. 必须证明的核心定理与推导任务

1. **定理 M4.1（延迟+不确定度下的不可避免碰撞集 ICS 精确刻画与 HJI 粘性解下界）**：
   - 写出该 4D 延迟随机微分博弈的 **Hamilton–Jacobi–Isaacs（HJI）变分不等式**；
   - 构造显式**高阶随机控制障碍函数（High-Order Stochastic CBF）** $h_{\tau,\sigma}(\mathbf{x})$，同时耦合：
     (i) 延迟前推预测状态 $\hat{\mathbf{p}}(t+\tau) = \mathbf{p} + \mathbf{v}\tau$；
     (ii) 制动与侧步最优避让边界（解析最小脱险距离 $d_{\mathrm{esc}}(\mathbf{v}, a_{\mathrm{brake}}, a_{\mathrm{side}})$）；
     (iii) Itô 引理（Itô's Lemma）产生的二阶扩散项 $\frac{1}{2}\mathrm{Tr}\!\big(G(\mathbf{x})^\top \nabla^2 h(\mathbf{x}) G(\mathbf{x})\big)$——**特别注意：当 $\sigma_Z \propto Z^2$ 时，该二阶项对远场告警边界有非平凡的曲率修正！**
   - **严格证明（Soundness + Tightness）**：
     - 证明满足 Itô 漂移条件 $\mathcal{L} h + \kappa(h) \ge 0$ 时，有限时域 $T_H$ 内的碰撞概率满足严格指数/Doob 上鞅上界 $P(\exists t \in [0, T_H], \mathbf{x}_t \notin \mathcal{S}) \le \delta$；
     - 给出该闭式解析边界与 4D 网格精确 HJI 粘性解之间的**紧致度间隙上界（Conservatism Gap Bound）**，证明它不像朴素最坏情况球包络那样在高速下退化为无用早报。
2. **算法 M4.2（轻量因果 3D 场景流速度估计算子 + `<0.4 ms` 解析 SCBF 告警引擎）**：
   - 仅用连续两帧 `v4` 深度图 $(D_{t-1}, D_t)$ + 因果位姿 $(R, \mathbf{t})$ + 轻量稀疏光流/特征匹配（OpenCV/NumPy），推导静态背景剔除 + 动态障碍物 3D 相对速度 $(\hat{v}_x, \hat{v}_y)$ 与协方差 $\hat{\Sigma}_v$ 的闭式加权最小二乘估计；
   - 将 $(\hat{\mathbf{p}}, \hat{\mathbf{v}}, \hat{\Sigma}_p, \hat{\Sigma}_v)$ 代入解析 SCBF 函数 $h_{\tau,\sigma}(\mathbf{x})$，输出连续风险标量与**物理最优触发时刻 $t^*_{\mathrm{alert}}$**。

## 3. 验证基线与验收门槛

- **精确 PDE 数值基准验证（`verify_theory.py`）**：
  - 用 SymPy 推导 Itô 二阶扩散项与最优避让解析解；
  - 在 2D/4D 网格上用 Lax–Friedrichs 迎风格式数值求解参考 HJI 偏微分方程，覆盖静止柱子（$v=0$）、迎面行人（$v=-1.5\text{ m/s}$）、高速电动车（$v=-6.0\text{ m/s}$）与侧向穿行目标；
  - 跑 `10,000` 条 Euler–Maruyama 随机微分方程蒙特卡洛轨迹：验证解析 SCBF 保证的碰撞率严格 `<= 目标风险 δ`（如 $\delta=0.01$），同时相比保守固定距离阈值，**对静止障碍的无效过早告警距离缩短 `>= 30%`，对 `6 m/s` 高速迎面障碍的碰撞避免率从 `<20%`（固定 2 m 阈值根本来不及刹车！）提升至 `>= 98%`**！
- **真实 `v4` 序列实证（`eval_on_v4.py`）**：
  - 在 `UT Austin CODa`（含真实校园移动行人/车辆，2,819 帧）与 `ARKit val30` 上，用 `v4` 输出 + 因果场景流对比：**固定距离阈值 vs 经典 TTC vs 解析 SCBF** 的闭环 `Late@FA<=5/km` 与预警提前量分布（Time-to-Collision margin）。

---

# 规格书 M5：NPU FP16 / INT8 反演深度 $Z = 1/\mathrm{disp}$ 的变分最优信息几何重参数化

> 任务目录：`research/active/egogeom-omni-v0/math_theory/m5_info_geom_quant_reparam/`

## 1. 精确数学问题定义

`EgoGeom-Omni v4` 输出预测视差 $d = 1/Z \in [d_{\min}, d_{\max}] = [0.04, 6.67]\text{ m}^{-1}$（对应 $Z \in [0.15, 25.0]\text{ m}$）。在真实 S24 Ultra Hexagon v75 FP16 实机验收（[s24u_v4_final_npu_report.json](s24u_v4_final_npu_report.json)）中，发现一个精确硬件现象：
- `Pure-RGB` 在 NPU FP16 与 PyTorch FP32 之间出现 **`depth_rel_pct = 3.330%`（`depth_mae = 155.74 mm`）**，主要由远场 $Z \in [12, 25]\text{ m}$ 时 $d \to 0.04$ 附近反演导数 $\left|\frac{dZ}{dd}\right| = Z^2$ 放大 FP16 量化舍入所致；若进一步压至 **W8A8 INT8（均匀网格 256 级）**，直接量化 $d$ 或 $Z$ 都会在远场或近场产生严重精度崩塌。

设下游盲人避障的安全敏感黎曼失真权重为已知正函数 $w(Z) > 0$（例如综合近场碰撞时间误差与全量程相对误差的加权度量）：
$$w(Z) = \frac{\beta_1}{Z^2} + \frac{\beta_2}{Z^4} + \beta_3 \mathbb{I}_{[0.3, 4.0]}(Z)$$
硬件表示变量 $u = \phi(Z) \in [u_{\min}, u_{\max}]$（$\phi$ 为严格单调 $C^2$ 双射），其硬件量化网格点密度函数为 $\rho_{\mathrm{hw}}(u)$：
- **INT8 均匀定点量化**：$\rho_{\mathrm{INT8}}(u) = \text{const}$；
- **IEEE-754 FP16 浮点量化**（10-bit 尾数，区间 $[2^e, 2^{e+1})$ 内均匀）：高率连续近似 $\rho_{\mathrm{FP16}}(u) \propto \frac{1}{|u|}$（$u \ge u_{\mathrm{subnormal}}$）。

高率量化失真泛函（Bennett Integral under Risk Density $p(Z)$）：
$$\mathcal{J}[\phi] = \frac{1}{12} \int_{Z_{\min}}^{Z_{\max}} p(Z)\, w(Z) \, \frac{1}{\big(\rho_{\mathrm{hw}}(\phi(Z)) \, |\phi'(Z)|\big)^2}\, dZ$$
以及极大极小最坏情况失真：
$$\mathcal{J}_\infty[\phi] = \sup_{Z \in [Z_{\min}, Z_{\max}]} \frac{\sqrt{w(Z)}}{\rho_{\mathrm{hw}}(\phi(Z)) \, |\phi'(Z)|}$$

## 2. 必须证明的核心定理与推导任务

1. **定理 M5.1（Euler–Lagrange 变分最优信息几何重参数化闭式解）**：
   - 分别在 **期望失真 $\mathcal{J}[\phi]$**（Hölder 不等式 / Euler–Lagrange 方程）与 **极大极小最坏失真 $\mathcal{J}_\infty[\phi]$** 两种准则下，严格求出针对 **INT8** 与 **FP16** 两种真实硬件算术的**唯一解析最优映射** $\phi^*_{\mathrm{INT8}}(Z)$ 与 $\phi^*_{\mathrm{FP16}}(Z)$！
   - 证明关键推论：
     - 当 $w(Z) = Z^{-2}$（纯相对误差）且硬件为 **INT8** 时，极大极小最优映射严格为 $\phi^*(Z) = \log Z$；
     - 当 $w(Z) = Z^{-4}$（纯视差 / 1/TTC 误差）且硬件为 **INT8** 时，极大极小最优映射严格为 $\phi^*(Z) = 1/Z$；
     - 当兼顾近场 1/TTC + 全量程相对误差（$w(Z) = \beta_1 Z^{-2} + \beta_2 Z^{-4}$）时，**严格导出广义双曲/反双曲正弦闭式最优压扩映射** $\phi^*(Z) = \operatorname{asinh}(\gamma / Z)$ 或 $\log(Z + c) - \dots$，并计算其相较线性深度 $Z$、纯视差 $1/Z$、纯对数 $\log Z$ 在全量程 `[0.15, 25.0 m]` 上的**理论 SNR / 最大误差改善倍率（严格闭式比值）**！
     - 针对 **FP16**（$\rho_{\mathrm{FP16}}(u) \propto 1/|u|$），推导为什么直接输出未平移的小视差 $d \in [0.04, 6.67]$ 或接近 0 的激活会在 `softplus + reciprocal` 链路中损失有效比特，并给出**零重训图融合重参数化变换**（仅在导出 ONNX 的最后 3 个算子插入仿射/分段有理重缩放，不改主干权重！）。
2. **定理 M5.2（PTQ 无重训逐层灵敏度上界与 Padé 硬件算子逼近界）**：
   - 对反演函数 $\phi^{*-1}(u)$ 构造 `[2/2]` 或 `[3/2]` 有理 Padé 逼近（仅需加减乘除，100% 原生支持 Hexagon v75 HTP，零 `exp/log/asinh` 查表退化），严格证明最大逼近误差 `< 10^{-4}`；
   - 证明在已知最终特征层前馈量化噪声上界 $\epsilon_{\mathrm{act}}$ 下，输出米制深度的全局误差严格上界。

## 3. 验证基线与验收门槛

- **符号与数值最优性验证（`verify_theory.py`）**：
  - 用 SymPy 符号求解 Euler–Lagrange 方程与 $\mathcal{J}_\infty$ 变分最优解，验证最优解满足二阶变分正定（全局最小）；
  - 在真实 IEEE-754 `np.float16` 与真实对称/非对称 8-bit 定点仿真器上，对比 5 种参数化（线性 $Z$、纯视差 $1/Z$、$\log Z$、$\sqrt{Z}$、变分最优 $\phi^*(Z)$）在 `[0.15, 25.0 m]` 的近场 `AbsRel`、远场 `AbsRel` 与最坏点相对误差。
- **真实 `v4` ONNX-17 实证（`eval_on_v4.py`，零重训！）**：
  - 直接加载 `artifacts.local/work/egogeom-omni-v0/egogeom_omni_v4_best_pure_rgb_fp32.onnx` 与 `egogeom_omni_v4_best_omni_fp32.onnx`（冻结基线：`Pure-RGB` S24U FP16 相对误差 `3.330%`，`Omni+CNH` `1.782%`，`Omni+Depth` `1.697%`）；
  - 构造“零重训图末端重参数化 + Padé 反演”新 ONNX 并在本地模拟 FP16 & INT8（有条件时直接走 `qairt-converter` 生成 DLC 验证全算子通过，零 unsupported op！）；
  - **验收门槛**：
    1. INT8 全量程 `[0.15, 25.0 m]` 输出头最坏相对量化误差较直接量化 `disp` 降低 **`>= 5×`**、较直接量化 `depth` 降低 **`>= 8×`**；
    2. FP16 输出头反演误差在 `[0.15, 25.0 m]` 全量程压降至 **`< 0.15%`**（定位并拆解原 `3.33%` 误差中“主干 ConvNeXt 累积 FP16 误差” vs “末层反演放大误差”的精确占比）；
    3. 导出带 Padé 重参数化的 ONNX-17 通过 `onnx.checker` 且全部算子被 QNN 2.47 `qairt-converter` 原生接受。

---

## 推荐给 Codex 的并行派单顺序

1. **第一波（立即可见业务质变，零 GPU 重训）**：并行派发 **M2（李群尺度不变滤波 -> 直接打穿 `Aria Gen 2` 跨眼镜尺度漂移）** + **M1（16-bin 泊松强串扰双峰反卷积与 CRLB 相变定理 -> 直接筑牢 `$3` ToF 玻璃物理护城河）**。
2. **第二波（顶会论文理论闭环）**：并行派发 **M3（非平稳+删失在线共形安全包络）** + **M5（FP16/INT8 信息几何最优重参数化）**。
3. **第三波（动态避障终局）**：派发 **M4（随机 CBF + 不可避免碰撞集解析界，对接 `CODa` 动态场景）**。
