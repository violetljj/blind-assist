# RCLE-Minimal Phase A 预注册

状态：`PREREGISTERED_BEFORE_FORMAL_RUN`

冻结时间：`2026-07-26T00:43:48+08:00`

机器协议 SHA-256：`d20e77f3ea5f7ac55376006f1d14feb0ffb5daffd10a42792912fb89cdb1b502`

唯一问题：在程序生成、旋转真值已知的连续图像中，旋转补偿能否压低纯旋转造成的局部 expansion 泄漏，同时保留尺度增大造成的正 expansion？

## 权限与非目标

本轮只产生 synthetic mechanism / coordinate / implementation evidence。它不证明 Bonn、真实场景、视障用户、安全、Android 或生产有效性。本轮不做深度、bearing、shear、双区域、风险场、Android、新 UI、Phase B 或 Replay Demo。

R1.1 的较宽 synthetic 清单还包含 horizontal/vertical translation 与 rotation+translation；用户本轮明确的最小问题只包含 rotation、scale 及三类退化，因此 translation 在 R0 标为 `DEFERRED_NON_GATING`，不进入 Kill Gate A。

## 冻结实验单位

- 统计单位：trial，不是 frame 或 grid。
- 图像：480×360 灰度；`fx=fy=420 px`，主点 `(239.5, 179.5)`。
- 时长：每 trial 0.6 s；帧率为 15/30/60 FPS，时间戳决定真实 `dt`。
- 20 个不可替换 seed：1000–1019；同 seed 的所有条件在 bootstrap 中作为一个 cluster。
- clean：yaw/pitch/roll 正负方向 × 15/30 deg/s × 3 FPS；scale up/down × 0.15/0.30 s^-1 × 3 FPS；rotation+scale up 使用相同旋转矩阵与 `+0.15 s^-1`。
- stress sentinel：30 FPS、30 deg/s、`|scale rate|=0.15 s^-1`，分别单独施加 noise、blur、partial occlusion。
- 计划总数：clean 1680、stress 840、合计 2520 trials；失败 seed 不补抽。

每帧从同一底图与累计解析 pair homography 直接渲染，避免反复插值造成 FPS 相关累积模糊。每对帧冻结
`H_pair = H_rotation × H_scale`；补偿将 current frame 用 `H_rotation^-1` warp 回 previous 坐标，因此解析残差只能是 scale。

## 算法与主指标

Sparse LK 的位移除以实际 `dt` 后进入 3×3 fixed-grid affine fit：

`v(p) = A(p-c) + b`，`expansion = 0.5 × trace(A)`，单位为 `s^-1`。

纯旋转主指标：

`L_rot = median_pairs(median_evaluable_grids(|expansion|))`。

closing 主指标：trial robust expansion 与解析 scale rate 的绝对误差及符号正确率。RSR/CRR 只作诊断；raw 分母低于 `0.03 s^-1` 时必须 `NOT_EVALUABLE`，不得加 epsilon。所有不可评价项保留在完整分母中，不能回填为零。

## 数值 Kill Gate A

1. Clean yaw/pitch：informative raw trial 的 pooled compensated leakage 中位数 95% CI 上界 ≤0.030 s^-1；配对绝对下降中位数 CI 下界 ≥0.015 s^-1；改善 trial 比例 CI 下界 ≥0.80。每个 cell 的 compensated leakage 中位数 ≤0.040 s^-1 且改善比例 ≥0.80。
2. Clean roll：compensated leakage 中位数 CI 上界 ≤0.030 s^-1；`L_comp-L_raw` 中位数 CI 上界 ≤0.010 s^-1。roll 的 raw divergence 可能本来近零，不要求制造 RSR。
3. Clean scale：closing error 中位数 CI 上界 ≤0.030 s^-1；rotation+scale 为 ≤0.040 s^-1；符号正确率点估计 ≥0.98 且 Wilson 下界 ≥0.95；`error_comp-error_raw` 中位数 CI 上界 ≤0.015 s^-1；mixed 与 matched scale-only 的 compensated expansion 差异 CI 上界 ≤0.030 s^-1。
4. FPS：matched seed/motion 的 15/30/60 FPS trial expansion range，其 CI 上界 ≤0.030 s^-1。
5. Coverage：clean paired point/CI-lower ≥0.90/0.85，单 cell ≥0.80；stress paired point/CI-lower ≥0.80/0.75，单 cell ≥0.70。
6. 每种 stress：compensated rotation leakage CI 上界 ≤0.050 s^-1；closing error CI 上界 ≤0.050 s^-1；closing 符号正确率点估计/CI-lower ≥0.95/0.90。

置信区间使用固定 seed `20260726` 的 10,000 次 seed-cluster percentile bootstrap；比例同时给 Wilson 95% CI。runtime 只报告 mean/median/P95，不是 Phase A gate。

## 判定

- `PASS`：全部 clean、FPS、coverage、stress、receipt 与解析不变量通过。
- `REVISE`：clean 核心没有被否证，但 challenge 不充分，或局部实现、FPS、coverage、stress、receipt 门失败；保留原 run，不改阈值，只允许版本化实现修复。
- `STOP`：坐标/oracle 已验证、clean challenge 充分，但旋转抑制或 closing 保留失败；或一次版本化 REVISE 后仍失败。

机器可读细节、参数、不可评价原因和精确门值以同 Module 的
`configs/phase_a_synthetic_signal_audit_r0.json` 为唯一真源。正式 runner 必须绑定其 SHA-256；任何漂移都 fail closed。
