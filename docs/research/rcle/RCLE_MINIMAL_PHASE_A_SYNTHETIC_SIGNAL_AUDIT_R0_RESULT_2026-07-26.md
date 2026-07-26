# RCLE-Minimal Phase A Synthetic Signal Audit R0 结果

状态：`REVISE / VALID`

日期：2026-07-26

最大权限：`SYNTHETIC_MECHANISM_AND_IMPLEMENTATION_EVIDENCE_ONLY`

## 结论

旋转补偿在 clean synthetic trial 上显著压低了 yaw/pitch 的纯旋转局部 expansion 泄漏，并保留了 scale up/down 与 rotation+scale 的解析 closing rate；roll 没有被要求制造本来不存在的高 RSR。clean 核心、15/30/60 FPS 一致性和 noise/blur/partial-occlusion 中可评价 trial 的误差门均通过。

Kill Gate A 仍判 `REVISE`，原因是预注册的逐 condition cell 覆盖率硬门失败。总体高覆盖率不能覆盖最差 cell：

- clean：`1661/1680 = 98.87%` 可评价，但 `pitch +30°/s @ 15 FPS` pure rotation 仅 `10/20`，低于 cell `0.80` 门；对应 mixed 为 `15/20`；
- stress：`758/840 = 90.24%` 可评价，但 partial occlusion 下 pure `pitch -30°/s` 为 `0/20`、pure `pitch +30°/s` 为 `1/20`，mixed pitch 两个方向为 `0/20` 与 `2/20`，低于 cell `0.70` 门；
- 101 个 trial 全部以 `EVALUABLE_PAIR_FRACTION_BELOW_0_80` 保留，不补抽、不回填零。

因此 Phase B、Replay Demo、深度、bearing/shear、双区域、风险场、Android 与新 UI 均未开放。下一独立边界只能是一次版本化、实现侧的 Phase A coverage revision；必须保留本轮全结果、协议、seed、trial、阈值和 gate，不得改条件或降门。

## 预注册与实现

机器协议在任何正式候选输出前锁定：

- protocol SHA-256：`d20e77f3ea5f7ac55376006f1d14feb0ffb5daffd10a42792912fb89cdb1b502`；
- 2520 trials：clean 1680、stress 840；
- 20 个固定 seed，15/30/60 FPS，0.6 s/trial；
- yaw/pitch/roll `±15/±30 deg/s`；
- scale `±0.15/±0.30 s^-1`，mixed scale `+0.15 s^-1`；
- trial 是统计单位，10,000 次 seed-cluster bootstrap；
- RSR/CRR 仅诊断，分母 `<0.03 s^-1` 时 `NOT_EVALUABLE`。

实现限定在：

```text
scripts/research/egomotion_compensated_looming/rcle_minimal/
同一 canonical Module 的 Phase A runner
```

使用 source-known rotation warp、Sparse LK、3×3 fixed-grid local affine、`e=0.5 trace(A)`、trial-level leakage/closing error、固定分母保护、自动图表与 hash-bound receipt。

## 主结果

| Component | 估计 | 95% CI | Frozen gate | 结果 |
| --- | ---: | ---: | ---: | --- |
| yaw/pitch compensated leakage median | `0.006676 s^-1` | `[0.006466, 0.006879]` | upper `≤0.030` | PASS |
| yaw/pitch paired leakage reduction median | `0.135966 s^-1` | `[0.135155, 0.136814]` | lower `≥0.015` | PASS |
| yaw/pitch improved trial fraction | `1.000` | `[1.000, 1.000]` | lower `≥0.80` | PASS |
| roll compensated leakage median | `0.002276 s^-1` | `[0.002132, 0.002520]` | upper `≤0.030` | PASS |
| scale closing MAE median | `0.002386 s^-1` | `[0.002243, 0.002618]` | upper `≤0.030` | PASS |
| rotation+scale closing MAE median | `0.002186 s^-1` | `[0.002061, 0.002323]` | upper `≤0.040` | PASS |
| closing sign accuracy | `1.000` | Wilson `[0.995989, 1.000]` | point/lower `≥0.98/0.95` | PASS |
| FPS expansion range median | `0.003378 s^-1` | `[0.003227, 0.003526]` | upper `≤0.030` | PASS |
| clean worst-cell coverage | `0.50` | point | `≥0.80` | FAIL |
| stress worst-cell coverage | `0.00` | point | `≥0.70` | FAIL |

noise、blur、partial occlusion 三个 profile 在可评价 closing trial 上均为 100% 符号正确，profile-level closing error 与 compensated rotation leakage 均通过冻结误差门。partial occlusion 的问题是 coverage，不是用剩余样本计算出的误差方向反转。

## Runtime

四 worker 离线正式运行期间的 host 计时：

| Module | Mean ms/pair | Median ms/pair | P95 ms/pair |
| --- | ---: | ---: | ---: |
| Rotation warp | `2.014` | `2.018` | `2.267` |
| Sparse LK | `37.214` | `36.154` | `48.571` |
| Local affine | `1.869` | `1.873` | `2.124` |
| Total including generation | `43.646` | `42.806` | `54.940` |

该计时受并行 worker 竞争影响，只是 host offline profile，不是 Android、实时或 Kill Gate 权限。

## 结果包与复算

本地论文级结果包：

```text
artifacts.local/evidence/rcle_minimal_r0/formal_run_r0/
artifacts.local/datasets/rcle_minimal_r0/formal_run_r0/
```

包含完整 trial manifest/metrics、主表、95% CI、flow 与 expansion 图、成功/失败案例、runtime、coverage、不可评价原因、代表 synthetic 序列、protocol 和 receipt。

独立复算命令由 canonical Module README 维护，使用 Phase A runner 的
`--validate-existing`，只读上述 evidence/dataset roots。

结果：`VALID`，2520 trials，verdict `REVISE`；receipt SHA-256 为
`14ed23e38bacc913207aaa56903a7b2cd3bebe52631338c4760f02dc5c2041ca`。

## 解释边界

本轮只说明在给定 synthetic 生成、source-known rotation 和 Sparse LK/3×3 affine 实现下，clean 核心机制成立但预注册 coverage 鲁棒性尚未闭合。它不证明真实相机 pose、滚动快门、真实遮挡、Bonn、真实用户、安全、Android 或生产有效性。
