# RCLE-Minimal Phase A Coverage Revision R1 结果

状态：`STOP_CURRENT_IMPLEMENTATION / VALID`

机器判定：`REVISE`（coverage component failed）

日期：2026-07-26

最大权限：`SYNTHETIC_MECHANISM_AND_IMPLEMENTATION_EVIDENCE_ONLY`

## 结论

唯一允许的 implementation-only coverage revision 已完整运行。它保留了
R0 协议 SHA-256 `d20e77f3…1b502`、全部 20 seeds、2520 trial IDs、输入
序列 hashes、数值阈值、gate、统计方法和 101 个 R0 负结果；R0 receipt
`14ed23e3…041ca` 仍可独立复算为 `REVISE / VALID`。

R1 修复有效但不充分：

- clean 从 `1661/1680` 提高到 `1680/1680`，worst cell 从 `0.50` 提高到
  `1.00`；
- stress 从 `758/840` 提高到 `810/840`；
- affine residual abstention 从 R0 的 raw/comp `1505/1424` 降为 `0/0`；
- 仍有 30 个 partial-occlusion pitch trial 不可评价，stress worst cell
  只有 `12/20 = 0.60`，低于冻结的 `0.70` 门。

因此正式 summary 仍为 `REVISE / VALID`。但 R0 协议同时冻结了“单次版本化
REVISE rerun 仍失败则停止”的语义，所以研究层终态是
`STOP_CURRENT_IMPLEMENTATION / VALID`：不得进行第二次 coverage revision，
不得降低 support/hull/residual/common-cell/pair/coverage 门，也不得进入
Phase B、Replay Demo、Bonn、Android、Risk Field 或任何安全/生产路径。

## 冻结修订

R1 只改两处实现：

1. forward LK 保持不变；backward cycle 在原最大 pyramid `3` 内执行
   `3→0` 多层验证，以 previous point 初始化，并对每个点保留最小有限
   forward-backward error；接受门仍为 `≤1.0 px`；
2. 每个 3×3 cell 使用 deterministic affine RANSAC 选择 consensus，
   reprojection threshold 复用原 `0.75 px/frame`；随后仍按原模型重拟合，
   并原样应用 support `≥12`、hull `≥0.10`、condition `≤1000`、
   residual `≤0.75 px/frame`。

没有修改 generator、feature detector、LK forward 参数、grid、common
`5/9`、trial pair fraction `0.80`、coverage gate、seed、trial 或退化矩阵。
R1 使用新 package、runner、tests 和输出根；R0 源码与产物未被覆盖。

## 主结果

| Component | R1 估计 | Frozen gate | 结果 |
| --- | ---: | ---: | --- |
| yaw/pitch compensated leakage median | `0.006296 s^-1`，95% CI `[0.006168, 0.006340]` | CI upper `≤0.030` | PASS |
| yaw/pitch paired leakage reduction median | `0.140708 s^-1`，95% CI `[0.139267, 0.142160]` | CI lower `≥0.015` | PASS |
| roll compensated leakage median | `0.002220 s^-1` | CI upper `≤0.030` | PASS |
| scale closing MAE median | `0.002398 s^-1` | CI upper `≤0.030` | PASS |
| rotation+scale closing MAE median | `0.002202 s^-1` | CI upper `≤0.040` | PASS |
| closing sign accuracy | `1.000` | point/Wilson lower `≥0.98/0.95` | PASS |
| FPS expansion range median | `0.003410 s^-1` | CI upper `≤0.030` | PASS |
| clean coverage | `1680/1680 = 1.000` | pooled/cell gates | PASS |
| stress coverage | `810/840 = 0.9643` | pooled gates | PASS |
| stress worst-cell coverage | `12/20 = 0.60` | `≥0.70` | **FAIL** |
| noise/blur/partial-occlusion error guardrails | all profile gates pass | frozen profile gates | PASS |

剩余四个失败 cell 全部是 partial occlusion pitch：

| Motion | Direction | Evaluable / planned | Point |
| --- | ---: | ---: | ---: |
| pure rotation | `-30°/s` | `12/20` | `0.60` |
| pure rotation | `+30°/s` | `12/20` | `0.60` |
| rotation + scale up | `-30°/s` | `13/20` | `0.65` |
| rotation + scale up | `+30°/s` | `13/20` | `0.65` |

30 个失败 trial 全部保留为
`EVALUABLE_PAIR_FRACTION_BELOW_0_80`。失败 trial 中不再出现 affine residual
超门；剩余链路是 consensus 后 `LK_TRACK_SUPPORT_BELOW_12` 与
`TRACK_HULL_COVERAGE_BELOW_0_10` 使 common grid `<5/9`。不得把 RANSAC 前
tracked count 当作合格 consensus support，也不得把不可评价回填为零。

## 完整性与复算

R0/R1 `trial_manifest.jsonl` 逐行完全一致，证明 seed、trial、base/sequence
hash 和 pair inventory 没有漂移。R1 结果包：

```text
artifacts.local/evidence/rcle_minimal_r1/formal_run_r1/
artifacts.local/datasets/rcle_minimal_r1/formal_run_r1/
```

独立复算为 `VALID / REVISE`，2520 trials；R1 receipt SHA-256：

`d5edb9528abfa6d79b973bddfed5f4234795262fb303258c9e1a9e2628ca2b15`

验证同时重跑了 R0 receipt，仍为原 SHA-256
`14ed23e38bacc913207aaa56903a7b2cd3bebe52631338c4760f02dc5c2041ca`。
全 Module 67 tests 通过；新增 R1 tests 覆盖低帧率 boundary support、
partial-occlusion pitch、determinism、affine outlier consensus，以及
support/hull 负回归。

## 精确边界

当前实现已按冻结停止语义终止。没有自动的“再修一次”或 Phase B 后继。
若未来重新研究，必须由新的明确授权建立独立、结果前冻结的信号/观测模型
假设；它不能复用本轮 2520 trials 做第二次阈值或实现选择，也不能改写 R0/R1
负结果。无论未来方向为何，真实来源、Android、人体与生产权限仍需独立门禁。
