# RCLE periodic self-motion counterfactual R2 generator geometry P1 R0

终态：`INTERVENTION_NOT_EVALUABLE / HOLD_P1`

权限：`EXECUTION_NOT_AUTHORIZED`

日期：2026-07-28（Asia/Hong_Kong）

## 结论

P1 已完成实现、身份物化和独立验证，但没有通过全部冻结门，因此不得进入 P2。
G01–G12 与 G14 通过；G13 `MONOTONIC_APPROACH_TRUTH` 失败。按照预先写死的
fail-closed 规则，本 evidence version 不换 seed、不降门、不调整轨迹，也不做
blur/low-texture calibration。

失败揭示冻结设计内部存在两个与 seed 无关的可执行性冲突：

1. guardrail 被定义为在约 10 秒、602 frame 内恰好产生 25% inverse-depth
   increase。其端点 log-radial expansion 上限为
   `ln(1.25) / duration = 0.022275–0.022279 /s`，严格低于 G13 的
   `>=0.05/s`；
2. `MONOTONIC_APPROACH_PLUS_PERIODIC` 保留冻结 periodic 6DoF 波形，9 个
   source-known persistent target points 的逐 pair depth monotonic fraction 为
   `0.0`，低于 `>=0.95`。纯 approach arm 的 monotonic fraction 为 `1.0`，
   说明失败不是 seed 或场景筛选造成。

因此当前只能把 generator、fixture、manifest 和失败回执保留为 implementation
asset / regression counterexample；不能把 13/14 项通过包装成
`GENERATOR_GEOMETRY_PASS`。

## P1 实现锁

[implementation lock](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_GENERATOR_GEOMETRY_IMPLEMENTATION_LOCK_R0_2026-07-28.json)
冻结：

- deterministic analytic ray/rectangle z-buffer v1；
- Python 3.11.9、NumPy 2.1.3、SciPy 1.17.1、OpenCV 4.13.0；
- MAIN / CAL / GUARD 三个 SHA-256 uint64 seed namespace；
- source-native `360×640` pinhole K、无畸变、`0.5–25 m` clipping；
- 四条 response-blind 602-frame trajectory 的 relative-SE(3)、60 Hz detrend、
  `0.7–3.0 Hz` FFT band 与 exact endpoint bridge；
- canonical UTF-8 JSON/JSONL manifest schema 和 producer / independent
  validator source hash。

80 个 main scene 与 8 个 guardrail scene 均使用 28 个显式 two-triangle rectangle
mesh，覆盖 near/middle/far depth，并通过独立 ray intersection、z-buffer、
reprojection 和 visibility 复算。它不是单平面、单 homography 或 painted 2D flow。

## 独立门结果

| 门 | 结果 | 独立复算摘要 |
| --- | --- | --- |
| G01 | PASS | 88/88 scene；valid depth 与三 depth band 全过 |
| G02 | PASS | 80/80 main scene；3×3 depth diversity 全过 |
| G03 | PASS | 四 block 各 10,000 samples；RMS `3.78e-14 px`，p99 `1.17e-13 px` |
| G04 | PASS | static fixture p99 `0 px`；240 static arm identity |
| G05 | PASS | Spearman `0.9562`；near/far flow ratio `9.0` |
| G06 | PASS | fixture residual `10.02 px`；四 block `20.14–55.66 px` |
| G07 | PASS | shared-bearing rotation p99 disagreement `5.97e-14 px` |
| G08 | PASS | 160 motion-sequence identities、865,440 visibility samples；mismatch `0` |
| G09 | PASS | 四 trajectory hash match；translation error `0 m`，rotation max `3.65e-8 rad` |
| G10 | PASS | endpoint translation/rotation error均为 `0` |
| G11 | PASS | 80/80 cluster，每个恰有六个唯一 arm |
| G12 | PASS | 160/160 motion-level group quality-geometry hash identity |
| G13 | **FAIL** | 16/16 sequence 均未同时达到全部冻结门 |
| G14 | PASS | 6 fixture manifest + 8 CAL replay frame，hash mismatch `0` |

独立 validator 不导入 generator producer，也不导入或运行 RCLE。

## 证据与边界

机器证据位于：

```text
artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/p1_geometry_r0/
```

其中包含 runtime / trajectory manifest、88-record all-seed geometry manifest、
analytic fixture ledger、四 block × 10,000 projective sample ledger、deterministic
replay ledger、producer receipt 和 independent receipt。

- implementation lock SHA-256：
  `3d45daebf95f190024197b29543989067d86655fd0621038bd96f0104565bcad`；
- independent receipt SHA-256：
  `72e0b8e042be9eb6208389eb8d83e9e9e4ad28e54ec82f7064b5387cc1abd279`。

本阶段没有：

- 读取或运行 RCLE output；
- 校准 blur 或 low-texture strength；
- 运行 8 条 performance preflight；
- 运行 480 main 或 16 guardrail 正式序列；
- 修改 R3、`0.01/s`、three-pair 或 PairState；
- 访问 sequence16，运行 CoTracker、Android 或 realtime。

## 后继规则

当前不得另立 P2。若未来要继续，只能先另立、审查并版本化修订 geometry spec，
明确解决 G13 的 endpoint-rate 与 monotonic-plus-periodic estimand 冲突；不得在本
R0 中改 gate、换 seed 或用已见失败结果择优修补。
