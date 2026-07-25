# USTRF bbox-route 归因 R1 结果（2026-07-25）

状态：`VALID / STOP_ROUTE_CONDITIONED_USTRF_DOWNGRADE_TO_DETECTOR_BASELINE`

最大权限：`RESEARCH_SIGNAL_ATTRIBUTION_ONLY`

## 结论

`A_bbox_matched_route` 的主 q90 `12/15` 已逐帧零偏差复现，但不能归因于“正确路线稳定增加了 bbox 信号”。

matched bbox 没有稳定超过 uniform、shuffled 与 bbox-only 三个对照，并且 `lilocbench_dynamics_0_front` 的正窗减负窗中位差为 `-0.771457`，直接触发预先规定的停止规则。因此：

- 正式停止当前 route-conditioned USTRF；
- 将现有 bbox 能力降级为普通 detector baseline；
- 不进入 causal lifecycle；
- 不补中等规模独立验证集；
- 不做 architecture convergence；
- 不扩到 120 episode；
- 不调 detector、bbox、route、窗口汇总或报警阈值回救。

这不是“detector 已获得产品授权”。它只说明当前 15 对窗口内的聚合 `12/15` 很大程度上与 bbox / 来源结构共同出现，而现有 matched route 没有形成跨来源稳定的独立增量。

## 冻结范围

本轮只复用：

- 15 个正窗口；
- 15 个同来源、等长负窗口；
- 4,594 个冻结 frame identity；
- G1b `4594/4594` canonical semantic parity；
- 上一轮四臂探针的 exact common-eligible frame subset（4,108 帧）；
- 同一份 `post_nms_detections_canonical_320` person bbox 与未改写 confidence；
- 同一 past-only matched route；
- 同来源 window ID cyclic shift-one shuffled route。

上一轮 dense 结果必须先为 `STOP_CURRENT_DENSE_USTRF_EXPRESSION`。本轮不读取新来源，不生成新 bbox，不运行 metric-depth dense、tracker、TTC、lifecycle、Android 或 App 链。

## 四个 arms

同一 bbox confidence field 定义为：每个冻结 person bbox 在其矩形区域内保持原 confidence。

| Arm | 唯一变化 |
| --- | --- |
| `A_bbox_matched_route` | 对 matched route patch 应用原 support-overlap 算子 |
| `B_bbox_uniform_route` | 将完整 source frame 作为 equal route support |
| `C_bbox_shuffled_route` | 对同来源 cyclic-shuffled route patch 应用同一算子 |
| `D_bbox_only` | 移除 route support，只取同一 field 的最大 person confidence |

每帧 matched 分数与上一轮 A arm 比较，mismatch 为 `0`。窗口只汇总 q50/q90/q95 连续分数；主口径保持 q90。没有选择报警阈值。

## 主结果

### 各 arm 的正负配对排序

| Arm | q90 W/T/L | 全局中位差 | dynamics_0 中位差 | lt_changes 中位差 |
| --- | ---: | ---: | ---: | ---: |
| bbox + matched route | `12/1/2` | `0.364860` | `-0.771457` | `0.473516` |
| bbox + uniform route | `11/0/4` | `0.052502` | `0.029861` | `0.154376` |
| bbox + shuffled route | `9/2/4` | `0.045066` | `-0.868812` | `0.236922` |
| bbox-only | `8/0/7` | `0.001689` | `-0.024074` | `0.045360` |

matched 的 12 次 win 中，两个来源分别为：

- `lilocbench_dynamics_0_front`：`1/3` win，median delta `-0.771457`；
- `lilocbench_lt_changes_dynamics_0_front`：`11/12` win，median delta `0.473516`。

所以 pooled `12/15` 由 `3 + 12` 的来源不平衡显著影响，不能替代 worst-source 判断。

### matched 相对对照的直接增量

直接增量定义为每对窗口的：

```text
(matched positive - matched negative)
-
(control positive - control negative)
```

| 对照 | q90 matched 直接 W/T/L | 中位增量 | Wilson 95% 下界 | 稳定超过 |
| --- | ---: | ---: | ---: | --- |
| uniform route | `11/0/4` | `0.325776` | `0.480` | 否 |
| shuffled route | `9/1/5` | `0.129304` | `0.357` | 否 |
| bbox-only | `9/0/6` | `0.222806` | `0.357` | 否 |

三项都没有达到冻结的 `>=12/15` 与 Wilson 下界 `>0.5`。uniform 与 bbox-only 的 dynamics_0 来源中位增量也分别为负；bbox-only 在 q50 的全局中位增量为负。

## 单一决策规则

只有同时满足以下两项才允许进入 causal lifecycle：

1. matched bbox 在每个对照上都稳定胜出；
2. matched bbox 的主 q90 正负窗中位差在每个来源都严格为正。

实际：

- `stable_exceeds_all_three_controls = false`；
- `matched_primary_median_positive_in_each_source = false`。

终态唯一确定为：

```text
STOP_ROUTE_CONDITIONED_USTRF_DOWNGRADE_TO_DETECTOR_BASELINE
```

## 证据与复算

机器证据：

- `artifacts.local/evidence/ustrf-bbox-route-attribution-r1/report.json`
- `artifacts.local/evidence/ustrf-bbox-route-attribution-r1/frame-scores.jsonl`

独立第二进程全量重算：

```text
status = VALID_REPLAY_MATCH
report_sha256 = 02c49982f9a3abb5f7bfe93b8092005c38ac278593655661dc2b951e7f07d1c6
frame_scores_sha256 = 860d27fb70e4ec28c82e86fcd3fd91b6a2e86573f6762d4a4425326b07ecf2ef
decision = STOP_ROUTE_CONDITIONED_USTRF_DOWNGRADE_TO_DETECTOR_BASELINE
```

Focused tests：

```text
6 tests passed
```

## 权限边界

本结果不授权：

- 报警阈值；
- tracker / TTC / lifecycle；
- 新数据或 120 episode；
- Kotlin / Android / App 架构改动；
- shadow、人体、安全或生产；
- 将 ordinary detector baseline 解释为真实提醒效果已通过。

若未来重启 route-conditioned 研究，必须是新的、独立预注册并具有新增路线特异性证据的假设；不得在本轮 15 对窗口上调 route 宽度、bbox 权重、quantile、来源规则或阈值回救本终态。
