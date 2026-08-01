# Objective image-space candidate increment D0 result

终态：

```text
COMPLETE / VALID /
STOP_FIXED_PIDNET_OBJECTIVE_CANDIDATE_NO_ROBUST_INCREMENT /
TIMING_NOT_EVALUABLE_ONSET_INCOMPLETE /
KEEP_CURRENT_YOLO
```

机器结果：
[OBJECTIVE_IMAGE_SPACE_CANDIDATE_INCREMENT_D0_RESULT_2026-08-02.json](OBJECTIVE_IMAGE_SPACE_CANDIDATE_INCREMENT_D0_RESULT_2026-08-02.json)。

## 结论

固定 PIDNet-S seed-20260801 的原生 `{blocking_obstacle, boundary_level_change}`
argmax 确实在 YOLO 框外增加了客观 source-mask coverage，但没有形成可接受的稳健系统
增量。关闭 exact seed+raw-candidate operator，保持当前 YOLO；不增加 confidence/面积
gate、组件分类器、时序 latch、parallel-curb 特例或新风险规则。

这不是性能失败。fixed semantic budget 和 host objective-operator cost 均过门；否决来自
boundary coverage、residual component recall 和 false-activation 分布。

## 数据与防火墙

主集为 30 source sessions / 1,920 observations。它与 PIDNet 的 520-frame train/dev
按 source session 隔离，但已被前序 RISKSEG 研究消费，因此本结果只是
`VALID_NEGATIVE_CONSUMED_DEVELOPMENT_RESULT`。

执行用 objective-only manifest 已剥离
`positive/bucket/alertable/passed/event_candidate_id/parent_event_id`。evaluator
未读取旧 actionability、风险、反馈、mask adapter 或事件链。独立单位是 30 个
source sessions；1,920 帧、像素和组件只是重复观测。

## 三臂 coverage

| 口径 | recall |
|---|---:|
| A YOLO-only | 0.219061 |
| B PIDNet-only | 0.089849 |
| C YOLO + PIDNet residual | 0.288441 |
| C − A | +0.069380 |

总 recall gain 通过冻结的 `>=.05` 门，30/30 sessions 都是非负增量，session median
gain 为 `.051848`。但只有 16/30 sessions 达到 `>=.05`，同时满足 gain 与
added-FP 门的只有 8/30。

YOLO-uncovered residual candidate 的 pixel precision/recall 只有
`.278398/.088842`。17,475 个 residual truth components 中命中 4,988 个，
component recall `.285436 < .50`；26,349 个 predicted components 中有 15,105 个
完全不命中 truth，平均 `7.867188/frame > 3`。

## 类别和 false activation

`blocking_obstacle` recall gain 为 `+.059330`，通过 `>=.02`；但
`boundary_level_change` 只有 `+.004039`，失败。换言之，额外信号主要继续来自实体类，
没有解决台阶/路沿/落差这一关键表征缺口。

pooled added FP area 为 `.046586 <= .05`，但分布不稳：session P90 为
`.127410 > .05`，只有 21/30 sessions 守住 `.05`。因此 pooled 平均数会掩盖部分
source sessions 的高额误激活。

## 客观连续量

候选 valid-area fraction 的 P10/P50/P90 为 `.005943/.053229/.229726`；
middle-third fraction 为 `.000718/.018055/.092375`；冻结梯形 ROI fraction 为
`.001782/.030925/.128211`。相邻 mask IoU P10/P50/P90 为
`.019928/.278234/.560791`，面积变化 P10/P50/P90 为
`-1371.8/+1/+1436.2` pixels。

raw candidate 在多数序列里非常“持久”：每 session 非零帧比例 P10/P50/P90 为
`.975/1/1`，最大连续非零 run 的 P10/P50/P90 为 `46.2/50/102` 帧。但这种持续性
没有带来可靠 truth recovery；它同样包含持续 false activation。因此
“持续存在”本身不能充当风险证据，也没有在看到输出后补一个 component-track threshold。

## 固定预算

- host objective operator P95：`3.729 ms <= 30 ms`；
- host PIDNet inference P95（仅描述）：`11.675 ms`；
- SM-S9280 frozen total P95：`77.374 ms <= 100 ms`；
- SM-S9280 inference P95：`5.198 ms`；
- 10 分钟末/初 P95：`1.07624x <= 1.20x`；
- thermal/failure：`0/0`，QNN 完整委派。

预算 PASS 不覆盖客观 utility 的四项失败门。

## Timing

冻结 onset 要求先有 5 个连续零 truth observations，再有至少 3 个连续非零 truth
observations，并至少取得 12 个独立 positive sessions。当前只有 4 个，因此 timing
固定为 `NOT_EVALUABLE_ONSET_INCOMPLETE`，不得从本结果声称“更早覆盖”。

## 最终规则

1. 固定 PIDNet raw candidate operator 关闭；
2. 不换 seed、不重训、不在已消费 cohort 上改阈值或加规则救援；
3. 不修改默认 App；
4. 若未来继续，必须提出不同因果表征，并先冻结新的 onset-complete、
   session-disjoint 自然 cohort；本 D0 不自动授权 successor。
