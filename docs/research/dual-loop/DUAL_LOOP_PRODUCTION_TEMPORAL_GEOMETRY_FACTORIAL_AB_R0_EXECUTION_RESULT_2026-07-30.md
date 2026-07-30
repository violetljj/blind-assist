# Production Temporal Geometry Factorial A/B R0 执行结果

日期：2026-07-30（Asia/Hong_Kong）

```text
EXECUTION_STATUS: COMPLETE
PRODUCER_STATUS: COMPLETE
INDEPENDENT_VALIDATION: VALID
SEAL_STATUS: SEALED
EVALUATION_STATUS: VALID
SCIENTIFIC_TERMINAL: NO_INCREMENT
EXECUTION_AUTHORITY: CONSUMED / NO_RERUN
CONFIRMATION: NOT_AUTHORIZED
CLAIM_CEILING: TWO_SESSION_ONE_CONTEXT_DEVELOPMENT_DIRECTIONAL_SCREEN_ONLY
```

## 结论

在冻结的两个 CrowdBot decision session 上，现有生产
`TemporalRiskTracker` 的 object-detector temporal geometry 相对完全同构的
neutralized 分支，没有改变任何实际可交付提醒、事件或距离/风险等级。两个分支在
全部 8 个可评分正例和 7 个负窗上的结果逐项相同，因此终点为
`VALID / NO_INCREMENT`。

这个结果拒绝“当前生产 temporal geometry 已经形成有效双环决策增量”的主张；
它不等于 temporal geometry 在所有数据或实现上无效，也不授权在相同 evidence
version 上调参、救援或重跑。

## 执行完整性

- 设备：`SM-S9280 / SM8650 / Android 16 / SDK 36 / arm64-v8a`。
- backend：`qualcomm_qnn_htp`，QNN Maven `2.47.0`，runtime `[0,24,0]`。
- 冻结输入：2 sessions，4,422 帧，2,612,679,375 bytes。
- detector：每帧严格一次，共 4,422 次；A/B trace 共 8,844 行。
- producer wall：124,701 ms；failure count `0`；truth 未在 producer/validator
  阶段打开。
- 正式 marker 已消费；独立 validator 的 branch-pair mismatch 为 `0`，
  `errors=[]`；随后原子 seal 才授权 truth join。

## 冻结终点

| 指标 | neutralized A | current production B | B−A |
| --- | ---: | ---: | ---: |
| 可评分正例召回 | 8/8 | 8/8 | 0 |
| missed positives | 0/8 | 0/8 | 0 |
| premature positive events | 6/8 | 6/8 | 0 |
| false-alert negative windows | 7/7 | 7/7 | 0 |
| unscored trigger rows | 325 | 325 | 0 |
| 首次有效提醒正增益事件 | 0/8 | 0/8 | 0 |
| pooled median normalized gain | 0 frame | 0 frame | 0 |
| session median gain | 0 / 0 ns | 0 / 0 ns | 0 / 0 ns |
| paired correctness delta | — | — | 0/15 |

8 个正例的首次有效提醒 timestamp 均逐项相同。两个 session 的 early-response
median 均为 0；risk-discrimination 在每个 session 的 improvement count 均为 0。
全部 common 与 session guardrails 通过，但 early-response 和
risk-discrimination 的增量谓词都不通过，所以唯一终点是 `NO_INCREMENT`。

`F1A-P-007` 与 `F1A-P-009` 因冻结 valid interval 内零 RGB 帧，继续保持
`TEMPORAL_SCORING_NOT_EVALUABLE`，没有被静默移入分母。

## Seal 后描述性定位

该段只用于后继路线定位，不属于冻结终点，也不能作为新有效性主张：

- 4,422 个 paired frames 中，A/B `raw_risk` 有 4,396 帧不同，
  `stable_risk` 有 4,419 帧不同；
- `approach_trend` 有 3,285 帧不同，`risk_score/total_score` 有 973 帧不同；
- 但 `level`、`proximity`、`feedback_triggered` 与 `risk_event` 的差异都为 0。

因此信号并非完全没有进入 risk object；它在进入实际提醒前失去决策作用。当前
CrowdBot 检测几乎全部落入 `NEAR/CRITICAL`，而现有 motion fusion 主要做单向提升，
无法校正两臂共同出现的 7/7 负窗误触。后继不应继续只调
`approachScoreBoost` 或重复同构 A/B。

## 身份与收据

- implementation commit：
  `2c53e89a67ec7848a7d2290ebf9e627f6bc96ff6`
- activation commit：
  `0b9181c9c39b0e898242c00486f38fb654676bf1`
- implementation lock：
  `d7383b9339d46935599d1f0da9bd163b78dd159050e8409a0578969ef9bb23de`
- activation：
  `257fa996be12f9b01b919aa6a27c01d55bfc84f85228ab16ab398f2c83807546`
- formal marker：
  `a644e5a2ab68ad569e081df69d94d01ff80102573cef5863adf2ea411421ce18`
- producer receipt：
  `7aa07b6ae8bbd382018cc2fa6df5f62c78de54487e1baa28441ea5e4ee157cac`
- trace：
  `37b267c56c5aad710418eb6b099cf32b910214e9eafd8d6c5b02d3a6ddf9b0e5`
- independent validation：
  `30d2e00c2c0bb7b5ccd7dbeed66a59e566e70d1be40555cd40ab7288f094719b`
- seal：
  `6c1b1e24da9f0a34599124b285b306c1299cb4b4b67f3e090e11ea5692a7c5e3`
- evaluation：
  `56a1ad9bd9cf502cc385e561987caf88692596735d97312aad52ad05a5efcbcf`

## 后继

按预先声明的回退路线，下一步是另立统计修复后的
`D0_EGOMOTION_ERROR_ATTRIBUTION_R1`：它只在已烧毁的 REveL LITE R2 evidence 上
给出 `EGO_CANARY_PRIORITY / TEMPORAL_TREND_PRIORITY / NO_PRIORITY_IDENTIFIED`
的 operational routing，不再声称识别 dominant causal mechanism。
只有新的独立合同、实现复核与 activation 才能执行；本结果不自动授权
EVIMO2v2、JRDB、Confirmation、Android 产品接线或安全主张。
