# HFTF Stage C semantic-independent label readiness result D0

日期：2026-08-01

终态：`D0_SEMANTIC_INDEPENDENT_LABEL_READINESS_SUPPORTED`

## 1. 结论

冻结的 depth-only ground-plane + horizontal-support reader 在两个 consumed EgoWalk
calibration sources 上通过全部 mechanics、plane、profile、opportunity 与 determinism
顺序门。

这支持 semantic-independent geometry proxy label readiness，不支持 hazard truth、
自然 prevalence、student agreement/effect 或助盲事件效果。

## 2. 报告绑定

- report：
  `artifacts.local/evidence/hftf/stage-c-d0-label-readiness-20260801/label_readiness.json`
- SHA-256：
  `8a267e07e48f70abbfe9e2d184e53ca5464331fd848e256aebd9b1cb2239952b`
- protocol commit：`9ed3bc8`
- runner commit：`f277ed8`

## 3. Source metrics

| metric | outdoor `2024_08_15...` | indoor `2024_07_11...` |
| --- | ---: | ---: |
| formal frames | 131 | 134 |
| plane known | `131/131` | `134/134` |
| median plane inlier fraction | `.6543` | `.5556` |
| median height error m | `.1102` | `.1828` |
| height error P90 m | `.2158` | `.2164` |
| direction known fraction | `.9176` | `.7821` |
| known no-risk cells | 594 | 524 |
| risk-proxy cells | 7 | 0 |
| UNKNOWN→SAFE | 0 | 0 |

cohort 7 个 risk-proxy cells 分布于 7 个 frames、4 个 directions
`[-30,0,+15,+30]°`，越过冻结的 `2/2/2` opportunity 门。它们包括 planter/curb、
horizontal support rise/drop 等 geometry proxy；没有 reference truth，不能解释为
七个真实危险。

七个 structural canaries 全过：flat、`.25 m` rise、`.20 m` drop、三 section
UNKNOWN、vertical wall rejection、missing-depth UNKNOWN 与 identical-input
determinism。完整 payload 第二次运行 byte-exact。

## 4. 下一边界

唯一新权限：

`FREEZE_FRESH_SOURCE_LABEL_OPPORTUNITY_AND_STUDENT_CANARY_PROTOCOL_ONLY`

在 student 前仍需把 `[0,.4,.8] s` future label mechanics 做到 phone-causal：
history pose 可决定 advected origin，future depth/pose 只作为 teacher observation，
不能让真实 future path 选择输出方向或 origin。随后 fresh train/dev/held-out 必须按完整
trajectory/session 分割。

当前不授权 fresh acquisition、teacher corpus、student training/effect、研究主线、
Android/App 或安全/产品 claim。
