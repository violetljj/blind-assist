# RCLE 研究主线

状态：`current / TERMINAL_HOLD`

最后核验：2026-07-28 17:17（Asia/Hong_Kong）

## 当前结论

RCLE-RF 仍是 BlindAssist 的论文研究主线，但当前没有获授权的自动后继实验。

最新阶段 `RCLE_RGB_SEGMENT_CONFIRMATION_R1` 已闭合为：

`RGB_SEGMENT_CONFIRMATION_R1_NOT_EVALUABLE / VALID_FAIL_CLOSED_TERMINAL`

两个冻结真实片段都未在各自已签名、one-shot、预算受限的 opaque identity extraction 内闭合 RGB identity。因此：

- eligible RGB frame 为 `0`；
- pixel decode 与 RGB algorithm call 都为 `0`；
- alignment pair denominator 为 `0`，所有指标为 `null`，不是 `0`；
- 两个 claim 均已消费且禁止重试；
- 本终态不证明 RGB 算法成功，也不证明 RGB 算法失败。

权威结果是 [RGB Segment Confirmation R1 result](RCLE_RGB_SEGMENT_CONFIRMATION_R1_RESULT_2026-07-28.md)，绑定 terminal SHA-256：

`03fbac1d815072639b00393cb058f31aacba5de0b0a270c6d440f2e0bab10753`

## 两个冻结片段的终态

| 冻结片段 | 原角色 | immutable terminal | RGB frames | 约束 |
| --- | --- | --- | ---: | --- |
| OpenLORIS `corridor1-1:w004` | positive approach | `INVALID_IDENTITY_EXTRACTION_CLOSE_ATTEMPT / URLError` | 0 | claim consumed；禁止重试、换 URL 或整源回退 |
| DLR `extreme_geometry/hexagon_01:w001` | below-trigger reference | `SEGMENT_IDENTITY_NOT_EVALUABLE / DLR_BYTE_BUDGET_EXHAUSTED_OR_RGB_GUARD_ABSENT` | 0 | claim consumed；禁止扩预算、换窗或整源回退 |

OpenLORIS 与 DLR 仍然 source-role confounded；没有执行 positive/below discrimination。MVSEC 未访问。

## 独立复核

- OpenLORIS terminal independent review：`PASS`，SHA-256 `99830376dcba266f607df636fd92e44c5f6164856d0d599b998f171fbe657cd9`
- DLR terminal independent review：`PASS`，SHA-256 `d56edd0d6dd1afed4e652fd4c0ce4a3ad92f577e9e22628b356ad43e0b58c69f`
- protocol terminal validation：`37/37 PASS`，SHA-256 `bbab3cd47e88108896503fbf2a8f52184ade5786ca91b0215b63b718c93bd64d`
- final independent review：`PASS`、无阻断项，SHA-256 `d93267d3cd13d6da4f45a1a0f507339ba32817f3f45d02c365764e7a127b221a`

对应本地 evidence 路径固定在：

`artifacts.local/evidence/rcle_rgb_segment_confirmation_r1/`

不得通过读取 raw claim、progress、partial transport payload 或未复核中间文件改写上述终态。

## 当前权限

| 能力 | 当前 authority |
| --- | --- |
| 两个 exact segment 的 fail-closed transport terminal | `VALID` |
| RGB segment mechanism evidence | `NOT_EVALUABLE` |
| positive/below discrimination | `NOT_AUTHORIZED` |
| performance / generalization | `NOT_AUTHORIZED` |
| RGB algorithm execution | `NOT_AUTHORIZED` |
| host replay | `NOT_AUTHORIZED` |
| Android / App 集成 | `NOT_AUTHORIZED` |
| 真人、产品、安全或生产结论 | `NOT_AUTHORIZED` |

BlindAssist 仍是论文、毕业设计、院内演示和竞赛原型，不面向视障人士独立使用，也不形成真实用户有效性或安全认证结论。

## 禁止事项

- 不重试或替换两个已消费 claim；
- 不扩大 DLR byte budget，不用整源下载、换窗、换源或 MVSEC 回救本 R1；
- 不把 null 指标当成零，不填补 zero-frame ledger；
- 不把 transport/identity 失败写成 RGB algorithm failure；
- 不从历史 synthetic、geometry canary、development RGB 或设计审查推导 confirmation、performance、Android、product 或 safety authority；
- 不在当前 working tree 的终态包完成窄交付前启动新的 formal claim 或 protected-outcome access。

## 下一步

当前无自动研究后继。

如果未来确有新的科学假设、独立来源或新的可执行 identity contract，必须另立新版本协议、重新冻结 selection/identity/access/claim/validator/停止条件，并与本 R1 已消费 claim 完全隔离。不得从本终态直接进入算法执行、性能资格、host replay 或 Android。

仓库工程上的唯一近期待办，是在不吸收其他共享工作树改动的前提下，对本终态 result、实现、tests、review 与 current 做窄范围交付；用户未明确要求提交推送前，不自动执行 Git 交付。

## 历史入口

- [2026-07-27 原 current 快照](RCLE_CURRENT_SNAPSHOT_2026-07-27.md)：保存 Phase A、Phase B、Progressive Discovery、geometry/RGB canary 与来源准入的完整历史叙事；其中任何“当前”“下一步”均已失效。
- [渐进式研究治理](../../RESEARCH_GOVERNANCE.md)：定义 Discovery/Canary/Development/Confirmation/Deployment 与 fail-closed 规则。
- [RCLE 前序历史索引](../ustrf-sc/README.md)：route-conditioned USTRF 与 egomotion-compensated looming 历史证据，不自动产生当前 authority。

本页是 RCLE 动态阶段、终态、权限、禁止事项和下一步的唯一 current 真源。其他 README、报告、日期化结果与 handoff 只能链接本页或保存当时快照，不得复制新的动态结论。
