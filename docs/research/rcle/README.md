# RCLE 研究主线

状态：`current / PUBLIC_DATA_CONFIRMATION_CLOSED_NOT_EVALUABLE`

最后核验：2026-07-28 18:22（Asia/Hong_Kong）

## 当前结论

RCLE-RF 仍是 BlindAssist 的论文研究主线。历史公开数据搜索与确认工作已经正式收束：

```text
CURRENT PUBLIC-DATA CONFIRMATION CONTRACT: CLOSED / NOT_EVALUABLE
ALGORITHM STATUS: DEVELOPMENT_EVIDENCE_PROMISING
EXTERNAL CONFIRMATION STATUS: PENDING
NEXT ALLOWED STUDY: RCLE_ECOLOGICAL_RESPONSE_DISCOVERY_R0
NEXT STUDY EXECUTION: NOT_AUTHORIZED / NOT_SCIENTIFICALLY_ADMITTED
```

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

OpenLORIS 与 DLR 仍然 source-role confounded；没有执行 positive/below discrimination。R1 终止时 MVSEC 未访问；其后另立的 MVSEC R1/R2 身份 claims 均已消费并以 `MVSEC_RGB_IDENTITY_NOT_EVALUABLE` 结束，不能回写或救援本 R1。

## 独立复核

- OpenLORIS terminal independent review：`PASS`，SHA-256 `99830376dcba266f607df636fd92e44c5f6164856d0d599b998f171fbe657cd9`
- DLR terminal independent review：`PASS`，SHA-256 `d56edd0d6dd1afed4e652fd4c0ce4a3ad92f577e9e22628b356ad43e0b58c69f`
- protocol terminal validation：`37/37 PASS`，SHA-256 `bbab3cd47e88108896503fbf2a8f52184ade5786ca91b0215b63b718c93bd64d`
- final independent review：`PASS`、无阻断项，SHA-256 `d93267d3cd13d6da4f45a1a0f507339ba32817f3f45d02c365764e7a127b221a`

对应本地 evidence 路径固定在：

`artifacts.local/evidence/rcle_rgb_segment_confirmation_r1/`

不得通过读取 raw claim、progress、partial transport payload 或未复核中间文件改写上述终态。

## 历史数据工作收束

- [历史数据工作收束报告 R0](RCLE_DATA_WORK_CLOSURE_R0_2026-07-28.md)：冻结所有旧终态、失败原因、资产去向、停止事项和下一阶段边界。
- [数据能力与访问状态表 R0](RCLE_DATA_CAPABILITY_MAP_R0_2026-07-28.csv)：统一记录 source/capture/window、outcome access、当前证据角色、claim ceiling 与建议处置。

后续 transport/identity 失败只作为历史资产：

- OpenLORIS R2：`NOT_EVALUABLE_PARTIAL_QUARANTINED`；
- DLR R2：claim 已占用，worker 已消失且没有合法 terminal，固定为 `ORPHAN/HOLD`，禁止重启或续扫；
- MVSEC R2：`indoor_flying2:w004` 的 200 个 geometry timestamp 仅 63 个通过冻结 5 ms pairing，137 个超门；`indoor_flying1:w002` 未启动；像素与算法调用均为零。

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
- 不继续公开数据漫游，不以新来源、新窗口、追加下载或新 claim 默认救援旧终态；
- 不启动新的 formal claim、protected-outcome access、算法执行或 Android。

## 下一步

当前无自动研究后继。唯一允许被提出、但尚未启动的下一实验是：

`RCLE_ECOLOGICAL_RESPONSE_DISCOVERY_R0`

它只允许比较 `BBox Growth / Uncompensated Local Expansion / Rotation-Compensated Local Expansion` 在自然第一视角行走中的响应规律与失败模式，不预设 RCLE 获胜。选择来源、下载数据、运行算法、调参、性能资格、host replay 与 Android 均未被本入口授权。

收束审查期间检测并停止了并发本地 pilot 进程；其未授权输出内容未被本收束任务读取，
只按 `UNAUTHORIZED_EXECUTION_ARTIFACT / NOT_ADMITTED` 归档。这不构成下一研究已启动，
也不授权续跑、解释或追认。

## 历史入口

- [2026-07-27 原 current 快照](RCLE_CURRENT_SNAPSHOT_2026-07-27.md)：保存 Phase A、Phase B、Progressive Discovery、geometry/RGB canary 与来源准入的完整历史叙事；其中任何“当前”“下一步”均已失效。
- [渐进式研究治理](../../RESEARCH_GOVERNANCE.md)：定义 Discovery/Canary/Development/Confirmation/Deployment 与 fail-closed 规则。
- [RCLE 前序历史索引](../ustrf-sc/README.md)：route-conditioned USTRF 与 egomotion-compensated looming 历史证据，不自动产生当前 authority。

本页是 RCLE 动态阶段、终态、权限、禁止事项和下一步的唯一 current 真源。其他 README、报告、日期化结果与 handoff 只能链接本页或保存当时快照，不得复制新的动态结论。
