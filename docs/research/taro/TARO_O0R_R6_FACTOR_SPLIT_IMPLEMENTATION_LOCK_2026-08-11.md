# TARO O0R R6 factor-split implementation lock

状态：`FACTOR_COMPOSITOR_IMPLEMENTATION_FROZEN / FORMATION_REPLAY_PASS / UNTOUCHED_EXECUTION_FALSE`

R6 已从 post-hoc 统计拼接推进为 roster-independent、可校验的 factor-level compositor：

- SUPPORT 与 BOUNDARY exact-copy Phase-A 已选 source-only component；
- QUERY_CLEARANCE exact-copy R1 baseline；
- 每个 factor 分别绑定实际 depth SHA-256；
- formation replay 与 untouched confirmation 使用不同 role；
- 24 个 formation parents 不能伪装成 untouched；
- 正式 confirmation 少于 8 个 parent 会 fail closed；
- 重新自封错误 query owner 仍触发 `R6_EXACT_COPY_DRIFT`。

15 个 focused tests 全部通过。随后对已消费的 R5 R3 证据执行 implementation replay：8 parents、211 frames、
1,899 components 与 1,899 composites 全部通过。重放保持 height parent-macro `+0.271048054088 m`、normal
parent-macro `+0.027702103489 rad`、8/8 parents jointly positive，并得到：

- extraction evaluability：`1,522 → 1,566`；
- boundary evaluability：`112 → 129`；
- query knownness：`7 → 7`。

该重放的 `promotion_allowed=false` 且无 confirmation 权限。它只证明冻结算法已被忠实实现，不能把已观察的
R5 cohort 变成 fresh evidence。

绑定真源是同名 JSON lock。formation replay result file SHA-256 为
`21B2506A226BC960FE27393103DF482780E405F0446E35B852BFD618F60DC336`，sealed content SHA-256 为
`2D6D6CA54DF37EAD029E5221E6E9EB0AC79C39AE090AAD7884F3A6A72CDF220F`。

唯一后继是 `TARO_O0R_R6_UNTOUCHED_COHORT_AND_DATA_USE_LOCK`：必须先冻结至少 8 个未参与 R4/R5/R6
形成的新 parents、exact frame counts、source/truth hashes 与明确数据使用权限，之后才能实现和签署 one-shot
untouched executor。当前已授权的 24 个 Training parents 均已消费，不能重复充当独立确认。
