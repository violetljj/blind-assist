# TARO O0R ARKitScenes direct Apple hybrid R5 result and R6 factor split canary

状态：`R5_TASK_METRIC_CONFIRMATION_FAIL / R6_PROTOCOL_FROZEN / R6_FACTOR_COMPOSITOR_IMPLEMENTATION_FROZEN / UNTOUCHED_EXECUTION_FALSE`

## R5 正式终态

最终有效终态是：

`TARO_O0R_DIRECT_APPLE_HYBRID_R5_TASK_METRIC_CONFIRMATION_FAIL`

exact cohort 为 8 个 former `ADAPTER_FIT` parents、211 个 physical frames、1,899 个 query slots。候选推理和
source-only Phase A 各执行一次并封存；165 frames 选择 direct Apple SUPPORT，46 frames 选择 R1 baseline。
Phase B 读取全部 211 份绑定 FARO；37 frames 因冻结的 support-unobservable codes 保留 333 个 UNKNOWN slots，
没有伪造 support plane，也没有从 truth 重新选择分支。

R5 的 8 个冻结 gates 中 7 个通过：

- exact cohort/lineage 与 Phase firewall：通过；
- parent metric denominators：8/8，通过；
- height parent-macro：`+0.271048054088 m`，8/8 parents 为正，通过；
- normal parent-macro：`+0.027702103488 rad`，8/8 parents 为正，通过；
- extraction coverage：baseline `1,522`，hybrid `1,566`；恢复 44、丢失 0，通过；
- boundary evaluability：恢复 17、丢失 0；
- query knownness coverage：baseline `7`，hybrid `5`；恢复 5、丢失 7，净少 2，**失败**。

因此 R4A 的 frame-level `DIRECT_WHEN_SOURCE_SUPPORT_AVAILABLE_ELSE_R1_BASELINE_V1` 不得晋级。它证明了
SUPPORT/BOUNDARY 的 parent-disjoint headroom，但把同一 frame-level 分支无条件扩展到 QUERY_CLEARANCE 会产生
knownness regret。

最终 R3 evidence 已逐文件重放 manifest：214 个 manifest-bound files、manifest 前 `706,089` bytes。

- manifest SHA-256：`E449799B257E26DFB7F05D104188649ED31B3066957B873F6A685A3EE3236FA0`
- result SHA-256：`5508EC6851ADD6069914D70A643534245426A3DFA256A301DCA5504AB4B46A39`
- summary SHA-256：`3B6EF8014A3A3FD018ADA4105FEC6B9A0F42DC6C87DED6B7B1AD0F64956CB882`

## 执行修复边界

原 R5 one-shot 在第 4 帧把 `SUPPORT_PLAUSIBLE_INSUFFICIENT` 错误升级为 execution-invalid。修复只把冻结的
7 类 support-unobservable code 映射为 9 个保留 UNKNOWN slots；其他错误继续 fail-closed。随后 R2 暴露
1,485/1,485 direct slots 的 `DIRECT_APPLE_CAMERA_BINDING_DRIFT`：Phase A 对数值相同的 K 哈希 JSON list，
extractor 哈希 normalized float64 array。R3 只验证 consumed legacy hash 并桥接到 extractor representation；
数值 K、候选、Phase-A branch、统计和 gates 均未改变。真实帧 canary 从 direct `0/9` 恢复为 `9/9`。

这些修复只决定执行是否能忠实实现冻结算法；它们不改变上面的 R5 scientific FAIL。

## R6 factor-split post-hoc canary

R5 负结果定位出一个 factor interaction。固定候选策略：

- `SUPPORT = R5 selected source-only branch`；
- `BOUNDARY = R5 selected source-only branch`；
- `QUERY_CLEARANCE = always R1 baseline`。

在只读复用 R5 R3 records 的 post-hoc landscape 中：height/normal、8/8 positive parents、extraction
`1,522 → 1,566` 和 boundary `+17/-0` 全部保留；query knownness 变为 `7 → 7`。按同形 landscape，7/7
candidate gates would pass。

这不是 confirmation PASS：策略是在读取 R5 outcome 后形成。canary 明确写入
`promotion_allowed=false`、`requires_untouched_confirmation=true`，没有 `passed` 或 `terminal` 字段。

- canary result file SHA-256：`96A66A67F77CA13E37F0F8E74E5FC9120FA666AD0810FEB8AB5A97D43693989E`
- sealed content SHA-256：`539F54174B22C7D91D503F8C67C13C7B9F505A3EF1013F3442AC47E3D327AE4C`

## R6 协议与实现锁

R6 factor ownership 已冻结为正式协议，并实现 roster-independent component schema、factor-specific depth
lineage、exact-copy compositor、formation/untouched role firewall、最少 8-parent confirmation gate 和
query→frame→parent reducer。错误 query owner 即使重新自封也会被 validator 拒绝。

实现随后对 consumed R5 R3 evidence 重放 1,899/1,899 records；extraction `1,522 → 1,566`、boundary
`112 → 129`、query knownness `7 → 7`，8/8 parents 仍 jointly positive。该 replay 只证明实现忠实性，
`confirmation_eligible=false`、`promotion_allowed=false`。

- [R6 protocol lock](TARO_O0R_R6_FACTOR_SPLIT_UNTOUCHED_PARENT_CONFIRMATION_PROTOCOL_LOCK_2026-08-11.md)
- [R6 implementation lock](TARO_O0R_R6_FACTOR_SPLIT_IMPLEMENTATION_LOCK_2026-08-11.md)
- implementation replay result file SHA-256：`21B2506A226BC960FE27393103DF482780E405F0446E35B852BFD618F60DC336`
- implementation replay sealed content SHA-256：`2D6D6CA54DF37EAD029E5221E6E9EB0AC79C39AE090AAD7884F3A6A72CDF220F`

## 唯一科学后继

`R5_SELECTED_SUPPORT_BOUNDARY_PLUS_ALWAYS_R1_QUERY_CLEARANCE_V1` 的 protocol 和 factor compositor 已冻结。
现在唯一允许的是先签 `TARO_O0R_R6_UNTOUCHED_COHORT_AND_DATA_USE_LOCK`，再在未参与 R4/R5/R6 形成的
parent-disjoint source/truth 上做一次独立确认。当前授权的 24 个
ARKitScenes Training parents 已分别用于旧 16-parent route 与本次 8-parent R5，不能再伪装成 untouched
confirmation。未获得新的数据 authority 前，不得把 R6 canary 写成 PASS。

Claim ceiling 仍为 WILD_LAB source characterization；不证明 formal O0R、external validation、deployment、
product 或 safety。
