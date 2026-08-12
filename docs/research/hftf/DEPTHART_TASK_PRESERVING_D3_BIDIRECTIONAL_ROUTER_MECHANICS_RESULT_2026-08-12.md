# DepthART task-preserving D3 bidirectional router mechanics result

终态：`D3_ROUTE_ESTABLISHED_SYNTHETIC_MECHANICS_PASS_PRE_DATA_STOP`

D3 已作为独立的新版本建立。它不重跑 D2 的 direct state head，而是只学习两种纠错证书：
`CLEAR release` 纠正 false-block，`OCCUPIED veto` 纠正 false-clear；强冲突转为
`UNKNOWN_GROUND`，弱证据或没有 hard evidence 时保持 baseline。

纯 CPU synthetic canary 与 7 项定向单元测试均 PASS。验证范围包括：

- 中性证书逐状态保持 baseline；
- release 与 veto 分别只改动有相应强证书的 cell；
- 双强证书 fail-closed 为 UNKNOWN；
- 缺少 hard evidence 不能覆盖 baseline；
- UNKNOWN 只有在强证书加 hard evidence 时才能晋升；
- occupied/corridor 证书的 horizon 投影以及 deterministic replay。

该结果没有访问新媒体、source truth、D3 Development 或 R2，也没有训练任何 head。它只建立
协议和 mechanics，不构成 accuracy、candidate、performance、默认 App、production 或 safety
证据。

当前唯一 successor：
`EXPLICIT_D3_FRESH_SOURCE_SCOPE_AND_PARENT_DISJOINT_METADATA_ROSTER_LOCK`。
在该门显式激活前，只能保持 pre-data 状态。
