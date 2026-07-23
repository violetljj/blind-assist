# USTRF route-target R2-L1X-L2P 结果（2026-07-24）

## 结论

本阶段以唯一合法终态 `FAIL_CLOSED_EXECUTION_ABORTED` 结束。C1–C3 从未运行，权威 trace 与 profile 均为 `0`；因此没有机制成绩矩阵，也不能宣布 `L2_FRESH_SELECTION_PREREG_READY`。

阶段在任何新候选输出可见前完成并冻结了 L2 fresh-selection 执行级预注册和 non-executable L3 confirmation lockbox 模板。两者可供未来独立阶段使用，但不改变本次执行失败终态，也没有选择、排名、推荐或 provisional selection。

## 父绑定与旧结果保留

- 已推送基线：`ed1e9ce7f508bebbebcc9ab9532a4aa0c4bd5a4d`。
- R1 terminal：`2be31cca0b64f195e648293a2c0ef4a85e1e57dd5d9730099c58bbaf73df7e6d`。
- R1 validation：`9cd1b5d6bd2b564c1787c25005a7e7fa733af53bea7f53d3d8180fa687ebecff`。
- R1 config：`9cd46c71426ce97420b1565808af58388299e4bf7f2ab58a2f22ffb476136922`。
- R1 core：`11c971f69ece0c5bea8e093b212072255842646318add065e35e7bfe0a8b608d`。

父 R1 的 `FAIL_CLOSED_EXECUTION_ABORTED`、三次内存 guard attempt、2 条 compact/successor 与逐帧 gap matrix 均保持原样。新阶段和 A1 amendment 使用独立 evidence root；父尝试均明确不计入新预算。

父 validator 重新确认：

- evidence maturity V2：`VALID_EVIDENCE_MATURITY_STANDARD_V2`；
- metric eligibility R2-L1：`VALID_METRIC_ELIGIBILITY_R2_L1 checks=18 events=6369`；
- R1 exploratory：`VALID`，41 ledger、62,229 帧、15 reset，已验证 2 ledger / 4,594 帧。

## 输出不可见时冻结的 L2 与 L3

### L2 fresh-selection

`configs/ustrf_route_target_l2_fresh_selection_prereg_r1.json` 的 SHA-256 为 `1ebd6fc26ee0f4d1fe511f1749a184cfa1ce7eb9fc2b667e7380089e1cc77b29`。它在新 C1–C3 输出之前冻结：

- 唯一数据角色 `fresh_selection`，与 seen/development/exploratory 和 future confirmation 数据隔离；
- C1、C2、C3 固定顺序、每候选一次完整运行、first-valid-complete trace；
- 8 个 required metrics、原性能门、逐来源与 worst-source 门、promotion veto；
- 至少 2 session family、单 family 占比不超过 `.7`；
- 总计 20 recall、5 critical、15 clearance、15 complete repeat、15 complete regeneration、20min negative exposure；相对主张另需 10 matched pairs；
- 每 family 5 recall、1 critical、3 clearance、3 repeat、3 regeneration、5min negative exposure；
- primary metric `event_recall`；依次用 false alerts/min、clearance rate、clearance P95、evidence age、固定 candidate order tie-break；
- 仅当全部 required metric 达到 `evaluable_powered`，point 与可判定 worst-source 门通过且 veto=0，才允许未来输出 `PROVISIONAL_SELECTION_FOR_FRESH_CONFIRMATION_ONLY`；
- 最多 2 个新来源 family、每来源 2 个 canary、默认 2 GiB、连续 2 family 不合格即停止。本阶段未下载或新增数据。

### L3 non-executable template

`configs/ustrf_route_target_l3_confirmation_lockbox_template_r1.json` 的 SHA-256 为 `d804fd675ddb4df33311c2004261c90c91b3992eba9783c3cd3b47d2c16c8d98`。模板保持 `executable=false`、`candidate_id=null`，冻结 6 sessions、至少 2 provenance families、60 complete positive-negative matched pairs、60 complete repeat、60 complete regeneration、至少 5 strata、6-fold LOSO、family share `.6`、critical `>=59`、每个 required metric 至少 2 family、10,000 次 session-within-family bootstrap、seed `20260723`、worst-family 与 LOSO worst-session sentinel。

只有未来独立 L2 pass 后，才能生成新的版本化 executable L3 prereg；本阶段没有运行 L3。

L2/L3 validator 输出 `VALID_L2_L3_PREREG_R1`，38 项 mutation tests 全部通过。

## R2 与 transport amendment 的执行终态

### 原 R2-L1E-R2

新 R2 preflight 验证 41/41 ledger、62,229/62,229 mask membership 与 15/15 discontinuity reset，并在新 namespace 重建 2 条 LILocBench successor。系统内存和磁盘当时满足冻结门。

第一条 CrowdBot ledger 创建新 attempt 时，host 使用了 `r2l1e-r2/<ledger>/attempt-00N`，而继承的清理 helper 只允许旧 `r2l1e/<single-leaf>`。初始尝试与两次重试都在 adb 清理、设备执行和 raw 创建前被安全白名单拒绝。失败 terminal receipt SHA-256 为：

`66396d7a55dcbc3d69e49f942ae13fd014056c1d2d56455157a08259ed811fb1`

### Outcome-unseen A1

因为任何 raw、候选输出或评分结果都未出现，发布 hash-bound `R2-L1E-R2-TRANSPORT-A1`，唯一改变是让清理白名单接受精确的 versioned ledger/attempt 叶路径；模型、labels、`.35/.45`、Canvas preprocess、TFLite、NMS、tracker、C1–C3、指标和 L2/L3 规则不变。原 R2 terminal 与三份失败 manifest 均作为不可变父输入。

A1 的首两次 instrumentation 找到了新 exporter class，但 app 侧 `targetContext.getExternalFilesDir()` 无法把 shell 推送的 manifest 识别为 `isFile`，因此没有 device receipt 或 raw。第三次在 bundle load 后观测系统可用物理内存 `5,512,597,504` bytes，低于不可降低的 `6,442,450,944` bytes 门，安全停止。A1 尝试耗尽，不再建立 A2 或无限重试。

最终 A1 terminal receipt：

- 路径：`artifacts.local/evidence/ustrf-route-target-r2-l1x-l2p-a1/terminal-receipt-r2-l1x-l2p.json`
- SHA-256：`8dd1d88973cdd965282e31a0de030938dbd7597d355cf82bb76a80dda20c3473`
- validator：`VALID_FAIL_CLOSED_EXECUTION_ABORTED`
- validation receipt SHA-256：`c30a420c78ac34123712e8cab306842dace4b591558ae846f92c4fd017ac750c`
- 已验证输入：2/41 ledger、4,594/62,229 帧、15/15 reset
- 缺失输入：39 ledger、57,635 帧
- 候选执行：未开始
- 权威 trace/profile：0/0

## 验证

- R2/L2/L3 Python compile 通过。
- L2/L3 mutation tests：38/38。
- R2 recovery mutation tests：12/12。
- A1 amendment tests：3/3。
- 独立只读复核：`PASS_NO_BLOCKERS`；复制后的 compact/successor、terminal binding、A1 parent 与 A1 scope mutation/replacement attacks 均被拒绝。ignored review receipt SHA-256：`2a25d513e5b99b7da5ada892ec2ef3f8fa99dbbb82a157138a07227c44cf5aba`。
- Android `:app:assembleDebug :device-benchmark:assembleDebug :device-benchmark:assembleAndroidTest --rerun-tasks`：`BUILD SUCCESSFUL`。
- app 与 device-benchmark APK 均成功安装到 SM-S9280；开始时 thermal status `0`、电池约 `28.2°C`。
- 文档索引、structure smoke、scoped secret scan 与 tracked/untracked diff check 通过。仓库卫生门只剩 8 项早于本轮的 R1 配置直引研究 Implementation 路径告警；本轮新增配置已登记为精确的 hash-bound contract 例外，没有新增卫生告警。
- 原 R2 与 A1 的失败 receipts、attempt manifests、instrumentation stdout 和最后安全 checkpoint 均保留在各自 ignored evidence root。

## 不能声称的内容

- 没有 C1、C2、C3 的 L1、条件 L1 或 L0 候选观察结果。
- 没有共同机制失败、候选特有失败、来源族性能差异或 winner/rank/best/总分。
- L2 只是预注册已冻结，不是 fresh-selection 执行或 provisional selection。
- L3 只是 non-executable 模板，不是 confirmation。
- 没有新增 replay 数据、Android shadow、H2、人体效果、安全独立行走、生产授权或默认模型替换结论。
