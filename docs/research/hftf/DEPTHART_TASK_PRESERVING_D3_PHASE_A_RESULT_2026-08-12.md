# DepthART task-preserving D3 Phase-A result

状态：`FAIL / 21_OF_32_CONTINUITY_QUALIFIED / NO_SELECTION_LOCK / NO_PHASE_B`

冻结的 48 个 ARKitScenes Training identity 已全部处理；精确 96 个
`lowres_wide_intrinsics.zip / lowres_wide.traj` body 共 `41,979,912` bytes，逐项满足
GET `200`，且 Content-Length、ETag、Last-Modified 与冻结 HEAD 一致。所有 ZIP 通过 CRC、
member path 与 stem 唯一性检查，`190,028` 个 `.pincam` payload 和 `31,185` 个 trajectory
row 均完成 parse/finite/schema 校验。没有请求或读取 RGB、depth、confidence、source truth
或模型输出。

冻结 continuity 门为：至少连续 300 帧、相邻 `0 < gap <= 0.5s`、pose bracket
`<=0.25s`、orientation index 属于 `{1,3}`；非 portrait、pose reject、零/负 gap 或超限 gap
都会切断 run。完整 48 身份中只有 21 个通过，少于所需 32 个，短缺 11 个。因此 materializer
按 fail-closed 规则发布空 `selected_phase_a`，`phase_a_selection_locked=false`，没有 TRAIN 或
DEVELOPMENT role，也没有进入 Phase-B。

执行留下 48 份 identity checkpoint 与 SHA sidecar，并保留全 48 份 intrinsics archive 和
trajectory。预冻结 validator 的入口只接受 PASS manifest，合法 FAIL 出现后才暴露该覆盖缺口；
原 validator、协议、manifest 和门限均保持不变，另冻
[终态 validator repair](DEPTHART_TASK_PRESERVING_D3_PHASE_A_TERMINAL_VALIDATOR_REPAIR_2026-08-12.json)
并对全部保留源做只读独立复算。审计结果为
`D3_PHASE_A_OFFLINE_TERMINAL_AUDIT_PASS / VALID_WITH_POST_TERMINAL_VALIDATOR_COVERAGE_REPAIR`，
终态仍是 `D3_PHASE_A_FAIL_FEWER_THAN_32_ELIGIBLE_IDENTITIES`。

[机器结果](DEPTHART_TASK_PRESERVING_D3_PHASE_A_RESULT_2026-08-12.json) 为 `6,996` bytes，
SHA-256 `0B7BC025CAFF3C34FC4FCE5A5BB171B206E635700879B5005280DD8106605F5A`；本地 immutable
manifest SHA-256 为 `C3A8D20B6E94E5F7F5A4DB89EDAD7570D1DA71C311B5C2D2F1188BDF8E4791A4`，
离线 audit SHA-256 为 `36EE786674C820503D75871FC7F2C041AA7BD713E17D86F858A3F5D509E674C4`。

当前 D3 evidence version 没有自动 successor。若未来恢复，必须另立新版本，在任何新媒体或
eligibility 访问前重新冻结 fresh identity-disjoint roster、source scope 和协议；不能扩大当前 48
身份池、降低 continuity 门或把这 21 个身份作为 partial role 继续。R2 保持 sealed；性能、默认
App、production 与 safety 均不授权。
