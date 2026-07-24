# USTRF route-target L1 candidate replay R2 结果（2026-07-24）

## 结论

本轮 replay-only 独立边界已闭合为 `CANDIDATE_REPLAY_COMPLETE / VALID`：

- 冻结候选 C1、C2、C3 各覆盖 `41/41` masked sequence ledger；
- 每个候选各覆盖 `62,229/62,229` 输入帧和 `15/15` discontinuity reset；
- 总计 `123/123` 条 first-valid-complete 权威 trace、`186,687` candidate-frame、`45` 次断点 reset；
- validator 逐条复核输入 ledger/successor、候选×ledger 笛卡尔积、frame identity/order、trace/receipt/config hash，并重新执行确定性状态机对比，终态 `VALID`。

本轮只证明三种冻结状态机已对完整 canonical input 可恢复地运行并产生可复算 trace。没有生成指标 profile，没有把 truth 或 scoring label 输入候选状态，没有比较候选，也没有 winner、ranking、selection、L2/L3、Android shadow、H2、人体结果、独立行走安全或生产权限。

## 输入与候选边界

输入保持两个唯一只读 authority root：

- LILocBench：旧 exploratory root `2` 条；
- CrowdBot：R3 recovery root `39` 条。

同一 ledger 多根命中、compact/successor 半写或哈希漂移都会在候选启动前失败关闭。父 R3 completion receipt 仍绑定 `41/41` ledger、`62,229/62,229` 帧、`15/15` reset，且候选输出为 0。

候选名册和顺序保持：

1. `C1_CAUSAL_ROUTE_RELATION_FSM`
2. `C2_ROUTE_OCCUPANCY_EPISODE_FSM`
3. `C3_DUAL_KEY_CLEARANCE_FSM`

候选实现、正式 App 模型与 labels、`.35/.45`、T0 association、route margin `.08`、alert/clear frame 参数均未改变。sequence 起点和 15 个冻结断点重置候选及 replay-local history；不按评分窗口重置。

## 执行恢复

初始 R2 在新 namespace 启动前通过 41/62,229/15 preflight，但 Windows 长路径在首个 attempt 目录创建前触发 `WinError 206`，权威 trace 为 0。A1 使用较短 output root 后完成 10 条 trace，随后原子临时文件后缀仍使路径超限；这 10 条完整 trace 保留、没有 profile 权限。

用户明确澄清本次 C1–C3 replay 的主机可用内存门也从 6 GiB 修订为 4 GiB。A2 因此冻结：

- `minimum_available_memory_bytes = 4,294,967,296`；
- 输入、候选逻辑、阈值、T0、reset、重试与权限边界不变；
- trace authority 路径改为 source/sequence identity 的短哈希目录；
- A1 的 10 条完整 trace 以父 receipt/hash 引用继承，不重跑；其余 113 条新运行。

A2 完成后，A3 只做严格 schema finalization，绑定 A2 terminal 与独立 validation receipt，不运行候选。

原 A2 启动器确实在启动时以 4 GiB 门检查并观测到 `9,615,626,240` bytes 可用内存，但这次观测未进入持久回执，A2 内层 runner 也没有逐 ledger 记录主机可用内存。因此不能把 A2 terminal 单独表述为“113 条均有持久逐条 4 GiB 观测”。

为闭合这一证据缺口，A4 作为**独立验证、非新权威 trace**，在每条确定性复演前调用真实主机可用内存采样，低于 `4,294,967,296` bytes 即 fail closed，并将全部 123 次观测写入哈希绑定回执。A4 复演了既有 123 条 trace 的 `186,687` 帧，没有创建或改写任何权威 trace，也没有 profile、比较或 selection 权限。123 次观测最小值为 `7,592,321,024` bytes，最大值为 `9,203,879,936` bytes，全部高于 4 GiB。

## 收据

- A2 terminal SHA-256：`192b6c3b961945bab675a01ba0d500be119a66a444ae195200e35408276c80ac`
- A2 validation SHA-256：`19eaf7776e7c41bdca314b338d08c90398aa060ddfadcf2682e15205d63dc409`
- A3 final terminal SHA-256：`05ea7dc69dd5045b9d632ae5bcf6f6fe10d6a93d30adc064c3528a6fbca97ddc`
- A3 final validation SHA-256：`5992d402a255002c484513fad0e9ef737a99f12fc04cf5626f1d23db7c756fa3`
- A4 4 GiB memory-guard validation SHA-256：`71302ae2455cec2d4edf5ee8c2284f3b380f672318d5a24c0dcfe313e7425bc7`

本地权威输出位于忽略目录 `artifacts.local/r2a2/`；schema-strict finalization 位于 `artifacts.local/r2a3/`；A4 独立内存门验证回执位于 `artifacts.local/r2a4/memory-guard-validation-a4.json`。

## 验证

- replay R2 contract tests：`8 tests OK`
- continuation A2 path/resource tests：`5 tests OK`
- finalization A3 tests：`3 tests OK`
- memory-guard A4 boundary tests：`3 tests OK`（恰好 4 GiB 通过，少 1 byte 失败）
- input preflight：`41/41` ledger、`62,229/62,229` 帧、`15/15` reset
- A2 replay：`123/123` authoritative traces
- A2 independent validator：`VALID`，`186,687` trace frames、`45` resets
- A3 strict schema validator：`VALID`
- A4 independent 4 GiB memory-guard validator：`PASS`，123 次逐条真实采样，最小 `7,592,321,024` bytes；确定性复演 `123` trace / `186,687` 帧；新权威 trace 为 0
- documentation index 与 tracked `git diff --check`：通过
- project structure / repository hygiene：只剩 8 项早于本轮的 R1 config 直引研究 Implementation 告警，本轮未新增同类告警
- untracked diff check：A2 core 仅报告 EOF 新空行；该文件已被执行配置哈希绑定，为保持已执行证据可复核而不作事后改写

## 下一边界

下一独立边界可以只做 metric profile construction：读取本轮冻结 trace，在候选输出之后联结 truth，按既有 L1、条件 L1、L0 权限分别计算；仍不得产生总分、winner、ranking、selection 或 shadow/H2/生产授权。
