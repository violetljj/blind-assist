# USTRF route-target L1E materialization recovery R3 结果（2026-07-24）

## 结论

此前的运输、资源调度和 Windows 长路径阻塞均已闭合；41 条 canonical input 已完整物化：

- Android 失败的根因是 scoped storage 下 shell、测试包和目标 App 的 UID 隔离。新链路改为 `adb push -> /data/local/tmp -> run-as com.linnan.blindassist cp -> targetContext.filesDir`，不再依赖 app external-files。
- B1 与 A1 保留 `6,442,450,944 bytes`（6 GiB）主机可用物理内存门。用户随后明确授权 continuation A2/A3 将该资源门修订为 4 GiB；连续 readiness 采样、每分片独立主机进程、数据完整性和权限边界均保持不变。
- 目标私有目录 canary 已逐一核验首个 CrowdBot 分片的 `1,455/1,455` 张 RGB，状态为 `TARGET_PRIVATE_TRANSPORT_CANARY_PASS`，TFLite、C1–C3 和候选输出均为 0。
- 同一链路随后完成 Android Canvas + 正式 TFLite CPU 4-thread raw 导出、流式拉取、逐帧哈希核验、host 解码和 compact successor 验证；continuation A1–A3 再按“一进程一分片”完成其余 38 条。
- 最终覆盖为 `41/41` ledger、`62,229/62,229` 帧、`15/15` reset；终态 `CANONICAL_INPUT_41_OF_41_COMPLETE`，C1–C3 仍未运行。

这证明 canonical detector input 已可在同一冻结运输与验证链上完整复建。它仍不是 L1 candidate profile，也不提供候选效果结论：C1–C3、trace、profile、selection 和生产权限均未运行或开放。

## 冻结边界

- 新阶段：`R2-L1E-RECOVERY-B1`
- 新 namespace：`r2-l1e-recovery-b1`
- 父 R2/A1 的 `FAIL_CLOSED_EXECUTION_ABORTED` 收据只读保留，不改写、不续算旧重试预算。
- canary 预算为初始 1 次加 1 次有界重试；真正 materialization 每 ledger 为初始 1 次加 2 次有界重试。
- B1 的 `maximum_crowdbot_shards=1` 只证明首个新分片；其余 38 条由独立冻结的 continuation A1–A3 串行扩展，没有从 B1 自动落入候选阶段。
- selection、ranking、recommendation、provisional selection、L2、L3、Android shadow、H2、人体结果、独立行走安全和生产权限全部为 false。

## 设备与资源证据

设备为 SM-S9280（serial `R5CX10M8Y8X`）。

传输 canary：

- 6 次主机可用内存采样：`8,039,772,160`、`8,024,350,720`、`8,061,296,640`、`7,907,364,864`、`9,921,601,536`、`10,571,825,152 bytes`
- 输入核验后：`9,929,842,688 bytes`
- instrumentation 前：`9,691,701,248 bytes`
- 设备传输前可用：`106,212,225,024 bytes`
- manifest SHA-256：`bb261cfa343c5b1a12a65fe5923928be094cbed4b4d7b38d4d3862eb17fa1d58`
- 核验 RGB：`1,455`

单分片 materialization：

- 6 次主机可用内存采样：`9,027,448,832`、`9,426,812,928`、`9,178,951,680`、`9,064,513,536`、`9,271,836,672`、`9,213,652,992 bytes`
- 输入核验后：`9,460,912,128 bytes`
- instrumentation 前：`9,405,534,208 bytes`
- 设备传输前可用：`105,810,399,232 bytes`
- 完成帧：`1,455/1,455`
- 压缩 raw 大小：`945,805,350 bytes`
- raw SHA-256：`ce206001fddc91666b1ca8fefc47955dd9c8841693491abd02c94f4d80da7f3e`
- compact ledger SHA-256：`f09787f33fbf5f585d104ebd975fbd2644a6694a6e9bfb904847de851c342066`
- successor SHA-256：`ee0523f10148a20d38abdc02d030e2383b0a002f5317b7f34b06340b45ee07a6`
- 设备 thermal status 最大值：`0`
- 电池温度：`27.5°C -> 28.7°C`
- 端到端分片 wall time：约 `236.297s`

raw 只作为产生 successor 前的临时运输证据；验证成功后已删除。设备 `/data/local/tmp` 与目标 App 私有 staging 已清理。

## B1 停止点与 continuation 边界

旧阶段已验证 `2/41` ledger、`4,594/62,229` 帧。B1 新增 1 条 CrowdBot ledger、1,455 帧，因此它的停止点为：

- `3/41` ledger
- `6,049/62,229` 帧
- 当时剩余 `38` 条 CrowdBot ledger、`56,180` 帧

continuation 随后把同一 R3 transport/materialization 合同扩展为“每进程一个分片”，并在每条 successor 验证后退出进程。只有达到 `41/41`、`62,229/62,229` 和冻结的 15 个 reset 后，才允许另开候选 replay 阶段运行 C1–C3；materialization 分片不命名为 trace、profile 或候选成绩。

## Continuation A1–A3 结果

`R2-L1E-RECOVERY-B1-CONTINUATION-A1` 没有改写首分片 B1 配置或实现哈希。父编排器严格串行启动 child；每个 child 通过双 canonical root 查找唯一下一缺失 CrowdBot ledger，只运行一条 B1 materialization，验证恰好新增一个 compact ledger/successor 后退出。独占锁禁止并发 child；每条 ledger 的初始 1 次 + 2 次有界重试保持不变。任何无效或半写 canonical pair、重复权威根、额外 detector-ledger 文件、非 CrowdBot 缺口、覆盖漂移或重试耗尽均 fail closed。

A1 在原 6 GiB 主机可用内存门下成功新增 9 条，使覆盖从 `3/41` 到 `12/41`。随后同一 ledger 的第一次尝试因真实可用内存不足停止，第二、三次在进入 materializer 前因 Windows 长控制回执路径写入失败；A1 因此按合同写出 `FAIL_CLOSED_LEDGER_ATTEMPTS_EXHAUSTED`。这两次路径失败没有伪装成设备推理失败。

用户随后明确授权把主机可用内存门修订为 4 GiB。A2 保留串行与每分片独立进程合同，改用短哈希控制路径；它成功生成并验证下一条 compact ledger/successor，使覆盖达到 `13/41`，但在 successor 完成后写 host materialization receipt 时再次遇到 Windows 长路径失败。A3 保留 4 GiB 门，并仅对长原子写使用 Windows extended path；随后严格串行完成剩余 28 条，28 个 child 成功、0 失败。

最终独立覆盖重算为：

- `41/41` canonical ledger；
- `62,229/62,229` 帧；
- `15/15` discontinuity reset；
- 缺失 ledger 与帧均为 0；
- 相对初始 `3/41` 共补齐 38 条、56,180 帧。

终态回执为 `CANONICAL_INPUT_41_OF_41_COMPLETE`，明确记录 `c1_c2_c3_executed=false`、candidate trace/profile count 均为 0，且所有 selection、ranking、recommendation、L2/L3、Android shadow、H2、human outcome 和 production authority 均为 false。达到输入完整门不会自动启动候选 replay；C1–C3 仍是下一独立阶段。

## 验证

- `:app:assembleDebug`
- `:device-benchmark:assembleDebug`
- `:device-benchmark:assembleAndroidTest`
- SM-S9280 transport canary：`OK (1 test)`
- SM-S9280 one-shard exporter：`OK (1 test)`
- R3 focused contract tests：`6 tests OK`
- continuation A1 contract tests：`4 tests OK`
- continuation A2 contract tests：4 项通过；1 项“仅有 A1 耐久回执”的执行前状态断言在 A2 已实际写出第二份耐久回执后按预期不再成立，冻结测试与配置未事后改写
- continuation A3 terminal contract tests：`4 tests OK`
- continuation A3 串行执行：`28 success / 0 failure`
- 独立覆盖重算：`41/41`、`62,229/62,229`、`15/15 reset`
- Python compile：通过
- compact ledger/successor 重建验证：通过
- 文档索引、project-structure smoke 与 tracked diff check：通过
- untracked diff check：6 个已被执行配置哈希绑定的 continuation Python 文件报告 `new blank line at EOF`；为避免事后破坏冻结实现哈希而原样保留，不影响执行语义
- 完整 repository hygiene 仍报告 8 项早于本轮的 R1 配置直引研究 Implementation 告警；R3 两份 hash-bound 配置已登记为精确例外，没有新增该类告警。

本结果只说明执行链恢复和全部 41 条 detector input ledger 可复现；不证明 C1–C3 的效果，更不构成辅助行走安全或生产授权。
