# USTRF route-target L1E materialization recovery R3 结果（2026-07-24）

## 结论

此前阻塞已经按原门槛解决，不需要降低标准：

- Android 失败的根因是 scoped storage 下 shell、测试包和目标 App 的 UID 隔离。新链路改为 `adb push -> /data/local/tmp -> run-as com.linnan.blindassist cp -> targetContext.filesDir`，不再依赖 app external-files。
- 冻结的主机可用物理内存门仍为 `6,442,450,944 bytes`（6 GiB）。执行改为连续 6 次 readiness 采样、输入加载后复查、instrumentation 前复查，并且每个分片由独立主机进程执行；没有放宽内存、数据、指标或权限标准。
- 目标私有目录 canary 已逐一核验首个 CrowdBot 分片的 `1,455/1,455` 张 RGB，状态为 `TARGET_PRIVATE_TRANSPORT_CANARY_PASS`，TFLite、C1–C3 和候选输出均为 0。
- 随后同一分片已完成 Android Canvas + 正式 TFLite CPU 4-thread raw 导出、流式拉取、逐帧哈希核验、host 解码和 compact successor 验证，状态为 `FIRST_CROWDBOT_SHARD_MATERIALIZED`。

这证明“manifest 不可见”和“6 GiB 门导致无法启动”两个执行阻塞均已解除。它不是完整 L1 profile：当前只恢复了 39 个缺失 CrowdBot ledger 中的第 1 个，C1–C3 仍未运行。

## 冻结边界

- 新阶段：`R2-L1E-RECOVERY-B1`
- 新 namespace：`r2-l1e-recovery-b1`
- 父 R2/A1 的 `FAIL_CLOSED_EXECUTION_ABORTED` 收据只读保留，不改写、不续算旧重试预算。
- canary 预算为初始 1 次加 1 次有界重试；真正 materialization 每 ledger 为初始 1 次加 2 次有界重试。
- 本轮 `maximum_crowdbot_shards=1`，只证明首个新分片；没有自动扩展到其余 38 条。
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

## 当前覆盖与剩余工作

旧阶段已验证 `2/41` ledger、`4,594/62,229` 帧。本轮新增 1 条 CrowdBot ledger、1,455 帧，因此跨阶段可复核的 canonical input 进度为：

- `3/41` ledger
- `6,049/62,229` 帧
- 剩余 `38` 条 CrowdBot ledger、`56,180` 帧

下一独立边界应先把同一 R3 transport/materialization 合同扩展为“每进程一个分片”的续跑编排，并在每条 successor 验证后退出进程；只有达到 `41/41`、`62,229/62,229` 和冻结的 15 个 reset 后，才允许另开候选 replay 阶段运行 C1–C3。局部分片不得命名为 trace、profile 或候选成绩。

## 验证

- `:app:assembleDebug`
- `:device-benchmark:assembleDebug`
- `:device-benchmark:assembleAndroidTest`
- SM-S9280 transport canary：`OK (1 test)`
- SM-S9280 one-shard exporter：`OK (1 test)`
- R3 focused contract tests：`6 tests OK`
- Python compile：通过
- compact ledger/successor 重建验证：通过
- 文档索引、project-structure smoke 与 tracked/untracked diff check：通过
- 完整 repository hygiene 仍报告 8 项早于本轮的 R1 配置直引研究 Implementation 告警；R3 两份 hash-bound 配置已登记为精确例外，没有新增该类告警。

本结果只说明执行链恢复和一条 detector input ledger 可复现；不证明 C1–C3 的效果，更不构成辅助行走安全或生产授权。
