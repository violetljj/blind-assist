# USTRF route-target R2-L1E 单次探索 profile 结果（2026-07-24）

## 结论

本阶段以唯一合法终态 `FAIL_CLOSED_EXECUTION_ABORTED` 结束。冻结的系统可用物理内存门为 `6 GiB`；初始尝试和两次有界重试分别观测到 `6,051,557,376`、`5,983,371,264`、`6,261,772,288` bytes，三次均低于 `6,442,450,944` bytes。资源守卫在首个 CrowdBot device raw attempt 创建前停止执行，重试额度已耗尽且禁止自动绕过。

候选执行从未开始，权威 trace 为 `0`，探索 profile 为 `0`。本结果不选择、不比较或晋级 C1–C3；不新增数据，不开放 selection、Android shadow、H2、人体结果或生产权限。

## 冻结输入与逐 ledger 覆盖

五个父输入 SHA-256 均重新复算并精确匹配通宵目标：

| 输入 | SHA-256 |
| --- | --- |
| R2-L1 protocol | `4ab0c5dd687f7c9a3b791795271e4ecf5f23c1bbea41d2561503e6ce72e196ac` |
| eligibility mask | `b7dd5cfacc6f14153900bfaf811f3e76a1e188d1064013823f717f615e528157` |
| denominator receipt | `3f356ca69eb50bd176210d01bd9deb69e35acca46764b14603d0bb155d7b82bd` |
| parent validation receipt | `6bae841457ed0bd98cea6653e3c291b8cac5ba80603ec155811fcf95057196b9` |
| frozen C1–C3 `candidates.py` | `82fb1a6391a6cb5fd5dd5116f16f26e250bb2421ee7211f57d691738176816c4` |

Mask membership、source/sequence/frame 顺序和逐帧 capture timestamp 覆盖为 `62,229/62,229`，共 `41/41` 条 masked sequence ledger。causal route 投影也覆盖 `62,229/62,229`。冻结的 `15` 个 `frame_id_not_consecutive` reset 已完整写入配置和终态收据，其中 `14` 个同时超过一秒，另一个 gap 为 `333,329,408ns`；候选未启动，因此没有任何跨 gap 状态或指标观察区间。

Android Canvas canonical detector raw successor 的逐 ledger 结果为：

| 来源 | ledger | 已验证帧 |
| --- | --- | ---: |
| LILocBench | `lilocbench_dynamics_0_front` | 558 |
| LILocBench | `lilocbench_lt_changes_dynamics_0_front` | 4,036 |
| CrowdBot | 39 条冻结 ledger | 0 / 57,635 |

两条既有 LILocBench raw 已解码为独立 compact detector ledger，并分别形成 successor receipt。其余 `39` 条 CrowdBot ledger、`57,635` 帧逐 frame 记录缺失字段 `android_canvas_canonical_detector_raw_successor`，未缩小 `41` ledger / `62,229` frame 宇宙。冻结 T0 association 和 consume timestamp 均为 replay/runtime 字段；资源守卫发生在首个缺失 raw shard 设备尝试前，因此没有生成 association trace、candidate consume timestamp、部分候选 trace 或 profile。

## 执行链与 fail-closed 收据

R2-L1E 新增了独立配置、终态 schema、Android Canvas/TFLite 逐 ledger exporter、host compact successor、receipt-aware runner、独立 validator 与 mutation tests。raw 合同固定为 production-same `ImagePreprocessor`、现有模型/labels、CPU 4 threads、`[1,84,2100]` 输出；没有改模型、阈值、NMS、tracker 或候选实现。每个 shard 的预定顺序是 `device raw -> raw receipt -> host compact detector ledger -> successor receipt -> bounded cleanup`，但本次 CrowdBot 流程在 raw attempt 创建前由内存门终止，因此没有可清理的 device staging。

机器证据位于忽略目录：

- `artifacts.local/evidence/ustrf-route-target-l1-exploratory-profile-r1/terminal-receipt-r1.json`
- `artifacts.local/evidence/ustrf-route-target-l1-exploratory-profile-r1/validation-receipt-r1.json`
- `artifacts.local/evidence/ustrf-route-target-l1-exploratory-profile-r1/resource-guard-attempts-r1.json`
- `artifacts.local/evidence/ustrf-route-target-l1-exploratory-profile-r1/detector-ledgers/`

关键 SHA-256：

| 证据/实现 | SHA-256 |
| --- | --- |
| frozen config | `9cd46c71426ce97420b1565808af58388299e4bf7f2ab58a2f22ffb476136922` |
| terminal schema | `c379bf26992838e343979ad80e98e62f1e4698d1364dd149ea7f4070d590700e` |
| core implementation | `11c971f69ece0c5bea8e093b212072255842646318add065e35e7bfe0a8b608d` |
| runner implementation | `3e0188b48f56f0168488d71b2d5b01826856a92dd984335b12e85ad6fa93a32c` |
| validator implementation | `0eff9a0a92631c0a9af2f21512712f349ea0ccfdce4ea4f02c04e2134016b603` |
| mutation tests | `dd75a0167f1d0f198dae87a978d66a6a5eb9faa3b64e791fa0144df6558e8e38` |
| Android exporter | `926c10155f481e12fb17372bee1148e9c16f98cc4656be07a73b40731a61e2e7` |
| resource-guard attempts | `b0f1940c225fb4eed966a853a65bc4b6600a8d81f356312e6bd6bb7e9e1846a5` |
| terminal receipt | `2be31cca0b64f195e648293a2c0ef4a85e1e57dd5d9730099c58bbaf73df7e6d` |
| validation receipt | `9cd1b5d6bd2b564c1787c25005a7e7fa733af53bea7f53d3d8180fa687ebecff` |

## 验证

- 父 R2-L1 validator：`VALID_METRIC_ELIGIBILITY_R2_L1 checks=18 events=6369`。
- R2-L1E validator：`VALID`；重建 `41` ledger、`62,229` 帧、`15` reset、`4,594` 已验证帧和 `57,635` 缺失帧，并确认终态为 `FAIL_CLOSED_EXECUTION_ABORTED`。
- R2-L1E mutation tests：`16 tests OK`；覆盖禁入 truth/clear/eligibility/scoring 字段、sequence/discontinuity reset、候选 SHA 漂移、0/0、pre-clear、repeat 真值池、evidence-age 缺帧和 L0 gate 等反例。
- Android 构建：`:app:assembleDebug :device-benchmark:assembleDebug`，`BUILD SUCCESSFUL`；两份 APK 已安装到 SM-S9280，但 exporter instrumentation 未进入首个 attempt。
- 正常 runner 在耗尽收据后的再次调用直接以 exit `3` 锁定于 `pre_device_resource_guard`，没有形成第四次设备尝试。独立只读 agent 复算全部绑定并执行 compact ledger + successor 成对替换攻击；局部配对虽能自洽，终态 validator 仍以 `verified compact/successor artifact binding drift` 拒绝，最终无阻断项。
- 文档索引通过。项目结构与仓库卫生门仅报告 8 条本任务开始前已存在的 R1 配置直连 Module Implementation 告警；R2-L1E 配置已改用稳定 root adapter + domain/module-relative 绑定，没有新增结构告警。

## 不能声称的内容

- 没有任何 C1、C2、C3 profile 或指标结果。
- `critical_miss`、`clearance`、`unknown_or_stale_alert` 的 L1 探索资格没有在本阶段兑现为候选观察值。
- `repeat` 与 `evidence_age` 的条件 L1 没有形成实际候选分母。
- 三个 L0 指标仍不具备 pass/fail、门禁或候选比较权限。
- 本阶段不支持安全独立行走、人体效果、Android shadow、H2、生产或模型替换结论。

## 后继边界

本任务已经闭合，不再自动重试。若未来另立任务，必须先重新核验所有冻结 SHA 和现有 successor receipt，并在系统可用物理内存稳定高于 `6 GiB` 后从第一条 CrowdBot ledger 起按原逐 shard 合同恢复；这不是当前收据授予的执行、比较或生产权限。
