# RCLE 历史数据工作收束与证据资产化报告 R0

日期：2026-07-28
状态：`CLOSED / HISTORICAL DATA WORK INVENTORIED / NO AUTOMATIC SUCCESSOR`

## 1. 执行摘要

过去数日的公开数据搜索、传输、身份闭合、geometry 角色筛选与 RGB 确认工作至此正式停止。当前正式外部确认终态保持：

`RGB_SEGMENT_CONFIRMATION_R1_NOT_EVALUABLE`

它表示两个冻结片段在各自 one-shot、预算受限的身份提取中均未形成 eligible RGB frame，因此像素解码和冻结 RGB 算法调用均为零。它既不是算法成功，也不是算法失败。

当前算法状态应拆开表述：

- 算法开发证据：`DEVELOPMENT_EVIDENCE_PROMISING`；
- 外部确认：`PENDING / CURRENT CONTRACT CLOSED_NOT_EVALUABLE`；
- 数据协议：存在来源—角色混杂、整源准入单位过宽、传输不可观测和孤儿 claim 等失败；
- 工程交付：历史小型文档、实现和测试可窄范围入库；大型 payload 与本地 evidence 继续留在 `artifacts.local/`。

本任务没有搜索或下载新数据，没有主动启动算法，没有修改阈值、三连续 pair 规则、状态机或 Android。

收束期间检测到并停止了并非由本任务授权的
`RCLE_ECOLOGICAL_RESPONSE_DISCOVERY_R0` 本地 pilot 进程链。最终留下两个已写入目录
（已完成两次）和两个被停止的目录，共 11 个文件、435,199 bytes。本任务没有打开其结果内容，也不承认其为
已启动研究或科学证据；这些目录按
`UNAUTHORIZED_EXECUTION_ARTIFACT / NOT_ADMITTED` 原样归档。

## 2. 本轮收束范围

本轮只读取仓库和本地已有文件，覆盖：

- Phase A synthetic、Bonn、TUM、EVIMO2、ETH3D、CID-SIMS、TartanAir；
- OpenLORIS、MultiScan、DLR、MVSEC、VECtor、BundleFusion；
- ADT、AV2、Waymo、CODa、HoloAssist、M3ED、KITTI/nuScenes、CAVERS 等历史或 metadata-only 候选；
- source authority、source discovery、geometry admission、RGB development/confirmation、传输与失败终态；
- 当前未提交的 RCLE 文档、实现、测试与结果文件；
- `artifacts.local/` 中已有数据、下载和 evidence 的去向与大文件风险。

统一逐项状态见 [RCLE 数据能力与访问状态表](RCLE_DATA_CAPABILITY_MAP_R0_2026-07-28.csv)。
该表包含 38 条数据/claim/工程资产记录和 19 个统一字段。

## 3. 过去数据工作的时间线

1. **2026-07-25**：route-conditioned USTRF 关闭，RCLE-RF 成为研究主线；R1.0 保留长期愿景，Minimal-First R1.1 成为执行原则。
2. **2026-07-26**：Phase A synthetic R0 与一次 coverage R1 完成；旧实现按冻结停止语义关闭。Observable Support Recovery 的 development 与 sealed validation 随后通过，但只构成 synthetic 机制证据。
3. **2026-07-26 至 07-27**：Bonn metadata/transport 逐版闭合，B1A 因独立 replay 的 ledger key-set mismatch 以 `INVALID_EXECUTION_CLOSE_B1` 关闭；TUM geometry audit 与 real-data geometry canary 形成实现/几何证据，不读取 RGB 算法输出。
4. **2026-07-27**：EVIMO2、ETH3D、CID-SIMS、TartanAir 等承担 role admission、development canary 或 source characterization；CID-SIMS RGB development 与后验 geometry 对齐推动了三连续 pair 修订，但不构成 confirmation。
5. **2026-07-27 至 07-28**：OpenLORIS、MultiScan、DLR、MVSEC、VECtor、ETH3D 等进入外部来源发现。R3 共评估 6 个 source family、15 个 capture、161 个固定窗，仍为 0 个同源 role-complete source；协议可用性审计判定继续全市场漫游不再是默认动作。
6. **2026-07-28**：R4 将合理准入单位纠偏为 exact fixed window，选出 OpenLORIS positive 与 DLR below，但保留来源—角色混杂；RGB Segment Confirmation R1 因两个身份提取均未闭合而合法终止。
7. **2026-07-28 后续失败资产**：OpenLORIS R2 传输在首个逻辑请求三次尝试后终止；DLR R2 index claim 进程消失且没有合法 terminal；MVSEC R1/R2 claims 均消费，R2 在 `indoor_flying2:w004` 的冻结 5 ms 配对门失败，`indoor_flying1:w002` 未启动。它们不改写 R1 正式终态。

## 4. 涉及过的数据来源和实验

来源可分为四类：

- **开发与机制资产**：programmatic synthetic、Bonn、TUM `fr2/rpy`、ETH3D `desk_changing_1`、TartanAir `P002`、CID-SIMS `floor3_1/2`；
- **外部来源发现与角色筛选**：OpenLORIS corridor、MultiScan、DLR、MVSEC、VECtor、ETH3D fresh captures、BundleFusion；
- **历史跨项目访问**：OpenLORIS office/cafe、ETH3D `cables_1`、ADT 16 sequences、AV2、Waymo、CODa、HoloAssist；
- **metadata/preflight 候选**：M3ED、KITTI、nuScenes、CAVERS、CoRBS、KITTI-360、ICL-NUIM 等。

这些名称只说明历史涉及范围，不自动授予未来角色。精确 capture/window、访问状态、终态与去向以 capability map 为准。

## 5. 当前正式终态

权威结果仍是 [RCLE RGB Segment Confirmation R1 result](RCLE_RGB_SEGMENT_CONFIRMATION_R1_RESULT_2026-07-28.md)：

`RGB_SEGMENT_CONFIRMATION_R1_NOT_EVALUABLE / VALID_FAIL_CLOSED_TERMINAL`

- OpenLORIS `corridor1-1:w004`：`INVALID_IDENTITY_EXTRACTION_CLOSE_ATTEMPT / URLError`；
- DLR `extreme_geometry/hexagon_01:w001`：`SEGMENT_IDENTITY_NOT_EVALUABLE / DLR_BYTE_BUDGET_EXHAUSTED_OR_RGB_GUARD_ABSENT`；
- eligible RGB frames：`0`；
- pixel decode calls：`0`；
- RGB algorithm calls：`0`；
- alignment denominator：`0`，指标为 `null`，不是零；
- 两个 R1 claim 均已消费，禁止重试、换窗、扩预算或整源回退。

后续 R2/MVSEC 失败仅作为历史 transport/identity 资产：

- OpenLORIS R2：`NOT_EVALUABLE_PARTIAL_QUARANTINED`，首请求三次尝试、成功字节为零；
- DLR R2：claim 已占用，最后进度 `2,172,649,564 / 3,633,353,305` accounted bytes；实际 worker 已消失且无 `TERMINAL.json`/`FAILURE.json`，因此只能记 `ORPHAN/HOLD`，不得重启或补写旧 claim；
- MVSEC R2：`MVSEC_RGB_IDENTITY_NOT_EVALUABLE / IMAGE_GEOMETRY_PAIRING`；`indoor_flying2:w004` 有 317 条 mono8 image metadata，200 个 geometry timestamp 中仅 63 个在冻结 5 ms 内、137 个超门，像素和算法调用仍为零；`indoor_flying1:w002` 未启动。

## 6. 为什么 NOT_EVALUABLE 不是算法失败

算法失败要求冻结算法在合法、身份闭合、时间同步、分母明确的输入上运行后未达到判据。本轮缺少的是算法运行的前置输入资格：

```text
frame identity 未闭合
+ guard / pairing gate 未闭合
+ eligible frame denominator = 0
+ decode calls = 0
+ algorithm calls = 0
```

因此可支持的结论只有“该数据协议未产生可评价输入”。把它改写成算法成功或失败，都会凭空增加未发生的像素观察。

## 7. 数据身份与传输闭合失败

主要失败模式不是同一种：

- OpenLORIS R1/R2：solid archive 的远程 range 传输失败，R1 缺少充分的失败位置诊断，R2 补足了 ledger 但仍在起始阶段终止；
- DLR R1：冻结 1 GiB 预算不足以扫到并闭合 bag 中目标 RGB topic/guard；R2 顺序索引发生孤儿运行，无合法终态；
- MVSEC R1：只保留了宽泛 `ValueError`；R2 将首因定位到 `IMAGE_GEOMETRY_PAIRING`；
- MVSEC R2：raw APS 约 31.86 ms cadence，rectified depth 约 50 ms cadence；nearest delta 正负近似对称且最大约 15.76 ms，固定 offset 不能把全部 200 对救入 5 ms。该现象更符合共享时间域中的异步采样，而不是单一固定时差；但冻结协议仍必须按 5 ms 失败处理；
- Source Discovery R2：科学流程的 geometry ledger 可复算，但重复 downloader 造成累计 acquisition 超出冻结 40 GiB 预算，completion audit 因 operational breach 为 `FAIL`。

这些失败应分别保留为 transport、index、timestamp pairing、protocol usability 和 resource-governance 回归资产。

## 8. 来源—角色混杂

R4 的两个精确片段来自不同来源：

- OpenLORIS `corridor1-1:w004` 承担 positive；
- DLR `extreme_geometry/hexagon_01:w001` 承担 below-reference。

因此 source domain 与 motion role 完全共线。即使未来曾成功取得两段 RGB，也最多只能逐片段描述“RGB 行为是否与各自 geometry 一致”，不能声称算法区分 positive/below，更不能声称泛化、性能、产品或安全有效性。

MVSEC 曾被考虑作为同一来源内的补充对照，但 exact RGB identity/sync 未闭合，不能用于救援该结论。

## 9. 旧来源准入单位为何不合理

旧规则要求一个 source family 或 capture 内同时出现 positive 与 below-reference，导致：

- OpenLORIS 已有 34 个 positive 窗仍被整体判不完整；
- DLR 已有 9 个 below-reference 窗仍被整体判不完整；
- MVSEC 已有 1 个 below-reference 窗仍被整体判不完整。

这把无关窗口的失败传播到了已通过的精确窗口。R4 的纠偏是把准入单位改为 `EXACT_FIXED_WINDOW`，禁止 capture/source-wide failure propagation。但 R4 设计与后续 RGB candidate 只具有历史发现价值，不能追溯改写 R3 或 R1 terminal，也不能消除跨来源角色混杂。

## 10. 仍具有复用价值的工作

应永久保留并可复用：

- source identity、member directory、CRC/SHA、range/phase/stage ledger；
- OpenLORIS、DLR、MVSEC 的失败 terminal 与独立复核；
- ROS bag index、timestamp、topic/type、mono8 layout 与配对诊断代码；
- geometry source/result/pair ledger 与独立 validator；
- frozen algorithm config、三连续 pair 修订及其 development tests；
- CID-SIMS、TUM、Bonn、EVIMO2 的 development/geometry assets；
- synthetic generator、oracle、support-manager、回归矩阵；
- 下载成本、预算超限与 I/O 放大记录；
- 典型反例：低参考假触发、异步 cadence、空网格 ledger mismatch、source-native timestamp 为零。

这些资产可以承担 discovery、development、regression、failure analysis 或 demo 角色；不能自动升级为 unseen confirmation。

## 11. 已查看输出或用于调试的数据

`TUNED_ON` 的核心单元包括：

- Phase A synthetic R0/R1 与 support-manager development；
- Bonn/TUM 已访问 geometry scopes；
- OpenLORIS `corridor1-1:w004`、DLR `hexagon_01:w001`；
- MVSEC `indoor_flying2:w004` 与已选但未执行的 `indoor_flying1:w002` tuple；
- ETH3D `desk_changing_1`、TartanAir `P002`、TUM `fr2/rpy@2/@7`；
- CID-SIMS `floor3_1` development windows；
- 历史 OpenLORIS office/cafe 与 ADT 16 sequences。

`OUTPUT_INSPECTED` 的主要单元包括 OpenLORIS/MultiScan/DLR/MVSEC/VECtor/ETH3D 的 geometry result、EVIMO2 13 sequences、CID-SIMS disjoint/cross-sequence holdout 与已执行 sealed synthetic validation。完整分项见 CSV；不得仅凭“看过 RGB”或“文件在本地”扩大污染范围。

## 12. 仍可能作为 sealed candidate 的资产

当前没有一个 exact RGB segment 能由现有证据稳健标记为 `SEALED_UNSEEN`。

- 历史 synthetic sealed seeds 已运行并查看输出，当前是 `OUTPUT_INSPECTED`；
- `indoor_flying1:w002` 虽未执行 RGB 提取，但已被选入 identity tuple，不能冒充未参与选择的 sealed candidate；
- metadata-only 的未访问 sequence 只能记 `UNKNOWN`，未来若要成为 sealed candidate，必须另立 discovery，在结果前闭合 exact identity、partition、ancestry 和无 outcome influence；本任务不执行该工作。

## 13. 正式停止的工作

从本报告起停止：

- 旧公开数据市场漫游和“再找一个来源”；
- OpenLORIS、DLR、MVSEC 的旧 claim 重试、resume、替换或扩预算；
- 以换窗、滑窗、pooled rescue 或放宽 guard/tolerance 挽救旧终态；
- 继续开发新的通用 dataset adapter；
- 继续读取旧 protected outcome 或运行冻结 RGB 算法；
- host replay、Android、主动告警、产品和安全外推；
- 把 `NOT_EVALUABLE` 改写为算法结论。

## 14. 当前工作树的工程交付状态

提交前盘点显示：

- 既有无关修改：`DEVELOPMENT_LOG.md`、`docs/research/GROUP_MEETING_PROGRESS.md`；
- 既有无关未跟踪文件：`docs/research/GROUP_MEETING_FIRST_REPORT_2026-07-29.md`；
- RCLE 未跟踪小文件：30 个文档、30 个实现/测试文件，均小于 1 MiB；
- 暂存区原为空，分支相对 upstream 为 `0/0`；
- `artifacts.local/`、`*.zip`、`*.npy` 已由 `.gitignore` 隔离。
- 交付审查检测并停止了 `ecological_response_discovery_r0` pilot 进程链；最终核对时
  已无相关进程。两个完成目录和两个停止目录共 11 个文件、435,199 bytes；均仅按
  未授权产物保留，未打开内容、未解释、未纳入结论。

本任务只允许精确 staging：本报告、capability map、RCLE current，以及盘点后确认属于历史 RCLE 数据工作的上述小型文档/实现/测试。无关三项保持原样；任何原始 archive、视频、图像、npz/npy、claim payload 或本地 evidence 均不得进入 Git。

## 15. 本轮经验和治理纠偏

1. `NOT_EVALUABLE` 应保护结论，不应成为无限新 claim 的触发器。
2. admission unit 应与科学观察单位一致；连续 frame、同一长序列 clip 和整源 family 不能默认当独立样本。
3. transport presence、content inspection、algorithm outcome 与 tuning influence 必须分开记录。
4. 先做真实 access-mechanics preflight，再签一次性大 claim；进程存在不等于健康，进度文件也不等于 worker 存活。
5. 超时、range、solid archive 与 DEFLATE 问题是工程诊断；不能把工程失败写成算法结果。
6. 失败资产优先转为 regression、counterexample 和 cost model，不再通过新来源堆叠制造进度感。

## 16. 下一阶段允许做什么

唯一允许被提出、但本任务没有启动的下一实验是：

`RCLE_ECOLOGICAL_RESPONSE_DISCOVERY_R0`

研究问题是：在自然第一视角行走视频中，rotation-compensated local expansion 及三连续 pair 修订，相较 bbox growth 与 uncompensated local expansion，呈现哪些响应规律、优势和失败模式？

第一轮必须是 Discovery，观察正常行走、转头、接近静态障碍、迎面动态目标、横向经过、步态振荡、运动模糊、低纹理、检测框抖动、flow support 和 fit residual 退化；不预设 RCLE 必须获胜。

## 17. 下一阶段明确禁止做什么

本轮以及任何自动后继均不得：

- 自动选择或下载新的正式数据来源；
- 运行旧 confirmation 或新 performance claim；
- 重新调参、改阈值、改三连续 pair 或状态机；
- 把 discovery 结果称为 confirmation/generalization；
- 直接进入 host replay、Android、产品、安全或真人结论。

下一阶段是否启动、使用何种输入与访问边界，需要新的明确任务；本报告本身不构成执行授权。
收束期间出现的未授权 pilot 产物不构成启动，也不得被后续工作直接读取或追认。

## 18. Git 交付信息

建议并采用提交主题：

`docs(rcle): close legacy data search and inventory evidence assets`

提交使用精确路径 staging，不吸收共享工作树中的组会与 development-log 改动。commit hash 由 Git 对最终树生成，因提交无法在自身内容中自引用，权威 hash、纳入文件清单和提交后工作树状态以该提交对象及本任务最终汇报为准。

## 资产处置摘要

### 建议保留

- 30 个 `artifacts.local/evidence/rcle*` 根：当前扫描共 4,815 个文件、约 2.140107 GiB；
- 8 个 RCLE/CID-SIMS dataset 根：14,619 个文件、约 12.642064 GiB；
- 3 个 RCLE download 根：15 个文件、约 2.737721 GiB；
- 所有正式结果、contract、audit、manifest、hash、ledger、失败 receipt、实现和测试。

### 建议归档

- CID-SIMS `floor3_1.zip`（2,211,008,069 bytes）与 `floor3_2.zip`（3,274,014,381 bytes）；
- TUM `rgbd_dataset_freiburg2_rpy.tgz`（2,045,614,831 bytes）；
- EVIMO2 `npz_flea3_7_sanity_ll.tar.gz`（1,706,608,737 bytes）；
- Bonn 六个本地 archive（合计约 2.26 GB）；
- 这些文件继续留在 ignored 本地资产区，仅由 manifest/hash 引用。

### 建议删除但尚未删除

- Source Discovery R2 审计记录的重复 `corridor1-1` 下载和非稀疏 `corridor1-2` partial；
- 已被更完整 ledger/terminal 替代的空临时日志或中断压缩片段；
- 删除前必须重新核对 exact path、identity、引用关系和可恢复性，本任务没有删除任何文件。

### 身份未知，暂不处理

- M3ED/KITTI/nuScenes/CAVERS 等 metadata/preflight 缓存；
- 没有明确 owner 或不能证明被新版完全替代的碎片；
- 默认保留，不因目录名或本地存在推断可删除。
