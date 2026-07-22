# SANPO 可通行区域 / 通用障碍候选基线

## 结论

首版采用两阶段候选通道，不改生产 APK 的默认 `yolo11n_fp16_320.tflite`：

1. **SANPO mask oracle**：先用 SANPO 真值分割验证通行走廊、路沿、台阶、通用障碍、杆状物等规则与指标。
2. **移动分割模型**：规则通过后，自行训练并导出 INT8 TFLite，以完全相同的输出接口替换 oracle。

这样可以分别回答“风险规则是否有效”和“模型分割是否足够准”，避免模型误差掩盖规则问题。

## 为什么不能直接下载一个 SANPO 分割模型

SANPO 官方论文给出的端侧分割基线是量化 MobileNetV3、512×512、三类输出（safe / not safe / obstacles），论文报告 Pixel 6 可达 30 FPS。但截至本基线建立时，SANPO 官方数据仓库没有列出这套三分类权重。Project Guideline 公开的 `guideline.tflite` 是紫色引导线分割，不是 SANPO 通用语义分割；其 `depth.tflite` 可作为后续几何证据，但不能代替区域语义。

官方资料：

- [SANPO WACV 2025 论文](https://openaccess.thecvf.com/content/WACV2025/papers/Waghmare_SANPO_A_Scene_Understanding_Accessibility_and_Human_Navigation_Dataset_WACV_2025_paper.pdf)
- [SANPO 补充材料（含完整标签与三类映射）](https://openaccess.thecvf.com/content/WACV2025/supplemental/Waghmare_SANPO_A_Scene_WACV_2025_supplemental.pdf)
- [SANPO 官方数据仓库](https://github.com/google-research-datasets/sanpo_dataset)
- [Project Guideline 模型目录](https://github.com/google-research/project-guideline/tree/main/project_guideline/vision/models)

## BlindAssist 映射

官方三类映射把 `stairs` 和 `terrain` 归入 safe-to-walk。BlindAssist 不直接照搬：

- `stairs` 保留为明确的台阶风险；
- `curb`、`inaccessible surface`、`obstacle`、`pole` 等保留为通用风险区域；
- `terrain` 仅作为候选可通行面，后续必须由深度突变、坡度或时序证据否决；
- 大面积 building/tree/vegetation 不直接触发提醒，只在通行走廊内形成近场占据时参与风险。

代码中的 `BlindAssistSanpoTaxonomy`、`TraversabilitySegmentationAnalyzer` 会输出：

- 中心梯形走廊的 safe / not-safe / obstacle 覆盖率；
- 位于走廊内、达到最小面积的通用风险框；
- `DetectionSource.SEGMENTATION` 来源标记，避免把分割区域伪装成 COCO 检测结果。

## 运行 oracle 基线

离线 mask 分析：

```powershell
.\.venv-export312\Scripts\python.exe scripts\benchmark_sanpo_traversability.py `
  --dataset test-artifacts.local\datasets\blindassist-sanpo-pilot-20260711
```

真机风险 A/B（同一 YOLO，区别仅在是否加入 SANPO oracle 区域）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_detector_ab_device_benchmark.ps1 `
  -DatasetKind BlindAssistEvalSet `
  -DatasetRoot test-artifacts.local\datasets\blindassist-sanpo-pilot-20260711 `
  -ComparisonMode SanpoTraversabilityOracle `
  -ImageLimit 30
```

oracle 区域只进入风险分析，不进入 YOLO 的 AP/precision/recall 统计。

当前 30 帧本地 pilot 的离线 oracle 结果（mask 最近邻缩放到 512×512）：中心走廊 30/30 帧存在候选风险，由模型/程序化规则指定并记录 provenance 的主分割区域有 26/30（86.67%）被同口径四连通域规则覆盖；走廊平均像素构成为 safe 71.22%、not-safe 15.26%、obstacle 13.53%。该结果只证明规则能覆盖当前单序列，不代表真实模型召回率，也不能作为上线依据。

## 2026-07-11 真机 A/B

设备：Samsung SM-S9280（同一设备、同一 30 帧、每帧 3 次）。默认模型 90 秒回归通过。

| 指标 | YOLO 几何基线 | SANPO oracle 候选 |
| --- | ---: | ---: |
| 错误提醒率 | 3.3% | 90.0% |
| 距离档准确率 | 3.3% | 10.0% |
| 风险等级准确率 | 3.3% | 10.0% |
| SANPO 主风险命中 | 0% | 10.0% |
| 逼近召回 | 0% | 40.0% |
| 逼近误报率 | 0% | 33.3% |
| 总延迟 P50 | 53.0 ms | 92.404 ms |
| 总延迟 P95 | 53.0 ms | 96.678 ms |

结论：`do_not_replace_default_model`。当前序列 30 帧均标注为不应提醒；候选在第 1–26 帧持续选择右侧 `curb` 并升到 `MEDIUM/NEAR`，造成主要误报。最后 3 帧选择中心 `generic obstacle`，但保持 `LOW` 且未提醒。下一轮应把路沿改为“边界证据”而不是独立障碍框：没有中心走廊侵入、近场深度突变或连续逼近证据时，不得从 LOW 升到提醒级；分割证据的单帧晋级也应限制为最多一级。完整真机报告位于 `test-artifacts.local/detector-ab-device-benchmark/20260711-153543/`。

## Traversability v2 第一阶段结果

2026-07-11 在 Samsung SM-S9280、同一 30 帧、每帧 3 次条件下复测。路沿不再作为普通障碍 Detection；通用障碍使用完整连通域中心重叠、底部位置与中心优先排序；mask 从 512×512 降到 256×256并复用工作缓冲和同帧解码结果。

| 指标 | v1 Oracle | v2 Oracle |
| --- | ---: | ---: |
| 错误提醒率 | 90.0% | 3.3% |
| SANPO 主风险命中 | 10.0% | 86.7% |
| 总延迟 P95 | 96.678 ms | 65.919 ms |

YOLO 指标无退化，benchmark 判定 `traversability_rules_ok_for_model_stage`。该结论只代表当前否定集通过；公开正负连续序列完成前仍不训练、不替换模型。证据目录：`test-artifacts.local/detector-ab-device-benchmark/20260711-163629/`。

## 公开正负连续序列扩展验证

本轮仅接入 SANPO-Real 官方公开会话，原始 RGB/mask 留在 `test-artifacts.local`。三条 10 FPS、各 30 帧序列均经过来源哈希、official split、review CSV 与 finalize 门禁：

- 平行路沿负例：原 SANPO pilot 的已复核连续边界场景；
- 正例：官方 test split、session metadata 标记 `ELEVATION_CHANGE_STAIRS` 的正前方连续台阶；
- 正例：官方 test split 高障碍街景，近距离垃圾桶占据人行通道。

为验证“单帧不提醒、连续证据可晋级”，分割候选只在中心路径满足两帧稳定或连续逼近后由 `LOW/MID` 升至 `MEDIUM/MID`；反馈层仅接收包含 `STABILITY_PROMOTED` 或 `MOTION_PROMOTED` 的中心分割证据。非台阶通用障碍另要求底部位置 `>=65%`，防止远距区域被稳定性误升。

最终 SM-S9280 90 帧 benchmark（证据：`test-artifacts.local/detector-ab-device-benchmark/20260711-191206/`）：候选危险提醒召回 `88.9%`、中心风险召回 `83.3%`、主区域命中 `93.9%`、total P95 `58.405ms`；YOLO AP50/precision/recall 与几何基线一致。但错误提醒率 `25.9%`，高于 `5.3%`，原因是登阶后的 receding 段仍有重复提醒，以及平行路沿负例中部分区域被标成 generic obstacle。因此本扩展集未通过 Oracle v2 晋级，保持 `do_not_replace_default_model`；不得训练或接入 MobileNetV3 + LR-ASPP。

2026-07-15：为收紧已通过台阶的重现漏洞，`RiskEventTracker` 对“已实际反馈、连续三帧远离或缺失后清除”的分割事件保留同标签、同中心锚点的 `1,000 ms` 短暂重现抑制。该窗口内即使上游趋势短暂回弹为 `APPROACHING`，仍返回原事件并阻断第二次反馈；窗口届满后，同锚点候选才可建立新事件。连续基准将事件状态机与 `TemporalRiskTracker` 共同绑定到 `frame_index × 100 ms` 的序列时钟，禁止以墙钟吞吐决定该窗口。此规则不作用于 YOLO、标签变化或中心位置变化的候选，不能替代真实事件级评测。核心 JVM 测试和 `device-benchmark` 编译已通过；尚未重跑真机 90 帧基准，因此仍不得晋级或替换默认模型。

盲道占用是已记录但未伪造的缺口：本轮未找到同时满足“连续、许可明确、可小规模下载、明确盲道被占用”的公开序列。后续可在获得许可明确的小分片后接入 VIP-Mobility360 或其他来源，但必须使用独立 importer 与同样 review/finalize 门禁。

## 下一阶段模型契约

建议训练 `MobileNetV3 + LR-ASPP` 或轻量 DeepLab 解码头，512×512、INT8，输出至少四个风险头：

1. `walkable`
2. `boundary_step_curb`
3. `obstacle`
4. `unknown_nonwalkable`

训练顺序为 SANPO-Synthetic 预训练、SANPO-Real 微调；另建中国场景回归集，覆盖路桩、低矮台阶、盲道占用、电动车和临时施工。晋级门槛看近场危险召回、台阶/路沿召回、错误提醒率、时序抖动和端侧 P95 延迟，不以总体 mIoU 单独决定上线。

## 2026-07-15 事件生命周期复测（SM-S9280）

固定评测集为 `test-artifacts.local/datasets/blindassist-sanpo-v2-event-labeled-20260711`（90 帧、3 条连续序列、每帧 3 次 App 推理）。在 `SanpoTraversabilityOracle` / `current` 风险配置下，设备端报告为 `test-artifacts.local/detector-ab-device-benchmark/20260715-183528/`，模型资产 SHA-256 为 `00edb41a528b0a7e709c4af8ce3e685491492c4539274804e5cfc17a1a867cd2`。

| 指标 | YOLO 几何基线 | SANPO oracle |
| --- | ---: | ---: |
| 事件级提醒召回 | 0.50 | 1.00 |
| 关键事件漏报 | 1 | 0 |
| 实际交付的重复提醒 | 2 | 0 |
| 通过窗口清除率 | 0.50 | 1.00 |
| 通过窗口实际提醒 | 1 | 0 |
| 误提醒 / 分钟 | 6.67 | 0 |
| 平行路沿误提醒 | 0 / 30 | 0 / 30 |
| App 总耗时 P95 | 70.0 ms | 76.0 ms |

此轮证明“实际用户收到的重复提醒”和通过窗口误报已被事件门控消除；但 oracle 仍报告 2 次运行时事件 ID 再生。逐帧审计表明，其中已提醒楼梯在通过后短时间内横向位移，超出活动事件的 0.25 中心偏移阈值而被重新编号。后续修复只对“已通过且已实际提醒”的 1 秒缓存放宽匹配至 0.5；活动事件仍保留 0.25 阈值。该修复及横向位移回归用例已通过 `:core:assist:test`（112 tests）和 `:device-benchmark:compileDebugKotlin`，但设备在同配置复跑前从 ADB 断开，因此尚未以此报告作为最终真机闭环或候选模型晋级依据。

本轮 recommendation 仍为 `do_not_replace_default_model`：oracle 是带真值 mask 的规则诊断，并非可部署的分割模型；即使事件指标改善，也不能替换默认 YOLO 或解锁训练/生产晋级。

### 横向位移修复的同设备复验

修复后的同配置复验已在同一 SM-S9280 完成，产物为 `test-artifacts.local/detector-ab-device-benchmark/20260715-224608/`。结果保持 `eventAlertRecall=1.0`、`criticalEventMissCount=0`、`deliveredRepeatedAlertCount=0`、`postEventClearanceRate=1.0`、`falseAlertCount=0` 和 `falseAlertsPerMinute=0`；45 次重复尝试均被事件门控抑制。`eventRegenerationCount` 仍为 2，但逐帧审计确认它们没有形成第二次实际反馈：其中一条已提醒楼梯在通过后以超过 0.5 中心比例的横向跳变重新出现。

因此当前结论是：用户可感知的重复提醒/通过窗口误报已在这组固定数据上关闭，运行时身份分裂仍作为诊断信号保留。不得为了压低该诊断数而把短时匹配放宽到整个中心走廊，因为这会在短窗口内吞掉同标签但确实不同的中心风险。保持 `do_not_replace_default_model`；下一步由来源 Agent 获取新增连续事件，并以互盲双模型复核/第三模型裁决验证身份分裂是否对应新的风险，而不是由候选 oracle 或公开 mask 自证答案。
