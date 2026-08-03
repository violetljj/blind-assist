# BlindAssist 算法研究全历程复盘：瓶颈、失败、突破与经验

> 截止时间：2026-08-01  
> 项目：BlindAssist（论文/演示/竞赛原型，不是独立安全产品证据）  
> 范围：从项目开始尝试检测、深度、时序、分割、几何、路线、未来可 traversability 预测等算法，到当前各研究分支的终态。

## 结论先行

截至当前，没有任何新算法获得替换默认 YOLO11n 的证据。正式 App 仍保持 `YOLO11n + 现有风险分析/反馈链路`，没有因为离线指标、oracle、合成数据、单次成功或设备性能通过而切换默认模型。

但这段研究并不是“什么都没做出来”。最重要的成果是把问题从“再换一个模型、再调一组阈值”推进到了可解释的证据边界：

1. 早期瓶颈主要在检测器输出和相对框几何；加入深度、运动、时序后，真正的限制转移到了数据权限、动作性标签、事件生命周期和因果信息。
2. 许多路线在帧级、合成或机制级指标上可以成立，但一到独立来源、父事件、误报成本和真实设备上，就没有证明能改善用户可感知结果。
3. 当前最硬的负结果是：对同一失败事件做信息上限审计时，oracle bbox 能找回事件但不能清除误报，source-native oracle mask 才能同时找回事件并清除误报；这说明当前失败不是简单的 YOLO 阈值或事件规则问题，而是观察表示/行动性信息不足。
4. 语义分割路线完成了从候选工具、失败图谱、门控、采样、结构残差到 PIDNet-S 的完整验证。PIDNet-S 的 QNN/设备性能通过，但三种训练种子在事件质量上全部未通过，因此只能保留 YOLO 基线。
5. 研究方法本身取得了突破：建立了父事件级评估、数据角色账本、协议冻结、独立验证、证据强度分层和 `REJECTED / NOT_EVALUABLE / DEVELOPMENT_ONLY` 终态。现在知道什么时候是算法失败，什么时候只是数据或执行条件不具备。

## 终态的三种含义

本复盘严格区分以下三类，不把它们统称为“算法失败”。

| 终态 | 含义 | 可以说什么 | 不能说什么 |
|---|---|---|---|
| `REJECTED / NOT_SUPPORTED` | 在冻结协议和合格数据上，候选没有达到门槛 | 该候选在限定范围内没有增量，或应停止该候选家族 | 不能扩大成“所有同类算法都不行” |
| `NOT_EVALUABLE / HOLD` | 数据、标签、权限、依赖、协议或执行条件不满足 | 当前实验不能产生科学结论，应修复或暂停 | 不能把不可评估叫算法失败，也不能把局部 canary 当成功 |
| `DEVELOPMENT_ONLY` | 机制、工程、合成或 oracle 证据成立，但尚未跨过独立真实事件/产品门槛 | 可作为研究线索、诊断工具或候选设计 | 不能当作安全、生产、默认 App 或用户效果证据 |

## 一、研究主线是怎样逐步变化的

### 1. 项目起点：YOLO11n 加规则几何

最初的系统是轻量 TFLite 检测器加 `RiskAnalyzer`、稳定器、反馈和会话状态。风险不是物理距离，而是由 bbox 的底边、面积、中心位置、类别和置信度构造 `FAR / MID / NEAR / CRITICAL` 相对等级，再叠加前方/侧方规则。

这条路线最先暴露出两个问题：

- bbox 只告诉我们“检测到某个类别的大致矩形”，不知道障碍是否真正位于行进路径、是否是台阶/路缘/坑洼、是否会在未来几帧造成不可通过；
- 规则可以改变告警密度，却不能凭空生成检测器没有观察到的目标身份、局部几何和事件生命周期。

早期做过中心区域和面积阈值的放宽，也加入过稳定器和反馈，但这些变化主要是局部回放或单元测试层面的改善，缺少独立真实事件证明。因此它们没有获得产品升级权限。

### 2. 检测器替换：YOLO26n 没有形成无回归优势

在同一设备、COCO100 条件下，YOLO11n 与 YOLO26n 的结果大致为：

| 指标 | YOLO11n | YOLO26n |
|---|---:|---:|
| AP50 | .285 | .279 |
| Precision | .859 | .872 |
| Recall | .299 | .294 |
| F1 | .444 | .440 |
| FP/image | .41 | .36 |
| FN/image | 5.84 | 5.88 |
| Risk FN | 15 | 12 |
| Risk FP | 1 | 1 |
| 总耗时 P50/P95 | 54/56 ms | 49/51 ms |

YOLO26n 更快、普通 FP 略少，但 AP/recall 没有变好。后续 BlindAssist EvalSet 还显示其中心风险 recall 下降、critical miss 增加、false-alert rate 翻倍。因此结论不是“YOLO26n 完全不能用”，而是“它没有通过本项目的无回归和风险级门槛”，默认保持 YOLO11n。

需要特别注意，历史文档还保留过另一套阈值调整后的 100 图 benchmark 口径。两套口径的绝对数字不能拼接，但结论一致：YOLO26n 更快并不等于通过风险级无回归门。

**这一阶段的经验：**检测器换代不能只看通用 AP、平均延迟或某一类 FN；必须把 critical miss、事件级 recall、误报和设备尾延迟放在同一个门槛中。

### 3. 深度路线：物理距离直觉没有转化成可用的端侧增量

#### 3.1 Depth Anything V2：算法 smoke pass，端侧落地受阻

Depth Anything V2 Small 的 PyTorch smoke test 可以得到有限、非零的深度，CPU 20 图均值约 88.059 ms、P95 约 111.098 ms；ONNX 图可以导出，但在 `onnx2tf` 的 `Reshape` 环节失败，未形成可用 TFLite 链路。

这里的瓶颈首先不是模型表达能力，而是端侧转换链、算子兼容性和运行预算。它说明“能在 Python 跑通”距离“能在 Android 默认链路中稳定、可审计地运行”还有很长一段。

#### 3.2 MiDaS 融合：误报和延迟代价超过收益

MiDaS TFLite 候选在设备 A/B 中有少量积极信号：critical miss 从 9 降到 7；但距离准确率从 .73 降到 .69，中心风险 recall 不变，alert recall 不变，alert FPR 从 .037 升到 .185，深度 P50/P95 约 222/276–292 ms，总耗时 P50/P95 约 276/292 ms，明显超过预算。

所以它被拒绝不是因为“深度完全没有信息”，而是因为收益不能抵消误报和端侧成本。项目中随后加入了保守的深度/运动融合策略：深度最多上调一级、拒绝 far 直接跳到 near/critical、运动只做有限提升、侧方目标不直接成为高风险；这些实现和单元测试通过，但没有完整的独立时序标签来证明用户效果。

#### 3.3 时序跟踪：代码容易，真值困难

`TemporalRiskTracker` 以约 5 帧/900 ms 的窗口跟踪 bbox 底边、面积、连续性和可选深度。实现层面没有问题，但旧评估集没有足够的序列级 approaching / event / clear 标签，导致 approach 类指标不可用。这个结果把瓶颈从“有没有 temporal code”明确转成“有没有可独立验证的事件生命周期真值”。

### 4. SANPO/语义分割：从像素质量转向可行动性

#### 4.1 oracle 首先揭示：像素更好不等于告警更好

90 帧 oracle 逻辑中，hazard recall 达到 88.9%，但 false alert 仍为 25.9%，并且楼梯/边界被错误升级后反复告警。由此发现，分割路线最核心的问题不是单纯提高 mIoU，而是：

- 当前场景中什么是“应该提醒”的障碍；
- 如何把连续像素变化归并为一个事件；
- 何时确认、何时清除、如何避免同一障碍反复触发；
- 误报和漏报的动作代价如何定义。

这成为后续所有 segmentation、route、future field 研究的共同约束。

SANPO 文档中还存在多个不同协议的 90 帧 oracle/benchmark 结果：早期固定回放曾报告约 3.3% 的 error-alert rate、约 86.7% 的主区域命中和约 65.919 ms 总 P95；另一组行动性/事件审计报告 hazard recall 88.9%、false alert 25.9%。这些结果的 cohort、定义和门槛不同，不能合并成一个“SANPO 已通过”的数字。它们共同支持的只是：oracle 逻辑能够说明事件/边界语义有价值，而 learned model 和生产效果仍未获得授权。

#### 4.2 数据和标签先于模型

SANPO 早期数据存在 raw depth/mask/weather/session 记录不完整、来源与标签 provenance 不清楚、source mask 不等于当前行动性 truth、程序化标签和 teacher pseudo-label 不能直接充当安全真值等问题。

在模型方面，程序/真实混合的 MobileNetV3 + LR-ASPP 最佳 dev mIoU 约 .3175，boundary IoU 约 .00038；real-only v4 的不同模型种子结果从约 .1804 到 .4344 mIoU，最差场景约 .2680，macro-session 约 .3283，未过离线门槛。后续 P1 结构修正能把个别最佳点提高到约 .4642/.5235，但模型种子波动仍约 .2951，OS4/OS32 的 boundary 还会崩溃；采样器波动约 .0112，而模型种子波动约 .2685，约为 24 倍。

因此停止了无边界的 architecture/seed 扫描，转而审计数据角色、边界标签和行动性定义。

#### 4.3 一个关键工程突破：Torch/TF32 等价性问题被定位并修复

Torch/TF 后端最初出现 `max_abs ≈ .0875、argmax agreement ≈ .990448` 的漂移。关闭 TF32 后收敛到约 `max_abs .0000634、argmax 1.0`。这不是算法效果突破，却是研究可信度突破：如果训练、导出、评估后端不等价，后续所有 seed、阈值和模型结论都可能是在比较数值实现差异。

#### 4.4 数据发现的负结果同样重要

围绕中心障碍、台阶和路缘做 source discovery 后发现，source mask 的“obstacle”不等于当前路线应该告警；相机视角、覆盖范围和路线关系有限，许多 step/curb 候选可能只是平行街道设施或不确定区域。也就是说，拥有 segmentation mask 不代表拥有可训练的 actionability label。

### 5. 公共数据、轨迹和 foundation feature：帧级分离不等于事件级因果

在公共 silver 数据上，曾尝试 YOLO12n 固定轨迹表示、scale/bottom/corridor/persistence/slope、DINO/DINOv2、DepthAnything 特征、chromatic marker、radial approach、route relation、temporal route head、future-route teacher 等。

典型结果：

- r3 的固定 YOLO12n 轨迹线性 baseline 单次 BA 约 .8167，但五次 bootstrap 中位数约 .7333；一个 hard case 每次都漏，静态障碍仍产生 false positive；
- r4/r5 的 BA、bootstrap、DINO、relative-depth corridor 和 lifecycle pair 结果反复波动，稳定性门槛没有通过；
- 某些 DINO regional feature 在帧级 AUROC 可到约 .8427/.8116，某些 exact risk-field LOSO linear head 可达 1.0，但二分类 DINO/global/route 只有约 .500/.6249；这更像来源或表示泄漏/不匹配，而不是已经获得可部署的事件预测能力；
- 使用更多历史帧、未来教师或更复杂 route 特征后，opening/alert timing 仍然没有稳定提前，说明加上“时序”这个名词并不会自动获得因果信息。

这条线最后不是简单的“foundation model 失败”，而是证明：如果正负样本、事件起止、路线占用和独立来源没有被定义好，强特征只能产生漂亮的离线分数，不能产生可审计的用户效果。

其中几条常被遗漏的旁线也应记录清楚：DINO/距离特征的 boundary IoU 约 .1312；IMU 路线意图 AUROC 约 .4770/.4746/.3465，未支持路线意图；future-route teacher 在受限 proxy 上 BA 约 .9167，显式路线 synthetic BA 约 .9156，但多 seed 均值约 .8774、最差 no-alert 约 .7971，未过稳定门。Corridor-Causal Student 的 62,689 参数 INT8 TCN 组件 P95 约 .3155 ms、CameraX+YOLO+TCN 总 P95 约 69 ms，证明的是工程承载能力；由于没有 event truth，它没有训练/提醒效果结论。

### 6. USTRF-SC：把“算法问题”与“真实世界证据权限问题”分开

USTRF-SC 研究了检测、bbox-route、scale growth、TTC、跨相机、传感器回放、因果轨迹和 egomotion compensation。REveL YOLO11n canary 在 512 帧/8,580 个观测上 AP50 约 .92747、precision .83313、recall .88831、F1 .85984；但 small target recall 只有约 .24324，且设备 metric geometry admission 没有通过。

bbox-route R1 用 15 个正窗口和 15 个负窗口比较 q90 权重/时间/位置等策略，虽然个别 matched 数值看起来有优势，但相对于 uniform、shuffled 和 bbox-only 没有稳定 superiority，dynamics_0 matched median delta 约 -0.771457，最终关闭路线条件 USTRF。

跨相机/小目标的局部结果也没有改变这个终态：YOLOE 小目标 recall 平均约 +2.20 个百分点，但事件仍约 4/6，FP 增加约 .236/image。RGB-D transport 或 QNN 回放可以复现，只能证明传输/工程路径，不能补上 intended route、event lifecycle 和独立 metric geometry truth。

更根本的阻塞来自真实世界 authority：JRDB、THÖR、REveL 只有部分 canary，时间、姿态、坐标变换、独立 trajectory truth 和 session role 不完整，形成 `EVIDENCE_PROGRAM_BLOCKED_BY_REAL_WORLD_AUTHORITY`。Bonn 等候选里也出现了“变换数值看起来合理，但可用深度帧少于正式要求”的情况。

因此 USTRF 的正确结论是两层：

1. 已有 bbox-route/密集风险场/因果生命周期路线没有证明超过 detector baseline，应关闭对应研究家族；
2. 部分传感器/数据来源不能评估，不应把 authority failure 误报成算法失败。

### 7. RCLE：受控机制可以成立，生态场景效果仍然不成立

RCLE 主要研究旋转、平移、深度、目标接近与 egomotion 补偿。受控 generator 中，motion component localization 和周期性自运动敏感性可以复现：4 个 cluster × 16 sequences × 9,616 pairs 的阶段 A 结果支持旋转泄漏和 translation response direction；四块 DEV diagnostic 观察到 8/8 clusters 的周期性自运动敏感性。

但这些是 controlled-generator internal evidence。自然 session 中，raw 与 compensation 后 trigger density 都约 .4000，Spearman 约从 .3498 变成 .3804，不能形成事件级增量；高角速度子集在 4 个 session 中 3 个变高，更像运动相关偏差而不是稳定纠正。

研究中有两个值得保留的工程突破：

- 修正了 quaternion `wxyz/xyzw` 混用问题；
- 补齐并审计了 `T_cam_imu`，使合成 arms 能正确运行。

同时也出现了典型的不可评估：旋转真值 oracle block 的绝对 P90 约 .0940–.1806/s，远超 `<= .01/s` 门槛；R3 因数值表示/OLS 支持数量和纯绝对误差比较协议有缺陷而只能 audit-only；P1 generator G13 无法达到要求的 inverse-depth 端点速率。RCLE 当前暂停，不能把这些都称为模型失败，也不能从 controlled mechanism 直接跳到 Android 或安全结论。

规模上，RCLE Stage B 还记录了 8 clusters、40 sequences、24,040 pairs 和 865,440 cell fits；这些数字说明执行量很大，但执行量不能抵消 rotation oracle 的 0/8 边界通过和 18 个 coverage failure。

### 8. Dual-loop 几何路线：先做可达性审计，停止“看起来合理”的特征

#### 8.1 Sparse LK 结构性负结果

现有 Sparse LK 输出只有 `success / inlierRatio / validCorridorFraction / corridorResidual / lowerCorridorResidual`，没有目标身份、目标区域、接近方向、径向扩张或 TTC。新鲜语义状态中，19 个状态里没有一个能形成可验证的 action-reachable 分支，首告警 lead 上限为 0 帧。

这不是某个阈值没调好，而是接口本身不包含研究问题所需的变量。因此 F-1B 以 `STRUCTURAL_REACHABILITY` 负结论停止，没有进入 Android 或 confirmation。

#### 8.2 radial flow / area growth / warp residual

在 469 个事件上，bbox area growth 正确 204/469、错误符号 153；ROI sparse radial flow 正确 188/469、错误符号 161，flow 相对 area 的 gain 为 -16。两者都没有进入确认器。

局部背景 homography residual 在 469 个事件中约 233 个正确、91 个错误符号，整体约 49.7%，quasi-static 子集约 25%，不满足独立信息准备度。Depth Anything range/temporal direction 的 Spearman 约 -.75，但方向正确率约 49.0%、错误符号约 29.4%，direct depth derivative 关闭。

#### 8.3 active correction：帧密度下降不是事件效果

在 Shiraz 未见自然来源 rank-2 的 4,891 帧上，candidate 与 baseline 都是 7/7 正事件、7/7 timely，5 个 baseline-false windows 全部保留，corrected 0，feedback rows 从 508 降到 494。表面上 density 下降，但候选只有 frame-level veto，没有 event identity、latch 或状态闭合，因此终态是 `FIRST_UNSEEN_SOURCE_NO_EVENT_LEVEL_EFFECT / DENSITY_SIGNAL_ONLY`。

这次结果很重要：减少反馈行数、减少某类帧告警、降低平均密度，不能代替“是否减少了一个用户事件”。

同一阶段的 production temporal geometry factorial A/B 在 8 个正例、7 个负窗、4,422 帧上两臂完全相同，正例 8/8、负窗 7/7，增量为零。因果框尺度 tri-state 的非 abstain accuracy 可达 1,008/1,017=99.12%，但 coverage 只有约 2.391%，因此只能保留为机制确认，不能驱动默认告警。这两项共同说明：实现了更稳定的趋势，或对少数样本判断得很准，都不等于有足够覆盖的事件级收益。

#### 8.4 其他不可评估项

egomotion attribution 的多个 repair 尝试分别遇到 `rosbags`、`yaml/PyYAML` 缺失和 semantic BBOX 字段双重含义，均为 consumed/invalid，未产生科学结果。目标局部背景 warp residual 的 B 终态为 `NO_DEVELOPMENT_INCREMENT`，没有进入 C1/C2。正确做法是保留失败归因和不执行后续授权，而不是用部分 trace 拼出积极结论。

### 9. 信息上限审计与当前 segmentation 主线

#### 9.1 三臂 audit：当前失败究竟缺了什么

在同一 90 帧、3 个父事件的 consumed regression cohort 上做三臂比较：

| 臂 | 事件找回 | 误报/清除结果 | 能说明什么 |
|---|---|---|---|
| 当前 YOLO bbox/规则 | 0/2 positives | critical miss 至少 1 个 | 当前观测和规则无法覆盖该失败 |
| bbox-derived oracle box | 2/2 | 53 个 FP frames，0/2 清除 | 框信息能定位目标，但不能提供足够的行动性/可清除信息 |
| source-native oracle mask | 2/2 | 无 FP，2/2 clear | 更细的区域/可通行性信息可能是关键，但这是 oracle，不是已学模型证据 |

该结果支持“停止继续堆 YOLO taxonomy、阈值和 event patch 来解决这个失败模式”。它不证明所有 bbox 方法都存在普适 ceiling；新的 learned candidate 必须改变任务、数据角色和事件合同，不能直接在同一 consumed cohort 上继续调优。

#### 9.2 Failure Atlas：把误报机制显式化

Atlas 扩展覆盖 200 帧、5,043 个 components，其中 3,062 个为 same-class false activations。主要机制包括上部视野污染、YOLO attribution ambiguity、时序 flicker、稳定高置信错误和小碎片。由于缺少 instance correspondence、depth、pose，残差类别本身也只有 weakly labelable 性质。

这个突破把“分割误报很多”变成了可审计的失败族谱，也解释了为什么一个简单静态 gate 不能解决所有机制。

#### 9.3 segmentation candidate utility、gating、sampling、structural residual

- 早期 candidate utility 在 120 帧/2 session 上 pixel recall 增量约 .073670、component recall .688129，但 false activation 约 13.833/frame，host P95 24.62 ms，增量总 host P95 138.444 ms，故 `CURRENT_SEGMENTATION_REFERENCE_REJECTED`；
- conditional gating 在 520 帧/10 session 上 FP reduction 约 .092572，低于 .30；minimum session recall 约 .774580，低于 .80，R0/R0.1 均无稳定增量；
- FP-aware DDRNet 的一个 seed 误报面积下降约 .1987，但 overall/min session recall 约 .822/.739，其他 seed 反而增加 FP，绝对 FP area/false component gates 全部不稳，故 `FP_WEIGHTED_SAMPLING_NOT_SUPPORTED`；
- DG-SRF/F0 的 Depth Anything V2 Small structural residual D4 macro AUPRC 约 .309456，低于冻结的 binary DDRNet B 约 .362109，LOSO 仅 4/9，obstacle recall retention 极低，故 `STRUCTURAL_SIGNAL_NOT_SUPPORTED_STOP`，没有继续 temporal/video-depth rescue。

这些负结果共同说明：可分辨的像素、可计算的 residual 和可下降的某个 FP 指标，只有在保持事件 recall、跨 session 稳定性和真实动作性成本时才有意义。

#### 9.4 RISKSEG-R0：端侧通过，事件质量失败

RISKSEG-R0 选择 PIDNet-S 四类任务：walkable、blocking obstacle、boundary level change、unknown nonwalkable。技术预检在 `512x288 / W8A8 / QNN HTP` 上通过，163/163 canary，P95 约 75.739 ms，10 分钟稳定。

最终三种 seed 的 event quality 均未通过：

- YOLO baseline：13/16 positives，6/14 false events；
- learned seeds：正事件分别约 13/16、14/16、13/16，false events 约 13/14、13/14、14/14；
- common hit delay 约 +3/+5/+3 frames；
- 设备训练最终版本共 7,727 samples，推理约 5.198 ms，P95 约 77.374 ms，final/initial P95 约 1.07624x，性能和热稳定性本身通过。

因此终态是 `RISKSEG_R0_TRAINED_NOT_PROMOTABLE_KEEP_YOLO`：性能不是问题，事件质量和误报是问题。

#### 9.5 RISKSEG-R1 P0：软 dense adapter 仍不能补行动性

P0 仅是 validation-only 的 truth-mask soft adapter，不是已授权生产路线。truth-mask 版本约 14/16 positives、12/14 false、4/16 clear，相比 YOLO 的 13/16、6/14、5/16 clear 没有形成可接受的清除增量；learned seeds 约 11/16、12/16、7/16，3/3 guardrails 均未通过。

终态 `TRUTH_MASK_SOFT_ADAPTER_FAIL_CHANGE_ACTIONABILITY_LABELS`。这直接要求下一候选改变 actionability/event supervision 和新的 session-disjoint cohort，而不是继续在同一标签上堆网络。

中心阻塞标签 readiness 也未达到门槛：D0-A1 fresh observation agreement 约 63.16%，successor diagnostic 约 16/24=.6667。故该标签目前只能做 auxiliary，不应驱动 alert。

### 10. HFTF：未来 traversability 是独立候选侧线，不是主线升级

HFTF 假设用 current/history RGB 学习短未来的分层风险场，teacher 可以使用深度、姿态和未来帧，student 只使用当前/历史 RGB。该方向的边界必须写清：所有几何标签都是 proxy，不是人类安全真值。

这条线有一些 `DEVELOPMENT_ONLY` 进展：独立 SANPO proxy authority、split-source teacher mechanics、causal future label mechanics 和部分 signed-clearance mechanics 可以支持候选设计；但 multi-height point-support、source/teacher 传输、foot-ground route 等环节先后出现 proxy 不支持或不可评估，foot-ground source route 已关闭。

F0.1 的 body/head cross-source temporal student 在 351 个 official-test heldout predictions 上，三个 seed delta 约为 -0.007233、+0.015577、-0.025393，中位数 -0.007233；head 中位数约 -0.008473；当前风险 F1 中位数约 .173267，远低于 .6 gate，未显示 temporal gain。该结果只说明 direct RGB → geometry-proxy risk 的跨来源可学性不足，不代表所有时序表示都不可能。

当前 G0-D1 仍是 design/frozen 状态，不应写成已完成结果。HFTF 仍是候选侧线，创新性也尚未评估，`RESEARCH_MAINLINE_UNCHANGED / DEFAULT_APP_UNCHANGED`。

HFTF F0.1 使用 6 个 train、3 个 dev、3 个 official-test heldout sources；351 个 official-test predictions 已消耗，不能再次拿来调参。最新 G0-D1 只冻结了 6 train + 3 model-selection sources 的执行合同，尚无 D1 effect result、Android、生产或安全授权。

## 三、没有形成独立算法结论的候选与旁线

项目早期方案池还出现过 YOLOv10n、YOLO12n、RT-DETR、SAM、Grounding DINO、OWL/VLM offline teacher 等想法。它们有些作为离线教师、特征探针或候选清单出现，有些没有进入冻结实验合同；因此不能把“被讨论过”写成“已经失败”，也不能把未执行的方案写成成功路线。

同样需要单独标记的还有：

- Q0 semantic refresh：固定频率/hold 的 Development reference-preservation，未证明 learned scheduler 或 Android 效果；
- target-local background warp residual：完成设计和实现审查，没有 Development increment，也没有 C1/C2 授权；
- D0 ego-motion attribution：单 capture 无法识别 dominant mechanism，后续 repair 因依赖/字段/协议问题没有形成合法科学退出；
- public silver 中的 proxy/teacher/future-field：可用于假设生成和机制开发，不是运行时 truth。

## 四、所有主要路线的终态总表

| 路线/算法族 | 主要尝试 | 关键瓶颈 | 当前终态 |
|---|---|---|---|
| YOLO11n + bbox rules | bbox 底边/面积/中心/置信度、稳定器、反馈 | 表示有限、行动性缺失、事件闭合不足 | 当前 incumbent；不等于安全证明 |
| YOLO26n | 同设备 detector swap | recall/critical miss/false alert 无无回归 | 不替换默认 |
| MiDaS depth fusion | 端侧深度、保守融合 | FPR 和尾延迟显著增加 | 拒绝 |
| Depth Anything V2 | PyTorch/ONNX/TFLite | 转换算子与端侧预算 | 落地不可评估；后续 structural route 负 |
| TemporalRiskTracker | 5 帧/900 ms bbox/深度状态 | 缺序列真值、事件定义不足 | 工程候选；无效果授权 |
| SANPO segmentation | MobileNet/LR-ASPP、real-only、P1、采样 | label/provenance、seed variance、boundary/actionability | 研究参考，不进默认 |
| public silver/trajectory | YOLO12n trajectory、DINO、depth、route head | 样本少、正负/生命周期/来源不稳 | 诊断证据；无训练/生产授权 |
| USTRF-SC | route field、TTC、crosscam、sensor replay | 独立真实 authority、small target、route superiority | 路线关闭；部分子项不可评估 |
| RCLE | rotation/translation/depth/egomotion | controlled vs natural gap、真值和 transform | 暂停；内部机制仅 Development |
| Sparse LK dual-loop | corridor residual、structural reachability | 输出没有 target/approach/TTC | 结构性停止 |
| area/radial flow/warp/depth | causal geometry side features | signed direction、source readiness、无事件增量 | 不进入 confirmation |
| active correction R1 | unseen-source frame veto | 只有 density effect，无 event effect | `DENSITY_SIGNAL_ONLY` |
| segmentation gates/sampler | conditional gate、FP-aware DDRNet | FP/recall trade-off、seed 不稳 | 停止对应 bounded family |
| DG-SRF | image-space structural depth residual | AUPRC/LOSO/recall retention 不足 | `STRUCTURAL_SIGNAL_NOT_SUPPORTED_STOP` |
| RISKSEG PIDNet-S | 四类 segmentation、QNN/设备 | 事件质量、误报、actionability | 训练完成但不晋级，保留 YOLO |
| HFTF | future layered field、temporal RGB student | proxy/source/teacher authority，cross-source learnability | 独立候选侧线；当前不具备主线权限 |

## 五、瓶颈总分类

### 1. 信息与表示瓶颈

当前 bbox 不能表达局部可通行区域、台阶/边界的行动性、目标身份连续性和可靠的 approach geometry。三臂 oracle 已经说明，更多规则不能补全不存在的信息。新的候选必须在信息源或任务定义上真正改变，而不是只改阈值。

### 2. 标签与行动性瓶颈

现有 mask、类别、深度和 proxy label 经常回答的是“像素/物体是什么”，而不是“现在是否应告警、何时清除、是否会挡住当前路线”。RISKSEG P0 的 truth-mask 失败和中心 obstruction label readiness 不足，是这一瓶颈的直接证据。

### 3. 时序与事件瓶颈

项目早期容易把连续帧平滑、面积增长、反馈次数下降当作事件改善。实际需要的是 parent event identity、开始/确认/升级/清除、timely response、重复告警和 critical miss。当前多条路线在帧级有改善，但在事件级没有增量。

### 4. 数据 authority 与独立性瓶颈

真实来源往往缺时间、姿态、变换、独立轨迹 truth 或完整 session role；公共数据的 silver label 不能自动成为安全真值；开发数据、模型选择数据、官方测试数据和 consumed regression cohort 必须分离。没有 source authority 时，正确终态是 `NOT_EVALUABLE`。

### 5. 工程与端侧瓶颈

模型转换、算子兼容、Torch/TF 数值等价、QNN/HTP、热稳定、P95 而非平均延迟、Android 集成依赖，都会在模型质量之前决定路线能否继续。MiDaS 的 P95 代价、DepthAnything 的转换失败、TF32 修复和 PIDNet 的设备通过/事件失败分别展示了这些边界。

### 6. 评价与治理瓶颈

单次结果、帧级 AUROC、通用 AP、最佳 seed、oracle、合成 generator、平台 benchmark 都不能直接产生生产或安全授权。实验必须先冻结协议、数据角色和门槛；结果出来后不能因为想救路线而改变合同，不能把 consumed case 调成 fresh validation。

## 六、真正的突破和可复用经验

### 1. 研究对象从“模型”变成“证据链”

项目后期不再把问题写成“哪个模型分数最高”，而是追问：

`输入是否含有信息 → 标签是否代表动作 → 事件是否独立 → 评估是否无泄漏 → 设备是否可运行 → 是否改善用户事件 → 是否有权限进入产品`

这让很多路线能在更早阶段停止，减少了无效训练和阈值调参。

### 2. 先做 information ceiling，再做新模型

如果当前模型、bbox oracle、mask oracle 在同一父事件 cohort 上已经形成清楚的上限差异，就应该先决定需要什么新信息，而不是继续搜索模型 taxonomy。oracle 是诊断上界，不是 learned-model evidence。

### 3. 父事件/session 是研究单位，不是帧

后续所有关键结论都应优先报告：event recall、critical miss、false event、timely response、clearance、重复告警、每 session 最差表现和延迟/热成本。帧数只能作为辅助分母，不能替代事件结果。

### 4. 把“性能通过”和“效果通过”拆开

PIDNet-S 的 QNN、P95、热稳定通过，但 event quality 失败；MiDaS 有 critical miss 的局部改善，但 FPR 和 P95 恶化；这说明 runtime admission、model quality、event effect、promotion authority 是四道不同的门。

### 5. 失败要精确归因，不能扩大结论

`DG-SRF` 只否定冻结的 structural-depth setup；`RISKSEG-R0` 只否定当前四类 PIDNet-S/事件合同组合；bbox ceiling audit 不否定所有 bbox 方法；HFTF F0.1 只否定当前 direct RGB→geometry-proxy cross-source learnability。精确的负终态比泛化的“这类算法都不行”更有价值。

### 6. 数据/协议错误也要留下来

缺 `rosbags`、缺 `PyYAML`、字段双重含义、数值协议过严、媒体采样率不匹配、teacher 依赖没有锁定，这些不是尴尬的杂项，而是决定实验是否能产生科学结论的一部分。修复后再实验也必须保持新协议、新数据角色和新 fresh cohort。

### 7. 研究治理本身是生产力

当前采用 thesis-first 的 `THESIS_DEVELOPMENT` 默认路径：开发数据、合成 canary、模型选择和平台 benchmark 可以用于排序和诊断；只有明确进入 `PRODUCTION_PROMOTION` 时，才需要完整的独立来源、盲测、设备、回归、热稳定和发布门槛。这样既能快速探索，也不会把探索结果冒充生产证据。

## 七、曾经最容易走错、后来已经纠正的方向

| 早期直觉 | 后来发现 | 现在的规则 |
|---|---|---|
| 换更大的/更新的 detector 就会解决漏检 | 关键失败可能根本不在 detector taxonomy | 先做同 cohort information ceiling |
| 加深度就能得到真实距离 | 深度可能降低 FPR/时延表现，且转换链先失败 | 先做端侧与事件级 A/B |
| mIoU/boundary IoU 上升就能减少误报 | mask 语义不等于当前路线行动性 | 先冻结 actionability/event label |
| temporal smoothing 会自然减少告警 | 没有事件 identity/clearance 时只会改变密度 | 报告 event-level effect |
| synthetic/oracle 通过就是突破 | 只能证明机制或上限，不证明现实可学 | 标记 `DEVELOPMENT_ONLY` |
| 反馈行数下降就是用户效果 | 可能只是 frame veto，事件完全没变 | 必须看 event recall/clearance |
| 一次最佳 seed 足够 | 模型 seed 波动可能远大于采样波动 | 报告跨 seed、session 和最差分组 |
| 数据不可用可以先用替代数据 | source/label authority 不同会改变 claim | `NOT_EVALUABLE`，不伪造结论 |
| 功能已接入 App 就是算法完成 | 接入、性能、效果、发布授权是不同层 | 分层记录 implementation/validation/promotion |
| 失败路线值得再加一轮 patch | consumed 结果不能调成 fresh validation | 负终态触发停止规则或新协议 |

## 八、当前应如何继续

1. 默认 App 继续使用 YOLO11n；不因为 RISKSEG、HFTF、oracle 或开发期 feature 的局部结果改变默认路径。
2. 对当前信息缺口对应的 YOLO taxonomy、阈值和事件 patch 停止继续投入；若要提出新候选，必须同时提交新的信息源/任务定义、actionability label、父事件合同和独立 session-disjoint cohort。
3. 语义分割不能只沿着“更多类别、更高 mIoU、更复杂 gate”继续。下一候选应先证明标签回答的是行动问题，再证明事件级清除和误报成本，最后才谈模型结构。
4. RISKSEG-R1 P1 当前未授权；P0 的结果只能作为“需要改变 actionability supervision”的证据，不能在原 consumed cohort 上继续调参。
5. Dual-loop 的几何路线保留为研究记录和少量 shadow/diagnostic，不进入 alert、confirmation 或 Android 默认路径。
6. HFTF 保留为独立候选侧线。G0-D1 目前只是 frozen design，必须等结果产生后再判断；不能把已经支持的 teacher mechanics 写成已证明的 student 效果。
7. 新候选的最低 admission checklist 应包含：

   - 明确 claim scope 和失败终态；
   - parent event/session 级 source authority；
   - train/dev/model-selection/official-test 角色隔离；
   - actionability、timely response、clearance、重复告警标签；
   - 与 baseline 同一信息表示或明确新增信息的对照；
   - 跨 seed、跨 session、最差组、critical miss 和 false event；
   - Android/QNN/热稳定/P95 预算；
   - 预先写好的停止条件和 `NOT_EVALUABLE` 条件。

## 九、证据索引

以下文件是本复盘的主要项目内证据源；`DEVELOPMENT_LOG.md` 提供跨月时间线，专项 result/protocol 文件提供可复核数字和冻结合同。

- [项目总览与当前主线](../../README.md)
- [开发时间线](../../DEVELOPMENT_LOG.md)
- [发布变更与已落地能力](../../CHANGELOG.md)
- [Dual-loop 当前状态与总路由](dual-loop/README.md)
- [信息上限三臂审计](dual-loop/INFORMATION_CEILING_THREE_ARM_D0_RESULT_2026-08-01.md)
- [RISKSEG-R0 最终结果](dual-loop/RISKSEG_R0_FINAL_RESULT_2026-08-01.md)
- [RISKSEG-R1 P0 结果](dual-loop/RISKSEG_R1_P0_SOFT_DENSE_ADAPTER_AUDIT_RESULT_2026-08-01.md)
- [Failure Atlas 与 residual labelability](dual-loop/DUAL_LOOP_SEGMENTATION_FAILURE_ATLAS_AND_RESIDUAL_LABELABILITY_R0_RESULT_2026-08-01.md)
- [条件门控结果](dual-loop/DUAL_LOOP_SEGMENTATION_CONDITIONAL_GATING_R0_RESULT_2026-08-01.md)
- [FP-aware DDRNet 结果](dual-loop/DUAL_LOOP_SEGMENTATION_FP_AWARE_DDRNET_R0_RESULT_2026-08-01.md)
- [DG-SRF 结构残差结果](dual-loop/DG_SRF_IMAGE_SPACE_STRUCTURAL_COMPLEMENTARITY_F0_RESULT_2026-08-01.md)
- [Dual-loop R1 未见自然事件](dual-loop/DUAL_LOOP_R1_UNSEEN_NATURAL_EVENT_R0_RANK2_EFFECT_RESULT_2026-07-31.md)
- [Dual-loop 当前数据/研究合同](dual-loop/BLINDASSIST_DUAL_LOOP_PHASE_MINUS1_ADMISSION_CONTRACT_R0_2026-07-30.md)
- [USTRF-SC 当前状态](ustrf-sc/README.md)
- [USTRF 路线关闭结果](ustrf-sc/USTRF_ROUTE_CONDITIONED_PROGRAM_CLOSURE_R1_2026-07-25.md)
- [USTRF 真实世界 authority 终态](ustrf-sc/USTRF_OBSERVABILITY_PROGRAM_REAL_WORLD_AUTHORITY_TERMINAL_R0_RESULT_2026-07-25.md)
- [RCLE 当前状态](rcle/README.md)
- [HFTF 当前状态](hftf/README.md)
- [前沿算法与论文升级复盘](frontier-upgrade-2026-07/BLINDASSIST_FRONTIER_PAPER_UPGRADE_REPORT_2026-07.md)

补充边界：根目录 README 的部分 HFTF 摘要可能滞后于专项 `hftf/README.md` 和最新 `DEVELOPMENT_LOG.md`；涉及 HFTF 当前状态时以后两者为准。全程没有盲人/低视力参与者实验，也没有独立生产安全证据；当前项目仍是研究原型。

## 最后一句话

BlindAssist 目前最大的突破，不是找到一个可以宣布“更安全”的新模型，而是找到了不应再继续盲目试错的边界：在没有新的可行动性信息、独立事件真值和严格证据权限之前，继续换 backbone、加阈值、加平滑或加 oracle，只会增加实验数量，不会增加可用结论。
