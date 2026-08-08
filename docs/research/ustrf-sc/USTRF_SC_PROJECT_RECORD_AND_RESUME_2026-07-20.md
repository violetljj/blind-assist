# USTRF-SC 项目工作记录与恢复入口（2026-07-20）

状态：snapshot / conservative GPU baseline、resolution sensitivity、source-range、source-radial-motion 与 crop/tiling canary 已完成；等待下一独立变量
记录范围：本轮 USTRF-SC 理论实现、SANPO/公开数据实验、手机设备观察、REveL detector 尝试及恢复约束。
生产授权：否。正式 BlindAssist App 的默认模型和运行路径未因本轮研究改变。

## 1. 结论先行

本轮已经把原方案从“架构图和算法建议”推进到三类可复现资产：

1. pure Kotlin 的双环数据合同、安全内核、故障注入、风险场、人体胶囊走廊、结构化影子输出与确定性回放；
2. 合成数据和多套公开数据上的时序几何、动态 TTC、RGB-D、轨迹及跨模态对齐量化；
3. SM-S9280 上隔离的 CameraX/ARCore 时间戳、姿态、深度候选、内参和 reference-free 慢环事件观察。

研究汇总 V13 的判断是 `CONDITIONAL_RESEARCH_GO`：15 道门中 14 道提供了离线理论、source-data-screening、bounded-public-rgb baseline、source-range 或 source-radial-motion stratification 证据，唯一未通过的是 `device_metric_geometry_admission`。因此目前可以说“核心算法合同和若干数据量化链已实现并可复现”，不能说“已经达到文档描述的真实助盲效果”。缺口不是再加一个识别模型，而是设备坐标/深度登记、身体外参、地面与危险事件真值，以及目标设备时延、热和连续运行收据。

## 2. 需求与决策演化

本轮形成并应继续遵守的项目决策如下：

- 近期优先理论实现、确定性回放和数据量化，不以立即工程落地为目标。
- 当前采集设备是手持手机，未来可能改为眼镜；手机证据不得迁移为眼镜的 frame、clock、intrinsics、extrinsics 或设备性能证据。
- 项目默认不依赖人工采集，使用来源 Agent、合成/仿真数据、公开许可数据和自动设备脚本推进；公开数据标签、Vicon 或 source-native pose 只能证明对应数据合同，不能自动成为客观助盲事件事实。
- 手机阶段采用 `reference-free shadow`：不把棋盘格或物理标定物设为继续研究的前置条件，但米制几何和空间持久化门保持关闭。
- 快环与慢环必须解耦。慢环语义只能修改目标或解释，不能绕过本地安全监督器发出“向前走”。
- 所有结论必须区分：接口存在、离线测试通过、公开数据筛查通过、设备候选可观察、设备准入、生产授权。
- GPU 可以正常用于研究。对已经完成稳定 bounded 试跑的同类配置，batch、样本数和分辨率可按显存、温度和系统响应余量灵活选择；只有新型或明显更重的负载、长时间连续满载以及已有崩溃记录的路线，才要求先做可停止的小样本试跑并保留守护收据。

## 3. 总体技术思路

### 3.1 双环架构

安全快环目标是 10–30 Hz、离线可运行，处理：

```text
RGB / IMU / optional depth
  -> source-frame timestamp + pose receipt
  -> metric geometry / ground / drop / head bands
  -> ego-compensated motion + TTC
  -> uncertainty-aware local risk field
  -> five body-capsule corridor candidates
  -> fail-closed supervisor
  -> structured shadow action
```

语义慢环为事件驱动或约 0.2–2 Hz，处理 OCR、开放词汇查询、VLM、场景候选和任务目标。慢环输出没有 corridor、heading 或 speed 的直接权限；其结果必须带 source frame、TTL、置信度和坐标系，并受快环安全结果约束。

### 3.2 USTRF-SC 核心

USTRF-SC 表示 *Uncertainty-aware Spatio-Temporal Risk Field and Safe Corridor*。核心不是逐帧回答“看见了什么”，而是对未来 1–3 秒回答：

- 身体局部 0–5 m 内哪些区域占用、可通行、下坠、头部危险或未知；
- 目标相对运动在人体包络上是否产生低 TTC；
- 五条离散人体候选轨迹中是否存在满足硬阈值的走廊；
- 若证据过期、错帧、低置信、失跟或冲突，应继续、修正、减速、停止还是请求扫描。

关键工程不变量：统一单调时间、严格 source-frame binding、latest-only、有界队列、TTL、显式坐标系、人体胶囊而非中心线、无法证明安全时 fail closed。

### 3.3 与现有 BlindAssist/SANPO 的关系

- 正式 App 继续使用 `yolo11n_fp16_320.tflite`；USTRF、SANPO 候选和 Corridor-Causal Student 均未替换默认模型。
- SANPO segmentation 用于研究可通行/边界表征，但单帧 boundary IoU 不等于连续助盲事件能力。
- Corridor-Causal Student 是独立的 benchmark-only 因果序列候选；其性能形态可行，但缺少 96 episode / 48 matched pair 的人工生命周期真值。
- `ROUTE_CONDITIONED_OBJECT_AGNOSTIC_RISK_FIELD_PLAN_2026-07-20.md` 是另一条 proposal-only 路线，强调路线条件化、类别无关和显式不确定性；不得与当前 USTRF 实现或生产路径混称。

## 4. 已实现资产

### 4.1 核心代码

主要代码位于 `core/ustrf/`，包括：

- 时间、采集、位姿、几何、运动和校准 receipt 合同；
- `UstrfMetricDepthGeometryAdapter`、`UstrfGroundVisibilityDropProposer`；
- `UstrfEgoCompensatedMotionPromoter` 与 TTC estimator；
- `UstrfRiskField`、已验证 pose delta warp 和故障后 reset；
- 五米 profile、五候选人体胶囊走廊和安全监督器；
- `UstrfStructuredSafetyOutput` 与确定性 trace digest；
- 慢环事件、语义、场景、任务合同及隐私最小化 digest；
- synthetic geometry / dynamic TTC / corridor replay；
- controlled fault replay 和 offline safety simulation。

Android 隔离观察位于 `apps/benchmarks/ustrf-shadow-benchmark/`，不接入默认 App，包括 CameraX timestamp、ARCore raw pose/depth/intrinsics 和 reference-free event gate。

### 4.2 离线与公开数据脚本

关键入口包括：

- `scripts/generate_ustrf_synthetic_temporal_geometry_benchmark.py`
- `scripts/generate_ustrf_synthetic_dynamic_ttc_benchmark.py`
- `scripts/generate_ustrf_synthetic_corridor_safety_benchmark.py`
- `scripts/audit_revel_dynamic_rgb_labels.py`
- `scripts/audit_revel_dynamic_vicon_trajectories.py`
- `scripts/audit_revel_rgb_vicon_reprojection.py`
- `scripts/report_ustrf_sc_research_benchmark.py`
- `scripts/benchmark_revel_yolo_person_detector.py`（已完成低负载、可分片和独立输出改造）

这些脚本均有相邻定向测试；完整通过状态应以最近一次实际测试输出或 artifact receipt 为准，而不是以文件存在推断。

## 5. 已完成实验与量化结果

### 5.1 分析型几何、运动与安全走廊

| 实验 | 主要结果 | 证据等级 |
| --- | --- | --- |
| temporal metric geometry | 14 个双帧场景；8 个静态目标重投影 RMSE `0 m`；2 个缺口不误报 DROP；真实下坠 `4/4` 检出 | offline-theory-only |
| dynamic TTC | 9 条解析轨迹中 7 条准入、2 条按合同拒绝；6 个 TTC 最大误差 `1 ms`；4/4 collision label 一致 | offline-theory-only |
| body-capsule corridor | 256 场景；应 STOP `59/59`；32 个 clear 场景误 STOP `0`；非故障 corridor selection `240/240`；16 个注入故障均 STOP | offline-theory-only |
| offline safety scenarios | clear、中央占用、全宽下坠、头部障碍、动态交汇、中央未知、过期几何、pose lost 均有确定性回放 | synthetic fixture only |

### 5.2 公开数据 source-native 审计

| 数据 | 主要结果 | 不能证明什么 |
| --- | --- | --- |
| TartanAir JapaneseAlley P000/P002 | 有效重投影约 97%；中位残差 `5.56 / 3.27 mm` | 候选地面未通过同一几何门；无 body/event truth |
| VKITTI2 Scene01 clone | 6,767 个连续轨迹对；source-moving precision `1.000`，recall `0.9972` | 无真实 timestamp/body receipt |
| Argoverse AV1 sample | 1,282 对；中位周期 `0.10017 s`；5 条 3 秒内 TTC 候选 | 无 egocentric RGB-D/body receipt |
| CARLA pedestrian RGB-D slice | 20 帧同步 depth/pose/timestamp 审计通过 | 坐标含 image reflection；无人体/地面绑定 |
| Bonn moving_obstructing_box | 590 RGB；20 ms RGB-depth 同步 `1.0`，RGB-pose 同步 `0.9983`；重投影中位残差 `13.33 mm` | 无用户身体与 assistive event truth |
| REveL Dynamic 2D | 8,580 RGB/label 帧全配对；13,018 boxes 几何有效率 `1.0`；约 `23.073 Hz` | 只有 2D helmet-colour boxes，无米制距离/TTC |
| REveL Dynamic Vicon | green/yellow 连续相对运动对 `22,644 / 22,465`；20 ms 同步率 `94.61% / 97.31%` | Vicon helmet/sensor-suite 不是手机/眼镜身体坐标 |
| REveL RGB-Vicon 对齐 | green/yellow 投影落入同类 RGB box `89.61% / 97.04%` | 只证明数据内部跨模态一致性 |

当前 V13 汇总文件：

- `artifacts.local/evidence/ustrf-sc/research-benchmark-v13-20260720/research_benchmark_report.json`
- `artifacts.local/evidence/ustrf-sc/research-benchmark-v13-20260720/research_benchmark_report.html`

V13 共 15 gate，14 个通过；新增 source Vicon radial-motion/TTC-proxy 分层后，唯一仍未通过的是 `device_metric_geometry_admission`，`production_authority=false`。

### 5.3 SANPO 表征与 boundary 消融

- 确定性 linear probe：raw `activation_1` global mIoU `0.1133`；`lraspp_fuse` global mIoU `0.2403`，但 boundary IoU 仍约 `0.000089`。
- v3 数据中 dev boundary 像素密度约为 train 的 16%，说明近零 boundary 能力包含明显的数据分布问题。
- real-only 覆盖匹配、100-step、三组 seed 的 distance auxiliary 平均 boundary 增益 `+0.00188`，同时 global mIoU 平均 `-0.00328`；是小幅 Pareto trade-off，不是胜出模型。
- 默认结构 300-step 两组 seed 的平均 boundary 增益 `+0.00053`、global mIoU `+0.00315`；边界效应仍不稳定。
- 结论：不再把“单纯增加训练步数或 signed-distance loss”当作解决台阶/路沿能力的充分路线。下一变量应是连续 boundary/step 来源、近场几何监督、route-field interaction 和按最差 session/scene 的事件指标。

详细记录：`USTRF_SC_RESEARCH_METRICS_2026-07-20.md`。

### 5.4 SM-S9280 手机观察

- CameraX 静止 30 帧：3 个 reference-free slow-loop event 与 source frame 绑定，27 个按节流规则抑制。
- ARCore 受控移动观察可出现稳定的 `TRACKING` candidate，但 raw pose 固定标记为 `EPHEMERAL_PER_FRAME`，不能进入跨帧稳定 world。
- moving freshness 审计中 861 个 raw-depth candidate 只有 1 个与 source frame 严格对齐；reference-free handheld r5 中 843 个 depth candidate 全部 stale/reprojected，source-aligned 为 0。
- image intrinsics observation 在短窗内稳定，但它不是独立校准，也不证明 depth registration 或 camera-body full SE(3)。
- 结论：手机 reference-free shadow 可继续研究事件、时序和非米制接口；设备米制几何门保持关闭。未来眼镜必须从新的 frame/clock/calibration/device benchmark 重新开始。

## 6. REveL YOLO11n 有界 detector baseline

目标：用 COCO 预训练 YOLO11n person class 对 REveL 的 green/yellow helmet boxes 合并后做 2D person detection baseline，输出 AP50、固定阈值 precision/recall/F1、按 box area 的 recall 和吞吐。

实验资产：

- 数据：`artifacts.local/evidence/datasets/revel-dynamic-images-labels-v1-20260720/`
- 权重：`artifacts.local/evidence/datasets/revel-dynamic-bag-v1-20260720/models/yolo11n.pt`
- 脚本：`scripts/benchmark_revel_yolo_person_detector.py`
- 测试：`scripts/test_benchmark_revel_yolo_person_detector.py`
- 旧进程收据：`.../qa/revel_yolo11n.process.json`
- 低负载守护器：`scripts/run_guarded_revel_yolo_smoke.ps1`

中断事实：

- 原脚本默认 `batch=64`、`imgsz=320`、`half=True`，一次调用会构造全部 8,580 张图像路径并持续 CUDA 推理。
- 两次全量尝试期间电脑均蓝屏重启；预期结果 `revel_yolo11n_person_benchmark.json` 不存在，因此没有任何 AP、recall 或 FPS 可以引用。
- 两个 dump 均为 bugcheck `0x133 DPC_WATCHDOG_VIOLATION`；本地符号不完全匹配当前 Windows build，但两份堆栈都重复出现长段 `nvlddmkm.sys` 调用。当前只能作“GPU 驱动/显示栈强关联”的组件级判断，不能把责任精确到某个 CUDA kernel。
- 当前 NVIDIA driver 为 610.62。现有驱动包已导出到 `artifacts.local/crash-diagnosis/2026-07-20/driver-backup-610.62/`；596.36 WHQL Studio 安装包已下载但未验证、未安装。按用户指令，驱动修复已暂停。

恢复实施：

- benchmark 默认改为 `batch=1`、`imgsz=256`、FP32、CUDA allocator fraction `0.25`；新增 `--max-frames`、`uniform/head` 确定性选帧、独立输出路径、批次间延迟和峰值显存记录。
- 守护器在独立 Python 进程外逐秒记录温度、利用率、整卡显存和功耗；温度到 72°C、连续三次监控失败或超时会终止精确进程树；运行前后保存相关 System events。
- 8 / 32 / 128 帧阶梯均以 `batch=1`、256px、15% allocator limit、250ms batch delay 通过；一次 128 帧 r1 因 stdout 管道反压在第 41 帧被 180s 守护超时终止，非 CUDA/温度故障。修复异步排空日志并停止传递无效 `half=False` 弃用参数后，128 帧 r2 在 43.51s 完成，最高 48°C、22.05W、0 相关系统事件。
- 512 帧 uniform bounded r1 在 150.78s 完成；142 个 GPU 监控样本，最高 49°C、38% utilization、整卡显存 1,453MB、35.74W，0 个相关系统事件。r2 对相同 512 个 index 精确复跑，所有 aggregate 指标与 r1 完全相同；生成 512 行逐帧 `details.jsonl`，SHA-256 为 `47cfb30d7cf1862dd85628332f3b9526708c1de76deaa1e24691beeb4396f530`。r2 最高 46°C、41% utilization、1,302MB、22.38W，0 个相关系统事件。

512 帧结果（770 个合并 person boxes，score=0.25，IoU=0.5）：

| 指标 | 数值 |
| --- | ---: |
| AP50 over score floor | 0.92747 |
| precision / recall / F1 | 0.83313 / 0.88831 / 0.85984 |
| small recall（37 boxes） | 0.24324 |
| medium recall（354 boxes） | 0.87571 |
| large recall（379 boxes） | 0.96306 |

结果说明：COCO YOLO11n 在该 bounded sample 上对中/大目标表现较好，但 small-box 漏检 28/37，远处/小目标提前预警能力明显不足。它只能作为公开 RGB 2D baseline，不能证明米制距离、TTC、身体走廊、assistive event 或设备安全。

分辨率配对反证：同一 128 帧 index set 上，320px 相对 256px 的 small recall 仍为 `0.4`，没有增益；F1 从 `0.90176` 降到 `0.88614`（差 `-0.01562`）。因此记录为 `do_not_scale_candidate_to_512`，没有继续做 320px/512 帧扩载。

逐帧失败与 Vicon 距离分层：770 个框中 502 个可绑定 source helmet-to-sensor 距离。0–5m 为 420/448，recall `0.9375`，Wilson 95% CI `[0.9112, 0.9564]`；5m 以上为 39/54，recall `0.7222`，CI `[0.5911, 0.8238]`。14 个 Vicon-aligned small boxes 全在 5.3m 之外，small recall 为 3/14。该结果支持“小框问题主要是远距离提前发现问题”，但这里的距离是 REveL source sensor-suite 到 helmet，不是用户身体距离或物理助盲 TTC。

CPU source radial-motion 分层：使用完整原生 Vicon 轨迹严格包围每个 bag image timestamp，按 5–50ms、≤5m/s、≤20ms 同步门求 `v_r=(r1-r0)/dt`；冻结 ±0.10m/s deadband，只对 approaching 计算离线非因果 `TTC-proxy=r_mid/(-v_r)`。motion 可用 488/770；approaching / quasi-static / receding recall 为 `190/204=.93137`、`93/103=.90291`、`164/181=.90608`。TTC-proxy<3s 为 10/10，但其中 0–1s 无样本、1–2s 仅 1 个，不能解释为安全能力。5m 以上三种 motion recall 均明显下降，说明距离混杂仍是主要解释。alignment/details 精确复跑哈希完全相同；证据在 `artifacts.local/evidence/ustrf-sc/revel-yolo11n-vicon-radial-stratification-20260720-r1/`。

主要证据：

- `artifacts.local/evidence/ustrf-sc/revel-yolo11n-guarded-bounded-512-20260720-r2/benchmark.json`
- `artifacts.local/evidence/ustrf-sc/revel-yolo11n-guarded-bounded-512-20260720-r2/details.jsonl`
- `artifacts.local/evidence/ustrf-sc/revel-yolo11n-guarded-bounded-512-20260720-r2/failure_analysis.json`
- `artifacts.local/evidence/ustrf-sc/revel-yolo11n-vicon-radial-stratification-20260720-r1/alignment.json`
- `artifacts.local/evidence/ustrf-sc/revel-yolo11n-vicon-radial-stratification-20260720-r1/details.jsonl`
- `artifacts.local/evidence/ustrf-sc/revel-yolo11n-resolution-sensitivity-128-20260720-r1/comparison.json`
- `artifacts.local/evidence/ustrf-sc/research-benchmark-v13-20260720/research_benchmark_report.json`

硬约束继续有效：不得再次直接运行已两次触发蓝屏的 8,580 帧、`batch=64/imgsz=320/FP16` 旧入口；不得把 bounded desktop baseline 扩大为 target-device 性能或生产效果。普通短时或中等负载任务可在有资源余量时运行；长时间连续高负载、明显放大推理次数或未经验证的新配置仍需守护、分片和可恢复 receipt，不做无检查的整夜满载。

## 7. 后续实验计划

已完成 8/32/128/512 帧恢复链、512 r2 精确复跑、128 帧 256/320 配对、Vicon 距离分层、CPU radial-motion/TTC-proxy 分层，以及首个 8 帧 crop/tiling canary。后续每一步仍可以独立停止：

1. source radial range-rate / approach-recede / TTC-proxy CPU 分层已完成；保持 `source-motion-stratification-only`，不提升为 physical assistive TTC。
2. 首个固定四角 crop/tiling r1 已恢复 4/8 small miss，但 FP 从 4 增至 14，按预注册门在 8 帧停止，32 帧不补跑。若继续 detector，先冻结跨-view 一致性或其他 FP 准入作为新变量，再以可停止的 bounded canary 验证；GPU batch 和规模按资源余量选择，不把 r1 的 `batch=1/8 帧`提升为通用调度限制。
3. 若以后需要全 8,580 帧统计，采用独立分片进程和可合并 receipt；分片规模可以按已验证的稳定负载调整，并在阶段边界检查系统事件。当前不直接复用已崩溃的全量旧入口。
4. detector 只能与 Vicon trajectory 做 source-level 对齐和误差分层；在没有身体坐标与事件真值前，不生成助盲效果结论。

当前暂停点：V13 已生成；CPU radial-motion 分层完成；320px sensitivity 路线因无 small-recall 增益停止，首个固定四角 crop/tiling r1 也因 FP 过多在 8 帧停止。下一 detector 变量必须先改变 FP 准入机制并重新冻结 canary，不能在 r1 上扫描 overlap/NMS/score 回救；更高价值的主线仍是 route-conditioned、object-agnostic risk field 与真实事件真值。

## 8. 下一轮研究优先级

在“理论实现和数据量化优先”的目标下，推荐顺序为：

1. 若继续 detector，先提出能抑制 crop-view FP 的独立机制并冻结新的 paired canary；运行规模以能快速停止和覆盖 small-GT/空控制压力样本为准，batch 按资源余量选择，不把 r1 的 8/32 帧或 `batch=1` 当作通用 GPU 限制；
2. 研究 route-conditioned、object-agnostic risk field，以风险/距离/可见性为目标，不再只优化四类 segmentation mIoU；
3. 将已完成的 source radial-motion/TTC-proxy 分层仅用于挑选公开数据压力样本，不把它提升为 physical TTC 或 assistive event label；
4. 用 Bonn/REveL/CARLA 等公开序列构造“同源但非助盲真值”的预训练/算法筛查，严格保留 evidence authority；
5. 真机投入恢复后，优先 bounded-queue latency、memory、power、thermal p50/p95/p99，而不是继续堆桌面 GPU 吞吐；
6. 只有出现 body-local metric geometry、动态 assistive event truth 和同机连续热/时延证据，才讨论从 `CONDITIONAL_RESEARCH_GO` 进入设备 shadow gate。

## 9. 当前有效文档入口

### 原始方案包

外部评估包根目录：`D:\edge\助盲系统核心算法方案与研发优先级_Codex评估包\`

- `README.md`：完整技术方案、NAV-P0-01..14、指标、Gate 和 R1..R16 参考资料。
- `CODEX_EVALUATION_PROMPT.md`：评估任务和一票否决条件。
- `助盲系统核心算法方案与研发优先级_Codex评估稿.docx` / `.pdf`：固定交付版。
- `dual_loop_architecture.png`：双环架构图。
- `ustrf_sc_pipeline.png`：USTRF-SC 数据流图。

### 仓库当前真源与研究记录

- `docs/SANPO_CURRENT_STATUS.md`：SANPO 当前状态、禁区和下一门。
- `docs/SANPO_TRAINING_PROTOCOL.md`：数据、训练和 blind 隔离规则。
- `docs/SANPO_CANDIDATE_PROMOTION_GATES.md`：候选晋级、INT8 和设备事件门。
- `docs/CORRIDOR_CAUSAL_PROGRESS_2026-07-20.md`：Corridor-Causal Student 的性能可行性与真值阻塞。
- `docs/ROUTE_CONDITIONED_OBJECT_AGNOSTIC_RISK_FIELD_PLAN_2026-07-20.md`：独立 proposal-only 新路线。
- `USTRF_SC_IMPLEMENTATION_STATUS.md`：双环逐模块实现、证据、缺口和授权。
- `USTRF_SC_DEVICE_PHASE_POLICY.md`：手机到眼镜的证据隔离策略。
- `USTRF_SC_OFFLINE_SAFETY_SIMULATION.md`：离线安全场景与验证。
- `USTRF_SC_RESEARCH_METRICS_2026-07-20.md`：SANPO probes/ablations 和 GPU 研究指标。
- `USTRF_SC_SAFETY_KERNEL_EXPERIMENT_2026-07-20.md`：安全内核的完整实验演化记录。
- `USTRF_SC_CALIBRATION_PROTOCOL.md`：以后需要打开设备米制几何门时的协议。
- `USTRF_SC_SANPO_REPLAY_INTEGRATION.md`：SANPO 数据回放边界。

## 10. 参考研究

完整 R1..R16 列表保存在外部方案包 `README.md` 第 22 章。当前最直接的理论来源包括：

- Google Project Guideline：ARCore 位姿、多帧世界状态、深度点云和低延迟反馈。
- Google Running Guide Agent：本地低延迟视觉与事件驱动高层多模态推理解耦。
- EgoBlind：真实盲人第一视角语义问答和安全类能力差距。
- SANPO：可通行、深度和人类导航场景数据。
- AI Guide Dog：手机端轻量方向预测基线。
- Watch Your STEPP：pose-projected semantic traversability。
- Deep Patch Visual Odometry、Depth Anything V2、UniDepthV2：VIO、相对/米制深度候选。
- Clew、ObjectFinder、Snap&Nav：路线记忆、找物和室内拓扑能力的产品参考。

这些论文或系统只支持设计假设与对比基线，不能替代本项目的复现实验和设备门。

## 11. 工作区与接续注意事项

- 实际 Git checkout 是 `E:\linnan\linnan`。
- 当前工作树包含大量其他任务的未提交修改；不得清理、reset 或整树提交。若以后提交本记录，应只按精确文件路径/补丁范围暂存。
- `artifacts.local/` 是本地数据、模型、benchmark、崩溃分析和设备 receipt 的规范位置，不应把大数据或驱动安装包加入 Git。
- 本记录是日期化 snapshot；若后续实验改变了当前 gate，应更新对应 current 文档，而不是只改本页历史。
