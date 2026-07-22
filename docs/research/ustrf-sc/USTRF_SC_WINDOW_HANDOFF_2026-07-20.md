# USTRF-SC 新窗口交接（2026-07-20 23:00 +08:00）

## 1. 新窗口先看什么

按以下顺序读取，避免从聊天上下文猜状态：

1. 本文件；
2. `USTRF_SC_PROJECT_RECORD_AND_RESUME_2026-07-20.md`：总体思路、所有关键实验、失败路线和恢复边界；
3. `USTRF_SC_IMPLEMENTATION_STATUS.md`：逐模块实现、证据、缺口、授权；
4. `USTRF_SC_RESEARCH_METRICS_2026-07-20.md`：SANPO、公开数据和 detector 的量化细节；
5. `artifacts.local/evidence/ustrf-sc/research-benchmark-v13-20260720/research_benchmark_report.json`：当前机器可读总收据；
6. `artifacts.local/work/codex-handoffs/USTRF-SC-SAFETY-KERNEL-R1.md`：本地长任务交接单。

仓库根目录必须使用 `E:\linnan\linnan`，不是外层 `E:\linnan`。当前工作树包含大量其他任务改动和未跟踪研究资产；不得清理、回退或宽泛提交。正式 App、默认 YOLO 和生产路径没有因本轮研究改变。

## 2. 当前一句话结论

USTRF-SC 已从方案图推进到可测试的 pure-Kotlin 安全合同、确定性回放、多套公开/合成数据量化和保守 GPU detector 基线；V13 为 `CONDITIONAL_RESEARCH_GO`，15 gate 中 14 个通过，唯一失败为 `device_metric_geometry_admission`。新增 gate 只是 source radial-motion 分层，不是 physical TTC 或设备安全授权；尚未达到文档中的真实助盲效果，也没有生产授权。

## 3. 总体路线

- 快环：RGB/IMU/可选深度 → 严格帧和时间收据 → 米制几何/地面/下坠/头部带 → 自运动补偿与 TTC → 不确定性时空风险场 → 五条人体胶囊候选走廊 → fail-closed supervisor → 结构化影子输出。
- 慢环：事件驱动 OCR/开放词汇/VLM/场景图/目标管理；只能提出语义提示或改变高层目标，不能绕过本地安全监督器发出“向前走”。
- 证据原则：接口存在、离线测试、公开数据筛查、设备候选、设备准入和生产授权必须分开；公开 Vicon/source pose 不自动成为用户身体或助盲事件真值。
- 设备策略：当前手机只做 reference-free shadow；未来眼镜必须重新建立 frame、clock、intrinsics、extrinsics 和设备性能证据，不能继承手机证据。
- 当前生产：正式 App 仍使用 `yolo11n_fp16_320.tflite`；USTRF、SANPO 候选和 Corridor-Causal Student 均为 production-isolated experiment。

## 4. 已完成的核心实现与实验

### 安全内核与回放

- pure Kotlin 的 frame/time/TTL/health receipt、metric geometry/motion 原子装配、TTC、uncertainty、risk field、pose warp、人体胶囊走廊、fail-closed supervisor、结构化 shadow output、trace digest 和故障回放。
- 解析时序几何：14 个双帧场景，8 个静态重投影 RMSE=0，4/4 drop 检出；Kotlin replay 14/14 admitted。
- 解析人体胶囊走廊：256 场景，预期/实际 STOP 59/59，32 个 clear 零误 STOP，240/240 非故障走廊选择，16 个注入故障全部 STOP。
- 解析动态 TTC：9 条轨迹中 7 条准入、2 条拒绝；6/6 TTC 最大误差 1ms；4/4 碰撞标签一致。

### 设备观察

- SM-S9280 的隔离 CameraX 时间戳和 reference-free 慢环事件收据已取得。
- ARCore 移动窗口能观察 TRACKING 和 raw depth candidate，但 pose 固定为 `EPHEMERAL_PER_FRAME`；raw depth 大量 stale/reprojected，不能进入 risk field。
- 无独立 depth registration、完整 camera-body SE(3)、body-local ground/event truth，设备米制几何门保持关闭。

### 公开数据量化

- TartanAir P000/P002：时序重投影有效率约 97%，中位残差 5.56/3.27mm；地面候选被同一门槛拒绝。
- VKITTI2：6,767 连续轨迹对，source-moving precision 1.000、recall 0.9972；无真实 timestamp/body receipt。
- Argoverse AV1：1,282 timestamped pairs，5 条 3 秒内 source-native TTC 候选；无 egocentric RGB-D/body receipt。
- Bonn Dynamic：590 RGB，20ms RGB-depth 同步 1.0、RGB-pose 0.9983，重投影中位残差 13.33mm；无 body/event truth。
- REveL：8,580 RGB/label 全配对，13,018 2D boxes；Vicon green/yellow 相对运动对 22,644/22,465；RGB-Vicon 同类投影命中 89.61%/97.04%。

### REveL detector 基线

- 两次旧全量 `batch=64/imgsz=320/FP16` 导致 `0x133 DPC_WATCHDOG_VIOLATION` 蓝屏；dump 与 `nvlddmkm.sys` 强关联但不能定位到具体 kernel。旧运行无结果 JSON，不得引用指标。
- 已改成默认 `batch=1/imgsz=256/FP32`、确定性分片、allocator cap、批间延迟和外部 GPU 守护器。
- 8/32/128/512 阶梯通过；512 r2 对同一 index 精确复跑，aggregate 指标与 r1 完全相同；逐帧 receipt 为 512 行，SHA-256 `47cfb30d7cf1862dd85628332f3b9526708c1de76deaa1e24691beeb4396f530`。
- 512/8,580 uniform sample、770 boxes：AP50 0.92747；precision/recall/F1 0.83313/0.88831/0.85984；small/medium/large recall 0.24324/0.87571/0.96306。
- r2 守护最高 46°C、41% GPU、1,302MB、22.38W，0 相关系统事件。
- 128 帧 256/320 配对：320 的 small recall 无增益，F1 下降 0.01562；已决定不扩成 320px/512。
- Vicon source-range：502/770 boxes 可对齐；0–5m recall 420/448=0.9375，Wilson 95% CI [0.9112,0.9564]；5m 以上 39/54=0.7222，CI [0.5911,0.8238]。这不是用户身体距离或 physical assistive TTC。
- CPU source radial-motion：严格原生 Vicon 包围、5–50ms、≤5m/s、≤20ms 同步和 ±0.10m/s deadband 后，488/770 boxes 可用；approaching/quasi-static/receding recall 为 0.93137/0.90291/0.90608。TTC-proxy<3s 仅 10 个且 10/10，0–1s 无样本；这是小分母、离线非因果 marker-range proxy。两次复跑的 alignment SHA-256 均为 `4f2750d38869aecf3576f5635a3c1db36af186e074dea689c6113485de0cc012`，逐框 details 均为 `155863e2725ccac5a237b98153fd275fb4f64faf764fe4ab6f828e219059d3ef`。
- 8/32 帧 crop/tiling r1 已执行到预注册停止点：8 帧 failure-enriched canary 中，full-frame 为 `TP/FP/FN=6/4/8`、small `0/8`、F1 `.5000`；全帧 + 四个 60% corner crop 为 `10/14/4`、small `4/8`、F1 `.5263`。所有基线已匹配 GT 均保留，但 FP `4→14` 超过冻结上限 6；守护最高 `47/50°C`、0 System event，决定 `stop_after_8_frame_canary`，32 帧未运行。详见 [配对实验记录](USTRF_SC_REVEL_CROP_TILING_PAIRED_2026-07-20.md)。

## 5. 已停止或否定的路线

- 不再直接运行 8,580 帧、batch=64、FP16 的旧 detector 入口。
- 上述 full-frame 320px sensitivity 候选没有改善 small recall，因此不做该候选的 512-frame 扩载；首个固定四角 crop/tiling 候选虽恢复 4/8 small miss，但 FP 4→14，亦在 8 帧 canary 停止。两条失败均只约束各自冻结配置，不授权事后调参回救。
- SANPO 单纯延长训练、提高语义分辨率或 signed-distance loss 没有稳定解决 boundary/step，下一模型变量必须改变监督或结构。
- ARCore raw pose/depth candidate 不等于稳定 world、登记深度或设备几何，不可偷偷接入 risk field。
- 公开数据平均 AP、source Vicon、解析 TTC 或走廊回放都不能单独证明真实助盲效果。

## 6. 下一窗口推荐顺序

1. 先只读核验 V13、总记录和工作树；随后可按本节安全边界运行 GPU，不要求为了交接而一律停用。
2. source radial range-rate / approach-recede / TTC-proxy CPU 分层已完成；只把它用于公开源压力样本选择，不提升权限。
3. Crop/tiling r1 已因 crop-view FP 过多在 8 帧停止，32 帧不得补跑。若继续 detector，下一独立变量必须先冻结跨 view 一致性或其他 FP 准入策略，并从新的 bounded canary 开始；不得在 r1 上扫描 overlap/NMS/score。
4. 更高价值的主线仍是 route-conditioned、object-agnostic risk field 与事件级指标，而不是只追 global mIoU/AP。
5. device metric geometry gate 只接受自动设备脚本产生并哈希绑定的测量收据；无法自动取得时保持关闭并转向其他自主研究分支。未来眼镜另建整套证据链。

### 6.1 主线接管落地（2026-07-20）

- 研究主线已切换为 [route-conditioned、object-agnostic risk field](../../ROUTE_CONDITIONED_OBJECT_AGNOSTIC_RISK_FIELD_PLAN_2026-07-20.md)。新增 `UstrfRouteConditionedRiskInteractor` 直接计算当前 risk field 与当前显式路线的侵入关系；它不依赖 detector 类别、box 或 track ID，也不产出 alert/action。route 无效、过期、来自未来、由 risk model 自推或仅是 offline teacher 时一律 fail closed，并稳定输出 `route_unknown_or_invalid`，不会回退到固定中心走廊。
- 真实事件真值合同已冻结为 `configs/ustrf_sc_route_conditioned_event_collection_v1.json`：正式门为 6 session × 5 scene × 正/负匹配对，共 120 episode / 60 pair。`configs/ustrf_sc_route_conditioned_event_manifest_template_v1.json` 只是空模板；当前 eligible truth 仍为 0。10-episode pilot 只允许审计采集、哈希、时钟、路线和双人复核链，不能用于效果结论或 production gate。
- 设备米制几何硬门由 `scripts/validate_ustrf_sc_device_metric_geometry.py` 执行，要求同一目标设备上的独立标定、current-frame metric depth 登记、稳定 body pose、body-local ground truth、完整 route-event truth 和同机时延/热证据。`UstrfMetricGeometryReceiptPromoter` 现在必须原子消费已验证 calibration admission，且绑定 calibration ID/source SHA；自报 `independentlyVerified=true` 不再足够。当前仍无真实 evidence bundle，所以 gate 保持 false；即使未来通过，也只授权 geometry shadow，不自动进入 App 或生产。
- Crop/tiling r1 保持冻结，不补跑 32 帧，不扫描 NMS、overlap 或 score。detector 若继续，只能另建带独立 receipt 的 crop-view FP 抑制变量（首选跨 view 一致性准入）和新的 bounded canary；不得把它并入本主线或当作硬门替代品。
- 本轮只完成合同、验证器、空模板和 pure-Kotlin 研究 seam，没有采集真实事件、没有运行 ADB/GPU、没有修改正式 App 或默认 YOLO。

当前权限结论：`route-conditioned-mainline-active / real-event-truth-blocked / device-metric-geometry-blocked / production-unchanged`。

## 7. GPU 和系统安全边界

- 驱动 610.62 仍在用；备份在 `artifacts.local/crash-diagnosis/2026-07-20/driver-backup-610.62/`。
- 596.36 WHQL Studio 安装包仅下载，未验证、未安装；驱动修复已按用户指令暂停。
- 已完成稳定 bounded 试跑的同类 GPU 配置可灵活选择 batch、样本数和分辨率，不再固定 `batch=1`、8/32 帧或全程前台守护；仍应保留独立输出和可恢复 receipt。
- 新型或明显更重的配置、长时间连续高负载、显著增加 tile/分辨率以及已有崩溃记录的路线，先做可停止的 bounded pilot，并按需要启用守护、分片和阶段性系统事件检查。禁止直接复用曾两次蓝屏的 8,580 帧、`batch=64/imgsz=320/FP16` 旧入口，也不做无检查的整夜满载。

## 8. 原始方案和参考文档是否保存

已保存。原始评估包仍位于 `D:\edge\助盲系统核心算法方案与研发优先级_Codex评估包\`，本轮没有覆盖或移动。当前核验清单：

| 文件 | 字节 | SHA-256 |
| --- | ---: | --- |
| `CODEX_EVALUATION_PROMPT.md` | 2,752 | `90544bb206a711ea13a80814d68a67ee0573d7750d58004406313e25df3ba22c` |
| `PACKAGE_CONTENTS.txt` | 644 | `74c5ee25b5384823fb764ab8dde5dd5f13e57e1f8592292f9f17db63a388ed7b` |
| `README.md` | 44,985 | `3f4db09c944c367e5c1e241ba850b66b3e3c84fb4a324032fee82d31d2b7e027` |
| `助盲系统核心算法方案与研发优先级_Codex评估稿.docx` | 765,510 | `892525324b215e19881ee055ed28dd077739aed4978b3f559f80dc633e51932e` |
| `助盲系统核心算法方案与研发优先级_Codex评估稿.pdf` | 1,476,173 | `a93503f3eece01afdcc2b188c83682abb6e82f7b878f1cad9b34bcbf3949e8e5` |
| `dual_loop_architecture.png` | 411,739 | `e16421fe19f35665134b27c8a86d57ca570d397d04dc4c0baeca97c182bdb0c0` |
| `ustrf_sc_pipeline.png` | 341,340 | `3ec8fe61a81fb43098977dcd7175c27b704ffbcf41c9ae6b3cf289721a7c2775` |

外部包 `README.md` 第 22 章保存完整 R1..R16 参考列表；仓库总记录的“参考研究”一节保存了与当前路线最相关的研究映射。原始论文全文并未全部下载到仓库；当前保存的是方案包中的引用清单、设计映射和已生成实验收据。如后续需要离线归档论文原文，应单独建立来源/许可/哈希清单，不要把链接清单误称为论文全文归档。

## 9. 新窗口可复制提示词

```text
继续 E:\linnan\linnan 的 USTRF-SC 研究任务。先完整读取：
1) docs/research/ustrf-sc/USTRF_SC_WINDOW_HANDOFF_2026-07-20.md
2) docs/research/ustrf-sc/USTRF_SC_PROJECT_RECORD_AND_RESUME_2026-07-20.md
3) docs/research/ustrf-sc/USTRF_SC_IMPLEMENTATION_STATUS.md
4) docs/research/ustrf-sc/USTRF_SC_RESEARCH_METRICS_2026-07-20.md
5) artifacts.local/evidence/ustrf-sc/research-benchmark-v13-20260720/research_benchmark_report.json
6) artifacts.local/work/codex-handoffs/USTRF-SC-SAFETY-KERNEL-R1.md

先核验现场和脏工作树，不清理、不回退、不宽泛提交。当前 V13 是 15 gate / 14 pass，唯一 device_metric_geometry_admission 阻塞；320px detector sensitivity 已失败，CPU source radial-motion/TTC-proxy 分层已完成且仅有 source-only authority。Crop/tiling r1 在 8 帧恢复 4/8 small miss，但 FP 4→14，决定 stop_after_8_frame_canary，32 帧未运行；不得放宽门槛或扫描参数回救。GPU bounded 守护收据正常，但禁止复用已两次蓝屏的 8,580 帧、batch=64/imgsz=320/FP16 旧入口。更高价值主线是 route-conditioned、object-agnostic risk field 与事件级真值。正式 App 和默认 YOLO 不变。
```
