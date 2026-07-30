# Development Log
## 2026-07-30
- 时间：2026-07-30（Asia/Hong_Kong）；执行者：violjjet。冻结并通过
  [target/track-conditioned causal radial geometry LITE R0 设计评审](docs/research/dual-loop/DUAL_LOOP_TARGET_TRACK_CAUSAL_RADIAL_GEOMETRY_LITE_R0_DESIGN_REVIEW_RESULT_2026-07-30.md)。
  第一轮独立评审先因既有 512-frame 稀疏账本不具连续因果窗口和自然事件分母而
  `HOLD`；随后仅以 truth-only 准备层冻结完整 REveL Dynamic `8,580` 帧、
  `13,014` 个唯一 ROI replay 机会、`17,160` truth rows 和 `469` 个至少 5 帧的
  primary parent events。两个 target × 三 anchor region × 三 truth state
  `18/18` cell 全覆盖，最小 cell 为 `9`。设计固定两臂、仅当前/过去帧、正号接近、
  quality、`100 ms` TTL、abstention、固定 event 分母、失败与停止门；REveL Vicon
  只允许 producer hash 后 evaluator join，旧 F-1B decision 继续 `0` 消费。
  protocol 与 input-freeze validator 均 `VALID / errors=[]`，专项测试 `5/5 PASS`。
  当前只授权实现、fixtures 与 implementation identity lock；候选 replay、truth
  join、Confirmation、Android、融合、提醒、产品与安全均未授权。
- 时间：2026-07-30（Asia/Hong_Kong）；执行者：violjjet。完成
  [双环可归因区域级接近证据源 Discovery R0](docs/research/dual-loop/DUAL_LOOP_ATTRIBUTABLE_REGIONAL_APPROACH_SOURCE_DISCOVERY_R0_2026-07-30.md)：
  保持旧 Sparse LK F-1B `NO_INCREMENT / VALID / decision SEALED` 不变，只读连接
  REveL 既有 770 个 GT 框与 Vicon 径向运动账本，得到 `770/770` 精确连接、
  `488` 个 motion-aligned 框；approaching 在 LEFT/CENTER/RIGHT 为
  `79/79/46`，每区均覆盖 green/yellow 来源身份。REveL 因而准入为
  `SOURCE_FOUND_FOR_DEVELOPMENT`，但其严格包围 Vicon pose 是 offline
  noncausal oracle，helmet/sensor marker 也不是人体包络或手机 body frame。
  successor runtime 仅保留
  `target/track-conditioned causal radial geometry / NOT_EVALUATED` 设计候选；
  未实现算法、未运行候选输出、未打开确认集，也不产生 Android、融合器、提醒、
  产品或安全权限。下一步仅建议先冻结 LITE Development round。
## 2026-07-29
- 时间：2026-07-29（Asia/Hong_Kong）；执行者：Codex。完成
  [R3 rotation-leakage source-localization formal closeout](docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_R3_ROTATION_LEAKAGE_SOURCE_LOCALIZATION_EXECUTION_RESULT_R0_2026-07-29.md)：
  唯一一次 runner formal authority 已消费，4 workers 完成 `8/8` 冻结
  rotation-only clusters、每条 `601` pairs，wall `1344.6163 s`，minimum
  coordinator-observed available RAM minimum `6,226,071,552 bytes`、swap delta
  `0/0`、residual worker
  `0`，R3/阈值/三 pair/PairState/abstention 均未修改。强制独立 validator 在
  `2/8` 后以 `PAIR:519:LOCAL_CELL_EXPANSION:COMPENSATED_FINAL:6`
  fail-closed；无 analysis、独立 receipt 或 execution decision。只读单-cell
  复算确认该 cell 已因 support `3<12` 弃权，ledger/recomputed OLS 差
  `1.0231815394945443e-11/s`，相对误差约 `6.14e-14`，但超过纯绝对
  `1e-12/s` 容差；分类为 audit-only numeric-representation protocol defect，
  不是算法成败。终态
  `NO_VALIDATED_SCIENTIFIC_RESULT / INDEPENDENT_VALIDATION_INVALID /
  ONE_SHOT_CONSUMED / NO_RERUN`；不修改或重跑 validator，不进入 R3 repair、
  C/D、正式 `480+16`、Android、产品或安全结论。机器记录见
  [execution closeout R0](docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_R3_ROTATION_LEAKAGE_SOURCE_LOCALIZATION_EXECUTION_CLOSEOUT_R0_2026-07-29.json)。
- 时间：2026-07-29（Asia/Hong_Kong）；执行者：violjjet。现场复核 Lenovo
  Y7000X IAX11（Core Ultra 7 251HX / RTX 5060 Laptop 8GB / Intel AI Boost /
  单条 16GB DDR5-6400 / 双 1TB NVMe）并完成短时 CPU、内存、项目盘与 CUDA
  基准；保留 RCLE 实测 `interactive=8 / balanced=12 / throughput=16`，普通
  host 启动器的系统内存 reserve 从 2.5 GiB 收紧至 4 GiB。修复 guarded
  launcher 对 stale、缺字段、非 complete 和 completed/total 未闭合 progress
  的误接纳；外部监控改为递归汇总进程树，并增加可选 NVIDIA 利用率、显存、
  温度、功耗。同步纠正 E/F 同属一块 ZHITAI 盘、PC NPU 尚未项目准入、Gradle
  worker 尚不可评等边界。`test_run_guarded_host_research.ps1`、新增
  `test_monitor_host_research_process.ps1`、preflight 单测、脚本语法、docs/index、
  repo hygiene 与 diff checks 通过；未触碰并行 RCLE R3 文件，未改 BIOS、驱动、
  Windows 电源计划、科学参数或 Android 端路径。
- 时间：2026-07-29（Asia/Hong_Kong）；执行者：Codex。保持旧 R2 P4
  `INTERVENTION_NOT_EVALUABLE / VALID / COMPLETE_PRE_R3_TERMINAL` 不变，
  完成 response-blind [quality manipulation successor R1](docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QUALITY_MANIPULATION_SUCCESSOR_R1_RESULT_2026-07-29.md)。
  诊断确认旧 low-texture gate 把不受 alpha 控制的物体/遮挡/材质边界混入全帧
  梯度剂量；先实现的固定 linear-RGB bilateral 在真实旧 scenes 上八个 subgroup
  均 `0/20`，因此淘汰而未调参。QMS-R1 固定为一次共享 raycast 后的材质内部
  prequantization residual contraction；旧 development identities `160/160`
  通过，全新 disjoint CAL `32/32` 通过，八 subgroup 均 `4/4`，512 frame-state
  exact residual error 为零。独立标准库 validator 与 11 个 mutation tests
  通过，receipt 为 `VALID / QMS_R1_INDEPENDENT_VALIDATION_VALID`。未运行或读取
  R3/outcome，未建立新 formal identities，未激活 successor formal，也未修改
  R3、阈值、三-pair、sequence16、Android 或 realtime。
- 时间：2026-07-29（Asia/Hong_Kong）；执行者：Codex。完成并修订
  [RCLE periodic self-motion counterfactual R2 轻量 P3 R0](docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_TRANSPORT_ANALYSIS_RUNTIME_PREFLIGHT_R0_RESULT_2026-07-29.md)：
  新增 generator-native RGB/valid-mask/K/timestamp/pose transport adapter，并以
  Pillow PNG reference transport 对冻结 R3 pair core 做 4-pair row/state 精确等价；
  新增 80-cluster、六-arm、九-member shared-resample max-t analysis implementation
  与 mutation tests。运行前冻结 uppercase `FACTORIAL/GUARDRAIL` seed literal 和
  8 个 ADVIO_14 PREFLIGHT identities；W4/W8 均完成 4816 frames / 4808 pairs，
  四类 transport hashes 逐 identity 相同。W4/W8 wall 为
  `1277.166 / 1064.115 s`，launch/minimum available RAM 分别为
  `9.80/7.58 GiB` 与 `9.38/8.28 GiB`，heartbeat 最大
  `20.065/20.094 s`，swap delta 与 residual worker 均为 0。独立 validator 不导入
  producer/adapter/analysis/runner。初始 `24.1952 / 20.1591 h` 来自错误的均匀
  guardrail 比例外推，保留为 predecessor evidence、不再作为当前结论。用户授权
  scheduler successor 后，静态相同 pose 由每 identity 重复渲染 602 次改为渲染
  1 次并复用 601 次，仍逐 pair 调用冻结 R3；W8 复现 predecessor 的实际
  `OpenCV=1 / OpenBLAS observed=18` 调度，完整实测 `677.5074 s`，四类 transport
  hash 与 predecessor 差异 0。按 `480 factorial + 16 guardrail` 分项外推并含
  10% retry reserve 为 `7.1575 h`，终态修订为
  `PERFORMANCE_QUALIFIED / VALID / P4_NOT_ACTIVATED`，选择 W8。
  未运行 480+16、未访问 formal seed、未解释科学 outcome、未调 strength、未改
  R3/阈值/三-pair，未访问 sequence16/Android/realtime，也未激活 P4。
- 时间：2026-07-29（Asia/Hong_Kong）；执行者：violjjet。将 RCLE P1 暴露的“治理压过科学问题”风险落实为全项目研究风格优化：更新 [研究治理](docs/RESEARCH_GOVERNANCE.md)、[协议模板](docs/RESEARCH_PROTOCOL_TEMPLATE.md) 与 [文档治理](docs/DOCUMENT_GOVERNANCE.md)，统一采用 scientific status / protocol status / execution authority 三轴报告；INVALID 不再抹去已观察的计算，但不能产生可签署 claim 或后继权限。新增 `SCIENTIFIC / PROTOCOL_ONLY / NON_BLOCKING` 变更分级、LITE/STANDARD/STRICT profile、按阶段最小证据包和单一 current 入口；协议错误只做薄修订并只重验受影响门，非阻断命名/文案/未来监控进入 backlog，不得创建版本或阻塞算法。该调整是现有 R2 policy 的工作与报告澄清，不修改 policy hash、既有机器合同、冻结 receipt 或历史 terminal。
- 时间：2026-07-29（Asia/Hong_Kong）；执行者：violjjet。完成 [RCLE periodic self-motion counterfactual R2 P1 keyset repair](docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_GENERATOR_GEOMETRY_KEYSET_REPAIR_R0_RESULT_2026-07-29.md)：保留 R0/R1/R2 已消费失败 receipt 与 source/lock 身份，另立 `R2_KEYSET_REPAIR_R0`，仅把 R0 historical evidence key 从错误别名 `generator_receipt.json` 修正为真实 `producer_receipt.json`，并为 generator evidence directory 与正式 receipt 加入独占创建保护。新 all-seed manifest SHA `3dcf3749…7ac6`，88 条 record 与 R2 逐字节一致；只读预检为 `VALID` 后唯一写入正式 receipt `95646437…c079`，G01–G14 为 14/14 PASS、`failed_gates=[]`、`errors=[]`，终态 `GENERATOR_GEOMETRY_PASS / EXECUTION_NOT_AUTHORIZED`。定向 20 tests 与模块 76 tests 均 PASS；未读取/运行 RCLE output，未进行 P2 blur/low-texture 校准、P3 八序列预检、P4 480+16 正式运行，未改 R3/阈值/三-pair，也未进入 sequence16/CoTracker/Android/realtime。P2 仍须另立授权。
- 时间：2026-07-29 00:03（Asia/Hong_Kong）；执行者：violjjet。完成 [RCLE periodic self-motion counterfactual R2 P1 geometry R2](docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_GENERATOR_GEOMETRY_IMPLEMENTATION_R2_RESULT_2026-07-29.md) 的版本链恢复、validator 加固与唯一冻结运行：R1 恢复为实际消费的 angle/acos validator `5be754…`、lock `b49efb…`、amendment `521fd5…` 和失败 receipt `af00df…`，不覆盖历史；R2 保持 88 条 scene record 与 R1 逐字节一致、80 MAIN 与 R0 逐字节一致、numeric seed/trajectory change 均为 0，并显式冻结 MAIN v1 + GUARD v2 schema union。R2 独立重算 G01–G14 均 PASS，其中 G08 为 160 sequence identity / 865,440 samples / 1,053 disocclusion、G13 16/16 且 602/602 可见、G14 base/guard replay mismatch 0；但正式 receipt 因把 R0 真实 `producer_receipt.json` key 误期望为 `generator_receipt.json` 而 `INVALID / INTERVENTION_NOT_EVALUABLE / HOLD_P1`。按 one-shot fail-closed 规则不修改、不覆盖、不重跑 R2，不进入 P2/P3/P4，不读取 RCLE output，不修改 R3/阈值/三-pair，也不进入 sequence16/CoTracker/Android/realtime。
## 2026-07-28
- 时间：2026-07-28；执行者：Codex。冻结并完成 [RCLE periodic self-motion counterfactual R2 design review](docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_DESIGN_REVIEW_RESULT_2026-07-28.md)：采用四个不可择优的 ADVIO pose 波形作为 motion block，在新非平面 3D 场景中冻结 `static/periodic 6DoF × clean/blur/low-texture` 严格配对设计；`4×20×6=480` 主序列只构成 80 个 `scene_seed×motion_block` cluster，另有 16 条 source-known approach guardrail。主要机制、组合与 quality-accompaniment 共 9 个 contrast 使用按 block 分层的 20,000 次 paired cluster bootstrap、10 个百分点门槛与 simultaneous max-t 区间；R3 response、strict `>0.01/s`、三 pair、reset 和 PairState 不变。冻结 14 项 3D geometry gate、response-blind quality calibration/main manipulation check、`4 vs 8 workers` guarded-host budget；12/16 workers 在本版禁止。两路隔离 AI 终审 `accept/accept`、共识 receipt `VALID`；bundle/global protocol validator、19 项 mutation tests、compile、docs/structure/hygiene 与 diff 门通过。终态仅为 `DESIGN_REVIEW_PASS / NOT_RUN / EXECUTION_NOT_AUTHORIZED`；未实现 generator、未运行 480 序列、未访问 sequence16、未运行 CoTracker/RGB/Android，也不产生自然假警、现实 gait 因果、障碍/风险、产品或安全权限。
- 时间：2026-07-28；执行者：violjjet。完成 [RCLE 时间结构诊断 R1](docs/research/rcle/RCLE_TEMPORAL_STRUCTURE_DIAGNOSTIC_R1_RESULT_2026-07-28.md)：在新 flow-direction 输出前冻结四个 ADVIO Development session 的 `0.7–3.0 Hz` signed pose、全局/径向 flow direction、pose-derived cycles、absolute-response axial phase locking 与 measurement-failure events；Stage 1 response-blind，sequence16、风险/障碍/人工 gait 标签、Android 均未访问，R3、strict `>0.01/s`、三 pair 与窗口不变。四 session pose band-energy fraction `0.729–0.924`、direction coverage `0.754–0.992`、相邻 direction cosine `0.976–0.993`，但 flow-at-pose-frequency `R²=0.020–0.035`；高响应与 failure overlap `0.176–0.471`。motion/quality routing 均 `0/4`，终态 `HOLD_MIXED_OR_INSUFFICIENT_TEMPORAL_EVIDENCE / VALID`。独立 validator 未导入 production summarizer，复算 `failures=[]`；9 项 focused synthetic/real-LK/invalid-cycle-gap/contract tests 通过。结论不把高响应称为假警，不把 pose 周期称为正常步态或因果，也不授权 quality-gate、motion-model、performance、Android、产品或安全改动。
- 时间：2026-07-28；执行者：Codex。完成 [RCLE 退化归因与 flow-quality diagnostic R0](docs/research/rcle/RCLE_DEGRADATION_FLOW_QUALITY_DIAGNOSTIC_R0_RESULT_2026-07-28.md)：在新代理提取前冻结 ADVIO sequence13/14/15/17 的相同 601-pair 身份和两阶段防火墙，Stage 1 只读 RGB/pose，Stage 2 才连接既有 R3 ledger；sequence16 保持 `SEALED_UNSEEN`。R3、strict `>0.01/s` 和三连续 pair 不变，fixed gate 仅新增 abstention/reset。高 absolute response 对 gait proxy 的 RR 在 `3/4` session `>=1.5`，blur/low texture 各为 `2/4`；flow gate 只有 `1/4` 富集高响应，`0/4` 达到 20% trigger-density 降幅，终态 `HOLD_FLOW_QUALITY_GATE / VALID`。独立 validator 精确复算、`failures=[]`，5 项 focused tests 通过；不调 gate 追结果、不恢复 rotation-only，也不产生 false-trigger、performance、Android、产品或安全权限。
- 时间：2026-07-28；执行者：Codex。完成 [RCLE natural-session expansion Discovery R0](docs/research/rcle/RCLE_NATURAL_SESSION_EXPANSION_DISCOVERY_R0_RESULT_2026-07-28.md)：在任何新 session 算法输出前，metadata-only 固定 ADVIO sequence13/14/15/17 为 Discovery/Development、sequence16 为 `SEALED_UNSEEN`；每 session 只运行一个 `10.0159–10.0175 s`、601-pair 连续片段。R3 的 strict `>0.01/s`、三连续 pair、单一连续 `PairState`、`wxyz + T_cam_imu`、去畸变有效区域与 0.5 resize 均未改。四 session support 为 `0.9867–0.9967`；在各自最高20%角速度层中，sequence13/15/17 的 compensated 触发密度和 absolute response 同时高于 raw，sequence14 未恶化，达到预冻结 `>=2 sessions` 停止规则。独立 validator 从四份 ledger 精确复算且 `PASS / errors=[]`，sealed artifact 路径为空，AUROC/F1 未计算。正式结束 standalone rotation；下一机制诊断转向步态振荡、运动模糊、低纹理和 flow-quality gate，不立即实现 reference-track、temporal consistency、bearing 或 Android。
- 时间：2026-07-28；执行者：violjjet。完成 [RCLE rotation-compensation mechanism audit R1](scripts/research/egomotion_compensated_looming/rotation_compensation_mechanism_audit_r1/RESULT_2026-07-28.md)：确认首轮 ADVIO pose 将官方 `wxyz` 错作 `xyzw`，且遗漏官方 `T_cam_imu` 到 OpenCV optical basis；新增 R3（含去畸变有效区域掩膜）、官方标定去畸变 A/B、yaw/pitch/roll 双符号 raw/correct/reverse 审计、source-coordinate LK 对齐和单进程连续 600-pair 执行。最终原始/去畸变高角速度窗三-pair 触发分别 `0.7083→0.9417`、`0.7083→0.8417`，旋转主导自然假响应假设被削弱，独立 rotation-compensation 路线停止并保留论文级负结果。阈值 `0.01/s`、三-pair、AUROC/F1、Android 均未改/未运行；ADVIO sequence 16 在修实现前已原子预留为 `SEALED_UNSEEN` 且未访问。
- 时间：2026-07-28；执行者：violjjet。采用 [RCLE 数据能力驱动研究主线 R2](docs/research/rcle/RCLE_DATA_DRIVEN_RESEARCH_MAINLINE_R2_2026-07-28.md)：Discovery 默认只保留解码、时间顺序、基本身份、许可限制和成本上界，不再要求固定十秒、同源正负、精确闭合率、RGB/pose/depth 全模态或单来源全角色；数据用途分为 `CAPABILITY_DISCOVERY / DEVELOPMENT_DIAGNOSTIC / SEALED_EVALUATION`，跨来源另作 `EXTERNAL_TRANSFER`，结果访问分为 `CONTENT_INSPECTED / OUTPUT_INSPECTED / TUNED_ON / SEALED_UNSEEN`。同来源新 person/session/route/sequence 可作为独立 holdout，随机 frame/clip 切分不可。新增保留旧 R1 的治理 v2、协议 validator/tests、10 列 active capability map 和 RCLE current；ADVIO 首轮 600-pair 结果归为 `OUTPUT_INSPECTED / SINGLE_SESSION_DISCOVERY`，旧 `RGB_SEGMENT_CONFIRMATION_R1_NOT_EVALUABLE` 不变，sealed evaluation 尚未分配，Android/产品/安全仍关闭。
## 2026-07-27
- 时间：2026-07-27；执行者：violjjet。独立 [RCLE source authority repair R1](docs/research/rcle/RCLE_SOURCE_AUTHORITY_REPAIR_R1_RESULT_2026-07-27.md) 只冻结 OpenLORIS corridor 与 MultiScan。OpenLORIS `corridor1-1/2` 通过 guarded HTTP range 只读 7z header 共 `495,819` bytes，完整列出 `42,601 / 17,408` members；逐文件 path/bytes/CRC32 与发布方 LFS SHA-256 或 hashed outer-TAR exact slice 绑定，member extraction、geometry access、RGB visual access、算法执行均为 0。用户接受 MultiScan `CC BY-NC 4.0` 后，受控浏览器验证 Files tree 解锁；`scene_00000_00/01` 的 exact ZIP bytes、LFS/Xet identity、`5763/7789 @ 60 Hz` 四流同步、`96.05/129.8167 s` 时长、JSONL 行数、严格递增 timestamp 与 pose fields 均闭合，只读 metadata range 共 `2,930,400` bytes。两来源独立 validator 均 `PASS / errors=[]`，authority 达到 `2/2`。随后另立 Source Discovery R1 candidate lock（SHA `c1a0ea53…a3c`），separate review 在 payload `0 bytes` 时 PASS。post-lock transport preflight 发现 OpenLORIS solid 7z 将 depth 与 color 交错共包：`39/39` 固定窗 cadence 合格，但 RGB-free authorized windows 为 `0`；独立复核 `PASS / errors=[]`。按冻结 stop rule 未下载 geometry payload、未继续 MultiScan depth、未运行 geometry/RGB、未启动 Android，[终态](docs/research/rcle/RCLE_UNSEEN_EXTERNAL_CONFIRMATION_SOURCE_DISCOVERY_R1_RESULT_2026-07-27.md)保持 `EXTERNAL_COHORT_NOT_EVALUABLE / VALID`。
- 时间：2026-07-27；执行者：violjjet。完成 [RCLE low-reference false-trigger R1](docs/research/rcle/RCLE_LOW_REFERENCE_FALSE_TRIGGER_R1_RESULT_2026-07-27.md)：在冻结四窗/967 pair 上以 baseline-only support-manager 反事实和 source-native geometry 互斥归因，198 次 geometry-below 旧触发中 local flow、rotation compensation、support-manager 分别为 `160/26/12`。只实现 `CAUSAL_THREE_PAIR_CONFIRMATION_R1`，不换窗、不补数据、不改 `0.01/s`、rotation/LK/affine/support-manager；below coverage `0.34783→0.02508`、positive `0.74276→0.70488`、positive 保留 94.90%、最大首触发额外延迟 0.20 s，四项预冻结门与独立 967-row 状态机复算均 `PASS / VALID`。首次 attribution 包装调用因 stdout 超时无产物退出，干净重启后双进程 598-pair 墙钟 84.5 s。终态 `IMPLEMENTATION_READY_FOR_CONFIRMATION / VALID`，只支持另立未见 all-real cross-source 外部验证，不产生 confirmation、Android、产品或安全权限。
- 时间：2026-07-27；执行者：Codex。交付独立 `com.linnan.blindassist.npu.candidate` arm64 候选 APK，QNN 2.47 依赖未进入正式包，初始化失败禁止 CPU fallback。SM-S9280 上候选 graph finalize 成功；100图 NPU P50/P95 `12/15 ms`，风险/反馈 `100/100` 对齐 CPU，14图检测差异归因为7个阈值附近缺失、6个框几何差异、3个置信度差异。修复共享事件清除后稳定风险重复反馈，三后端90帧均 recall=1、重复提醒=0、身份重建=0、最终退出2/2。102,511,366-byte 候选通过 arm64 Android ELF 16KB检查、独立UID安装/启动/卸载；正式包路径、版本、安装时间及17文件数据指纹前后相同。随后将[NPU晋升策略](docs/NPU_DEFAULT_CANDIDATE.md)重写为 v2：只有 runtime、关键风险、提醒生命周期、持续稳定、设备路由和回滚属于阻断门；无预冻结阈值的包体/冷启动及逐框/能效诊断不得事后否决。当前唯一阻断项是正式选择器尚未实现“受支持 SM8650 走 NPU、其他设备走 CPU”，因此 `candidatePromotionReady=false`，而非因86/100严格逐框等价或能效未知。
- 时间：2026-07-27；执行者：violjjet。完成 [CID-SIMS `floor3_1` disjoint geometry-stratified holdout R0](docs/research/rcle/RCLE_RGB_ALGORITHM_CID_SIMS_FLOOR3_1_DISJOINT_GEOMETRY_STRATIFIED_HOLDOUT_R0_RESULT_2026-07-27.md)：先冻结 W3–W11、W2 guard、20 秒选中间隔、full-pair geometry roles 与精确 `2 positive + 2 below-reference`；七个身份合格窗均 `299/299` 可评价、positive fraction `1.0`，W5/W9 因帧数不符身份门不评价，低参考窗为 `0`，终态 `GEOMETRY_STRATIFIED_WINDOWS_NOT_EVALUABLE / VALID`。selected RGB identity/cache/ledger 均未创建，RGB bytes `0`、算法未运行；独立 validator 复算 geometry ledger identity/aggregate/selection 为 `errors=[] / VALID`。新增专用 frozen runner/validator、11 项规则与 firewall 测试、8-worker guarded preflight；首个 launcher 因默认裸 Python 缺 `cv2` 在 claim 前退出，随后显式使用项目 venv 完成唯一 claim，防止将 preclaim 环境错误冒充科学运行。结果只否定 floor3_1 剩余固定网格同时提供两类角色的假设，不构成 RGB 失败、跨序列泛化或性能资格。
- violjjet: CID-RGB R0.
- 2026-07-27 violjjet: [approach-role R0](docs/research/rcle/RCLE_PHASE_B_REAL_POSITIVE_APPROACH_ROLE_ADMISSION_R0_RESULT_2026-07-27.md)：EVIMO2 `sanity_ll` 13 窗/3895 pair，0 准入、replay mismatch 0，`HOLD / VALID`；无 RGB、替补或算法权限。
- 2026-07-27 violjjet: [RGB algorithm canary R0 F1 design](docs/research/rcle/RCLE_PHASE_B_RGB_ALGORITHM_CANARY_R0_DESIGN_REVIEW_RESULT_2026-07-27.md) 第三轮独立审查 `PASS`，30 tests；真实 approach role 缺失，保持 `HOLD / VALID / EXECUTION_NOT_AUTHORIZED`。
- 时间：2026-07-27；执行者：violjjet。完成 [real-data geometry canary R0 唯一正式执行](docs/research/rcle/RCLE_PHASE_B_REAL_DATA_GEOMETRY_CANARY_R0_RESULT_2026-07-27.md)：activation 绑定 implementation lock `0d833b83…e2387`，单一 claim 后只处理 TUM 窗 0/3/4/6；producer 与独立 validator 各 `1196` pair，identity/schema/abstention/branch/strict-float mismatch 全为 0，终态 `VALID_IMPLEMENTATION_DEBUGGED_GEOMETRY_INTERFACE_ONLY`，无 failure receipt且未读取 RCLE RGB algorithm outcome。正式运行约 2 h 07 min，监控记录约单核、`2156.646 GiB` 累计逻辑读取；gzip TGZ 逐 pair 反复回扫和未做真实 archive mechanics 性能预检记为下一 evidence version 的实现限制，不追溯修改或重跑 R0。
- violjjet: Added [host-only 8/12/16-worker scheduling](docs/HOST_RESEARCH_COMPUTE.md) and launcher; Android and scientific parameters remain unchanged.
- violjjet: Added `scripts/monitor_host_research_process.ps1` for non-invasive
  phase, CPU/I/O, memory, bottleneck, action, stall and terminal-state
  telemetry. New long runners must publish completed/total and ETA; compressed
  tar random member access is prohibited as a repeated sample path. Host work
  exceeding 3 minutes now requires workload classification, a representative
  bounded pilot, scheduling comparison and performance qualification before an
  irreversible claim.
- violjjet: Adopted [the project-wide engineering learning loop](docs/ENGINEERING_LEARNING_LOOP.md):
  expensive work now requires explicit runtime/progress/resource expectations;
  anomalous black-box execution, resource mismatch, I/O amplification and
  repeated failure trigger diagnosis and must leave a durable prevention
  mechanism rather than only a conversational postmortem.
- violjjet: Added a mechanical host long-run gate:
  `validate_host_research_preflight.py` rejects unbound runner hashes,
  non-representative or unbounded pilots, missing progress/terminal contracts
  and incomplete formal one-shot declarations; `run_guarded_host_research.ps1`
  also checks live RAM/VRAM, injects the receipt-selected worker count, attaches
  monitoring and refuses to treat exit code 0 without progress/success evidence
  as completion. Existing claimed R0 remains untouched.
## 2026-07-26
- 2026-07-26 violjjet: [R0 review](docs/research/rcle/RCLE_PHASE_B_REAL_DATA_GEOMETRY_CANARY_R0_IMPLEMENTATION_REVIEW_RESULT_2026-07-26.md) PASS; 18 tests; lock `0d833b83…e2387`; formal TUM/RGB not run or authorized.
- 时间：2026-07-26；执行者：violjjet。完成 [PB-H1 role proxy R0](docs/research/rcle/RCLE_PHASE_B_PB_H1_ROLE_PROXY_R0_RESULT_2026-07-26.md)：实现 `R·X` 对 `R·X+t` 的 pose+depth radial expansion/parallax，受控纯旋转/横移/同速前向接近六项物理检查全部通过；固定 burned `rgbd_bonn_crowd2:0` 的 `294/294` pair 可评价。结论为旧 raw-speed gate 因果错位，同时 absolute radial aggregate 单独也不是 approach 判据；result SHA `50bc54d0…3de7` 且实现/输入哈希复验 `VALID`，下一步仅值得审计 TUM `fr2/rpy` source-native geometry。
- 时间：2026-07-26；执行者：Codex。将外部 GPT 提供的 TUM/ETH3D/ICL-NUIM/EVIMO2 清单降级整理为 [Phase B 动态数据候选池](docs/research/rcle/RCLE_PHASE_B_DYNAMIC_DATA_CANDIDATE_POOL_2026-07-26.md)：候选排序可随 PB-H1、本地缓存与取得成本调整；先做合成+Bonn burned window 的几何代理实验，再逐个 pose-first 审计，禁止批量下载和按 sequence 名称直接授予角色。
- 时间：2026-07-26；执行者：violjjet。建立[渐进式研究治理](docs/RESEARCH_GOVERNANCE.md)、机器策略和 validator：分离五个研究阶段，要求失败学习、规则质疑、信息增益/成本排序、比例化验证、实质差异重开和失败资产复用；产品认证不再阻塞论文机制研究。20 项含恶意反例的专项测试通过，高权限数据/退役证据必须引用可复算的仓库 JSON。B1 R5 不改写，closure overlay 只关闭 evidence/protocol version 与依赖 B1B，RCLE 问题保持开放并进入 Progressive Discovery。
- 时间：2026-07-26；执行者：violjjet。B1A 唯一 run 完成 6 sequence / 10 window geometry；independent replay 因 24 个 abstaining pair 的 `216` 个 blank-grid key mismatch 加 ledger identity mismatch 判 INVALID。未运行 RGB/RCLE metric，原 artifacts 保留且同版本不重跑。详见 [B1A 结果](docs/research/rcle/RCLE_PHASE_B_BONN_B1A_RESULT_2026-07-26.md)。
- 时间：2026-07-26；执行者：violjjet。Bonn B0 R1 首次 GET 完成六包共 `2,262,988,443` bytes，6/6 archive/member/CRC/timestamp 可评价并固定 10 窗；receipt `dc0ffe9a…1f86` 独立复算 `PASS / VALID`。详见 [B0 R1 结果](docs/research/rcle/RCLE_PHASE_B_BONN_FORMAL_ENTRY_B0_R1_RESULT_2026-07-26.md)。
- 时间：2026-07-26；执行者：violjjet。Bonn metadata R3 `PASS/VALID`，receipt `05a283b8…489b`；B0设计 `a0b04ac5…c757` PASS，现 `AUTHORIZED/NOT_STARTED`，payload/metrics未读。
- 时间：2026-07-26；执行者：独立 Codex validation context。唯一 sealed validation `3000–3019` 完成 `2520/2520 / PASS / VALID`，receipt `d10afb25…6365c`；未 patch、换 seed、读 partial metrics、重跑或扩权。详见 [Sealed Validation 结果](docs/research/rcle/RCLE_OBSERVABLE_SUPPORT_RECOVERY_R0_SEALED_VALIDATION_RESULT_2026-07-26.md)。
- 时间：2026-07-26；执行者：Codex。唯一 support-manager development `2000–2019` 完成 `2520/2520 / PASS / VALID`，receipt `93b4c924…214e3c`；候选、环境、schema 与输出均锁定。详见 [Development Gate 结果](docs/research/rcle/RCLE_OBSERVABLE_SUPPORT_RECOVERY_R0_DEVELOPMENT_GATE_RESULT_2026-07-26.md)。
- 时间：2026-07-26；执行者：violjjet。Observable Support Recovery R0 冻结单一 support-manager 候选与独立 development/sealed matrices；最终 design lock `3fcc21e2…52bac` 复审 PASS。详见[预注册](docs/research/rcle/RCLE_OBSERVABLE_SUPPORT_RECOVERY_R0_PREREGISTRATION_2026-07-26.md)与[审查](docs/research/rcle/RCLE_OBSERVABLE_SUPPORT_RECOVERY_R0_DESIGN_REVIEW_RESULT_2026-07-26.md)。
- 时间：2026-07-26；执行者：violjjet。Phase A coverage R1 保持原 2520 trials/门，clean `1680/1680`、stress `810/840`，但 partial-occlusion pitch worst cell `0.60 < 0.70`，终态 `STOP_CURRENT_IMPLEMENTATION / VALID`。详见 [R1 结果](docs/research/rcle/RCLE_MINIMAL_PHASE_A_COVERAGE_REVISION_R1_RESULT_2026-07-26.md)。
- 时间：2026-07-26；执行者：violjjet。Phase A R0 完成冻结 2520 trials；数值误差门通过但 clean/partial-occlusion worst-cell coverage 失败，终态 `REVISE / VALID`，receipt `14ed23e3…041ca`。详见 [R0 结果](docs/research/rcle/RCLE_MINIMAL_PHASE_A_SYNTHETIC_SIGNAL_AUDIT_R0_RESULT_2026-07-26.md)。
## 2026-07-25
- 时间：2026-07-25；执行者：violjjet。裁决并整理用户中止后的未提交现场：保留 `egomotion_compensated_looming` 的 41 个 Module/依赖文件与 12 份日期化文档，冻结为 [RCLE 前序证据](docs/research/rcle/RCLE_PRECURSOR_FREEZE_2026-07-25.md)，原 R0/R1 不再续跑且不计作 Phase A。43 项 focused tests、39 个 Python 文件 compile 与三个只读 validator 通过；当前终态仍为 R0 fail-closed、R1 non-authoritative evaluation quarantined。清理 117 个 Python 缓存文件。根 `mobileclip_blt.ts` 是 599,764,649-byte TorchScript 模型，核验其与 `E:\codex-tools\models\ultralytics\mobileclip_blt.ts` 大小和 SHA-256 `A67804D1...ED95E54` 完全一致后移除误放副本；规范副本保留。同步压缩 current 状态、修正脚本索引和冻结 Module README；未删除唯一证据，未改 App 或默认模型。
- 时间：2026-07-25；执行者：violjjet。按用户最新指导将 RCLE-RF 确认为 BlindAssist 当前研究主线，并建立 [current 入口](docs/research/rcle/README.md)。R1.0 只作为长期能力地图，R1.1 是当前唯一执行协议且现阶段只授权 Phase A Synthetic Signal Audit；Phase B–D、Risk Field、Android 主动告警、人体与生产权限均未开放。用户同时中止本项目的其他工作，因此 RCLE 成为唯一活跃研究线；Route-conditioned USTRF 保持 closed，现有 `egomotion_compensated_looming` 代码、文档和 receipts 冻结为 RCLE 前序现场，不计作 Phase A 完成，也不再自动续跑原 Looming R1。同步更新 agent 规则、根 README、文档索引与当前状态口径；未修改算法代码、实验产物、正式 App 或默认模型。文档索引、scoped 链接、RCLE 文风和 diff whitespace 检查通过；project-structure / hygiene 仍只被 8 个任务外既有 USTRF config-to-implementation 引用阻塞。
- 时间：2026-07-25；执行者：violjjet。完成并行 Bonn 连续信号评价的 authority 复核。该执行真实联结 596 对 trace 与已被隔离的 central-ROI/full-frame truth proxy，共同支持 503 对；自报 `STOP_R1A_BONN_C2_ORACLE_GATE_FAILED`，oracle equal-session Spearman `.07039`、session-block CI `[-.03435,.17513]`，uncompensated `-.04413`，oracle delta `+.11452`。该 stop 被正式降级：signal 是全图 q90 positive radial rate，truth 是中央 `u25–75% / v15–90%` ROI 的 q05 static-depth rate，空间单元不一致；truth ledger 非预注册、把派生 map projection 错标 A 级，canonical 3×3/500ms B 级 truth 仍为 `0/18`，两个 Bonn session 还共享 capture volume且缺 controlled family。因此唯一合法结论是“当前全局汇总对探索性中央 ROI proxy 的相关性弱且 session 不稳”；不能停止 R1-A Looming、oracle rotation 或 local expansion。实际 `candidate_signal_result_evaluated=true`、`truth_join_or_scoring_run=true` 已如实入账，但 `authoritative_algorithm_result_available=false`，终态 `BONN_NONAUTHORITATIVE_CONTINUOUS_SIGNAL_EVALUATION_QUARANTINED / VALID`。
- 时间：2026-07-25；执行者：violjjet。共享工作树的并行流在 Bonn truth 权威冲突尚未统一时冻结了 R1-A signal contract 与 596 对 metadata manifest，并实际解码 598 个 discovery RGB、生成 `bonn_r1a_base_flow_traces_r0.json`（9,737,654 bytes，SHA `2d6205c3...1c5`）；随后又读取 orientation/full pose 与 594 个 source-depth member，生成 594 对 oracle rotation/full-6DoF trace（SHA `756f63cc...7fc5`）。这些 producer 未读 closing truth/cell/outcome/旧窗口/validation/holdout，故 trace 不删除；但 canonical 3×3/500ms truth canary 仍为 `3<4`、18/18 cell abstain。现已如实写回 `candidate_signal_computed=true`、`oracle_trace_computed=true`，同时固定 `candidate_signal_result_evaluated=false`、`truth_join_or_scoring_run=false`；base/oracle trace 与并行 central-ROI truth ledger均隔离为无结果权威，不得 join、评分或形成算法/产品结论。当前 terminal 为 `R1_CLAIM_SCOPED_SOURCE_PROGRAM_ORACLE_TRACE_FROZEN_INPUT_AUTHORITY_PENDING / VALID`。
- 时间：2026-07-25；执行者：violjjet。完成 Looming R1 的 Bonn Leica 静态表面 truth 审计。先冻结两个既有 discovery 窗、20 个 500ms anchor、固定 3×3 网格、官方 `T_ROS/T_m`、RGB 内参、1/64 deterministic map sample、六个 depth canary 与四帧 quorum；随后流式核验 PLY `54,676,774` 点并投影 `856,075` 点 sample，只解码六个预冻结 depth，RGB/validation/holdout/signal 均为 0。官方公式数值检查与投影 canary 通过；3 个可用 depth frame 的中位 absolute/relative agreement 为 `0.054646m / 0.021345`，但两个起始单元 pose join 失败、`person_tracking2 9.9s` 无 common map support，故只有 `3<4`，未事后降门。终态 `BONN_C2_STATIC_SURFACE_TRANSFORM_CANARY_FAILED / VALID`；18/18 网格轨迹显式 `TRANSFORM_GEOMETRY_CANARY_FAILED` abstain，诊断轨迹仍保留但不升级 B 级 C2 truth。这不是 Looming 算法失败；独立复跑 receipt SHA `7ea241f9...478c` 完全一致，focused+mutation tests `7/7`。下一主边界仍是受控硬件/calibration manifest 与三个无人体刚性目标 discovery cluster。
- 时间：2026-07-25；执行者：violjjet。在 Bonn canonical discovery 上只读取 ZIP central directory、`rgb.txt/depth.txt/groundtruth.txt`，没有解码图像或运行 signal。`person_tracking2` 与 `balloon` 各形成一个从首个 RGB timestamp 起固定、不滑动的 10 秒窗，pose join 为 `298/299`、最大时差 `24.28ms / 16.59ms`；两窗 translation path/end 分别为 `1.956/1.362m` 与 `2.198/0.574m`，所以 C2 translation mechanics 为 `2/2`，C1 pure-rotation mechanics 为 `0/2`。这只使 `Bonn × C1` abstain，不否决 Bonn、也不换样。另取得官方 Leica 静态地图 section：`676,032,657` bytes，SHA `1ce51526...7d35`，单 PLY 解压后 `2,318,666,764` bytes，CRC 通过但尚未解压/读点。当前只允许下一步验证官方 `T_g = T_ROS^-1 T_0 T_ROS T_m` 与静态表面 truth；signal/图像 decode 仍为 0。
- 时间：2026-07-25；执行者：violjjet。完成 Looming R1 的 Bonn claim-scoped metadata freeze 与 discovery acquisition。官方页面列出 24 dynamic + 2 static sequence；在排除三条 prior-inspected sequence、固定 `<=550MB` 与 hash 规则后，canonical discovery 为 `rgbd_bonn_person_tracking2`、`rgbd_bonn_balloon`，validation/holdout 各两条继续密封。两包共 `568,081,295` bytes、`1,006` RGB index frame，ZIP CRC 全通过，本地 SHA 分别 `d3ef7898...39b5` / `36fb4aa5...175a`；depth index 有 3 / 2 个引用 member 缺失，按最小单元 abstain 保留，不判整包失败。共享工作区冻结切换期间误取的 prior-inspected `rgbd_bonn_crowd` 原包已移入 quarantine，receipt 明确 `NEVER_EVALUATE`，未解码图像、未运行 signal。当前 Bonn 终态 `BONN_DISCOVERY_ARCHIVES_ACQUIRED_METADATA_VALID_EXTRACTION_NOT_RUN / VALID`；下一边界是只打开 discovery 的 pose/index 形成 cell ledger，并等待受控硬件 receipt。
- 时间：2026-07-25；执行者：violjjet。落实 Looming 的声明级修正但不事后改写已收口 R0：R1-A 先检验 oracle 物理上界，冻结 raw flow、bbox growth、未补偿扩张、oracle rotation 与 full-6DoF diagnostic，只有两个 claim-support family 同方向后才另立 R1-B 部署方法。[受控采集与来源子集协议](docs/research/ustrf-sc/USTRF_EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R1_CONTROLLED_CAPTURE_AND_SOURCE_SUBSET_PROTOCOL_2026-07-25.md) 明确无人体刚性板/滑轨、独立测距/轨迹 truth、三个 discovery cluster、84 个机械 trial 和完整 unit abstention 字段。识别出本地 REveL `dynamic` 与 Bonn 旧 session 已被旧研究读取：不可冒充新 validation/holdout，REveL 单 bag 不得切段伪造 session，Bonn 后续确认须从未读 sequence metadata-only 冻结。HOT3D metadata 显示公开 train-Aria 有可组成 10 秒的连续 clip pair，但按两周来源冻结仅保留未选记录，tar/RGB 读取为 0。当前 terminal 为 `R1_CLAIM_SCOPED_SOURCE_PROGRAM_FROZEN_INPUT_AUTHORITY_PENDING / VALID`，signal、阈值、App、route/lifecycle 仍为 0。
- 时间：2026-07-25；执行者：violjjet。按独立复核修正 Looming 公共来源审计并完成 R0→R1 边界转换：AV2 的 `3761/3762` 仅降格为 lidar-filename→camera-filename 描述，annotation truth join 为 `NOT_EVALUATED`；CODa 明确区分未绑定 TACC tiny 的连续性 `0` 与 checksum-bound TDR tiny 连续性 `NOT_EVALUATED`，Range 请求新增 HTTP 206、Content-Range、精确长度和约 1 MB 总读取防护，TDR snapshot 绑定固定 SHA 与三个 exact file tuple。三来源组合改为 `NON_TERMINAL_SOURCE_AUDIT_BOUNDARY_SUMMARY`，不再制造第五个父终态。冻结 R0 以其合法 `FAIL_CLOSED_NEW_DATA_OR_TRUTH_AUTHORITY_BLOCKED / VALID` 收口，六个 signal arm 均未运行。另立 [Looming R1 声明级证据目标](docs/research/ustrf-sc/USTRF_EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R1_CLAIM_SCOPED_EVIDENCE_GOAL_2026-07-25.md)：停止漫游式完美数据集搜索，改用受控刚性目标+Bonn+REveL 的分声明确认、JRDB 近场迁移诊断、单元级 abstention 与 A–D 证据等级；route/event/lifecycle、报警、App、人体和生产不开放。
- 时间：2026-07-25；执行者：violjjet。闭合 `EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R0` 的 ADT groundtruth-only geometry cell prescreen。16 条预先冻结且永久 `SOURCE_PRESCREEN_ONLY` 的 sequence / 16 个 singleton component 中，按 10 秒不滑窗、source-native pose/object geometry 与 visibility firewall 生成的 accepted-eligible 非 skeleton object proposal 为 `PURE_EGO_ROTATION_NO_CLOSING=0`、`EGO_APPROACH_STATIC_SURFACE=5`、`STATIONARY_EGO_ACTIVE_TARGET_APPROACH=0`、`LATERAL_PASS_NO_SUSTAINED_CLOSING=0`；三个必需 cell 小于最低分母，终态 `ADT_CELL_PRESCREEN_INSUFFICIENT / VALID`。skeleton coverage 明确为 `NOT_IMPLEMENTED` 且只能 diagnostic，不能修复 accepted 分母。独立审查复算一条 positive 与一条 zero-proposal archive，先后闭合 prereg/implementation blockers；5 项 focused tests、terminal validator 与 source inventory validator 通过。RGB/VRS、旧窗口、candidate signal 读取均为 0；不扩 ADT RGB、不冻结 role split、不运行 arm，不改 App/Kotlin/YOLO/route/lifecycle。这是 ADT source/cell availability 失败，不是 looming 算法失败；下一合法边界只能是另一真实来源的 outcome-blind prescreen 或新的受控采集。详见 [ADT 预筛结果](docs/research/ustrf-sc/USTRF_EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R0_ADT_GEOMETRY_CELL_PRESCREEN_RESULT_2026-07-25.md)。
- 时间：2026-07-25；执行者：violjjet。在 ADT downloadable inventory 闭合后、读取任何 groundtruth 前，以固定 salt 和 activity/person metadata proxy 冻结 16 条 singleton-name-base sequence（四 proxy stratum 各 4），总 `main_groundtruth` 预算 705,566,181 bytes；随后只取得这 16 个 ZIP 并逐文件通过官方 SHA-1，RGB/VRS/depth/segmentation 下载为 0。另立 [ADT geometry cell prescreen goal](docs/research/ustrf-sc/USTRF_EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R0_ADT_GEOMETRY_CELL_PRESCREEN_GOAL_2026-07-25.md)，冻结 10 秒不滑窗、camera/object/skeleton clock/geometry、四类 proposal 与双模型 review；尚未运行 prescreen producer，activity proxy 不算 cell truth，ADT 仍 `HOLD_R0_ADMISSION`。
- 时间：2026-07-25；执行者：Codex。完成 `EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R0` 的 metadata-only 来源权威与 counterfactual cell 预筛。ADT Explorer 实际列出 236 sequence / 2,832 个 SHA-1+size download entry，但无官方 capture-cluster UID；UT CODa 的 23 个 TACC archive 实际对应 12 个 capture date，且 TACC archive 无发布密码学 checksum 或 TDR v2.3 immutable binding；AV2 匿名 S3 实际列出 `700/150/150=1000` 个 train/val/test log，三条 deterministic sample 的 camera/lidar/calibration/pose/map member 完整。三者均未下载/解码 payload，且没有任何来源仅凭 metadata 证明 discovery/validation/sealed holdout 的四 cell × session 分母，故终态 `SOURCE_AUTHORITY_CANDIDATES_PRESENT_CELL_PRESCREEN_REQUIRED / VALID`、`ADMITTED=0`。同时建立 `OLD_WINDOW_ADMISSION_FIREWALL_READY`：纠正旧 30 窗口本体仅含 2 个 LILocBench source，另把 canonical 41-sequence input 的 4 个 CrowdBot source 单独拒收；新 producer 只能看到 deny receipt，不能读旧 frame/outcome/threshold/score。focused validator 通过；未运行 signal、未冻结 split、未下载大包、未改 App/Kotlin/YOLO/route/lifecycle。详见 [source/cell 结果](docs/research/ustrf-sc/USTRF_EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R0_SOURCE_AUTHORITY_AND_CELL_PRESCREEN_RESULT_2026-07-25.md)。
- 时间：2026-07-25；执行者：violjjet。采用 closure 后首个独立算法研究目标 `EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R0`：并列比较相机运动补偿 looming、开放集未知障碍和短时未来占用预测，优先选择物理含义与反事实最清楚的 looming 路线。新 R0 永久禁用旧 15 对窗口及旧 route/lifecycle truth，要求至少 3 个真实 source family、按 session 隔离 discovery/validation/sealed holdout，并冻结 raw flow、bbox growth、无补偿局部扩张、rotation-compensated 主候选和 full-6DoF oracle/self-motion 上界。先报告连续 `G_t` 可分性、common support、cluster CI、反事实 suppression/retention 和 worst-source；无明确增益或最坏来源不稳立即停止，不选择报警阈值，不改 App/Kotlin/YOLO/route/lifecycle/feedback。详见 [R0 goal](docs/research/ustrf-sc/USTRF_EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R0_GOAL_2026-07-25.md)。
- 时间：2026-07-25；执行者：violjjet。完成 `USTRF_ROUTE_CONDITIONED_PROGRAM_CLOSURE_R1` 文档收口：以 [bbox-route 归因 R1](docs/research/ustrf-sc/USTRF_BBOX_ROUTE_ATTRIBUTION_R1_RESULT_2026-07-25.md) 的 `STOP_ROUTE_CONDITIONED_USTRF_DOWNGRADE_TO_DETECTOR_BASELINE / VALID` 为父终态，撤销旧状态、roadmap、持续 goal 与 handoff 中所有 active/conditional/blocked-waiting/自动后继解释。dense、bbox-route、causal lifecycle、120 episode / U0 与 architecture convergence 全部关闭；现有 YOLO/bbox 仅保留为普通 detector baseline，不删除、不重构、不替换默认模型、不改变 App。未来算法研究必须另立全新信号假设和独立证据，禁止继续使用既有 15 对窗口调 route、quantile、窗口汇总或阈值；完整规则见 [closure R1](docs/research/ustrf-sc/USTRF_ROUTE_CONDITIONED_PROGRAM_CLOSURE_R1_2026-07-25.md)。
- 时间：2026-07-25；执行者：Codex。完成 `USTRF-BBOX-ROUTE-ATTRIBUTION-R1`：正式继承 `STOP_CURRENT_DENSE_USTRF_EXPRESSION`，不调参、不加 lifecycle、不补数据、不改 App/Kotlin/Python 架构；严格复用 15 正/15 同源等长负窗口、4594 帧和父探针 4108 个 common-eligible frame，将同一 post-NMS person bbox confidence field 接到 matched、uniform、within-source cyclic shuffled 与 bbox-only。matched 逐帧 mismatch `0`，主 q90 W/T/L 为 `12/1/2`、`11/0/4`、`9/2/4`、`8/0/7`；matched 相对三个对照的直接 W/T/L 仅 `11/0/4`、`9/1/5`、`9/0/6`，且 dynamics_0 matched median delta `-0.771457`。按单一停止规则终态 `STOP_ROUTE_CONDITIONED_USTRF_DOWNGRADE_TO_DETECTOR_BASELINE / VALID`；不进入 lifecycle、独立扩样、120 episode 或 architecture convergence。独立复算 `VALID_REPLAY_MATCH`，report/frame SHA 为 `02c49982...1c6` / `860d27fb...2ef`；详见 [R1 结果](docs/research/ustrf-sc/USTRF_BBOX_ROUTE_ATTRIBUTION_R1_RESULT_2026-07-25.md)。
- 时间：2026-07-25；执行者：violjjet。按路线重启边界完成 `USTRF-FOUR-ARM-SIGNAL-PROBE-R1`：冻结新来源与 JRDB/THÖR authority、centroid/deskew/外参、Android/生产和架构扩张，只复用 15 正/15 同源等长负窗口、4594 帧、G1b `4594/4594` 语义 parity、past-only matched route 与注册 metric depth。单一 harness 在每帧一次生成同一 63×63 dense proximity field，比较 A bbox+matched route、B dense+matched route、C 同 dense+uniform、D 同 dense+within-source cyclic shuffled route；只汇总 q50/q90/q95 连续分数和正负配对排序，不选报警阈值、不运行 tracker/TTC/lifecycle。主 q90 的 W/T/L 为 A `12/1/2`、B `5/0/10`、C `6/0/9`、D `3/0/12`；B median paired delta `-0.0423`，两来源分别 `1/3` 与 `4/12` wins，固定五项稳定胜出条件全部失败，终态 `STOP_CURRENT_DENSE_USTRF_EXPRESSION`。独立第二进程全量复算为 `VALID_REPLAY_MATCH`，report/frame SHA 为 `bf0d1de6...a4a` / `3da84704...a57b`；证据在 `artifacts.local/evidence/ustrf-four-arm-signal-probe-r1/`。该结论只停止当前 metric-depth proximity dense 表达，不授权 lifecycle、扩样、Android、人体或生产。
- 时间：2026-07-25；执行者：violjjet。完成 `THOR_SOURCE_NATIVE_ID_TIME_TRANSFORM_AUTHORITY_RECOVERY_R1`：只审计官方 THÖR people/point-cloud records、论文、Qualisys 6DOF/时间格式和唯一 moving-robot run5 paired-bag canary，保持 R0 source/member/整文件窗口、9 tracks、`Citi_1`、missing policy 与五带不变。Qualisys 官方 `_6D.tsv` 格式明确 X/Y/Z 为 rigid-body local-origin 的毫米位置，THÖR record 明确 Helmet reference 是 marker-set centre，故原 `/1000` 单位门恢复；但 48-file people inventory 无 raw `.qtm`、逐帧 ID repair/recovery mask，冻结 run2 无 paired bags/实测 offset-jitter，两个 official records 无 marker→LiDAR calibration/extrinsic/error。run5 两 bag MD5 通过，只证明可配对 timestamps；约 `±5ms` nearest-QTM residual 是 100Hz sampling phase，不是 clock measurement，且 bag 无 `/tf`、`/tf_static`、`/clock` 或 calibration topic。focused tests `3/3`、独立 validator `24/24 VALID`，终态 `INDEPENDENT_PERSON_TRAJECTORY_TRUTH_AUTHORITY_ABSENT / VALID`；未读候选输出、未比较 centroid/tracker/deskew，算法/route/event/Android/人体/生产权限关闭。详见 [R1 结果](docs/research/ustrf-sc/USTRF_THOR_SOURCE_NATIVE_ID_TIME_TRANSFORM_AUTHORITY_RECOVERY_R1_RESULT_2026-07-25.md)。
- 时间：2026-07-25；执行者：violjjet。完成 `INDEPENDENT_PERSON_TRAJECTORY_TRUTH_SOURCE_AUTHORITY_AND_ADMISSION_R0`：在候选算法输出不可见时冻结 source/member/整文件 window、`Helmet_2..Helmet_10`、`Citi_1`、`0–5 / 5–10 / 10–20 / 20–40 / 40m+`、产品重点 `5–20m` 及 missing 守恒。JRDB 因 3D box→PCD point-in-box→box-conditioned centroid 循环论证拒绝；REveL 与 THÖR 虽有独立 mocap，但稳定 ID 的人工恢复 provenance、TSV unit、测得的跨系统同步误差、marker/world→sensor measurement-frame 外参与不确定度未闭合。THÖR `25,912` frame / 9 tracks 形成 `233,208` opportunities，`92,142` valid、`140,004` person missing、`1,062` reference missing；仅按非权威 `/1000` 假设的五带分母为 `43,821 / 41,035 / 7,286 / 0 / 0`，不得升级为 metric truth。focused tests `3/3`、独立 validator `39/39 VALID`，终态 `INDEPENDENT_PERSON_TRAJECTORY_TRUTH_AUTHORITY_ABSENT / VALID`；未读候选输出、未比较算法，Android/人体/独立行走/生产继续关闭。详见 [R0 结果](docs/research/ustrf-sc/USTRF_INDEPENDENT_PERSON_TRAJECTORY_TRUTH_SOURCE_AUTHORITY_AND_ADMISSION_R0_RESULT_2026-07-25.md)。
- 时间：2026-07-25；执行者：violjjet。完成 `JRDB_PERSON_3D_TRAJECTORY_FAR_RANGE_DENOMINATOR_ADEQUACY_METADATA_BLIND_REPLICATION_R0`：任何候选 label/PCD payload 前只按 timestamp、ZIP member 与 bag metadata 固定 hash 冻结 8 条未见 sequence × 360 连续帧；预注册每条 `>=100` 个 `40m+`、至少 3 条 sequence 的 label-only 分母门，4 条通过，4 条零远距分母没有运行 PCD support，也未换窗。精确复用原 LZF/field-major、upper/lower、logical-rgb360 oriented-box、`>=3` 点、四类 ledger、算术质心和 quantile/motion kernel；4 条 `40m+` support 分别为 `0.43% / 41.58% / 6.23% / 12.66%`，相对各自 `90.52% / 90.57% / 91.85% / 91.39%` 的 `0-20m` support 全部下降，终态 `FAR_RANGE_SUPPORT_DECLINE_REPLICATED / VALID`。同步画像显示 3D-only support `4/4` 更低但 residual median 仅 `3/4` 更差；fully-visible/fully-occluded pooled support `90.53%/29.11%`，零点/1–2点分母 `5,704/6,145` 严格 annotation-only/abstained，3–9点 residual P95 `0.626m` 高于 10+ 点 `0.375m`。focused tests `5/5`、独立 validator `22/22`、compile 通过；仍无 independent person trajectory truth，不开放 centroid 比较/选择、deskew、route/event、Android、人体或生产。详见 [R0 结果](docs/research/ustrf-sc/USTRF_JRDB_PERSON_3D_TRAJECTORY_FAR_RANGE_DENOMINATOR_ADEQUACY_METADATA_BLIND_REPLICATION_R0_RESULT_2026-07-25.md)。
- 时间：2026-07-25；执行者：violjjet。完成 `JRDB_PERSON_3D_TRAJECTORY_SENSOR_SUPPORT_AND_BIAS_CROSS_SEQUENCE_REPLICATION_R0`：在读取支持结果前，只按 source timestamp / ZIP central-directory metadata 从 26 个合格 train sequence 中以固定 hash 排序冻结 3 个新 sequence × 120 帧；全部 input packet/eligibility 物化后再整体 hash-bind，并精确复用原 PCD LZF/field-major 解码、logical-rgb360 oriented-box、`>=3` 点门、四类 object/pair ledger 与 quantile kernel。新序列合计 object/pair support `8,118/9,771=83.08%`、`7,822/9,679=80.81%`，centroid residual median/P95 `0.168/0.446m`；但 worst support 为 `73.67%/70.15%`、worst P95 `0.669m`，说明单序列 aggregate 不可外推。3D-only residual 方向在 3/3 可评 sequence 复现，远距仅 Clark `43` 个 object-frame 可评，跨 sequence 为 `NOT_EVALUABLE`；终态 `CROSS_SEQUENCE_PROFILE_AVAILABLE_WITH_PARTIAL_REPLICATION / VALID`。新 bag 的 dataset-wide static calibration fallback、non-consumed IMU 缺口和 RGB-PCD simultaneity 限制已逐 packet 披露；focused tests `5/5`、独立 validator `16/16`、compile 通过。仍不开放 selection、route/event、alert、Android、人体或生产；详见 [R0 结果](docs/research/ustrf-sc/USTRF_JRDB_PERSON_3D_TRAJECTORY_SENSOR_SUPPORT_AND_BIAS_CROSS_SEQUENCE_REPLICATION_R0_RESULT_2026-07-25.md)。
- 时间：2026-07-25；执行者：violjjet。完成 `JRDB_PERSON_3D_TRAJECTORY_SENSOR_SUPPORT_AND_BIAS_CANARY_R0`：从同一 Meyer Green 120 帧 immutable packet 真实解码 240 份双 LiDAR `binary_compressed` PCD，逐传感器变换和 oriented-box 查询后输出 object-frame、pair、acceleration triple 的 `sensor-supported / annotation-only / abstained / invalid` ledger。1,105/1,350 个可计算 3D object-frame、1,044/1,336 个 motion pair 得到冻结的 `>=3` 点支持；质心残差 median/P95 `0.195/0.481m`，支持率随 3D-only、遮挡与距离显著退化，`>=40m` 仅 15.28%。局部缺云/稀疏支持未关闭整段；5 项 focused tests、16 项独立 validator checks 与 Python compile 通过，终态 `SENSOR_SUPPORT_AND_BIAS_PROFILE_AVAILABLE_WITH_ABSTENTION / VALID`。该结果仍为 annotation-conditioned、单序列 diagnostic evidence，不开放 selection、route/event、alert、Android、人体或生产；详见 [R0 结果](docs/research/ustrf-sc/USTRF_JRDB_PERSON_3D_TRAJECTORY_SENSOR_SUPPORT_AND_BIAS_CANARY_R0_RESULT_2026-07-25.md)。
- 时间：2026-07-25；执行者：Codex。完成弹性证据与降级标准 R1、近期 fail-closed 粒度回顾和 `JRDB_SINGLE_SEQUENCE_NATIVE_MULTISENSOR_PERSON_GEOMETRY_CANARY_R1`。R0 packet/receipt 与 `FAIL_CLOSED_LABEL_JOIN / VALID` 历史不改写；R1 修正“完整 2D join 是 3D-native claim 前置条件”的误设，强制 source-native denominator 守恒与最小单元 abstention。同一 120 帧 packet 上，1,350/1,350 个 robot-relative 3D geometry、1,336/1,336 个相邻 annotation-derived motion pair（14 tracks）可计算；29 个 3D-only 与 24 个 2D-only 只使 cross-modal identity 降级。全部 1,350 个 3D label 均为 source-interpolated，direct observation 为 0，故仅 `DIAGNOSTIC`，selection/route/event/alert/Android/人体/生产仍关闭。标准 validator、R1 12 项 validator checks、5 项 tests 与 compile 均通过；详见 [弹性标准](docs/research/ustrf-sc/USTRF_ELASTIC_EVIDENCE_AND_DEGRADATION_STANDARD_R1.md)、[回顾审计](docs/research/ustrf-sc/USTRF_FAIL_CLOSED_GRANULARITY_RETROSPECTIVE_AUDIT_R1_RESULT_2026-07-25.md)及 [R1 结果](docs/research/ustrf-sc/USTRF_JRDB_SINGLE_SEQUENCE_NATIVE_MULTISENSOR_PERSON_GEOMETRY_CANARY_R1_RESULT_2026-07-25.md)。
- 时间：2026-07-25；执行者：violjjet。完成 `JRDB_SINGLE_SEQUENCE_NATIVE_MULTISENSOR_PERSON_GEOMETRY_CANARY_R0`：只 range-read 124,209,382 bytes，物化 Meyer Green 前 120 帧的 120 stitched RGB、240 PCD 与 2 份 label JSON，共 362 members / 110,596,529 bytes；逐帧绑定 bag RGB/LiDAR header、动态 `odom -> base_link`、`imu/data` 和 `/tf_static`。immutable packet 由第二进程从 raw payload + bag canonical JSON 精确重建；clock、双 PCD、静态链与 pose/IMU interpolation 均通过，但 1,350 个 3D object-frame 中 29 个无同帧 2D `label_id`，按冻结全量门以 `FAIL_CLOSED_LABEL_JOIN / VALID` 关闭。未用 1,321 交集回救，motion pair 为 0；route/event/alert/Android/人体/生产/commit/push 均关闭。4 项 tests、compile 与 12 项 validator checks 通过；详见 [P2 结果](docs/research/ustrf-sc/USTRF_JRDB_SINGLE_SEQUENCE_NATIVE_MULTISENSOR_PERSON_GEOMETRY_CANARY_R0_RESULT_2026-07-25.md)。
- 时间：2026-07-25；执行者：violjjet。完成 `JRDB_SINGLE_ROSBAG_NATIVE_POSE_IMU_TIME_AUTHORITY_CANARY_R0`：冻结 27 条 train bag 中最小且 timestamps/labels/双 PCD 同名存在的 `meyer-green-2019-03-16_0.bag`，只 range-read 690,606,150 bytes 单 member，CRC/SHA 绑定 725,607,175-byte bag，未下载 40 GB 全包或第二条 bag。原生 `tf: odom -> base_link` 3,183 条、`imu/data` 622 条、upper/lower Velodyne 471/478 条均覆盖外部前 120 帧时间窗，0 header 倒退；第二进程完整重解码并逐字段一致，终态 `NATIVE_POSE_IMU_TIME_AUTHORITY_PRESENT / VALID`。P1B 完成，只开放另立 P2 perception/geometry canary；person-relative motion、route/event/safety、Android、人体与生产仍关闭。3 项 tests、compile、docs index 与 scoped diff checks 通过；全库 hygiene 仍被 8 个任务外旧 config 的既有 Implementation-path 引用阻塞。详见 [R0 结果](docs/research/ustrf-sc/USTRF_JRDB_SINGLE_ROSBAG_NATIVE_POSE_IMU_TIME_AUTHORITY_CANARY_R0_RESULT_2026-07-25.md)。
- 时间：2026-07-25；执行者：violjjet。完成 `JRDB_NATIVE_POSE_AND_3D_PERSON_MOTION_AUTHORITY_AUDIT_R0`：在每进程 64 MiB 门内只读取官方 train rosbags/images/pointclouds/labels 的 central directory 与两个 label JSON，未下载 40 GB/22.3 GB/11 GB 全包。`cubberly-auditorium-2019-04-22_0` 的 RGB、双 Velodyne、逐源 timestamp、2D/3D person track、静态 transform 与同名 rosbag 均存在，前 120 帧目录完整；整段两路 PCD 各缺末两帧。官方材料证明 IMU/encoder/synchronized sensor 与 rosbag TF 的 existence，第三方 consumer 证明 `odom -> base_link -> base_chassis_link` 和双 Velodyne 可读，但 native pose/IMU topic/message/header-time 尚未直接审 payload，故终态 `NATIVE_MULTISENSOR_CANARY_ELIGIBLE_POSE_IMU_TOPIC_AUDIT_REQUIRED / VALID`，P2、risk primitive 与 route/event/safety 权限关闭。producer/validator 各 35,569,929 bytes，deterministic validator、2 项 stdlib tests、compile 与 scoped diff check 通过。详见 [R0 结果](docs/research/ustrf-sc/USTRF_JRDB_NATIVE_POSE_AND_3D_PERSON_MOTION_AUTHORITY_AUDIT_R0_RESULT_2026-07-25.md)。
- 时间：2026-07-25；执行者：Codex。完成 USTRF observability program terminal R0：hash-bound G0、JRDB single-frame/ego-motion、ARCore freshness、RGB-D replay R2/R3 与当前状态七族证据，producer/validator 两进程独立复算为 `EVIDENCE_PROGRAM_BLOCKED_BY_REAL_WORLD_AUTHORITY / VALID`。当前 source transport 可行，故不是“所有 source 不可得”；但 current canonical authority 缺失、ARCore fresh depth 仅 1/861 且 pose 为 `EPHEMERAL_PER_FRAME`、R2/R3 route/event authority 均 false，正式 G1–G7 无法执行。显式拒绝把 terminal 写成核心算法失败或权威输入下任务不可观测；Android、人体、生产仍关闭。恢复须带来 fresh metric geometry+stable pose、intended-route truth、独立 event lifecycle truth 或明确的新参与者/采集授权。详见 [terminal R0 结果](docs/research/ustrf-sc/USTRF_OBSERVABILITY_PROGRAM_REAL_WORLD_AUTHORITY_TERMINAL_R0_RESULT_2026-07-25.md)。
- 时间：2026-07-25；执行者：Codex。完成 JRDB pre-G3 `RGB_CONTINUITY_EGOMOTION_AVAILABILITY_R0`：冻结同一 sequence 的 32 stitched RGB / 31 pair、person bbox+16px mask、sparse LK、单一 full-affine RANSAC 与 28/31 availability 门。producer/validator 各 range-read 32,941,877 bytes，32/32 JPEG hash-bound，未下载 full archive。timestamp gap、657–803 features、649–792 tracks、11–12 grid cells、residual、affine condition 与 determinant 全部通过，但 20 pair 的 inlier ratio `<0.65`，仅 11/31 pair 通过，终态 `EGOMOTION_QUALITY_AVAILABILITY_INSUFFICIENT / VALID`。不降门、不加 homography/dense/source fallback、不扩 sequence、不运行 G3/G4；下一信息增量只接受 metric depth、VIO/IMU、真实 route provider 或 route-authoritative 新数据。详见 [R0 结果](docs/research/ustrf-sc/USTRF_JRDB_RGB_CONTINUITY_EGOMOTION_AVAILABILITY_R0_RESULT_2026-07-25.md)。
- 时间：2026-07-25；执行者：Codex。用户自行建立 JRDB 登录态后完成 `JRDB_SINGLE_FRAME_RGB_TIME_TRANSFORM_CANARY_R1`。登录页显示旧版 test images/timestamps/calibration 为 21 GB/1.9 MB/4 KB；27/27 timestamp sequence 与 test labels 重合。对 22,527,101,047-byte ZIP64 不做整包下载，而是冻结 64 MiB 门并仅 range-read 21,915,466-byte central directory、local header 与 341,740-byte compressed JPEG；producer/validator 各读取 22,257,329 bytes。同一 `cubberly-auditorium-2019-04-22_1/000000.jpg` 的 9-object label、capture timestamp `1555960991.4668088`、3760×480 RGB、CRC/SHA 与 calibration 闭合，独立终态 `RGB_TIME_TRANSFORM_CANARY_PRESENT / VALID`。只开放短连续窗口的 RGB/ego-motion availability 规划；父 G0、G1、route truth、signal、Android、人体和生产仍关闭。详见 [R1 结果](docs/research/ustrf-sc/USTRF_JRDB_SINGLE_FRAME_RGB_TIME_TRANSFORM_CANARY_R1_RESULT_2026-07-25.md)。
- 时间：2026-07-25；执行者：Codex。完成 `JRDB_RGB_TIME_FRAME_TRANSFORM_ACCESS_CANARY_R0`：固定官方 toolkit commit `4fbf7d6...`、公开下载页与 4,064-byte sample structure。toolkit 的 static calibration 和 stitched image 路径合同存在，sample 声明 `timestamps/` 但只有 16 个空目录、0 payload；visualiser 所称 timestamps 来自 label key，不能充当 capture clock。公开页明确 dataset 下载必须登录，Chrome 与内置浏览器均无 JRDB 登录态，故 producer/validator 两 PID 独立复算为 `ACCESS_BLOCKED_LOGIN_REQUIRED / VALID`。未猜受限 URL、未下载 RGB/point cloud/rosbag；G1、signal、route truth、Android、人体和生产继续关闭。详见 [access canary 结果](docs/research/ustrf-sc/USTRF_JRDB_RGB_TIME_FRAME_TRANSFORM_ACCESS_CANARY_R0_RESULT_2026-07-25.md)。
- 时间：2026-07-25；执行者：Codex。承接 G0 `SOURCE_AUTHORITY_ABSENT` 完成 `CANONICAL_OBSERVATION_SOURCE_AUTHORITY_DATA_PACK_R0` labels/calibration canary：两 family 有界筛选中，nuScenes 因 visibility 是六相机聚合比例且为 vehicle perspective 在 metadata 门拒绝；JRDB 因 human-comparable mobile robot、source-native truncation/occlusion 与官方 sensor transform 成为唯一 canary。官方 test-label archive 498,600,976 bytes、SHA `a6247ef...6b10d`、723 entries、0 unsafe path；未解压 5.208 GB payload，只流式复算 27 stitched sequence、27,661 frame、956,803 object、1,781 track。truncation false/true/missing 为 925,799/30,889/115，覆盖 99.98798%，missing 未默认 false；producer/validator 两 PID 得到 `AUTHORITY_CANARY_PRESENT_ROUTE_ROLE_PENDING / VALID`。RGB identity、timestamp、route-role truth 与 G1/signal/Android/人体/生产仍关闭；全量下载页面需登录，本轮未绕过。详见 [R0 结果](docs/research/ustrf-sc/USTRF_CANONICAL_OBSERVATION_SOURCE_AUTHORITY_DATA_PACK_R0_RESULT_2026-07-25.md)。
- 时间：2026-07-25；执行者：Codex。完成 `CANONICAL_OBSERVATION_AUTHORITY_AND_REPAIRABILITY_AUDIT_R0`：纠正旧 scale producer“先完整解码 candidate/lifecycle 再投影”不能作为 source-only G0-A 的协议问题，建立 A source/transport inventory、B aggregate-denominator-only availability 与第三进程 validator。A 全量重新哈希 39 条 CrowdBot + 2 条 LILoc 的 `62,229/62,229` RGB，并核验 source ledger、PNG geometry、timestamp、41/41 membership 与 263,680 person box；source geometry/RGB/time/membership 为 authoritative、bbox frame 为 verifiable transform，canonical transform 全部 unknown、authoritative severe truncation 全部 absent。B 在先验 inventory SHA 后运行，event/cell/negative/truth/oracle/outcome/signal/candidate decode 均为 0；全局缺口支配出 `0/11` independent event、`0/33` mechanical cell、`0/836` negative interval 的乐观上界，三 PID 独立复算终态 `SOURCE_AUTHORITY_ABSENT / VALID`。8 项 focused tests、Python compile 与后续仓库门通过；G1、signal、Android、人体和生产继续关闭，下一合法边界只能是新的 authoritative source/data pack。详见 [日期化结果](docs/research/ustrf-sc/USTRF_CANONICAL_OBSERVATION_AUTHORITY_AND_REPAIRABILITY_AUDIT_R0_RESULT_2026-07-25.md)。
- 时间：2026-07-25；执行者：violjjet。采纳 USTRF-SC 可观测性优先架构与持续研究总目标：保留 route-conditioned dense risk、事件级 lifecycle、安全监督与 fail-closed，正式关闭 current-input timing/token 策略搜索，将 canonical observation authority/repairability audit 设为第一边界，并冻结 scale、ego-motion、temporal-depth teacher、新 validation 与正式六臂 U0 的逐级路线、投入产出规则和永久停止门。持续授权仅覆盖最小离线可证伪研究，不开放正式 App、人体、独立行走、生产、commit 或 push；详见 [当前持续研究总目标](docs/research/ustrf-sc/USTRF_SC_OBSERVABILITY_FIRST_CONTINUOUS_RESEARCH_GOAL_2026-07-25.md)。
- 时间：2026-07-25；执行者：violjjet。完成 `ROUTE_CONDITIONED_SCALE_GROWTH_SEPARABILITY_R0` 的输出前合同审计与 fail-closed 收口：配置冻结 normalized bbox-area、600ms/5-observation/150ms-gap Theil–Sen 单变量协议、11/11+33/33+负机会<=2 门，并因父 evaluator 无 deadline 而在 signal outcome 前独立冻结 5000ms event-window delay 门。producer-preflight 复验父收据和 123→41 candidate-blind 投影后确认 62,229 帧均未绑定 canonical source-size/rotation receipt，263,680 个 observed-track 均无 severe-truncation authority；因此 signal/truth/event/oracle/negative/candidate decode 全为 0，inventory/frontier/candidate 均未生成，终态 `FAIL_CLOSED_INPUT_OR_CONTRACT_BLOCKED / VALID`。10 项 focused tests 通过；producer-preflight、独立 audit、第三进程 validator 均闭合。剩余风险只允许另立 canonical geometry input-contract repair goal，不假定 rotation=0、不硬编码 640×480、不自动进入 ego-motion/Android/opener。详见 [日期化结果](docs/research/ustrf-sc/USTRF_ROUTE_CONDITIONED_SCALE_GROWTH_SEPARABILITY_R0_RESULT_2026-07-25.md)。
- 时间：2026-07-25；执行者：violjjet。采用 USTRF-SC 下一阶段新信号可分性目标：不重复关闭已为 `CURRENT_INPUT_POLICY_FAMILY_NOT_FEASIBLE / VALID` 的旧 family，当前唯一可执行边界为 route-conditioned normalized bbox area growth 的 standalone token-qualification Pareto 审计；冻结 producer/audit 隔离、单阈值扫描、coverage/经验风险/延迟门和三种 fail-closed terminal。纯尺度失败只关闭当前 standalone 角色；表面成功只冻结 discovery candidate。后继背景运动 availability 与 ego-motion-aware expansion decomposition 必须另立 goal，保留 absolute/ego/residual 并禁止 residual 独自解释危险；Android、opener、shadow、H2、人体与生产权限均未开放。详见 [下一阶段 goal](docs/research/ustrf-sc/USTRF_SC_NEXT_STAGE_SIGNAL_SEPARABILITY_GOAL_2026-07-25.md)。

## 2026-07-24
- 时间：2026-07-24；执行者：Codex。完成候选无关 causal route-relative intrusion signal R0：不再调资格时长/TTL/renewal，而是在任何 signal output 前冻结 5-frame route-relative radial convergence、lateral convergence 与 normalized bbox expansion 的 `2-of-3` 单信号。producer 先证明 C1/C2/C3 的 123 条 preoutput trace 在 bbox/route/reset/time 上逐帧一致，再折叠为 41 序列 / 62,229 帧；signal inventory 冻结前 truth/event/oracle/负暴露解码均为 0。结果 `SIGNAL_REJECT / VALID`：1,903 个激活只覆盖 `7/11=21/33`，比旧 timing family 乐观上界少 `1` event / `3` cell；负暴露 `43/4.956min=8.6759/min`，95% Poisson UCB `11.1877/min`，远高于 `<=2 / <=0.50/min`。该新信号直接淘汰，不调窗口/组合/阈值，不生成 policy、不接 opener。详见 [R0 结果](docs/research/ustrf-sc/USTRF_CAUSAL_ROUTE_INTRUSION_SIGNAL_R0_RESULT_2026-07-24.md)。
- 时间：2026-07-24；执行者：violjjet。完成 `CURRENT_INPUT_POLICY_FEASIBILITY_BOUND_R0`：只对 track scope、active route relation、route validity、reset 与 causal elapsed timestamp 构成的共享单调 lease family 求经验上界；保留 two-frame、one-token/track-reset、fail-closed、no-renewal，为 coverage 上界仅乐观忽略 nominal TTL。36 candidate cell 先去重为 12 个候选无关事件，41 序列 / 62,229 帧形成 31,500 个 activation interval 与 29,424 个完整 frontier segment；最大 coverage 仅 `8/11=24/33`，在 `4.956min × 0.50/min` 即最多 2 个负 token 约束下仅 `2/11=6/33`。终态 `CURRENT_INPUT_POLICY_FAMILY_NOT_FEASIBLE / VALID`；未输出 policy/threshold/witness，未改 TTL/renewal，未接 opener。可信风险 floor 仍不足且不作为不可行理由。详见 [R0 结果](docs/research/ustrf-sc/USTRF_CURRENT_INPUT_POLICY_FEASIBILITY_BOUND_R0_RESULT_2026-07-24.md)。
- 时间：2026-07-24；执行者：Codex。完成 candidate-independent policy failure attribution R1：只读复算冻结 policy gate，不改 500ms 资格/TTL/失效顺序。以 timestamp 半开区间将 24 个 supported-cell miss 展开为 96 次互斥 oracle qualification opportunity：资格不足 `39`、TTL 后 oracle `39`、relation gap 提前失效 `12`、route unknown 提前失效 `6`、track `0`、unexplained `0`；24 cell 中 6 个显式保留 mixed 原因。34 个负暴露 token 全部按 source/sequence/invalidation reason 联结，TTL/relation/track/route 为 `16/9/8/1`。终态 `POLICY_FAILURE_ATTRIBUTION_CLOSED / VALID`；父 `POLICY_COVERAGE_REJECT` 不变，不生成 successor policy、不接 opener、不开放更高权限。详见 [R1 结果](docs/research/ustrf-sc/USTRF_CANDIDATE_INDEPENDENT_POLICY_FAILURE_ATTRIBUTION_R1_RESULT_2026-07-24.md)。
- 时间：2026-07-24；执行者：violjjet。冻结 candidate-independent causal token policy/risk gate R1：只读取父 R0 的 41 条 truth-blind full-sequence ledger，采用 `2 frames + 500ms` active-relation 资格、500ms TTL、reset/route unknown/track unobserved/relation gap/TTL 即时失效与同 track/reset 再资格化抑制；41 ledger / 62,229 帧 inventory 冻结、truth/event/oracle 解码为 0 后才 post-hoc 联结。结果为 `POLICY_COVERAGE_REJECT / VALID`：有效期内 supported coverage 仅 `9/33`，3 个 no-active-relation cell 继续关闭；1,448 token 中 1,445 extra，半开负暴露 `34/4.956min=6.86/min`，95% Poisson UCB `9.13/min`，cluster 支持不足。未接 opener、未改 C1–C3/clearance，不能进入比较、selection、L2/L3、shadow/H2、人体或生产。详见 [R1 结果](docs/research/ustrf-sc/USTRF_CANDIDATE_INDEPENDENT_CAUSAL_TOKEN_POLICY_RISK_GATE_R1_RESULT_2026-07-24.md)。
- 时间：2026-07-24；执行者：violjjet。冻结 truth-blind causal per-track attribution-token producer R0：只读取 detector/T0 track、route relation、route validity 与 reset，硬拒绝 event/truth/window/future/clearance/oracle/candidate 输入；先在独立进程验证 123 条 C1–C3 runtime 投影逐帧一致并折叠为 41 条候选无关 full-sequence ledger / 62,229 帧，truth/event/oracle 解码均为 0，第二进程复验 inventory 后才 post-hoc 联结。结果为 `HOLD_FOR_POLICY_GATE / VALID`：33/33 oracle-supported cell 覆盖，3 个无 active relation cell 继续关闭，unknown/reset/duplicate token 为 0；但 5,126 枚 token 中 5,113 枚为 extra，4.956 分钟负暴露内 153 枚（30.87/min），并完整记录 6,328 次被抑制重复激活。未修改 C1–C3、opener 或 clearance，不能进入集成、比较、selection、L2/L3、shadow/H2、人体或生产。详见 [R0 结果](docs/research/ustrf-sc/USTRF_TRUTH_BLIND_CAUSAL_PER_TRACK_ATTRIBUTION_TOKEN_PRODUCER_AUDIT_R0_RESULT_2026-07-24.md)。
- 时间：2026-07-24；执行者：Codex。单变量 route-invalid + reset-scoped lifecycle guard 只读 A2 `123/123` trace；unknown/stale active 从 `12,621 / 7,165 / 12,759` 降为 0，known→invalid-active `1,235 / 801 / 1,238` 同帧关闭且跨 reset key 为 0；validator 重算父/新 trace 各 `186,687` 帧为 `VALID`，但仅 known-route relation closure 可获 credit，clearance 仍为 `0/12 / 1/12 / 0/12`，overall gate false。未重跑候选/detector、补 consume timestamp、比较或开放 L2/L3/shadow/H2/人体/独立行走/生产权限。详见 [R1 结果](docs/research/ustrf-sc/USTRF_ROUTE_INVALID_RESET_LIFECYCLE_DIAGNOSTIC_R1_RESULT_2026-07-24.md)。
### USTRF route-target R2-L1 trace-only metric profiles
- 时间：2026-07-24；执行者：violjjet。新增 profile-only 冻结合同、schema、runner、validator 与 7 项 focused tests；只读采用 A2 terminal 的 `123/123` 权威 trace，并绑定 A3 completion、A4 memory validation、eligibility protocol/mask/receipt 和三份 post-output truth。评分前逐条复核 trace/authoritative-receipt SHA、四元 frame identity、每候选 `41` ledger / `62,229` 帧 / `15` reset；候选重跑、新权威 trace 和新数据均为 0。
- 结果：终态 `METRIC_PROFILES_COMPLETE / VALID`，validator 重算 `186,687` candidate-frame。三个 profile 的 critical miss 均为 `0/8`，但 `n=8 < 59`，只能 `estimate_only / bound_sufficient=false`；clearance 分别为 `0/12`、`1/12`、`0/12`，unknown/stale active alert 分别为 `12,621`、`7,165`、`12,759` / `62,229`，均存在硬 veto；repeat 分母仅 `3/2/1`，evidence age 因 consume timestamp `0/62,229` 为 `not_evaluable`。L0 三项保持 diagnostic-only；未比较、排名、selection 或开放 L2/L3、shadow/H2、人体、独立行走和生产权限。详见 [R2-L1 metric profile 结果](docs/research/ustrf-sc/USTRF_ROUTE_TARGET_R2_L1_METRIC_PROFILE_R1_RESULT_2026-07-24.md)。

### USTRF route-target L1 candidate replay R2
- 时间：2026-07-24；执行者：violjjet。新建 replay-only R2 namespace，精确绑定旧 exploratory failure、R3 `41/41` completion、两个唯一 canonical input root、冻结 C1–C3/T0/route/reset 与独立 terminal/schema；同一 ledger 多根命中、compact/successor 漂移、partial attempt 或候选/config hash 漂移均 fail closed。权威 trace 只保留确定性状态/decision，wall time/RSS 留在 receipt；attempt-local trace+receipt 验证后才发布 authority。
- 执行：初始 R2 在首个 attempt 前因 Windows 长路径失败、trace 为 0；A1 短 root 完成 10 条后仍在原子临时后缀处触发长路径。用户明确将本次 C1–C3 replay 内存门从 6 GiB 修订为 4 GiB；A2 同时使用短哈希 trace path，按父 receipt/hash 引用继承 A1 的 10 条完整 trace、不重跑，并新运行其余 113 条。
- 结果：C1/C2/C3 各 `41/41` ledger、每候选 `62,229` 帧与 `15` reset，总 `123/123` 权威 trace、`186,687` candidate-frame、`45` reset。A2 independent validator 逐 trace 重放确定性状态为 `VALID`；A3 strict-schema finalization 也为 `VALID`。原 A2 启动时 4 GiB 检查观测到 `9,615,626,240` bytes，但未持久化逐 ledger 观测；A4 因而在 123 条独立确定性复演前逐条执行真实 4 GiB fail-closed 检查，最小 `7,592,321,024` bytes，`PASS`，且新权威 trace 为 0。未做 truth join、metric profile、比较、winner/ranking/selection，也未开放 L2/L3、Android shadow、H2、人体、独立行走或生产权限。详见 [R2 结果](docs/research/ustrf-sc/USTRF_ROUTE_TARGET_L1_CANDIDATE_REPLAY_R2_RESULT_2026-07-24.md)。

### USTRF route-target L1E R3 remaining-shard continuation A1–A3
- 时间：2026-07-24；执行者：Codex。冻结 `R2-L1E-RECOVERY-B1-CONTINUATION-A1`，保留首分片 B1 配置、实现与收据哈希；新增双 canonical root 覆盖复核、严格串行父编排器和独占 child 锁。每个 fresh child 只处理冻结顺序中的下一缺失 CrowdBot ledger，compact successor 验证后立即退出；无效/半写 pair、重复权威根、额外 ledger、并发 child、非 CrowdBot 缺口、覆盖漂移或单 ledger 三次尝试耗尽均 fail closed。完整输入门固定为 `41/41`、`62,229/62,229`、`15/15 reset`，父流程不导入或执行 C1–C3。
- A1 在原 6 GiB 门下成功补齐 9 条至 `12/41`；随后一次真实 readiness 内存失败和两次 Windows 长控制回执路径失败，按冻结尝试预算写出 `FAIL_CLOSED_LEDGER_ATTEMPTS_EXHAUSTED`。经用户明确指示将门修订为 4 GiB 后，A2 用短哈希控制路径补齐 1 条至 `13/41`，但在 successor 已验证后写 host receipt 时再次触发 Windows 长路径失败。A3 保留 4 GiB 门并使用 Windows extended-path 原子写，严格串行完成剩余 28 条，28 个 child 成功、0 失败。
- 最终独立重算为 `41/41` ledger、`62,229/62,229` 帧、`15/15 reset`；相对初始 `3/41` 共补齐 38 条、56,180 帧。终态为 `CANONICAL_INPUT_41_OF_41_COMPLETE`，`c1_c2_c3_executed=false`，candidate trace/profile 均为 0；C1–C3 仍须作为下一独立阶段显式启动。

### USTRF route-target L1E materialization recovery R3
- 时间：2026-07-24；执行者：violjjet。新建独立 `R2-L1E-RECOVERY-B1` 阶段，保留父 R2/A1 `FAIL_CLOSED_EXECUTION_ABORTED` 与旧重试预算；将 Android 输入运输改为 `/data/local/tmp -> run-as target -> targetContext.filesDir`，并保持冻结 6 GiB 可用内存门，以 6 次 readiness 采样、加载后/启动前复查和单进程单分片控制资源。SM-S9280 transport canary 逐一通过首个 CrowdBot 分片 `1,455/1,455` RGB 哈希；随后同一分片完成 Android Canvas/TFLite raw、流式拉取、逐帧回执、host decode、compact ledger 与 successor 验证。跨阶段 canonical input 进度为 `3/41` ledger、`6,049/62,229` 帧，剩余 38 条；C1–C3、trace/profile/selection/L2/L3/shadow/H2/人体/生产仍为 0 或 false。双 APK build、真机两个 `OK (1 test)`、6 项 focused contract tests、Python compile 和 cleanup 通过。详见 [R3 结果](docs/research/ustrf-sc/USTRF_ROUTE_TARGET_L1E_MATERIALIZATION_RECOVERY_R3_RESULT_2026-07-24.md)。

### USTRF route-target R2-L1X-L2P fail-closed recovery and preregistration
- 时间：2026-07-24；执行者：violjjet。在任何新 C1–C3 输出前冻结 L2 fresh-selection 的 8 required metrics、原性能门、primary/tie-break、单次运行、两-family/逐-family分母、worst-source、hard veto、数据角色与唯一 provisional selection 语义，并建立 `executable=false`、`candidate_id=null` 的 L3 6-session/60-pair/LOSO/bootstrap lockbox 模板；L2/L3 validator 与 38 项 mutation tests 通过，未下载或新增 replay 数据。
- 执行：R2 使用独立 evidence/attempt/device namespace 并保留父 R1 failure。原 R2 三次在新远端路径与旧 cleanup 白名单不兼容处、设备/raw 前 fail closed；outcome-unseen A1 仅修路径白名单，前两次 instrumentation 因 app external-files manifest materialization 不可见而无 receipt/raw，第三次在 bundle load 后可用内存 `5,512,597,504 < 6,442,450,944` bytes，尝试耗尽。最终 `FAIL_CLOSED_EXECUTION_ABORTED` validator 有效，仍为 2/41 ledger、4,594/62,229 帧、15/15 reset，C1–C3/trace/profile/selection/L3/shadow/H2/人体/生产均未运行或开放。详见 [日期化结果](docs/research/ustrf-sc/USTRF_ROUTE_TARGET_R2_L1X_L2P_RESULT_2026-07-24.md)。

## 2026-07-23
### USTRF route-target 证据成熟度分层标准 V2
- 时间：2026-07-23；执行者：violjjet。保留 R1 `DATA_BLOCKED / STOP_SOURCE_SEARCH` 与候选未运行事实，新建机器可校验的 L0–L4 evidence-maturity 标准：recall/critical/repeat/clearance/false-alert exposure/evidence-age 各自使用独立 eligibility 与分母；terminal clear 缺失只限制 clearance，right-censored 不得记成成功、失败或零延迟，空分母不得以 `0` 通过。
- 权限：现有 LILocBench/CrowdBot 最高只能在新 eligibility mask 冻结后进入 L1 exploratory profile；L2 需要新鲜两-family selection，L3 需要 6-session/60+60/LOSO confirmation，L4 仍是 production-isolated Android shadow。R1 性能门不降低，unknown/stale alert、身份唯一性、因果路线、missing 不得 clear 与 candidate-blind truth 均保留。
- 停止与验证：来源工作每轮最多 2 family、每来源 2 canary、默认 2 GiB；连续两个 family 不合格后停止为 `STOP_DATA_COLLECTION_AT_CURRENT_LEVEL`，不再用无限搜索或全局 block 抹掉局部证据。Clearance 删失只从 observed truth clear 起算，pre-clear 缺失单列 observability；L3 按来源族固定分层、族内 session 重采。验证器锁定各层权限与超预算预注册；validator、Python compile、diff check 和 28 项含 mutation 的 focused tests 全部通过。当前仍为 L0，候选、Android shadow、人体与生产权限均未开放。详见 [V2 当前标准](docs/research/ustrf-sc/USTRF_ROUTE_TARGET_EVIDENCE_MATURITY_STANDARD_V2.md)。2026-07-24 续记（执行者：violjjet）：R2-L1 已用候选盲 materializer 和完整重建 validator 覆盖 LILocBench/CrowdBot 6,369 个 event/proposal unit × 8 指标，固定 62,229 帧、62,188 个相邻 pair、4.956268516min 严格负暴露及全部排除/删失原因；`critical_miss`、`clearance`、`unknown_or_stale_alert` 获 L1 探索资格，`repeat`/`evidence_age` 为条件 L1，其余三项保持 L0。未运行或读取 C1–C3。validator 18 项、R2-L1 38/38、父 V2 28/28 tests、compile、文档索引、secret scan 和 scoped diff check 通过；结构/卫生门仅余 8 项旧 R1 告警。下一任务须在冻结断点重置并逐 ledger 分片闭合 canonical raw，只产探索 profile。详见 [R2-L1 结果](docs/research/ustrf-sc/USTRF_ROUTE_TARGET_METRIC_ELIGIBILITY_R2_L1_RESULT_2026-07-23.md)与 [R2-L1E 通宵目标](docs/research/ustrf-sc/USTRF_ROUTE_TARGET_L1_EXPLORATORY_PROFILE_OVERNIGHT_GOAL_2026-07-24.md)。R2-L1E 续记（执行者：violjjet）：新增逐 ledger Android Canvas raw exporter、compact successor、独立 runner/schema/validator；2/41 ledger、4,594/62,229 帧 canonical raw 已验证，39 ledger、57,635 帧形成精确缺口。冻结 6 GiB 系统可用内存门在初始尝试和两次有界重试中均触发，首个 CrowdBot device attempt 未创建，终态合法收口为 `FAIL_CLOSED_EXECUTION_ABORTED`；候选/trace/profile 均为 0，selection、Android shadow、H2、人体和生产权限关闭。父 validator、R2-L1E validator、16 项 mutation tests、独立替换攻击复核与双 APK build 通过。详见 [R2-L1E 结果](docs/research/ustrf-sc/USTRF_ROUTE_TARGET_L1_EXPLORATORY_PROFILE_R1_RESULT_2026-07-24.md)。
### USTRF route-target evidence closure R1 启动
- 时间：2026-07-23；执行者：violjjet。基于 detector coverage 硬门通过与 T0–T3 事件门失败，新建 production-isolated Module/预注册，冻结五态逐人路线角色、三条单接缝 oracle、三个结构候选及两来源 sealed holdout 门；父 evidence 已绑定，seen 盲审 bundle 已复核 `4594/4594` RGB 哈希并联结 route receipt，含 3,745 个 truth seed boxes。15+15 只具归因权限；route-role truth/holdout 未物化，detector/`.35`/NMS/tracker/App/H2/深度/TTC/route-risk flip 均未开放。详见 [R1 预注册](docs/research/ustrf-sc/USTRF_ROUTE_TARGET_EVIDENCE_CLOSURE_R1_PREREG_2026-07-23.md)。
- 时间：2026-07-23；执行者：violjjet。首组 CrowdBot 两来源完成 16/16、22,856 唯一 RGB 的物化、重复时间戳 last-message-wins 修复、TF 回填、0 精确/近重复审计、双视觉 pass 与 route/person 融合，但真值窗口准入为 `0/2`：旧协议把无关未知人升级为整窗不可评，并继承相机不可见的 LiDAR event onset，6,340/6,340 event proposal 全隔离；未运行任何候选。随后在候选盲条件下以固定容量门拒绝仅 4.526min 的 `1203 manual`，冻结 `0410 mds + 1203 shared-control` 23 条替换 holdout 与“可见 metric-person 正事件 / route-relevant complete 负帧”修正协议；候选对未知人物告警逐来源硬失败，`.35`/NMS/tracker/C1–C3/H2 不变。两来源 RGB-D/TF canary 均通过；`1203` 的 BGR8 无损转 RGB 与来源原生坏行单列限制。最终预注册 `f68a59cf...7f72a1`；候选盲静态审计再以 `9af8c307...34d148` scoring amendment 修复全量 false-alert numerator/分母不一致，36 tests 通过，23 条正式顺序物化已启动。详见 [R1 阶段结果](docs/research/ustrf-sc/USTRF_ROUTE_TARGET_EVIDENCE_CLOSURE_R1_RESULT_2026-07-23.md)。
## 2026-07-22

### USTRF 前沿论文研究指导与 R3 LILocBench 来源准入
- 时间：2026-07-22；执行者：violjjet。OpenLORIS 准入 0/3 后按冻结 24/12/0.03/0.50 替换到 LILocBench：`dynamics_0` 与 `lt_changes_dynamics_0` 完整适配为 2397 / 8377 帧，双隔离 reviewer 均准入并由第三模型冻结 3 / 12 个 canonical 事件，累计 `2/3`；三源 evaluator、U0 与生产仍关闭。后续 tracker/TTC R1 的 4594 帧 host 0-person 结果被 `detector_taxonomy_coverage_v1` 定位为 `[1,84,2100]` 解码轴错误：正确 host/SM-S9280 分别有 2639/2617 个 person proposal 帧，15/15 正事件区间至少出现 proposal；但 PIL/Android Canvas input 与 raw exact parity 均为 `0/4594`，30 帧在 `.35` 上分歧，且缺目标 person bbox truth，故 G3–G5、T0–T3、H2、训练/App/生产继续 fail closed。模块 13 tests、逐帧 manifest/device/host receipt 身份绑定、双 APK build、SM-S9280 全量 4594 帧 0 failure 与完整仓库门以本轮最终复跑为准。
- 时间：2026-07-22；执行者：violjjet。将 13 篇有效论文保存到本地 `artifacts.local/downloads/papers/2026-07-22-ustrf-frontier-guidance/`，排除已撤回的 Eye4B 旧稿并改用作者指定后继版本；新增证据—论点映射、固定实验协议和停止条件。论文只进入可证伪研究臂，不改变 App、默认模型或生产权限；13/13 PDF 的解析、首页、页数和 SHA-256 已核对，仓库门禁以本轮最终复跑为准。

### USTRF 二维路线代理独立实验 App
- 时间：2026-07-22；执行者：violjjet。
- 范围：新增可与正式包并存的 `ustrfExperiment` build type（包名 `com.linnan.blindassist.ustrf.experimental`），在该变体中用 `UstrfImageRouteProxy` 和 object-agnostic risk evidence 直接替代旧 `RiskAnalyzer` 决策入口；debug/release、正式模型资产和默认 App 行为不变。
- 边界：当前只使用 CameraX 同帧时间戳、YOLO bbox 与固定画面中心假设路线，不提供米制深度、稳定姿态、地面、物理 TTC 或真实导航路线；无证据不等于安全，输入/时钟异常一律 HIGH 并提示停下重扫。实验版常驻显示“不可用于独立行走”。
- 验证：`:core:ustrf:test`、`:core:assist:test`、`:core:ui:testDebugUnitTest`、`:feature:assist:testDebugUnitTest`、App debug/实验变体编译与实验 APK 构建通过；SM-S9280 已并存安装、完成首次授权并进入实时相机页，确认实验包名、版本和常驻警示。最终 lint、结构/文档/卫生门与 APK 哈希以本轮交付记录为准。

### USTRF-SENSOR-REPLAY-R2 多来源同步 RGB-D+pose 回放
- 时间：2026-07-22；执行者：violjjet。
- 范围：新建 production-isolated `ustrf_sensor_replay` Module，冻结统一 RGB/metric-depth/camera-pose 合同、三来源许可/哈希和跨来源门；Agent 自动取得 ETH3D、ICL-NUIM，并复用已哈希 TartanAir archive，各规范化连续 120 帧。未扩 120 episode、未运行 U0/ARCore、未改 App 或默认模型。
- 结果：三来源 source-aligned 均为 `1.0`，时序 depth reprojection p95 分别为 `4.13/9.15/288.83mm`，geometry transport `3/3` 通过，TartanAir 为 worst source。两次隔离模型审核均拒绝 route/event admission；独立 pose estimate、route truth 和事件 lifecycle truth 缺失，五项闭环指标保持不可评，verdict 为 `DO_NOT_SELECT_HARDWARE`。
- 验证：Module tests、Python compile、规范化哈希重算、三来源 replay、review consensus、项目结构、仓库卫生、文档索引与 diff check；详见 [R2 结果](docs/research/ustrf-sc/USTRF_SENSOR_REPLAY_R2_RESULT_2026-07-22.md)。

### USTRF 全模型代理 pilot 与 ARCore frame-bound 停止门
- 时间：2026-07-22；执行者：violjjet。
- 范围：按用户的全模型替代规则，以图像模型生成 5 场景/10 episode matched pair，并用两个隔离模型 run 复核；新增生成/review/raw-media 哈希审计与 17 个反绕过测试。随后在 SM-S9280 自动运行独占 ARCore `Session` 单 `Frame` canary，不要求用户移动或人工验收。
- 结果：模型代理 10/10 接受、1000 解码帧重算通过，只开放正式代理矩阵扩展，U0 永远 false。设备 150 行中有 139 个唯一 Camera2 时间戳与 139 个 camera-image pair，但 raw depth/confidence、tracking、valid pair 与稳定 Anchor 均为 0；host verdict 为 `FREEZE_FRAME_BOUND_METRIC_GEOMETRY`。
- 决策：按 `100 / 0.95 / INTER_FRAME_STABLE` 停止条件，不扩 120 episode、不运行 U0、不以人工动作回救；App/runtime/training/production authority 均未改变。详见 [R1 结果](docs/research/ustrf-sc/USTRF_MODEL_PROXY_FRAMEBOUND_R1_RESULT_2026-07-22.md)。
- 验证：model-proxy 17 tests + ARCore host 9 tests（合计 26）通过、Python compile、`:ustrf-shadow-benchmark` compile/assemble、SM-S9280 instrumentation `OK (1 test)`；项目结构、文档索引、仓库卫生与 diff check 通过。

### USTRF stride-4/P2 小目标 detector R1.2d 受控研究
- 时间：2026-07-22；执行者：violjjet。
- 范围：在许可/哈希/精确 bbox 几何闭合的数据收据上，以 YOLO26n 共享骨干做三 seed P2 stride-4/P3 配对训练；固定 `.05/.45/.30` 阈值、640 输入和既有 12 事件 truth-blind 协议，并把 YOLOE-768 只作为外部参考。未读 R1.3、未改 App、默认模型或生产权限。
- 数据与受控性：2,106 个唯一图像，Pittsburgh 训练、17 个其他城市来源验证，事件帧与 synthetic/provisional 标签均未入训。先后在结果前拒绝未显式标签去重、trainer 丢失共享初始化和残留进程显存竞争三轮；最终 v4 六份训练回执的同 seed 骨干哈希全部一致。
- 结果：P2/P3 三 seed 均为正事件 `4/6`，均漏 London `0/22` 与 Bridge；YOLOE 为 `5/6`。P2 离线 small/London-like recall 配对均值提高 `+2.20pp/+2.54pp`，但正事件增益为 0、假检测增加 `+0.236/图`，路线内未分配检测压力 `486–640` 且波动更大。假设判定 false，不选 seed、不调阈值、不运行 R1.3/INT8/Android/生产替换。
- 后续：停止新 detector 训练，第一优先执行既有 `route_obstacle` 正/负 matched pair 与 10-episode 采集链；第二优先只做独占 ARCore 单 frame 绑定 RGB/raw-depth/pose 的可停止 canary。正式 truth 与 device metric geometry 双门前不运行 U0。详见 [post-R1.2d 计划](docs/research/ustrf-sc/USTRF_POST_R12D_NEXT_WORK_PLAN_2026-07-22.md)。
- 验证：R1.2d focused `6 tests OK`；六份训练/七份评测与多 seed 汇总收据闭合；文档索引、仓库卫生与差异检查以本轮最终命令为准。详见 [R1.2d 结果](docs/research/ustrf-sc/USTRF_CROSSCAM_SMALL_TARGET_R12D_RESULT_2026-07-22.md)。

### USTRF Bangkok 替换与 R1.2c v2 真机事件门
- 时间：2026-07-22；执行者：Codex。
- 范围：前向物化 Bangkok 替换 Japan 的 12 事件清单并重跑六正例 truth—路线 oracle；只在 `6/6` 后导出并执行唯一同权重 FP16-768 GPU 候选，事件门先于 600 秒 soak。未读 R1.3、未改 App 默认模型/反馈路径。
- 结果：oracle `6/6`；机械 canary 通过。SM-S9280 完整事件门为正例 `5/6`，London 22 帧仍未关联；负例假告警、重复交付、共现接管、身份切换均为 0，歧义率 `3.65%`。事件门失败后跳过 soak，设备门显式未评估，R1.3/训练/生产权限保持 false。
- 后续：关闭同权重分辨率搜索，前瞻冻结 stride-4/P2 小目标 detector 假设；当前候选数 0，须先补唯一权重、训练 manifest 与审查/许可/精确几何收据。验证为 cross-camera `30 tests OK`、JDK 17 双 APK 构建通过。详见 [R1.2c v2 结果](docs/research/ustrf-sc/USTRF_CROSSCAM_R12C_V2_RESULT_2026-07-22.md)。

### USTRF R1.2c 非 R1.3 seen positive 预注册
- 时间：2026-07-22；执行者：Codex。
- 范围：从 2026-07-19 已打开的 Wikimedia Commons/POPtravel Bangkok Modern Center 来源冻结 `bangkok_tactile_cone_intrusion`，以两份互盲模型复核、唯一目标与保守凸路线多边形补 Japan 排除后的第六正例；未读取或占用 R1.3。
- 结果：333s/336s 在 `.01/.02/.03` 三档均 robust inside，边界距离 `28.45/61.34px`；328s 固定为 non-gate uncertain，339s 仅作 robust-outside clear proxy。验证后 eligible seen positives 为 `5+1=6`，但 768/连续/soak/R1.3 权限仍为 false，须先物化 R1.2c v2 并重跑全六例 oracle。
- 验证：focused `3 tests OK`，完整 crosscam `25 tests OK`；收据 SHA-256 `94154e091bec1e80cb1accc15fe20de0472c90db96619c1948e75d98ae70d083`。详见 [seen positive 预注册](docs/research/ustrf-sc/USTRF_CROSSCAM_SEEN_POSITIVE_R12C_PREREG_2026-07-22.md)。

## 2026-07-21
### USTRF truth—路线几何一致性 R1.2c
- 时间：2026-07-21；执行者：violjjet。
- 范围：冻结六个正事件的独立 alertable-anchor oracle、冲突仲裁与 London-only FP16-768 GPU 单变量协议；完整事件门先于 600 秒 soak，R1.3 v2 保留 12 个未打开槽位并把双 VLM 分歧或 truth/geometry 冲突转人工仲裁。未改 App、默认模型、阈值、bbox/contact、旧 polygon 或 tracker。
- 结果：oracle `5/6` 一致；Japan 在 `10000/12000ms` 两个 alertable anchors 均为 robust outside，固定标为 `truth_geometry_conflict`。因此 768 执行、连续重放、soak 与 R1.3 解封权限全部为 false；禁止移动旧 polygon 回救历史结果。
- 模型仲裁：按项目约定由两个 fresh-context 模型独立复核、第三模型仲裁，不等待真人。最终将 Japan 裁为 `event_truth_unknown / route_relation_inconsistent / EXCLUDE_FROM_SCORE`，不保留 positive，也不越权改成 strict negative；当前缺少第六个合格正事件。
- 验证：R1.2c focused Python 合同 `4 tests` 通过；hash-bound adjudicated oracle 无未决冲突，但 Japan 被排除后仅剩 `5` 个合格正事件。详见 [R1.2c 结果](docs/research/ustrf-sc/USTRF_CROSSCAM_TRUTH_GEOMETRY_R12C_RESULT_2026-07-21.md)。

### USTRF 跨相机移动端连续事件 R1.2b
- 时间：2026-07-21；执行者：Codex。
- 范围：只在 R1.2a 的 12 个 seen diagnostic 上冻结移动端 OFAT 候选顺序、分段延迟、取帧等价门与逐帧 SHA-256 传输；正式 App/core runtime 未改，R1.3 未解封，Vancouver 未用于候选选择，prompt/类别/`.05/.30/.45`/bbox/polygon 均未改变。
- 结果：首个候选“同 FP16-640 模型 + benchmark-only GPU delegate”即通过 canary，故未运行 320 候选。SM-S9280 600 秒、4,795 次检测为 inference p50/p95 `40/54ms`、总检测 `84/105ms`、0 解码/推理失败、温升 `4.0°C`、thermal status 最大 `0`，设备门通过。
- 事件门：正例 `4/6`，Japan 已持续关联但与冻结 route polygon 的事件关系不相容，London 22 帧关联为 0；负例假告警、重复交付、共现接管、身份切换均为 0，出画门随 London fail-closed。总体失败并保持 `do_not_replace_default_model`。
- 验证：R1.2b Python 合同测试；`:device-benchmark:assembleDebug`；精确帧设备准入 `OK (1 test)`；完整连续 instrumentation 跑满 600 秒并按预期在写出收据后因事件门失败。详见 [R1.2b 结果](docs/research/ustrf-sc/USTRF_CROSSCAM_MOBILE_R12B_RESULT_2026-07-21.md)。

### GPT/Codex 端到端自主工作流 authority 收口
- 时间：2026-07-21；执行者：violjjet。
- 范围：来源发现/获取、采集编排、标注、评测集、P3 准入、连续事件参考、设备米制几何、实验验收与候选发布统一改为 GPT/Codex/自动 Agent 工作流；隔离双模型一致即共识，分歧由全新第三模型仲裁，缺证据或 abstain 仅让相关样本/分支隔离或失败关闭，不创建人工待办。
- 自动推进：完整事件模型共识可授权研究训练与冻结评测；生产模型替换仍需 benchmark、INT8、同机事件、Android 证据全部通过，并附独立发布模型复核收据。参与者同意、许可证、设备真实测量和签名凭据仍必须来自真实主体或设备，模型通过授权自动化获取/审计收据但不得伪造。
- 治理：新增 `configs/ai_review_workflows_v1.json`、`docs/AI_REVIEW_GOVERNANCE.md` 与哈希绑定 receipt validator；扩展项目结构门，递归扫描现行 scripts/configs/docs，拒绝重新引入 human/manual authority，并增加正反 smoke test。
- 验证：聚焦 Python/JVM 回归、项目结构门及其 smoke tests、文档索引、仓库卫生、统一 research contracts 与 diff check；最终结果以本次实际复跑为准。

### Stacked PR CI bootstrap fixes
- 时间：2026-07-21；执行者：violjjet。
- 范围：仓库卫生门对 PR base 中已存在、当前仅删除的历史二进制视为清理，同时仍拒绝新增二进制；release signing 只在显式请求 `assembleRelease`/`bundleRelease` 时要求本地 keystore，不再被 `mergeReleaseAssets` 的任务图误触发。
- Android lint：Camera2 interop 的 camera ID 读取在窄函数边界显式 opt-in，不通过关闭 `UnsafeOptInUsageError` 或 lint baseline 掩盖错误。
- 验证：hygiene smoke 含 deleted-only/added-from-base 反例；`master...HEAD` hygiene 通过；CI debug/merge-assets 任务不需要 release keystore，显式 release 打包仍 fail closed。

### USTRF 跨相机连续事件 R1.2a 与 R1.3 预注册
- 时间：2026-07-21；执行者：violjjet。
- 范围：将已解封 R1.1/R1.2 降级为 12 段 5–15 秒 seen diagnostic；新增 benchmark-only 冻结 anchor 双向关联、一次交付/重复抑制、出画 fail-closed、共现隔离、设备延迟/soak/thermal 收据。App、core runtime、默认模型与反馈路径未改变。
- 结果：SM-S9280 600 秒、648 次 inference 无 decode/inference failure，温升 `5.1°C`、thermal status 最大 `0`；但正例事件仅 `4/6`，出画证据不完整，inference p50/p95 `762/978ms` 远超 `120ms` 门，事件门和设备门均失败，维持 `do_not_replace_default_model`。
- R1.3：只冻结 12 个未打开来源槽位（6 正/6 负）与双 VLM 独立复核 provisional event truth；未发现、下载、解码或消耗新 held-out。Vancouver 仅作漏检线索，不回调 prompt、`.05/.30`、bbox、polygon 或门槛。
- 验证：Python R1.2a/R1.3 合同 2 tests；`:device-benchmark:assembleDebug`；SM-S9280 association `OK (2 tests)`；完整 instrumentation 按预期以 frozen gate failure 结束并先写出可审计 JSON。详见 [R1.2a 结果](docs/research/ustrf-sc/USTRF_CROSSCAM_CONTINUOUS_R12A_RESULT_2026-07-21.md)。

本文件是 BlindAssist 的追加式工程历史：记录有代码、配置、模型、测试或已采纳技术决定的任务。近期条目应简洁写明范围、验证、风险并链接证据；长篇实验结论应写入对应 `docs/` 页面，当前状态以 [docs/SANPO_CURRENT_STATUS.md](docs/SANPO_CURRENT_STATUS.md) 为准。

## 使用与历史保留
- 历史条目全部保留，不能因重整索引而删除或重写结论。
- 月度历史归档：[2026-05](docs/history/development-log/2026-05.md) · [2026-06](docs/history/development-log/2026-06.md) · [2026-07](docs/history/development-log/2026-07.md)。根文件只保留最近 2–4 周；归档仅迁移原文，不改变当时结论。
- 新条目使用 `## YYYY-MM-DD` 和其下的 `### 任务标题`；追加在当前日期块，不为追求排序而搬动旧条目。
- 早期恢复式记录存在非单调日期与混合标题层级，以下原文作为历史证据保留；按日期或任务标题检索，不将其目录顺序当作当前优先级。
- 发布事实写入 `CHANGELOG.md`，待决方向写入 `idea.md`，当前协议/状态写入 `docs/`，具体规则见 [docs/DOCUMENT_GOVERNANCE.md](docs/DOCUMENT_GOVERNANCE.md)。

## 2026-07-21

### CI 结构门基线语义修复
- 时间：2026-07-21；执行者：violjjet。
- 范围：结构门只在 base 已含同一 policy 时检查新增稳定根 Interface，避免把门禁引入前的历史脚本追溯判错；`test_*` 与 `README.md` 的索引豁免由 `scripts/policy/project_structure.json` 单点声明，根 allowlist 仍保持精确失败门。
- 边界：不向脚本索引灌入历史测试名，不放宽根目录清单、研究 Module、跨域 import 或开发日志预算；不改 App、模型、研究结论或生产权限。
- 验证：结构门新增 bootstrap/test-exemption 反例；structure smoke 13 场景通过，`origin/master` 与 `eea9ea3` 两种 CI base-ref 的 repo hygiene 均通过。

### USTRF 真实事件与同设备米制几何 evidence pivot
- 时间：2026-07-21；执行者：violjjet。
- 范围：冻结新的 detector/teacher/dense/public/synthetic 实验轮次；物化首个 `route_obstacle` matched-pair 真实采集执行目录与 SM-S9280 红灯几何包。真实媒体、双审和标定尚未采集，authority 全部保持 false。
- 几何门：五类 evidence artifact 必须解析为 typed JSON，精确绑定设备/mount/calibration 与顶层 metrics，并继续哈希绑定 raw/gate source；`blocked/in_progress` 包也校验已有收据。空 `{name: ...}` artifact 与汇总漂移不再可能通过。
- 现状：手机已通过 ADB 确认为 `R5CX10M8Y8X / SM-S9280 / Android 16 API 36`；红灯 blocker 为 `BLOCKED_ON_SOURCE_ALIGNED_METRIC_DEPTH_AND_INTER_FRAME_STABLE_POSE`，绑定 r3 `1/861`、r5 `0/843` 与 `EPHEMERAL_PER_FRAME`。不重复同一 ARCore 窗口碰运气。
- 验证：geometry validator 6 tests 与统一 dependency-free research contracts 通过；blocked bundle 审计成功且 admission/shadow=false；空真实 pilot manifest exit 2 且未生成报告。

### USTRF P0 时间、风险场、栅格、路线与 dense 第三臂合同硬化
- 时间：2026-07-21；执行者：violjjet。
- 生产时间链：CameraX `ImageProxy.imageInfo.timestamp` 现以带 clock-domain/source/frame identity 的 `FrameStamp` 贯穿 VisionFrame、detector、Assist evaluation 与 session trace；采集时钟用于趋势，decision/effect 时钟独立，处理延迟变化不再改变 approach/recede。新增 `feature:assist -> core:ustrf` write-only shadow adapter；未获米制 geometry/pose/route 时只记录 fail-closed/abstain，结果不进入 UI、语音或震动。
- USTRF 内核：过期 risk cell 会移除，新鲜覆盖可从 unknown 恢复 known；新增共享 `UstrfGridSpec`，投影、运动、风险场、包络规划与结构化输出强制同 spec，五候选 profile 扩为 ±3 cell 并拒绝包络越界，corridor width 来自 body profile。连续 `RouteFieldReceipt` 已能进入同一 shadow session 候选规划并保留 intrusion evidence，但仍是直线候选，不宣称曲率轨迹完成。
- U0/dense：runner 改为执行并复验 bundle 内哈希副本，消除 implementation/threshold TOCTOU；dense field v2 使用 `uint32-le / 1e6` 固定点，source 与 route-interaction hash 分离，admission 从序列化 cell 重算摘要，不再信任自报 SHA/分数；LOSO artifact 移除运行耗时噪声并补模型、实现、样本 provenance。四个真实 dense/control adapter、真实 teacher 运行、人类 truth 与 device metric geometry 仍缺失，所有晋级/生产权限保持关闭。
- 仓库治理：repo hygiene 对 base-ref deleted-only 历史禁用产物放行，但新增同类产物仍拒绝；PR #1 仍需拆分，本轮未改写或推送远端。
- 验证：JDK 17 `:core:assist:test`、`:core:vision:testDebugUnitTest`、`:core:device:testDebugUnitTest`、`:core:ustrf:test`、`:feature:assist:testDebugUnitTest` 与 `:app:assembleDebug` 全绿；dense/runner/admission/LOSO 18 tests 在 USTRF teacher Python 环境全绿，LOSO 双跑 artifact SHA 一致；structure smoke/current gate、docs index、repo hygiene smoke/current/base-ref 与 diff check 通过。

### USTRF 研究合同持续验证与 current truth 收口
- 时间：2026-07-21 01:10:00 +08:00；执行者：violjjet。
- 范围：补充 `:core:ustrf:test` 与无设备 Python research-contract 的本地/CI 统一入口；统一 route-conditioned 主线在文档索引、SANPO 当前状态和计划表中的状态表述。
- 边界：只提升验证与文档 Locality，不改变 App、默认 YOLO、模型、设备行为或任何研究晋级结论；真实事件仍为 0，设备米制几何 admission 仍为 false。
- 验证：Python suite 24 tests 通过；JDK 17 `:core:ustrf:test`、文档索引、仓库卫生与目标差异检查通过。

### RC-OARF E0 事件门加固与 wrong-route 负控
- 时间：2026-07-21；执行者：violjjet。
- 范围：修复 route-conditioned event validator 的合同/来源/authority fail-open；为 route-risk seam 增加风险场新鲜度与独立 abstain reason；冻结并执行 r816 within-image wrong-route 负控。
- 结果：正确路线 BA `.91555`，两种错路线 BA `.72492/.79515`，3 个父来源均同方向下降；但旧 r816 report 缺逐预测 example ID，正式 gate fail-closed 为 `BLOCKED_ON_PREDICTION_IDENTITY_BINDING`。只保留 provisional 合成机制信号，不解除 r818、真实事件 0、设备或生产门。证据见 `artifacts.local/evidence/ustrf-sc/rc-oarf-route-specificity-control-v1-20260721-r3/report.json`。
- 验证：route validator 4 tests、legacy validator 6 tests、route-specificity 4 tests、JDK 17 `:core:ustrf:test`、空模板 fail-closed、docs index 与 diff check。

### scripts 研究模块下沉与开发日志月度归档
- 时间：2026-07-21；执行者：violjjet。
- 范围：把已冻结 public-video/public-silver campaign 的 315 个 CLI、合同、测试和 PowerShell Adapter 从 `scripts/` 根目录迁入 `scripts/research/public_video/`，新增稳定运行/测试 Interface；将 2026-05、2026-06 开发日志原文迁入月度历史档案。
- 边界：目录迁移不改研究算法、历史失败结论、模型、数据、App 或任何授权；当前 route-conditioned 主线不在旧 campaign 目录继续堆叠。
- 验证：以迁移后完整 campaign 测试、研究合同测试、文档索引、仓库卫生、路径引用审计和差异检查为准。

### 项目结构自动门禁
- 时间：2026-07-21；执行者：violjjet。
- 范围：新增统一结构检查与 smoke tests，并接入既有 repository hygiene 和 CI；冻结 `scripts/` 根文件 allowlist、开发日志行数/字节/28 天预算、研究 Module README 合同、内部脚本路径泄漏和跨 Module 私有 import。
- 架构：共享 RGB 脱敏 Implementation 下沉到 `research/common`，public-video 保留薄兼容 Adapter；真机闭环通过根 Adapter 调用，保持 Interface 稳定和实现 Locality。
- 边界：只改变仓库治理与脚本组织，不改研究算法、模型、数据、App、设备行为或晋级结论；结构预算不得用提额回避，应归档或深化 Module。
- 验证：以结构门 smoke、repo hygiene smoke、public-video/root Python 回归、research contracts、docs index、repo hygiene、PowerShell parse 和 diff check 为准。

### 项目结构门执行规则固化
- 时间：2026-07-21；执行者：violjjet。
- 范围：在 `AGENTS.md` 补充 research Module 放置规则、README 合同、跨域调用方式与统一结构门入口；具体根文件清单和预算仍由 `scripts/policy/` 单点维护。
- 边界：只固化协作指令，不改脚本 Implementation、研究算法、模型、数据、App、设备行为或晋级结论。
- 验证：项目结构检查、仓库卫生、文档索引和差异格式检查通过。

### RC-OARF E0 identity-bound 复跑与稳定门收口
- 时间：2026-07-21；执行者：violjjet。
- 范围：r816 输出增加唯一且保序的 example ID；用原 Python 环境和冻结参数复跑，并在 RC-OARF 收据中同时绑定旧 r816 全 evaluation parity、执行参数和 r818 稳定门。
- 结果：216 个 ID 与 route rows 逐项一致，新旧 r816 的 global/route/exact 预测、指标、fold 和系数 SHA 精确一致；路线特异性转为 `PASS_IDENTITY_BOUND_SYNTHETIC_ROUTE_SPECIFICITY`。r818 仍因 mean BA `.87737 < .90` 与 worst no-alert recall `.79710 < .80` 失败，组合决策 `BLOCKED_ON_R818_STABILITY`，不授权学生训练、设备或生产。
- 验证：r816 runner 9 tests、route-specificity 5 tests、dependency-free research contracts 24 tests、JDK 17 `:core:ustrf:test`、docs index、repo hygiene 与 diff check 通过；正式收据为 `artifacts.local/evidence/ustrf-sc/rc-oarf-route-specificity-control-identity-bound-v1-20260721-r4/report.json`。

### USTRF P0 生产与 benchmark shared decision parity
- 时间：2026-07-21；执行者：violjjet。
- 范围：抽取 Android-free `AssistDecisionKernel`，让生产 Coordinator 与 device benchmark 共用 temporal、stabilization、event、confirmation、feedback receipt 和 trace 顺序；不改风险阈值、默认 YOLO、UI 或生产 lifecycle gate。
- 契约：benchmark 报告升级为 v2，显式绑定 shared-kernel、STANDARD profile、manifest scenario、100ms 合成时钟与 planner adapter；保留旧 raw `model_risk` alias，新增 stable risk，并标记旧新聚合不可直接比较。device-event extractor 对旧 schema/旧 kernel/未知 adapter fail closed，且明确 planner 接受不等于物理设备投递。
- 回归：新增独立四帧 segmentation 黄金矩阵，锁定 `DISTANCE_TOO_FAR -> UNSTABLE -> TRIGGERED -> EVENT_ALREADY_ALERTED`，并覆盖 feedback unavailable 后事件不被消费、下一帧可重试；生产 wrapper 与 shared kernel 逐帧 raw/stable/event/feedback/trace 一致。
- 边界：本轮只关闭 P0 code/host parity，没有生成新的真机 benchmark 或物理反馈证据，不解除 r818、真实事件 0、设备米制几何、U0 或生产授权。
- 验证：extractor 3 tests 通过；JDK 17 `:core:assist:test`、`:core:device:testDebugUnitTest`、`:feature:assist:testDebugUnitTest`、`:app:testDebugUnitTest`、`:device-benchmark:compileDebugKotlin` 与 `:app:assembleDebug` 组合构建通过。首次 `--offline` 因本机缺 AndroidX test AAR 停在依赖解析，切换正常解析后同一任务组全绿；docs index、structure、repo hygiene 与 diff check 另行闭合。

### USTRF U0 teacher upper-bound 可执行门
- 时间：2026-07-21；执行者：violjjet。
- 范围：新增 U0 六臂预注册合同与 dependency-free evaluator；四个正式臂和 uniform/shuffled route 负控共用 frame ledger/shared decision kernel，并绑定 truth、实现、artifact、阈值、视频、route、frame IDs 与 trace SHA。不实现 teacher、不读取 blind、不生成标签或训练模型。
- 门禁：评价前重算完整 120 episode / 60 matched-pair route-conditioned 双人人类真值门；额外拒绝重复 episode/event ID、pair route samples 漂移、LOSO 错绑、critical fold 零分母、future/blind、漏臂/漏 episode 和 synthetic 授权。冻结逐 fold 事件硬门、route/control BA `.10` 增益、unknown-low-obstacle `.10`/2-session 增益与 causal lifecycle 不退化门。
- 结果：合成 fixture 3 tests 与统一 research-contract suite 27 tests 通过；当前正式空 template 的 CLI fail-closed 实跑 exit 2 且未写报告。状态 `U0_EVALUATOR_READY_BLOCKED_ON_TRUTH`，S0/student/Android/production 权限均未打开。

### USTRF U0 十集试采证据链与正式 truth 防绕过
- 时间：2026-07-21；执行者：violjjet。
- 范围：把 1 session × 5 scene × 1 matched pair 的 10-episode pilot 冻结为独立 contract/schema/scope；新增确定性空槽生成器、逐帧 video/clock/route 原子绑定、两份互不可见的人类 review 与独立 adjudication 校验，并把同一验证链接入正式 full-matrix truth。未采集媒体、未生成标签、未读取 blind/test。
- 合同修正：matched pair 不再要求两个独立拍摄复制逐像素 route trace，而是共享 `route_plan_id + provider policy + route choice`；每个 episode 的 current-camera 投影必须分别绑定自身 frame ledger、video、camera/calibration 和 projection receipt。U0 额外钉死官方 truth config SHA 与 route/base/frame/review validator bundle SHA，并逐 episode 绑定 source frame-ledger SHA。
- 结果：本地 ignored capture plan R2 精确生成 10 slot / 5 pair；空 pilot template CLI 以 exit 2 拒绝且不写报告；source receipt/episode 的 origin scope 能阻止仅改状态把 pilot 升格成正式 truth。官方 JSON 使用 canonical hash、validator 使用 LF-normalized text hash，CRLF/LF 回归通过；统一 dependency-free research-contract suite 7 files / 35 tests 全绿。已审计的 non-blind SANPO-Real 只有约 5 秒正式片段；相邻 `d3CK...` 素材也仅是待补收据、隐私、时钟、路线和人工双审的 raw candidate，不能冒充 pilot episode。
- 边界：pilot 审计输出的 route truth/U0/S0/training/Android/production 字段永远为 false；正式 eligible truth 仍为 0，状态保持 `U0_EVALUATOR_READY_BLOCKED_ON_TRUTH`。

### USTRF P0 真机 shared-kernel v2 与 U0 prediction evidence admission
- 时间：2026-07-21；执行者：violjjet。
- 真机：在 SM-S9280/API 36 上按历史同一 90 帧 SANPO v2 连续基准重跑 `SanpoTraversabilityOracle/current`；报告 schema v2、shared-kernel、STANDARD profile、100ms 序列时钟与 planner adapter 均已绑定。candidate total P95 `57.674ms`，event recall `1.0`、critical miss `0`、delivered repeat `0`、clearance `1.0`、false alerts/min `0`，49 次 duplicate attempt 被抑制；报告 SHA256 `6b2d39b36996613515a6988654c16d06d62c00ee023eda4cabf99563b96b4a25`。仍有 2 次 event ID regeneration，且 planner acceptance 不代表物理反馈送达。
- U0 加固：新增 prediction-bundle validator 并在 evaluator 指标计算前强制调用。六臂必须绑定真实本地 implementation/artifact/threshold/execution receipt 和逐帧 shared-kernel trace；逐帧核对 truth ledger、video/route/ledger hash、candidate adapter、kernel 顺序和 feedback receipt，提醒时间由 trace 重算，手写摘要不再是评价输入真源。
- 负控：占位/漂移文件 SHA、trace 单字节篡改、漏帧、手改提醒、adapter 漂移和 execution failure 均 fail closed；valid synthetic bundle 覆盖 6 arms / 12 episode traces / 252 frame traces，但正式 authority 仍强制 `u0_passed=false`。统一 dependency-free suite 为 8 files / 39 tests 全绿，空正式 truth CLI exit 2 且零报告。
- 边界：该设备数据是 historical benchmark-only，不是 U0 双审人类 truth；未实现或运行真实 teacher 六臂，不训练、不改 App/默认模型、不授权 S0/Android/production。

### USTRF U0 v2 unified candidate runner 与 LOSO/去标签执行合同
- 时间：2026-07-21；执行者：violjjet。
- 范围：新增稳定 `run_ustrf_sc_u0_candidate_bundle.py`，把 U0 从“可校验手工 bundle”推进为实际 subprocess-bound adapter 执行；prediction/evidence schema 升级 v2。修复 evaluator 对 adapter 字符串使用对象身份比较及重复 arm 可覆盖的问题。
- 执行边界：adapter 只收到不含 review、adjudication、`should_alert` 或事件标签的 sanitized inference manifest；U0 cadence 冻结为采集合同的 500ms exact grid。正式 backend 必须绑定 Android/Kotlin shared `AssistDecisionKernel`；synthetic process proof 使用独立 fixture backend，不冒充真机或模型证据。
- 实验设计：fixed baseline 声明 no-fit；拟合臂逐 held-out session 绑定 exact train-session/episode inventory、fold artifact 与 training receipt。uniform 由 runner 生成 constant full-frame field；shuffled 使用 session 内 sorted episode cyclic shift-one，control 禁止标签、seed 与 refit。truth route 与 adapter route input 分别记账。
- 兼容修正：trace 状态改为 kernel 原生 `APPROACHING/ALERTED/PASSED_OR_RECEDING/CLEARED`，feedback outcome 显式绑定 Kotlin reason；YOLO/bbox 保留生产现状的 optional event ID，dense 臂强制 kernel-native ID，禁止 writer 补造。
- 验证：统一 dependency-free suite 9 files / 44 tests 全绿；synthetic proof 实际执行 6 arms / 12 subprocess / 252 frames。LOSO held-out 泄漏、漏臂/重复臂、独立 JSON identity、非零退出、漏帧、标签注入、route/control 漂移、文件/registry/kernel/hash 漂移及 feedback/event 映射漂移均 fail closed。
- 边界：没有真实六臂 adapter、人类 full-matrix truth 或 device metric geometry；不训练、不改 App/默认模型/阈值，不授权 U0、S0、Android 或 production。

### USTRF U0 baseline Android adapter 与可审计真机 receipt
- 时间：2026-07-21；执行者：violjjet。
- 范围：新增 `baseline_yolo_geometry` 的稳定 host ADB adapter、device-benchmark instrumentation 与冻结配置。host 不生成 decision；Android 重算 request/manifest/video/ledger/artifact/config，枚举编码 sample PTS、解码 canonical RGBA8888、调用 shipped YOLO11n TFLite 和 shared `AssistDecisionKernel`，再生成最终 adapter JSON。
- 证据加固：Android receipt 绑定 device/build fingerprint、app/test APK SHA、模型/标签 SHA、host/device 源码 SHA、ledger、逐帧 requested/selected PTS、20ms 误差上限、压缩 video sample SHA、RGBA8888 内容 SHA 与 detector timing。runner/admission 对正式 Android backend 强制此 receipt，对 synthetic backend 反而拒绝伪 Android receipt。
- 真机：使用与设备现有安装包一致的 `.android-home` 调试证书，无清数据覆盖安装 app/benchmark。SM-S9280/API 36 对 3 帧公开视频完成 r2 两次 smoke，设备/APK/encoded-sample/RGBA/决策稳定字段一致；首次/repeat output SHA256 为 `592ad572...ef26d9d3` / `1f344b9f...f70d1b55`，receipt SHA256 `50fe0692...5bb086c7`。证据在 `artifacts.local/evidence/ustrf-u0-baseline-device-smoke-20260721-r2/`。
- 验证：统一 dependency-free suite 10 files / 48 tests、额外 device-event extractor 3 tests、JDK 17 `:core:assist:test`、`:core:ustrf:test`、benchmark compile/assemble 与 App assemble 通过；同签名 APK 安装与双次真机复跑通过。
- 边界：smoke 无 U0 人类事件真值，不证明安全精度或模型晋级；其他五臂、120-episode truth、r818 稳定性与 device metric geometry 仍未闭合，不改 App 运行时/默认模型/阈值。

### USTRF U0 detector bbox × explicit route 第二真实 Android adapter
- 时间：2026-07-21；执行者：violjjet。
- 范围：实现 `detector_bbox_explicit_route_adapter_v1` 的 host/device 全链与冻结配置。设备在每个 500ms truth-ledger frame 只选择当前或过去最新且仍有效的外部显式路线 sample，以相机底部中心连接 1/2/3 秒 waypoint、0.08 frame-width 半宽走廊和 bbox 底部 25% footprint 做二值 gate；保留 detection 原 bbox/置信度后送入同一个 `AssistDecisionKernel`。future/stale/低置信/invalid route 统一向 kernel 传空列表，禁止 intervention upgrade。
- 证据：新增 route-conditioning receipt，逐帧绑定 provider/projection、selected sample/waypoints、每个 source bbox/footprint、最短走廊距离与 keep；host 独立重算 sample 因果、footprint、距离和 keep，U0 runner/admission 另强制 Android bbox-route receipt 与 threshold/source hash 绑定。SM-S9280/API 36 公开视频负控保持 encoded sample、RGBA、app/test APK 和模型不变，仅把路线从中心改为左侧：同一 person bbox 从 `669.07px > 172.8px` 的排除/raw `NONE` 变为 `75.75px` 的保留/raw `MEDIUM`。左侧路线复跑的 backend、gate 与 decision 稳定字段一致；证据在 `artifacts.local/evidence/ustrf-u0-bbox-route-device-smoke-20260721-r1/`。
- 验证：route gate instrumentation 6 tests、统一 dependency-free suite 11 files / 52 tests、JDK 17 benchmark compile/assemble、同签名 APK 安装、三次完整 host→ADB→device→host 执行与 admission 重验通过。
- 边界：这是无人类事件真值的 public-video mechanism/pipeline smoke，不证明安全精度或 U0 通过；四个 dense/control 臂、120-episode truth 与 device metric geometry 仍未闭合，不训练、不改 App 运行时/默认模型/阈值。当前状态为 `U0_TWO_ANDROID_ADAPTERS_DEVICE_VERIFIED_BLOCKED_ON_HUMAN_TRUTH_AND_FOUR_REAL_ADAPTERS`。

### USTRF U0 dense risk-evidence seam 与最终内核真机重封存
- 时间：2026-07-21；执行者：violjjet。
- 范围：为 shared `AssistDecisionKernel` 新增 object-agnostic risk-evidence 输入，复用 temporal/stabilizer/event/confirmation/feedback；冻结 `UstrfU0DenseRiskEvidenceAdapter` 的 route-intrusion/local-peak 归一化，并把 kernel facade 加 7 个直接依赖文件纳入 U0 bundle hash inventory。未实现 teacher field generator、模型或第三臂。
- Fail-closed：拒绝 bbox、检测式 distance、预置 trend/event/feedback、矛盾 NONE 语义、越界/不一致分数、stale/current-frame 漂移和非单调时间。prediction admission 新增四个 dense/control 臂的 teacher 名称/版本/许可证/权重/实现、LOSO fold、route、逐帧 field SHA/evidence/unknown/归一化算术 receipt，缺项或篡改均拒绝。
- 执行修正：发现 baseline host 误用类级 instrumentation selector，导致同类 bbox test 被一并运行；改为方法级 selector并新增回归。最终 shared-kernel SHA `d28ea341...d7ac04d` 下重新同签名安装 APK，SM-S9280/API 36 完成 baseline r4 双跑和 bbox-route r3 中心/左侧/左侧复跑，五份输出全部通过 formal admission；证据分别在 `artifacts.local/evidence/ustrf-u0-baseline-device-smoke-20260721-r4/` 与 `artifacts.local/evidence/ustrf-u0-bbox-route-device-smoke-20260721-r3/`。
- 验证：dependency-free research contracts `11 files / 54 tests`；JDK 17 `:core:assist:test`、`:core:ustrf:test`、`:device-benchmark:compileDebugKotlin`、`:app:assembleDebug`；SM-S9280/API 36 dense seam `3/3`、route gate `6/6` instrumentation，五次 host→ADB→Android→host 与 admission 重验全绿。
- 边界：当前仅为两条真实 Android adapter 加 dense kernel seam，不是第三臂或 U0 PASS；正式人类 truth 仍为 0/120，teacher generator/LOSO artifact、四个 dense/control adapter 与 device metric geometry 仍缺。状态为 `U0_TWO_ANDROID_ADAPTERS_AND_DENSE_KERNEL_SEAM_DEVICE_VERIFIED_BLOCKED_ON_HUMAN_TRUTH_AND_FOUR_REAL_ADAPTERS`。

### USTRF U0 第三臂离线 dense teacher 前置原型
- 时间：2026-07-21；执行者：violjjet。
- 范围：在隔离的 `scripts/research/ustrf_sc` Module 新增 Apache-2.0 Depth Anything V2 Small ONNX teacher field 原型，并新增 label-free、fold-local 校准 artifact/receipt 稳定入口；输入合同拒绝 event/review/adjudication、blind、future 与 held-out 泄漏，所有输出保持 auxiliary-only 且 authority false。
- 验证：隔离 Python 3.11 venv 中 field 与 fitter 各 3 tests 通过，Python compile 与 diff check 通过；尚未运行真实 fold fit、Android field consumer、第三臂 device smoke 或 formal admission。
- 边界：审计发现现有 dense receipt 仍缺可从 fixed-point cells 重算的 field/route 证据，Android backend receipt 仍为 YOLO 专用，runner 另有复制后执行原文件的 TOCTOU。原型因此不得计为第三条真实 adapter；状态和 human truth `0/120` 均不变。

## 2026-07-11

### SANPO v3 训练前总门禁加固
- 时间：2026-07-12 23:45:00 +08:00
- 执行者：violjjet
- 类型：训练数据治理 / benchmark 隔离 / 自动化门禁 / 测试
- 修改范围：`scripts/sanpo_training_gate.py`、`scripts/train_export_sanpo_segmentation.py`、`scripts/validate_sanpo_v3_dataset.py`、`scripts/prepare_sanpo_v3_dataset_views.py`、训练脚本测试、README、CHANGELOG。
- 修改内容：将训练入口从任意 `--manifest` 收紧为唯一 `--dataset-root`，每次训练在 TensorFlow 导入前自动运行总门禁并写入 JSON + `.sha256` sidecar。门禁机器校验固定的 300 train/dev + 120 blind、六条 50 帧训练/dev 序列、两条各 60 帧且不同 session 的 blind 序列、四类 0..3 掩码、图片/掩码 SHA256、来源许可证与允许的隐私状态；policy 同时把恰好两条 blind session 锁为 `benchmark_only`，并显式禁止训练和阈值选择访问其路径或 session。报告不是 green 一律拒绝输出 MobileNetV3 + LR-ASPP 候选。
- 验证方式：临时 420 帧 fixture 的总门禁报告为 green，840 个图片/掩码哈希均匹配且 report sidecar SHA256 已生成；当前本机 dense annotation queue 缺 canonical training/blind manifests 与 policy，实际报告为 red，训练按预期不启动。
- 当前判断：这是本地 benchmark 训练治理加固，不改变 Android 运行时、默认 YOLO11n、风险阈值或 app 版本，故保持 `v10.9.0` / `versionCode=37`，不构建或归档 APK。后续必须先补齐受许可、自动门禁可验证的完整 v3 数据集，不能用人工覆盖或普通 manifest 绕过。

### v10.9.0 SANPO 风险事件闭环与边界形态否决
- 时间：2026-07-11 20:15:00 +08:00
- 执行者：violjjet
- 类型：核心风险策略 / 时序反馈闭环 / 分割候选否决 / benchmark schema / 测试 / 版本升级
- 修改范围：`RiskEventTracker`、`AssistSessionCoordinator`、`AssistEngine`、`TraversabilitySegmentationAnalyzer`、反馈本地化/相机解释、`DetectorAbDeviceBenchmarkTest`、SANPO finalize/review/事件阶段克隆脚本、JVM 测试、README、CHANGELOG 与 app 版本。
- 修改内容：新增纯 Kotlin 风险事件状态机，以分割来源、标签和中心走廊位置匹配同一事件；首次语音/震动实际触发后进入 `ALERTED` 并抑制重复反馈，连续 3 帧远离/缺失或离开中心走廊后清除。新反馈原因 `EVENT_ALREADY_ALERTED` 与 cooldown 分开呈现。分割 `generic obstacle` 对贴边长条区域执行边界否决，保留台阶和紧凑中心障碍。benchmark 改为按序列复用事件 tracker，并输出 event ID/state、反馈原因、已通过窗口及平行路沿统计；标注 finalize 支持 `expected_event_phase`，事件阶段 clone 工具从既有已复核的 approach/alert 标签生成新的不可变 90 帧 manifest。
- 修改原因：90 帧扩展集的 25.9% 错误提醒主要来自登阶后同一事件重复播报和平行路沿泛化；本轮先闭环风险规则与评测口径，不替换 YOLO11n，也不训练模型。
- 验证方式：提权运行 `:core:assist:test :device-benchmark:compileDebugKotlin --no-daemon --console=plain`，通过（105 个核心 JVM 测试，benchmark Kotlin 编译成功）；`:app:lintDebug :app:assembleDebug :device-benchmark:assembleDebug --no-daemon --console=plain` 通过。SM-S9280 上以事件阶段 clone 执行 90 帧 SANPO Oracle A/B：候选 P95 `57.581ms`、已通过窗口错误提醒 `0`、平行路沿报告错误提醒 `0`，但 alert FP `5.6%`、逐帧 alert recall `5.6%`，未达到门槛。90 秒 CameraX 回归安装与冷启动成功（`TotalTime=1081ms`），但等待 `检测中 | Detecting` 文本超时失败。
- 当前判断：版本从 `v10.4.0` / `36` 升至 `v10.9.0` / `37`，属于核心风险反馈行为的 `+0.5` 更新。事件闭环已证明能消除已通过台阶重复播报，但当前 benchmark 的逐帧 alert recall 口径应改为按事件计算；另外需消除 1 次 YOLO `person` 与 2 次 `generic obstacle` motion promotion 的负例提醒，再复跑设备回归。APK 已构建但未归档，因为固定集与 CameraX 回归均未通过；维持 `do_not_replace_default_model`，不得启动 MobileNetV3 + LR-ASPP INT8 训练。

### BlindAssist 项目综合评估与报告留存

- 时间：2026-07-10 03:35:32 +08:00
- 执行者：violjjet
- 类型：分析 / 架构审查 / 测试 / 工程化 / 无障碍 / 文档
- 修改范围：
  - `docs/PROJECT_AUDIT_2026-07-10.md`
  - `DEVELOPMENT_LOG.md`
- 修改内容：
  - 对当前 BlindAssist 工作树进行只读综合评估，覆盖多模块架构、CameraX/TFLite 运行链路、风险与反馈语义、session 生命周期、测试、Gradle、CI、仓库卫生、产品完成度、隐私权限、无障碍和文档一致性。
  - 将完整评估结论、验证数据、证据位置、架构深化候选和分阶段修复顺序保存到 `docs/PROJECT_AUDIT_2026-07-10.md`，避免评估只留在聊天或临时 HTML 中。
  - 项目总体判断为：作为毕设/课程技术原型完成度较高，但尚不能作为可依赖的助盲安全产品；最高优先级问题是“未检测到”被表达为“安全观察中”，以及旧帧可能越过关闭动作进入新 session。
  - 记录仓库卫生脚本正则失效、完整 Gradle 合并任务图存在隐式依赖、仪器功能测试与 benchmark 未隔离、设备回归断言不足、文档版本漂移和局部无障碍缺口。
- 修改原因：
  - 用户要求将本次项目评估写入开发文档或保存报告，需要形成可在 Git 中审阅和后续追踪的耐久产物。
  - 项目协作规范要求即使只做分析，也应记录评估结论、验证证据和后续事项。
- 验证方式：
  - 运行 `.\.venv-export312\Scripts\python.exe scripts\inspect_tflite.py`，模型检查通过：输入 `[1, 320, 320, 3] float32`，输出 `[1, 84, 2100] float32`。
  - 使用仓库本地 JDK/SDK/Gradle 缓存并加 `--rerun-tasks` 强制重跑核心 JVM 测试，共 189 tests，0 failures，0 errors，0 skipped。
  - 单独运行 `:app:lintDebug :core:vision:lintDebug :core:device:lintDebug :core:ui:lintDebug :feature:assist:lintDebug`，Lint 矩阵通过。
  - 单独运行 `:app:assembleDebug :app:assembleDebugAndroidTest`，Debug APK 与 AndroidTest APK 构建通过，大小分别为 47,288,840 bytes 和 76,238,831 bytes。
  - 将 JVM 测试、Lint、Debug APK 和 AndroidTest APK 合并到一次 Gradle 调用时失败；Gradle 报告 `prepareYolo26nBenchmarkAssets`、`prepareDepthBenchmarkAssets` 与 `generateDebugAndroidTestLintModel` 之间存在未声明的隐式依赖，已按真实问题写入报告。
  - 运行仓库卫生脚本时表面输出通过，但代码审查和正则实测确认多个 `\\.` 表达式无法匹配普通扩展名路径，因此没有把该结果视为可靠门禁。
  - 运行 `adb devices` 未发现连接设备，本次未执行真机回归或 connected instrumentation；报告已明确这一验证边界。
  - 生成后检查报告文件存在、Markdown 标题结构完整，并核对其中引用的仓库文件路径。
- 版本判断：
  - 本次仅新增评估报告并追加开发日志，不修改应用功能、Gradle 配置、Manifest、资源、权限、模型资产或 APK 行为，因此不调整版本号，不归档新 APK。
  - 项目版本保持 `v8.9.0` / `versionCode=33`。
- 后续事项：
  - 第一优先级修复安全措辞和运行时 session 生命周期，再修复卫生脚本、Gradle 任务依赖和测试隔离。
  - 后续应同步 README/DEMO_GUIDE 与真实 UI，并补充无障碍修复和真实连续场景验证。
  - 实施核心运行时修复后，需要重新执行完整 JVM、Lint、AndroidTest 编译、真机回归和 APK 归档流程。

## 2026-07-10

### v9.4.0 安全语义、Session 生命周期与测试架构修复

- 修改文件：
  - 风险与反馈：`core/assist`、`core/device`、`core/ui` 对应实现和 JVM 测试。
  - 运行时：`feature/assist` 的 lifecycle gate、frame processor、state machine、renderer、controller 及测试；`core/device` 的 CameraX source-generation 校验。
  - 测试架构：新增 `device-benchmark/`，迁移 Detector A/B benchmark，删除旧 `Yolo26nDeviceBenchmarkTest`，收窄 `app/src/androidTest`。
  - 工程与文档：Gradle/CI、卫生脚本与 smoke、`.gitignore`、README、CHANGELOG、DEMO_GUIDE、benchmark/APK/审计文档。
- 修改内容：
  - 新增 `RiskEvidenceState`。`RiskLevel.NONE` 只表示未达到提醒等级；空帧、不支持类别和低置信度为没有支持目标证据，远处支持目标为已有证据但未达阈值。中英文运行时、无障碍、预览和测试移除“安全观察中、无风险、Safe、No risk detected、Ahead is stable”等承诺性表达。
  - `AssistRuntimeLifecycleGate` 改为单调 `SessionToken` 和 `commitIfCurrent()`。detector 保持锁外阻塞执行，但统计、coordinator、反馈、UI render 和错误提交必须通过 token；停止先失效旧 token，shutdown 等最后 lease 完成后再关闭 CameraX executor、detector 与 feedback。
  - 独立审查发现 CameraX provider callback 在 generation 检查与 `cameraProvider/started/onStarted` 提交之间仍可被 stop/start 插入；最终将 generation 校验、provider 绑定、started 写入和回调收敛到同一 `lifecycleLock` 临界区，并对错误回调做同样的原子代际校验。
  - `ResetSession` 收敛为 `StopSession`；`StopCamera` 只停止 CameraX。`AssistFrameProcessor.resetSessionStats()` 不再篡改 `isProcessing`；renderer 使用当前 `AssistFrameResult.sessionSummary`。
  - `FeedbackGateway.resetSession()` 改为必实现接口；新 session 清空 feedback cooldown 与 fatigue，新增首条提醒不受旧 session 冷却影响的测试。
  - 修复仓库卫生扩展名正则，增加 `.android-home/`、`.kotlin-home/`、`**/__pycache__/` 和 `work/` 规则；新增无 Pester 的临时 Git 仓库 smoke，覆盖 13 个允许/拒绝场景。
  - 新增 `:device-benchmark` `com.android.test` 模块，目标 `:app` debug；资产准备任务显式依赖 merge consumer。功能 AndroidTest 只保留 11 个 Compose 测试，不再携带 TFLite benchmark 依赖和大资产。
  - 版本升级为 `versionName=9.4.0` / `versionCode=34`。
- 验证方式：
  - 使用仓库 JDK 17 完整运行 31 个 JVM test suite，共 `198 tests`，`0 failures / 0 errors / 0 skipped`。
  - `scripts/test_repo_hygiene.ps1` 的 13 个 smoke 场景全部通过；对当前真实变更运行 `scripts/check_repo_hygiene.ps1` 通过。
  - 一次 Gradle invocation 同时执行全部 JVM、App/library Lint、`:app:assembleDebug`、`:app:assembleDebugAndroidTest` 和 `:device-benchmark:assembleDebug`，`319 actionable tasks`，`BUILD SUCCESSFUL`。AGP 8.7.3 的 `com.android.test` 实际不提供 `lintDebug` 或 lint-model task，因此 benchmark 模块以编译和资产任务图作为本地覆盖，不虚构 lint 结果。
  - 功能 AndroidTest APK 为 `1,010,446` bytes，不含 yolo26、深度模型、COCO100 或 BlindAssist EvalSet；最终 device-benchmark APK 为 `75,594,654` bytes，包含本机存在的 yolo11/yolo26、COCO100 和 BlindAssist EvalSet。本机默认 Depth Anything TFLite 不存在，因此本轮 benchmark APK 不包含该深度模型。
  - `scripts/inspect_tflite.py` 通过：默认 yolo11 输入 `[1,320,320,3] float32`，输出 `[1,84,2100] float32`。
  - 独立审查修复后重跑 `:core:device:testDebugUnitTest :core:device:lintDebug :app:assembleDebug :device-benchmark:assembleDebug`，通过。
  - 最终 `scripts/verify_release_apk.ps1` 通过：包名 `com.linnan.blindassist`，`versionName=9.4.0`，`versionCode=34`，debug APK 大小 `47,297,084` bytes，SHA256 `E4DB467B77F9628F04E4E2CF00AC8737C5FABE95ED60AC6EF6A8ED1518E067BC`。
- APK 归档：
  - 最终完整本地归档：`E:\linnan\blind-assist-apk-archive\apks\BlindAssist-v9.4.0-debug-20260710-084153.apk`，并更新本地 `APK_ARCHIVE_MANIFEST.csv`。独立审查前的本地中间构建继续作为构建历史保留，但不作为最终交付。
  - 最终 Git 里程碑：`releases/apk/BlindAssist-v9.4.0-debug-20260710-084153.apk`；相对最新 Git 里程碑 v8.8.0 的版本差值为 0.6，符合归档规则。
- 验证边界：
  - 当前无设备，本轮没有运行 `:app:connectedDebugAndroidTest` 的 11 个 Compose 功能测试、`:device-benchmark:connectedDebugAndroidTest` 的 Detector A/B 或 Depth-fusion，也没有执行 90 秒真机回归。
  - 当前结论是“本地验证完成、真机验证待执行”；benchmark APK 构建成功不描述为 benchmark 通过。
- 保留内容：
  - 未回滚或清理本轮开始前已有的 `AGENTS.md`、`DEVELOPMENT_LOG.md`、`idea.md` 修改及本地实验目录；本轮未提交、未推送。

### v9.4.0 真机闭环、Benchmark 修复与可信回归

- 设备：Samsung `SM-S9280`，Android 16 / API 36，序列号 `R5CX10M8Y8X`。
- 修改内容：
  - `:device-benchmark` 显式对齐 `androidx.lifecycle:lifecycle-common` 到项目 `2.8.7`，修复测试 APK 中 Lifecycle `2.3.1` 覆盖目标 App 类后触发的 `Lifecycle.Event.Companion` `NoSuchFieldError`。
  - `run_detector_ab_device_benchmark.ps1` 与 Depth 脚本统一使用仓库 `.android-home`，避免两条测试链使用不同 debug keystore 导致 `INSTALL_FAILED_UPDATE_INCOMPATIBLE`。
  - `run_device_regression.ps1` 从被动采集升级为可信相机回归：语义关闭 Android 16 兼容提示、跳过 onboarding、预授权 CAMERA、进入手机摄像头，并断言 `检测中`、前台 resumed Activity、模型就绪性能帧和无 Crash/ANR；脚本保存为 UTF-8 BOM，确保 Windows PowerShell 正确解析中文 UI 文本。
- 真机验证：
  - `:app:connectedDebugAndroidTest`：`11/11` tests passed，`0 failed / 0 skipped`。
  - Detector A/B：BlindAssist EvalSet 100 图完整运行。YOLO11n / YOLO26n total P50 为 `53/48ms`，centerRiskRecall 为 `0.688/0.667`，alertFalsePositiveRate 为 `0.037/0.074`，criticalMissCount 为 `9/10`；结论为保留 YOLO11n。证据：`test-artifacts.local/detector-ab-device-benchmark/20260710-165143`。
  - MiDaS Depth-fusion：baseline / candidate total P50 为 `54/277ms`，centerRiskRecall 为 `0.688/0.667`，alertFalsePositiveRate 为 `0.037/0.148`，distanceBandAccuracy 为 `0.73/0.70`，criticalMissCount 为 `9/7`；漏报改善不足以抵消误报、准确率与延迟退化，候选不晋级。证据：`test-artifacts.local/depth-fusion-benchmark/20260710-170003`。
  - 最终增强版 90 秒回归通过，证据：`test-artifacts.local/device-regression/20260710-172506`。脚本在采样前后分别断言相机 UI、前台 resumed Activity 和性能帧增长；最终记录到 `101` 条性能帧，session 达 `1分44秒`，末帧约 `13.2 FPS`、推理 P50/P95 `42/46ms`，PSS 约 `264–284MB`，最终累计 jank `1.37%`，无目标进程 Java/native crash、ANR 或异常死亡，最终 UI 仍为 `检测中`。
- 发现但未在本轮解决：
  - Android 16 调试兼容提示指出 `libtensorflowlite_jni.so`、`libimage_processing_util_jni.so`、`libtensorflowlite_gpu_jni.so` 和 `libandroidx.graphics.path.so` 未全部满足 16KB page-size 对齐。当前 debug 运行与测试通过，但后续依赖升级或发布准备必须处理。
- 版本判断：
  - 本轮修复测试基础设施并完成既有 v9.4.0 真机验证，不改变生产运行逻辑、模型资产、权限集合或用户可见功能，因此保持 `versionName=9.4.0` / `versionCode=34`，不新增 APK 归档。

### v9.9.0 16KB 兼容、离线回放与眼镜设备模拟中心

- 时间：2026-07-10 23:39 +08:00
- 执行者：violjjet
- 类型：Android 实现 / 兼容性 / 无障碍 / 测试 / 发布归档
- 版本：`versionName=9.9.0` / `versionCode=35`
- 修改内容：
  - 将 `org.tensorflow:*:2.16.1` 迁移到 LiteRT `1.4.2` core/GPU/GPU API，保留现有 Interpreter、模型和 CPU fallback；新增 `scripts/verify_apk_16kb.ps1`，并把 APK/AAB ELF、zipalign、bundletool 和旧依赖排除接入发布检查与 CI。
  - 新增 `AssistInputSource`、`ReplayScenario`、`ReplayFrameSource` 和独立 RGBA 帧；debug 四种 COCO 素材以 2 FPS 进入真实 detector、risk、feedback、overlay 和 session summary 链路。stop/shutdown 使用 generation 隔离，旧会话不得提交帧、错误或 started 回调。
  - 用全屏 `GlassesSimulatorScreen` 替换眼镜占位弹窗，支持 800ms 模拟连接、82% 电量、15% 低电量、模拟断连与重置；debug 连接态可选择离线回放，release 只显示模拟状态与链路，不包含 replay 入口或资产。
  - CameraX 继续要求非空 `PreviewView` 和 CAMERA 权限；Replay 不依赖 CameraX、CAMERA、存储或网络权限。运行期间切换来源会先完整停止旧 session，再创建新 source/token。
  - 中英文可见文本、按钮与 TalkBack 说明均明确使用“模拟”，并明确未扫描蓝牙、未联网、未连接真实眼镜；状态只保存在 ViewModel 生命周期内。
- LiteRT GPU 回归与处置：
  - 同设备 BlindAssist EvalSet 100 图连续两次复跑，LiteRT GPU 的关键漏报保持 `9`，但提醒误报率稳定从历史 `0.037` 增为 `0.074`，不满足发布门禁。
  - 按计划允许的回退条款，保留 GPU delegate 代码和依赖，但默认启用 LiteRT CPU 兼容模式。CPU 结果与历史基线对齐：AP50 `0.289`、precision/recall `0.826/0.300`、关键漏报 `9`、提醒误报率 `0.037`，total P50/P95 `53/55ms`；历史 P95 为 `61ms`，无超过 10% 的退化。证据：`test-artifacts.local/detector-ab-device-benchmark/20260710-232929`。
- 验证方式：
  - 最终组合 Gradle 矩阵一次通过：全部 `testDebugUnitTest`、`lintDebug`、`:app:assembleDebug`、`:app:assembleDebugAndroidTest`、`:app:bundleDebug`、`:app:mergeReleaseAssets`、`:device-benchmark:assembleDebug`；`335 actionable tasks`，`BUILD SUCCESSFUL`。
  - Compose 真机测试 `12/12` 通过，覆盖模拟中心中英文、返回、按钮状态与 debug-only replay 入口。
  - `scripts/inspect_tflite.py` 通过：LiteRT backend，输入 `[1,320,320,3] float32`、输出 `[1,84,2100] float32`。
  - 仓库卫生 smoke、真实工作树卫生与 `git diff --check` 通过；release merged assets 只有 labels、README 和 YOLO11 模型，不含 replay。
  - 最终 APK 通过 `verify_release_apk.ps1`：包名 `com.linnan.blindassist`，版本 `35/9.9.0`，大小 `55,879,856` bytes，SHA256 `53065A54A43ABF6256994CDC9E6C89F2F5680BE5BEA1FC9BDF88CAF26AA77BDD`。
  - 最终 APK 与 debug AAB 的 16 个 native library 全部满足每个 `PT_LOAD p_align >= 16384`；APK `zipalign -P 16` 通过，AAB bundletool 输出 `PAGE_ALIGNMENT_16K`。
  - Samsung `SM-S9280` / Android 16 最终 90 秒 CameraX 回归通过，证据：`test-artifacts.local/device-regression/20260710-233619`。最终约 15 FPS，CPU 推理 P50/P95 约 `31/33ms`，无目标进程 Crash/ANR，关闭重开与性能帧增长断言通过。
- APK 归档：
  - 本地完整归档：`E:\linnan\blind-assist-apk-archive\apks\BlindAssist-v9.9.0-debug-20260710-233951.apk`。
  - Git 里程碑：`releases/apk/BlindAssist-v9.9.0-debug-20260710-233951.apk`。
  - SHA256：`53065A54A43ABF6256994CDC9E6C89F2F5680BE5BEA1FC9BDF88CAF26AA77BDD`。
- 验证边界与后续：
  - 当前三星设备为 4KB page-size，本轮完成的是 16KB 静态发布门禁，尚未在真实 16KB Android 15/16 环境验证安装、冷启动和连续推理。
  - `keystore.properties` 指向的本地 release keystore 不存在，因此没有生成签名 release APK/AAB；debug 里程碑按既有归档规则交付。
  - 实际场景采集、算法调参、模型替换和真实眼镜 BLE/USB/视频/反馈接入继续延期；模拟中心不得表述为真实硬件能力。

## 2026-07-11

### SANPO-Real 连续真实场景证据首批接入

- 时间：2026-07-11 00:39 +08:00
- 执行者：Codex
- 类型：公开数据调研 / 连续序列导入 / 人工复核门禁 / 文档；不改变生产 App、模型、权限或版本号
- 修改范围：
  - `scripts/build_sanpo_sequence_evalset.py`
  - `scripts/finalize_sanpo_sequence_evalset.py`
  - `scripts/test_build_sanpo_sequence_evalset.py`
  - `docs/SANPO_SEQUENCE_EVALSET.md`
  - `docs/BLINDASSIST_EVALSET.md`
  - `README.md`
  - `idea.md`
  - `DEVELOPMENT_LOG.md`
- 资源核验：
  - 原计划优先检查 PEDESTRIAN；Zenodo 官方 API 对 `10.5281/zenodo.10907945` 返回未注册，GitHub `CYENS/PEDESTRIAN` 也不存在，无法核验许可证、文件清单和哈希，因此没有接入或下载。
  - 改用 SANPO-Real 官方公开存储桶。数据集页面与官方仓库确认数据许可证为 CC BY 4.0，可按 session 选择性下载；代码仓库许可证与数据许可证分开记录。
- 实现内容：
  - 导入脚本从 GCS API 枚举指定 session 的 RGB 与 segmentation 对齐帧，读取 session FPS、相机参数、场景属性、labelmap 和逐帧人工/机器标注类型。
  - 15 FPS 源序列以最近帧方式重采样为 10 FPS，目标 `frame_index` 从 0 连续编号，另外保留源帧号与源/目标时间戳，匹配现有 benchmark 的 100ms 时间步。
  - 解析 SANPO RGB 分割掩码：红通道为类别 ID，后两通道为实例 ID；保留全部 `source_regions`，只对 `pedestrian -> person` 和 `traffic light -> traffic light` 做无歧义 COCO 映射，拒绝把通用 `vehicle/animal/traffic sign/obstacle` 伪造为具体 COCO 类。
  - 下载结果只生成 `manifest.draft.jsonl`。所有 BlindAssist 风险、提醒和逼近字段保持 null，状态为 `pending_review`；`finalize_sanpo_sequence_evalset.py` 只有在每行 CSV 明确 `accepted_manual_review` 且全部枚举、布尔值、图片与连续 frame index 合法时才生成 canonical `manifest.jsonl`。
  - 每行保留官方 URL、GCS object name、RGB/掩码官方 MD5、派生 SHA256、许可证、再分发策略和隐私复核提示；下载后强制比对官方 MD5，并校验 RGB、掩码和 session description 尺寸一致；原始帧、掩码和 QA 全部位于 Git 忽略目录。
- 首批本地候选集：
  - 路径：`test-artifacts.local/datasets/blindassist-sanpo-pilot-20260711`。
  - session：`-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG`，`camera_chest/left`，城市道路交叉口、晴天、高可见度、步行、中等障碍和较高车辆流量。
  - 30 帧，覆盖 3 秒，2208×1242；源帧 15 FPS，目标 10 FPS。RGB `110,536,034` bytes，掩码 `698,298` bytes，30 个唯一 SHA256。
  - 官方 SANPO split 为 `train`；8 帧为 `HUMAN_ANNOTATED`、22 帧为 `MACHINE_ANNOTATED`。只有 2 个同时具备人工分割和无歧义 COCO 映射的帧进入 detection GT `objects`；其余映射只保留为 `source_mapped_objects` 候选。
  - 分割区域帧计数包括 vehicle 80、tree 60、obstacle 47、curb 29、pedestrian 11；这些是来源候选区域，不是 BlindAssist 风险真值。
- 验证：
  - `.venv-export312\Scripts\python.exe scripts\test_build_sanpo_sequence_evalset.py`：11 tests passed，覆盖 15→10 FPS 无重复采样、拒绝上采样、SANPO 类别保留/精确映射、未复核风险字段拒绝、finalize 正负路径、非 COCO primary 隔离、阻断 issue tag、GCS MD5 损坏、finalize 二次 SHA256 校验和机器标注 detection GT 拒绝。
  - 首批下载脚本完成，`qa/manifest_validation.json`：`ok=true`、`image_count=30`、`unique_hashes=30`、`pending_review_count=30`、`benchmark_ready=false`。
  - CLI 负向验证通过：`--target-fps 5` 在网络访问前被拒绝，`--lens right` 被 argparse 拒绝；当前导入只允许与 benchmark 100ms 时间步一致的 10 FPS 左目序列。
  - 抽查首、中、末三张 boxed QA 图，连续画面、障碍/车辆/行人候选框和尺度变化可见；强光、阴影、脚手架、路桩、车辆与远处行人提供了比静态 COCO 更接近步行场景的复核素材。
- 后续：
  - 先完成 30 帧人工风险与 approach 复核，运行 finalize 并验证现有 benchmark；随后再选择 5–9 个不同环境 SANPO session 和本地受控自采场景。
  - canonical manifest 生成前不得描述为 benchmark 已通过；本轮不升级 `versionName=9.9.0` / `versionCode=35`，不生成或归档 APK。

### SANPO 首批序列 AI 多轮复核与正式清单

- 时间：2026-07-11 01:17 +08:00
- 执行者：Codex；复核身份明确记录为 `ai_assistant`，不冒充人工复核。
- 复核方式：两路逐帧独立视觉复核分别覆盖 0–14、15–29 帧，第三路复核全序列时序一致性，主流程再检查全部 30 张 boxed 图并裁决分歧。
- 共识标签：主风险区域 30/30 为 `sanpo_20_2`；距离均为 `MID`、风险均为 `LOW`、不触发即时或 approach 告警。方向为 CENTER 22 帧、LEFT 8 帧；时序为 UNKNOWN 1 帧、STABLE 4 帧、APPROACHING 25 帧。
- 分歧处理：时序复核曾建议把末段设为立即告警；综合障碍仍处中距离、画面保留绕行空间以及两路逐帧复核意见，最终保留不告警。强光、方向边界帧下调置信度，最终逐帧范围为 0.66–0.87。
- detection GT：第 24 帧 1 个人框、第 28 帧 2 个人框通过 AI 视觉复核；其他机器分割映射仍不进入 detection GT。
- 门禁增强：finalize 新增 `accepted_ai_review` 路径，但必须显式传入 `--allow-ai-review`，并要求 `reviewer_type=ai_assistant`、reviewer ID、置信度至少 0.65、至少两次独立复核；provenance 写入每行 manifest。人工路径保持兼容。
- 产物：`test-artifacts.local/datasets/blindassist-sanpo-pilot-20260711/manifest.jsonl`，30 行，SHA256 `879eea31021e68de1648f0d8818b0f8f30fea51d3fad9bf56cc4834cf32023e8`；finalize report `ok=true`、errors 为空。
- 验证：`scripts/test_build_sanpo_sequence_evalset.py` 13 tests passed；逐行语义断言、复核 provenance、方向分段、approach 分段、提醒字段和 detection GT 复核状态全部通过。
- 边界：这次完成的是工程数据复核与 benchmark 输入晋级，不是人工/目标用户安全验证，也尚未执行固定真机连续序列 benchmark。

### SANPO 真机 A/B benchmark 与解析器修正

- 时间：2026-07-11 01:36–01:46 +08:00
- 设备：Samsung `SM-S9280` / Android 16，序列 30 帧，`current` 风险配置，每图 3 次 App 链路运行。
- 首轮结果：A/B instrumentation `BUILD SUCCESSFUL`，随后默认 YOLO11n 90 秒 CameraX 回归 `status=passed`，证据目录为 `test-artifacts.local/detector-ab-device-benchmark/20260711-013632` 和 `test-artifacts.local/device-regression/20260711-013904`。
- 复核发现：benchmark 使用 `JSONObject.optString()` 读取 manifest 的 `primary_object_id: null` 时得到字符串 `"null"`，把无主 COCO 目标的样本错误纳入 `primaryObjectHitRate` 分母。修正为 null-aware `optionalString()`，同时用于 `sequence_id`、`scene_bucket` 和 approach 枚举解析；`BlindAssistExpectedRisk.primaryObjectId` 改为 nullable，JSON 输出保持真正的 null。
- 修正后重跑：证据目录 `test-artifacts.local/detector-ab-device-benchmark/20260711-014344`，instrumentation 与汇总均 `status=passed`；30 帧的 `primary_object_id` 已验证全部为 JSON null，不再是字符串。
- YOLO11n：AP50/precision/recall 均 0，FP/img `0.233`、FN/img `0.1`、错误提醒率 `0.033`、approach recall `0`；第 0 帧把远处内容误认成 person 并产生 RIGHT/NEAR/MEDIUM 提醒。total P50/P95 `60/68ms`。
- YOLO26n：AP50/precision/recall 均 0，FP/img `0.433`、FN/img `0.1`、错误提醒率 `0.033`、approach recall `0`；第 25 帧误认成 truck 并产生 RIGHT/NEAR/MEDIUM 提醒。total P50/P95 `47/48ms`。
- 两款模型都漏掉第 24 帧 1 个、第 28 帧 2 个 person GT；25 帧标注为 `APPROACHING` 的通用障碍均未被时序跟踪。YOLO26n 虽快，但误检更多且检测/风险质量无改善，推荐保持 `do_not_replace_default_model`。
- 决策：默认 App 继续使用 YOLO11n；下一算法实验优先考虑可通行区域、语义分割或深度几何候选，不继续只比较 COCO detector。该结论是单序列工程证据，不是安全保证。

### SANPO 可通行区域 oracle 基线、真机否决与远端授权

- 时间：2026-07-11 16:08 +08:00
- 执行者：violjjet
- 类型：算法候选 / benchmark / 真机验证 / 文档 / 协作规则
- 修改范围：
  - `core/assist/.../Detection.kt`
  - `core/assist/.../RiskAnalyzer.kt`
  - `core/assist/.../TraversabilitySegmentation.kt`
  - `core/assist/.../TraversabilitySegmentationAnalyzerTest.kt`
  - `device-benchmark/build.gradle.kts`
  - `device-benchmark/.../DetectorAbDeviceBenchmarkTest.kt`
  - `scripts/benchmark_sanpo_traversability.py`
  - `scripts/run_detector_ab_device_benchmark.ps1`
  - `docs/SANPO_TRAVERSABILITY_BASELINE.md`
  - `AGENTS.md`
- 实现与验证：
  - 新增 SANPO 三类通行区域、梯形走廊、四连通域和路沿/台阶/不可通行面/通用障碍/杆状物风险提取；使用 `DetectionSource.SEGMENTATION` 隔离分割证据，避免污染 COCO 检测指标。
  - 离线 30 帧 512×512 oracle：主风险区域覆盖 `26/30`（86.67%），平均 safe/not-safe/obstacle 覆盖率为 `71.22%/15.26%/13.53%`。
  - Samsung SM-S9280 真机同设备 A/B、每帧 3 次：候选错误提醒率从 `3.3%` 升至 `90.0%`，SANPO 主风险命中仅 `10.0%`，总延迟 P50/P95 从 `53/53ms` 增至 `92.404/96.678ms`；主要原因是第 1–26 帧持续把右侧 curb 升为 MEDIUM/NEAR。
  - 默认模型 90 秒 CameraX 回归通过；证据目录：`test-artifacts.local/detector-ab-device-benchmark/20260711-153543`、`test-artifacts.local/device-regression/20260711-153801`。
  - 结论保持 `do_not_replace_default_model`；后续应将 curb 降为边界证据，并要求中心侵入、深度突变或连续逼近后才允许提醒。
- Git 与协作授权：
  - 功能提交：`1fdcff4 feat: add SANPO traversability oracle baseline`。
  - 用户明确确认 `git@github.com:violetljj/blind-assist.git` 为其控制的可信远端，并授权后续常规推送不再重复询问外发可信性；授权边界已写入 `AGENTS.md`。
- 版本判断：
  - 当前实现仅存在于 benchmark/oracle 实验通道，真机门禁明确否决，未进入生产运行链路、未替换模型、未改变用户可见功能，因此保持 `versionName=9.9.0` / `versionCode=35`，不新增 APK 归档。

### SANPO Traversability v2 Oracle 第一阶段
- 时间：2026-07-11 16:50:00 +08:00
- 执行者：violjjet
- 类型：重构、性能、测试、真机验证、文档、版本
- 修改范围：`TraversabilitySegmentation.kt`、`RiskAnalyzer.kt`、`DetectorAbDeviceBenchmarkTest.kt`、离线 oracle 脚本、单元测试、README/CHANGELOG/idea/基线文档和版本配置。
- 修改内容与原因：
  - 将 curb 从普通障碍输出中移除，保留为后续深度/连续逼近佐证的边界证据，修复首版 26 帧持续路沿误报。
  - 连通域改为在完整类别区域生长，再计算中心走廊重叠与底部位置；通用障碍只转发中心路径最强候选，防止边缘大区域压过中心风险。
  - `DetectionSource.SEGMENTATION` 且无深度证据时，RiskAnalyzer 防御性限制为 `RiskLevel.LOW / ProximityBand.MID`，单帧不能直接提醒。
  - benchmark mask 改为 256×256、同一图片三次运行只解码一次；分析器复用 corridor、visited、queue 缓冲。
- 验证：
  - `:core:assist:test` 两次通过；`:device-benchmark:compileDebugKotlin` 通过。
  - 离线 30 帧 v2 oracle：主区域候选覆盖 `27/30=90%`。
  - 首次真机复测：错误提醒率 `3.3%`、主区域命中 `60%`、P95 `57.287ms`，据此修正最终候选优先级。
  - 最终 Samsung SM-S9280 A/B：错误提醒率 `3.3%`、主区域命中 `86.7%`、total P95 `65.919ms`，YOLO AP50/precision/recall 无退化；判定 `traversability_rules_ok_for_model_stage`。证据：`test-artifacts.local/detector-ab-device-benchmark/20260711-163629`。
  - 同轮默认模型 90 秒回归通过。证据：`test-artifacts.local/device-regression/20260711-163908`。
- 后续事项：仅完成当前否定集。仍需从许可明确公开数据补齐平行路沿、正前方台阶/横向路沿、路桩/低矮障碍/盲道占用连续序列；危险召回通过前不得训练模型。
- 版本判断：核心风险规则、性能与验证能力形成阶段升级，按较大更新从 v9.9.0 升至 v10.4.0（versionCode 36）；默认 YOLO 模型资产不变。
- 构建与归档：`:core:assist:test :app:lintDebug :app:assembleDebug` 通过；APK 已从 `app/build/outputs/apk/debug/app-debug.apk` 归档到 `E:\linnan\blind-assist-apk-archive\apks\BlindAssist-v10.4.0-debug-20260711-164742.apk`，大小 `55,844,975` bytes，SHA256 `7456034404F83CA8600FD39DEF09AB7741B219973B6B5CC774C6CBD4250455F8`。本次不提交 Git 里程碑 APK。

### SANPO Traversability v2 公开连续序列扩展与真机否决
- 时间：2026-07-11 19:20:00 +08:00
- 执行者：violjjet
- 类型：数据集、规则、测试、真机验证、文档
- 修改范围：公开 SANPO 候选发现、review profile、sequence clone/merge 工具；`TemporalRiskTracker`、`FeedbackPlanner` 与对应 JVM 测试；README、CHANGELOG、idea、基线文档。
- 数据与审查：
  - 扫描 SANPO-Real official test session 的稀疏 segmentation mask，筛选并下载三类 10 FPS 连续序列；来源为 CC BY 4.0，RGB/mask/QA 仅保存在忽略的 `test-artifacts.local`。
  - 台阶来源为 metadata 明确标记 `ELEVATION_CHANGE_STAIRS` 的 session `i2jglnBfoIqIIA7ojQGe-4vK07hUm4T3`；中心垃圾桶通道障碍来源为 high-obstacle test session `GxMb4zhAvoM5jbF54kfcs8wxTL4fqNnT`；平行边界负例复用已复核的 SANPO public pilot。三条均经 AI 双重复核、review CSV 和 finalize gate。
  - 未发现同时满足连续、可小规模下载、许可明确且显式“盲道占用”的公开序列；该缺口明确保留，不把普通通道占用伪称为盲道占用。
- 规则与验证：
  - 新增稳定分割候选的单级 `LOW -> MEDIUM` 晋级，要求两帧稳定或连续逼近；普通通用障碍还要满足近场底部 `>=65%`。路沿不进入该路径。
  - 反馈只接受带 `STABILITY_PROMOTED` 或 `MOTION_PROMOTED` 的中心分割候选，单帧不提醒。
  - 多轮 SM-S9280 真机 benchmark 逐步发现并修正：稳定证据未进入反馈、候选轨迹因 mask 形变重置、远距 generic obstacle 被误升。最终 90 帧/3 序列结果：危险提醒召回 `88.9%`、中心风险召回 `83.3%`、主区域命中 `93.9%`、total P95 `58.405ms`，但错误提醒率 `25.9%`，高于 `5.3%` 门槛。证据目录：`test-artifacts.local/detector-ab-device-benchmark/20260711-191206`；默认模型回归：`test-artifacts.local/device-regression/20260711-191513`。
- 结论与后续：扩展集未通过 Oracle v2 门槛，维持 `do_not_replace_default_model`；不开始 MobileNetV3 + LR-ASPP 训练。下一轮优先区分“已通过的台阶”与“仍在前方的台阶”，并对平行边界中的 generic obstacle 建立更严格的实例几何/深度否决。
- 版本判断：本轮仅增强实验 oracle、benchmark 和本地公开数据评测，默认生产模型与发布 APK 不变；保持 `v10.4.0` / `versionCode=36`，不新增 APK 归档。
- 构建与归档：`:core:assist:test :app:lintDebug :app:assembleDebug` 通过。为保留本轮公开序列验证对应的可测试 APK，已归档 `E:\linnan\blind-assist-apk-archive\apks\BlindAssist-v10.4.0-debug-20260711-192336.apk`，大小 `55,844,975` bytes，SHA256 `5670746F36B57DC556B1A8010E009604509F30BF0B1158E5C89D4E08F51C5774`；不提交 Git 里程碑 APK。

### SANPO v3 模型复核契约与连续素材筛选

- 时间：2026-07-12 +08:00
- 类型：数据治理、模型复核、候选采集、测试、文档
- 修改范围：`build_sanpo_sequence_evalset.py`、`discover_sanpo_sequence_candidates.py`、`review_sanpo_sequence_with_model.py`、`test_sanpo_v3_dataset_controls.py`、`docs/SANPO_V3_REGRESSION_DATASET.md`。
- 实现：将草稿初筛改为可追溯的大模型复核。每条序列生成首/中/末三帧证据及 SHA256；模型结果强制记录型号/版本、场景桶、走廊事件、`alert/no_alert`、置信度、证据帧和局限。模型复核只能决定是否进入四类像素级掩码标注队列，不能直接生成掩码、训练样本或 benchmark 晋级。平行路沿与侧向目标允许作为 `no_alert` 负例，避免被规则误改写为障碍正例。
- 首轮结果：道路横穿/入口台阶草稿被模型判为 `reject`（台阶未进入行进走廊）；侧向行人/犬只草稿判为 `needs_recapture`（同时存在中心构筑物，不能作为干净负例）。两条均为 `not_promoted`，没有进入 v3 训练或盲测集。
- 连续素材：候选扫描补充 `pedestrian`、`rider`、`vehicle` 语义类；中心障碍候选 `SRHpBZXk_0pKjk6SK23VOhoLZGZfnKFF` 已启动 50 帧公开 SANPO 下载与哈希校验，完成后必须再次经模型复核，未完成前不计入 420 帧覆盖。
- 验证：`C:\Users\26442\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe scripts\test_sanpo_v3_dataset_controls.py` 通过，4 tests passed；包含模型复核可接受高置信 `no_alert` 负例和证据帧缺失拒绝测试。首次沙箱运行因系统临时目录写入限制失败，受控权限重跑通过。
- 版本判断：仅修改 benchmark 数据治理与候选发现流程；生产 YOLO 路径、APK 和模型资产未变，不升级版本、不归档 APK。

### AGENTS 分层收敛与发布流程减负

- 时间：2026-07-13 +08:00
- 执行者：violjjet
- 类型：协作规则、文档、发布流程
- 修改范围：`AGENTS.md`、`docs/RELEASE_AND_VERIFICATION.md`、`docs/GLASSES_HARDWARE_ROUTE.md`、`docs/APK_ARCHIVE.md`、`docs/NEW_COMPUTER_HANDOFF.md`。
- 修改内容与原因：
  - 将项目 AGENTS 从长篇环境状态与逐任务强制流程收敛为项目边界、变更卫生、风险分级验证、产物/硬件/推送安全四类规则；避免只读任务也写日志、普通改动也跑发布级构建、普通 debug 包也自动归档。
  - 新增发布与验证工作流，明确文档、模块、风险链路和交付候选的最低验证级别；交付 APK 校验提供带必填 `-ApkPath` 的最小命令，并说明可选版本断言。
  - 新增眼镜硬件接入路线，把旧源码路径、占位功能边界和迁移前接口设计从 AGENTS 迁出，防止日常 Android 工作携带无关上下文。
  - 将 APK 归档调整为演示、老师查看、交付候选、里程碑或用户明确要求时才执行；既有历史归档保持不删除。
- 验证：
  - 检查全局、工作区和项目 AGENTS 的层级职责与相对 Markdown 链接；项目 AGENTS 指向的四份专项文档均存在。
  - `git diff --check` 通过；未运行 Gradle，因为本轮仅修改协作与流程文档，不影响 Android 源码、模型或构建配置。
- 版本判断：协作与文档流程优化，不改变用户可见功能、模型、权限或构建产物；保持现有版本，不归档 APK。

### Runtime session and root UI contract consolidation

- Time: 2026-07-13 +08:00
- Executor: violjjet
- Type: architecture refactor / runtime / UI
- Scope: `feature/assist/runtime/AssistRuntimeSession.kt`, `AssistRuntimeController.kt`, `AssistRuntimeControllerFactory.kt`, `core/ui/compose/BlindAssistAppContract.kt`, `BlindAssistApp.kt`, and `app/MainActivity.kt`.
- Implementation: introduced the `AssistSession` seam and routed camera, permission, replay, and runtime configuration through `AssistRuntimeIntent`; state machine, lifecycle gate, camera, detector, risk, and feedback implementations remain unchanged. `BlindAssistApp` now receives a state object and runtime/navigation/glasses action groups, reducing startup-shell coupling to product details.
- Verification: with local JDK 17 at `E:\codex-tools\tools\jdk17.0.19_10`, `:core:assist:test :feature:assist:testDebugUnitTest --rerun-tasks --no-daemon --console=plain` passed; `:core:ui:testDebugUnitTest :app:lintDebug :app:assembleDebug --no-daemon --console=plain` passed. The first attempt was blocked by system JDK 26.0.1 and a stale core:assist ABI cache; switching to JDK 17 and forcing recompilation resolved both.
- Version decision: no user-visible behavior, model, permission, or risk-rule change; version unchanged and no APK archived.

### SANPO 候选离线质量、INT8 保真与设备事件门拆分

- 时间：2026-07-13 +08:00
- 执行者：violjjet
- 类型：模型评价、门禁、测试、文档
- 修改范围：`scripts/sanpo_candidate_quality_gate.py`、`scripts/test_sanpo_candidate_quality_gate.py`、`docs/SANPO_CANDIDATE_PROMOTION_GATES.md`。
- 修改内容与原因：
  - 新增逐 session、逐 scene 的四类 precision/recall/IoU，以及 session 宏平均、最差 session/scene mIoU，防止长序列或大类别掩盖跨场景失败。
  - 将 boundary precision/recall/IoU 和 unknown abstain rate、known coverage、unknown precision/recall/IoU、covered accuracy 提升为显式诊断与硬门。
  - 新增 Keras→full-INT8 语义保真门，同时约束 argmax agreement、逐类预测 IoU、逐类真值 IoU 最大退化和平均 mIoU 退化；避免仅凭总体一致率漏掉稀有类别量化坍塌。
  - 离线训练质量、INT8 保真、同机连续序列设备事件三门完全拆分。无 TFLite 时可先做离线质量审计；无与模型 SHA256 绑定的设备报告时，设备门保持 `not_evaluated`，不得晋级。最终报告始终固定 `production_model_replacement_authorized=false`。
  - `backbone_alpha`（仅 0.75/1.0）与 `decoder_channels` 已贯通跨后端 worker、TensorFlow 导出和候选质量门。等价报告 schema v2 保存完整模型配置及 canonical JSON SHA256；consumer 按调用方预期配置 fail-closed，拒绝 alpha/decoder 错配、字段缺失和配置哈希篡改，防止同名权重被错误图加载。
  - 输入尺寸同样贯通并只允许 256/384/512，用于 SANPO 官方 512 路线的受控消融。固定合成输入 shape/hash、共享模型图、INT8 输入输出合同和质量报告均绑定同一尺寸；consumer 拒绝分辨率错配，既有跨后端与质量阈值保持不变。
- 验证：`.venv-export312` 下候选质量门 8 项、跨后端等价门 11 项、训练/INT8 导出 12 项单元测试及 Python `py_compile` 通过；导出测试包含真实 TensorFlow→full-INT8 TFLite smoke 和 256 模型被 512 合同拒绝的负向检查，配置/分辨率错配与配置哈希篡改均有测试。尝试复核 evidence-v4 旧权重时，因当前共享模型定义 SHA256 已变化而被既有跨后端 consumer 正确拒绝；未绕过绑定，需等待新模型重新训练并生成新的等价报告后实跑。
- 剩余风险：默认阈值是本轮预注册的首版安全门，后续只能基于独立 blind/真机证据版本化调整，不能用待晋级候选反向调阈值。本轮未导出 TFLite、未运行真机事件 benchmark、未替换生产模型。
- 版本判断：仅增加 benchmark 评价与晋级保护，不改变 App 用户可见功能、模型资产或运行时；版本不变，不归档 APK。

### SANPO v4 真实 session 扩展与 official-split 闭环

- 时间：2026-07-13 +08:00
- 执行者：violjjet
- 类型：公开数据扩展、来源治理、训练门禁、测试
- 修改范围：`configs/sanpo_v4_real_session_recipe_20260713.json`、`configs/sanpo_v4_real_source_receipts_20260713.json`、`scripts/build_public_v3_canonical_dataset.py`、`scripts/validate_sanpo_v3_dataset.py`、`scripts/sanpo_training_gate.py` 及对应测试。
- 数据与拆分：从 SANPO-Real v0 官方 GCS 新增 7 个 official-train 50 帧 session，并下载 2 个此前未进入 train/dev 的 official-test 60 帧 session；所有下载均保留对象 generation/MD5/CRC/source manifest。最终 recipe 使用 8 train + 4 session-held-out dev（两者都只来自 official train）和 2 benchmark-only blind（只来自 official test），共 720 帧。四个有可程序化语义证据的关键场景桶均达到 `train>=2`、`dev>=1`、总 session `>=3`；未把缺少可信筛选证据的普通 SANPO 片段伪称为低照或盲道占用。
- 门禁升级：固定 `6×50 + 2×60` 改为 recipe-bound expanded coverage policy，强制最小 train/dev session 数、逐场景独立 session 数、目标 split 到 SANPO official split 的映射，以及 exactly two 真实 blind session。builder 逐序列校验 `expected_frame_count` 和 `official_split`，继续保持 raw asset SHA inventory、session 隔离和跨 split 原资产否决。
- 质量观察：SANPO 机器标注可在不同 RGB 帧间复用同一 raw mask，且四类投影也可能相同；重复样本继续由 RGB SHA 拒绝，raw mask 跨 split 继续硬拒绝，同 split 重复 mask 改为报告型指标。最终 train/dev raw 与 semantic duplicate row 均为 `1/600=0.1667%`，blind 为 `0/120`。
- 构建结果：`test-artifacts.local/datasets/sanpo-v4-real-canonical-r3-20260713` 成功发布；600 train/dev + 120 real blind，12+2 独立 session，1440 个图像/掩码哈希匹配。最终 10 项检查全绿，training gate SHA256 `4c68e43494012f0499d8f9f01a5160a80276682fcd2e78a6ac5ca4cf98a1d5e1`；build report SHA256 `f7f7b11e4ca0f733dd4c5ccfb9f01ccf30548014c406edd72f308bb1fd6967b5`。
- 验证：builder/数据治理单元测试、Python `py_compile` 和最终 canonical 发布前/发布后门禁通过。生产 App、训练脚本和模型资产未修改；本轮不改变版本、不归档 APK。

### Workspace、工具链与本地产物收敛

- 时间：2026-07-13 +08:00
- 执行者：violjjet
- 类型：仓库结构、工具链迁移、文档信息架构
- 修改范围：外层 `E:\linnan\README.md`、根 `README.md`、`.gitignore`、`docs/README.md`、`scripts/README.md`、`docs/LOCAL_ARTIFACTS.md`、`docs/NEW_COMPUTER_HANDOFF.md` 及本机忽略目录。
- 修改内容与原因：
  - 明确 `E:\linnan` 仅为 workspace，`E:\linnan\linnan` 是 BlindAssist 唯一源码和 Git 命令入口。
  - 将约 20.7 GiB 实验证据、1.0 GiB 下载和本地工作目录归并到 ignored `artifacts.local/`；旧 `test-artifacts.local`、`.downloads`、`work`、`tmp` 保留 junction，避免一次性破坏历史命令。
  - 将 JDK、Android SDK、Gradle/Android/Kotlin 状态和缓存迁到 `E:\codex-tools\projects\blindassist`，旧隐藏路径保留 junction。`.python311` 与 `.venv-export312` 因 Windows DLL 锁和 venv 绝对路径暂保留原位，待重建验证后再切换；已回填并验证 Python 环境完整。
  - 新增 docs/scripts 稳定索引；根 README 从逐日工程日志收敛为产品定位、当前状态、构建方式和导航。历史与详细执行继续分别由 CHANGELOG/DEVELOPMENT_LOG 承担。
- 验证：迁移 `robocopy` 对实验物报告 21,372 + 541 + 1,592 个文件、0 failed；回填后的 `.venv-export312` 与标准 `E:\codex-tools\bin\blindassist-python.cmd` 均通过 `scripts\inspect_tflite.py`；迁移后的 JDK/SDK/Gradle 状态执行 `:app:tasks --offline` 为 `BUILD SUCCESSFUL`；新增 Markdown 导航目标存在。`check_repo_hygiene.ps1 -AllTracked` 仍被既有受跟踪 skills snapshot ZIP 拦截，属于本轮未处理的历史二进制策略问题。
- 版本判断：不改变 App 行为、模型资产、权限和风险规则；版本不变，不归档 APK。

### Workspace 布局规范固化与最终复核

- 时间：2026-07-13 +08:00
- 执行者：violjjet
- 类型：仓库规范、文档治理、最终核验
- 修改范围：`E:\linnan\AGENTS.md`、`AGENTS.md`、`DEVELOPMENT_LOG.md`。
- 修改内容与原因：将唯一源码入口、`E:\codex-tools` 工具链、`artifacts.local/` 产物职责、旧路径 junction 的迁移规则、README/docs/scripts 的知识导航职责，以及布局变更后的最低验证要求写入长期协作规范，避免后续重新在仓库根创建工具、缓存、下载或未索引文档。
- 验证：确认 Git 根为 `E:\linnan\linnan`；`test-artifacts.local`、`.downloads`、`work`、`tmp` 均指向 `artifacts.local/`；JDK/SDK/Gradle/Android/Kotlin 路径均指向 `E:\codex-tools`；`E:\codex-tools\bin\blindassist-python.cmd scripts\inspect_tflite.py` 通过；`scripts\check_repo_hygiene.ps1` 对 35 个变更路径通过；顶层 `docs/` 索引覆盖通过；`git diff --check` 通过。
- 剩余风险：历史受跟踪的 skills snapshot ZIP 仍会使 `check_repo_hygiene.ps1 -AllTracked` 报告二进制策略问题；它不属于本轮布局变更，需在单独的 Git 历史/归档治理任务中处理。
- 版本判断：仅固化协作和文件管理规范，不改变 App 行为、模型资产、权限或风险规则；版本不变，不归档 APK。
# SANPO P3-B chest-view boundary continuation

- Time: 2026-07-13 +08:00
- Type: public-data discovery / isolated evidence acquisition / dataset governance
- Scope: `scripts/discover_sanpo_sequence_candidates.py`, `artifacts.local/evidence/p3/`, `artifacts.local/evidence/datasets/`, and `docs/SANPO_P3_SPLIT_RECONSTRUCTION_2026-07-13.md`.
- Evidence: official-train chest-left index 20-29 yielded six new sparse `step_curb` candidates; five passed the mandatory 50-frame mask geometry screen. Isolated drafts for `JtMYI6rJ4wiDsEVffAkee0kR5Zmrf8vM` and `W1ZpmAq74J8xfKg23BcwLx0QHfxM6W_w` completed 50-frame RGB/mask download, manifest validation, and local geometry replay. W1's inspection frame depicts a stair/descending entrance, but it remains `pending_review`; source segmentation and machine inspection cannot create a safety label. The non-overlapping index 30-39 scan completed with zero candidates and zero failures; report SHA256 `659f553fdb418efd083d54a68365cccd03a1a4d02bdfa99771a9292b80121824`.
- Decision: do not rebuild canonical data, start training, merge head and chest cameras, alter risk labels, or replace the production model. Continue only with official-train candidate discovery and fail-closed 50-frame screening until coverage and independent review gates are met.

- Continuation: the disjoint chest-only `40-49` scan attempted ten official-train sessions, produced seven sparse `step_curb` candidates with zero network/data failures, and all seven passed the remote 50-frame geometry gate. Sparse report SHA256: `219aaa995139d74dd5285f4e8b43b3810d5fdb7c1c1eb1285d273cc5e2f13287`. The strongest, `DD9W-6F3D126azdsR_Usvu6zkNqkP8XG`, completed a 50 RGB+mask isolated draft, manifest validation, and local geometry replay (selection SHA256 `c74c49fbc1cbbfd7dc11a2fa2d115519e54c7937692c7bc88519fcd50a2f34cb`). Its inspection-only middle RGB is an ordinary city sidewalk with street furniture and curb/snow edge, not a clear step; retain `pending_review` and do not create a risk label. This is evidence that the source-mask boundary profile needs semantic review, not a shortcut around it.

- Continuation: chest-only `50-59` discovered two candidates and no terminal network/data failures; a TLS EOF recovered inside the configured retry contract. Both passed remote 50-frame geometry. `qFDP9gJDz4MyXNxjl6mIEPFCUb8n1guU` completed an isolated RGB+mask draft, manifest validation, and local replay (selection SHA256 `74ce730413f5ab6827e5b257b3c59a1661d21be6c5bdc66e5b36f87dcaed8a70`). Its inspection-only middle frame shows a curb-cut/crosswalk transition with pedestrians. Keep it `pending_review`: it is a semantically plausible step/curb review candidate, not a machine-approved alert or training label.

### SANPO counterfactual episode manifest gate and P3-B continuation
- Time: 2026-07-13 +08:00
- Scope: `scripts/validate_sanpo_counterfactual_episodes.py`, its deterministic tests, `docs/SANPO_COUNTERFACTUAL_EPISODE_COLLECTION.md`, plus official-train chest-only discovery evidence under `artifacts.local/`.
- Counterfactual gate: converted the collection protocol from documentation-only into a fail-closed manifest validator. It verifies local-only file paths and SHA256s, accepted license and green privacy receipts, non-empty annotator IDs, positive/negative anchor rules, same-session/same-scene/same-receipt matched pairs, exact capture-context equality, and complete per-session/per-scene quotas when requested. It may report training eligibility only for a complete, reviewed matrix and always reports `production_model_replacement_authorized=false`; it never generates labels or downloads data.
- Verification: validator negative tests cover mismatched negative anchors, counterfactual capture-context mismatch, and incomplete collection promotion. Its 3 tests plus candidate discovery, local geometry, and P3 planner tests passed `25/25`; Python compile and `git diff --check` passed.
- Data continuation: official-train chest-only index `60-69` attempted 10 sessions (6 chest inventories and 4 insufficient pages), found four sparse candidates with zero failures, and all four passed the remote 50-frame geometry gate. Sparse report SHA256: `0579a75754c3429b9e33830575edababff9cc102d56dd20154b8b18b3671cab0`. Their maximum corridor-blocking ratio is <= `0.04938`, so they remain mask-only review candidates; no RGB download or training decision was made.
- Collection handoff: added `configs/sanpo_counterfactual_episode_manifest_template_v1.json` and documented the exact validator invocation. The empty template validates only as `collection_status=in_review`, `episode_count=0`, `training_eligible=false`, and `production_model_replacement_authorized=false`; it is an explicit field scaffold, not collected evidence.
- Seed queue: recorded the downloaded chest-view windows in ignored `artifacts.local/evidence/p3/sanpo_counterfactual_seed_queue_20260713.json` (current SHA256 `29cfa415c8190d45f4ea07005da24f73045ab05281445eb6835caa29c4d17ec0`). qFDP is a curb-cut/crosswalk-transition seed and DD9W an ordinary-boundary seed. Each is only five seconds, lacks a same-session matched counterpart and human event anchors, and is explicitly forbidden from use as a complete episode, label, calibration input, or benchmark truth.
- Continuation: official-train chest-only `70-79` attempted 10 sessions, yielded five sparse candidates and no failures (report SHA256 `a13c05a45fd750276c0c323930429be485c38d96245eded3c115421d08c71d3a`); all five passed the remote 50-frame geometry gate. `vczXAwthxnadTYUS_TiR7IHiqEQrdSJx` was downloaded as an isolated 50 RGB+mask draft and passed manifest validation plus local replay (selection SHA256 `2cf176bae0239f763f2473654fef82a05b538993a7e7aa90c96f52c1e0d0aad6`). Inspection-only middle RGB shows entrance stairs beside, rather than entering, the forward sidewalk corridor. It is added only as a `parallel_boundary`/matched-negative review seed; no alert label, P3 recipe entry, canonical row, training, or model promotion is authorized.
- Continuation: official-train chest-only `80-89` attempted 10 sessions, yielded three sparse candidates and no failures (report SHA256 `6bbeda410b728d423a3161a68d428e4fd032ee7f21343ee4024a5326ff1a4720`). All passed 50-frame geometry but had maximum target corridor-blocking ratios <= `0.02550`; retain mask-only evidence and do not download RGB merely to inflate the candidate pool.
- Continuation: official-train chest-only `90-99` attempted 10 sessions, yielded five sparse candidates and no failures (report SHA256 `c9600534e7129f68d5112e42070602420bd88672589352eb75720864818ca422`). All passed 50-frame geometry; the best maximum corridor-blocking ratio was `0.13272`, below existing higher-information reviewed drafts and without a new scene signal. Retain mask-only evidence; do not download RGB or infer alert labels.
- Continuation: official-train chest-only `100-109` attempted 10 sessions, yielded two sparse candidates and no failures (report SHA256 `4130f8cb1059805461d88b75b3ec4694c56b50f4a76061a4f1cdd63186c74d67`). `gie8...` was already an isolated center-obstacle draft and was not duplicated; new `yGcm...` passed 50-frame geometry but had `max_block=0.07497`, so remains mask-only. No RGB download, risk label, canonical entry, training, or model promotion is authorized.

### SANPO P3 full official-train discovery, resumable evidence contract

- 时间：2026-07-13 +08:00
- 类型：公开数据只读发现、断点恢复、数据治理、测试
- 修改范围：`scripts/run_sanpo_p3_discovery_batches.py`、对应单测、view/source contract 与 P3 full-discovery record。
- 实现：将 official-train 候选扫描按 20 session 分批 checkpoint；checkpoint 绑定官方 session 顺序和全部筛选参数，`--resume` 对参数/顺序变化 fail closed。每批要求完整尝试且零 unresolved failure；只有全部 batch hash/coverage 通过才写 aggregate。
- 结果：official-train index `0–559` 的 560 session 已完成 `28/28` 批次、0 unresolved failure，aggregate 146 candidate row（chest 86/63 unique session，head 60/42 unique session）。aggregate SHA256 `a5031bea47fae0c66bd59aa12b036a2ac420d3b0681d82ee7d25867078dd9889`；checkpoint SHA256 `9b0430be23724822b2ff3c3994cd226800b8296d287b796fd0c33cc80229d7ee`。完整记录见 `docs/SANPO_P3_OFFICIAL_TRAIN_FULL_DISCOVERY_2026-07-13.md`；证据位于 ignored `artifacts.local/evidence/sanpo-p3-discovery-auto-20260713/`。
- 决策：解除长扫描不可恢复与 official candidate-pool 未穷尽的工程阻塞；不解除 P3 canonical/训练门。候选仍需 50-frame、RGB/PII、人工语义、annotation-quality、session/split、camera cross-view 审计。未读 blind、未启动训练、未改 App/默认模型。

### SANPO P3 lateral bottleneck closure and consented-capture planner gate

- 时间：2026-07-13 +08:00
- 类型：公开数据精确筛选、P3 真实来源接入、训练门治理、测试
- 证据：full official-train pool 的四条 chest lateral 候选均完成 exact 50-frame remote-mask gate，结果 `0/4` accepted。拒绝原因分别为中心 target 污染、lateral target 仅 15 帧、中心 hazard + target 污染、以及中心 hazard 污染；因此不能伪称为 clean-lateral negative。该结果使“只靠 SANPO public official-train 补齐 lateral P3 session”明确不可行。
- 实现：P3 planner 现在新增 `consented_forward_phone_v1` 的 source admission。它在打开 manifest 前验证无 PII 的同意 receipt，强制 granted consent、已过 residual-PII、human-verified pixel annotation、human annotation quality、approved scene review 和仅允许的前向胸前/手持手机采集模式；逐 manifest row 再绑定 receipt SHA、human quality 与 PII clearance。machine-only、缺 receipt、official-test 或其他未批准 source 均 fail closed。
- 验证：planner 11 项单测通过，包含 consented source 通过、machine-only receipt 预读取拒绝、official-test sentinel、raw-mask 跨 split 泄漏和分布门。未采集/写入任何真实个人数据，未下载 RGB、未读 blind、未训练、未改 App。

### SANPO P3 exact lateral closure and center review drafts

- 时间：2026-07-13 +08:00
- 类型：官方公开 mask 精确筛选、隔离 draft、审核队列
- lateral：4 条全量发现的 chest lateral candidate 全部在 exact 50-frame gate 拒绝。`4P1…` 是 center target 污染，`edl…` 仅 15 帧 clean lateral，`JtMY…` 同时存在 center hazard/target，`vcz…` 存在 center hazard。四份独立 evidence SHA256 为 `b7bf03fcabf9dc8a43cec0ac0063076e8a1661798233dbf794aaabdc01c50124`、`bf0c7a034e5bfc4b1cbccbf580c603d2eaccdffee701b35b96b0fde101197588`、`c2444abec8e10c8252753a55fb31eb25de3884905322655ddfaba919e482eea0`、`c9eb5445224e963a745ebfb064d4aefb1f4d0134df7533c79bd4d185586cc872`。因此 official-train 不能单独补齐 lateral P3 配额。
- center：8 条新 chest candidate 的 exact gate 得到 3 accept / 5 reject。`3ok1…` 和 `cBVS…` 已完成官方 train 50 RGB+mask 隔离 draft、本地 geometry replay 和 hash-bound model-review request，仍为 `pending_review`；selection evidence SHA256 分别为 `93a9a34ebe8cb3b7363c520200f9639c05d82d3af409c695026e093ce764e659`、`6b1128d27ba201178bc314043f695971e2580a621e0db0820eb2d7aa2d834ff8`。`JtMY…` 的 RGB 下载在上限超时且未写 manifest，已仅终止本任务孤儿进程并排除，不删除不完整目录。
- 决策：公开 center 候选可继续人工/PII 审核；lateral 缺口改由 consented forward-phone A 层补齐。未把 source-mask 几何、模型审核请求或 pending draft 计为人工标注；未训练、未读 blind、未改 App。

### SANPO P3 boundary semantic-review closure

- 时间：2026-07-13 +08:00
- 类型：隔离公开 draft 的模型审核路由、数据治理
- 证据：已对 qFDP（curb-cut/crosswalk）、vcz（侧方入口台阶）和 W1（有界连续下坡/栏杆）完成 hash-bound model-review response 验证。结果依次为 `needs_recapture`、`reject`、`needs_recapture`，均显式返回 `promotion=not_promoted`；预期均为 `no_alert`。这不是人工语义审核，也不产生风险标签、训练样本、标定/benchmark 真值或模型升级授权。
- 决策：source-mask 中持续出现的 boundary/curb 几何与事件级告警语义多次不一致，禁止将该 profile 用作伪标签。下一步只可将这些五秒公开窗口作为未标注审核 seed；补齐 P3 必须依赖 `10–20 s`、同 session 匹配对、人工锚点和隐私/同意收据均齐备的真实 episode。P3 coverage 仍红，未重建 canonical、未训练、未改默认模型、未读 blind。

### SANPO P3 center-obstacle discovery continuation

- 时间：2026-07-13 +08:00
- 类型：公开 official-train 只读候选发现
- 证据：auto-view index `110-129` 完成 20 个独立 session 的稀疏扫描，产生 7 条 `center_obstacle` candidate、0 failures，报告 SHA256 `10e4dab53e6aecf385f35671977071b846c1dfc29ae6a9a023463d70d5ef3476`。其中 `Nta3...` 是既有隔离候选，避免重复；另有 3 条 chest 和 3 条 head-only discovery row。
- 决策：这些均未经过 exact 50-frame gate，故不是 RGB 下载、语义审核、P3 recipe、canonical、训练或模型升级候选。head-only 行继续受独立 cross-view gate 约束；未读 blind、未训练、未改默认模型。

### SANPO P3 consented-capture taxonomy and RGB binding closure

- 时间：2026-07-13 +08:00
- 类型：真实采集 intake 合同修复、数据准入治理
- 根因修复：此前 P3 planner 的像素统计只适配 SANPO 原始 31 类 ID；经同意手机来源即使具备人工四类 mask，也可能被按 SANPO taxonomy 错误映射。现将 `sanpo_real_v0_panoptic_class_id_v1` 与 `blindassist_4class_mask_v1` 分离，并把后者强制绑定到 consent receipt 和每帧 `human_pixel_mask` provenance。任何未声明、错配或含非 `0–3` 像素值的经同意 mask 都 fail closed。
- 证据绑定：经同意 session 的每帧现在还必须绑定 RGB `image_path`/SHA256、前向 capture mode、`lens=not_applicable`、正整数原始宽高；planner 重算文件哈希，验证图片尺寸，并拒绝 session 内 camera/lens/分辨率变化。真实手机分辨率记录为 session invariant，不再误用 SANPO 的固定分辨率白名单。
- 验证：新增 taxonomy mismatch 与 RGB 尺寸错配拒绝测试；P3 planner、counterfactual manifest、候选发现与本地 geometry 共 27 项测试通过。未采集真实个人数据、未读 blind、未训练、未生成或替换默认模型。

### SANPO P3 completion-evidence audit

- 时间：2026-07-13 +08:00
- 类型：训练前事实核验
- 核验：本地 `artifacts.local/evidence/` 未发现受同意长时 episode manifest 或完成的同意采集 receipt；唯一 counterfactual manifest template 的验证结果为 `collection_status=in_review`、`episode_count=0`、`matched_pair_count=0`、`training_eligible=false`、`production_model_replacement_authorized=false`。现有 SANPO draft/model-review result 及 procedural reviewed-source manifest 均不构成该真实事件数据。
- 结论：P3 数据问题尚未解决，且没有证据支持启动训练、校准、canonical 重建或默认模型替换。下一份可使状态前进的外部证据必须是满足 `10–20 s`、同 session 匹配正负 pair、人工事件锚点、同意与 PII 审核、人工四类 pixel mask 的真实采集包；公开 SANPO 候选精筛可并行恢复，但不能代替该包。

### SANPO P3 resumed center-obstacle exact screening

- 时间：2026-07-13 +08:00
- 类型：公开 official-train 50-frame source-mask 精筛、下载完整性治理
- 证据：`We-WV...` 与 `J4P...` 的 chest 50-frame remote-mask gate 分别 accepted（`26/17/50` 和 `41/35/50` 的 target/run/path），screen report SHA256 分别为 `eedf3ff1ff54289141b524e1d68c387c561fbe473c67124b109b7f35606941c0` 与 `bd0acd7645bc3e2e1c111bf2c21691024e43dff29f8558656034b76982f68b04`。`HEm...` 仅有 30 个对齐 source mask，按不足 50 帧拒绝（SHA256 `f479f9cfcd6abff5b0035bbd430b3dce456316e6ac971e0a52b93e210c36a70fc`）。
- 完整性决策：优先 J4P 的隔离 RGB+mask download 在 10 分钟上限超时，未写 `manifest.draft.jsonl`；目录中只有空的 images/qa/source_masks 框架。保留但明确排除，不删除、不重试到同一路径。未执行本地 replay、PII/语义审核、annotation queue、训练或模型替换。

- Retry result: J4P was redownloaded into a fresh isolated root and completed 50 RGB/mask frames with `validation_ok=True` and `benchmark_ready=False`. Local geometry replay exactly matched remote selection (`target=41`, `run=35`, `path=50`, `median=0.9140`, `max_block=0.10590`; selection SHA256 `7d86dccd97a365c4aa9efb1765fdef8751e586e457cab3d2fe8e241babfbf88e`). Hash-bound model review returned `needs_recapture`, `center_obstacle`, corridor event present, expected `no_alert`, confidence `0.76`, selection disagreement; verifier output is `promotion=not_promoted`. The visual route context is insufficient to decide whether a large fixed structure blocks the actual intended path. It is an unclassified counterfactual seed only (current queue SHA256 `e7a0253a41372151c246ebab883c6d39e63547ec91684220c4cdea5d4cc23a7e`), never a label, canonical row, training sample, calibration input, benchmark truth, or model-promotion input.

- Continuation: the other accepted chest candidate, We-WV, completed 50 RGB/mask isolated download and local replay (`target=26`, `run=17`, `path=50`, `median=0.8065`, `max_block=0.21535`; selection SHA256 `bed76cb0b28dfecc21e1469813b90f14bc145bb6f354cf0aabdd63151760b06e`). Its hash-bound model review returned `reject`, `parallel_boundary`, corridor event absent, expected `no_alert`, confidence `0.95`, selection disagreement; verifier output `promotion=not_promoted`. The visible scene is an open plaza/walkway with static benches, planters and entrance fencing; the near-forward route remains open. This is a second direct static-boundary false-positive, retained only as an unclassified counterfactual seed. The two complete center-profile drafts admit neither dense annotation nor P3 canonical/training/calibration/promotion; source geometry alone is now explicitly insufficient for that profile.

- Historical-draft closure: `3ok...` (50/50 source-mask candidate) was visual-model rejected as an open plaza with static seating/fence/planting/pigeons, `parallel_boundary`, no corridor event, expected `no_alert`, confidence `0.97`, `promotion=not_promoted` (selection SHA256 `93a9a34ebe8cb3b7363c520200f9639c05d82d3af409c695026e093ce764e659`). `cBVS...` (target 43/run 31/path 50) is a crowded market walkway and remains `needs_recapture`, `center_obstacle`, corridor event present, expected `no_alert`, confidence `0.72`, selection disagreement, `promotion=not_promoted` (selection SHA256 `6b1128d27ba201178bc314043f695971e2580a621e0db0820eb2d7aa2d834ff8`). Both are unclassified counterfactual seeds only; no annotation, training, calibration or promotion use is allowed.

### SANPO risk-profile and lifecycle primary-supervision gate

- 时间：2026-07-14 +08:00
- 类型：事件数据合同、训练前治理、测试
- 实现：counterfactual manifest validator 现在强制每个 episode 具有 10–20 秒整数毫秒 `duration_ms`、与 `scene_id` 绑定的 `risk_profile`、和严格由人工事件锚点导出的 `lifecycle_intervals_ms`。positive 必须是 `enters_or_blocks / approach_alertable_clear`，且三段区间精确覆盖 approach、alertable 与 post-event；matched negative 必须是 `outside_or_nonblocking / no_alert` 且完整覆盖 `non_alert`。短片段、浮点/失序锚点、错配风险轮廓或生命周期区间均 fail closed。
- 原因：将未来风险轮廓 + 生命周期头的主监督固定在可审计事件语义上，像素 mask 只作辅助。这直接针对本轮反复发现的围栏、长椅、广场边界、固定结构等 geometry false positive，禁止其仅凭 mask 进入 alert 正例。
- 验证：新增时长不足与 lifecycle interval 错配拒绝测试；并新增 `build_sanpo_risk_lifecycle_targets.py`，它只从 complete、人工复核的 episode manifest 生成确定性风险轮廓/生命周期 target，输出明确 `pixel_supervision_role=auxiliary_only`、`training_execution_authorized=false`、`production_model_replacement_authorized=false`。counterfactual、P3 planner、发现器与本地几何共 29 项测试通过。未读 blind、未训练、未改 App/默认模型。

### SANPO counterfactual collection-plan generator

- 时间：2026-07-14 +08:00
- 类型：真实 episode 采集执行支架
- 实现：新增 `generate_sanpo_counterfactual_capture_plan.py`，从冻结的 6 session × 4 scene × 2 matched pair 合同生成 96 个 `not_captured` slot。每个 slot 固定正/负采集合同、10–20 秒范围、pair 必须共享的上下文、风险轮廓与生命周期模板，并显式禁止将清单当作证据、标签、receipt 或训练许可。
- 决策：采集清单是填补 P3 根因数据缺口的执行工具；它不伪造任何真实视频、人工标注、隐私回执或训练状态。
# 2026-07-15 — Counterfactual independent-review evidence gate

- Strengthened the counterfactual episode validator: every episode now requires a local SHA256-bound review record, at least two independent human reviewers, and reviewer-ID agreement with the manifest. Positive-event anchor disagreement above 500 ms fails closed; model output cannot serve as event-label evidence.
- Verified with the project Python under a workspace-local temporary directory: 31 focused tests passed, including rejection of a single-review record and unstable positive anchor adjudication. This remains a collection/annotation gate only; no real episode was created, no training was run, and default model replacement remains unauthorized.
# 2026-07-15 — Autonomous public-video candidate path

- Added a fail-closed public-source acquisition path. `acquire_public_gnd_candidate.py` verifies a bounded CC0 GND file through Dataverse metadata before any download and writes a receipt that permanently excludes the material from calibration, safety truth, production training, and model replacement. The remote endpoint timed out during live metadata validation, so no GND data download is claimed.
- Added `acquire_sanpo_rgb_timeline_candidate.py` for low-bandwidth, resumable CC-BY SANPO RGB-only timelines. It intentionally omits masks and event labels; model output can only decide whether a longer public clip merits later review. Tests cover sparse 10-second selection, incomplete windows, resumability, public-file bounds, and counterfactual gates (12 focused tests passed).
- Live candidate review stopped the J4P expansion after its first verified RGB frame: the center sidewalk is visually clear while benches, pedestrians, and parked vehicles remain lateral. This agrees with the prior `needs_recapture/no_alert` finding and is evidence against using this source-mask profile as a weak positive; no training, calibration, or promotion was run.
- Extended the model-only RGB screen to four more local SANPO candidates and recorded every decision in `artifacts.local/evidence/p3/public_model_candidate_screening_20260715.json`. `5ll`, `J4P`, and `gie8` were rejected as weak positives: their apparent targets are edge-static or leave an open route; `cBVS` needs longer context rather than an inferred event label. `k7` is retained as the sole high-priority human-review candidate because multiple oncoming pedestrians approach within a corridor constrained by a food truck and storefront. This is acquisition prioritization only—not a `should_alert` label, a lifecycle target, or training authorization.
- Extended autonomous visual review to the already-local cBVS and blind 5ll sequences. cBVS needs longer temporal context before any path-cut-in claim; 5ll is rejected as a weak positive because lateral snowbanks/vehicles/columns leave the center corridor usable. The model screening receipt is local-only and explicitly has no event-label or training authority.

### 2026-07-15 — Public first-person sequence acquisition and privacy-prefilter audit

- Source boundary: acquired the public uB-VisioGeoloc sequence-10 archive through its Harvard Dataverse metadata contract (declared CC0 1.0). The 1,319,673,592-byte archive matched the publisher MD5 `c92eff31b8e391dc6539640de1891341`; the local receipt and source config bind it to *unlabeled first-person representation candidate* use only. RGB extraction retained exactly 804 `10/Color/` PNG frames and deliberately did not extract Labels, Depth, GPS, IMU, boxes, or any source geometry as risk supervision.
- Privacy-prefilter result: an initial Haar-plate pass was rejected for pavement false positives. The subsequent dedicated LPD_YuNet pass still missed a visible vehicle plate in spot review, and was also rejected. The retained conservative pass uses YuNet face detection, LPD_YuNet, and whole-person/vehicle blur (YOLOv8n): it produced 804 redacted frames, 199 frames with regions, and 585 blurred regions. A visual spot check confirms the visible near vehicle is blurred as a whole; it does **not** prove that distant faces, plates, or all identifying text were found.
- Gate: the resulting receipt remains `privacy_audit_required=true`, `training_execution_authorized=false`, and `production_model_replacement_authorized=false`. This short single-scene (~27 s) sequence is neither event truth nor a sufficiently diverse training corpus. It may not be consumed until a separate privacy audit explicitly clears it, and even then only as unlabeled representation data followed by the frozen SANPO linear probe—not as risk/lifecycle/pixel truth.
- Verification: Dataverse acquisition, RGB-only extraction, and redaction helpers have 10 focused unit tests passing. No blind data was read; no safety training, calibration, canonical rebuild, or model promotion was run.

### 2026-07-15 — Foundation-feature and boundary-distance root-cause diagnostics

- Frozen-feature triangulation: added a train/dev-only, deterministic Depth Anything V2 Small DINO feature probe. It reached global mIoU `0.423488` but boundary IoU only `0.131167`; repeated closed-form ridge coefficients and dev argmax matched exactly. An exploratory frozen concatenation with the current raw MobileNet OS8+OS32 features reached only `0.414418/0.138427` (mIoU/boundary), also exactly repeatable and below the preregistered `.35/.20` separability gate. Neither run opened blind data, trained a feature/decoder, or authorized a model change.
- Distance-field admissibility: independently diagnosed signed 16-pixel boundary targets at 384² before wiring them into the trainer. Real train has boundary pixels `0.8569%` and near-boundary pixels `3.8388%`; dev has `16.9761%` and `8.0587%` respectively. This ~`19.8×` boundary-coverage mismatch makes a current auxiliary-loss short run non-attributable: it would test split composition as much as the loss. Do not present any distance-field training result before reconstructing coverage.
- Decision: the evidence strengthens the original diagnosis—this is not a head-only optimization failure. Do not start prototype/bootstrap short runs, SAM/ASAM, or distance-loss promotion on the current split. First repair session/scene boundary coverage and collect human-reviewed event pairs; then run exactly one frozen P1-A OFAT distance-field experiment under the documented contract. Detailed evidence is in `docs/SANPO_FEATURE_AND_DISTANCE_DIAGNOSTICS_2026-07-15.md`.

- Continuation: the promised OFAT distance-field test is now complete under a train-only session-held-out replay to remove the original train/dev boundary-density confound. Two held-out canonical-train sessions (100 frames) had `0.79325%` boundary pixels and the six optimization sessions (300 frames) had `0.87812%`, a permitted coverage ratio of `0.90336`; canonical dev/blind assets were not read and no weights were saved. Across five fixed model/sampler seed pairs and 100 steps per arm, signed 16 px weighted-SmoothL1 distance supervision at weight `0.20` changed mean IoU by `-0.001918`, boundary IoU by `-0.000610`, and selection score by `-0.001230` versus paired baseline. Only one pair improved boundary IoU and its mean IoU fell. Report SHA256: `f6482912c37e111e08d17214b5f3b15b30e7b99e20a58880ac672c79842fabef`. This is a negative result for the current distance-head recipe, not a model promotion or event-label result; retain `do_not_replace_default_model`.

- Continuation: added `sanpo_risk_lifecycle_prototype.py`, a no-trainer temporal multi-head prototype over externally supplied frame features. It accepts only hash-attested, complete human-reviewed risk/lifecycle targets, maps their half-open millisecond intervals deterministically to `non_alert/approach/alertable/post_event`, and emits episode-level hazard/corridor/should-alert logits plus per-timestep lifecycle logits. It rejects incomplete/unauthenticated targets and any non-`auxiliary_only` pixel role; public masks, geometry and model output cannot enter as event truth. Focused contract tests passed (15 total with the adjacent counterfactual and distance tests), and a Keras-torch smoke produced shapes `(2,4)`, `(2,2)`, `(2,1)`, `(2,5,4)`. No real target collection exists, no training or weights were produced, and the contract keeps both training execution and default-model replacement unauthorized.

- Continuation: added and ran `run_sanpo_corridor_anomaly_probe.py`, an explicitly non-alert, no-weight frozen Depth Anything V2 DINO familiarity diagnostic. It fitted a 32-component PCA reconstruction subspace only to 25,600 canonical-train source-semantic walkable patches, then scored held-out canonical-dev source classes. AUROC against walkable was `0.891081` for boundary_step_curb, `0.941669` for obstacle, `0.857822` for unknown_nonwalkable, and `0.898064` for all nonwalkable; the preregistered source-semantic interpretation gate passed. Report SHA256: `c7ad14b2e9ada2637f65b6a2a4db6f20298a1d0ff350bb4668292bf34c053a61`. This only permits a future `unknown_motion_or_surface` auxiliary/abstention candidate—not an alert, risk/event truth, threshold calibration, benchmark claim, or model promotion. During the first run the now-CUDA-enabled Depth Anything image tensor was on GPU while the model was on CPU; `smoke_depth_anything_v2_pytorch.py` now moves loaded frozen weights to the same selected device, with a focused unit test. No blind data, risk target, trainable model weight, or default-model change was produced.

- Continuation: explicitly tested whether the current frozen MobileNetV3 raw OS8+OS32 representation could make the DINO familiarity score practical on-device before building a new head. The fixed teacher/student ridge probe used 25,600 train and 12,800 held-out dev feature rows, without training either encoder or saving a student weight. Dev teacher reproduction was poor: `R²=-0.101451`, Spearman `0.601385`, failing the preregistered `R²>=0.50 && Spearman>=0.70` gate (report SHA256 `ab84880d5026db94fe30699dbc04cdf30f7f0231b74ffcbb4d4b61e34defde1`). Therefore do not add a MobileNet unknown head, non-linear student, SAM/ASAM, or hyperparameter rescue search; this evidence further localizes the current limitation to the lightweight representation rather than an omitted scalar head. The test remains source-geometry-only, with no blind data, event truth, calibration, exported weights, or default-model change.

- Continuation: completed an external event-source admissibility audit rather than treating public-video claims as data access. PEDESTRIAN is highly relevant on paper (egocentric pavement obstacles), but both its DOI resolver and Zenodo API `records/10907945` returned a persistent 404 on 2026-07-15, so it is unavailable. SideGuide remains approval-gated; VIEW360 is data-coming-soon and semantically out of scope; EgoTraj is privacy-blurred but carries non-commercial terms and VLM-generated scene descriptions, so it can at most be a separately authorized unlabeled representation candidate. No audited public source satisfies the counterfactual human event/lifecycle truth contract. The detailed source/permission matrix is `docs/SANPO_EXTERNAL_EVENT_SOURCE_AUDIT_2026-07-15.md`; no source data was downloaded or used for training.

### 2026-07-15 — Auxiliary-only public boundary-candidate pipeline

- Planned a separate public SANPO source-mask route strictly for pixel/geometry auxiliary work. The plan excludes every current canonical session, limits candidates to official train chest/left views, ranks sparse step/curb boundary coverage, and prohibits risk/event/lifecycle labels, calibration, benchmark truth, training execution, and default-model replacement. The initial plan selected eight candidates; its SHA256 is `aeeb445e800995d2db00be0cacbbcd6c7acd1c529009049aeaaeca687db3160f`.
- Three candidates (`vcz`, `qtty`, `4P1`) passed exact remote 50-frame source-mask geometry screens. This is only mask geometry evidence. The qtty RGB+mask draft subsequently completed with 50/50 MD5-verified files and 50 null risk-field rows; validation explicitly reports `benchmark_ready=false`.
- Privacy/utility check: qtty is a dense-pedestrian scene. Its conservative YuNet/LPD-YuNet/whole-person-or-vehicle prefilter created 684 blur regions across all 50 frames. The redacted derivative remains `privacy_audit_required=true`, `risk_or_event_truth_present=false`, and `training_execution_authorized=false`; it is not admitted to any training path. This prevents the public-mask coverage fix from silently introducing private or heavily occluded RGB data.

- Continuation: low-density candidate `wBP` completed its isolated 50 RGB/50 source-mask draft with 50 null risk/event/lifecycle records. Three raw-frame spot checks showed no near person or readable face, but vehicles and street signage meant that no privacy clearance was inferred. The whole-object redaction pass retained 50 RGB frames and blurred 186 regions in 48 frames; receipt SHA256 `d99b794c235e0f24a0656fe8da3f1719b283a795571e6de323e03997a9501129`. It remains `privacy_audit_required=true`, `auxiliary_pixel_geometry_only`, and has no training, risk-truth, calibration, benchmark, or production-replacement authority.

### 2026-07-16 — Public-silver trajectory root cause and provisional lifecycle-head experiment

- The r3 public-video package contains 11 non-abstaining episodes from 9 packages: 6 alert and 5 no-alert sources, with two explicit matched counterfactual pairs and no cross-source duplicate frame hashes. Licensing, source manifests, frame hashes and provisional-training attestations passed; calibration, blind evaluation and production replacement remain unauthorized.
- Static MobileNet/DINO pooling, segmentation-corridor pooling and fixed residual flow all remained below the frozen-feature gate. A new fixed YOLO12n-proposal trajectory representation (relative scale, bottom/corridor overlap, persistence and temporal slopes) passed source-isolated linear diagnosis at `0.816667` balanced accuracy with confusion `[[4,1],[1,5]]`; report SHA256 `ab54c01002f6a18830750a8185855e299916e1a24d18ddb191e930d2835f6596`.
- Five prototype-initialized, within-class source-bootstrap 80-step linear-head runs produced balanced accuracy `.65/.7333/.7333/.8167/.7333`; 4/5 passed the preregistered per-run gate and the median was `.7333`, so the head-stability gate passed. Report SHA256 `84ae652907ca774990968177d12e86bc2230f04ddca74def81b3699c9d0f7b39`; no weights were saved.
- Added a weakly supervised risk-profile MIL prototype: per-frame object/corridor features feed a tiny linear risk head and smooth-max episode pooling. Only episode alert/no-alert is supervised; lifecycle states are latent curve-derived diagnostics because the silver package has no trustworthy frame boundary labels. Five LOSO bootstrap runs had median balanced accuracy `.7333` (range `.6333–.9167`), while both matched pairs preserved correct alert-over-no-alert probability order in all five runs (10/10). Report SHA256 `af329f062b60a94c8650f6bf5232c20ad348f63cc79853067bbd4f6b6d4719d6`.
- Remaining failure is structured rather than optimizer-only: `1sft` (persistent pedestrians in the corridor with weak scale growth) was missed in every run, while the open `We` scene and static `gie8` obstruction were bootstrap-sensitive. Next data should target those two hard counterfactual families and add explicit alertable/cleared boundaries before claiming supervised lifecycle performance. SAM/ASAM remains third priority; pixel segmentation stays auxiliary-only.
- Post-hoc quarantine sensitivity kept the full 11-episode baseline at `.816667`. Quarantining `1sft` raised balanced accuracy to `.90`, but quarantining the explicit SK1 matched-positive also raised it to `.90`; this demonstrates severe single-positive influence rather than permission to cherry-pick. Visual review found `1sft` semantically ambiguous (wide path, persistent pedestrians, weak scale growth), so it is routed to independent policy/label review and same-type data expansion. The sensitivity report SHA256 is `b707bda5ebbc1073d43dea73e3b35e7dabb7af2626294b06ce03d7bbbe825841`, with `post_hoc_analysis_only=true` and no training-gate authority.
- Verification: 51 focused Python tests passed, the new scripts compiled, and `git diff --check` reported no errors. Existing LF-to-CRLF warnings were unchanged.

### 2026-07-17 — Public-silver r4/r5 mechanism-stratified root-cause continuation

- Isolation correction: confirmed that `secondary-corridor-causal` is a separate model direction. Removed the unrun cross-line replication artifacts created earlier and kept all subsequent data, scripts, reports and gates inside the public-video silver mainline. The new r4/r5 builders and depth/mechanism audits reject any input or output path containing that independent direction; no independent data, metrics, weights or promotion decisions were consumed.
- r4 data expansion: built an immutable child of r3 with the SANPO `gie8` timeline split into a passable early static-furniture episode and a near-field narrowing episode. r4 has 12 non-abstaining episodes, 6/6 class balance, three explicit matched pairs, no cross-source frame-hash reuse and all package validations green. The generic data gate passed.
- r4 diagnosis: frozen object trajectories remained linearly separable at balanced accuracy `.75`, but the new pair reduced prototype-direction alignment and the five prototype/bootstrap runs fell to `.6667` median with only `2/5` passing. Frozen DINO reached `.6667`; a new explicit relative-depth corridor-profile probe reached only `.50`. The five-run risk/lifecycle MIL median was `.5833`, with `.6667` median pair-order rate. This rejects the simple fixes “use a stronger pooled backbone”, “add hand-built depth occupancy” and “tune the tiny head”.
- Targeted acquisition: downloaded ten MD5-verified public SANPO RGB frames from `Chcne...` source frames 117–252, with no masks or human-truth claim. Visual temporal review retained only the stable 177–207 passable interval and 222–252 near static chokepoint interval; the original later 252–387 turning window remains excluded. The new pair stays GPT-5 model-provisional supervision and cannot calibrate, blind-evaluate or replace the production model.
- r5 mechanism expansion: added Chcne as a second independent static-corridor matched source, producing 14 non-abstaining episodes, 7/7 class balance and four explicit pairs. Frozen trajectory balanced accuracy recovered to `.785714` (`[[5,2],[1,6]]`). Five prototype/bootstrap runs were `.6429/.7857/.6429/.7857/.9286`; median `.7857`, but only `3/5` passed and the stability gate remained false. Both static-source no-alert episodes (`gie8`, `Chcne`) were still false alerts under source holdout.
- Lifecycle result: the r5 weak MIL runs were `.7857/.8571/.7143/.6429/.7857`, median `.7857`; matched-pair alert-over-no-alert ordering was `.75/.75/1/.75/1`, median `.75`. This is a material improvement over r4 and supports mechanism-matched data expansion, but the worst seed `.6429` remains below promotion quality.
- Governance fix: added `audit_public_silver_mechanism_coverage.py`. The default hard gate separately requires at least three independent matched sources for `dynamic_agent_approach` and `static_corridor_narrowing`. r5 has only `2/2` for each mechanism, so the mechanism coverage gate correctly fails even though the old overall pair-count gate passes. Next action is data expansion, not SAM/ASAM.
- Evidence SHA256: r5 trajectory `36b80d545c09af9fe2594bd580da8d18cda3a4b9f691c6229e736f3f12c6f5be`; r5 short runs `453d605012402532e21b8499d5a9e4b5e5f79c0cda363b349b2d89ddf377cf72`; r5 MIL `b5171d4b649aede351fb23156c678d3971c6ca5cbdf766e1eedf39a91556429e`; mechanism coverage `bf5c267194ff5c8ef9f25d57ec125a90444a06db0ce06867059e775f7608a032`. No weights were saved, no App/default model changed, and production replacement remains unauthorized.

### 2026-07-17 — Public-silver r6 low-confidence dynamic-pair diagnostic

- Scope/isolation: built immutable child `public-video-provisional-training-r6-20260717` from r5 and a new consolidated RGB-only timeline `sanpo-weak-jtmy-counterfactual-20260717`. It copies three existing official-train JtMY frames from 226–300 and three from 339–429; no masks, independent-model data, weights or metrics were consumed. All new paths fail closed on `secondary-corridor-causal`.
- Label contract: added an early passable approach episode and paired it with the existing horse-carriage crossing episode. The original alert confidence `.63` was retained. The mechanism audit now requires every episode in a counted pair to meet confidence `.65`; JtMY is therefore exploratory and excluded from qualified coverage rather than used to make the gate green.
- Coverage: dynamic `all=3`, confidence-qualified `2`, independent sources `2`; static `all=2`, qualified `2`, independent sources `2`. The generic aggregate readiness gate passes, but the mechanism+confidence gate remains authoritative and fails. No optimizer escalation or model promotion is authorized.
- Results: frozen object trajectory balanced accuracy `.732143`, confusion `[[6,2],[2,5]]`, JtMY source-holdout pair correct, but pair-delta mean cosine `-.00651`. Five prototype/bootstrap runs `.6696/.75/.9375/.6696/.7946`, median `.75`, only `3/5` pass. Weak lifecycle MIL runs `.8661/.8036/.7946/.8661/.8661`, median `.8661`; pair-order median `1.0`, with JtMY correctly ordered in all five seeds. This strengthens the lifecycle-head direction while leaving data coverage red.
- Sensitivity: quarantining the JtMY alert lowers balanced accuracy by `.0863`, while quarantining its early no-alert raises it by `.0536`; the dynamic signal is useful, but the negative interval is distributionally different enough to require better same-route evidence.
- Evidence SHA256: build `c6652fc72db9c8c53eba36d51412272cd79ff611cd87e55be50b3c36057e8456`; mechanism coverage `ead765a49f94ea63d9b439ecdf42366bcf56ea4593a7759ba54a923af8149490`; trajectory `c035f3889635151e7033e0e4a70c3da83dbfa0ae120392d315bf51c8a2afb519`; short runs `30d9b1b6698abd94ea2407df30b03f7812d0a579f8d13dc3b1eaf48ebe2c49e0`; MIL `1fa5d796d0c90c1d2532821877a0aea244705e3f5b1b3a5f8df7d0294b63a7f9`; sensitivity `660318742d894cbf96696f374a4e79d380fab10a3ae1662f0e5479cddfc5e2af`. No weights were saved and the App/default model was not touched.
- Confidence-weighting OFAT: extended the same MIL head with an optional class-normalized linear confidence loss weight. The equal-loss rerun exactly reproduced the original five metrics. The weighted run produced the same balanced-accuracy vector `.8661/.8036/.7946/.8661/.8661` and pair-order vector `.8/1/1/.8/1`; the largest episode-probability movement was about `.01`, with no systematic correction of static false alerts. Stop this optimization branch and retain confidence for gating/review routing only. Equal/weighted report SHA256: `e724d0493191d2aa8a37ddd30dc21b90d16312c54e0839d20dd8a556359ca33d` / `bddc0ed8e269a96e9d5d963096cbf4832063cad5f1c6d15ed2d42507b4c4b01b`.

### 2026-07-17 — Public-silver r7 reviewed-Wikimedia mechanism closure and representation diagnosis

- Isolation: kept `secondary-corridor-causal` fully independent. The r7 builder rejects any path containing that direction, and this run consumed none of its code, data, labels, weights, metrics or outputs. Only method-level lessons may be referenced.
- Source acquisition: used the Wikimedia Commons page for POPtravel's first-person Bangkok Sukhumvit walk, licensed CC BY 3.0 and carrying a YouTubeReviewBot license-confirmation record. Downloaded the 240p WebM transcode to `artifacts.local/downloads/`; size `241402495` bytes, SHA256 `8f0efe24eddd939e8396abc60cfa35789003e9a3b9f115b9538182d0060e6a17`. No source masks or external event labels were used.
- Governance: extended the v2 silver validator so CC BY 3.0 is accepted only with bound review status, Commons file page, original source, author, review time and exact license URL. Added positive and negative tests for reviewed/unreviewed CC BY 3.0.
- r7 build: added a deterministic builder and tests. It verifies the exact video SHA, extracts 12 preregistered frames, writes a hash-bound source manifest and four provisional episodes, then creates immutable child `public-video-provisional-training-r7-20260717`. The pairs are driveway-clear → near-field van crossing and sand-pile blockage → clear after passing. Labels remain GPT/VLM provisional and cannot authorize calibration, blind evaluation or production replacement.
- Coverage: r7 has 19 non-abstaining episodes and seven matched pairs. Confidence-qualified dynamic and static mechanisms each reach three independent pairs/sources, so the mechanism gate passes for the first time; low-confidence JtMY remains excluded.
- Frozen diagnosis: object trajectory balanced accuracy is `.738889`; pooled DINO/Depth is `.361111`; explicit relative-depth corridor profile is `.522222`. The latter two fail, so neither generic foundation pooling nor the current hand-built depth profile fixes the new mechanisms.
- Head tests: prototype/bootstrap balanced accuracy `.6833/.7444/.7889/.5833/.6778` (median `.6833`, `2/5` pass). Weak lifecycle MIL `.7222/.6722/.7389/.5833/.7833` (median `.7222`), with pair-order median `.714286`. No weights were saved.
- Failure localization: the van-crossing alert has only one persistent COCO trajectory; the sand-pile alert and post-pass no-alert have identical detection/track counts. Source-holdout trajectory predictions for the four new episodes are `1/1/1/0` versus expected `0/1/1/0`. Quarantining the driveway-clear episode raises balanced accuracy by about `.15`, which routes it to representation/semantic review but does not authorize deletion or relabeling.
- Decision: the minimum data-mechanism gate is now green, while the feature contract remains red. Stop optimizer and SAM/ASAM escalation. The next isolated mainline prototype must represent near-field occupancy, static surface obstruction and temporal occlusion without importing the independent direction's implementation or artifacts.
- Evidence SHA256: build `dc0cbe6de714e3ac67bef5fdc3bf4ae8e2c9b85cbbd39c42da43e866e1ac933a`; mechanism `59dfa67b7bab5a07c6ae7a6464e26081004d6030e5c90b93d869888356dc8e52`; trajectory `0a460982ea9b703b33905613fb0abe523dba6b8afcf942cff3cd0dbb01bf9162`; DINO `16a940f949ee04512c08f090ed004d7151d27e26e7c8b7b3e3fa25a5f2861120`; depth corridor `5034cc56309155d0c1a3e94ab60e6b28456d71d7c9b9fef85ef08add971e7f9c`; short runs `2ab461a3537843ff8a85d1c4877be85dddcc18fca2811146635e1f47e5893cbc`; MIL `d047c04ab1cb0094b3add9e8e45cf4613de67d3ed2f6ce8174cb4f805ddd4515`; sensitivity `f35378391948da77a0df2109c383684f06f8e1f3a1265baab548b273911c8f02`.
- Verification: 51 focused Python tests passed with the repository's `scripts` import path; related files passed `py_compile`; every r7 JSON sidecar matched its SHA256; the r7 package contained no `secondary-corridor-causal` reference; `git diff --check` reported no errors beyond the repository's existing line-ending warnings.
- Follow-up frozen representation OFAT: independently added `run_public_silver_free_space_topology_probe.py`. It consumes only the current mainline frozen SANPO segmentation logits, traces an adaptive walkable path, summarizes width/bottleneck/lateral shift/nonwalkable class probabilities, and evaluates both topology-only and trajectory+topology fusion under source holdout. Pure tests cover full blockage, lateral rerouting, temporal sensitivity and independent-path rejection.
- Result: topology-only balanced accuracy `.522222` (`no-alert=.60`, `alert=.444444`); trajectory+topology `.627778` (`.70/.555556`), below the trajectory-only `.738889`. Counterfactual delta mean cosine is `.034916` for topology and `-.027240` after fusion. Every new Wikimedia episode has minimum walkable width `0` at the fixed `.50` threshold, including the post-sand clear segment, so the current segmentation logits are not calibrated for free-space topology. This branch is stopped without threshold or hyperparameter search and is not passed to a head.
- Topology evidence SHA256: `be519fc2034e72933bcb7140e80ca773bd2bcbd07f273199f0d8e0d5f39fd4a9`. Its sidecar matches, and the regenerated r7 evidence tree contains zero references to the independent direction name.
- Final verification: the expanded r4–r7 focused suite passed `55/55`; all related Python files compiled; every r7 JSON sidecar matched; the r7 evidence tree retained zero independent-direction references; and `git diff --check` remained clean apart from unchanged line-ending warnings.

### 2026-07-17 — Public-silver r7.1 train-only static counterfactual stabilization

- Isolation remained non-negotiable: `secondary-corridor-causal` was not read, modified, imported or evaluated. The mainline r7 evidence tree and the new controlled synthetic tree contain zero references to that direction. Builders and probes retain explicit fail-closed path checks.
- Added a motion-compensated lower-corridor occupancy probe. Motion-only balanced accuracy was `.516667`; trajectory+motion was `.683333`. A compact motion variant fused with trajectory reached `.738889` but produced the same predictions as trajectory-only. The residual exposed registration failure on the near-field van and larger alert residual ranges in static pairs, but did not add generalizable classification power. Report SHA256: `b26e18ea87e31bb1d2b5494c453098b3357a80c01f99740b23192331d7f754d6`.
- Added a mechanism temporal-range audit. Dynamic and static source-internal pair ordering each passed `3/3`; held-out dynamic endpoints passed `6/6`, while static endpoints passed only `4/6`. A single absolute static threshold is therefore rejected in favor of source-relative baselines and lifecycle structure. Report SHA256: `5b5f59422bc30f3d2d249cd39998b42c7b6777ba32d4e2fb99a07ac7b9f96d88`.
- Extended the weak lifecycle MIL head with registered-residual reliability, temporal change from the first reliable baseline, complete-pair logistic ranking loss, optional train-only augmentation and optional lower-corridor appearance statistics. The baseline+pairwise candidate produced balanced accuracy `.8389/.7333/.7389/.6833/.8389`, median `.7389`, minimum `.6833`, with `4/5` seeds at or above `.70` and pair-order median `.8571`. Report SHA256: `c8b7c28993913a6f7f8f84126a3505208287e71dd3c92808387571795cbc04e7`.
- Continued public-video discovery without forcing a real r8. The Novi Sad pillar interval was rejected due to people, a truck and turn/route-change confounds. The Trubarjeva roadworks clip was rejected because the camera was largely stationary and a white van crossed the view. Neither candidate entered training or evaluation.
- Built a controlled train-only static-counterfactual package from three exact real no-alert parent episodes. Clear frames are byte-identical copies; positive frames deterministically composite perspective-growing barricade or sand-pile cutouts. The package contains three pairs, six provisional episodes and 18 images: nine positive images with alpha-derived masks/bboxes and nine exact clear negatives. Manifest, YOLO and COCO checks passed. All 18 contact-sheet images were visually reviewed and accepted with no issue tags.
- Leakage control is code-enforced: every synthetic pair binds a `parent_source_id`; during real LOSO evaluation it is training-only and is excluded whenever that parent source is held out. Synthetic episodes are never included in confusion matrices, balanced accuracy, counterfactual metrics, calibration, blind evaluation or production authorization.
- The best candidate (`temporal baseline + pairwise ranking + train-only synthetic static pairs`) produced real-episode balanced accuracy `.7944/.7944/.8944/.7944/.7444`, median `.7944`, minimum `.7444`, maximum `.8944`, with `5/5` seeds at or above `.70`; pair-order median remained `.8571`. Relative to the no-synthetic candidate, median improved from `.7389` to `.7944` and minimum from `.6833` to `.7444`. Report SHA256: `4f8e68c04d5c40787fb5df262318c1a4bbd414c974660f2ab0f9203601f5eb3a`.
- The appearance-channel OFAT did not win: `.7389/.6889/.8389/.7889/.8389`, median `.7889`, minimum `.6889`. This branch is stopped without further feature or threshold search. Report SHA256: `d851a1b99e501a29373c39dbcd61d70a65a9ae6784445ffe9e351d26092f6040`.
- Remaining hard failure: the Bangkok sand-pile alert is correct in all five seeds, but the post-pass clear episode remains a false alert in four of five. The current result closes minimum head stability only; it does not close obstacle-exit/event-termination semantics and does not authorize saved deployment weights or production replacement.
- Synthetic build receipt SHA256: `79273244d32694ab9bd566eb48969b5d40862ba89c367cc75f0fc63978cabae6`. Manual visual-QA receipt SHA256: `f30494a89c949e31d10984cef83c1747d7b2ea0d651506119ef9c2db1098f02e`.
- Verification: the expanded focused suite passed `74/74`; all related implementation and test files passed `py_compile`; `18` JSON sidecars across r7 evidence and the controlled synthetic tree matched their payloads; both artifact trees retained zero independent-direction references; and `git diff --check` reported no errors beyond existing line-ending warnings.

### 2026-07-17 — Public-silver r7.2 causal static-event exit

- Root-cause refinement: the Bangkok post-sand clear episode has low first-frame risk in every seed, a spurious middle-frame peak from generic object/street-scene features, and partial terminal recovery. Smooth-max retains the stale middle peak. The source frames visually confirm that the forward sidewalk is clear after the pile is passed; this is not an obvious relabeling case.
- Strict-terminal negative control: added a selectable `terminal` episode pooling mode with exact one-hot gradient tests. Five runs fell to `.6278/.6333/.7833/.6833/.6333`, median `.6333`, with pair-order median `.7143`. Current-state-only pooling loses dynamic approach evidence and is stopped. Report SHA256 `ef4f54bef65173ff6d7c1198eca1522d06c633e3ea45ad877d78062be28ae81a`.
- Synthetic-mask teacher negative control: added a frozen dense DINO teacher trained only on matched composite-mask patches versus the same exact-clear locations. Parent-matched synthetic records are excluded in every real source holdout. Real balanced accuracy was `.5778`; the Bangkok sand/clear direction reversed. Do not attach this teacher to MIL. Report SHA256 `6aa21bc02b29907d9dcde632f070550f495c5bb7f3275fffd559efa3309da319`.
- Downloaded and hash-bound Ultralytics YOLOE-11s prompt-free segmentation weights (`292bdf157a9ec7315f34b567cb93467c5043cd1889a1cc18abbfdeB88d7a948d`). The model uses its fixed 4,585-class vocabulary and no per-run text prompt, silver label, source mask or synthetic image. It detected `sand box` at `.5845` in the nearest pile frame and no surface-material object in any of the three clear frames.
- Direct semantic fusion negative control: semantic-only balanced accuracy was `.6111`; trajectory+semantic was `.5778`. Sparse static semantics should not be concatenated into a global linear head. However, the held-out Bangkok source was `0/1/1/0` under fusion, correctly separating all four episodes. Report SHA256 `041a97745832e0457c8ffa1bf607e020d9ad56586ae59ef8544939a11a0301e2`.
- Added a zero-learned-parameter causal exit router. It closes a static surface event only when the immediately preceding same-source episode has a surface-material detection, the current episode has none, the gap is at most 5 seconds (or three manifest indices), and the independent source-isolated trajectory probe predicts no current dynamic hazard. Router conditions never read the current label and contain no episode-ID exception.
- The router found exactly one candidate: the sand-pile alert followed by the clear episode after `2000 ms`. Five balanced-accuracy values improved from `.7944/.7944/.8944/.7944/.7444` to `.8444/.8444/.8944/.8444/.7944`; median `.8444`, minimum `.7944`, every seed non-degrading. Post-event clear became correct `5/5`; no-alert recall rose by `.10` in four seeds and alert recall remained `.8889`. Report SHA256 `80b4f4badad700e5931d4f4aace88cc49f756f65d016ac4009b237d28a176697`.
- A preregistered 1-second gap negative control produced no exit candidate and exactly reproduced the un-routed metrics. This confirms the gain depends on the observed 2-second causal continuity rather than unconditional suppression. Negative-control SHA256 `2d5d6c0bd18894e5c5da9a8b1240bd8d1869e0f49591a2ef42c23ba406e105f5`.
- Decision: the known post-event clear failure is resolved by a lifecycle router rather than a different optimizer or global feature concatenation. This remains a one-source provisional prototype. Do not modify Android `RiskEventTracker` or claim general event-exit closure until at least two to three additional independent continuous exit sources pass the same frozen contract.
- Verification: the expanded focused suite passed `86/86`; all related implementation and test files passed `py_compile`; all `23` JSON sidecars across the r7 evidence and controlled-synthetic trees matched their payloads; both artifact trees retained zero independent-direction references; and `git diff --check` reported no errors beyond existing line-ending warnings.

### 2026-07-17 — Public-silver r7.3 cross-source exit discovery audit

- Added four license-bound Commons walking sources (Greenwich, Shanghai, Harbin and Worms), about 112 minutes total. A 5-second 320px prompt-free scan covered 1,346 frames and proposed 92 exits (6 surface, 86 barrier). GPT multiframe review rejected every surface candidate and the top barrier candidates as ordinary buildings, fences, fixed entrances, vehicles, stairs, crowds or texture errors.
- Added hash-bound persistence re-filtering. Ten seconds of consecutive absence reduced 92 proposals to 77; twenty seconds reduced them to 62, while all six surface false positives survived. A 640px, `.05` confidence, 10-second persistence rescan reduced the set to 51 (3 surface, 48 barrier), but the remaining reviewed surface and top barrier proposals were still false. Broad YOLOE-only discovery is stopped.
- Added an optional near-field box gate as a negative experiment. It removed valid construction-boundary detections and retained low-confidence noise, so it is not adopted. Future prompt-free reports now attest input size and Ultralytics/Torch/OpenCV runtime versions.
- Built a deterministic 301-frame temporal-reversal counterfactual from a CC BY 4.0 Hof bus rear-window construction video. Raw adjacent absence produced three proposals; ten consecutive one-second absent samples retained only the `18s -> 19s` boundary, matching GPT's roughly `19s -> 21s` visual-clearance judgment. The artifact is discovery-only and excluded from real evaluation and training because its viewpoint is not pedestrian first-person and time direction is synthetic.
- Added persistence refilter, temporal-reversal builder, overview/review artifacts and pure tests. No Android runtime or default model was modified. The independent `secondary-corridor-causal` direction remains unread and untouched.

### 2026-07-17 — Public-silver r7.4 real-source exit challenge and risk-profile router

- VLM-first review found a new self-published CC BY 4.0 Hof bus-window construction clip in original temporal order. Construction is visible through six seconds, leaves the view at seven seconds, and remains absent through ten seconds. The downloaded 240p video SHA256 is `1d9dff54e8ff89c4f40b66f29818eddca5948c564ef64c2f92a02d09efcc9e4c`.
- A frozen YOLOE-11s prompt-free scan at 640px, `.05` confidence and one-second sampling detected `construction site` through six seconds (`.5028877` at the final present sample) and no selected semantic object in four consecutive samples from seven through ten seconds. GPT timestamped multiframe review accepted `6s -> 7s` as a discovery-only exit boundary with `.92` confidence.
- The clip exposed a structural failure in the original surface-only exit idea: a transient `sand bar` detection disappears at two seconds even though the construction scene continues to six seconds. With three-sample persistence, the isolated external challenge rejected surface-only at `1s -> 2s`; barrier-only and the union of surface plus barrier risk groups both matched `6s -> 7s`. Challenge SHA256: `7c4cdc7b724ee6a3a1ef326c8e1f81d5231b724e6601bb4f4eda3a8ab83195d8`.
- Added a separate risk-profile router experiment without modifying the r7.2 v1 router. The main r7 replay still produced exactly one candidate, Bangkok post-sand clear, and preserved the routed five-run balanced accuracies `.8444/.8444/.8944/.8444/.7944` (median `.8444`, minimum `.7944`, all non-degrading). Report SHA256: `d18b89e6a70d392fa842874eb2adf6e61ec9990bca844a336cb94be1b1d9eb3f`.
- The one-second-gap negative control produced no candidate and exactly restored the five baseline values `.7944/.7944/.8944/.7944/.7444`. Negative-control SHA256: `ffc79d68561e3a8d30e6abf7594b463bd28cf0b63aea5652ef06e3f653390668`.
- Decision: the risk-profile union is a stronger lifecycle prototype than surface disappearance alone, but the Hof capture is a lateral bus view rather than a pedestrian forward corridor. It remains external discovery evidence only. No training set, Android `RiskEventTracker`, runtime, or default model was modified; calibration, blind evaluation and production replacement remain unauthorized.

### 2026-07-17 — Public-silver r7.5 licensed-source acquisition gate

- Reviewed two CC BY 3.0 Ljubljana worksite videos and one CC BY 3.0 Vimeo Gympie site-visit video. The Ljubljana clips were static or panning soundscapes; Gympie was a hard-cut montage mixing road, walking, aerial, machinery, plan and title-card shots. All three were hash-bound in an acquisition rejection report and excluded from exit evaluation and training.
- Generalized the public discovery registry to v2 for non-Commons platforms while preserving v1 normalization. Added a bounded Vimeo CC-BY search ledger plus PowerShell single-request wrapper; it never paginates or logs in, records the raw response hash, and marks every search result ineligible for training until item-level license and continuity review pass.
- The only strict Vimeo search result was licensed CC BY but semantically irrelevant: walking and roadwork were generic description examples rather than the depicted task. It was rejected before download.
- Public YouTube search produced a high-fit 22:50 Addis Ababa construction-corridor walking tour. Logged-in Chrome confirmed the continuity claim, but no reusable video license was established, so it remains `hold_no_download`. A separate 90-second walk beside a construction site was rejected after the expanded description showed that CC BY 3.0 applied only to music while the footage itself carried a copyright assertion.
- Added machine-readable triage with SHA256 `8a6c22482e590a3f030d769b5da00ce2aa3372e863ce2cafbf8d8cef9a3541c4`. The new qualified pedestrian-exit source count remains `0/2`; provisional training, Android runtime changes and `RiskEventTracker` integration remain closed. No independent-direction artifact, code, data, weight or metric was read or modified.
- Verification: the new Vimeo ledger pure suite passed `6/6`; the PowerShell wrapper parsed cleanly; the registry v2 tests had previously passed `7/7`. Further online discovery paused when the execution approval service reported its usage limit, without using an alternate path to bypass it.

### 2026-07-19 — Public-silver r7.6 frozen-DINO retrieval negative experiment

- Added a deterministic zero-trainable-parameter retrieval probe from the frozen Depth Anything V2 DINO-S encoder. The direction is defined only by Hof present frames at `0/3/6s` versus clear frames at `8/9/10s`; it scanned the four licensed continuous walking videos every five seconds, covering 1,346 frames.
- Raw direction projections were not treated as cross-source probabilities. Candidates were ranked only by within-source robust-z sustained drops, then expanded into two-second multiframe review windows.
- GPT multiframe review rejected all 12 top windows. The dominant failure modes were indoor/outdoor domain shifts, close objects and shop displays, historic masonry/rock texture, fixed gates/gardens, and camera turns. No window represented a continuous construction-or-obstacle risk exit.
- With `0/2` accepted independent sources, downstream risk-profile/persistence/trajectory challenges were intentionally not run. No training record, weight, Android runtime change, `RiskEventTracker` change, calibration or blind claim was produced. Review report SHA256: `05b5fe9637598144fce86a7fb5fb839baccf8714e4f8520ff1d33f6908d05112`.
- Decision: do not rescue the failed retrieval by post-hoc DINO/YOLOE union weights. The bottleneck remains acquisition of item-level reusable, causally continuous pedestrian-view sources, not head optimization or SAM/ASAM.

### 2026-07-19 — Public-silver r7.7 Fremantle long-work-zone lifecycle challenge

- Reused the already licensed CC BY 2.5 AU Fremantle original-order dashcam source as a second external mechanism challenge, without counting it toward the pedestrian `0/2` gate. Dense GPT frame review placed the visual cone-channel exit at `178s -> 179s`, followed by a stable clear road through 195 seconds.
- A one-second, 640px, `.05` frozen prompt-free scan covered 262 frames. Barrier-only and risk-profile union each emitted 10 exits: eight premature, two late and zero exact matches; surface-only emitted none.
- During the reviewed risk-present interval `150–178s`, barrier evidence was active for only `2/29` samples (`.06897`) with a 20-sample absent run. The stable-clear interval `179–195s` still contained one false `construction site` activation. Challenge report SHA256: `8e37b59d75ca2c2985c06e4cd826491bebb59eaa0d4fc05415744953b277a7a4`.
- This falsifies the assumption that consecutive semantic absence is positive clear evidence. A future lifecycle prototype needs explicit `present/uncertain/clear` state and an independent clear/traversable signal; extending a timeout alone would preserve both the long in-risk dropout and the post-exit false activation. No training, Android runtime or `RiskEventTracker` change was authorized.

### 2026-07-19 — Public-silver r7.8 work-zone markers and tri-state lifecycle prototype

- The frozen prompt-free vocabulary contains direct work-zone classes that the original preregistered barrier subset omitted: `barricade`, `cone`, `construction worker` and `traffic cone`. Added an explicitly exploratory, default-off scanner flag that maps only these fixed built-in classes to barrier; the established baseline remains unchanged and no text prompt or trainable parameter is introduced.
- On Fremantle, risk-window activation improved from `2/29` to `23/29`; longest absence fell from 20 to 2 samples and raw exits fell from 10 to 3. The adjacent-exit rule still failed with one premature and two late candidates, including isolated post-clear activations.
- Added a zero-learned-parameter `present/uncertain/clear` external lifecycle evaluator. Entry requires `2/3` active samples; absence first enters uncertain; three consecutive absent samples confirm clear; a lone activation from clear cannot reopen an event. Fremantle produced exactly one event with `last_active=177s`, `first_absent=178s`, `confirmed_clear=180s`, containing the GPT `178→179s` reference. Report SHA256: `302152ec2cb3ab2287bec387fdbc064a447f0fe5987a3d14f3694b2d56ce5744`.
- Without changing parameters, the original Hof baseline scan also produced exactly one event with `last_active=6s`, `first_absent=7s`, `confirmed_clear=9s`, containing its existing `6→7s` reference. Report SHA256: `3f73e28d97278d68533ed434ec077939b3b8dc2a405c21c234e66e3a2a9de37e`.
- This is a two-source vehicle-view mechanism prototype, not prospective pedestrian validation. The marker expansion was proposed after inspecting the Fremantle failure, so the pedestrian `0/2` gate, training, calibration, blind evaluation, Android runtime and `RiskEventTracker` remain closed. The next licensed continuous source must freeze the marker set and `2-of-3 / 3-clear` contract before visual review.

### 2026-07-19 — Public-silver r7.9 prospective marker contract failed nuisance controls

- Frozen the full work-zone marker vocabulary and `2-of-3 / 3-clear` lifecycle before opening four nuisance controls. Harbin and Worms passed, but ordinary Greenwich gates and a Shanghai street turn created sustained false events; only `2/4` controls passed. Contract SHA256: `48a8319f61bb58f9e319460e5dbb655340f8729de06871718cb258550cce5fa1`; failure report SHA256: `b4ce3f45ec2791924492df35c0402dd5325d888a08ed733cd816fc4e8b97a038`.
- No leave-one-marker-out variant passed both Fremantle and all controls. The existing near-field corridor gate reduced Fremantle risk coverage to `.241` and still retained both false events. r7.9 remains a prospective failure and was not rescued by threshold or label edits.

### 2026-07-19 — Public-silver r7.10 multi-cone expert boundary

- A post-failure count audit tested `1–4` traffic-cone detections per frame. Only threshold `2` passed the Fremantle exit plus all four prior nuisance controls; audit SHA256: `517b17d00432c35b3f742f7aa752202f9e5adcdb27f01e5dc390fe0d32467ad7`.
- After freezing the r7.10 contract (`109521ccc692d5f711ebcabf2ed0f26b4694f6c42773f72e13921235fd29c35d`), four new nuisance windows passed `4/4` with zero events. A separate persistent-risk Hof clip then exposed the boundary: only `2/18` risk frames met the dense multi-cone policy and no event opened. Reports: `d81541eeefc43997ce26def42761364e66455f01e3387fec9182db19cd0ec348` / `3335a4e466c3e9b679c9406f64dc54aad9cc20a7408eacd09fd7926dea5a9c79`.
- Decision: retain multi-cone as a narrow mechanism expert only. Do not lower its count threshold or call it a general construction detector.

### 2026-07-19 — Public-silver r7.11 chromatic marker expert

- Extracted per-detection box color and geometry for true dense cones, sparse red-white delineators and fixed-gate false detections. Geometry and box shape did not separate them. The scale-free rule `high_saturation_fraction > dark_fraction` retained 75/115 Fremantle cone detections and 9/12 Hof sparse detections while rejecting 5/5 Greenwich false-cone detections; adding `barricade` brought Hof entry to 22 seconds without reopening the gate controls.
- Testing clear persistence `3–6` showed 5 and 6 pass the dense exit, Greenwich/Shanghai controls and sparse persistent entry; selected the minimum sufficient value 5. Audit SHA256: `1ace0742e75c63df14e34c60aef66e253c4c38d07c6a53d807b763f3d61f6065`.
- Frozen r7.11 contract SHA256: `3e6e6f410ce03053f4e8d6e38475156f69cf6a41352d8619541a7c8ae526bca7`. On four new nuisance windows, 190 frames contained 35 raw cone/barricade detections but the frozen chromatic expert produced zero events and terminal clear for all `4/4`; report SHA256: `c0c7dde0cdf0545a45d9a36fce2a971afe79ad02c0b7a931b87371e133bb746f`.
- This supports mechanism-specific positive evidence plus tri-state lifecycle, not global barrier absence. No prospective independent positive exit or pedestrian first-person source exists yet, so training, calibration, blind evaluation, Android runtime and `RiskEventTracker` remain unchanged.

### 2026-07-19 — Public-silver r7.12 source-lineage gate

- Added a SHA256-deduplicated inventory covering all 14 videos in the seven local public-video registries. The audit rejects registry omissions, duplicate bytes under aliases, video drift, and any source that influenced the frozen r7.8/r7.10/r7.11 parameters.
- Large-model review of the full local overview evidence classified Moira/Ljubljana/Anasskoko as static or panning, Duisburg/Vimeo as edited or non-causal, the four long walking sources as having no construction exit, Fremantle/Hof as derivation sources, and the longer Hof clip as persistent-risk-only.
- The enforced result is `0/1` independent held-out positive exits and `0/1` pedestrian-view held-out positive exits across 14 unique videos. Inventory/report SHA256: `2110df3b8973a68efa7f466cb2ffe2d7f2f8a4c60c281339b11054fa62af7090` / `9070787e41d701129ec56597a09cd78c0dce834e48ce9efeefaaab7fc99bb370`.
- Decision: the remaining bottleneck is source-level positive evidence, not source aliases or head optimization. Keep training and Android integration closed until a genuinely new eligible video passes the gate.

### 2026-07-19 — Public-silver r7.13 prospective-positive acceptance harness

- Added a fail-closed evaluator that binds the r7.11 contract, frozen chromatic feature report, r7.12 source-inventory audit and timestamped large-model review by SHA256. The review must attest that the policy and feature report were frozen before visual inspection.
- Acceptance requires an eligible held-out lineage, an accepted original-order no-cut review, exactly one lifecycle interval containing the visual boundary, risk-window activity at least `.4`, stable-clear activity at most `.1`, and terminal clear with no open event. Pedestrian eligibility remains a separate reported condition.
- Added a fixed review template (`07d872826ff68231e327c231dd52de2002b34b16256fc6bf45255673a0e355bd`) and eight unit tests covering the valid path plus derivation contamination, boundary mismatch, clear-window false activation, contract drift, feature-report substitution and hard cuts.
- No real report was fabricated because the r7.12 inventory contains no held-out positive source. The harness is ready for the first newly acquired item-level licensed continuous video; all downstream authorizations remain false.

### 2026-07-19 — Public-silver r7.14 pair-relative lifecycle probe

- Added a zero-trainable-parameter probe over the six qualified same-source matched pairs. It verifies sidecars and report lineage, orders episodes only by SHA-bound source `frame_index`, rejects overlap/cross-source/incomplete pairs, and maps the sign of the later-minus-earlier mechanism score to open/close/abstain without an absolute threshold.
- All `6/6` transitions matched the provisional episode chronology: five clear-to-risk opens and one risk-to-clear close across both required mechanisms. The Bangkok post-sand clear was correctly closed with signed delta `-.0012401`.
- The close evidence is fragile: its normalized margin is `.0412`, so a post-result `>5%` margin stress diagnostic abstains on that one transition while retaining the other `5/5` correctly. The stress rows are diagnostics, not calibrated thresholds.
- Final report SHA256: `421e2d5be2c7ca81991ebe36ccadc0033c4503515acb828345fbb0c63a51ec68`. The result supports a future mechanism-channel + relative-baseline + tri-state-lifecycle head, but requires a recent trusted reference state and does not solve arbitrary-frame cold start, continuous streaming, or cross-source positive validation. No Android runtime, `RiskEventTracker`, default model, calibration, blind or production state changed.

### 2026-07-19 — Public-silver r7.15 retrospective dynamic-close stress case

- Audited unused bound frames after each qualified pair. Only SK1 retained consecutive frames after its risk episode. GPT multiframe review was hash-frozen before detector scoring and accepted frames `5–7 risk → 8–9 clear` as a retrospective dynamic close stress case; review SHA256: `d620eb14494751ca61e9e97f591857e36e21a01c9cf6e33b5adbe3372dc15452`.
- The frozen YOLO12n/320/`.15` contract reproduced the published risk temporal-range score exactly (`.0349100`). The later clear window scored `.0123757`, yielding delta `-.0225343`, normalized margin `.6455`, and a correct close event. Result SHA256: `8e1d212893a3e77729b911b4d93d0d8588cef1586cbe68478ecea6c43a0bf6bb`; six pure tests passed.
- The review JSON contained a manually mistyped UTC value later than the inference output. The immutable review was not rewritten; erratum SHA256 `4458000e09c595e1a7b66cc456da55ab826330873ea5b3c204ea92023f099497` records that review and sidecar file writes preceded inference, while explicitly treating filesystem timestamps as supplementary rather than cryptographic evidence.
- This adds a dynamic-mechanism close stress case but not a new source or prospective sample. The r7.12 independent positive count remains zero, and no training, calibration, blind, Android runtime, `RiskEventTracker`, default-model or production authorization changed.

### 2026-07-19 — Public-silver r7.16 dual-evidence lifecycle fusion

- Added a zero-learned-parameter fail-closed fusion over r7.14 relative changes, the r7.15 dynamic close, the existing 5-second risk-profile semantic-exit router, and its 1-second gap negative control. It never uses an absolute scene threshold.
- A trusted clear reference plus a strong relative increase opens an event. A trusted risk reference plus a strong decrease closes directly. A weak decrease closes only when the frozen causal semantic router identifies the exact same source and previous/current episode boundary. Missing reference, zero change, missing corroboration and conflicting evidence remain uncertain.
- All seven available transitions passed: five strong opens, one strong dynamic close and one weak semantic-corroborated static close. Four fail-closed controls also passed, including removal of the Bangkok exit under the 1-second gap contract.
- Report SHA256: `b0f0a4c46bd2a0de6df6ba9a859811280c594898017e47aa98f2d21172acddc7`. The `.05` strong-margin boundary came from the already-published r7.14 post-result stress grid, so retrospective acceptance is true while prospective acceptance remains explicitly false. No training, calibration, blind, Android runtime, `RiskEventTracker` or production authorization changed.

### 2026-07-19 — Public-silver r7.17 frozen prospective dual-evidence contract

- Frozen the dual-evidence lifecycle policy before opening any new source visual content. Contract SHA256: `e7439ab3beac677ac913a0bb51155378ce2b2898c61dc4c38399c31235cd6175`. It discloses r7.16 as post-hoc and fixes the `.05` strong margin, model hashes, input sizes, semantic groups, 5-second corroboration gap, three states, review sample counts and all authorization flags false. The post-clear dynamic guard is recomputed from frozen occupancy peaks; feature reports cannot inject a hazard verdict.
- Added a prospective evaluator that requires complete-video one-second features to be frozen before original-order large-model review. The reviewer may select only pre-clear, risk-present and stable post-clear windows; open, close, stability and controls are recomputed from immutable features.
- Added a bounded-batch full-video extractor for frozen YOLO12n corridor occupancy, registered lower-corridor residual and prompt-free semantic counts. It follows the half-open one-second schedule, binds video/model/contract hashes, never receives review windows or labels, and emits no hazard or lifecycle verdict.
- The evaluator rejects contract/feature/inventory substitution, feature generation after window selection, irregular or missing samples, invalid hashes, hard cuts, ineligible lineage, boundary mismatch, weak closes without semantic corroboration and stable-clear reopening. Five extractor, seven contract and thirteen evaluator tests passed, including a real temporary-sidecar `run()` and output-hash path.
- Fixed review-template SHA256: `03923a78b3dd48bee49581d36f2fbe2ac8d2110a1e8f49c2fd3ee82742b48ffa`. No real result was fabricated because no r7.12 source is both new and independently positive. Training, calibration, blind, Android runtime, `RiskEventTracker`, default-model and production authorization remain unchanged.
- Executed a real engineering smoke on the known-ineligible 40.117-second Hof bus source: 41 scheduled samples across multiple batches completed frozen YOLO12n, YOLOE and motion compensation. Feature report SHA256: `95f2bb6f8bc60f68622b4ecee57669b2850cde128bb39d49da8bc320b7b2a031`.
- A deliberately invalid pre-contract review was then rejected before lifecycle scoring with a structured chronology error; neither result nor sidecar was created. Smoke audit SHA256: `9e0897599d2057411f1c606982213422b118878b9dc9f748bfc5190db9a9f59e`. This verifies execution and provenance rejection only; the independent positive source count remains zero.

### 2026-07-19 — Public-silver r7.18 first real pedestrian prospective challenge failed open

- Resumed licensed-source acquisition after network recovery. Registered every source before download or visual review. A CC BY 3.0 Commons clip was rejected as near-static with no lifecycle; a Pexels-licensed continuous POV clip was rejected because construction context persisted and road traffic did not create a pedestrian-path exit.
- Found YouTube `TVCX9tpaty8` through the Creative Commons search filter and confirmed item metadata `Creative Commons Attribution license (reuse allowed)`. Downloaded format 18: `31,679,092` bytes, SHA256 `551f8483e112b38afd0a91840fb56c8539cfcbe3a6475465677a342c695e63ec`.
- Before any visual window selection, extracted all 457 scheduled one-second samples with the frozen r7.17 YOLO12n, YOLOE and registered-residual channels. Feature report SHA256: `9ff5ecbe0abba930463cc24c4679a2419a53364f0df472008df706c7743504e4`.
- Full 5-second review found no visible hard cut. Dense 1-second review selected `340–345s clear`, `365–371s concrete-pipe near-field risk` and `378–389s stable clear`. The candidate-only lineage audit passed `1/1` independent and `1/1` pedestrian held-out positive; SHA256 `7847f19e6494dda6af56c4a4ee4f8b101664009d16044437b151090d57d3bb50`.
- Frozen evaluation failed only `strong_open_passed`. Static residual range was `.3058160` in visual clear versus `.1055915` in pipe risk, producing normalized open change `-.6547222` and `uncertain`. Strong close (`-.6958017`), stable no-reopen, lineage, boundary and all three controls passed. Result SHA256: `633591eafd019e2e5d6ae91c63268bde666de4564465798d7364884dcd3aae96`.
- Diagnosis: this is prospective evidence of a static-feature robustness failure under camera rotation/compression, not a head-optimization failure. The result is not rescued by window edits, margin changes, SAM/ASAM or post-hoc calibration. Training, calibration, blind, Android runtime, `RiskEventTracker`, default-model and production authorization remain closed. Consolidated audit SHA256: `3cd003c196de87b94d4b5afe209f247a54b3bf64f12a572ba8e27f31816520fe`.

### 2026-07-19 — Public-silver r7.19 static-representation diagnostics

- Added a zero-parameter background-normalized registered-residual probe with robust median/q75 pooling. Five pure tests passed. Rice risk-versus-clear ordering passed, but no candidate also ordered all three legacy static pairs; report SHA256 `7a15d0e0443b359c6db0b23a4c6dd1c2dda0e0b1c3a5f63af8436e848bc695c9`.
- Installed project-local Transformers and downloaded the frozen public `nvidia/segformer-b2-finetuned-ade-512-512` teacher under ignored `artifacts.local/`. A no-threshold soft free-space probe ordered Chcne and the sand pile correctly but reversed gie8 and ranked Rice pre-clear as more nonwalkable than pipe risk; report SHA256 `26ef0ea50707d94a7be5e2ec6faeb6a35f571d0103c4bea32c8362b51fda4f8f`.
- One structural follow-up used a fixed continuity-constrained adaptive path over the same soft map without changing labels or fitting thresholds. Four pure tests passed, but no feature ordered all legacy pairs plus Rice open/close; report SHA256 `642ad6b6ce6e266f76d0367d68896092b2bf463e584820b0a1f8a120abc24be7`.
- Decision: stop post-hoc mask/class/threshold and optical-flow searches. Resume independent matched-source acquisition and preserve multi-channel, mechanism-routed risk profiles. No training, calibration, blind, Android runtime, `RiskEventTracker`, default-model or production authorization changed.
- Acquisition resumed with item-level license fail-closed behavior. Eighteen new YouTube candidates had no explicit CC license metadata and were not downloaded. Two CC BY 3.0 Commons Ljubljana worksite videos were registered before downloading only 360p transcodes; 5-second overview triage found near-static/across-street cameras, persistent construction, no pedestrian forward progress and no clear-risk-clear lifecycle. Both were rejected before feature extraction; triage SHA256 `d6ab3842d651ed3576fa55ae341c0754319e93edef08d4aaa03433377aee90e1`.
- Added `run_public_silver_multichannel_risk_profile_probe.py` and four pure contract tests. The fixed 16-channel profile covers registered local change, absolute clearance, adaptive-path occupancy and detour offset; it is evaluated alone and concatenated with the unchanged 163-D frozen trajectory vector without feature, mask, class or threshold search.
- On the same 19 episodes, source-isolated balanced accuracy was `.738889` for trajectory-only, `.422222` for profile-only and `.738889` for fusion. Profile/fusion matched-pair mean cosine was `-.062585/-.026414`; Rice profile-only open and close projections both failed. The fusion did not strictly improve the trajectory baseline, so the preregistered feature gate failed and five prototype/bootstrap runs were not launched. Report SHA256: `d370192ca5ca81d1693eb26274e43daa859210251afd2983d1ead561be005db1`.
- After connectivity returned, Vimeo's CC-BY endpoint still returned HTTP 500 for three queries. Item-level YouTube checks over 30 results produced one explicit CC candidate, `SI7uinNg7jk`. It was registered before downloading a 360p copy, then rejected before feature extraction because the 10-second overview showed repeated multi-camera montage cuts rather than one ego-pedestrian clear-risk-clear episode. No training, calibration, blind, Android runtime, `RiskEventTracker`, default-model or production authorization changed.

### 2026-07-19 — Public-silver r7.20 mechanism diagnosis and static counterfactual entry

- Added a nested source-isolated mechanism-routed expert probe over six qualified pairs. Neither observable routing nor oracle mechanism labels produced a stable gate: best routed balanced accuracy was `.6667` for trajectory, `.5000` for risk-profile and `.5833` for fusion. Report SHA256: `8c29ea04130b2af9a65a407c58d9a318586166b13a8d0ff8320eaf5c531f5f7a`.
- Created an ignored train-only photo-real static counterfactual dataset with six independent real clear parents, two sources for each of construction pipe, temporary barrier and surface debris, and twelve final 1280x720 images. Natural-language generation geometry was not treated as verified object geometry: annotations intentionally contain zero bbox/mask objects, while pair labels and full parent/source/image SHA lineage are retained. Model review SHA256: `522c6a98b90370b42f21081f95740b2d68cac193d7ef0332bf4b3182eba953de`.
- Added a schema-v2 provenance/response auditor. All train-only, hash, parent-source exclusion, family coverage and no-pixel-truth checks pass. One visually accepted barrier is a frozen-teacher hard nonresponse and remains in the data rather than being filtered by teacher agreement. Audit SHA256: `2e7dcbcdb30278098aee2d33669a11b7b0d3b38a912d243d846560b168cd4b8d`.
- Added parent-source-isolated fixed pair-delta and full ADE semantic adapter probes. Six synthetic pairs improved legacy real static ordering from `1/3` to `2/3`, but Rice open remained false; SHA256 `eb2e574bc4c86d2d7aaa454c990c240dbc9346dc9062511f2610bd159d9578bd`. The 450-D semantic adapter passed Rice open/close but only `1/3` legacy pairs; SHA256 `1d5dc9f2700c8cdf179e4a5bdd11574d6ffc02f215acbdab8f0490ea93a0d42b`.
- The apparent complementarity is post-hoc and was not promoted into a fusion rule. Next work is actual representation-adapter training with expanded hard counterexamples and strict parent-source exclusion, followed by a newly frozen prospective source. Android runtime, default model and production gates remain unchanged.
- Ran exactly one fixed nonlinear representation-adapter short run: seed 0, four softplus positive-evidence units, fixed unweighted readout, 300 steps, frozen teacher, no hyperparameter sweep and no saved weights. Source-isolated real ordering remained `1/3`; Rice open failed and close passed. Report SHA256: `b8630d4c08823b3bce026f9db77203e4a9d61c17c1e06b564e32f41ef1ffeb7c`. The five-seed bootstrap was therefore skipped by contract.
- Began an inverse-counterfactual batch after network recovery: licensed real obstacle photos remain the risk endpoints and GPT image editing removes only the obstruction to create provisional clear endpoints. The first two source-isolated pairs cover a discarded sofa (CC0) and a utility pole blocking a narrow sidewalk (CC BY-SA 4.0). Four image hashes and source/license lineage validate; manifest SHA256 `fbf3a2ab6b7df429823a3f23fb68703847cdd23276672000957287808826639b`, model-review SHA256 `dbf60fd344fa67da98f1e1c495d7446245a0bb516bb63ab286645b7e3c656292`. This is train-only ranking/classification evidence with no bbox, mask, distance-field, evaluation or deployment authority.

### 2026-07-19 — Public-silver r7.21–r7.23 frozen DINO and multi-expert lifecycle

- The two inverse pairs passed provenance, license, hash, no-pixel-truth and parent-source isolation audit; SHA256 `784adac3c5b3d707c2159ec29e13ef174307235ab56dbd495aec2f494f4fc943`.
- A fixed 1920-D DINOv2-S regional pair representation passed all three legacy real static pairs and Rice open/close; SHA256 `37f90be695d16cf3937a7c019cc8e2e8c04a5cb482489ee89aecb3b3fbcdfd5f`. Five fixed source-bootstrap short runs passed `5/5`; SHA256 `69223aae8d1ff4d6273cf0acf4f37d7ab01b49dc6182054d72265f1e38fa7852`.
- Froze the r7.22 DINO prospective contract before new video review; SHA256 `6b5924ab35b9866a4ec23994518e5316de1b7ccc1e127e2e79a5535392389845`. On the independent Japan riverside pedestrian video, DINO failed the fixed open projection but passed close; immutable result SHA256 `cde840bb604766d5dcd0c3f471e32d9db64c5e4b171c271113622d16c97a4995`.
- The older independently frozen r7.11 chromatic construction-marker expert passed the same video's small traffic-cone lifecycle without rule changes; official result SHA256 `ad71ad76a2859611401e13fad7ebc1de80c1927d1dc466d60593fd5bfb190191`.
- Added and tested a multi-expert risk-profile/lifecycle prototype. Independent channels OR-open; every channel that opened must independently close; conflicts remain present/uncertain and absence never asserts clear. Japan derivation prototype SHA256 `ed147893f8ee49f4b9cc3648aa7ec0d6227d8671e94ad6e711ba3327c0218f63`. Before registering any new source, corrected the contract's local-time-as-UTC metadata; frozen r7.23 contract SHA256 `4918fe3e6053a3dd7b13d200ea219dfbdfee2a6e20dbc7be67fefc5b18317071`. It requires a new post-freeze source. Training, calibration, blind, Android runtime, `RiskEventTracker` and production authorizations remain false.

### 2026-07-19 — Public-silver r7.24 prospective multi-expert negative failure

- Registered and downloaded the CC BY 3.0 Matoaka pedestrian walk only after the corrected r7.23 contract freeze. Video SHA256 is `017f860e002c75d093206772800bd68cb1c19f226b74e4d5a933798916347821`. All 1,073 DINO and chromatic samples were frozen before visual review; report SHA256 values are `1fb898825505deedb8f611e6de62406a628e2b055a7e14a797646f90c6318a9a` and `1e526a7a2e155856559766008fab3ad8c4f75315cbb441db15ed2db5fa1e008f`.
- Original-order review fixed a visually clear negative challenge around a traffic cone across the road. DINO falsely opened (`+.0375678`) and failed to close (`-.0551055`); the chromatic expert opened at 174 seconds and cleared at 192 seconds. Frozen result SHA256 `938c03fa3ac03a341ecb3b0885946496ad5ebe67268586f2c7b55ccf7f0ac0ef`.
- This prospectively rejects unconditional expert OR. The next open rule must combine mechanism evidence with adaptive path relevance and approach/intrusion dynamics. A fixed center nearfield gate is insufficient because it also rejects the valid Japan side-cone event. No existing result is rewritten and all training/runtime authorizations remain false.

### 2026-07-19 — Public-silver r7.25–r7.26 radial approach freeze and negative pass

- Added a deterministic retrospective radial-approach diagnostic over frozen r7.11 detections. The fixed gate requires five accepted samples, first/last three-sample medians, at least `.05` bottom-coordinate progress greater than horizontal sweep, and positive log-area growth. Japan passed; all Matoaka and historical vehicle events failed. Diagnostic SHA256: `8881048ddd5f3fab4f10fc56eba435d0bc572bf3b039d65b6f988cebae70efcc`.
- Froze r7.25 before registering any new source. Contract SHA256: `6746957ce89d4f133c21802632ba4c5c972b01d0d61e34895b11b12392e9a8ce`. DINO is support-only; an event may open only when the bound chromatic lifecycle and radial-approach gate agree.
- Registered and downloaded two new CC BY 3.0 continuous pedestrian videos from Commons before review. Bramwell video SHA256 is `9c582c178a67b12b70df7c5a9542b18f5f00583a69c2e61e2cbcf9dd64d29b45`; Stegna is `ae0385d336785b0179e8655ab0a3feaef507c7e9885fddb5d457ebd2fd8fc3d8`. Froze 2,163 full-video samples before visual review (SHA256 `efd3127df05aca4a280dacbdb92d2e47b083989f1845918bc941cb63b827e160`) and then froze zero r7.25 candidates (SHA256 `576fe7da8cdd213e2d4ee71ef6c4dff7fb56f9cdd8907bba563aed7bdc222c9a`).
- Dense post-freeze review classified Bramwell 375–385 seconds as a close-pass negative: the cone remains on the grassy shoulder, the paved route stays open, and the brief detection jump is caused by turn/pan rather than sustained intrusion. The scored negative passed; Stegna remains context-only because no marker lifecycle was reviewed. Result SHA256: `9c3910fc5467a0890831c721eeba3eda9371164ca121c2d11658f9ca58ec2752`.
- Added evaluator and tests. This closes one false-open case only; it provides no independent positive recall evidence. Training, calibration, blind, Android runtime, `RiskEventTracker`, and production replacement remain unauthorized.

### 2026-07-19 — Public-silver r7.27 post-freeze Tai Wo source rejection

- After network recovery, registered the 9.4-second CC0 Commons clip `Roadwork with no dust control 02` before download or frame review. Video SHA256: `4e6b47fc218a7e96cfa601309315dd73462957516fb1a07d15343d0baf0ba252`.
- Froze ten one-second chromatic samples before review (SHA256 `da2202521c783a37d9a3babc6ca70709e3a37050651b8aa7196b016c5bafc833`) and then froze zero r7.25 candidates (SHA256 `1d87766d92bc69b30bb6098f5ac7b5aaf6198e42b359ff160839eb34a00fc0f1`).

### 2026-07-19 — Public-silver r7.28–r7.29 independent lifecycle failure

- Preserved Rice Street as context-only after pre-review freezing: chromatic features `1baa692d...`, zero candidates `4acf7972...`; ambiguous route-choice windows were not forced into a score.
- Registered Edmonton and Kampala before download/review. Frozen combined chromatic features SHA256: `ea351ffe25201fb0e2ef6fa99d6e5cb6261784d5e429cc0fde673f519bd86ade`; frozen radial candidates SHA256: `0c350a4659485c9df490e3937fbb586c303d6a6da5aee46209aa7d123394c78c`.
- Edmonton opened correctly at 671 s but the frozen r7.25 lifecycle cleared at 697 s while GPT multiframe silver review kept the same constrained pedestrian episode active through 735 s. The preserved 38 s false-clear result SHA256 is `d6230f066e0684c1f65a88b2c6564071e08376011263d776fc71ed498bc7ca42`.

### 2026-07-19 — Public-silver r7.30–r7.32 gap bridge freeze and source stress

- Retrospective absence sweep selected the minimum passing value of 9 samples; probe SHA256 `a1070be1326b455c2ae83789c2254ada97c2465eedc4523c4633888d6e1528ca`. Froze the asymmetric entry/exit contract before further source search, SHA256 `b692f72758d7f34021a4dd02dd65371fa24a9ddc7faa48b821fb6003dd158169`.
- Installed a local FFmpeg essentials runtime under `E:\codex-tools\media\ffmpeg` to cut pre-registered CC BY source windows without changing project dependencies or Android code.
- r7.31 Dallas: pre-review feature SHA256 `e6777a75890a3549d9f97fa4105baf8b3e43b7e9e1383349a2e3974c7b9ad899`, zero-candidate SHA256 `6c530633e6636e5c2f716ce921fde5709acbc11fce6ab5cd17056840283b6ac9`, negative pass SHA256 `0186c526aef0b728d08f1aedaa6e46f9c5e4e73cbba52a9109688383d0a5fa5a`.
- r7.32 object-rich garage demo was rejected after freezing 592 samples and zero candidates; rejection SHA256 `264d0d6f6f9628a78d11a96a2eb446046281a568ac8b96600193f4f519c9f08f`. It receives no event gate credit.
- r7.33 Cape Town: registered a chapter-selected continuous CC BY window before download/review, froze 220 samples (`968082ec...`) and zero radial candidates (`38c550e9...`), then passed the wide-forecourt/roadside-marker negative control. Result SHA256: `eafef19730cf4e62774e35063f1677300f5a6913ebd5405f241085c11aee8556`.
- No Android runtime, `RiskEventTracker`, calibration, blind set, or production model was changed. The remaining r7.30 blocker is a true-radial-entry episode with rapid visual clear, needed to measure over-persistence.

### 2026-07-19 — Public-silver r7.34 dense-marker negative and local-window rejection

- Pre-registered the CC BY Jakarta reopening chapter before download/review and froze an exact 80-second clip (`f2a0d3ec...84fdb`). The frozen feature report contains 221 traffic-cone detections (`ab5b9cba...34b4b`) but the r7.25 scanner produced zero candidates (`988b87ea...4735`).
- Post-freeze multiframe review found an open curved route with markers along its boundaries, not a path-intruding obstacle. The prospective negative passed; result SHA256 `3ef05cb89c6df9cc939fa6370f673b96c23c330b484bdfc5518d214f8ff2ee3b`.
- Added a diagnostic-only local radial-window probe. Widths 5/7/9 retain Japan but false-open Jakarta and other negatives; width 12 also false-opens Jakarta. Report SHA256 `a410ef8eeebd78bf16098d1beccda519c6f0e977484bac9378cbded50d4e9fb9`. No contract was frozen; the next representation target is adaptive path occupancy.

### 2026-07-19 — Public-silver r7.35–r7.36 path relation and rapid-clear search

- Rejected a whole-image generative edit because it redrew the background. Built three equal-count path-relation pairs instead using one generated transparent cone asset plus deterministic compositing. Every clear/risk image has four identical cones at identical depth/scale/spacing, translated only laterally; all pixels outside intervention masks remain unchanged. Generation report SHA256 `7a1297a6d0c6f7cf5dbf9513204a904dc89a15458f02925d9df642311a270671`.
- The frozen DINO regional direction ordered all synthetic leave-one-out and mirrored pairs, Japan and Edmonton positives, and Cape Town negative. Jakarta remained a false positive at `+.00882`; the strict diagnostic failed. Report SHA256 `a4494f81b9cccca082bb4c65ac5b34ff3d8a45d50453f51058848745e85b7d33`. No contract or head was promoted.
- Registered the CC BY 3.0 Trubarjeva roadworks walk before download/review. Frozen features `aad2aec4...160d8`, zero candidates `cd67d219...c2d9`; post-freeze review found lateral construction and a temporary vehicle occlusion, not ordered rapid clearance. Rejection SHA256 `70730ac7a3cd55b837ecac323dcac0b22739c73e8c45f8bc7a5d61f71adcedfc`.
- No Android runtime, `RiskEventTracker`, training, calibration, blind or production model was changed.

### 2026-07-19 — Public-silver r7.37 Tampere metadata/viewpoint mismatch

- Pre-registered the full CC BY 3.0 Tampere tram-worksite item because its description explicitly says pedestrian walkways were narrowed. The fixed 137.28-second 640x360 video has SHA256 `7f930a1399165aad8822ac27b54b6924b9d8abd40a77ac0f664d66b47f47f850`.
- Before visual review, froze 138 one-second samples and 192 target detections (`18d85399...30b2b`); r7.25 yielded zero radial candidates (`e6918d21...36d99`).
- Post-freeze overview found a static roadside observation/pan, not a forward pedestrian traversal. Rejected before event scoring; rejection SHA256 `6ce7687fec64670fafb3ec178b08724bc24e12fe3d6afc57f51925695f1fe74a`. It receives no gate credit and authorizes no runtime or model change.

### 2026-07-19 — Public-silver r7.38–r7.41 obstacle-aware route-width rejection

- Added `run_public_video_obstacle_aware_route_width_probe.py` with frozen ADE20K walkable support, frozen chromatic-marker obstacles, deterministic safety expansion, and widest-route or adaptive-centerline distance-field descriptors. Eight pure geometry tests pass.
- Hard argmax connectivity collapsed clear and risk medians to zero (`7e5a72b1...d15d5`). Soft walkable-probability capacity restored paths but reversed Japan/Edmonton and false-constrained Cape Town (`c4e6ca6d...e5861`).
- Adaptive-centerline distance q10 retained Japan/Edmonton but false-constrained Jakarta/Cape Town and tied all original synthetic pairs (`cba812f7...ce73d`). A single physical correction from one full object height per side to half-height per side still failed (`7c906401...ba03`); no further scale search is allowed.
- Distance-field evidence remains auxiliary only. No contract, training, Android runtime, calibration, blind, or production authorization changed.

### 2026-07-19 — Public-silver r7.42 positive/negative DINO prototype gate

- Added a zero-parameter source-isolated positive-minus-negative prototype over frozen DINO regional pair deltas. Synthetic descendants and real Jakarta share one parent source and leave every training fold together.
- All five positives and all three held-parent mirrored synthetic pairs passed, but only two of five negatives passed. Jakarta boundary markers, the Bramwell turn/shoulder cone and the Dallas road-edge cone remained false positives; balanced accuracy is `.70`. Report SHA256 `4fd55dfcd57f84295a56c2335b203a62fb85f3dd2bc71b302ddf58f7446d249f`.
- The gate rejects five bootstrap runs: feature deltas still mix camera turn/scene drift with route intrusion. No head, SAM/ASAM, contract, Android or production change was authorized.

### 2026-07-19 — Public-silver r7.43 clear-drift nuisance projection

- Kept every r7.42 sample, source fold, prototype and zero threshold fixed. The only OFAT projects each real marker-minus-clear DINO delta away from the direction measured between the two halves of its own clear window.
- Negative recall improved from `2/5` to `3/5` and balanced accuracy from `.70` to `.80`; Bramwell was repaired, while Jakarta and the Dallas road-edge cone remained false positives. Report SHA256 `738d0ce65b1e80ecc67a1556f318c1ff6dd8954f33e804702bfdf83c1c435ca0`.
- Camera drift is a partial nuisance, not the full missing representation. Bootstrap, SAM/ASAM, Android and production gates remain closed.

### 2026-07-19 — Public-silver r7.44–r7.46 multisource counterfactual direction failure

- Added `build_public_video_multisource_equal_count_pairs.py` with frozen parent hashes, per-pair parent source inheritance, equal-count/equal-scale/horizontal-only checks, pixel invariants, and fail-closed train/runtime authorization fields. Five generator tests pass.
- Selected continuous forward frames at Bramwell 700s and Dallas 128s. A first Bramwell placement remained on the route edge and was visually rejected; one pre-registered position correction produced two accepted pairs. Accepted generation SHA256 is `ac294ee6224dfd5bc2313d0d3e4be7ae0250d199722778af9928d0bba3f59e96`; provisional GPT/VLM review SHA256 is `b9963c763513fb4ccebf8ce916759e77164e875104a2f4a59186004e9b7bd08a`.
- Generalized the deterministic prototype so every synthetic descendant inherits its own public-video parent source and is removed with that source's real samples. Seven prototype tests pass. With the r7.43 nuisance contract unchanged, r7.45 fell to `.3714` balanced accuracy and `1/7` positive recall (`9be9736241976bb3ce780b526738b6a4ff5f263440f5850486ffda504e0bb422`).
- Added a fixed position-sensitive `4x4` DINO patch-grid mode as a single OFAT. r7.46 still failed at `.3429` balanced accuracy (`7864be4266be053bba6053b19e028e1e4951d8de3ea30e522da82fd49f6aac0b`). This rejects both regional pooling and coarse spatial-grid DINO as the shared route-relation representation; no bootstrap, SAM/ASAM, Android, calibration, blind, or production authorization changed.

### 2026-07-19 — Public-silver r7.47–r7.48 explicit route relation and event composition

- Added `run_public_video_explicit_ego_route_relation_probe.py`. It restores frozen walkable support only inside expanded marker masks before tracing the ego route, then scores q10 route-to-obstacle distance. Four pure geometry tests pass.
- r7.47 retained Japan/Edmonton (`2/2`) but only two of five real negatives; Jakarta, Cape Town and Bramwell remained false positives. Balanced accuracy is `.70`; original synthetic ordering is `3/5`, mirror ordering `5/5`. Report SHA256 `6a72165c166ac64b46e71a54149b928c3ea5da3b662a18419bd6c6ad36d0f87e`. No inpaint/width/threshold search follows.
- Added `run_public_video_event_risk_profile_lifecycle_gate.py`: frozen radial approach AND positive route relation opens an event; frozen chromatic continuity and nine absent samples manage persistence/clear; segmentation is auxiliary and cannot open alone. Three pure tests pass.
- r7.48 achieves `TN=5/FP=0/FN=0/TP=2` and balanced accuracy `1.0` on current retrospective event rows, while retaining the r7.30 Edmonton one-reminder lifecycle. Report SHA256 `11f75c2697d45b329301667d89ca8dcae3355d483e4557f8725aea8dc2a8ecb1`.
- Full closure remains false: all reviewed negatives have zero frozen radial entries, so no independent real true-radial safe-lateral veto exists; Japan has not been causally replayed through the r7.30 lifecycle. A renewed licensed-source search found no eligible source (`e8e244517e7bdd1080137d261d3050f39225db00baea76c624c66bafc2a3382d`). Bootstrap, training, SAM/ASAM, Android and production gates remain closed.

### 2026-07-19 — Public-silver r7.49 Japan causal lifecycle timing failure

- Added `evaluate_public_video_japan_causal_lifecycle_replay.py`; every r7.25 decision uses only the current prefix, the route baseline is the frozen 2–7s pre-risk clear median, and r7.30 persistence/clear remains unchanged. Two helper tests pass.
- The five-sample radial minimum first passes at 8s and route support is already above baseline, so the composed event opens at 8s. The frozen GPT/VLM risk window starts at 10s; entry is two seconds early and the timing gate fails.
- Persistence and close are correct: one reminder, risk covered through 15s, and clear at 22s after nine absent samples inside the frozen 17–22s clear window. Report SHA256 `95ce8201dea8979634e07c07cbefe5edaa73f74c428dc1172676c1f644601d56`.
- Event identity separation is therefore not lifecycle closure. No post-hoc onset expansion or relation-threshold fit is allowed; training, bootstrap, SAM/ASAM, Android and production remain unauthorized.

### 2026-07-19 — Public-silver r7.50 prospective event-timing contract

- Froze `configs/public_video_event_timing_contract_r750.json` before reviewing another source. Contract SHA256 is `9b3c9fb4ed42e9ad8723592e31c2ac506584d12cf54036c5202d21df52041067`; four validator tests pass.
- The fixed reminder band begins 3000 ms before reviewed material-risk onset and ends at the latest useful reminder. This limit is explicitly disclosed as a post-r7.49 hypothesis; Japan remains an immutable r7.49 timing failure and is forbidden from r7.50 acceptance.
- Full closure requires two distinct real source hashes: an independent positive with one useful reminder and valid clear, plus a true frozen-radial safe-lateral negative vetoed by the frozen route relation. Synthetic and GPT-only examples receive no gate credit; large-model review remains provisional silver.
- The unfilled review template SHA256 is `2e823daa9fe379e61e158a5c071792a9295f9b87ec5fba43511e0a1a4ed95797`. Training, bootstrap, SAM/ASAM, calibration, blind, Android runtime and production authorizations remain false.

### 2026-07-19 — Public-silver r7.51 prospective London positive

- Pexels item 3874684 was stopped by item-identity lineage because it had already been used and rejected in r7.17; a new resolution/hash cannot create prospective credit. New Pexels item 5234995 froze 25 samples but yielded zero r7.25 candidates (`c547fa66...1a714`) and was rejected before visual role review.
- Registered the CC BY 3.0 POPtravel London first-person walk before download/review. The 240p transcode SHA256 is `ee68ca320e07a84358a7adc557187811a45242dac8dd33b884c7eba922d82836`; 3301 frozen one-second samples contain 534 traffic-cone and 923 total target detections (`81e5317e...83680`). One frozen radial event spans 2678–2687s (`088f37bb...5102b`).
- Post-candidate original-order large-model review assigned a positive role, not the requested safe-lateral negative: real red cones/delineators move into the lower center route. The frozen review uses material onset 2681s, latest useful reminder 2684s, risk end 2687s and stable clear 2688–2699s.
- Added `evaluate_public_video_event_timing_positive.py` with three timing/veto unit tests. Frozen route delta is `+.953596`; reminder at 2678s is exactly the contract's three-second early boundary, clear occurs at 2696s after nine absent samples, and reminder count is one. Prospective positive passes; report SHA256 `7043eeaf60427f688e18aa75e458d8536a1ee4ffa1eb91a6644bd5d14361eb8c`.
- Full r7.50 closure remains false until a distinct real true-radial safe-lateral negative is vetoed by the frozen route relation. Bootstrap, training, SAM/ASAM, Android and production remain unauthorized.

### 2026-07-19 — Public-silver r7.52 Ulm true-radial route-veto failure

- License screening rejected Commons items still awaiting license review and YouTube results without item-level open-license metadata. Registered CC BY 3.0 Maribor froze 2100 one-second samples and 215 target detections but zero r7.25 candidates, so it was rejected before visual role review.
- Registered CC BY 3.0 POPtravel Ulm before download or review. Its 240p video SHA256 is `67efb35b1e0bbb15bc5be1b5289ec02b1d88a3a8a895a51e16b6de5ac49caa48`; 2177 frozen samples contain 282 target detections (`7053981f...18cf`) and three frozen radial events (`bf1c1fc6...a3b9`).
- Froze all three review windows and 18 keyframe timestamps before viewing (`81ca66c6...e6c1`). Provisional large-model review rejected candidates 1 and 3 because construction barriers coincided with route turns, and assigned candidate 2 (`1504–1510s`) as the only true-radial safe-lateral event: delineator posts remain on both road edges while the central passage stays open.
- Added `evaluate_public_video_true_radial_route_veto.py` and three unit tests. The frozen r7.47 relation failed the required veto with clear median intrusion `0`, marker median `.928303`, and delta `+.928303`; report SHA256 `cb02a1c0de75f1c07de05666958d6c4ece2bcc4818401939b4e18d6fba7d9ea1`.
- This is a representation failure, not a head-optimization result. No post-review geometry tuning is permitted; bootstrap, training, SAM/ASAM, Android and production gates remain closed.

### 2026-07-19 — Public-silver r7.53 offline future ego-trace diagnostic

- Froze `public_video_future_ego_trace_contract_r753.json` only after the Ulm failure, explicitly making Ulm derivation-only. The contract fixes 1/2/3-second ORB+RANSAC homographies, a future lower-center anchor, half-object-height expansion and median aggregation; no parameter or aggregation search is allowed.
- Added `run_public_video_future_ego_trace_probe.py` and three geometry tests. Both London and Ulm produced valid traces for every event frame. Ulm mean/median intrusion are `0`; London mean is `.266667`, but its fixed median is also `0`, so the diagnostic gate fails. Report SHA256 `fc9f4b0a5185a7c6d4538423dd1cd4b09628e702ca7a31f89fad84bf408362f3`.
- Future ego trace contains sparse positive signal but is not stable enough for the primary relation score and cannot run causally at inference. It remains an offline auxiliary-teacher idea; all training/runtime authorizations stay false.

### 2026-07-19 — Public-silver r7.54–r7.60 future-route teacher and distillation diagnosis

- r7.54 expanded the future-anchor hypothesis to three positive and six negative real events. Valid ORB scores strictly separated the classes, but Japan coverage was only `.25`; report SHA256 `31ede4fc0f702bbab9f80543a9f602fded2d849255970e473532653fb6fcf464`.
- r7.55 froze DIS medium dense future-to-current flow with all other horizons/readout rules unchanged. Every event reached full coverage; positive scores `.0833/.1875/.1667` all exceeded the maximum negative `.0476` from Ulm. Gate passed, SHA256 `5b097322e6e5977d7539b7332d51cc5ecb1217913e6d5ff2da2f99b2b673eea6`. This is offline auxiliary truth only.
- r7.56 causal current-to-past constant-motion projection failed event separation (`42d3f616...e91a0`). r7.57 current geometry/past anchors and r7.58 frozen DINO regional features passed frame-level teacher-active AUROC (`.8427/.8116`) but failed event separation (`923f4583...50af`, `744e27ba...78f84`). Global/head variants stop here.
- r7.59 spatial merged route distance fields achieved source-isolated pixel AUROC `.9311` but failed marker-overlap event separation (`547f087c...30471`). r7.60 retained three horizon heatmaps and exact argmax-point hit semantics, reaching pixel AUROC `.9161` but still failing event separation (`c4571611...f401c`).
- The future-route auxiliary target is valid and spatially learnable; the current 126-frame static representation is insufficient for cross-source future route choice. Next work expands automatic source-isolated causal-clip/future-route targets for a dedicated temporal route head. Android, production, bootstrap and SAM/ASAM remain closed.

### 2026-07-19 — Public-silver r7.61 automatic temporal-route auxiliary manifest

- Froze `public_video_temporal_route_auxiliary_dataset_contract_r761.json` before materialization. Eligibility requires a frozen marker detection, three seconds of causal history and all 1/2/3-second future anchors; each source is deterministically capped at 128 evenly spaced eligible timestamps. Event roles and labels are never read.
- Materialized 753 unique items from 10 sources. Every item has a null event label and all three valid future-route anchors. Manifest SHA256 is `05424d63fdb84cd384a75078c7df842329358e714bae3c41afe70bb97c938428`; audit report SHA256 is `86a186eb517a44a0096a829ee4569ca8d690ba109b5baad346ab7cbde6f7d6e6`.
- The 10 sources include context-only Kampala and Stegna because their already-bound joint feature reports contain independent source rows; they receive route-auxiliary training data only, never event-gate credit. Long London/Ulm/Edmonton sources are capped at 128.
- r7.61 authorizes only a source-isolated temporal-route auxiliary train-only prototype. Risk-event training, calibration, blind, Android and production remain unauthorized.

### 2026-07-19 — Public-silver r7.62–r7.64 temporal route head

- Added the fixed 43-channel temporal input (32 projected DINO patch channels, RGB, full spatial 1/2/3-second causal DIS flow and coordinates) and a 61,955-parameter convolutional route head with ten whole-source held-out folds.
- Marker-only r7.62 reached route-field AUROC `.91058` but localization `.1222` and failed event separation (`98ec998a...902cc`). r7.63 expanded the label-free auxiliary manifest to 2,102 continuous frames across ten sources (`59923fbc...e0e`).
- The exact same seed-0 r7.64 head improved to AUROC `.91974` and MAE `.0420`, but localization remained `.11787`; Japan scored zero while Ulm remained above London. Report SHA256 is `eaa3c8ef0e03b50254e73fb1eb34e504051f4593c115ba528a116f1dab96aa0c`. Five-seed runs remain closed.

### 2026-07-19 — Public-silver r7.65–r7.66 frozen temporal risk profile

- Audited the exact frozen r7.64 outputs without retraining. Only the expanded-marker relative peak ratio retrospectively separated the old three positive and six negative events; audit SHA256 is `f4f535dbc2abffd94f1802c6494cfde87a48e9d9a588c0a89222a36a1bd1a9cf`.
- Froze the disclosed post-hoc midpoint threshold `.68`, half-object-height marker expansion, r7.25 radial entry and r7.30 nine-absent/one-reminder lifecycle before any new source scoring. Contract SHA256 is `73076ff9a37cf97218167068285924792aa3a5b7d43372fc163b6d8f6617ecf3`; diagnostic checkpoint SHA256 is `18690367a33fd5857739e41961ffac6d1d7f1f99b23ec953922401da54ad7901`.
- Added a strictly offline prospective evaluator that reconstructs the exact r7.64 43-channel input, rejects every r7.54–r7.65 derivation source ID and video hash, binds all inputs by SHA, and requires the frozen timing/lifecycle gate for positives. It never grants acceptance, training, calibration, blind, Android or production authority by itself.

### 2026-07-19 — r7.66 prospective-source acquisition round 1

- Registered and downloaded the CC0 Spiegelgasse item before any visual review. Its 121 one-second samples produced zero frozen radial candidates, so it was rejected pre-visual.
- Registered the CC BY 3.0 Alicante POPtravel item and downloaded its complete 240p transcode (`fb325c92...5445a`). The first frozen 1,200 samples produced zero candidates. Sequential proposal search over the still-unseen remainder froze another 2,581 samples and one `2408–2424s` radial candidate (`aa61c079...4f56`).
- Candidate-bound visual review found a continuous promenade walk but no real construction marker. YOLOE had expanded over a restaurant terrace, stairs, railings and bright fascia/panels as `barricade`; the event was rejected before temporal risk-profile scoring and receives neither positive nor safe-lateral-negative credit.
- Pexels item 2980886 was item/license registered before download but decoded to only 3.47 seconds. It cannot meet r7.25's five accepted one-second samples and was rejected before feature extraction or visual review. r7.66 remains open and diagnostic-only.

### 2026-07-19 — r7.66 Bristol previsual source rejection

- Selected the license-reviewed CC BY 3.0 POPtravel Bristol walk from Commons text metadata and pre-registered the first 20 minutes before download or frame review. The 240p video SHA256 is `1d2d74adab3023ea8f9abbb48f31d683f3600afc8d1641f28ac44df60399dc76`.
- The first interval froze 1,200 samples, 66 traffic-cone and 188 total target detections (`cfd6dbf0...78ae`), but zero r7.25 candidates (`f8aac721...6cfb`). The still-unseen remainder was then registered as adaptive proposal-only search, not blind evidence.
- The remainder froze 2,161 samples, 87 traffic-cone and 344 total target detections (`91b2f6cb...40a4`) and again zero candidates (`828b8186...7e0e`). The complete 56-minute source was rejected before visual review or r7.66 scoring (`92a0ccf7...ff88`).
- A first extractor invocation omitted the r7.11 contract; the candidate freezer rejected it. That report was preserved under a `missing_contract_rejected` name, and the same fixed detector settings were rerun with the contract. A separate 7 ms duration endpoint mismatch was also recorded and mechanically normalized before remainder sampling.
- No event role, training, calibration, blind, Android runtime, production, five-seed or SAM/ASAM authorization changed.

### 2026-07-19 — r7.66–r7.69 Bangkok pressure pair and marker-relation diagnostics

- Pre-registered and scanned the distinct CC BY 3.0 Bangkok Modern Center walk. Frozen review identified a real radial safe-lateral cone event at `300–311s` and a route-intruding cone event at `328–339s`. Frozen r7.66 scored them `.80168/.87608`; the negative crossed `.68`, so r7.66 failed without threshold retuning. The positive event-local lifecycle replay produced one reminder and a valid clear.
- Fixed the prospective evaluator to replay lifecycle only over the selected event rather than all source candidates. Added a pair-error audit; the offline future teacher retained the correct `.11111/.33333` order while the old head compressed both above threshold (`c231b9ad...e5ec`).
- r7.67a retained all 324 marker frames, including four sub-patch detections via a frozen nearest-patch fallback. The 132-D deterministic marker-conditioned ridge reached ten-source pooled OOF AUROC `.883995` and a post-hoc Bangkok margin `.15003` (`e4ae6777...bc12`).
- Pre-run independent review removed Bangkok from authorization and replaced whole-source bootstrap with equal-mass `(source,class)` blocks plus source-macro metrics. r7.68a passed `0/5`: prototype-only balanced median `.8534`, optimized `.7245`, with positive recall `.39–.56` (`3e17d71a...f246`).
- r7.69 tested distance-field auxiliary supervision as a paired five-seed OFAT. Distance MAE improved in all runs, but the median primary balanced-accuracy delta was only `+.00019`; retention failed (`8edd27a4...d59a`). No Android/runtime/default-model or production authorization changed.
- r7.70 materialized the two Bangkok event intervals into 24 unique one-second PNGs with a parent-bound JSONL manifest (`90287f20...2612`). The package is a real same-source matched contrast and representation-training candidate only; training execution and independent evaluation remain unauthorized.

### 2026-07-19 — r7.71–r7.77 pair ranking, target correction, and causal lifecycle failure

- r7.71 nearest-time within-source pair ranking passed with median/minimum source AUROC `.85/.67385` and pair ordering `.792857` (`caa37275...c6d`). Five r7.72 optimized bootstrap heads were individually viable, but their median `.84889` underperformed the zero-training bootstrap prototype median `.90833` by `.05944`; the stability gate failed (`be949071...cf7`).
- The r7.73 zero-training prototype lifecycle opened the Bangkok safe-lateral event at 304s and the positive at 337s, one second after the latest useful reminder (`1615d161...b25`). Geometry-matched r7.74 and training-fold geometry-residual r7.75 reduced median AUROC to `.7733/.675`, so both branches stopped (`48914904...6f7`, `7c03f19d...a2f`).
- Target audit found counts `268/15/10/31` for `0/3,1/3,2/3,3/3` future-horizon hits. r7.76 froze strong-positive as at least two of three hits and reached median/minimum source AUROC `.96774/.71959` with pair ordering `.94231` (`7b2f5d89...9d2`).
- r7.77 changed only that target, yet causal Bangkok openings remained 304s/337s (`c18cff4e...db1`). Offline separability is no longer the main blocker; current RGB, DINO, and past flow do not expose route choice early enough. Optimizer/SAM/ASAM, Android runtime, and default-model changes remain unauthorized.

### 2026-07-19 — r7.78–r7.78a Düsseldorf external safe-lateral challenge

- Pre-registered the license-reviewed CC BY 3.0 POPtravel Düsseldorf source and first 20 minutes before download or frame review. The 240p video SHA256 is `c9bedb93...fc5f`; 1,200 frozen samples contained 85 traffic-cone and 223 total detections, producing two radial candidates (`529f29b5...fbd4`).
- Original-order model review rejected 117–127s as a real-barricade/route-turn confound and assigned 900–910s as a true-radial safe-lateral provisional negative: the barricade stays on the left curb while the camera proceeds through the open right corridor.
- Frozen r7.66 scored the negative `.771165`, above `.68`, and failed (`19ba400d...19d4`). The first r7.78 attempt failed closed before scoring because two marker-absent frames exposed an implicit non-empty-mask assumption.
- r7.78a froze gap handling before scoring, retained exactly the nine r7.25 accepted marker samples, and still opened at 906s with maximum relative score `2.6933` (`ab880466...54af`). The majority target improves offline separation but does not solve causal safe-lateral false alerts. No training, SAM/ASAM, Android, or production authorization changed.

### 2026-07-19 — r7.79–r7.81c causal readout and event-role diagnostics

- r7.79 replaced the dense readout with fixed block pooling plus multi-output waypoint ridge over the same causal cache. Mean localization error was `.11583`, only `.00204` better than r7.64, and strict event separation failed (`cc54e30...81e6`).
- r7.80 split generic radial context from a past-only committed-route upgrade. It upgraded the positive at 336s and did not upgrade Düsseldorf safe, but falsely upgraded Bangkok safe at 307s. Fixed the lifecycle report to read `confirmed_clear_timestamp_ms`; r7.80a confirms clears at 320/348/919s while the overall gate remains false (`6e466af8...95e9`).
- r7.81c retained 106 marker-present frames across eight events/sources for event-role LOSO ridge. Event AUROC was `0`, balanced accuracy `.2`, and positive recall `0` (`3f1a69ff...beb3`), exposing source shortcuts and insufficient matched mechanism coverage. The r7.81 attempts are diagnostic failures, not prospective credit.

### 2026-07-19 — r7.82–r7.84 Cologne/Cardiff acquisition audit

- Registered and downloaded distinct license-reviewed CC BY 3.0 Cologne and Cardiff sources. Cologne's first 40 minutes yielded 2,400 samples and 190 target detections but no frozen r7.25 candidate. Cardiff's full 3,241 samples yielded 875 target detections and likewise no r7.25 candidate.
- A diagnostic five-accepted-sample local gate found three Cardiff windows; widths 7/9/12 found none. Review windows were frozen and hash-bound before visual inspection.
- GPT/VLM original-order review retained two real yellow floor-warning-sign sequences as provisional path-intrusion/right-side-pass candidates and excluded the road-cone junction because the camera follows a planned right turn (`088bad8f...17a9`). Both retained events share the Cardiff parent source, add no independent-source credit, and cannot be promoted to canonical r7.25 events.
- Training readiness remains false. No SAM/ASAM, Android runtime, default-model, calibration, blind, or production authorization changed.

### 2026-07-19 — r7.85 causal actionability target correction

- Froze a post-hoc semantics audit over the hash-bound r7.80a report. The state machine uses only current/past committed-motion trace: two consecutive `>=1/3` samples enter intervention and two later `<1/3` samples confirm route clear.
- The Bangkok event formerly labeled safe-lateral entered intervention at 307s and cleared at 311s; the positive entered at 336s and remained persistent; Düsseldorf remained context-only. All frozen expectations passed (`0120009d...07e5`).
- This exposes one causal label contradiction: eventual successful avoidance cannot be used as a no-warning target when the no-change corridor first required intervention. The original review role remains preserved for auditability.
- Added `blindassist_public_video_silver_labels_v3` validation. It requires `silver_actionability`, `causal_evidence_basis=past_or_current_only`, and prevents `safe_pass` or `route_changed` from relabeling an earlier context/intervention state as `candidate_no_alert`.
- This corrects supervision semantics only. It does not authorize training promotion, calibration, blind evaluation, Android runtime changes, or default-model replacement.
## 2026-07-19：因果 actionability 与视觉惯性路线意图诊断

- r7.86–r7.89 完成事件重标、Cardiff/Ulm 因果复核和 source-isolated manifest：16 个事件、11 个来源、3 个独立 intervention 来源；旧 role 标签矛盾率 `25%`。
- r7.90 事件均值 probe 失败；r7.91 风险轮廓 + 生命周期改善但仍失败，证明当前纯视觉因果输入缺少前瞻路线选择，而非继续优化 head 即可解决。
- 下载并校验 ADVIO-15：实际 `54,845,329` 字节，官方 MD5 完全匹配；r7.93 同步/采样率/四元数审计全通过。
- r7.94 原始 IMU 连续块 probe AUROC `.4770`；r7.95 旋转不变 OFAT `.4746`，无改善。停止单序列未来意图 head 变体；架构方向优先改为显式 route-intent + 视觉风险轮廓/生命周期，并把 IMU 转向确认单列为待证假设。
- r7.96 current-only 转向确认负控 AUROC `.3465`、balanced `.3878`，说明未对齐的手机旋转也不能直接确认世界路线转向；撤回“IMU 可直接作确认器”，不新增 Android sensor 接口。架构方向收紧为显式 route-intent 优先，IMU/VIO 必须先独立证明姿态对齐。
- 新增 `docs/PUBLIC_VISUAL_INERTIAL_ROUTE_INTENT_2026-07-19.md`、三份 ADVIO 合同、三个执行脚本及配套测试。ADVIO 路线保持 CC BY-NC、research-only、production-isolated。
- r7.97a 在 16 个事件/11 个来源上用冻结 future-route teacher 模拟外部显式路线输入，无学习参数的连续两秒交叠规则达到 intervention recall `1.0`、context recall `.8333`、balanced `.9167`；勘误版重算 12 份 feature report 与 13 个本地视频 SHA，指标不变。新增生产隔离的 route-risk 生命周期原型、机器接口模板和模型合同；未来视频仍严禁进入 eval/runtime，默认 App 未改变。
- r7.98 将同一生命周期移植到 `device-benchmark`，JDK 17 `assembleDebug` 成功，APK `5c44bcab...284c`；SM-S9280/API 36 定向 instrumentation `3/3` 通过。未接入 App/core/default model。后续全量 connected rebuild 被并行 SparseLK 文件的既有类型错误阻塞，本任务未修改该文件，且不影响已哈希绑定 APK 的定向结果。
- r7.99/r7.99a tested fixed LEFT/STRAIGHT/RIGHT camera-space templates. The 16-event replay scored balanced `1.0`, but all four interventions were STRAIGHT; LEFT and RIGHT had zero intervention coverage, so the full provider gate stayed closed.
- r8.00/r8.02 searched r7.61 mean future-anchor x and visually rejected all five candidates. The audit established that a future lower-center flow correspondence is not a categorical turn label and is confounded by moving objects, camera motion, and detector false positives.
- r8.03 replaced direction inference with robust upper-background yaw flow, producing 13 LEFT and 17 RIGHT candidates across multiple sources. r8.05 deterministic frozen-template intersection retained three windows. Dense r8.06 review and r8.07 VLM adjudication rejected all three as parallel construction boundaries or a Bramwell building false positive; LEFT/RIGHT intervention coverage remains `0/0`.
- Preserved all reports and SHA sidecars, including the r8.01a timestamp erratum. No App/core/default-model wiring, training promotion, SAM/ASAM, calibration, blind, or production authorization changed.
- r8.08 froze three Commons and three Vimeo directional-obstruction searches before retrieval. The original Commons Python TLS handshake produced no response; r8.08a repeated the exact queries once via Windows TLS without pagination or rewriting.
- Commons produced one police/dashcam semantic false hit and it was rejected without download. Vimeo produced one title-relevant, item-level CC BY 3.0 Burwell candidate; 5-second whole-video review rejected it as an edited public-works/news package rather than a continuous pedestrian POV causal episode.
- r8.09 admitted zero events; LEFT/RIGHT intervention coverage remains `0/0`. The full route-field interface result remains valid, while categorical-provider, training, Android, calibration, blind, and production gates remain closed.
- r8.10 added a frozen, one-page Internet Archive discovery path with item license metadata and four offline parser tests. Three queries returned three archival films from 1930–1945; r8.11 rejected all three from metadata without downloading because aggregate full-text hits did not describe continuous pedestrian POV events.
- A separate exact web search for YouTube watch pages produced no usable item-level candidate in the returned results. No query result was treated as a license or event label.
- r8.12/r8.12a built a three-parent train-only route-conditioned dataset. The generic `static_obstacle` v1 was rejected for class-semantic leakage; the corrected `inserted_temporary_obstacle` v2 passed 36-image, 108-route-label, mask, YOLO, COCO, and full visual QA.
- r8.13a separated feature from head: an exact-risk-field LOSO linear head scored balanced `1.0`, while binary DINO global/route readouts scored `.5000/.6249`. This localized the failure to risk representation or source/asset coverage.
- r8.14a reexecuted a two-asset-per-parent factorial after rejecting r8.14 for an embedded future contract timestamp. All 72 regenerated image hashes matched the visually reviewed invalid run, geometry QA passed, and binary route BA remained only `.6332`.
- r8.16 changed only the auxiliary target to a frozen bbox distance field. Route-conditioned BA passed at `.9156` with clear/block recall `.8696/.9615`, worst source `.89`, and exact repeat equality; global BA remained `.5446`.
- r8.17 fixed two-consecutive open lifecycle passed at `.9429`, worst source `.90`. It does not test route-clear because no post-obstacle departure sequences exist.
- r8.18 five prototype/bootstrap 80-step runs were stable (BA std `.0090`) but failed the frozen gate: mean `.8774`, worst seed `.8682`, worst clear recall `.7971`. No parameter retry, SAM/ASAM, App wiring, calibration, blind, or production authorization followed.

## 2026-07-19：r8.19–r8.23 距离场真实迁移与风险生命周期诊断

- 新增 r8.19 source-LOSO 真实迁移 probe。r8.16 合成距离场在 16 个真实 provisional 事件上，路线/全局 balanced 为 `.7083/.5833`；路线提升成立，但 context recall `.6667` 未过门。
- r8.20 只追加全场背景统计后 balanced 降到 `.4583`；r8.22 只追加跨三场景 traffic-cone 合成族后降到 `.5833`。两条失败均按预注册合同保留，未改阈值或重试。
- r8.21 新增可复现 chroma-key 资产准备脚本。无 alpha 棋盘格首稿和有色边首版均拒绝；v2 交通锥经完整视觉/几何 QA，形成 108 图、324 路线样本的 train-only 数据集。
- r8.23 以 218 个真实 provisional 帧状态做层级等权 ridge 和固定两帧 open/clear，事件 balanced `.5417`、context recall `.3333`，证明 lifecycle 不能修复底层真实风险字段误激活。
- 结论：生产、Android、calibration、blind、SAM/ASAM 全部保持关闭。后续优先真实训练折内的局部风险/距离监督，不再堆 head 或单个合成资产族。

### r8.24 真实 provisional marker 距离场负控

- 新增按完整 parent source 留一的真实 marker bbox 距离场 teacher。所有 held-out bbox 均从 teacher 拟合中排除；218 帧、28 个无检测全零场，near/far、来源、帧分层平衡。
- 结果全局/路线 balanced `.6250/.5833`，路线 context recall `.4167`，低于 r8.19 `.7083`。报告 `7750e172...d0013`，冻结门失败。
- 由此停止“只改善 object-distance map”的路线；后续目标改为显式 route-field × visual-patch interaction，仍先做确定性 probe，不启动优化器或端侧改动。

### r8.25 route-field × frozen patch interaction

- 新增固定 32 维投影与连续三点 route polyline 场，直接对冻结 DINO patch 做 route/off-route/contrast pooling；均匀场为同维负控，完整来源留一。
- 均匀/路线 balanced `.4583/.5000`，路线 intervention recall `.25`，报告 `ec1b3684...93b60`。预注册门失败，未搜索 sigma、维度、ridge 或非线性 head。
- 主线回到已获最强证据的 external route provider + detector bbox deterministic intersection + lifecycle；模型风险场降为辅助，不修改 App/default model。

### r8.26–r8.26a explicit-route device geometry and aspect-ratio erratum

- Added benchmark-only Kotlin conversion from validated 1/2/3-second route waypoints and normalized detector boxes to the frozen lifecycle score. The first build and 9 targeted device tests passed, but a new all-frame audit rejected r8.26: 59/654 anchor hits and 30/218 frame scores differed from r7.97a.
- Diagnosed a coordinate-unit bug: the Python oracle expands in pixels by object pixel height, while r8.26 reused normalized y-height for both axes. Froze r8.26a before reexecution, added positive frame dimensions, separate x/y normalized margins, and a non-square-frame regression.
- r8.26a passed 4 offline geometry tests, reproduced all 654 anchors and 218 frame scores with zero mismatch across 16 events/11 sources, built successfully offline, and passed `OK (10 tests)` on SM-S9280/API 36. APK SHA256 is `b543dd9d...62fe`.
- Evidence remains benchmark-only. No App/core/default-runtime wiring or real route-provider, LEFT/RIGHT intervention, calibration, blind, or production authorization changed.

### r8.27–r8.27a Android external non-future route payload boundary

- Added a benchmark-only Android Intent parser requiring provider/projection receipts, issued/valid timestamps, confidence, and exact 1/2/3-second normalized camera waypoints. Risk-model-generated routes, future-video routes, future-issued, expired, overlong, low-confidence, missing-receipt and malformed payloads fail closed.
- r8.27 failed before execution because the test fixture treated `Intent.removeExtra()` as returning Intent. r8.27a changed only the fixture to retain the Intent via `also`; provider logic, policy and assertions stayed frozen.
- Offline build then succeeded. On SM-S9280/API 36, 6 provider tests plus 7 geometry and 3 lifecycle regressions passed: `OK (16 tests)`, APK `ffb540c1...5bcc`.
- This validates the external Android payload boundary, not a real navigation provider, camera projection accuracy, LEFT/RIGHT intervention coverage, App wiring or production readiness.

### r8.28–r8.28a world-route camera projection

- Added benchmark-only local-ENU route projection using an externally supplied world-to-camera rotation, camera origin and pinhole intrinsics. Receipt age/confidence, right-handed orthonormal rotation, positive depth, in-frame coordinates and exact 1/2/3-second horizons all fail closed.
- r8.28 device execution rejected three nominal valid cases as `invalid_pose`; the identity test fixture had eight values. r8.28a added only the missing zero and preserved projector logic and gates.
- On SM-S9280/API 36, 6 projection, 6 external-payload, 7 geometry and 3 lifecycle tests passed: `OK (22 tests)`, APK `e4eeabf3...2716`.
- Projection math is validated only in benchmark. Real pose/calibration/navigation accuracy, LEFT/RIGHT intervention coverage, App wiring and production authorization remain open.

### r8.29–r8.29d real-device projection input capability

- Audited all real back-facing Camera2 characteristics and sampled the SM-S9280 rotation-vector sensor for 2.5 seconds. r8.29 had a compile-only nullable FloatArray issue; r8.29a–c passed all device capability assertions but failed only while exporting the receipt from the com.android.test storage context. r8.29d returned the same JSON through instrumentation status and passed `OK (1 test)`.
- Two back cameras expose focal length, physical sensor size and pixel arrays, allowing derived intrinsics. Camera 0 derives approximately `fx=fy=2625px` at 4080×3060. The apparent absence of exact intrinsic calibration and distortion was observed from a context without `CAMERA` permission and is corrected by r8.30 below.
- The QTI rotation-vector stream produced 119 samples over 2453.9ms, median interval 20.80ms, maximum quaternion norm error `4.16e-8`, and observed maximum stationary delta `.0255°`.
- Real pose-stream capability is supported; permissioned exact calibration is audited in r8.30. Reprojection error, navigation route accuracy, LEFT/RIGHT events, App wiring and production remain open.

### r8.30 permissioned Camera2 calibration, lens-pose and timestamp audit

- Froze a permission-aware audit contract before build, grant, and device execution. Android's Camera2 contract requires `CAMERA` permission for intrinsic, distortion, and lens-pose metadata, so r8.29d nulls were not evidence of device absence.
- Built offline and passed `OK (1 test)` on SM-S9280/API 36 using the CAMERA-permissioned target App context. Both back cameras expose exact intrinsic calibration, distortion, lens pose rotation/translation, and `REALTIME` timestamp source. Camera 0 reports `fx=2766.1165px`, `fy=2771.1763px`; camera 2 reports `fx=1653.87px`, `fy=1656.9772px`.
- Both cameras report `PRIMARY_CAMERA` pose reference. The lens-pose rotation still maps Android sensor coordinates to camera-aligned coordinates, while the rotation-vector matrix maps device coordinates to ENU world coordinates. The deterministic candidate is `R_camera_from_world = R_lens_pose * transpose(R_device_to_world)`.
- Independent rotation-extrinsic calibration is no longer a prerequisite. Matrix-composition conformance, analysis-stream crop/scale mapping, real reprojection direction/error, navigation/LEFT-RIGHT events, App wiring, and production authorization remain open.

### r8.31 deterministic Android rotation-vector and Camera2 lens-pose composition

- Froze a benchmark-only contract and implemented `R_camera_from_world = R_lens_pose * transpose(R_device_to_world)` with quaternion normalization checks, orthonormal rotation validation, and fail-closed errors.
- SM-S9280/API 36 passed `OK (5 tests)`: identity, device-to-world inversion order, the r8.30 camera-0 `xyzw` quaternion, compatibility with the existing world projector, and invalid input rejection.
- The output intentionally remains in raw camera-aligned sensor coordinates. `SENSOR_ORIENTATION`, CameraX analysis-stream crop/scale, camera/IMU timestamp pairing, and real-image reprojection remain unvalidated; App and production authorization remain false.

### r8.32 production-matched CameraX stream geometry and clock audit

- Captured 30 frames on SM-S9280/API 36 with the production resolution selector, RGBA output and KEEP_ONLY_LATEST. The bound camera was ID 0; every frame was 640×480 with full `[0,0,640,480]` crop, 90-degree clockwise rotation, 2560-byte row stride and 4-byte pixel stride. Device test passed `OK (1 test)`.
- Capture timestamps were strictly monotonic, median interval `66.64ms`; callback minus capture timestamp was `77.43ms` median. This is an observed latency distribution, not yet an interpolation error bound.

### r8.33 authoritative CameraX sensor-to-buffer intrinsic transform

- Read `ImageInfo.sensorToBufferTransformMatrix`, which CameraX defines as the active-array-to-buffer mapping. Device test passed `OK (1 test)`.
- Camera 0 reported a uniform no-offset transform `diag(0.15686275, 0.15686275, 1)`, mapping 4080×3060 to 640×480. Exact principal point mapped sensor `(2041.33,1530.07)` -> buffer `(320.21,240.01)` -> rotated 480×640 display `(238.99,320.21)`.

### r8.34 deterministic analysis-display camera geometry

- Implemented a fail-closed benchmark mapper combining exact Camera2 intrinsics, CameraX sensor-to-buffer affine, clockwise buffer/display rotation, matching camera-axis rotation, and the r8.31 world-to-raw-camera pose.
- SM-S9280/API 36 passed `OK (4 tests)`: r8.33 numeric intrinsic mapping, 90-degree camera-axis rotation, world-projector principal projection, and rejection of shear/skew/unsupported rotation/invalid pose.
- The deterministic coordinate-formula gap is closed. Per-frame camera/rotation-vector pairing, distortion-aware real-image reprojection error, route/LEFT-RIGHT events, App wiring and production authorization remain open.

### r8.35 CameraX capture timestamp and rotation-vector bracketing

- Concurrently sampled the production-matched CameraX stream and `TYPE_ROTATION_VECTOR` at `SENSOR_DELAY_GAME`, pairing by the common capture timestamp timebase.
- SM-S9280/API 36 passed `OK (1 test)`: all 30 camera frames were bracketed by rotation samples. Nearest-sample delta was `4.81ms` median / `9.67ms` maximum; bracket span was `19.756ms` median / `19.756ms` maximum.
- Quaternion SLERP at each `ImageInfo.timestamp` is therefore supported by the observed sampling coverage. Interpolated pose accuracy, distortion-aware real-image reprojection error, route/LEFT-RIGHT events, App wiring and production authorization remain open.

### r8.36–r8.36a automatic real-frame gravity-axis reprojection

- r8.36 captured and persisted a real pose-linked display frame, predicted the world-up vanishing point through the full camera geometry, and ran a frozen Hough-line comparison. The upward-facing scene was extremely dark and the fixed Canny 60/160 pipeline returned zero candidates, so the result is non-informative rather than a geometry failure.
- r8.36a froze a disclosed low-light rescreen after visual diagnosis and reused the exact same frame without recapture. CLAHE plus Canny 15/50 found 89 candidates and 10 lines within 10 degrees; the smallest error was `.42°`. Aligned length fraction was `8.782%`, below the frozen 10% gate, so the result is an informative fail and thresholds were not changed afterward.
- Diagnosis: most long lines belong to horizontal ceiling-plane boundaries, invalidating the gate's assumption that all scene-line length is a useful denominator for the world-vertical family. The next independent test will compare IMU-predicted rotational homography `K R K^-1` with LK optical flow under a short automatic vibration, avoiding semantic line labels.

## 2026-07-20：USTRF-SC source radial-motion 分层与 V13

- 执行者：violjjet
- 新增独立 `:core:ustrf` pure-Kotlin 安全合同与回放模块、隔离的 `:ustrf-shadow-benchmark` Android 测试 App，以及公开/合成来源审计、REveL detector 与研究汇总脚本；未接入正式 App、默认 YOLO、risk/feedback runtime 或生产资产。
- 将既有 REveL detector/Vicon 对齐升级为 CPU/NumPy v2：严格原生 Vicon 时间包围、连续性/同步门、冻结 ±0.10m/s deadband、approach/recede/quasi-static、离线非因果 TTC-proxy、逐框 JSONL/hash 和 fail-closed 缺失原因；未改 App、默认 YOLO、runtime 或 GPU 路径。
- 770 个框中 range 仍为 502，motion 可用 488；三类 recall 为 `.93137/.90291/.90608`，TTC-proxy<3s 仅 10 个且 10/10。alignment/details 两次精确复跑哈希一致；5 个相关 Python test module 共 22 tests 通过。
- V13 为 `15 gate / 14 pass / CONDITIONAL_RESEARCH_GO`，新增 gate 仅获 `source-motion-stratification-only`；`device_metric_geometry_admission` 仍是唯一失败，`production_authority=false`。详细协议、CI、分母和证据见 `docs/research/ustrf-sc/USTRF_SC_RESEARCH_METRICS_2026-07-20.md`。
- GPU 调度改为风险分级：已稳定的同类 bounded 配置可按显存、温度和系统余量灵活选择 batch/规模；新型重负载或长跑才要求先做可停止 pilot、守护/分片 receipt。曾两次蓝屏的 8,580 帧、`batch=64/imgsz=320/FP16` 旧入口组合仍禁止复用。
- 验证：22 个 USTRF Python test 文件共 60 tests 通过；46 个 Python 文件 `py_compile` 与 PowerShell guard 语法检查通过；JDK 17 下 `:core:ustrf:test`、`:ustrf-shadow-benchmark` Kotlin/AndroidTest 编译和 `:device-benchmark:compileDebugKotlin` 通过；文档索引与仓库卫生检查通过。

## 2026-07-27：放宽近期开发日志容量预算

- 执行者：violjjet
- 将根 `DEVELOPMENT_LOG.md` 的结构门禁从 1500 行 / 300000 bytes 调整为 6000 行 / 1200000 bytes，解决中文研究记录在行数仍合理时过早触发字节上限的问题。
- 继续保留最老日期不超过 28 天的近期窗口和月度原文归档要求；本次不改变日志职责、历史归档路径或其他项目结构门禁。

## 2026-07-27：CID-SIMS floor3_2 cross-sequence development holdout

- 在首次 ZIP open 前冻结官方 transport、科学合同、新 runner/独立 validator 与 implementation lock；官方 `floor3_2.zip` 的 exact bytes、MD5 和本地 SHA-256 全部通过，geometry 固定使用 8 workers。
- 全部 18 个完整 10 秒窗的 geometry coverage 均为 1.0；角色为 `17 positive / 0 below-reference / 1 ambiguous`，无法形成精确 2+2，故 selected RGB identity/cache/ledger 未创建，RGB bytes 为 0，冻结 RGB algorithm 未运行。
- Formal validator 因 W3 偶数样本 median 的 Decimal/float 表示级精确比较保留 `CROSS_SEQUENCE_HOLDOUT_INVALID / INVALID`。Post-hoc R1 不重跑算法，只以 `rel_tol=1e-12, abs_tol=1e-15` 修复聚合数值等价，其余 identity/hash/ledger/role/selection 检查不变，结果 `errors=[] / VALID`；不回写 formal terminal。
- 最大权限仍为同源、同场景、不同 run 的 development holdout，不是 confirmation、cross-source generalization 或性能资格。Floor3 两个 run 都缺冻结低参考角色，后继只能另立 outcome-blind source discovery，不能自动用 floor3_3 或改门。

## 2026-07-27：冻结 R1 并启动未见数据外部确认

- 执行者：violjjet
- 冻结 `CAUSAL_THREE_PAIR_CONFIRMATION_R1`：严格 `> 0.01/s`、连续三 pair、所有
  abstention/窗边界/不满足门的 pair 均重置；旧版与底层 RGB/geometry 实现不变。
- 新建 outcome-blind 外部确认预注册和 F0 来源发现合同。正式 cohort 固定为两个
  ancestry-independent all-real 来源，每个来源各一个 10 秒 positive 与 below-reference
  窗；四门逐窗口、逐来源取逻辑 AND，pooled aggregate 只作诊断，失败后禁止换窗补救。
- 新建纯 pair-ledger 指标模块，同一遍历派生 old/R1，拒绝 `dt > 0.1s`、窗口/时间链漂移、
  角色缺失和非两来源四窗口输入；专项 `17/17` tests 通过。
- metadata-only 审计终态为 `CANDIDATE_NOT_FOUND / EXTERNAL_COHORT_NOT_EVALUABLE`：
  Ground-Challenge、MultiScan、ARKitScenes raw 与 SUN3D 均未同时闭合许可、精确
  payload identity 和 pose/depth/timestamp 绑定；OpenLORIS exact corridor 与 CoRBS
  authority 修复复核仍为 HOLD。该 discovery version 已按 stop condition 关闭；未下载候选
  payload、未读 claim-relevant RGB outcome、未消费正式 claim；Android/实时集成继续关闭。

## 2026-07-27：RCLE 证据访问与传输规则纠偏

- 用户明确指出“不得运行 RGB”未授权把压缩传输和 solid decoder 的瞬时经过也定义为
  RGB 使用；该机械禁令由 agent 自行引入，R1 因此产生了人工 transport terminal。
- 新增风险导向访问标准，将 transport presence、transient decode、内容物化、人工/
  模型查看、claim algorithm consumption 与 selection influence 分层记录。允许在
  identity、许可和 `40 GiB` 预算内选择 range/member/solid block/完整 archive，并允许
  技术上不可分离的 RGB 在内存中经过后立即丢弃。
- Source Discovery R2 access correction 的独立 stdlib review 为
  `PASS / errors=[] / EXECUTION_AUTHORIZED`。候选、exact capture、窗口、公式、科学门槛、
  fixed denominator 和 tie-break 全部未变；RGB 输出、缓存、查看、算法消费和选窗影响
  仍禁止，Android 仍关闭。后续等价 transport 切换只记 execution ledger，不再反复另立
  科学协议。
- 纠偏后科学流程消费的唯一 candidate payload 合计 `19.843352 GiB`。OpenLORIS materialize
  `8,518 + 3,481` 个 exact geometry 文件，MultiScan 每 capture 只物化 JSON、JSONL、
  depth；逐项 identity/bytes/CRC/hash 审计均 PASS，RGB 持久化/查看/算法调用为 0。
- geometry-only 运行完成全部 60 个固定窗：OpenLORIS 为
  `34 positive / 0 below / 5 ambiguous`，MultiScan 为 `0 / 0 / 21`。独立 validator
  从完整 pair ledger 重算 coverage、fixed-denominator fractions、连续段和角色，
  `errors=[] / PASS`。两个来源均缺精确 `1 positive + 1 below` tuple，role-complete
  source 为 `0/2`，科学终态保持 `EXTERNAL_COHORT_NOT_EVALUABLE`；原因是实际角色
  不足，不再是 transport firewall。未扩候选、降门、运行 RGB 或启动 Android。
- 交付复核发现旧 Hugging Face downloader 未随 selective-prefix 路径完成而退出：
  它重复取得完整 `corridor1-1`，并写入 `15,961,011,157` bytes 的非稀疏
  `corridor1-2` partial。保守累计 acquisition 为 `47.610525 GiB`，超过冻结
  `40 GiB` 预算 `7.610525 GiB`；两个冗余进程已停止，partial 保留。该 breach 不改变
  geometry ledgers 或角色，但 R2 completion audit 的 `PASS / within_budget=true`
  已撤销为 `FAIL / false`。

## 2026-07-27：正式 NPU 设备能力路由与回滚复验

- 正式 `com.linnan.blindassist` 通过 manifest provider 接入设备能力路由：仅
  `SM8650 + arm64-v8a + live QNN HTP FP16 capability` 走 QNN 2.47；不支持、
  API 26–30、能力或 delegate/graph 初始化失败均记录原因并走 CPU，VM 致命错误不吞掉。
- 最终正式 APK 在 SM-S9280 上命中 `qualcomm_qnn_htp`。100图风险、稳定风险、反馈和
  事件状态均与 CPU `100/100` 一致，P50/P95 为 `12/15 ms`；90帧事件 recall=1、
  关键漏报/重复提醒/事件再生均为0，最终退出率1。
- 正式路由600秒稳定性为5938帧、9.895 FPS、0失败、无安全终止，P50/P95
  `16/21 ms`，温度 `30.1°C → 30.1°C`，thermal status 最大0。
- CPU 基线 APK 以精确 SHA-256 回滚并冷启动成功，随后恢复 NPU 正式包。旧“17文件”
  指纹由错误的 `find -exec` 转义产生，已修复为逐文件枚举与单文件 SHA-256。当前设备
  用户自有状态文件为0，因此数据保留子项记为 `NOT_EVALUABLE`；两个版本相关的
  Android/ProfileInstaller 标记单独披露，不再伪装成用户数据。
- 最终 APK SHA-256 为
  `A1BD48CBDC41000477183BB8579725A9039818F4489563A9F7DE3643B966FDD5`，
  设备安装哈希一致；候选和 benchmark 包已删除，正式路由包保留。
- 时间：2026-07-28；执行者：violjjet。完成 `RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_GENERATOR_GEOMETRY_IMPLEMENTATION_R0` 的 P1-only 实现与独立验证：新增 deterministic analytic ray/rectangle z-buffer、MAIN/CAL/GUARD SHA-256 seed 派生、source-native `360×640` K、四条 602-frame endpoint-closed trajectory、80 main + 8 guardrail all-seed manifest、6 个解析 fixture、四 block 各 10,000 projective sample 和独立 validator。G01–G12、G14 PASS；G13 FAIL：冻结的 10 秒 exact 25% inverse-depth endpoint 只能产生约 `0.0223/s` radial expansion，达不到 `>=0.05/s`，且 approach-plus-periodic 的逐 pair depth monotonic fraction 为 `0.0`。按预注册 fail-closed 规则终态写为 `INTERVENTION_NOT_EVALUABLE / HOLD_P1 / EXECUTION_NOT_AUTHORIZED`；未换 seed、降门或进入 P2，也未读取/运行 RCLE output、P3、P4、sequence16、CoTracker、Android 或实时集成。详细证据见 [P1 result](docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_GENERATOR_GEOMETRY_IMPLEMENTATION_R0_RESULT_2026-07-28.md)。

## 2026-07-29：R2 response-blind quality calibration P2 R0

- 执行者：violjjet
- 新增 CAL-only source-known 32-edge plate、linear-RGB Gaussian PSF、pre-render material-albedo contraction，以及冻结 Laplacian variance、local RMS、multiscale gradient density 与 edge-spread 指标；专项实现/独立验证测试 `23/23 PASS`。
- 完成 `4 block × 4 CAL seed × 2 motion × 16 frame × 12 state = 6144` 行 response-blind ledger。Low-texture 最大可行值为 `alpha=0.15`；全部 blur 候选低于 Laplacian-ratio 下界，最小 `sigma=0.75` 的 overall/subgroup 仅 `0.132784 / 0.128384–0.136307`。
- 独立 validator 不导入 producer、quality implementation 或 RCLE algorithm，复算全部 ratio、层级、门、选择方向、hash/read allowlist/firewall，`errors=[]`。终态为 `NO_GLOBAL_QUALITY_STRENGTH / VALID / HOLD_P2`；未扩 grid、换 seed、做分 block strength，未运行 RCLE、P3/P4、sequence16、CoTracker、Android 或实时集成。
- 详细证据见 [P2 result](docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QUALITY_CALIBRATION_R0_RESULT_2026-07-29.md)。

## 2026-07-29：R2 P2 一次性 blur-grid repair R1

- 执行者：violjjet
- 在任何新增 CAL 访问前冻结 9 点小 sigma grid
  `[0.35, 0.40, 0.425, 0.45, 0.475, 0.50, 0.55, 0.60, 0.65]`；复用 R0
  的 512 个帧身份，仅生成 `1 clean + 9 blur` 的 `5120/5120` 行
  response-blind ledger，未重跑或重调 low-texture。
- `sigma=0.475 px` 是满足 overall 与全部 8 个 block×motion subgroup 门的最小
  候选：Laplacian ratio overall `0.525336`、subgroup
  `0.513451–0.533071`，local RMS overall `0.931832`、subgroup minimum
  `0.919136`。更小的 `sigma=0.45` 因 Laplacian ratio 高于 `0.55` 上界失败。
- 与 hash-bound R0 `alpha=0.15` 形成唯一全局 strength lock。独立 validator
  不导入 R1 producer、quality implementation 或 RCLE algorithm，复算全部
  5120 行、层级、8 subgroup、最小 sigma、R0 继承、hash/read allowlist/firewall，
  `errors=[] / validated=true`。
- 终态为 `QUALITY_CALIBRATION_PASS / VALID / P3_NOT_AUTHORIZED`。未运行或读取
  RCLE，未运行 P3、480+16、sequence16、CoTracker、Android 或实时集成，未换 seed、
  分 block、修改 R3/阈值/三-pair 或开启第二次修复。
- 详细证据见 [P2 R1 result](docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QUALITY_CALIBRATION_BLUR_GRID_REPAIR_R1_RESULT_2026-07-29.md)。

## 2026-07-29：R2 P3 qualified 与 P4 formal pre-R3 terminal

- 执行者：violjjet
- P3 已完成 R3 transport equivalence、analysis implementation/mutation、
  8 个固定 PREFLIGHT identity 的 W4/W8 guarded-host qualification；W8 successor
  实测 `677.507 s / 8 arms`，OpenBLAS 18、OpenCV 1，正式 496-arm 投影含
  10% reserve 为 `7.1575 h`，终态
  `PERFORMANCE_QUALIFIED / VALID / P4_NOT_ACTIVATED`。
- 随后按用户正式授权冻结 W8 scheduler amendment、精确
  `480+16=496` identity lock、formal runner/bundle closure/独立 validator，
  activation 前 P4 集成与 mutation tests `33/33 PASS`。
- 一次性 P4 activation 后，response-blind formal manipulation check 完成
  `80 cluster × 2 motion × 16 frame = 2560` frame-state evaluations。blur 八个
  subgroup 均 `20/20`；low-texture 在 ADVIO_13 periodic `17/20`、
  ADVIO_15 periodic `14/20`、ADVIO_17 static `17/20`、ADVIO_17 periodic
  `17/20`，未达到冻结的 `18/20`。
- 独立重算为 `VALID / INTERVENTION_NOT_EVALUABLE`，正式终态 receipt
  `validated=true / errors=[]`。按合同在任何正式 R3 之前停止：formal arm
  `0`、pair-core call `0`、outcome analysis `false`；未调 strength、换 seed、
  修改 R3/阈值/三-pair，也未访问 sequence16、Android 或 realtime。
- 详细证据见 [P4 formal result](docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_P4_FORMAL_RESULT_2026-07-29.md)。

## 2026-07-30：神经—几何双环 F-1A 至 F-1B 主线终结

- 执行者：violjjet
- 按用户 `F-1A_EXISTING_RGB_LABEL_REPAIR_ONLY` 与后续连续推进授权，固定既有 RGB
  修复 R0 保持 `HOLD_DATA` 不变；独立 R1 只从 development-only Ulm 既有 RGB
  补入 1 个双复核一致的静态负窗，合并账本达到 `17 positive / 20 negative /
  4 categories >=2 / 4 sessions`，F-1A 为 `READY / VALID`。两条单方低纹理候选经
  第三复核全部隔离，未降低门或回收 R0 项。
- F-1B0 在 `SM-S9280 / SM8650` 上补做 baseline-only 因果时序：24 条生产 QNN
  (`qualcomm_qnn_htp`) 与 24 条隔离 Sparse LK 记录均具完整 publish/available/consume
  顺序、同一 `ANDROID_ELAPSED_REALTIME_NANOS` 时钟与零未来读取。语义 available-age
  P50/P95 `86.017/107.773 ms`，几何 `9.397/11.309 ms`；终态 `READY / VALID`，
  未访问风险或提醒效果。
- F-1B 在 decision 候选输出零消费时完成现有 Sparse LK 与生产提醒状态机的结构可达性
  审计。R0 validator 因相信自报 truth-table boolean、漏绑下游状态和未证明 secondary
  endpoint，被独立复核判 `INVALID`；R0 凭据原样保留。
- protocol-only R1 虽绑定 13 个实现 identity，但独立复核发现侧向 temporal NEAR
  被错列为 HIGH，且非 planner-eligible CENTER/MID confirmation substitution 可能
  改变 stabilizer history，因此同样 fail-closed 为 `INVALID`。最终 R2 正确保持侧向
  `MEDIUM / 2-frame`，只允许 planner-eligible pair 触发确认替代，从规则派生 19 个
  fresh states，并以历史归纳覆盖 temporal promotion、stabilizer hold、side-person
  gate、event、cooldown、fatigue 与 effect acceptance。`fusion action reachable=0`，
  `EARLY_RESPONSE / RISK_DISCRIMINATION / RISK_CONTINUITY / MULTIPLE_INCREMENT`
  均零可达，首次实际提醒提前上界 `0 frame`；两条独立复核 PASS，终态
  `NO_INCREMENT / VALID`。
- 按冻结合同，双环论文主张在 F-1B 停止，F-1C、正式融合器、生产 CameraX 接线均未
  运行且不授权。decision 非访问只作为协议声明，不伪装成机器可证明事实；当前 claim
  ceiling 为 `DEVELOPMENT_ROUTE_REJECTION_ONLY`。

## 2026-07-30：target/track-conditioned causal radial geometry LITE R0

- 冻结 REveL single-capture Development-only 输入、13,014 个 target/ROI replay
  opportunities、469 个 primary parent natural events、两条最小 arm、target/region/
  truth-state 评价、100 ms TTL、abstention、失败与停止门；design review PASS。
- 实现 causal bbox log-area baseline、ROI sparse radial flow、pre-truth producer、
  post-keyset truth evaluator 与 24 个 synthetic fixtures；implementation review PASS。
  旧 F-1B decision 输出继续密封。
- 一次性 activation review PASS 后，唯一 full producer attempt 在 replay line 1,728
  的同目标同 epoch 相邻 RGB 尺寸 `260×346 → 258×346` 处触发
  `cv2.calcOpticalFlowPyrLK` size assertion。失败发生在任何 candidate ledger 写盘前；
  producer output/receipt/evaluation 均不存在，evaluator 和 truth join 未运行。
- 按预注册 one-shot/no-repair 规则终点为
  `EXECUTION_INVALID_STOP_NO_RERUN / NOT_EVALUABLE`。不得 resize/pad、增加 shape-change
  reset/abstention、修补或重跑 R0；若未来另行授权，只能新建并重新评审 evidence
  version。详细证据见
  [execution result](docs/research/dual-loop/DUAL_LOOP_TARGET_TRACK_CAUSAL_RADIAL_GEOMETRY_LITE_R0_EXECUTION_RESULT_2026-07-30.md)。

## 2026-07-30：causal radial geometry LITE R1/R2 Development 封存

- 执行者：violjjet
- 独立 R1 冻结跨尺寸 reset/abstention 后，formal producer 完成 13,014 输入与
  26,028 双臂输出，但共享 host guard 将 JSON UTC `Z` 时间戳误解释为本地时间。
  R1 按执行包络门关闭为 `EXECUTION_INVALID_STOP_NO_RERUN / NOT_EVALUABLE`；
  evaluator 未运行，完整 producer 输出不作科学救援。
- 修复 shared guard 的 DateTime/DateTimeOffset UTC 处理并加入 trailing-`Z` integration
  regression。独立 R2 仅继承冻结科学合同、绑定新 identity/namespace 与修复后的
  execution envelope；design、implementation、双线程等价 pilot、host preflight 和
  one-shot activation 均通过独立复核。
- R2 唯一 guarded producer 为 `COMPLETE`：13,014 输入、26,028 输出、32/64
  shape-change 账本、`truth_joined=false`。全部 pre-truth 门通过后，唯一 evaluator
  在冻结的 469 个 primary 自然事件上完成 Development join。
- box 面积增长为 204/469 correct、153/469 wrong-signed；ROI sparse radial flow
  为 188/469 correct、161/469 wrong-signed。flow 的 correct-event gain 为 `-16`，
  两个 target 与 LEFT/CENTER/RIGHT 三个区域增量均为负；两臂均未达到 readiness
  floor。
- 终点为 `BOTH_NOT_READY_FOR_CONFIRMATION / IMPLEMENTATION_NOT_READY`。独立执行
  封存复核 PASS；R2 不重跑、不调阈值救援，旧 F-1B decision 继续密封，Confirmation、
  Android、产品、运行时与安全均不授权。详细证据见
  [R2 execution result](docs/research/dual-loop/DUAL_LOOP_TARGET_TRACK_CAUSAL_RADIAL_GEOMETRY_LITE_R2_EXECUTION_RESULT_2026-07-30.md)。

## 2026-07-30：D0 ego-motion error attribution R0 协议冻结

- 执行者：violjjet
- 将 LITE R2 后继冻结为 burned REveL single-capture Development 诊断：469 个
  primary parent natural events 是分析单位；frame、pair 与 flow track 只作事件内
  重复测量。按 target 与 anchor region 做描述性检查，不做泛化 p 值。
- 冻结 person/sensor 径向分量闭合、相机光心角速度与平移、ROI 面积/中心抖动、
  事件长度、flow MAD、sign flip、features、surviving tracks、quadrants、FB error
  与 coverage 的 event-level 表；有限差分 chord range gradient 保证分量求和闭合，
  相机光心派生不改写既有 sensor-marker truth。
- 独立设计审查要求 temporal 路由必须包含直接时间不稳定指标加独立
  support/persistence 指标，并加入 approaching/receding composition guard 和
  可评价反向 region 禁止救援；低支持相关指标不能机械叠加为 temporal dominance。
- 科学出口只允许 `EGO_MOTION_DOMINANT`、`TEMPORAL_NOISE_DOMINANT`、
  `MECHANISM_NOT_IDENTIFIABLE`。只有首个出口可另立一次 EVIMO2v2 背景 affine
  补偿 canary；canary 再失败即停止路线。JRDB 仅在 canary 通过后承担人员域
  Development，Confirmation 仍需独立未调参 source/session。
- 本轮仅合同与设计审查 `PASS / NOT_RUN`；未实现、未执行、未下载新数据、未改算法，
  R2 不重跑，旧 F-1B decision 继续密封。详见
  [D0 protocol](docs/research/dual-loop/DUAL_LOOP_D0_EGOMOTION_ERROR_ATTRIBUTION_R0_PROTOCOL_2026-07-30.json)。
