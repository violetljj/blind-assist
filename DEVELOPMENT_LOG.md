# Development Log
- 时间：2026-08-09（Asia/Hong_Kong）；执行者：violjjet。完成 AG-QSF H1 parent-level TRAIN support audit 并关闭 R0 支线。逐 bytes/SHA 扫描同一 16 parent × 300 target NPZ，不读 RGB、模型、feature 或 Development/Confirmation；只有 `41159448` 有 right-censor（selected-64 `18`，全 300 帧 `83`），其余 15 parent 在 selected-64 和全 300 帧均为 `0`。因此不存在 parent-disjoint fit/eval 两侧 censor 都非零的 split，冻结 terminal `BLINDASSIST_ASSISTIVE_GEOMETRY_QSF_R0_CLOSED_DATA_SUPPORT_INSUFFICIENT`。这不反证 H1 数学假设，但当前 TRAIN target contract 无法评价 H1 learnability；H2 与组合版从未授权，路线无 successor。重开需 pre-outcome 新数据/target contract 在至少两个 parent identity 上提供 censor support，并另立独立版本。
- 时间：2026-08-09（Asia/Hong_Kong）；执行者：violjjet。AG-QSF H1 Attempt 03 的 fixed-setup pilot 以 model-load `2.793 s`、16-frame extraction `4.950 s`、全量投影/保守上界 `349.621/699.242 s`、峰值 `388 MiB` 合格；full run 逐 bytes/SHA 复核 1024 RGB/target 并完成 frozen feature，但在 head 训练前由分母前门停止：fit `event/censor/occupied=1213/18/3162`，eval `262/0/784`。签署 `H1_TRAIN_CANARY_NOT_EVALUABLE_DATA_SUPPORT`，未物化 checkpoint、未形成 learnability PASS/FAIL，H2 继续未授权。唯一 successor 收窄为无 RGB/模型/feature 的 TRAIN parent-support audit；冻结读取同一 16-parent target roster，eval 取 manifest 顺序中前 4 个 selected-64 event/censor/occupied/clearance-event 均非零 parent，并明确披露 support-based roster selection。36 项 QSF 测试与 machine protocol validator 通过。
- 时间：2026-08-09（Asia/Hong_Kong）；执行者：Codex。完成 Assistive Geometry B1 A0 三 seed 正式 TRAIN 与冻结 Development Selection 评价。seed `17/29/43` 均从 DepthART 初始化完成 `20 epochs / 6,000 steps` 和四个留存 checkpoint；seed 29 Attempt 01 的 2097-step CUDA OOM 保留失败收据，Attempt 02 不恢复中间状态，从共同初始化完整重跑。只物化固定四 parent / 1,200 帧 Selection，并完成 3,600 个 seed-frame 观察；Calibration 与 Confirmation 未打开，selected seed 始终为 null。首次 evaluator 因错误强制 seed 29 匹配原三 seed 协议 SHA 而在读取 observation 前 INTERNAL_FAILURE；修正仅按冻结的 Attempt-02 binding 校验该 seed，未改身份、阈值、观察值或聚合，12 个 checkpoint 全部复核后写入独立 r1。A0 前门 PASS，但 clearance MAE `0.3152 m > 0.20`、false-block `0.7501 > 0.02`、geometry transition agreement `0.7728 < 0.90` 均为 `0/3` seed 通过，签署 `B1_A0_DEVELOPMENT_EVALUATION_FAIL_TASK_GATES`。冻结 A1 条件 successor 未激活；A1–A4、M0、C0、D0 执行均未授权，Selection 已消费且不得复用，重开需新的 pre-outcome 假设与独立选择证据。
- 时间：2026-08-09（Asia/Hong_Kong）；执行者：violjjet。AG-QSF H1 performance pilot Attempt 02 用独立 namespace 重锁 feature batch 16；16 帧 feature 仍 finite，峰值 VRAM `733 MiB`，但 combined setup+extraction `9.995 s`、全量投影 `669.664 s`、conservative maximum `1399.327 s > 900 s`，再次形成 performance-only `NOT_QUALIFIED`，仍未训练 head 或生成 checkpoint。对 Attempts 01/02 的估算器审计发现固定的一次性 DepthART model load 被错误乘以 `full/pilot=64`。Attempt 03 恢复原 batch 4，只把 estimator 冻结为 `model_load + variable_extraction × 64 + 30 s`，maximum 为投影的 2 倍；科学模型、loss、roster、frame selection 与 gates 不变，并切换到 Attempt-03 namespace。唯一 successor 为 `BLINDASSIST_ASSISTIVE_GEOMETRY_QSF_H1_TRAIN_CANARY_ATTEMPT_03_FIXED_SETUP_ESTIMATOR_PILOT`。
- 时间：2026-08-09（Asia/Hong_Kong）；执行者：violjjet。AG-QSF H1 performance pilot Attempt 01 在 foreign B1 formal runner 自然退出、runtime preflight 返回 READY 后执行；逐 SHA 复核 16 RGB/target，BF16 frozen feature 全 finite、shape `16×3×48`、峰值 VRAM `388 MiB`、实测 `8.549 s`、全量投影 `577.126 s`，但冻结的 conservative maximum `1214.252 s > 900 s`，因此签署 `H1_TRAIN_CANARY_PERFORMANCE_NOT_QUALIFIED`。该 pilot 未训练 task head、未生成 checkpoint、未打开科学 outcome；失败范围仅为本 evidence version。Attempt 02 只把 feature-extraction batch 从 4 重锁为 16，模型/损失/roster/frame selection/scientific gates 全部不变，并切到新的 evidence/model/work namespace；唯一 successor 为 `BLINDASSIST_ASSISTIVE_GEOMETRY_QSF_H1_TRAIN_CANARY_ATTEMPT_02_BATCH16_PERFORMANCE_PILOT`。
- 时间：2026-08-09（Asia/Hong_Kong）；执行者：violjjet。根据独立审计加固 AG-QSF H1 TRAIN canary lock：将 H1 protocol 明确冻结为 exact-three-input embedded shared-resource manifest，target 访问从不实的 metadata-only 修正为 `CONTENT_INSPECTED / TRAIN_TARGET_INPUT_ONLY`，并 hash-bind 运行时导入的全部项目代码与专项测试。runner 在模型使用前逐个复核所选 RGB/NPZ 的 producer bytes/SHA-256；fit/eval 的 event、right-censor、known-occupied、clearance-event 任一分母为零即形成 `H1_TRAIN_CANARY_NOT_EVALUABLE_DATA_SUPPORT`，不物化 checkpoint、不以伪分母继续。通用 preparation validator 新增 B1 producer/role/path-specific Development/Confirmation deny、schema-only 窄门和混合角色 B0 source 的 TRAIN-only file-manifest 要求，同时保留 formal TRAIN OOM diagnostic 与非 B1 generic policy 正例。33 项 QSF 测试及 H1 input lock 验证通过；runtime 仍因 foreign B1 formal runner 返回 `H1_CANARY_DEFERRED_RESOURCE_ISOLATION`，未启动 QSF GPU 或重 I/O。
- 时间：2026-08-09（Asia/Hong_Kong）；执行者：violjjet。关闭 AG-QSF 的 `H1-only implementation + TRAIN canary lock` successor。实现四桶 `0.5/1.0/1.5/2.0 m` robust q-contact hazard、严格 event/right-censor/UNKNOWN 编译、结构单调 CDF、horizon-capped clearance、right-censored NLL、false-clear 与独立 confidence loss；H1 与 direct task head 均为 `8,678` 参数。冻结 16-parent TRAIN 的 `12 fit / 4 eval` parent-disjoint canary，每 parent 取 source-order-even 64 帧，仅在 GPU 空闲时提取 frozen DepthART pooled band feature，再在 CPU 训练 head 50 epochs。H1 lock 明确拒绝额外/Development/Confirmation input、H2 和组合 authority；15 项 target/gradient/flip/checkpoint/zero-support/参数/协议/资源恶意反例测试通过。runtime preflight 现场发现 foreign B1 formal runner 活跃，因此正确返回 `H1_CANARY_DEFERRED_RESOURCE_ISOLATION`，未启动 QSF GPU、未读取 B1 checkpoint/progress 或 Development/Confirmation outcome。唯一 successor 为 `BLINDASSIST_ASSISTIVE_GEOMETRY_QSF_H1_TRAIN_CANARY_PERFORMANCE_PILOT_THEN_RUN_WHEN_FOREIGN_GPU_IDLE`。
- 时间：2026-08-09（Asia/Hong_Kong）；执行者：violjjet。建立独立 `BLINDASSIST_ASSISTIVE_GEOMETRY_QSF_R0` 并行 WILD_LAB 路线，为 H1 censored robust-contact survival 与未来 H2 profile-conditioned swept configuration clearance 提供独立 current、Module、机器准备协议、共享资源 manifest 和输出所有权 validator；当前只授权 H1-only 实现，H2 仅保留非可执行占位。允许按 producer/path/version/provenance/license/data-role/outcome-access/selection-influence 逐项登记后只读共享 source、TRAIN cache、冻结初始化、几何合同、fixture、工具和 operational lesson；QSF 的 target、checkpoint、optimizer/RNG、metrics、progress 与报告一律 copy-on-write 到 `artifacts.local/{work,models,evidence}/assistive-geometry-qsf/`。B1 Development/Confirmation、active checkpoint/progress、selection/threshold/stop decision 与可变目录禁止共享；B1 正式 seed 运行时 QSF 只做 CPU/synthetic/light-I/O，不竞争 GPU 或长物化。当前仅授权协议、H1 实现和 synthetic mechanics，真实 TRAIN canary、H2 实现、H1+H2、Android/HTP、默认 App、产品与 safety 均未授权；B1 successor 不变，QSF 唯一 successor 为 `BLINDASSIST_ASSISTIVE_GEOMETRY_QSF_H1_ONLY_IMPLEMENTATION_AND_TRAIN_CANARY_LOCK`。
- 时间：2026-08-09（Asia/Hong_Kong）；执行者：violjjet。并行关闭 DepthART task-preserving D1 的 pre-outcome contract/metadata-roster 门，未占用 Assistive Geometry A0 正式训练的 GPU。D0 三臂负终态与 strict G4-D 保持不变；冻结 CameraX `640×480 / 4:3`、display-upright portrait、full-FOV `1×3×608×448`、动态 K/gravity fail-closed、`left/center/right × 1.0/1.5/2.0m` 同后处理和与 R2 逐字段相同的质量门。metadata planner 同时扫描冻结提交中 HFTF/Assistive Geometry 605 份文档，排除 163 个既有官方身份，锁定 8 primary + 8 reserve ARKitScenes Training visit/session，零媒体、truth/model outcome 访问；8 项合同/防漂移测试 PASS。当前执行仍未激活，唯一 successor 为 reviewed use-scope 扩展后按冻结主备顺序执行 label-blind portrait/pose/RGB-D continuity preflight；产品图重建、任务 outcome、性能、R2、DA2 replacement、默认 App、生产与 safety 均未授权。
- 时间：2026-08-09（Asia/Hong_Kong）；执行者：violjjet。完成 Assistive Geometry B1 A0 depth-only three-seed TRAIN execution lock。冻结 16-parent/4,800-frame deterministic parent-balanced order、portrait/landscape 同方向 effective-batch 16 与跨 epoch carry、seed `17/29/43`、20 epoch、6,000 optimizer steps/seed、AdamW `2e-5`、300-step warmup + cosine-to-0.05x、gradient clip 1.0、BF16/FP16 fallback 和包含 model/optimizer/scheduler/scaler/sampler/RNG/protocol 的 checkpoint。真实 TRAIN smoke 在 BF16 下各执行 1 个 portrait/landscape optimizer step，loss `0.95634/1.48731`、clip 前 gradient norm `6.82721/6.62860`、每步 616 个 encoder/depth 参数非零梯度，epoch-0 计划 299 步并保留 carry `4/12`，峰值 CUDA memory 2,053,701,632 bytes，缺失 Autograd 警告为 0。Attempt 1 的 CUDA map-location 使 CPU RNG ByteTensor 恢复失败，保留为 HOLD；Attempt 02 改为 CPU load 后全状态 checkpoint roundtrip exact。未启动正式训练或读取 Development/Confirmation outcome；唯一 successor 为 `BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_DEPTH_ONLY_THREE_SEED_FORMAL_TRAIN_EXECUTION`。
- 时间：2026-08-09（Asia/Hong_Kong）；执行者：violjjet。完成 Assistive Geometry B1 dual-orientation target/model implementation lock。只物化冻结的 16 个 TRAIN video/visit 共 4,800 帧（portrait 2,724、landscape 2,076），逐一关闭文件大小/SHA、NPZ schema、方向/K、gravity/ground、clearance/occupancy 与 UNKNOWN 防泄漏；得到 ground-plane-valid 3,424 帧、已知 clearance band 6,991、occupancy cell 21,060、confidence-valid band 6,990。K bit-exact 复核的唯一差异量化为一次 FP32 舍入（`3.0517578e-05`），据此冻结 one-ULP 门而未放宽任务语义。实现 DepthART shared 48-channel feature 上的 Ground/Clearance/Occupancy/Confidence heads 和 A0–A4 losses；首次部署 `torch.library` smoke 因缺失 Autograd-key 注册警告保留为 HOLD，Attempt 02 训练路径直达 `_SelectiveScanAutograd`，其 forward 与注册路径 bit-exact，双方向完整 checkpoint loss/gradient finite，每个方向有 616 个 encoder/depth 和 12 个 head 参数非零梯度，缺失 Autograd 警告为 0。未启动训练或读取 Development/Confirmation outcome；唯一 successor 为 `BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_DEPTH_ONLY_THREE_SEED_TRAIN_EXECUTION_LOCK`。
- 时间：2026-08-09（Asia/Hong_Kong）；执行者：violjjet。在 B1 implementation 前 pose-only audit 中发现 Attempt 1 单一 portrait `608×448` 协议不满足 full-FOV 数据几何：4,800 个 TRAIN 帧中 portrait 2,724、landscape 2,076，强行单 shape 会丢 43.25% 数据或引入裁剪/重力旋转错误；原 calibration 四 parent 也只有 30 个 portrait 帧、没有 portrait-dominant parent。所有媒体/task outcome/model output 均未打开，因此将 Attempt 1 保留为 pre-outcome superseded negative，并冻结 Attempt 2：portrait `608×448` + landscape `448×608`、orientation-bucket batch、对应 full-FOV K 更新、orientation-stratified reporting；Development 在 outcome 前重分，使 calibration/selection 各含一个 portrait-dominant parent，但 portrait confidence claim ceiling 仍为 single-parent Development-only。正式训练继续未授权；唯一 successor 为 `BLINDASSIST_ASSISTIVE_GEOMETRY_B1_DUAL_ORIENTATION_TARGET_AND_MODEL_IMPLEMENTATION_LOCK`。
- 时间：2026-08-09（Asia/Hong_Kong）；执行者：violjjet。完成 Assistive Geometry B1 confidence/training protocol lock，仍未启动训练或读取任何 DEVELOPMENT/CONFIRMATION outcome。冻结 `DepthART-S + 48-channel stride-4 shared DPT feature + dense ground + fixed-third band MLP`，A0–A4 additive arms，八项 loss lambda、`0.25–2/2–5/5–6m=3/2/1` 近场权重、AdamW/20 epoch/effective batch 16/三 seed、checkpoint 与 selection 顺序。关闭三项语义冲突：primary confidence 为 band-level `[3]` 并重复到 `[3,3]` interface；无 intrusion clear 作为 censored-clear，只监督 occupancy/confidence；A0 的 gravity postprocess 使用所有 arm 共享的 exogenous `up_camera`，缺失时 UNKNOWN。原 8 个 DEVELOPMENT identity 在 outcome 前固定拆成 4 calibration / 4 selection，CONFIRMATION 8 个继续 sealed。当前只授权 TRAIN target materialization 与 model/loss smoke，正式训练仍未授权；唯一 successor 为 `BLINDASSIST_ASSISTIVE_GEOMETRY_B1_TARGET_MATERIALIZATION_AND_MODEL_IMPLEMENTATION_LOCK`。
- 时间：2026-08-09（Asia/Hong_Kong）；执行者：violjjet。完成 BlindAssist Assistive Geometry B0 truth reader 与 registration lock：冻结 ARKitScenes 官方毫米深度、inverse trajectory、SLERP pose、逐帧 upright RGB/depth/confidence/K、gravity-ground 与 Left/Center/Right body-swept clearance/UNKNOWN 实现。修复 upsampling ZIP 跨模态同 stem 误报、9 个候选中 3 个实际属于 DEVELOPMENT 的角色冲突（下载前 fail-closed，最终仅 6 个 TRAIN）及 `ground_valid` 误标全部有效深度的语义错误。TRAIN-only 157 个 AppleDepth/FARO 帧逐帧组合门通过率 `94.27%`，1,151 个 occupied decision 一致率 `95.48%`；主 TRAIN 固定 480 帧 ground/all-band opportunity `71.04%/70.83%`，16/16 视频可形成 ground，UNKNOWN clearance leakage 为 0，16 项 gate 全 PASS。保留 ground/clearance 最大差异 `0.754/2.205 m` 为 tail negative evidence；结果只授权 B0 sensor-derived reader，不是 human safety truth，B1 training、DEVELOPMENT/CONFIRMATION、HTP/default App/product/safety 仍未授权。唯一 successor 为 `BLINDASSIST_ASSISTIVE_GEOMETRY_B1_CONFIDENCE_THRESHOLD_AND_TRAINING_PROTOCOL_LOCK`。
- 时间：2026-08-09（Asia/Hong_Kong）；执行者：violjjet。完成 BlindAssist Assistive Geometry B0 data capability 与 ARKitScenes roster lock：metadata ledger 的 139 个 RGB-D session 中 66 个仅达结构候选；另从官方 metadata 冻结全新 visit/video-disjoint `TRAIN/DEVELOPMENT/CONFIRMATION=16/8/8`，排除 101 个已跟踪既有/失败身份并保留 consumed 120-frame 与 DepthART R2 防火墙。三个失败版本分别定位到 `159/160` HEAD（单个 trajectory HTTP 403）、源包仅 `219/300` 公共帧、以及 earliest-common 窗口 `32/32` 不完全位于 trajectory 域（最差 `78/300`）。未降 300 帧门槛、未改 roster；Attempt 4 重新物化轨迹域内 9,600 帧，逐 SHA 并实际解码 28,800 张 RGB/depth/confidence、解析 9,600 个内参映射，最大 pose 插值包络 `116.62 ms <= 250 ms`，label-blind integrity PASS。当前只授权 B0 数据能力与 roster；depth unit/registration/pose convention/ground/clearance truth 尚未关闭，B1 training、模型 outcome、默认 App、产品与 safety 均未授权。唯一 successor 为 `BLINDASSIST_ASSISTIVE_GEOMETRY_B0_TRUTH_READER_AND_REGISTRATION_LOCK`。
- 时间：2026-08-08（Asia/Hong_Kong）；执行者：violjjet。完成 DepthART G4-D 首个 SelectiveScan 前纯标准算子 ORT/SM8650-HTP parity bisect：冻结 canonical ONNX、`fixed_integer_formula_v1` RGB input、ORT `1.27.0` 与 `rtol=3e-5 / atol=3e-6`，从首个 scan 第一个输入反向裁剪 120 个唯一依赖节点、80 个 float checkpoint，以单终点子图执行 `79→70→35→0` probe。prefix 终点精确复现完整图的 `max_abs=0.008583426`；首个可观察异常即 node 1 `/patch_embed/patch_embed.0/c/Conv` 输出，ORT↔HTP `max_abs=9.006e-4`，而同 DLC QNN CPU↔ORT `max_abs=3.58e-7` 且 PASS。结论只收敛到 HTP 特有的 layout/precision lowering 或首 Conv primitive，不能继续断言内部罪因。G4-D 保持 FAIL；PyTorch↔canonical ONNX 仍未关闭，G4-E/F 继续 `NOT_EVALUATED`，DA2 replacement 继续 `NOT_AUTHORIZED`。
- 时间：2026-08-08（Asia/Hong_Kong）；执行者：Codex。执行 DepthART G4-C full-context gate：在 `SM-S9280 / SM8650 / HTP v75` 上复用冻结 850-op canonical DLC 与已通过的 SelectiveScan package，完整图成功 load/register/compose 并进入 HTP prepare，但 finalize `1002`。首个确定性 frontier 为 `/sfh/decoder/layers.0/norm1/LayerNormalization` 的 disabled `q::layernorm_2d_fp16_oneshot_moments_sf`。23 个 LayerNorm 的等价 rank-4 包装仍命中同一实现；标准公式展开及 rank-4+展开将 frontier 前移到 disabled `q::reduce_mean.fp16`，仍不能生成 context。签署 `G4-C_CONTEXT_HOLD_LAYERNORM_REDUCE_FP16`；G4-D full-model parity、G4-E partition purity、G4-F 性能继续未授权/未评价，不影响 G4-A/G4-B 单算子 PASS。
- 时间：2026-08-08（Asia/Hong_Kong）；执行者：violjjet。完成 DepthART SelectiveScan HTP 工具链与 package compile milestone：通过已登录的 Qualcomm QPM3 激活 HexagonSDK5.x Core license，安装并登记 Hexagon SDK `5.5.5.0`（Tools `8.7.06`、v73 libraries），并接入 `E:\codex-tools\bin\depthart-deploy-env.ps1`。QAIRT generator 从冻结 XML 生成 package interface；仓库构建脚本将 correctness-first kernel 编译链接为 v73 `elf32-hexagon` `.so`（65,616 bytes，SHA-256 `8A8E7B07...AE662`）与 Android AArch64 prepare-side `.so`（892,448 bytes，SHA-256 `289D7001...F1103`），两端均导出 `DepthArtSelectiveScanPackageInterfaceProvider`。新增只允许写入 `artifacts.local` 的可复现构建脚本与合同测试。当前仅为 `HTP_V73_AND_AARCH64_PACKAGE_COMPILED / RUNTIME_NOT_EVALUATED`；算子 parity、QNN context、HTP/设备执行、partition、latency/thermal 均未完成，G4 仍 `NOT_EVALUATED`。
- 时间：2026-08-08（Asia/Hong_Kong）；执行者：violjjet。启动 SelectiveScan HTP Op Package runtime-kernel spike：新增冻结 `G=4/N=8/L=196` 的 float32 scalar reference kernel，逐 channel 使用 8-float stack state、无 heap，完整实现 stable softplus/transition/B/C/D recurrence，源码合同测试通过。QAIRT 2.47 官方本地文档确认 SM8550/v73 编译需 Hexagon SDK 5.5.5 + Tools 8.7.06；本机没有 QPM3/SDK，普通 clang probe 在官方 HTP headers 的 `HVX_Vector` 缺失处停止，因此源码严格记为 `SOURCE_READY_NOT_COMPILED`。已自动安装公开依赖 Android NDK r26c 并登记 deployment env；Hexagon SDK 需用户完成 Qualcomm/QPM3 登录后继续。G4 未评价。
- 时间：2026-08-08（Asia/Hong_Kong）；执行者：violjjet。完成 DepthART SelectiveScan exact primitive-lowering feasibility：按冻结公式将每个 `L=196` recurrence 展开为 3,730 个标准 ONNX 节点，真实 `C=48/128/336/672, G=4, N=8` 随机 canary 的 ORT/reference parity 在 `rtol=3e-5 / atol=3e-6` 下通过；完整图 `2,723 -> 21,368` ONNX nodes，QAIRT 2.47 仍成功转换。但优化后 QNN IR `850 -> 21,440` ops（25.2×）、DLC `32,003,812 -> 47,687,076` bytes（1.49×）、转换约 789 秒，并保留 196 级串行链。因此签署 `TECHNICALLY_CONVERTIBLE_NOT_SELECTED_AS_CURRENT_MOBILE_IMPLEMENTATION`：primitive 图只作 parity oracle/upper bound，下一步转最小 HTP Op Package runtime kernel；G4 仍未评价。
- 时间：2026-08-08（Asia/Hong_Kong）；执行者：violjjet。完成 DepthART SelectiveScan converter-mapping feasibility：新增 `com.depthart::SelectiveScan` OpDef 与仅提供 shape/type inference 的 converter DLL，QAIRT 2.47 在 `--target_backend HTP` 参数下成功转换并写出 32,003,812-byte DLC（SHA-256 `6ACD65D82FF3C0ABC7E1BC4787FCBA881D7E5CC4F5D48722F00F814D897DC680`）。优化后 QNN IR 为 850 ops，5 个 SelectiveScan 均保留正确 rank-3 shape 与参数，且跨过后未出现新的 normal-converter hard blocker；LayerNorm/Resize 被转换，Erf 不在最终图。该 DLL 不是 QNN/HTP runtime kernel，因此只签署 `G3-C_CUSTOM_MAPPING_PASS`；primitive lowering、canonical end-to-end parity、QNN context、HTP/设备执行、latency/thermal 均未评价，G4 不变。
- 时间：2026-08-08（Asia/Hong_Kong）；执行者：violjjet。优化提交/推送门禁耗时：
  `scripts/check_repo_hygiene.ps1` 默认只执行仓库卫生检查，不再隐式重复结构扫描；
  仅 `-IncludeStructure` 时串联 `check_project_structure.ps1`，并保留
  `-SkipStructure` 兼容参数。普通提交使用 staged/task-owned `git diff --check`
  和相关测试；结构/政策变更单跑 structure；push/交付单跑 hygiene；确需两者时
  只调用一次 `check_repo_hygiene.ps1 -IncludeStructure`。
- 时间：2026-08-08（Asia/Hong_Kong）；执行者：violjjet。按效率审计收窄 Codex 工作约束：
  `DEVELOPMENT_LOG.md` 仅记录 durable decision/架构与 interface 变化/研究结论/重要验证/
  材料失败；冷启动读取改为“默认两个入口、直接依赖或冲突时可扩展”；Android build 仅由
  runtime、共享接口、resources/assets、权限、构建配置或不确定跨模块影响触发；远端 parity
  仅用于 push、交付候选或明确发布；六项合同对小任务隐式维护。同步将 AI-review 与研究
  authority 的详细语义路由回对应 current 文档，并压缩全局子代理固定模板。
- 时间：2026-08-08（Asia/Hong_Kong）；执行者：violjjet。精简根 `AGENTS.md`
  的启动上下文：保留权限/Git/研究 authority 硬边界、八类按需文档路由、六项执行
  合同入口和机械验证命令；将最小读取、日志输出预算、任务切换、handoff 与共享工作树
  细则迁入 current `docs/CODEX_WORKFLOW.md` 并登记文档索引。目标是让普通 Kotlin
  修改不再默认加载完整研究、设备、发布和长任务协议，同时不删除或降级对应 current
  文档的 authority。
- 时间：2026-08-08（Asia/Hong_Kong）；执行者：violjjet。采纳 `WILD_LAB + EVIDENCE_TRACK`
  双轨研究风格：新论文/算法工作默认允许大胆的跨数据集、Teacher/pseudo-label、合成、
  自监督和超出当前 Android/模型大小/默认 YOLO 约束的探索；只有 Confirmation、Deployment
  或 claim-critical 问题才激活 Evidence Track 的独立验证与完整门禁。保留四条硬线：
  不泄漏 blind、UNKNOWN 不当 negative、source/derived provenance 分离、claim ceiling
  与证据匹配。更新 `docs/RESEARCH_GOVERNANCE.md`、`configs/research_governance_v4.json`、
  `docs/SANPO_CURRENT_STATUS.md` 和 `AGENTS.md`；不改默认 App、不改历史 receipt。
- 时间：2026-08-08（Asia/Hong_Kong）；执行者：Codex。完成 DepthART fixed-S448 static-shape R0：静态求值证明 6 个 Expand 的 target 均为全 1 shape，对各自 `[1,C,H,W]`/`[1,8,128]` 输入为严格 no-op；4 个 Mod 均为常量 `2 mod 3 = 2`。旁路/折叠并反向 DCE 后节点数 `2823 -> 2723`，输出 SHA-256 `9C98479915FF2A34303DCD1E3C39638AE1B39023058CF36365A9C698E0BE07D5`，专属+hygiene+Einsum tests `7/7`。QAIRT normal frontier 仍首先停于 5 个 SelectiveScan；dry-run 候选收敛为 `Erf 27 / LayerNormalization 23 / Resize 13 / SelectiveScan 5`。未改写 LayerNorm、Resize 或 Erf，HTP 仍未评价。
- 时间：2026-08-08（Asia/Hong_Kong）；执行者：Codex。完成 hygiene 后的 shape-only 只读归因：6 组 `ConstantOfShape/Where/Expand` 都由 `Shape`/`Equal`/`Reshape` 生成 broadcast 形状并下游进入 Conv/Concat/Add，4 个 `Mod` 输入均来自 Constant。它们进入下一轮 fixed-S448 constant-fold 候选，但本节点不改写，也不触碰 LayerNorm/Resize/Erf；normal frontier 仍以 SelectiveScan 为唯一 confirmed blocker。
- 时间：2026-08-08（Asia/Hong_Kong）；执行者：Codex。按 `normal-conversion frontier + parity-preserving minimal rewrite` 完成 DepthART Graph Hygiene R0。审计确认 123 个 BN 全为 `training_mode=0`、108 个 Reshape 全为 `allowzero=0`；4 个 AveragePool 均 `ceil_mode=0`、pads 全零，故移除这些显式默认/零 padding 等价属性，共 239 项，图仍为 2823 nodes。专属 hygiene + Einsum tests `4/4` 通过，输出 SHA-256 `94D12AC706DC4A6F4DAC7B643839B60199F50234BE69FF60725351E6359F39A2`。QAIRT 2.47 HTP normal conversion frontier 未漂移，仍首先停止于 5 个 `onnx_selectivescan`；未主动改写 LayerNorm、Resize、Erf 或 shape path，HTP 仍 `NOT_EVALUATED`。

Active window: 2026-07-20 onward. Older July entries are archived in [2026-07](docs/history/development-log/2026-07.md).

- 时间：2026-08-07（Asia/Hong_Kong）；执行者：Codex。完成 QAIRT 2.47 Python 3.10 converter runtime 补齐（`E:\codex-tools\venvs\qairt310`，NumPy/ONNX/PyYAML/protobuf/scipy/packaging），并对外提 camera 图执行 HTP `--dry_run`。正常转换首个停止点仍为 5 个 `onnx_selectivescan`；dry-run 另枚举 `Erf` 27、`LayerNormalization` 23、`Resize` 13、`ConstantOfShape` 6、`Expand` 6、`Where` 6、`Mod` 4，以及 `BatchNormalization.training_mode` 123、`Reshape.allowzero` 108、AveragePool 属性 4。dry-run 仅为 diagnostic，不把这些候选直接升格为 conversion blocker，也不产生 HTP/partition authority。R1 A3 口径保持 G3-C `BLOCKED_SELECTIVESCAN`，但后续必须逐层清理/复核这些候选。
- 时间：2026-08-07（Asia/Hong_Kong）；执行者：Codex。完成 DepthART R1 A3 graph deployment follow-up：保留 reference `image,K -> depth`，新增 host Camera Embedder externalization（四级 `camera_prompt_*`），PyTorch prompt parity `max_abs=0.0`。外提图 2823 nodes、`Acos=0`、5 个 SelectiveScan；QAIRT 2.47 在 Einsum→MatMul 等价改写后已真正触达 `onnx_selectivescan`，当前日志为 `No translation registered for op type onnx_selectivescan`。因此正式口径更新为 `G3-A Export PASS / G3-B Numerical Parity PARTIAL_PASS / G3-C BLOCKED_SELECTIVESCAN`，Gate 4 HTP `NOT_EVALUATED`；这不是 HTP PASS 或 FAIL。R0 `DEPTHART_ADMISSION_R0=FAIL`、DA2 frozen baseline/teacher/fallback、Android default/production/safety authority 均不变。receipt：`artifacts.local/evidence/hftf/depthart-admission-r1/camera-externalization-receipt.json` 与 `qairt/blocker-ledger.json`。
- 时间：2026-08-07（Asia/Hong_Kong）；执行者：Codex。补齐 DepthART A3 Windows deployment toolchain：安装 MSVC Build Tools 17.14 与 CUDA 12.8.93，针对 RTX 5060 / SM 12.0 编译 Selective Scan extension，核心 CUDA/forward/backward tests `9/9` 通过。legacy exporter 成功生成 31,985,722-byte metric S448 `image,K -> depth` ONNX（3555 nodes、5 SelectiveScan、SHA `06A0C059...78C`）。确认 QAIRT 2.47 原已在 `E:\codex-tools\qairt`；converter 原图先拒绝 10 个 Einsum，等价重写为 MatMul 后继续至 Camera Embedder `Acos`，因无 translation 停止，尚未评价 SelectiveScan。故 A3 更新为 `ONNX_EXPORT_PASS / QNN_CONVERSION_BLOCKED`，HTP/Android/production authority 仍关闭。
- 时间：2026-08-07（Asia/Hong_Kong）；执行者：Codex。完成 `DEPTHART_ADMISSION_R1` A3 deployment preflight：PyTorch 2.11 新 exporter 在 `depthart.selective_scan` custom op translate 阶段停止，legacy exporter 又因缺少 `depthart_selective_scan_cuda` 停止；本机亦未找到 QAIRT/QNN converter/runtime 工具。没有生成 ONNX graph，故 A3 严格记为 `NOT_EVALUABLE / DEPLOYMENT_PREFLIGHT_BLOCKED`，不产生 ONNX parity、QNN、HTP、Android 或生产 authority。R0 `FAIL` 与 DepthART 研发主线边界不变。receipt 见 `artifacts.local/evidence/hftf/depthart-admission-r1/a3-onnx-qnn-preflight.json`。
- 时间：2026-08-07（Asia/Hong_Kong）；执行者：violjjet。根据 R0 终态启动
  `DEPTHART_ADMISSION_R1`，明确 R0 `FAIL` 永久不改写，但将 DepthART 设为研发主力候选。
  冻结 A0 内参/预处理审计、A1 false-block 分解与 contact sheet、A2 relative truth-aligned
  diagnostic control、A3 ONNX/QNN graph preflight；relative 的 truth scale 不具部署权威，
  R1 新 holdout 的非对称 false-clear/false-block 规则只能预注册后使用。
- 时间：2026-08-07（Asia/Hong_Kong）；执行者：violjjet。冻结
  `DEPTHART_ADMISSION_R0`：保留 DA2 metric 518 canonical 为 baseline/teacher，唯一
  首轮候选绑定官方 DepthART 提交 `0384521` 与 indoor S checkpoint
  `597631AC...667E65`。新增 hash-bound materializer、距离分段和时序诊断、任务/时序
  fail-closed AND gate 及单测；AbsRel 仅作诊断，不得抵消 clearance/false-clear/时序
  失败。120 帧 TUM 仅为 consumed Development regression；ONNX、Snapdragon、Android
  default、产品和安全 authority 均保持关闭，FRESH-TF pause 不变。
- 时间：2026-08-06（Asia/Hong_Kong）；执行者：violjjet。将 AtomS3R-M12 外设视频接入
  Android 现有感知链。新增 `GLASSES_HARDWARE` 与 `AtomS3rMjpegFrameSource`，按固件
  multipart `Content-Length` 读取 JPEG，读线程容量 1 覆盖旧完整帧，解码后同时更新
  App 预览并进入既有 `ObjectDetector -> AssistSessionCoordinator -> 语音/震动` 路线。
  每帧绑定 `X-Frame-Sequence`、capture、ToF timestamp/valid/range/age；设备时间显式
  标记 `EXTERNAL_DEVICE_MONOTONIC_UNMAPPED`，未对时前风险事件使用 Android decision
  clock，禁止跨时钟直接比较。ToF 仅保留为逐帧诊断元数据，不改变风险算法，相机—ToF
  标定继续暂停。增加 multipart/metadata fail-closed JVM 测试和 SM-S9280 真实流五帧
  instrumentation；本地 `core:assist`、`core:device`、`feature:assist` 测试及 debug APK
  构建、instrumentation 源集编译通过。手机无线调试恢复后，debug APK 覆盖安装至
  SM-S9280（Android 16），`:core:device:connectedDebugAndroidTest` 2/2 通过：手机进程
  读取状态/距离/流端点，并真实解码连续五帧 MJPEG，设备 capture 时间严格递增、外设
  时钟域正确，且至少一帧绑定有效 ToF 距离。
- 时间：2026-08-05（Asia/Hong_Kong）；执行者：violjjet。建立 Android 手机与
  AtomS3R-M12 + ToF4M 的首条真实外设链路。无线 ADB 发现并连接 Samsung SM-S9280
  (`192.168.5.4:43505`, Android 16)，手机到设备 `192.168.5.11` 三次 ping 0 丢包、
  RTT `9.79/17.41/25.91 ms`。将原“眼镜设备模拟中心”的产品入口升级为统一的
  “眼镜外界硬件连接”，新增 `GlassesConnectionRepository`：在 IO dispatcher 中
  fail-closed 校验 `/api/status` 的 AtomS3R 固件身份、读取 `/api/range`，并打开
  `:81/stream` 验证 multipart MJPEG。App 新增 INTERNET 权限，仅对白名单
  `192.168.5.11` 与 `atoms3r-tof.local` 允许 cleartext；连接状态、固件、RSSI、ToF
  距离、视频端点与错误进入 ViewModel/Compose 状态，默认相机/检测/提醒不变。
  JDK 17 下 `:core:device:testDebugUnitTest :feature:assist:testDebugUnitTest
  :app:assembleDebug` 通过；debug APK 经无线 ADB 覆盖安装。首次独立 test APK 因缺
  INTERNET 权限得到 `EPERM`，补测试 Manifest 后 SM-S9280 上
  `GlassesConnectionRepositoryDeviceTest` 通过，实证手机进程可读取设备状态、有效
  ToF 距离并打开 MJPEG 端点。当前仅建立控制/距离/流可达链路；MJPEG 解码、时间戳
  账本、latest-only 帧源和风险链路输入仍是下一里程碑，不授权产品或安全结论。
- 时间：2026-08-05（Asia/Hong_Kong）；执行者：Codex。完成 AtomS3R-M12 + ToF4M
  stream PSRAM JPEG copy-buffer reuse 单变量 R10。先从 R9 账本确认 JPEG P50/P95 约
  `34.5/35.4 KB`、copy/metadata prepare 约 `0.98/1.31 ms`，因此不采用会延长
  framebuffer 占用的零拷贝。工具链确认 `SO_SNDBUF` unimplemented、默认 TCP send
  buffer 5744 B，关闭无效扫描。正式相邻五分钟 per-frame/reuse 为
  `7,016/7,487 frames`、`23.341/24.903 fps`，均 0 reconnect/error/overwrite/gap；
  但直接 prepare P50/P95 `802/970→823/996 us` 未改善，write P50/P95/P99
  `24.849/33.244/38.786→24.807/33.651/38.515 ms` 等价。baseline actual core
  `[0,1]`、candidate `[1]`，故更高吞吐、较低 slow fraction 和较低端到端 P99 受
  调度/场景混杂，不授权晋升。终态 `STREAM_COPY_BUFFER_REUSE_NOT_PROMOTED /
  DIRECT_COST_NOT_IMPROVED / CORE_MIGRATION_CONFOUNDED`，恢复 per-frame 分配。
  baseline/candidate summary SHA-256 分别为
  `accd812aff6342dd7a105062582824d0651d70a15106e91db4be3e6e97ed70b0` 与
  `ccecd78191c19955822c24e0e9c9885005e007868a25f1e945d12202877a7913`。最终烧录
  `atoms3r_m12_tof4m_stream_r11_per_frame_copy_buffer`，program/RAM
  `1,078,575/62,608 bytes`，application binary SHA-256
  `5dd4afc81d880674a2e6dd0fe560f42644a85992766b1ed4088335220eb0c732`；20 帧 smoke
  0 reconnect/error/overwrite/gap，actual core/priority `[1]/[5]`，退出后
  `stream_clients=0`、自动曝光开启、ToF sampling/valid、Wi-Fi 0 重连。本结果仅为
  Development 传输实现证据，不授权画质、准确率、人体、产品或安全结论。
- 时间：2026-08-05（Asia/Hong_Kong）；执行者：Codex。完成 AtomS3R-M12 +
  ToF4M stream HTTPD task priority 单变量 R9。冻结 XGA/quality 10/自动曝光/
  DMA-off/ToF-on/TCP_NODELAY-on/preamble split/no-affinity/host 4 threads，只比较默认
  priority 5 与 6。新增状态 configured priority 和逐帧实际 handler priority；正式
  五分钟两臂实际 core 均为 1、priority 分别全为 5/6。两臂 `7,195/7,202 frames`、
  `23.940/23.957 fps`，均 0 reconnect/error/overwrite/gap。priority 6 将 response
  write P50/P95/P99 `27.930/35.089/38.916→28.627/35.835/40.147 ms`，
  JPEG-ready→host read start `3.445/6.883→3.737/7.274 ms`，capture→feedback
  P50/P95/P99 `83.717/121.438/129.448→84.800/122.407/132.683 ms`；ToF age/skew
  基本相同。两臂 camera capture P50/P95 均约 `36.6/72.6 ms`，RSSI P50
  `-35/-36 dBm`，没有支持候选的混杂优势。终态
  `STREAM_PRIORITY6_NOT_PROMOTED / NO_THROUGHPUT_GAIN / SMALL_LATENCY_REGRESSION`，
  正式恢复 priority 5，并停止更高 priority 扫描。priority-5 summary SHA-256
  `bac05123dc7e5fcba5da5bf55293f1f4e112eba70e71ce76cc0fa770ab521203`，
  priority-6 summary SHA-256 `ab4a1dcfd0fc739ac7d2d6ba384688ba4af7921fa480acf0ad76f1766febf3a2`。
  最终恢复并烧录 `atoms3r_m12_tof4m_stream_r10_priority5`；program/RAM
  `1,078,467/62,608 bytes`（`32%/19%`），固件 SHA-256
  `9230ba67004c793fe1711cfd52582856e1092b0b9ca82ddf0990dbd4bf8b3c54`。
  20 帧 release smoke 为 0 reconnect/error/overwrite/gap，实际 handler core/priority
  为 `[1]/[5]`；退出后状态 API 确认 `stream_clients=0`、自动曝光开启、ToF
  sampling/valid、Wi-Fi reconnect attempts 为 0。
  本结果仅为 Development 调度证据，不授权画质、准确率、人体、产品或安全结论。
- 时间：2026-08-05（Asia/Hong_Kong）；执行者：Codex。完成 AtomS3R-M12 +
  ToF4M stream server core affinity 单变量 R8。本地 sdkconfig/HTTPD contract 确认
  Wi-Fi/lwIP 固定 core 0、Arduino loop 和 timing UDP 固定 core 1、HTTPD 默认
  no-affinity/priority 5。新增 configured core/priority 状态与逐帧实际 handler core；
  no-affinity canary/五分钟实际全部为 core 1，因此不做无效的 core-1 候选，只比较
  no-affinity 与固定 core 0。两臂 `7,040/7,397 frames`、`23.419/24.603 fps`，均
  0 reconnect/error/overwrite/gap。core 0 将 response write P50/P95/P99 从
  `26.287/35.132/42.301` 恶化到 `31.439/39.009/43.308 ms`，JPEG-ready→host
  read start P50/P95 从 `3.561/7.587` 恶化到 `6.770/10.431 ms`，capture→feedback
  P50/P95 从 `82.549/121.248` 变为 `86.798/122.050 ms`。候选 P99/max 较好，但
  同时 camera capture P99/max 从 `108.413/144.687` 降至 `72.736/108.748 ms`，
  RSSI P50 也为 `-32/-35 dBm`，不能归因于 affinity。终态
  `STREAM_CORE0_AFFINITY_NOT_PROMOTED / NETWORK_START_AND_WRITE_MEDIAN_REGRESSED`，
  正式恢复 no-affinity（当前实际 core 1），priority 不扫描。no-affinity summary
  SHA-256 `667b90f98136b91d5022ce51c4b222bc139945b80d2e6ba87ca8d6f210f349b1`，
  core-0 summary SHA-256 `e5daa4f707d282aa5df4a01b920a40a5c7dc675fefec4e40e5604457aaa5deec`。
  本结果仅为 Development 调度证据，不授权画质、准确率、人体、产品或安全结论。
  正式 no-affinity 固件 program/RAM `1,078,375 B (32%) / 62,608 B (19%)`，app bin
  SHA-256 `928af057147198a233cb10db1c5e464307321e705f4c2b6647b1f056ea847bc8`；刷入
  COM5 后 20 帧验收实际 handler core 全为 1，0 reconnect/error/overwrite/gap，
  TCP_NODELAY=true、preamble split、自动曝光、ToF sampling/valid，退出后
  stream_clients=0。12 项测试、Ruff、py_compile、固件编译与 scoped diff check 通过。
- 时间：2026-08-05（Asia/Hong_Kong）；执行者：Codex。完成 AtomS3R-M12 +
  ToF4M stream preamble coalescing 单变量 R7。冻结 XGA/quality 10/自动曝光/
  DMA-off/ToF-on/TCP_NODELAY-on/host 4 threads，只比较 boundary + metadata header
  分两次或合为一次 HTTP chunk。split/coalesced 五分钟分别为 `7,223/7,358 frames`、
  `24.025/24.481 fps`，均 0 reconnect/error/overwrite/gap。合并后 response write
  P50/P95/P99 `25.531/35.616/39.244→24.193/33.504/38.988 ms`，正常收益仅约
  `0.3–2.1 ms`；但候选 frame 3879 发生 `1563.568 ms` device write、
  `1565.344 ms` host read 和 `1620.721 ms` capture→feedback，split 对应最大仅
  `96.897/95.620/187.649 ms`。异常帧相机采集 `36.604 ms`、JPEG `31,938 B`、
  RSSI `-32 dBm`、heap `149,048 B`，定位为设备写出/网络接收冻结，不是相机或主机
  queue。终态 `STREAM_PREAMBLE_COALESCE_NOT_PROMOTED / TYPICAL_GAIN_TOO_SMALL /
  EXTREME_WRITE_STALL_OBSERVED`；不声称 coalescing 必然导致尖峰，但不以小幅 P95
  收益掩盖 1.6 秒冻结，正式恢复 split。split/coalesced summary SHA-256 分别为
  `a4dd9d2fde42e3a568ecc57bfb60279571a871d8ddac36e31ff8b84b72bb289a` /
  `782e9e74c8a78e93a99db9bebd510f990eae391a997b944f8d4a834b597304b4`。
  本结果仅为当前 Development 传输配置证据，不授权准确率、人体、产品或安全结论。
  正式 split 固件 program/RAM 为 `1,078,267 B (32%) / 62,608 B (19%)`，app bin
  SHA-256 `a9f265e6db715b106438b6dfffb1e05d8680f7514cd3a5bcf4195de1d1a68a73`；刷入
  COM5 后 20 帧带模型验收 0 reconnect/error/overwrite/gap，全部帧
  TCP_NODELAY=true、preamble_coalesced=false、ToF sampling/valid、pipeline
  threads=4，退出后 stream_clients=0。12 项测试、Ruff、py_compile、固件编译和
  本任务 scoped diff check 通过；全仓 diff check 中另有并发 dataset ledger CSV
  尾空格，未修改且未纳入本提交。
- 时间：2026-08-05（Asia/Hong_Kong）；执行者：Codex。完成 AtomS3R-M12 +
  ToF4M MJPEG stream `TCP_NODELAY` 单变量 R6。设备保持 XGA/quality 10/自动曝光/
  DMA-off/ToF-on，主机保持 4-thread latest-frame pipeline。相邻五分钟 off/on 为
  `6,927/6,938 frames`、`23.043/23.085 fps`，均 0 reconnect/error/overwrite/gap。
  开启后 device response write P50/P95/P99 从 `22.13/30.83/35.80 ms` 变为
  `26.46/34.63/38.82 ms`，接受约 4 ms 常态成本；host JPEG read P95/P99 从
  `62.48/72.02 ms` 降至 `32.86/37.23 ms`，capture→feedback P95/P99 从
  `128.94/164.42 ms` 降至 `121.37/151.24 ms`，off 基线的约 1.1 秒 write/read
  尖峰在 on 运行中未复现，on 最大 capture→feedback 为 `197.79 ms`。因此终态为
  `TCP_NODELAY_PROMOTED_FOR_TAIL_LATENCY / MEDIAN_WRITE_COST_ACCEPTED /
  EXTREME_STALL_NOT_PROVEN_ELIMINATED`。固件对具体 stream socket 执行 set/readback，
  API、逐帧 header 和 host summary 绑定实际配置，失败时 fail closed。on summary
  SHA-256 `ada9f563f5a45136f48e0c4782c6d7f0bc2ded358bfd81f7cca9270779d4f540`。
  slow fraction `19.72%→22.50%`，故不声称相机变快；一次相邻 A/B 也不证明永久消除
  极端网络尖峰。本结果仅授权当前设备/网络的 Development 尾延迟配置选择，不授权
  准确率、人体、产品或安全结论。正式固件 program/RAM 为
  `1,078,167 B (32%) / 62,608 B (19%)`，app bin SHA-256
  `84f059606efa0cb0560a8f7fe7110c38d8df30b22ae7a66a188ee3a608cd1d3f`；刷入 COM5
  后 20 帧带模型回归 0 reconnect/error/overwrite/gap，全部帧 TCP_NODELAY=true、
  ToF sampling/valid、pipeline threads=4，退出后 stream_clients=0。12 项测试、
  Ruff、py_compile、固件编译及 diff check 通过；一次从仓库根目录直接加载测试文件
  因模块导入路径错误未运行，随后在模块目录用 discovery 正确执行并通过。
- 时间：2026-08-05（Asia/Hong_Kong）；执行者：Codex。完成主机 TFLite 4 线程
  优化 R5。当前 18 logical CPU 主机同帧微基准显示 1/2/4/8 threads P50
  `29.84/16.95/11.89/17.73 ms`，预先选择 4，不继续线程扫描。设备保持自动曝光、
  XGA/quality 10、DMA-off、ToF-on，五分钟 `300.609 s / 6,927 frames /
  23.04 fps`，run accepted，0 reconnect/error/overwrite/sequence gap。相对自动曝光
  legacy 单线程 R1，inference P50/P95/P99 `32.47/42.52/48.13→
  12.92/15.00/16.43 ms`，latest queue wait `7.90/28.67/35.39→
  0.07/0.18/0.27 ms`，capture→feedback `114.48/149.38/178.37→
  82.98/128.94/164.42 ms`。R5 设备侧 slow fraction 更高且含一次 1.095 s write
  尖峰，故主机收益不是设备本轮变快造成，也不授权设备优化结论。新增显式
  `--pipeline-num-threads`、identity/逐帧/summary 线程绑定和 host CPU count；4 线程
  晋升为当前主机参考默认，保留 CLI 覆盖。summary SHA-256
  `066fcea70270657db40a227603477bcd65407144498a96ed8d6c4917ac23f6a2`。
  本结果仅为 Development host 性能证据，不授权手机、准确率、人体或安全结论。
- 时间：2026-08-05（Asia/Hong_Kong）；执行者：Codex。完成 AtomS3R-M12 +
  ToF4M 固定曝光 490 单变量 R4。通过 session-only API 关闭自动曝光，冻结
  XGA/quality 10/brightness 1/double-buffer/LATEST/PSRAM DMA-off/ToF-on。五分钟
  `300.719 s / 7,557 processed / 25.13 fps`，0 reconnect/error，run accepted；全部
  帧 auto=false、exposure=490。与自动曝光 R1 相比，用同一 `36.320 ms` 阈值复算
  slow fraction `17.71%→8.84%`，capture→framebuffer return >54 ms 双周期帧
  `15.08%→7.58%`，>90 ms `0.86%→0.25%`；capture→完整 JPEG P95/P99
  `111.19/140.05→97.81/111.68 ms`。但 capture→return P95 仍约 72.44 ms，双峰未
  消失，故自动曝光是重要影响因素而非唯一机制。summary SHA-256
  `3e93f038163cf6ad38e0523d342e9264fdb76ba1a709bae23e5265098acf6a66`。
  因未验证照度阶跃和画质，禁止仅凭性能晋升固定曝光；已恢复正式自动曝光配置，3 帧
  验收 auto=true/exposure=490、ToF valid、0 reconnect/error、stream_clients=0。
  本结果仅为单场景 Development 性能机制证据，不授权画质、模型、人体或安全结论。
- 时间：2026-08-05（Asia/Hong_Kong）；执行者：Codex。完成 AtomS3R-M12 +
  ToF4M camera PSRAM DMA 单变量 R3。先用 R1 账本后验确认：所有 slow frame 的
  capture timestamp 早于 `fb_get` 调用，正常/slow capture→framebuffer return
  P50/P95 为 `36.55/36.75 ms` 与 `72.50/83.32 ms`，主现象是单帧交付偶发跨越
  两个约 36 ms 周期。R6 加入 PSRAM DMA、framebuffer count/grab mode 身份及上述
  两个派生阶段。唯一开启 PSRAM DMA 的五分钟实验仅交付 1 帧，产生 59 reconnect/
  59 error；与此同时 60 个状态样本持续显示 Wi-Fi connected、camera ready、ToF
  valid，camera total_frames 只增至 1。路线判为
  `PSRAM_DMA_REJECTED_INCOMPATIBLE_STREAM_ROUTE`，summary SHA-256
  `3b8aa6ca9d57523bf95c60173ac89043b1eab052910d403d2a239af74edfe081`。
  修正 host fail-open：以后必须有帧且 0 reconnect/0 error 才 run accepted/成功退出。
  最终恢复并刷入 DMA-off `slow_frame_r6`；20 帧带模型验收 0 reconnect/error，ToF
  sampling/valid，stream_clients=0。最终 program/RAM
  `1,077,911 B (32%) / 62,608 B (19%)`，app bin SHA-256
  `027f2142df98706e9cdf8d63464ba3abe16f7b2af75eaf207c2d9499cfd215b6`。
  该失败只约束当前硬件/固件路线，不外推到其他设备，也不授权精度、人体或安全结论。
- 时间：2026-08-05（Asia/Hong_Kong）；执行者：Codex。完成 AtomS3R-M12 +
  ToF4M 单变量争用对照 R2。仅关闭 ToF 连续读取，冻结 XGA/quality 10/自动曝光/
  MJPEG latest-frame/host reference pipeline；R5 将 `sampling_enabled` 写入状态 API
  和每帧 header。ToF-off 五分钟为 `300.468 s / 7,125 frames / 23.71 fps`，0
  reconnect、0 error，全部帧 sampling=false 且 update count=0。冻结规则 slow
  `15.31%`，R1 ToF-on 为 `17.71%`；但 camera wait 桶从 `981/7,070=13.88%`
  变为 `1,022/7,124=14.35%`，capture→JPEG ready P50/P95 仍为
  `36.56/72.59 ms` 对 `36.58/72.60 ms`。净下降主要来自 network write 桶
  `1.84%→0.28%`，单次顺序 A/B 不授权将网络变化归因于 ToF。结论：ToF 不是
  camera wait 主因，正式固件保持开启。off summary SHA-256
  `96c1cdd50cca088dc2489938c5ad2b76cab0a49aee7c0c9e8c5ffaa6c3078dc5`。
  修正缺失 ToF timestamp 的 skew 为不可评估。最终已恢复并刷入
  `atoms3r_m12_tof4m_slow_frame_r5`，状态 sampling=true、ToF ready/valid；最终
  3 帧协议验收 0 reconnect/error，退出后 stream_clients=0。最终 program/RAM
  `1,077,703 B (32%) / 62,608 B (19%)`，app bin SHA-256
  `1114cab4d6f4352484c8a32d91d6826eb4b3f7ccab79f8c5e0d1c920b5b3c5c5`。
  本结果仅为 Development 机制证据，不授权精度、风险、人体、产品或安全结论。
- 时间：2026-08-05（Asia/Hong_Kong）；执行者：Codex。正式关闭 host 串行接收
  backlog，完成 AtomS3R-M12 + ToF4M 设备慢帧归因 R1。冻结 XGA/quality 10/
  自动曝光/ToF，不扫参数；R4 固件逐帧加入 frame-ready interval、camera mutex +
  `esp_camera_fb_get` acquire、JPEG/metadata prepare、按下一帧 sequence 回填的前帧
  HTTP write、JPEG bytes、实际 exposure、RSSI、heap 与 ToF update count。主机补齐
  first-byte/full-frame/decode/queue 及独立 overwrite JSONL；慢帧规则预先固定为
  `interval > median+3×MAD OR >2×median`。五分钟 `300.578 s / 7,071 processed /
  23.52 fps`，0 reconnect、0 error；7,070 个 interval median/MAD
  `36.047/0.091 ms`，slow `1,252=17.71%`。slow/normal acquire P50
  `48.42/13.74 ms`，preceding write P50 `22.78/21.44 ms`，JPEG 中位数
  `31,360/31,371 B`，实际 exposure 全部 `490`。诊断分层：981/1,252 slow 为
  `acquire>=30 ms && preceding write<40 ms`，130/1,252 为 preceding write
  `>=40 ms`；最大 `1,280.338 ms` interval 由 `1,278.869 ms` write 尖峰形成。
  结论为 camera framebuffer/cadence 等待主导、Wi-Fi write 次要；JPEG 大小和曝光
  变化不是本轮主因。ToF 相关性仍受等待窗口长度混杂，下一合法单变量仅为关闭 ToF。
  测试结束 `stream_clients=0`；summary SHA-256
  `e2d542665bbea7b7c808c321295675c5f72611141978475c9569e3b813782b11`。结果不授权
  图像质量、ToF 精度、风险、物理反馈、人体、产品或安全结论。最终固件
  program/RAM `1,077,639 B (32%) / 62,608 B (19%)`，app bin SHA-256
  `713973e77c79f4f4c50508da6e07bc37490121c8fc2eb32711c206e4f0d2642a`；9 项测试、
  Ruff/format/py_compile 与 diff check 通过。
- 时间：2026-08-05（Asia/Hong_Kong）；执行者：Codex。完成 AtomS3R-M12 +
  ToF4M host backlog 定位与 latest-frame R1：对 43,230 帧 R0 长测逐阶段复算，确认
  tail 主因是原脚本将 MJPEG 读取与 decode/inference 串行化，P95
  `jpeg_ready→host_read_start=179.4 ms`，而设备 JPEG/inference P95 仅
  `72.6/46.1 ms`。新增独立 reader thread、容量 1 latest 队列、queue wait 与显式
  overwrite 账本；8 项专属测试通过。按用户要求，日常回归默认改为 300 秒，30–60
  分钟仅在明确要求时做压力测试。正式五分钟 XGA/quality 10 回归为
  `300.297 s / 7,158 frames / 23.84 fps`，0 reconnect、0 error；容量 1 队列覆盖
  2 个旧帧（约 0.028%，对应 2 个 sequence gap）。capture→feedback record
  P50/P95/P99 为 `109.3/146.8/180.3 ms`，旧长测 P95 为 `265.8 ms`；接收排队
  P95 降至 `7.2 ms`。不同持续时间不冒充同长度压力比较，但阶段账本支持 backlog
  机制已被移除。一次 0-frame 五分钟片段源自外层命令终止后遗留的旧 Python 子进程，
  已明确结束该进程树并从正式结果排除；随后测试结束 `stream_clients=0` 且无遗留进程。
  正式 summary SHA-256 为
  `ac6bd0ddf72f85d7bb282cc79000036d6c16cd99da89cc1853a1be4b215ac854`。
  仍不授权真实语音/震动、风险准确率、手机、空间标定、人体、产品或安全结论。
- 时间：2026-08-05（Asia/Hong_Kong）；执行者：Codex。完成真实 AtomS3R-M12 +
  Unit ToF4M 端到端时间基线 R0：固件升级为
  `atoms3r_m12_tof4m_timing_r3`，为抓拍和 MJPEG 逐帧加入 boot/clock domain、严格
  frame sequence、相机首 DMA、JPEG ready/send start、最近时刻 ToF 及有符号 skew；
  新增独立高优先级 3333/UDP 对时 task，将最小 RTT midpoint 对时误差界从 HTTP
  canary 的约 23.5 ms 降至正式运行 P50/P95 `1.45/2.20 ms`。新增可重连主机账本、
  OpenCV 解码和 `HOST_REFERENCE_YOLO11N_RAW_SCORE_RISK_R0_NOT_PRODUCTION` 测时
  pipeline；物理语音/震动明确未发出、未评估。XGA/quality 10 完整 30 分钟运行
  `1802.422 s / 43,230 frames`，0 stream reconnect、0 error、0 frame-sequence gap、
  单一 boot；capture→JPEG complete P50/P95/P99 `99.1/225.5/259.2 ms`，
  capture→反馈记录 `137.2/265.8/300.7 ms`，绝对 ToF—capture skew P50/P95/max
  `23.3/51.5/59.7 ms`。355 次状态采样中 free heap 首尾同为 `153,288 B`、最低
  `146,364 B`；ESP32 内部温度 `67.1→71.1 °C`、最高 `72.1 °C`，RSSI P50
  `-37 dBm`。结果揭示长时 Wi-Fi/接收排队尾延迟，但不授权风险准确率、空间标定、
  手机/物理反馈、人体、产品或安全结论。逐帧证据保存在 ignored
  `artifacts.local/evidence/atoms3r-e2e/20260805T090231.009682Z/`，summary SHA-256
  `c91218b37d22d82e3e6d707677d902f61e7f16e6d16fc9d824d6d83283fac1e5`。
  最终固件编译 program/RAM 为 `1,076,407 B (32%) / 62,600 B (19%)`，app bin
  SHA-256 `fef05a3ab307f498bc14ab9c60dc8833dbdde7cd9c0b59bda5e8976aff1ceade`；
  Python Ruff/format/py_compile、7 项专属测试和 diff check 均通过。
- 时间：2026-08-05（Asia/Hong_Kong）；执行者：violjjet。将真实
  AtomS3R-M12 网页固件升级到 `atoms3r_m12_tof4m_web_r2`：新增
  `VGA/SVGA/XGA/SXGA/UXGA`、JPEG quality `6..30`、亮度 `-2..2`、自动曝光补偿
  与手动曝光的设备端白名单控制；参数切换在相机 mutex 内完成并丢弃三帧过渡缓冲，
  MJPEG 帧先复制到 PSRAM 后释放相机，从而允许实时流期间并发调参和抓拍。新增
  `/api/status` 与 `/status`，报告 uptime、heap、Wi-Fi/IP/RSSI/重连计数、相机配置、
  recent FPS、流客户端和 fail-closed ToF 状态；浏览器抓拍下载 JPEG 与
  `blindassist_atoms3r_capture_browser_r0` JSON，绑定 boot sequence、frame timestamp
  及最近 ToF 样本/age，但不宣称硬件同步或标定。设备启用自动重连及 5 秒主动 retry，
  网页为距离/状态/MJPEG 增加超时、退避、停帧检测和错误提示。真实板五档逐一应用后，
  API 声明、JPEG SOF 和抓拍 metadata 宽高均精确匹配 `640x480`、`800x600`、
  `1024x768`、`1280x1024`、`1600x1200`，非法档返回 HTTP 400，最终恢复 XGA。
  并发测试在 1 个流客户端下仍可切换手动曝光/分辨率及抓拍；最终 XGA 4 秒观察
  103 帧（25.74 fps，状态 API 25.62 fps），Wi-Fi/camera/ToF 均 ready、距离
  `78 mm`，抓拍 30,451 B 且 nearest ToF age `83 ms`。最终编译 program/RAM 为
  `1062159 B (31%) / 60088 B (18%)`，app bin SHA-256 为
  `ca0c48a253d938c53b897acc562683ae30936a1961f2fdf68c0113cc30f88e14`，COM5
  各 flash 区段写入哈希均通过。两段页面 JavaScript `node --check` 通过；应用内
  浏览器环境未能完成局域网地址导航，因此不把视觉自动化计为通过。所有结论仍限于
  Development 单区显示/采集，不授权 RGB-ToF 标定、多区深度、精度或安全结论。
- 时间：2026-08-05（Asia/Hong_Kong）；执行者：violjjet。在真实
  AtomS3R-M12 上补齐 OV3660 + Unit ToF4M 局域网实时页：设备可保存 2.4 GHz
  Wi-Fi 到本地 NVS，连接失败时回退到受密码保护的配置 AP；控制/API 使用端口 80，
  MJPEG 使用端口 81，网页每 200 ms 更新中央单区距离并显式警示其不代表整幅深度或
  安全判断。相机电源 GPIO18 在初始化前拉低并稳定 1500 ms，解决过短上电稳定时间
  导致的 OV3660 init 失败；最终档位为 XGA `1024x768`、JPEG quality 10、双 PSRAM
  framebuffer 和 latest-frame grab。真实 `m5stack:esp32:m5stack_atoms3r` 编译为
  program/RAM `1044155 B (31%) / 60016 B (18%)`，刷入 COM5 的各 flash 区段均通过
  esptool 写入哈希校验，app bin SHA-256 为
  `4b0f962450be150de994038cb3b8e8357c4fbebcba4a7c05edd94469ec39b3b6`。设备重启后在
  station 模式返回 dashboard HTTP 200；实测首帧 JPEG SOF 为 `1024x768`，3.00 秒
  观察 74 帧（约 24.66 fps），5 次距离 API 均为 `VALID`、`73–75 mm`、age
  `21–71 ms`。这些结果只证明当前设备与局域网下的 Development 实时显示和单点测距，
  不授权相机标定、RGB-ToF 时空注册、多区深度、精度、持续可靠性、提醒或安全结论。
- 时间：2026-08-05（Asia/Hong_Kong）；执行者：violjjet。新增
  AtomS3R-M12 + Unit ToF4M (`VL53L1X`) 的最小设备联调工程：按官方 HY2.0-4P
  映射固定 `GPIO2/SDA`、`GPIO1/SCL` 与 7 位 `0x29`，固件输出 boot-unique
  monotonic clock domain、严格递增 sample index、read-complete 时间戳、range status、
  timeout、signal/ambient rate 和 fail-closed `range_m` JSONL；无效测量不伪造米制值，
  驱动未提供 per-sample sigma 时不填充 `sigma_m`。补充只写
  `artifacts.local/evidence/tof4m/` 且拒绝覆盖的串口采集/哈希 receipt，以及合成
  validator 回归。该入口明确属于单区 Development capture，不填充伪 zone，不覆盖
  现有 VL53L8CX 多区合同，也不授权 RGB 同步、标定、Android、提醒或安全结论。
  Arduino CLI 1.5.1、M5Stack core 3.3.8、ESP32-S3 toolchain/SDK 与 Pololu
  VL53L1X 1.3.1 已按官方 package-index SHA-256 安装到 `E:\codex-tools`；真实
  `m5stack:esp32:m5stack_atoms3r` 编译通过，program/RAM 为
  `343966 B (10%) / 24148 B (7%)`，app bin SHA-256 为
  `aa36c53c709a26e86a144deb1f69ef870e4c101ea936828823eb0b796b7954d5`。
  Pololu 1.3.1 真实头文件下的 C++17 syntax-only 检查、5 项 Python 单测、Ruff、
  PowerShell 解析、`git diff --check` 与文档索引也通过；全仓卫生门仍被既有 root
  allowlist、历史 Module README/内部引用等结构债务阻断，本任务未改动或吸收这些
  并发范围。随后在真实 `ESP32-S3-PICO-1` 上完成烧录，
  esptool 对 bootloader、partition、boot app 与 344112 B app image 的写入哈希均
  校验通过。设备退出 `DOWNLOAD(USB/UART0)` 后从 `COM5` 完成 10 秒 Development
  capture：validator 接受 8 条 event 和 42 条 sample，`0x29` 最终探测成功且
  `sensor_init=READY`；41 条为 `VALID`、1 条为 `INVALID_RANGE`，有效距离
  `0.052–0.059 m`（均值 `0.055 m`）。capture SHA-256 为
  `37ecef808e2f749c37fd0d762c5923f2c6f438c2ff2e0f3e4e7ea8fe4e3c7629`；串口打开/重启
  期间另丢弃 12 条非 JSONL ROM/残片，并观察到初始化早期 `0x29` 暂未发现后恢复，
  因此结果仅证明该实物组合的开发级连通、初始化与单点测距，不授权持续可靠性、
  精度、相机同步、多区、Android、提醒或安全结论。
- 时间：2026-08-04（Asia/Hong_Kong）；执行者：violjjet。补齐 DA V2 canonical
  CameraX 十分钟持续部署门的实现与证据绑定：持续测试显式调用
  `preprocessFp16CanonicalStrict()`，报告固定
  `canonical_native_official_fp32_then_integer_rnte_fp16_v1` 路径标识；设备 runner
  记录 Git、APK、cached DLC、设备与 Android 身份并对 instrumentation/base gate
  失败直接报错；新增独立 R1 十分钟入口，预冻结 preprocess+QNN P95 `<=250 ms`、
  full pipeline P95 `<=350 ms`、fresh result age P95 `<=750 ms`，不覆盖旧 fast/fused
  R0 证据。20 秒真实 CameraX 冒烟通过：287/287 个 `ImageProxy` 关闭，canonical
  route 命中，preprocess+QNN/full/result-age P95 为 `98.78/192.85/230.77 ms`，
  thermal max `0`。随后以冻结提交 `5f73f54` 在同机完成 600 秒 R1：
  8,993/8,993 个 `ImageProxy` 关闭，1,143 次完整处理，最大并发 1、三槽全归还、
  thermal before/max/after `0/0/0`；canonical preprocess+QNN、full pipeline、
  result-age P95 为 `99.00/195.23/215.71 ms`，base 与 R1 门均通过。该门只授予
  支持设备上的持续部署/性能证据，不授权精度、metric geometry、安全或默认 App。
- 时间：2026-08-04（Asia/Hong_Kong）；执行者：violjjet。完成 DA V2 Android
  `CPU_BOUNDARY_MICROBENCH_R0`、`PREPROCESS_KOTLIN_TABLE_R0` 与
  `PREPROCESS_NATIVE_OPENCV_R0`。冻结官方 `640x480 RGB -> float/255 -> OpenCV
  INTER_CUBIC 686x518 -> ImageNet normalize -> NCHW`，不改 crop、旋转、插值、
  归一化、模型或几何。`SM-S9280/SM8650/Android 16` USB 真机亮屏/锁屏各 100 次：
  旧 Double resize P50 `1212.64/1214.53 ms`；Kotlin 预计算 Float 表、融合 packing
  与 direct-buffer 复用后为 `60.88/60.86 ms`；Native OpenCV 四线程+NEON FP32 为
  `1.29/5.40 ms`，P95 `1.78/8.12 ms`。Native FP32 与官方张量最大误差
  `1.74e-6`，FP16 round-trip 最大误差 `9.77e-4`；Native 两状态观察到 0 Java
  allocation/0 GC。刻意 allocation/copy 对照约 `8.55 MB/次`并触发 19/14 次 GC。
  Native 已过 `<40 ms` 门，故 GPU 前处理不触发；下一步为普通内存复制版 App-native
  cached QNN context，结果仍仅属平台工程 benchmark，不授权默认 App 或生产路由。
- 时间：2026-08-04（Asia/Hong_Kong）；执行者：violjjet。完成公开 RGB-D
  scale-free traversability 独立实现复核，并把“已消费数据可主动复用、但必须披露
  最强证据角色与独立性的具体维度”写入项目/研究准则。R1 在两段已消费 Bonn
  registered RGB-D 上固定评价 192 帧；候选执行率 100%，观察到 19/19 推荐方向与
  sensor reference 一致，但 `bonn_person_tracking` truth-score coverage 仅
  `47/97 = 48.45%`，低于预冻结 50%，终态
  `SCALE_FREE_TRAVERSABILITY_R1_NOT_EVALUABLE_SOURCE_SUPPORT`。未降门后，R2
  另冻 20 个 parent-disjoint ARKitScenes visits、3,000 帧与 confidence-2 nearest
  reconstruction；独立 validator 复算无差异，但 visit `472626` coverage 74% 且
  truth directions 19<20，visit `469455` support 17<20，仍为
  `SCALE_FREE_TRAVERSABILITY_R2_NOT_EVALUABLE_SOURCE_SUPPORT`。诊断性 visit-macro
  directional agreement 为 94.90%、opposite error 1.01%，但只有 16/20 visits
  通过 recommendation coverage，visit `484248` accuracy 38.46%，即使忽略 source
  precondition 也不能支持候选。两轮均保留 Development/validator/counterexample
  价值；按用户“能用就用”的要求，明确 practical-use decision 为继续用于开发诊断、
  回归和下一候选，而不是因 formal gate 失败丢弃数据；仍不授权 App 集成、米制距离、
  提醒、安全或生产。
- 时间：2026-08-03（Asia/Hong_Kong）；执行者：Codex。完成 HFTF D45
  `SM-S9280 / Android 16` 物理 source canary 与外接相机运行时边界修正。R4 在
  `OPERATOR_CONTROLLED_TRANSLATION_TEXTURED_SCENE` 下执行 900 updates，得到 844
  tracking frames、864 distinct camera timestamps、0 exact-timestamp raw-depth
  observation，844 次 acquisition 全为 `DEPTH_TIMESTAMP_MISMATCH`；ARCore 报告
  raw-depth supported，但 hardware-depth camera config 为 0。终态为
  `D45_PHONE_METRIC_DEPTH_SOURCE_NOT_EVALUABLE`，未执行 1/2/3/5 m 人体测距，不把
  source 不可评估写成算法负结果。撤销 target-context 非法的重复 capability canary，
  capability/source/registration 合并为一个合法 benchmark receipt。鉴于最终普通
  外接摄像头不能假设 ARCore/depth，D45 同时降级为 teacher/diagnostic bridge；HFTF
  在线核心只要求 causal RGB、单调时间戳与冻结 camera profile，depth/pose/future
  保持 teacher-only。研究主线与默认 App 不变。
- 时间：2026-08-03（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D27
  THOR-MAGNI kinematic information-ceiling oracle。prediction 两臂均不读未来：
  current-static 冻结其他人体当前位置，history-kinematic 只用 anchor 前 0.4s
  世界位置估计恒速；truth 仍为 D26 的真实未来轨迹。2,927 个 current-body
  observations 中 2,787 个有历史速度，coverage `95.22%`。history 相对 static
  的 source-macro direction×horizon AUROC/AP 为 `+.10833/+.17781`，
  safest-direction accuracy `+.13955`，pooled AUROC/AP
  `+.09163/+.24982`；五折全部为正。left/center/right AUROC 分别
  `+.11345/+.08086/+.13069`，AP `+.16539/+.17820/+.18985`，三方向均
  5/5 folds 正。冻结 gate 11/11 通过，终态
  `D27_THOR_MAGNI_HISTORY_KINEMATIC_INFORMATION_CEILING_SUPPORTED`。
  这建立强 source-native history-motion information ceiling，定位 D26 瓶颈为
  whole-frame RGB dense-flow 没有恢复 object-centric motion；不撤销 D26 RGB
  总体负结果，也不升级为系统效用。下一学生直接蒸馏 current-static 与
  history-kinematic distance fields，不再让 full-truth loss 自行发现运动；主线、
  默认 App 与安全权限不变。
- 时间：2026-08-03（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D26
  THOR-MAGNI counterfactual collision field canary。对 530 个 current-clear
  anchors 生成 `-30°/0°/+30°` 恒速候选路径，与其他人体的真实未来世界轨迹计算
  三方向×五类首次 1.25m 冲突时间；287 个样本精确时间随方向变化，231 个在
  2 秒 collision/no-collision 上有方向分歧。相同 1,057,651 参数 current/history
  五折 seed17 完整训练。history-minus-current 的 source-macro
  direction×horizon AUROC/AP 为 `-.00051/+.00434`，2/5 与 3/5 folds 正；
  safest-direction accuracy 为 `+.00541`、3/5 正；冻结 gate 7/11 通过，终态
  `D26_THOR_MAGNI_COUNTERFACTUAL_COLLISION_FIELD_INCREMENT_NOT_SUPPORTED`。
  同时 right direction 的 horizon-macro AUROC/AP 为 `+.00802/+.01289`，均
  4/5 folds 正，保留
  `D26_RIGHT_CANDIDATE_COLLISION_FIELD_SIGNAL_SUPPORTED_DEVELOPMENT_ONLY`，但
  不切 primary direction、不扩 seed。总体负结果不覆盖该表示层信号，局部信号也不
  覆盖 center 与 choice effect 失败。下一步只做 current-static vs
  history-kinematic source-native information-ceiling oracle，先定位 target 还是 RGB
  motion representation 瓶颈；主线、默认 App 与安全权限不变。
- 时间：2026-08-03（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D25
  THOR-MAGNI ordinal time-to-entry canary。把 530 个 current-negative
  proximity anchors 按首次 1.25m 进入时间固定为五类
  `61/32/35/29/373`，四个累计 horizon positives 为 `61/93/128/157`，每折均
  有正负。相同 D22 encoder、五折、seed17、30 epochs 下独立训练等容量
  current/history 共 10 runs。history-minus-current 的 source-macro
  horizon-macro AUROC/AP 为 `-.04575/-.06348`，仅 2/5、1/5 folds 正；
  pooled 为 `-.03031/-.02951`，四个 horizon 的 AUROC/AP mean 均不为正。
  0.5/1.0s Brier 虽改善 `-.00710/-.00728`，不足以覆盖 ranking 负结果；终态
  `D25_THOR_MAGNI_TIME_TO_ENTRY_INCREMENT_NOT_SUPPORTED`。首次执行在 fold1
  held-out metric 前因 current 模型跨 arm 留在 GPU 触发 OOM；commit `9b65e37`
  改为逐 arm CPU checkpoint 后释放，从 fold0 完整重跑，属于工程无效，不烧毁
  cohort。D23 binary representation 正结果保留；当前 dense-flow timing successor
  停止。下一变量改为多候选方向的 counterfactual collision field，不再调 time head、
  seed 或 loss；主线、默认 App 与安全权限不变。
- 时间：2026-08-03（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D24
  THOR-MAGNI proximity event input ablation。复用 D23 的 15 个 history
  checkpoints，不新增训练；同一权重分别读取真实五帧+dense flow 与重复当前帧+零
  flow。D12 的 530 个 proximity-eligible anchors 形成 157 positive、373
  negative 与 107 个连续 positive events；157/157 个正 anchor 均从原始 scenario
  CSV 重建首次 1.25m 进入时间。15/15 paired units 完整产生。history 相对
  zero-dynamics 的 source-macro event AUROC/AP mean 为
  `-.00641/-.00873`，10% false-active 诊断包络下 event recall 为 `-.00132`，
  仅 5/15、7/15、6/15 units 为正；lead-time credit 虽为 `+.02175s`、9/15
  units 正，但仅 1/3 seed mean 与 1/5 fold seed-mean 为正。冻结 gate 2/7
  通过，终态 `D24_THOR_MAGNI_PROXIMITY_EVENT_DYNAMICS_NOT_SUPPORTED`。这只否定
  当前 checkpoint 的稳定事件级动态依赖，不撤销 D23 独立训练 history arm 的
  representation 正结果。下一变量限于 train-only 的单标量动态残差决策桥，不再
  扩 seed、阈值或主模型；主线、默认 App 与安全权限不变。
- 时间：2026-08-02（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D16
  TartanGround true-future-onset baseline。继承既有 15-environment 三折，物化
  495 samples、19,478 eligible cells、1,652 onset cells；near/far × body/head
  四 targets 每折均有正负。相同 14,484 参数 frozen-spatial current/history
  head 下，near AUROC/AP 增量仅约 `+.0005–+.0012`、2/3 folds 正，far body/head
  多数反向。终态 `D16_TARTANGROUND_FUTURE_ONSET_HISTORY_INCREMENT_NOT_SUPPORTED`
  与 `FROZEN_SINGLE_FRAME_FEATURE_PLUS_POSTHOC_TEMPORAL_RESIDUAL_FAMILY_STOP`。
  数据机会充足但表示仍失败；下一候选必须前移到五帧共同时空预训练，不再调
  residual head/seed/threshold，主线与默认 App 不变。
- 时间：2026-08-02（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D15 JRDB
  true-future-onset 独立复现。用 anchor-frame source-native 3D person geometry
  排除 current-risk，得到 proximity 14 positive / 102 eligible、corridor 10 / 71；
  两个固定 source-pair folds 均含正负例。相同 13,586 参数 frozen-spatial
  current/history head、seeds17/23/41 下，primary corridor AUROC/AP 两折
  seed-mean 均为负，aggregate `-.00618/-.03098`、0/2 folds 正。终态
  `D15_JRDB_FUTURE_ONSET_HISTORY_REPLICATION_NOT_SUPPORTED`。这是可评价后的科学
  负结果；D13 只保留为 THOR source-local weak signal，不切换 proximity target、
  不继续当前 frozen representation search，主线与默认 App 不变。
- 时间：2026-08-02（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D14 explicit
  motion future-onset canary。固定 pretrained RAFT-small 对 D12 的 1,078×4
  adjacent pairs 全量推理，保留 direction-preserving raw/residual 3×6 grid；
  4,312 pairs 无缺失。相同 49,490 参数下比较 current+zero-motion 与
  current+RAFT。走廊 AUROC/AP mean delta `+.0219/+.0240`，但 AP median
  `-.00485`、仅 2/5 folds 正；近距 AUROC/AP 为 `+.00048/-.01025`、均仅
  2/5 folds 正。终态 `D14_EXPLICIT_MOTION_FUTURE_ONSET_INCREMENT_NOT_SUPPORTED`。
  保留 folds0/1 corridor 局部信号和 D12/D13 true-onset 资产，但不切换 target、
  不调 RAFT/grid/head；主线与默认 App 不变。
- 时间：2026-08-02（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D11–D13
  true future-onset 任务修正。D11 发现原 0–2 秒 future-ever 标签从 `t=0` 开始，
  current-static QTM geometry 五折 AUROC 已约 `.89–.97`，causal-history
  kinematic AP 未稳定改善；原任务主要测当前占用。D12 只保留当前安全样本，物化
  近距 157 positive / 530 eligible、走廊 148 / 616，五折均有正负例。
  D13 用相同 13,586 参数 frozen-spatial head 比较 current/history；四项
  seed-mean fold median delta 均为正，正折数 `4/3/5/4`，终态
  `D13_FUTURE_ONSET_TEMPORAL_SPATIAL_INCREMENT_SUPPORTED`。效应仅
  `+.0008–+.0020`，且走廊 AP mean 略负；保留为弱 representation 正信号，
  下一步测试显式 motion，不升级主线、App 或安全主张。
- 时间：2026-08-02（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D10
  THOR-MAGNI trainable-tail temporal canary。先以可恢复 `.partial.npy` + atomic
  replace 物化 1,078×5 RGB cache；工程中断只重建 cache，不烧毁 source。固定
  五折 source-session isolation、seed17、8 epochs，冻结 MobileNet blocks `0..8`、
  训练 `9..12`；current/history 两臂共享相同 765,386 个 trainable parameters。
  history-minus-current 的近距 AUROC/AP mean 为 `-.000235/+.000004`，走廊为
  `-.000403/-.000546`，四项均仅 2/5 folds 为正。终态
  `D10_TRAINABLE_TAIL_TEMPORAL_INCREMENT_NOT_SUPPORTED_STOP`。不扩 seeds23/41，
  不启动 JRDB zero-shot，不改 epoch、解冻边界、学习率或 head 救援；该科学负结果
  只关闭当前 late-tail temporal residual successor，主线与默认 App 不变。
- 时间：2026-08-02（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D9 JRDB
  independent-dataset corridor replication。四个本地 RGB360+`labels_3d`
  sequences 各 120 连续帧，物化 104 个 samples；geometry-only census 后固定两个
  完整 source-pair folds。复用 D8 相同 13,586 参数 temporal-spatial head、120
  epochs 与 seeds `17/23/41`。主检验 corridor AUROC/AP history-minus-current
  mean `-.00235/-.00152`，0/2 folds 为正，individual units 仅 1/6、0/6 为正。
  终态 `D9_JRDB_TEMPORAL_SPATIAL_CORRIDOR_REPLICATION_NOT_SUPPORTED` 与
  `HFTF_FROZEN_FEATURE_HISTORY_ROUTE_STOP`。近距负对照虽为小正但不切换 target。
  这是完整执行后的科学负结果，不是工程/协议失败；保留 D8 的 19-session 监督资产，
  停止同一 frozen-backbone/head family 搜索，主线和默认 App 不变。
- 时间：2026-08-02（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D8
  equal-capacity temporal-spatial actionability head。冻结 MobileNet
  `5×576×4×7` maps，current/history 两臂共享 13,586 参数、相同五折三 seed
  训练预算。近距 AUROC/AP delta mean `-.0016/-.0006`，仅 2/5、1/5 fold 为正；
  走廊 AUROC/AP delta mean `+.0040/+.0038`，均 5/5 fold 为正，但 individual
  units 仅 13/15、9/15。记录
  `D8_TEMPORAL_SPATIAL_CORRIDOR_SIGNAL_WEAK_NOT_ACTIONABLE` 与
  `D8_EQUAL_CAPACITY_TEMPORAL_SPATIAL_ACTIONABILITY_INCREMENT_NOT_STABLE`。
  空间 layout 是机制一致变量，但效应小且未通过预定双目标门；停止当前 THOR
  frozen-backbone 搜索，不删 seed、不调模型救援，下一科学变量转向独立来源复现。
- 时间：2026-08-02（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D8
  equal-capacity temporal actionability head。两臂共享相同 `5×576` 接口、4,610
  参数、120 epochs、source-balanced BCE 与 seeds `17/23/41`；current arm 仅将
  current feature 重复五次，history arm 才读取真实五帧。seed-mean fold delta：
  近距 AUROC/AP mean `-.0039/-.0080`、各 2/5 折为正；走廊 AUROC mean
  `+.0071`、3/5 折为正，但 AP mean/median `+.0013/-.0009`、仅 2/5 折为正。
  终态 `D8_EQUAL_CAPACITY_TEMPORAL_ACTIONABILITY_INCREMENT_NOT_STABLE`。先前
  高维 screen 的 5/5 coarse AUROC 正信号保留为 separability observation，但因
  容量混杂不能升级为 history 独立增量。停止当前 pooled frozen-feature head，
  不调 epoch、seed、head 或 target 救援；主线和默认 App 不变。
- 时间：2026-08-02（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D8
  THOR-MAGNI local route supervision 与首个 RGB-history screen。19 个 Pupil/QTM
  sessions 物化 1,078 个 source-session-isolated 样本；近距正例 705、走廊侵入
  正例 610，五折均含正负例。冻结 pretrained MobileNetV3-small 后，history 相对
  current-only 的近距 AUROC delta mean/median 为 `+.0559/+.0358`、5/5 折为正，
  走廊侵入为 `+.0511/+.0473`、5/5 折为正；对应 AP 均 4/5 折为正。完整 48-cell
  occupancy AUROC/AP delta mean 为 `-.0103/-.0074`，AP 0/5 折为正；最小距离
  Spearman 仅 2/5 折为正。分别记录
  `D8_COARSE_ACTIONABILITY_HISTORY_INCREMENT_SUPPORTED_DEVELOPMENT_ONLY` 与
  `D8_FULL_LOCAL_FIELD_HISTORY_INCREMENT_NOT_SUPPORTED_ON_FROZEN_REPRESENTATION`。
  下一步只做等容量 compact temporal actionability head，不微调 backbone、不搜索
  field 表示；主线和默认 App 不变。
- 时间：2026-08-02（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D6 pretrained
  RAFT-small motion representation evaluation。权重固定为 torchvision
  `raft_small_C_T_V2-01064c6d.pth`，SHA-256
  `01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27`。比较 raw
  pixel、raw dense flow、去 dominant global motion 的 residual flow；三臂共享相同
  3×6 grid summary、train-only L2 projection 与五折。初版 partial-affine extraction
  coverage `.9616`，在监督投影前终止；outcome-blind 增加 median-flow translation
  fallback 后 coverage `1.0`。residual-flow 相对 raw pixel 的 AUROC/AP delta：
  fold0 `-.0833/-.0333`、fold1 `-.2222/-.2000`、fold2 `+.3333/+.3556`、
  fold3 `-.3333/-.3611`、fold4 `-.3333/-.3333`；raw flow 为 `0/5` 双增量。
  终态 `D6_RAFT_RESIDUAL_FLOW_SEPARABILITY_NOT_STABLE`。保留 classical folds0/1
  motion-alignment 局部正信号，但停止在当前 30-session global phase cohort 上继续
  更换 flow backbone/summary。下一科学需求是 source-diverse local
  route/actionability correspondence，而不是更多模型控制面。
- 时间：2026-08-02（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D6
  motion-alignment separability audit。只将 raw adjacent-frame residual 替换为
  sparse-LK + RANSAC partial-affine aligned residual；两臂共享 54 维 `3×6`
  grid summary、train-only weighted standardization、L2 projection 和同一 5-fold
  source split。初版 affine consensus `>=.50` 在监督投影前因 held-out coverage
  `.8951` 终止为 `NOT_EVALUABLE`；只根据 correspondence diagnostics 将机械门修复为
  `.40`，未读取 outcome 或改成功门，整体 coverage 升至 `.9685`。fold0 raw/aligned
  AUROC/AP 为 `.6667/.5889` 对 `1.0/1.0`，fold1 为 `.5556/.7222` 对
  `.7778/.8056`；fold4 为 `.5000/.7000` 对 `.3333/.4500`；fold2/3 各因一个
  short phase coverage `.64/.667` 保持不评价。终态为
  `D6_MOTION_ALIGNED_PAIR_SEPARABILITY_SIGNAL_MIXED_NOT_READY_TO_TRAIN`：保留
  folds0/1 的 real-domain representation 正信号，不让 fold4 或 claim ceiling
  覆盖它；但 `2 positive / 1 negative / 2 not evaluable` 不足以让当前 classical
  alignment 进入 field training。下一变量转向更可靠的 pretrained dense flow/
  correspondence representation，不再放松当前 coverage/feature 门。
- 时间：2026-08-02（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D6
  real-phase-supervised early-pair representation canary。沿用相同 5-fold
  source-session split，固定 `seed17/model-fold0/heldout-fold0`；23 train sessions /
  1,016 scored windows，7 held-out sessions / 286 windows。candidate 仅在 frozen
  directional-single inverse-risk comparator 上新增 zero-initialized early-pair
  field residual，以 class/source-session-phase balanced human-reviewed phase labels
  训练 20 个固定 epochs，不使用 held-out 选模。loss 从 `.7772` 降至 `.1111`，
  但 held-out event-phase p95 AUROC 从 baseline `.7500` 降至 `.4167`
  （delta `-.3333`），AP 从 `.6389` 降至 `.4444`（delta `-.1944`），positive
  passed-vs-alertable direction 从 `4/4` 降至 `1/4`。终态为
  `D6_REAL_PHASE_SUPERVISED_EARLY_PAIR_CANARY_INCREMENT_NOT_SUPPORTED_STOP`；不扩展
  seed/fold，不继续调 head/loss/threshold。首次 launcher 5 秒超时发生在输出写入前，
  按完全相同配置修复重跑；工程中断不消耗科学结论。下一变量必须改变
  motion-alignment/correspondence representation。
- 时间：2026-08-02（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D6
  source-session-held-out real-domain calibration ablation。30 个 SANPO source
  sessions 按正/负 strata 内稳定分为 5 folds（`7/6/6/6/5`），同一正事件的
  alertable/passed phases 不跨 fold。固定 `StandardScaler + L2 LogisticRegression`
  比较 baseline risk/known+空间统计与再增加 candidate mean/p95/max；没有 feature、
  C、model、fold 或 threshold search。跨 3 seeds × 3 folds，candidate-aware 的 OOF
  event-phase AUROC delta mean/median 为 `+0.01704/-0.00833`，AP delta 为
  `+0.00348/-0.00354`，positive paired-direction increment 为
  `+0.01197/-0.00302`，三项都只有 `3/9` 为正。终态为
  `D6_CANDIDATE_AWARE_REAL_CALIBRATION_INCREMENT_NOT_SUPPORTED`；停止当前
  candidate-score output calibration，下一实验只改变 representation，把 real-phase
  supervision 放回 early-pair RGB interaction/structured field。工程异常仍可按原
  配置修复重跑，不视为科学负结果。
- 时间：2026-08-02（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D6
  SANPO real veto transfer。通过 candidate index、review bundle 和逐帧 SHA join，
  将四段 RGB A/B/C 3/3 model-blind `REJECT` 区间物化为 150 个唯一帧、146 个完整
  五帧负例窗口；保留 clip-level 科学标签，不把缺失 authoritative timestamp/phase
  的 final `NOT_EVALUABLE` 误写成无科学观察。冻结 zero-training-true-alert threshold
  在 24,046 个 baseline active model-cells 中仅 veto 48 个；中央方向为
  `19/11,019`，仅一个模型清除一个窗口且无多数模型复现，全方向 field 清除
  `0/1,308` model-windows。随后在已消费 30-session / 1,920-frame 人审 SANPO
  Development cohort 上比较真实排序：candidate/comparator 的 pooled cell AUROC
  mean 为 `.5096/.5197`，event-phase p95 AUROC mean 为 `.4613/.5714`，
  candidate delta 仅 `3/9`、`2/9` 为正；143 个 positive passed-vs-alertable
  model×event pairs 中仅 56 个方向正确。终态为
  `D6_CONSERVATIVE_REAL_HARD_NEGATIVE_EXECUTION_NOT_SUPPORTED /
  D6_SYNTHETIC_VETO_RANKING_REAL_TRANSFER_NOT_SUPPORTED`；synthetic ranking 正结果
  保留，不继续搜索当前 threshold/top-k/votes。14 个 veto-focused tests 通过；详见
  `docs/research/hftf/HFTF_STAGE_C_D6_SANPO_REAL_VETO_TRANSFER_2026-08-02.md`。
- 时间：2026-08-02（Asia/Hong_Kong）；执行者：Codex。建立并推进
  `HFTF_D7_PUBLIC_REAL_R1` 公开真实关系监督数据集 intake。冻结
  `dataset/session/frame/source_receipt` schema、九类事件桶、RGB A/B/C、geometry、
  counterexample、final adjudication 与 source-session-disjoint split 边界；不得在
  数据集完成前修改 YOLO、HFTF、阈值、confirmation length 或 backbone。公开
  EgoWalk trajectories 元数据完成 239 trajectories、1,032,900 frame rows、51,645
  model-blind uniform windows；与既有 Development candidate 合并后为 52,216 个
  candidate windows，超过 50,000 发现目标，但 `adjudicated_events.jsonl` 仍为 0，
  未生成任何训练或 Confirmation split。THOR Zenodo 3382145 的 22 个开放
  tracks/LiDAR 文件已下载并以 MD5/SHA-256 receipt 登记；同步视频受限，未作事件真值。
  EgoWalk 提取 RGB 已完成 239/239 MP4（45,540,962,961 bytes），SANPO 官方 GCS metadata
  inventory 仍在异步获取。role isolation
  当前为 `HOLD_ROLE_REVIEW`（2 个 ancestry groups 跨历史角色）；assignment-only
  rows、NOT_EVALUABLE terminal、source receipt hash kind 与 fail-closed validator
  已补齐。当前终态为 `NOT_COMPLETE`，不凑类别数、不把候选当标签、不改变模型。
  详见 `scripts/research/hftf_d7_public_real/README.md` 与
  `F:\ba-data\hftf-d7-public-real\reports\d7_validation_report.json` 与
  `F:\ba-data\hftf-d7-public-real\reports\d7_final_report.md`。
- 时间：2026-08-02（Asia/Hong_Kong）；执行者：Codex。候选事件挖掘继续复用
  `cem-r0-real-20260802-2hz-yolo-depth-proxy` 的既有 `candidate_report.json`，未重跑
  模型推理，将剩余 `507` 条候选按 `128/128/128/123` 四批建立排他 queue 并由隔离
  Luna 完成复核；571/571 覆盖，合并结果为 `240 keep / 331 quarantine`。新增
  `--exclude-report` 与 `merge_candidate_pools.py`，每批 queue、bundle、review 和
  pool 都保留 hash-bound lineage。另从 Wikimedia Commons 下载并登记 3 条公开源
  （Boston crowd、descending staircase、walking in sands），位于
  `F:\ba-data\blindassist-candidate-event-mining\`，新基础 batch 实际运行 2 Hz、
  YOLO11n、Depth Anything V2，得到 313 帧。随后对同一 trace 实际运行
  `nvidia/segformer-b0-finetuned-ade-512-512` ADE20K SegFormer 与现有 HFTF
  `directional-history-finetune-seed17` checkpoint，各自输出 313/313 逐帧 sidecar；
  post-inference join 增加 4,373 个真实 segmentation/HFTF 归一化信号并通过 hash、
  完整覆盖、前缀和范围校验。新 run 产生 `128` raw、`92` 去重候选、`14` clusters；
  Luna 独立复核 `81 keep / 11 reject / 0 quarantine`，候选池为 `81`，其余 `11`
  quarantine。所有结果仍是 `THESIS_DEVELOPMENT / DISCOVERY_CANDIDATE_ONLY`；
  `event_truth/training/confirmation/production/safety/default_app` 均为 false，
  因此该池不构成训练集、生产授权或安全授权。7/7 candidate-module tests、sidecar
  join 回归、hash-bound run index 与文档门禁在提交前复验。详见
  `scripts/research/candidate_event_mining/README.md`。
- 时间：2026-08-02（Asia/Hong_Kong）；执行者：Codex。按“治理服务科学”的纠偏原则，
  完成 outcome-open、可修复的 TartanGround HFTF student Development，而不再为
  下载、路径、parser 或结果文件创建 one-shot/source-burning 终点。8 个互斥环境形成
  `6 train / 2 dev`、`198 / 66` samples；592 个 RGB/depth PNG 全部可解码，样本路径
  缺失为 0，samples SHA-256 为
  `649d8ffc1e550b209ed64fcc87de20858da707089a5c31b7c00fabc14591ec75`。
  相同 1,087,464 参数下，train cell-prior、single、随机初始化 history 的 future
  body/head macro F1 为 `0.2874 / 0.5435 / 0.4996`，证明 RGB 可学习但否定 naive
  history joint training。交叉输入显示 single checkpoint 使用真实 history 可达
  `0.5509`；据此从该 single checkpoint 以较小学习率微调 history，三个微调随机种子
  得到 `0.5549 / 0.5565 / 0.5512`，相对 single 增量
  `+0.0114 / +0.0130 / +0.0077`。但增益只出现在 `MiddleEast`；
  `WaterMillNight` 三次 macro delta 均为负，终态为
  `STAGED_HISTORY_SIGNAL_OBSERVED_BUT_ENVIRONMENT_ROBUST_INCREMENT_NOT_ESTABLISHED`。
  下一步先扩 outcome-open Development 环境并诊断最差环境，不打开 held-out，不建立
  主线、Android、产品或安全主张。实现、结果与复现命令见
  `docs/research/hftf/HFTF_STAGE_C_D5_TARTANGROUND_DEVELOPMENT_PILOT_2026-08-02.md`。
- 时间：2026-08-02（Asia/Hong_Kong）；执行者：violjjet。将候选事件自动挖掘从接口骨架推进到一条真实 `THESIS_DEVELOPMENT` host 链：4 个已登记公开视频 source/session 的 byte-verified 副本位于 `F:\ba-data\blindassist-candidate-event-mining\`，实际运行 2 Hz、YOLO11n 与 Depth Anything V2 Small，产出 2,566 帧 canonical trace。全量发现为 `715` raw windows、`571` 同 session 去重候选、`15` cluster；HFTF 无 sidecar，保持 `0`，segmentation 只以 manifest 明确标注的 `image_space_risk_proxy_not_a_segmentation_model` 参与，不冒充分割模型。新增确定性 review-budget queue：覆盖 source×taxonomy/cluster 选 `64` 条，另 `507` 条保留为 `not_reviewed_and_excluded_from_candidate_pool`。Luna 在 candidate-blind、hash-bound、独立上下文中复核 64 条，`24 keep / 27 reject / 13 quarantine`；candidate pool 只收 `24` 条，`40` 条进入 quarantine，未复核分母保持 `507`。完整 source/run 索引位于 `F:\ba-data\blindassist-candidate-event-mining\project_index.json` 与 `run_index.json`；探针 Norrköping run 也保留为 `0 keep / 26 quarantine`。代码、contract tests（5/5；完整 suite 14/14）、docs index 和权限边界均复验通过；结果仅用于候选发现与后续复核，不授权事件真值、训练、Confirmation、Android、默认 App、生产或安全结论。详见 `scripts/research/candidate_event_mining/README.md`。
- 时间：2026-08-02（Asia/Hong_Kong）；执行者：violjjet。新增独立
  `candidate_event_mining` discovery Module，冻结 `CANDIDATE_EVENT_MINING_DISCOVERY_R0`
  的 canonical frame trace、12 类候选触发、同 session 去重、跨 session 聚类、candidate-blind
  Luna review bundle、fail-closed review receipt 和 discovery candidate pool。明确数据下载
  目录为 `F:\ba-data\blindassist-candidate-event-mining\`，并提供 source/session/url/time/hash
  项目索引模板。该初始提交阶段只完成标准库合成回归与接口骨架，不下载媒体、不读取
  fresh/confirmation outcome，不授权事件真值、训练、Android、默认 App、生产或安全结论；
  后续真实 run 见上方条目；详见
  `scripts/research/candidate_event_mining/README.md`。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：Codex。`RISKSEG-R0` 已按完整授权
  顺序执行到负终态
  `RISKSEG_R0_TRAINED_NOT_PROMOTABLE_KEEP_YOLO`。三个固定 PIDNet-S seed 均完成
  `512x288 / four-class / full W8A8` 训练与导出；30 parent events / 30 source
  sessions / 1,920 frames 的 output-blind 三臂评价每 seed 产生 5,760 条 trace，并由
  主机独立复算。YOLO recall/false-alert 为 `13/16、6/14`；learned 三 seed 为
  `13/16、14/16、13/16` 与 `13/14、13/14、14/14`，质量门 `0/3` 通过，决策 seed
  失败。决策 seed trained INT8 在 SM-S9280 上最终 600 秒 QNN/HTP 门通过：7,727
  样本、173/173 nodes / 1 partition、total P95 `77.374 ms`、inference P95
  `5.198 ms`、末/初比 `1.07624x`、thermal/failure 均为 0。性能 PASS 不覆盖事件
  质量否决；不改默认 App，不调已消费 event-eval，不增加规则。最终结果见
  `docs/research/dual-loop/RISKSEG_R0_FINAL_RESULT_2026-08-01.md`。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。F0.1 cross-split
  metadata plan 与 exact source lock 正式双运行 byte-exact，固定
  `6 train / 3 dev / 3 official-test heldout`，12 个 parent sessions 全互斥。
  每条 source 均有 50 个连续 aligned RGB/mask/depth source frames、intrinsics 与
  pose receipt；5 FPS 固定 `0..24`，20→10 FPS 固定 `0,2,...,48`。
  cross-split plan SHA-256
  `edaa63a86ff0254b0887d437086be9bda6f3c1b0aa3c3c9cbfc72bc05d5d0f55`，
  source-lock SHA-256
  `f7353779315757b8b4ca5ba13b3544c4348c25f2ac4daa4befe47ad80fc79f62`。
  geometry/teacher/student outcome firewall 全为 false；只授权 exact media
  acquisition，尚不授权 teacher corpus 或 student training。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。在 F0 media、geometry、
  corpus 与 student outcome 全部未打开时，用独立 metadata-only source 审计把
  same-train-split heldout design 加强为 cross-split F0.1。train/dev 仍从排除
  60 个 burned sessions 的 official train 按字典序固定 `6/3` 个 source；
  heldout 改为 official test split 文件顺序前 3 个 metadata-eligible source。
  test split generation `1692794964058506`、SHA-256
  `0f701db54d2cc26b32bf2c636537a1353beb5d7e09f8914279cde2e7c06400df`、
  401 sessions 已只读复核。F0 的 field/teacher/UNKNOWN/三臂/训练/margins 全部
  继承；split-aware importer 必须 hash-bind actual train/test，test 只准一次性
  heldout evaluation。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。在任何新 SANPO
  geometry/student outcome 前冻结 Stage C F0 body/head temporal-student canary。
  source pool 排除 R4 前 56 个 burned sessions 与 R4 四个 outcome-open sessions；
  official train 中按完整 ID 字典序固定前 12 个 metadata-eligible source，
  rank `1–6/7–9/10–12` 为 train/dev/heldout。混合 5/20 FPS 全部按物理时间取
  `[-.8,-.6,-.4,-.2,0] s` history 与 `.4 s` future；三个同参数 arm 为
  `SF_CURRENT/SF_FUTURE/HIST_FUTURE`。12/12 source 的 authority、transport、
  body/head opportunity 与 teacher byte-determinism 全过前禁止 corpus/training。
  foot-ground、完整 HFTF、主线、Android/App 与安全 claim 均保持未授权。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。固定 E0.2 关闭为
  `E0_2_FIXED_BATCH_TEACHER_MECHANICS_NOT_EVALUABLE`。1,232,000,737 bytes 与
  transport 全过；dev/heldout role opportunity 为 `35/37 risk cells`、
  `32/32 anchors`、5 directions，但 3/6 source `.4 s` known fraction
  `.3257/.6515/.5000 < .70`，其中首条 plane known `.9088 < .95`。report
  SHA-256 `a58aff72e0207871ef80d9aa6f94bc9ef7db21ba08d15e7405436b0a60558eee`。
  按 stop rule 关闭 EgoWalk foot-ground student source route，不训练、不再扩源。
  HFTF 下一候选转向 R4 已支持的 SANPO body/head obstacle temporal student；
  foot-ground 保持未评价，不能混称完整 HFTF。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。在任何新媒体前冻结一次性
  multi-source E0.2。排除前十条 consumed trajectories 及其 recording dates 后，
  按总字节升序、日期互斥固定 6 条，排序位置交替为 `3 dev / 3 heldout`，总计
  1,232,000,737 bytes。E0.1 模型/训练/阈值/margins 全部不变；每角色预要求
  `4 risk cells / 4 anchors / 2 sources / 2 directions / 300 no-risk`。固定 batch
  任一门失败即关闭该 EgoWalk foot-ground student source route，不再扩大。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。E0.1 在 student 前停止为
  `E0_1_FOOT_GROUND_STUDENT_CANARY_NOT_EVALUABLE`。新 dev/heldout transport、
  plane/speed、`.4 s` known `.9329/.8312`、known loss/UNKNOWN gates 均通过；
  dev risk `4 cells/4 anchors`，heldout 仅 `1/1`，低于冻结 `2/2`。report
  SHA-256 `44240751e577dff8ae1ad55cc4263e143cf6d2762a68f61430c5226837d22e99`。
  未生成 corpus/训练。只允许一次性固定 3 dev + 3 heldout、与全部 consumed dates
  互斥的 E0.2；若仍不够则关闭 source route，不再逐条扩张。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。在新评价媒体前冻结
  `.4 s`-only E0.1 successor。原四条 E0 train 仅作 consumed training data，原
  dev/heldout 永久排除；从排除全部八条 consumed 后的 healthy inventory 按总字节
  升序、日期互斥锁定 `2024_12_01__15_29_33` dev 与
  `2024_07_10__11_01_46` heldout。三臂相同 MobileNetV3/head，只输出
  `[current,.4 s]`；训练、阈值与 `.03` margin 均冻结。新 transport/teacher/
  opportunity 全过前不生成 corpus 或训练，任何失败不得换样。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。E0 teacher-opportunity
  正式双运行在 student 前停止为
  `E0_FRESH_TEACHER_MECHANICS_NOT_EVALUABLE`。train/dev/heldout opportunity
  全过，risk cells `27/8/36`、物理 risk anchors `22/4/19`；plane known 与
  history-speed gates 全过。唯一 blocker 是 `.8 s` candidate known fraction：
  仅 2/6 source 达到 `.70`，另四条 `.6015–.6857`；`.4 s` 为 6/6 通过。
  report SHA-256
  `770928a2e44776703f23185e2152326147e580256c25d2a76b92bdfbe3277e6b`。
  不降低 E0 门、不训练；只允许另冻 `.4 s`-only E0.1 并换全新 dev/heldout。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。获取并永久 burned E0
  六条 exact fresh media；18 个 pose/RGB/depth files 合计 956,183,459 bytes，
  size/SHA 全部匹配，acquisition SHA-256
  `8b19ff024ed6eb8d1ed0afdeeffad78025af9a3c623c6df9c598b5a8161ffdc3`。
  transport 正式双运行达到 `E0_FRESH_MEDIA_TRANSPORT_SUPPORTED`：六条
  pose/RGB/depth counts 分别为 `530/657/703/705/1251/609`，全部物理 5.0 Hz、
  PTS 严格递增并 byte-exact；report SHA-256
  `a2a0c3e739d93c79afb613727a4946fb7967c087cfdeb49c9539ecb5e66c9ac7`。
  只授权 teacher mechanics/role-opportunity audit，尚不授权 corpus/training。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。Stage C E0 source-lock
  validator 正式双运行达到 `E0_FRESH_SOURCE_LOCK_VALIDATED`。D0/D1 parent、
  C0 inventory/metadata、MobileNetV3 权重 hashes，以及 6 条 source、4/1/1
  role、unique recording dates 和 18 个 pose/RGB/depth file bindings 全部复算
  一致；payload byte-exact。report SHA-256
  `9e3ce8793597907dbe87e6a9c57d9f3f9ffcfb1510f078ea31e01148eab046dc`。
  只授权获取精确选择媒体；teacher corpus 和 student training 尚未授权。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。在任何 fresh RGB/depth
  或 geometry-label outcome 前冻结 Stage C fresh foot-ground student canary E0。
  从 95 条 healthy EgoWalk inventory 排除两条 consumed source，按总字节升序、
  recording-date 互斥锁定 6 条，固定 `4 train / 1 dev / 1 heldout`。三臂共用相同
  frozen MobileNetV3-Small encoder 与同参数 head：single-frame future、history
  current-only、history future；ImageNet 权重 SHA-256 为
  `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`。
  source/transport/teacher/role-opportunity 顺序门全过前不训练，heldout opportunity
  不足不得换样。该 E0 只检验 foot-ground temporal geometry-proxy agreement，
  body/head、完整 HFTF、主线和 App 均未授权。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。Stage C causal
  future-label mechanics D1 正式双运行达到
  `D1_CAUSAL_FUTURE_LABEL_MECHANICS_SUPPORTED`。两条 consumed EgoWalk source 的
  history-speed eligibility 均为 1.0；outdoor `.4/.8 s` candidate known fraction
  `.9266/.8766`、future-added known `186/280`，indoor `.7954/.7588`、
  `303/490`；全部 known loss 与 UNKNOWN→SAFE violation 为 0。24 个
  risk-proxy cells 覆盖 5 方向，七个 structural canaries 和第二遍 payload
  byte-determinism 全过。report SHA-256
  `e0c86898539602d6323958edc0ac01935f3fbc74375c85575db187e3948fc8c3`。
  只授权冻结 fresh session-disjoint teacher corpus + student canary protocol；
  不授权 acquisition、corpus generation、training/effect、主线或 App。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。在 D0 support 后冻结
  Stage C causal future-label mechanics D1。history `anchor-2 -> anchor` 速度只用来
  外推 `.4/.8 s` origin，orientation 固定 current yaw；future pose 只转换 future
  depth observation，禁止选择 origin/direction。consumed calibration 的 motion-yaw
  resultant 为 outdoor `.899/.840`、indoor `.969/.962`；candidate 相对
  current-only 新增 known cells `186/280` 与 `303/490`，known loss 0。formal D1
  尚未运行；通过也不直接 acquisition 或训练。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。Stage C D0 达到
  `D0_SEMANTIC_INDEPENDENT_LABEL_READINESS_SUPPORTED`。两条 consumed source 共
  265 个 formal frames，plane known `265/265`；direction known fraction
  `.9176/.7821`，known no-risk `594/524`。outdoor 7 个 risk proxies 分布于 7 帧、
  4 方向，indoor 0；UNKNOWN→SAFE 为 0。七个 structural canaries 和第二遍 payload
  byte-determinism 全过。report SHA-256
  `8a267e07e48f70abbfe9e2d184e53ca5464331fd848e256aebd9b1cb2239952b`。
  只授权冻结 causal future-label + fresh student canary；不授权 acquisition/training。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：Codex。`RISKSEG-R0`
  successor 已冻结 30 parent events / 30 source sessions 的 output-blind event-eval，
  四桶为 `8/8/7/7`；520-frame train/dev 重编码视图为 `320/200` 且 session
  零重叠。唯一候选 PIDNet-S 完成 `512x288 / four-class / full W8A8` 技术预检：
  TFLite 7,911,768 bytes，SM-S9280 上 QNN HTP `163/163` nodes / 1 partition，
  7,619 次冻结全链路 total P95 `75.739 ms`，末/初 2 分钟 P95 比
  `1.00255x`，failure 与 thermal status 均为 0。训练实现 commit
  `943fae9...` 和三 seed recipe 已写入 implementation lock；未读取 event-eval
  模型 outcome，默认 App 仍为 YOLO。下一步按 `20260801/2/3` 依次训练。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。在两条明确 consumed
  EgoWalk calibration source 上完成 depth-only reader 校准并冻结 Stage C label
  readiness D0。64/64 preview frames 可恢复 height-constrained ground plane；相机
  近场约 `<1.2 m` 不可观测，固定 UNKNOWN，可评价 sections 改为
  `1.4/1.8/2.2/2.6/3.0 m`。support-mode local normal gate 消除室内 4 个
  wall-derived 假台阶；室外保留 2 个与花坛/路缘方向一致的 foot-risk proxy。
  formal D0 尚未运行；通过也只允许冻结 fresh-source label/student canary，不授权
  acquisition、teacher dataset、student、主线或 App。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。C0.1 同一 consumed
  replay 达到 `C0_1_STAGE_C_SOURCE_TRANSPORT_FEASIBILITY_SUPPORTED`。两条 parquet
  delta 均为 `198/200/201 ms`、有效 `5.0 Hz`；pose/RGB/depth frame count 为
  `647/647/647` 与 `664/664/664`，PTS 与原 surface gates 保持通过，container
  `100/100 Hz` 仅记录。report SHA-256
  `071c8e9aa7fd36ee6682ef836f7dfed09120f2db24e5779b0c109cc55bc72024`。
  claim 只到 consumed schema repair/natural depth observability；唯一新权限是冻结
  Stage C label-and-student canary protocol，不授权执行或训练。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。C0 media audit 按冻结门
  关闭为 `C0_EGOWALK_MEDIA_TRANSPORT_NOT_EVALUABLE`。两条 source 的 file SHA、
  `647/664` pose/RGB/depth rows、完整 decode、ordinal PTS 和 depth support
  `32/32`、adjacent common support `31/31` 均通过；唯一 blocker 是 RGB/depth
  container nominal rate 均为 `100 Hz`，不满足原合同 reported `5 Hz`。
  dataset `info.json=5` 且 parquet delta 约 `200 ms`。audit SHA-256
  `3dafbef91d09f13f63826d6f004be28da9d9af1ad8a680a5df83f26ad7887057`。
  保留 C0 负终态后冻结 C0.1：同一 consumed media、不得换样，物理 timeline 改由
  parquet frame/timestamp + meta fps 定义；container nominal rate 只记录，其余门与
  权限不变。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。C0 metadata inventory
  正式复算 `239/239`，严格健康 `95`，精确锁定冻结 cohort；report SHA-256
  `5ff6a4270f2319bd8d3e30b5d10e24cdee47c0025d22c8e12a9642e5f089b82b`。
  媒体仍未下载时，把 surface observability 的“相邻 sample 共同有限深度支持”
  消歧为共同正有限像素比例 `>=.25`、至少 20 个相邻 pair；随后才允许实现/运行
  media audit。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。Stage C C0 首次正式
  inventory planner 在任何 RGB/depth 下载或报告写入前 fail closed：机器合同中的
  `trajectories.json` SHA-256 漏写末尾 `b`、仅 63 位。本地 source 与冻结前 shell
  核验均保持 64 位
  `e9a4dad8e77b60e0d6bfb9b4ae764900ed81dcf58d72d19b279f1b558807037b`。
  只修正该转录绑定；source、cohort、选择规则、数值门和权限不变。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。在任何新 EgoWalk
  RGB/depth media 内容或 geometry outcome 前冻结 Stage C source-feasibility C0。
  SANPO 固定为 causal obstacle/future teacher role；EgoWalk exact dataset revision
  `8a167f27...` 固定为 natural RGB/depth/pose transport 与 semantic-independent
  surface observability canary。239 条 pose metadata 中有 91 条含 null；严格健康门
  后有 95 条，按 pose+RGB+depth 总体积升序并要求不同录制日期，冻结
  `2024_08_15__19_45_11 / 2024_07_11__12_33_57`。两条 media 此时未下载/打开。
  C0 成功也只允许冻结 label-and-student canary protocol，不授权 student training、
  effect、主线、Android/App 或安全 claim。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。完成正式 R4 split-source
  Stage B，joint 终态
  `R4_STAGE_B_SPLIT_SOURCE_TEACHER_MECHANICS_SUPPORTED`。SANPO obstacle role
  前四个字典序 candidates 全部 reference-only qualified 并立即停止；primary
  candidate/baseline F1 `.98756/.76000`，delta `+.22756`，precision/recall delta
  `+.37792/-.00493`，4/4 session 与三高度均通过。analytic terrain role 的 20 risk、
  16 safe、6 UNKNOWN 全部正确/弃权，candidate F1 `1.0`，高于最佳 baseline `+.25`。
  joint report SHA-256
  `cc7adb2b08ceb1ef4542a0c0c86957e4bb20fc6f50f1d01e31b22f66f1177453`。
  claim ceiling 仍是 split-source Development teacher mechanics；只授权冻结 Stage C
  source-feasibility contract，不授权执行、student、主线、Android、提醒或安全 claim。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。完成但尚未运行 R4
  obstacle arm 与 joint aggregator。cohort lock 要求 qualification reports 为字典序
  contiguous prefix、精确停在第 4 个 qualified source，并绑定 authority/manifest/
  spec/pose/qualification hashes。formal runner 复用 D1 candidate/baseline/disjoint
  reference metrics，但完全不导入 ground component，保持 R3 全部 obstacle effect
  gates。joint aggregator 才可把 obstacle 与 analytic-terrain terminals 合并。
  HFTF suite 83 项通过；实现须先提交，之后才允许第一次 arm outcome。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。完成但尚未执行 R4
  split-source 工具第一段：obstacle inventory planner 验证冻结 parent/56-session
  burn ledger 与 official split，只读 inventory；obstacle qualifier 只计算 stride-4
  dense reference，显式禁止 ground、candidate、baseline 和 arm delta；analytic
  terrain runner 在采样前由 42 个 exact profiles 导出真值，执行 five-section
  candidate、semantic-safe/endpoint-delta baselines 与 UNKNOWN 防火墙。HFTF suite
  76 项通过。任何 R4 outcome 均须在本实现提交后另行执行。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。在任何 R4 outcome 前冻结
  split-source Stage B successor。obstacle role 排除 R0–R3.1 共 56 个
  outcome-open SANPO sessions，字典序最多 reference-only screen 12 个、目标 4 个，
  保持 R3 的全部 obstacle effect gates；ground role 使用 42 个解析 metric height
  profiles，覆盖 16 no-risk、20 rise/drop/localized risk 与 6 occluded UNKNOWN，
  对照 semantic-safe 和 endpoint-delta 两基线。joint success 也只允许冻结 Stage C
  source-feasibility contract，不授权执行、student、主线、Android、提醒或安全 claim。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。完成 R3.1 冻结的
  40-session reference-only opportunity screen 与 cohort 聚合，终态
  `R3_1_REFERENCE_OPPORTUNITY_COHORT_NOT_EVALUABLE`：预算 `40/40`、qualified
  `0/4`。3 个 authority 失败、3 个缺完整 geometry binding；其余 34 个 dense
  reference ground reports 合计 0 risk cells、0 个非零会话，而 29/34 通过全部
  obstacle opportunity checks。cohort report SHA-256
  `6c61d8c333cc6bad59f37e2f0c3bc34c8baabfa138958ec14a484d56510979e7`。
  新增聚合器校验 protocol/ledger/plan/report hashes、rank/session 连续映射与
  reference-only firewall；HFTF suite 65 项通过。不得扩大或降低同一 R3.1 门；
  successor 只能把 obstacle 与 ground source role 拆分，保留 ground 任务与 Stage C
  禁令。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。R3.1 ranks 5–8
  qualification 中，5–7 完整拒绝且 ground risk 均为 0；rank 8 authority 总体准入但
  缺一个 manifest frame 的 local-ground-plane，旧 qualifier 产生 KeyError 且未写
  报告。修复为在任何 reference 计算前比较 manifest、pose binding 与 ground-plane
  IDs；缺口生成显式 source rejection 与 missing-ID atlas，不消失、不默认 safe。
  ranks 5–7 已有报告保持不重跑，只允许补完 rank 8。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。完成 R3.1 bounded
  inventory plan：official split 1,560 sessions 中按字典序记录 109 个 scanned
  entries，固定前 40 个 inventory-eligible candidates；不读取 reference/candidate/
  baseline outcome，报告 SHA-256
  `de42952c99236f7d1775732055076042ea2ca4986bb667ece47bd7f92cb3a599`。
  首次命令在 120 秒 wrapper 边界返回 124，但独占报告随后完整落盘并通过 JSON、
  40/40 count 与 outcome-read=false 检查，故保留而不重跑。qualifier 现强制绑定该
  plan hash 与 inventory rank，拒绝名单外 session；新 source 尚未下载或消费。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。实现但尚未执行 R3.1
  bounded inventory planner。planner 复核 official split generation/text hash 与
  16-session burn ledger，按完整 session ID 字典序只读 description 和三模态对象清单，
  固定前 40 个 inventory-eligible sessions、target fps 与 25 个 source frame indices，
  同时保留 burned/ineligible 跳过原因。它不下载 pixels，不读取 reference/candidate/
  baseline outcome。新增 split drift 与 burn uniqueness tests；HFTF suite 61 项通过。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。实现但尚未消费新 source 的
  R3.1 single-source reference-only qualifier，并冻结 16-session burn ledger。
  runner 固定 D0 mechanics SHA，重算 authority/manifest/spec/pose 与实际消费
  depth/mask hashes，只生成 stride-4 obstacle/ground reference opportunity；不导入
  angular baseline helper，不计算 stride-8 candidate、confusion、F1 或 arm delta。
  qualification gate 覆盖每高度正负 opportunity、known coverage 与 ground risk 的
  cells/frames/directions persistence。新增 reference gate、ground persistence 和
  forbidden-helper tests；HFTF suite 59 项通过。状态为
  `R3_1_QUALIFIER_IMPLEMENTED_NO_NEW_SOURCE_CONSUMED`。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。在任何 R3.1 arm outcome
  前冻结 `HFTF_STAGE_B_REFERENCE_ONLY_OPPORTUNITY_QUALIFICATION_R3_1`。排除
  R0–R3 16 个 burned sessions 后按 official train 字典序最多筛 40 个 inventory-
  eligible sessions，目标 4 个。qualification 代码只可读 disjoint stride-4 reference，
  禁止计算 stride-8 candidate、angular baseline 或 arm delta。obstacle 要求每高度
  5 positive/20 negative、known `.10`；ground 要求 known `.10` 且至少 5 risk cells
  分布于 3 frames/2 directions。若预算不足 4 个即 NOT_EVALUABLE，不降资格门、不
  无限扫描。后续 formal R3.1 保持 R3 全部 effect gates，claim 只限 opportunity-
  qualified challenge cohort；Stage C、H2、主线、Android、提醒、默认 App、生产与
  安全未授权。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。完成 formal
  `HFTF_STAGE_B_SWEPT_ENVELOPE_REFERENCE_COMPARISON_R3`，终态
  `R3_SOURCE_OR_REFERENCE_NOT_EVALUABLE`。authority/exact set、obstacle known 与
  ground known 均过门，但 `043db91a` 在 primary 下为 0 positive / 883 negative，
  违反预冻结 4/4 reference opportunity gate；ground shared-known 651 cells 也无
  step/drop opportunity。后序 diagnostic 的 cohort F1/precision/recall delta 为
  `+.1915/+.3273/-.0038`，其余三 session F1 delta `+.1670/+.2831/+.1455`，但
  不越过前序门。报告 SHA-256
  `512a5dda7e84148820e398af39eab4d5841f4a2ac6c94871cfb6754b374cb5af`。
  四 sessions burned；只允许 outcome 前冻结 reference-only opportunity-qualified
  R3.1，保持原 effect gates 并限制 claim ceiling。Stage C、H2、主线、Android、
  提醒、默认 App、生产与安全仍未授权。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。formal R3 首次调用读取并
  完成 fresh field metrics 后，在 gate 汇总阶段因某分层 arm 无 predicted positive、
  helper 将 F1 写为 undefined 而 fail closed；未创建报告，四 sessions 已视为 consumed，
  不换样本、不改门。修复按标准 `2TP/(2TP+FP+FN)` 定义 F1：reference 有正例而无预测
  正例时为 0，双方均无正例才为 undefined；同时把每 height reference opportunity
  纳入 readiness，并在报告绑定 D1/helper 两个 dependency hashes。新增定向测试后
  HFTF suite 56 项通过；只允许用相同 consumed inputs 完成本 evidence instance。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。实现但尚未运行 formal
  `HFTF_STAGE_B_SWEPT_ENVELOPE_REFERENCE_COMPARISON_R3` runner。实现复用 D1 的
  candidate/baseline/disjoint-reference obstacle confusion，并新增 disjoint ground
  sampling、candidate/reference/shared known coverage、step/drop opportunity 与
  ground precision/recall；五个 ordered-terminal tests 保证 source→obstacle→ground
  顺序停止。HFTF 全套 55 tests 通过。即使 full terminal，也只把下一 Stage C
  protocol freeze 标记为可授权，Stage C execution 与 student 仍为 false。当前为
  `R3_IMPLEMENTATION_READY_RESULT_NOT_RUN`。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。完成 R3 四个 fresh
  source acquisition 与 frozen-canonical authority，并在任何 field outcome 前冻结
  `HFTF_STAGE_B_SWEPT_ENVELOPE_REFERENCE_COMPARISON_PROTOCOL_R3`。`043db91a/
  0460c41f/047a3307/04bfa5b7` 均为 canonical rank 1、`+Z` ground 25/25、
  standard-body proxy admitted；完整 authority/manifest/spec/pose hashes 已绑定。
  obstacle gates 保持 D1 冻结值；ground 新增 disjoint stride-4 reference，candidate/
  reference/shared known coverage 门 `.10/.10/.08`，有 step/drop opportunity 时
  precision/recall 各须 `.80`。当前为 `PROTOCOL_FROZEN_IMPLEMENTATION_NOT_READY`；
  尚未计算 R3 outcome，不授权 future Stage C、H2、主线、Android、提醒、默认 App、
  生产或安全 claim。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。完成
  `HFTF_STAGE_B_REFERENCE_METRIC_PILOT_D1`，终态
  `D1_REFERENCE_METRICS_READY_FOR_R3_GATE_FREEZE`。四个 disjoint-reference count
  thresholds 上 candidate cohort micro-F1 `.9849–.9917`，baseline
  `.8129–.8306`，delta `+.1587–+.1720`；4/4 sessions 与 foot/body/head 均稳定为正。
  报告 SHA-256
  `d4eb37137f0c2502a7f860e29d7d2148c9dafb89dea261f1e31ca12b1c31e6cf`。
  随后在任何 fresh field outcome 前冻结 R3：primary threshold=2，cohort F1/precision
  delta `>=+.10`，recall delta `>=-.02`，4/4 session F1 delta `>=+.05`，并要求四
  sensitivity thresholds 方向一致。outcome-blind inventory preflight 在排除 12 个
  burned sessions 后拒绝 19 个无 chest-camera sessions，固定
  `043db91a/0460c41f/047a3307/04bfa5b7`；当前只授权 acquisition/source authority，
  不授权 R3 outcome、future Stage C、H2、主线、Android、提醒、默认 App、生产或安全。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。完成但尚未运行
  `HFTF_STAGE_B_REFERENCE_METRIC_PILOT_D1` runner。实现对 candidate/baseline 使用
  相同 stride-8 points，对 reference 使用不相交的 stride-4 grid；所有 arm 共享
  swept-prism known mask，并按 foot/body/head/micro 与 threshold `1/2/4/8` 输出
  confusion、precision/recall/F1/accuracy 和 paired correctness。新增 lattice、
  UNKNOWN mask、confusion、paired direction 与 JSON tests；HFTF 全套 50 tests
  通过。当前仍为 `D1_IMPLEMENTATION_READY_RESULT_NOT_RUN`，不授权 fresh source
  acquisition 或 outcome、future Stage C、H2、主线、Android、提醒、默认 App、生产或
  安全 claim。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。在读取任何 fresh R3
  outcome 前冻结 Development-only
  `HFTF_STAGE_B_REFERENCE_METRIC_PILOT_D1`。candidate 为 stride-8/offset-4
  swept envelope，baseline 使用完全相同 points 的 angular bins；reference 为与
  candidate pixel lattice 不相交的 stride-4/offset-2 dense swept geometry proxy。
  四个 burned R2 sessions 将同时报告 reference count threshold `1/2/4/8` 的
  precision/recall/F1、confusion 与 paired correctness；不允许只挑最好 threshold。
  D1 仅设计 formal R3 gate，不授权 fresh outcome、future Stage C、H2、主线、Android、
  提醒、默认 App、生产或安全 claim。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。完成 Development-only
  `HFTF_STAGE_B_SWEPT_ENVELOPE_LABEL_MECHANICS_CANARY_D0`，终态
  `STAGE_B_SWEPT_ENVELOPE_LABEL_MECHANICS_ADMITTED_FOR_FRESH_R3`。7/7 structural
  canaries、4/4 burned R2 source binding、三态 UNKNOWN 防火墙和非退化门全过；
  height disagreement 共 111 cells，相对旧 angular point-support 新增 209 个
  swept-collision cells，报告 SHA-256
  `52114e9fbf500f703188de14f41f0f88e6a0cc3a081421d1011bc9192554e57f`。但真实
  sources 的 ground risk 为 0，3,600 个 foot cells 有 2,905 个 ground-UNKNOWN；
  故只授权另行冻结 fresh-source formal R3，以独立高密度 geometry reference 比较
  candidate 与 baseline，不支持 Stage B 增益、future、H2、主线、Android、提醒、
  默认 App、生产或安全 claim。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。实现但尚未执行
  `HFTF_STAGE_B_SWEPT_ENVELOPE_LABEL_MECHANICS_CANARY_D0` runner：按 6 个候选方向、
  6 个距离区间及 foot/body/head effective half-width 对障碍点作 swept-prism
  collision，使用 9 probes 裁决可观测性，并以 5-section ground continuity 检出
  `.18m` rise/`.15m` drop。新增显式 `UNKNOWN/SAFE/RISK` 三态编码，数值 risk=0
  不会在 unknown cell 上变成 SAFE；修复三维 `np.add.at` 必须直接索引原张量的
  实现错误。结构 canary 与 HFTF 全套 46 tests 通过。首次 D0 调用在报告对象完成后
  因 NumPy boolean
  JSON 序列化失败，未形成可读取终态；runner 已改为原生 boolean 且先完整序列化、再
  独占创建输出，防止编码失败留下貌似有效的部分报告。当前仍为
  `IMPLEMENTATION_READY_RESULT_NOT_RUN`；仅可消费 burned R2 sources，不授权 fresh
  R3、H2、主线、Android、提醒、默认 App、生产或安全 claim。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。重新读取 HFTF 原始构想并
  完成 objective-alignment audit，发现 R0–R2 teacher 只实现
  `theta*distance*height` angular-cell point counts，缺少原 Stage B 要求的 body-width
  dilation、swept candidate trajectory collision 与 foot ground/step/drop。故撤回
  “直接降为 single-height R3”的当前决策；R2 正式终点保持不变，但只关闭 point-support
  proxy，不能外推为 human envelope failure。outcome 前冻结 Development-only
  `HFTF_STAGE_B_SWEPT_ENVELOPE_LABEL_MECHANICS_CANARY_D0`：标准代理 effective
  half-width `foot/body/head=.30/.40/.28m`，9 prism probes，5 个 ground sections，
  rise/drop `.18/.15m`，并修正 dynamic provenance IDs 为
  `10/11/12/13/14/21`。先在 R2 burned sources 上验证标签 mechanics；fresh R3、H2、
  主线、Android、提醒、默认 App、生产与安全均未授权。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。一次性完成
  `HFTF_H1_CAUSAL_ADVECTED_ORIGIN_GEOMETRY_TEACHER_R2`，终态
  `H1_MULTI_HEIGHT_PROXY_NOT_SUPPORTED_STOP`。4/4 authority、prep hash、
  independence/exact set、usable anchors `15/15/19/15`、consistency `0` 与
  current/near/far coverage 全过；worst coverage
  `.204191/.184698/.119136`。multi-height disagreement 为
  `.072222/.020370/.002924/.031481`，`03c87279` 低于 `.02`，故在第二顺序门停止；
  future `.079012/.087654/.013645/.069136` 只作 diagnostic。报告 SHA-256
  `600f37dea7940af5a4e2d09eb798547f3a8694b2dc4d04ce611e68f186023949`。
  R2 sources burned；不改该 evidence version 的 height bands/gates。后续
  objective-alignment audit 对下一步作了补正；H2、主线、Android、提醒、默认 App、
  生产与安全均未授权。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。完成但尚未正式运行 R2
  causal-advection runner。实现新增严格过去 history selection（等距时取更高 source
  frame）、history-to-anchor ground-tangent velocity、每 horizon 独立 rolling
  origin/probes/binning、predicted-vs-observed ground-origin diagnostic，以及
  source-preparation contract hash validation；future pose 不参与 origin 或方向。
  R0/R1 无 rolling contract 时保持原行为。新增 history tolerance/tie 与 tangent
  advection 三项测试，HFTF suite 共 41 项通过。独立只读审查逐项复核 causality、
  horizon wiring、U/denominator、hash fail-closed 与 diagnostic/gate 隔离，无 blocking
  finding。状态仍为
  `PROTOCOL_AND_IMPLEMENTATION_FROZEN_RESULT_NOT_RUN`，未计算正式 R2 outcome。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。完成 R2 四个 fresh source
  authority 并冻结
  `HFTF_H1_CAUSAL_ADVECTED_ORIGIN_GEOMETRY_TEACHER_PROTOCOL_R2`，状态
  `FROZEN_RESULT_NOT_RUN`。`03694304/03b6dc99/03c87279/03d70593` 均为
  `HFTF_H0_2_SANPO_CANONICAL_PROXY_REPLICATED`、canonical rank 1、`+Z` local
  ground 25/25、standard-body proxy admitted；完整 authority/manifest/spec/pose
  hashes 与 source-preparation contract hash 已绑定。R2 仅新增
  `anchor-400ms -> anchor` causal ground-tangent velocity 和
  `origin(h)=origin0+v*h`；anchor orientation、R1 field/UNKNOWN/denominators/gates
  不变。正式 runner 尚未实现/提交，teacher outcome 未运行。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。在任何 R2 teacher outcome
  前冻结 `HFTF_H1_CAUSAL_ADVECTED_ORIGIN_SOURCE_PREPARATION_R2`。R2 使用
  `anchor-400ms` 严格历史 pose 到 anchor 的 causal velocity，经 anchor local-ground
  plane 投影后把 origin 外推到 `.4/.8s`；future pose 不参与 origin/方向选择。
  6-bin sector、distance/height/horizon、9 probes、UNKNOWN、固定 denominator 与
  `.15/.10/.10/.02/.02`、4/4 门保持 R1 不变。排除 R0/R1 八个 burned sessions 后，
  official train 中 chest-left 且 25 帧可获取的字典序前四个冻结为
  `03694304/03b6dc99/03c87279/03d70593`，target fps=`min(10,source fps)`。当前只授权
  source acquisition/authority；完整 hashes 未绑定，R2 teacher 未授权、未运行。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。一次性完成
  `HFTF_H1_FORWARD_SECTOR_GEOMETRY_TEACHER_CANARY_R1`，终态
  `H1_GEOMETRY_TEACHER_NOT_EVALUABLE`。4/4 authority、unique/exact fresh session
  set、usable anchors `18/21/21/21`、consistency `0` 通过；6-bin forward sector
  current coverage 为 `.220679/.277778/.297178/.367725`，4/4 越过 `.15`，R0 的
  current coverage blocker 在新 evidence version 未复现；cohort 同时改变，不能将
  差异单独归因于 sector。但 `00c2a1cd` near/far 仅 `.033436/0`，低于
  `.10/.10`，故在 future observation coverage 顺序门停止；height/future fractions
  只作 diagnostic。正式报告 SHA-256
  `49b8a39119983b6c84187fc97b40365b4403e12c420d73a7f31bf73a194ab939`。
  burn 后 pose localization 显示该 source 的 `.4/.8 s` translation 中位数约
  `3.60/7.14 m`，形成 ego-motion-aware temporal support 假设但不是因果确认。
  R1 四 sessions 永久 burned；不改 sector/horizon/gates/UNKNOWN。H2、主线、Android、
  提醒、默认 App、生产与安全均未授权。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。完成但尚未正式运行 HFTF
  H1 R1 forward-sector runner。实现现在从冻结 protocol 读取 theta range/edges；
  full-circle R0 保留 `[-pi,pi)` wrap，partial-sector R1 的边界外 obstacle points 不
  wrap，`+45°` 上界按协议进入最后 bin，9-probe geometry 使用相同 edges。result schema
  与 claim ceiling 由 protocol version 显式选择，R0/R1 都能 fail closed。新增
  forward-sector exclusion、probe bounds、full-circle compatibility 三项测试，HFTF
  suite 共 38 项通过。状态仍为
  `PROTOCOL_AND_IMPLEMENTATION_FROZEN_RESULT_NOT_RUN`，尚未计算正式 R1 outcome。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。在读取任何 R1 teacher
  outcome 前冻结
  `HFTF_H1_FORWARD_SECTOR_GEOMETRY_TEACHER_CANARY_PROTOCOL_R1`，状态
  `FROZEN_RESULT_NOT_RUN`。R0 的 360° 单目 observation support 已在 known coverage
  门失败且四 sessions 永久 burned；R1 不是降门救援，而是预先定义 camera-forward
  `[-45°,45°]`、6 个 15° bins 的 action-agnostic locomotion sector。其余
  distance/height/horizon、9 probes、UNKNOWN、固定 denominator 与
  `.15/.10/.10/.02/.02` 门全部保持 R0 不变。fresh sources 按排除 R0 后 official
  train session ID 字典序前四个固定为 `00c2a1cd/013e2db5/01c00b13/026d78f9`；
  4/4 source authority 已通过，完整 IDs 与 authority/manifest/spec/pose hashes 已
  绑定。尚未计算 R1 field outcome；H2、主线、Android、提醒、默认 App、生产与安全均
  未授权。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。一次性完成
  `HFTF_H1_GEOMETRY_TEACHER_CANARY_R0` 正式四 session 执行，终态
  `H1_GEOMETRY_TEACHER_NOT_EVALUABLE`。4/4 authority、exact frozen session set、
  usable anchors `18/20/18/18` 与 single/multi consistency error `0` 通过；但冻结
  360° anchor-centric 9-probe field 的 current known coverage 仅
  `.056199–.096836 < .15`，near `.005401–.061343 < .10`，far
  `.000000–.042477 < .10`，故在第一顺序门停止。height/future fractions 只作
  diagnostic，不能形成支持或否定。报告
  `h1-geometry-teacher-canary-r0-20260801/teacher_canary.json` SHA-256
  `53261fd930c9a1ffc1de03468d974a1e16624383fb12e241da8b26df0cf7809e`。
  不调低 R0 门、不删 UNKNOWN、不挑最好 session；四 sessions 永久 burned。只允许在
  新 sessions 上 outcome 前冻结不同 field-support hypothesis 的 R1。H2、主线、
  Android、提醒、默认 App、生产与安全均未授权。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。完成但尚未正式运行
  HFTF H1 R0 geometry-teacher runner 与 9 项 outcome-free unit tests，状态
  `PROTOCOL_AND_IMPLEMENTATION_FROZEN_RESULT_NOT_RUN`。runner 重算 protocol、
  authority、manifest/spec/pose 与全部消费 depth/mask bytes hash；future field 使用
  anchor origin/normal/forward/right，nominal horizon 从 source frame/fps 复算；
  class 0、behind/out-of-image/invalid depth probe 保持 UNKNOWN，固定 denominator
  不因 unknown 缩小。独立实现审查发现并在运行前修复非冻结 `.1/80m` depth cutoff、
  manifest-time horizon、frame-byte 未绑定、zero-U 除零和 atlas unknown index。
  实现与 tests 将先提交，再执行一次正式四 session H1。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。根据 independent
  outcome-blind implementation review，在 H1 正式运行前完成最后一次 denominator 与
  authority 消歧。冻结 usable anchor 集 `U=current+near+far all bound`；之后
  current/near/far coverage 分母均为 `|U|*432`，height disagreement 为
  `|U|*144`，future union 为 `|U|*432`，UNKNOWN/invalid 不能缩小分母。冻结
  anchor-centric future、`n/f/right/theta` basis、`floor(x+.5)` depth lookup、
  camera-z、semantic 0 probe 为 unknown、闭开区间规则，并绑定 4 个完整 session ID
  及 authority/manifest/spec/pose SHA-256。尚未计算 H1 outcome，既有数值门与顺序终点
  未改变。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。在读取任何 H1 outcome 前，
  补齐 H1 R0 的纯实现消歧：point cloud 固定 x/y stride 8、offset 4；排除 semantic
  IDs `0/1/3/5/6/17/27/30`，dynamic 单列 `12/13/14/15/16/21`；9 probes 固定为
  cell center 加 `theta/distance/height` 八角点；single-height risk 明确定义为
  `max(foot,body,head)` 并复核 `1e-12` consistency。状态仍为
  `FROZEN_RESULT_NOT_RUN`，门槛与终点未改变。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。outcome 前冻结
  `HFTF_H1_GEOMETRY_TEACHER_CANARY_PROTOCOL_R0`，状态 `FROZEN_RESULT_NOT_RUN`。
  parent unit 为 4 个 source sessions；field 固定 24 theta × 6 distance ×
  `0/0.4/0.8 s` × `foot/body/head`，future nominal-time tolerance 100 ms，
  per-session usable anchors `>=12`。冻结 9-probe visibility/depth known、8-point
  risk saturation、single/multi exact consistency、current/near/far known coverage
  `.15/.10/.10`、height disagreement 与 future union change 各 `>=.02` 且 4/4
  sessions 全过。顺序终点为 source/mechanics `NOT_EVALUABLE`、multi-height stop、
  future stop 或 `GEOMETRY_PROXY_MECHANISM_SUPPORTED`。尚未运行或读取 H1 field
  outcome；成功也不自动授权 H2、主线、Android、提醒、默认 App、生产或安全。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF
  source-specific H0.1 discovery 与 H0.2 三独立 SANPO-Synthetic session replication，
  终态 `HFTF_H0_2_INDEPENDENT_SESSION_REPLICATION_ADMITTED`，只授权
  `H1_GEOMETRY_TEACHER_CANARY`。verifier 固定 official SANPO commit
  `11faca999b5c223b804cd3196541a1427834918b`、`common.py` hash、GCS
  generation/MD5/CRC32C 与本地 bytes，复算 official pose-row ↔ 同编号 RGB/depth/mask
  绑定；48 假设 discovery 唯一选择
  `p_world=R_xyzw@p_opencv_camera+translation_m`。三个 outcome-blind 字典序 sessions
  的 frozen transform 均 rank 1，median relative depth error
  `.000369–.000763`。改用确定性 per-frame semantic-ground RANSAC/PCA plane，而非把
  坡地高程变化误当相机高度；三会话均导出 `+Z` vertical，camera-to-plane proxy
  median `1.229–1.307 m`。新增 source verifier/6 tests 与 cohort
  aggregator/3 tests。物理 camera-to-person 标定、精确 capture timestamp、真实人体/
  event truth、student/effect、Android、提醒、安全、主线和默认 App 均未获准；official
  `right_handed_y_up` 与当前四回放 source-derived `+Z` 的冲突只按 evidence-version
  局部处理。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。执行 `RISKSEG-R0`
  session-disjoint event-eval 数据门并以 `HOLD_EVENT_EVAL_DATA` 关闭当前尝试。
  排除 520 train/dev 与固定 90-frame regression 的 11 个 native sessions 后，本地盘点
  27 个完整 RGB/source-mask sessions；另扫描 SANPO official test 的 48 个 sessions，
  得到 44 个合格 sparse candidates / 26 sessions。14 个 boundary broad windows 完成
  精确 source-mask 门，13 个通过；新物化 9 个完整 drafts、750 RGB + 750 masks，
  全部 `manifest_validation.ok=true`。两路互不可见的 RGB-only review 最终只有 14 个
  同桶一致 shortlist，`blocking/boundary/parallel/normal=7/2/1/4`，低于
  `8/8/7/7` 与总数 30；event truth 未冻结。新增
  `scripts/research/riskseg_r0_event_eval/` review-bundle/cohort validator 与 5 项 focused
  tests。PIDNet-S 预检、训练、YOLO/PIDNet/oracle 三臂均未启动，默认 App 保持 YOLO。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。启动当前唯一算法主线
  `RISKSEG-R0`，冻结四类风险/可通行性任务、520-frame session-disjoint train/dev、
  90-frame consumed regression、新 `>=30` parent-event session-disjoint 评价集、
  PIDNet-S 单候选 `512x288 / W8A8` 技术预检、三 seed 训练与事件/设备晋级合同。
  用户已授权按数据门 -> TFLite/QNN/SM-S9280 预检 -> 训练 -> 三臂事件评价 -> 条件默认
  App 替换的完整顺序执行；前置门失败仍 fail closed。旧 canonical ID
  `1=boundary_step_curb / 2=obstacle` 必须重编码为新
  `1=blocking_obstacle / 2=boundary_level_change`，禁止 mask passthrough。冻结现有
  mask adapter、risk/temporal/event/feedback chain 与 YOLO baseline，不再以规则、gate、
  FP sampler 或 component classifier 救结果。90-frame 集与 train 有两个 source-session
  重叠，其中 `i2jg` 含 22 张相同 RGB，故仅保留为 contaminated non-gating smoke，
  不支持泛化主张。当前停在
  `EVENT_EVAL_DATA_GATE_PENDING / DEFAULT_APP_UNCHANGED`，不是等待新授权。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。建立独立
  `HFTF_CANDIDATE_LANE_R0`，状态为
  `CANDIDATE_SIDE_LANE_ACTIVE / DEVELOPMENT_STANDARD /
  INNOVATION_NOT_EVALUABLE / MAINLINE_UNCHANGED / DEFAULT_APP_UNCHANGED`。文献核验确认
  AgniNav 已覆盖身体碰撞包络、几何/深度标签、极坐标 student 与边缘部署，故不再主张
  这些组件首次出现；HFTF 只保留 action-agnostic、history-RGB、显式 short-future
  layered cells 的助盲组合假设。新增 source-feasibility audit 和 17 项 focused
  tests；在本地声明为 SANPO-Synthetic 的单 session、25-frame、2.4 s replay 上核验
  75/75 RGB/mask/depth 文件与 hash、完整 PNG decode、depth
  header/shape/finite-positive 以及全部 row 的 session/sequence 归属，终态为
  `HFTF_H0_SOURCE_FEASIBILITY_PARTIAL`。静态 metric projection canary 可执行；通用
  H0 不认证本地 source identity，也永久禁止 pose/body sidecar 自签 multi-height、
  future 或 effect eligibility；它们仍因缺 source-specific mapping/calibration
  verifier、独立 session/event ledger 而 `NOT_EVALUABLE`。截断 PNG、伪造 QA、缺失
  group、重复 observation、字符串 false QA、荒谬 body geometry、无效 pose 与自报
  effect、bool 冒充 metric count/fraction/intrinsics 均有失败关闭测试。最终报告
  `h0-source-feasibility-r0-20260801-final-v3/source_feasibility.json` SHA-256
  `43e72db3395b698a6b0ee9753e5aa6088c64e85e3cbe396b53a5a732df13d8be`，独立重跑逐
  字节一致；此前输出只保留为非权威 implementation diagnostics。冻结 source-native
  raw-capture/event ancestry 去重、positive/negative/critical 分层最小分母、逐
  guardrail missing/censoring 记分、置信界、miss lead-time 与设备预算约束的 challenger
  晋级原则；当前不训练模型，不改 Android、提醒、双环主线或默认 App。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。完成
  `INFORMATION_CEILING_THREE_ARM_D0` 的无训练设备审计，冻结同一 90-frame /
  3-parent-event SANPO consumed Development cohort、`riskConfig=current`、
  `AlertProfile.STANDARD`、100 ms 因果时钟与
  `blindassist_shared_decision_kernel_v1`。SM-S9280 instrumentation 1/1
  `BUILD SUCCESSFUL`；三臂逐帧账本各 90 行。当前 YOLO 为正事件 `0/2`、关键漏报 1、
  误提醒 0、passed 清除 `2/2`；318 个 mask-derived 真值风险框恢复 `2/2` 与漏报 0，
  但产生 53 个误提醒帧、1 个负事件误报和 `0/2` 清除；source-native mask 经当前
  adapter/source policy 为 `2/2`、漏报 0、误提醒 0、清除 `2/2`，响应较真值框晚
  2–5 帧。独立 validator 从 manifest、90 个 RGB/mask hash、B 的 source-region 框、
  C 的 mask resize/component/corridor/`take(1)` 与逐帧 truth 重算，errors 0、
  `PASS`；冻结终态 `MIXED_DETECTOR_AND_REPRESENTATION_GAPS`。当前 YOLO 冻结为
  baseline，停止为同一失败模式继续加 post-YOLO 规则；若只推进一个主学习模型候选，
  下一 Development 候选优先轻量风险/可通行性分割。该 3-event pilot 不单独证明
  bbox 几何上限或 learned segmentation 效果，不改默认 App、提醒或安全权限。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。完成
  `DG_SRF_IMAGE_SPACE_STRUCTURAL_COMPLEMENTARITY_F0` 的冻结 520-frame consumed
  Development 执行，终态 `STRUCTURAL_SIGNAL_NOT_SUPPORTED_STOP`。Depth Anything V2
  Small producer 在 Git `32650abe1c0bb974626c61adcc31a8a47fa4a793` 上完成，520/520
  q 健康，方向 canary 4/4 同向且 transform gate 通过。macro AUPRC
  `B/D1/D2/D3/D4/D5=.362109/.278070/.359603/.311101/.309456/.281121`；
  D1-D4 无一满足跨组 stable signal，D4 只在 1/10 组优于最佳单信号。10 个 LOSO inner
  context 均无九门全过 operating point；cross-fitted D4 只过 4/9 门，FP reduction
  `.556665`，但 overall/minimum-group/obstacle recall retention 仅
  `.254913/.000019/.139797`，component recall `.252938`，false components/frame
  `6.823077`。独立 validator 不导入候选算子或 evaluator，从 raw depth、truth 和 A/B
  复算 29,031 项并 `VALID`。关闭当前精确定义的 F0；不在同一 520 帧调权、改尺度/
  trend/morphology/lambda 或引入 Video Depth/时序救援，不授权 F1-F5、Android/QNN/
  A568、risk/feedback、提醒、TTS、振动或默认 App。结果只是否定该 consumed
  Development image-space 方法，不外推为相对深度对所有类别无关障碍无效。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。冻结
  `DG_SRF_IMAGE_SPACE_STRUCTURAL_COMPLEMENTARITY_F0` 协议与 host-only 实现，当前
  `RESULT_NOT_RUN`。唯一问题是在 520 帧、10 个已消费 SANPO-Real source-session 上，
  固定 Depth Anything V2 Small 的图像空间结构信号能否在实际 YOLO coverage 外，以
  低于 frozen binary raw DDRNet residual 的假激活代价，对
  `boundary_step_curb / obstacle` canonical pixels 提供稳定互补。模型 source、
  checkpoint、official preprocess、`RAW_LARGER_IS_NEARER`、逐帧尺度、q 与
  `.95/.90` coverage、`N/E/R+/R-`、二阶 lower-image surface trend、D4
  `1:1:1:1`、D5 `lambda=.25`、19 点 LOSO maximin、九门、8/10 组合优势与四终态均已
  outcome 前冻结。10/10 focused tests、520/520 truth-minimized identity preflight 和
  8-frame GPU pilot 通过；pilot 未读取 canonical truth 或 A/B，方向 canary 4/4 同向、
  median margin `.707553`，8/8 depth output 健康。正式 520-frame terminal 尚未读取。
  所有数据均为 consumed Development；缺 participant/route/parent-capture identity，
  两套 YOLO detector 与 source role 完全混杂。F1-F5、Video Depth、Android/QNN/A568、
  risk/feedback、提醒、TTS、振动和默认 App 均未授权。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。完成
  `DUAL_LOOP_SEGMENTATION_LEARNED_COMPONENT_VALIDATOR_R0` 的 10-session nested
  LOSO grouped execution、host benchmark 与独立复算，终态
  `NOT_SUPPORTED_AND_GATING_STOP`。11,757 held-out component predictions 只通过
  4/9 utility 门；FP reduction / overall / minimum-session / boundary retention 为
  `.177920 / .855661 / .466375 / .207740`，`C-A` FP-area 为 `.087407`。模型/scaler
  `1,847 B`、bounded state/buffer `1,000,023 B` 过门，但 host P95
  `9.376145 ms >= 3 ms` 失败。validator 重建 11,757-row causal table，复核 10 outer /
  90 inner folds、纯 NumPy probabilities、520-frame ledger、九门、工程门和 terminal，
  9/9 top-level checks 均通过。near-miss 因 5 项 utility 门失败、latency 失败且
  stable-high-confidence retained-false area share 仅 `.373382 < .50` 而不成立；
  不授权 component-aware loss 或其他 classifier 救援，关闭当前 reference 上的 active
  learned segmentation gating，只保留 visual sidecar / coverage diagnostic。未访问
  fresh、未启动 Confirmation、设备/Android、risk/feedback、提醒或默认 App。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。冻结
  `DUAL_LOOP_SEGMENTATION_LEARNED_COMPONENT_VALIDATOR_R0` 协议与实现，当前
  `RESULT_NOT_RUN`。输入绑定为 520 帧、11,757 raw components、10 个已消费
  Development source-session；前向角色限定为
  `CONSUMED_DEVELOPMENT_CROSSFIT_CONTEXT_ONLY`，不修改历史 R1 amendment，也不恢复
  fresh/unseen/independent/Confirmation 身份。模型唯一固定为 21 列 current/past
  因果特征的 `StandardScaler + L2 Logistic Regression`；外层/内层均按 source-session
  LOSO，outer-heldout session 不进入 scaler、weight、模型或阈值。预检重建全部组件表
  并通过 6 项 causality/leakage/threshold focused tests；entropy、future persistence、
  truth/mechanism、session/scene routing 与伪造 YOLO same-class/overlap 均不进入模型。
  九项 utility 门、host P95 `<3 ms`、64 KiB model/scaler、1 MiB bounded state/buffer
  及三态 terminal 均已在 outcome 前数值冻结；未访问 fresh、未拟合正式 fold、未改
  Android、risk/feedback、提醒或默认 App。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。完成
  `DUAL_LOOP_SEGMENTATION_FP_AWARE_DDRNET_R0` 单一 successor。冻结 commit
  `e98b3efb7d556351c6536923553f46302b3ac47e` 上完成三 seed × 1200 steps；
  第一次前台启动在 seed 20260711 step 100 后被外部 60 秒进程组清理中断，未读取
  terminal truth，部分进度原样保留且未用于评价；同 config/seed/预算的
  `training-recovery-v2` 完整完成，cross-seed selection 明确未执行。
- consumed 320 帧 same-seed 评估产生 1,920 行；validator 重新装载六个 checkpoints、
  逐像素复核 prediction masks 并通过 28,861 项检查、错误数 0。三个 seed 的 FP
  reduction 为 `.198713 / -.138991 / -.043984`，false components/frame 为
  `4.41875 / 7.81875 / 5.61875`，均未通过全部 relative 五门与 absolute 四门。
  正式终态为 `FP_WEIGHTED_SAMPLING_NOT_SUPPORTED`，只关闭这个 FP-weighted
  full-frame sampler；不选择少数 seed，不在相同 outcome 上改 crop/loss/target 救援，
  不授权 INT8、runtime、Android、risk/feedback、提醒或默认 App。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。在 conditional gating
  精确三臂静态手工门家族关闭后，冻结单一
  `DUAL_LOOP_SEGMENTATION_FP_AWARE_DDRNET_R0` successor
  `FP_WEIGHTED_UNGUIDED_FULL_FRAME`。相对历史 R1 DDRNet 只改变原 30% unguided
  full-frame branch 的 session 内 frame probability：按同 seed baseline 在 train
  truth 上的 hazard FP pixel count 加权，但仍输入完整帧；backbone、官方初始化、四类
  target、loss、70% guided crop、三 seed、1200-step 预算和 dev checkpoint rule 均不变。
  新 config、独立 trainer/evaluator/validator 与 8 项 focused tests 已就绪；outcome-
  blind preflight 通过，三个 seed 分别有 `818,645 / 1,088,041 / 2,089,096` train FP
  pixels，均覆盖 400 frames 和 8 sessions。终态只允许 consumed old-blind 120 +
  R1-consumed 200，与同 seed baseline 配对且三 seed 各自通过 relative 五门和 absolute
  四门；不得选择最好 seed。validator 将重新装载六个 same-seed checkpoints、复推 320
  帧并逐像素核对 prediction mask，任何执行或合同异常都写
  `FP_WEIGHTED_SAMPLING_NOT_EVALUABLE`。候选尚未训练，320-frame terminal outcome
  未读取；不访问 fresh/Confirmation，不运行 INT8/runtime，不修改 Android、
  risk/feedback、提醒或默认 App。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。完成
  `DUAL_LOOP_SEGMENTATION_CONDITIONAL_GATING_R0_1` post-primary shadow closeout。
  冻结的 `CLASS_CONDITIONAL_TEMPORAL` 与 `MULTI_NEGATIVE` 在 Git
  `827dcda976394cd4d2a0c6f5bc29993ada9d9d5d` 上一次性处理 520 帧、11,757 raw
  components、10 个 consumed Development sessions；两臂 FP reduction / overall
  recall retention 分别为 `0.284667 / 0.781123` 与 `0.109286 / 0.922445`，最低
  session retention 为 `0.612024 / 0.629324`。前者失败 overall、minimum-session
  与 obstacle recall，后者失败 FP、minimum-session 与 boundary recall。两臂均无
  material signal，`H_min/H_cross` 均为 false。validator recovery Git
  `dd0daacc3d847e94fae1e0000179ffbb796ce33d` 只修 primary-summary schema，未修改
  已有 evidence；独立 validator 通过 `167,327` 项检查、错误数 0，第二次复算的
  frame/component JSONL 逐字节一致。R0 primary terminal 与全部 evidence 不变；
  family terminal 为 `TWO_SHADOWS_WEAK_FIXED_HANDCRAFTED_GATING_FAMILY_STOP`，只关闭
  这三个精确定义的固定阈值静态手工门，不扩大到 learned gating、postprocess 或语义
  分割。下一主边界为 residual-aware DDRNet Development；未修改模型、Android、
  risk/feedback、提醒或默认 App。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。R0.1 V2 implementation
  Git `827dcda976394cd4d2a0c6f5bc29993ada9d9d5d` 已完成一次 520-frame、23,514
  shadow component-decision 执行。初始独立 validator 在 0 项 aggregation checks 后以
  `reported primary binding drifted` 停止：runner 摘要包含
  `reference_only/terminal_unchanged`，validator 却期待不存在的 `protocol_id`。
  validation recovery 只修 primary-summary exact schema，不修改既有 result、
  frame/component 输出、candidate、阈值、material/heterogeneity 或 authority；
  当前 `EXECUTION_COMPLETE / RESULT_NOT_YET_VALID`。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。R0.1 初始冻结 Git
  `6ef3014dbea24b24ca31fadd1c9c9eda829d2481` 的首次 activation 在读取 raw shadow
  frame/component 文件前因 input binding list 被传给 single-binding loader 而
  `TypeError` 停止；未创建 output root，未计算 shadow mask、component decision 或
  指标。前向 V2 只把两组 input list 路由到既有 multi-file bound loader，并让
  `--preflight-only` 先加载并验证 520 帧、11,757 components 的完整 membership；
  candidate、阈值、角色、material/heterogeneity、terminal 与 authority 均不变。
  35 项 synthetic/legacy tests 与旧 R0 85,235 项 validator 将在 V2 freeze 前重跑；
  当前 `IMPLEMENTATION_RECOVERY_V2_FROZEN_NOT_RUN`。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。接受用户对单 primary
  假阴性风险的纠正：R0 的 result、hash、`CONDITIONAL_GATING_NO_ROBUST_INCREMENT_
  STOP_GATING_ROUTE` machine terminal 均保持不可变，但其科学 scope 收窄为
  `PRIMARY_CANDIDATE_ONLY`，不再表述为全部 conditional gating 已失败。前向建立
  `DUAL_LOOP_SEGMENTATION_CONDITIONAL_GATING_R0_1`，mode
  `POST_R0_FORWARD_SHADOW_DIAGNOSTIC`：
  `CLASS_CONDITIONAL_TEMPORAL` 与 `MULTI_NEGATIVE` 在 R0 outcome 前曾被概念性提出，
  但当时未 repo-freeze，现以新 config/hash/runner/validator 冻结为 diagnostic-only。
  两者一次全量报告、不选优、不救援 primary；execution terminal 固定为
  `POST_TERMINAL_SHADOW_ABLATION_COMPLETE_DIAGNOSTIC_ONLY`，family counterexample、
  alternative signal 和 bounded-family negative 的解释规则在结果前写死。当前
  `RESULT_NOT_RUN`；residual-aware DDRNet 训练排在 R0.1 closeout 后，未获授权或执行。
  未访问 fresh holdout，未修改模型、Android、risk/feedback、TTS、振动、提醒或默认 App。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。在已推送的冻结 implementation
  Git `2e46d76057becb1f85c22bf0c9ea4e8b59d26c31` 上一次性执行
  `DUAL_LOOP_SEGMENTATION_CONDITIONAL_GATING_R0` 的 520 帧、11,757 components、
  10 source sessions。单一 `CLASS_CONDITIONED_MULTI_NEGATIVE` 保持 overall、
  boundary/step/curb 和 obstacle recall（`0.942399 / 0.945451 / 0.946764`），但
  false-positive reduction 只有 `0.092572 < 0.30`，最弱 session recall retention
  为 `0.774580 < 0.80`；候选不支配既有参考点且不是新的 Pareto improvement。独立
  validator 从逐帧/逐组件账本复算 85,235 项检查、错误数 0，held-out/direct session
  metrics 全部一致；写入独立目录的第二次确定性复算再次 `VALID`，result/frame/component
  三个核心输出逐字节一致。终态为
  `CONDITIONAL_GATING_NO_ROBUST_INCREMENT_STOP_GATING_ROUTE`，gating 路线停止；只授权
  另立 residual-aware DDRNet Development 设计，未执行训练。未访问 fresh holdout，
  未改变模型、Android、QNN/A568、risk/feedback、TTS、振动、提醒或默认 App，
  Confirmation、产品与安全 authority 均未激活。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。在 Atlas `GATING_PARTIAL`
  之后，于任何 conditional-gating outcome 前冻结
  `DUAL_LOOP_SEGMENTATION_CONDITIONAL_GATING_R0`。两路隔离审计发现 Atlas
  `UPPER_FIELD_BACKGROUND_ACTIVATION_PROXY` 读取 truth-derived
  `dominant_truth_class`，且固定候选的 LOSO 只是 burned Development session 重排；
  第三路独立裁决因此选择单一 `CLASS_CONDITIONED_MULTI_NEGATIVE`，将 upper 输入改为
  纯几何 any-intersection、temporal history 按 predicted class 隔离并仅来自 raw mask，
  对 obstacle 保留 pixel-level causal 语义、对 boundary/step/curb 只整组件拒绝低置信
  小碎片。新增绑定 520 帧、11,757 components、10 source sessions、输入 SHA、五项
  停止门的 config、独立 Module、truth-firewall/component/fragment/held-out 账本、
  aggregation validator 与 19 项 synthetic/unit checks；结果尚未运行，当前仅为
  `PROTOCOL_AND_IMPLEMENTATION_FROZEN / RESULT_NOT_RUN`。未训练模型、访问 fresh
  holdout、修改旧 Atlas/sidecar、Android、risk/feedback、TTS、振动、提醒或默认 App。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。在 Atlas
  `GATING_PARTIAL` 主线暂停点启动 host-only visual sidecar R0。新增绑定 DDRNet INT8
  SHA 的 Development renderer：逐帧展示 YOLO known-object boxes、重新推理的 raw
  segmentation heatmap、rehearsal visual candidates、指定原有 probe 的 gate-passed、
  rejected/abstained pixels 与原因，固定
  `DEVELOPMENT VISUALIZATION ONLY / DOES NOT DRIVE ALERTS` 水印。输出 manifest 固定
  `VISUAL_CANDIDATE_ONLY / drives_alerts=false`，显式禁止 confirmed-danger、
  safe-route 与 verified-obstacle 文案。以 causal 2-of-3 的 success/failure 两帧完成
  smoke render，3 项 sidecar 单元测试通过并人工检查布局；未新增/选择 gate，未接
  Android、risk/feedback、TTS、振动、默认 App 或任何提醒路径。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。完成 Atlas 固定
  320-frame 定向扩展：以同一 DDRNet INT8、未过滤 postprocess、canonical evaluator
  和冻结 YOLO trace 分别重放 4 个 dev session（200 帧）与 2 个 consumed old blind
  session（120 帧），两组 rehearsal 独立全量复算均为 `VALID`。扩展共分析 6,714 个
  components；五类 pilot 机制均跨两角色复现，aggregate 排序 Spearman `0.90`，三态
  residual 仍为 `WEAKLY_LABELABLE`。原样运行的 causal 2-of-3 与 median confidence
  `>=0.65` 分别以 `0.7930 / 0.8528` overall recall retention 达到既有 `PARTIAL`，
  但最低 session retention 只有 `0.4729 / 0.4087`，没有 `SUFFICIENT` gate；按冻结
  决策树终态为 `GATING_PARTIAL`，因此未训练 residual-aware DDRNet、未选择或组合 gate。
  runner 新增唯一帧/完整 membership 多输入合同、五机制覆盖/排序复算、session FP
  汇总与固定成功/失败案例图；9 项单元测试通过，完整 Atlas 与 10 张案例图确定性复跑
  逐文件一致。未访问 fresh
  holdout，未修改 Android、risk/feedback、TTS、振动或默认 App；Confirmation、产品与
  安全 authority 均未激活。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。落实新科学主线
  `DUAL_LOOP_SEGMENTATION_FAILURE_ATLAS_AND_RESIDUAL_LABELABILITY_R0`：新增冻结 pilot
  配置、可复算 Atlas runner 与 5 项单元契约，只读消费已降级的 200-frame R1 rehearsal、
  5,043 个组件、canonical pixel truth 和冻结 YOLO trace。pilot 识别 3,062 个同类
  residual false activation component；错误由上部视场背景 proxy、YOLO 归因歧义、
  temporal flicker、稳定高置信错误与小碎片等非互斥机制共同构成。4 个空间、3 个因果
  时序和 2 个置信 probe 均未达到预声明 recall-retention 门，gating 终态为
  `INSUFFICIENT`；pixel residual 可复算，但缺少 instance correspondence、depth 和 pose，
  三态 attribution 仅为 `WEAKLY_LABELABLE`。五种机制满足跨 4 session 的定向扩展规则，
  只列出 6 个 dev/consumed candidate session，未执行扩展推理。未训练模型、访问 fresh
  holdout、实现可视化平台、运行 Android/QNN/A568 或修改 risk/feedback/TTS/振动/默认
  App。验证：5 项 module unit tests、完整 pilot identity/geometry/truth pairing 与后续
  仓库检查通过；Confirmation、产品与安全 authority 均未激活。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。完成全项目 current 权威面的
  R4 收口：SANPO 拆分为默认论文 `THESIS_DEVELOPMENT` 与显式
  `PRODUCTION_PROMOTION`，普通训练/utility/算法选模 benchmark 不再要求 fresh
  holdout、INT8、设备事件或发布门；生产默认模型晋级继续保留完整链条。双环、暂停
  RCLE 和历史 USTRF current/index 均加入前向 R4 marker：新 Development 不继承旧
  one-shot/formal 门，历史终态与数据角色保持不可变。同步修正根 README、docs/scripts
  索引中的旧主线口径，并在 governance unit tests 中检查六个 current/操作入口及三个
  导航入口的 lane markers；SANPO sequence、segmentation candidate、traversability
  baseline、v3 regression、counterfactual collection 和 public-video silver 六份
  workflow 也明确为“选择后生效”，防止产品门再次倒灌普通论文实验。未修改算法、模型、
  数据、历史协议、receipt、App 或提醒链路。验证：35 项 governance unit tests、
  13-file research contract suite、历史 R3 contract CLI、JSON/py_compile、repo
  hygiene、docs index 和 diff whitespace 全部通过；12 份 legacy-bound machine
  contract 数量不变，R3 policy 与 R1 result、closeout validator、failure receipt、
  formal freeze 的既有 SHA-256 保持一致。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。进一步落实 R4 的轻量实验
  流程：Discovery 默认不分配或消费 fresh holdout，算法早期优先使用
  Development/consumed/synthetic 数据；小型 label mapping、mask decoder、tensor
  layout 和 schema adapter 必须先在 synthetic canary 覆盖合法、未知、边界与预期失败
  路径。设备证据拆分为可参与 Development 候选排序的
  `ALGORITHM_SELECTION_BENCHMARK`，以及只验证 backend/build/operator/memory/thermal、
  不参与算法排序的 `PLATFORM_ENGINEERING_BENCHMARK`；两者都可在 formal 选模前进行，
  且都不产生 Confirmation、产品安全或默认 App authority。同步更新机器策略、validator、
  tests、治理模板和双环 current；历史协议、终态和证据不变。验证：34 项 governance
  unit tests、13-file research contract suite、历史 R3 contract CLI、JSON/py_compile、
  repo hygiene、docs index 和 diff whitespace 全部通过；R3 policy 与 R1 result、
  closeout validator、failure receipt、formal freeze 的既有 SHA-256 保持一致。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。按硕士论文、毕业设计和
  演示原型目标，前向采用 `THESIS_FIRST_RESEARCH_GOVERNANCE_R4`：新工作默认进入
  可逆 `DEVELOPMENT_STANDARD`，允许在声明的 Development/consumed 数据上做版本化
  操作修复和重跑、最多比较 3 个候选，并允许最终选模前采集 host/device runtime
  工程证据；默认取消 one-shot、逐文件 SHA、完整 hash chain 和底层全量独立复算。
  只有用户明确激活最终 Confirmation，才恢复冻结协议、独立数据、严格 validator 和
  receipt。技术故障在主张指标产出前只关闭 evidence version；结果驱动的算法修改会把
  同一数据限制为 Development。新增
  `configs/research_governance_v4.json`，validator 默认解析 R4 且按 policy ID 保留
  R1/R2/R3 历史兼容；双环 current 更新为 Development 可修复/重跑、可提前 device
  benchmark、路线 A/B 尚未选择、最终 Confirmation 未激活。历史 R1/R2-P0 文件与终态
  未修改，本次未运行模型、真机 benchmark、融合或提醒链路。验证：33 项 governance
  unit tests、13-file research contract suite、历史 R3 contract CLI、JSON/py_compile、
  repo hygiene、docs index、diff whitespace 均通过；R3 policy 与 R1 result、closeout
  validator、failure receipt、formal freeze 的既有 SHA-256 逐项保持一致。
- 时间：2026-08-01（Asia/Hong_Kong）；执行者：violjjet。完成
  `DUAL_LOOP_SEGMENTATION_R2_P0` candidate-qualification readiness：在不选择、下载或
  读取新 fresh mask truth 的前提下，冻结 SANPO native `0..30` 到 canonical `0..3`
  decoder/mapping、SHA-closed materialized canonical view、synthetic/consumed rehearsal、
  逐帧逐阶段 runtime rows、独立全量 validators 与一次预冻结 36 点 DDRNet
  postprocess refinement。924-row canonical view、200-frame/5,043-component rehearsal 和
  200-row runtime validators 均为 `VALID`；DDRNet baseline false activation
  `7.885/frame`，SegFormer total P95 `74.139 ms`，最接近 refinement 仍以 delta FP area
  `0.072513` 失败，qualified candidate 为 `0`。终态为
  `R2_NOT_WORTH_BURNING_FRESH_HOLDOUT`；R2/device/Android/risk/event/主动提醒均未授权，
  默认 App 不变。R1 四个 consumed session 永久降级为
  regression/rehearsal/validator-only，R1 result/failure/closeout/formal-freeze identities
  保持不可变。验证：17 项 module unit tests、3 个 independent validators、22 项 R1
  frozen identities 重算和 14/14 closeout requirement audit 通过；最终仓库/文档/Git
  parity 见本次提交交付。
- 时间：2026-07-31（Asia/Hong_Kong）；执行者：violjjet。为 R1 建立候选模型训练入口：
  `models.py` 固定 raw-RGB `256x256`、ImageNet normalization、四类输出和 NHWC export wrapper；
  统一 PyTorch runner 固定 400/200 train/dev、session-balanced guided crop、Adam 1200 steps、
  三 seeds、两阶段学习率、同一 weighted CE/soft Dice/focal loss 与 dev harmonic checkpoint
  规则。DDRNet-23-Slim 使用官方仓库架构与 ImageNet checkpoint 的兼容张量加载，SegFormer-B0
  使用 NVIDIA `nvidia/mit-b0` backbone 与新四类 decoder；两者均记录 source/checkpoint/config
  SHA256，且未打开 fresh holdout。当前仍仅授权 Development 训练与后续 FP32/INT8 工具链验证，
  不授权 Android/QNN/风险事件/主动提醒。验证：模型 forward/backward smoke、Python compile、
  配置 JSON 与 `git diff --check` 通过；正式训练和转换尚未完成。
- 时间：2026-07-31（Asia/Hong_Kong）；执行者：violjjet。优化研究执行治理的最低充分流程：在
  `AGENTS.md` 和 `docs/HOST_RESEARCH_COMPUTE.md` 中统一为
  `ROUTINE_ENGINEERING`、`REVERSIBLE_EXPLORATION`、`FORMAL_CONFIRMATION` 三档；普通工程和
  可重复探索不再默认触发阶段判定、双 Agent/第三方仲裁、hash-bound receipt、one-shot 或
  guarded host preflight，正式确认、受保护 outcome、不可逆或高风险任务仍保留冻结、validator、
  receipt/hash、风险分层审查和性能预检。handoff 明确为连续性工具而非研究门禁。验证：`git diff --check` 和
  文档索引检查通过；仓库卫生检查已执行，但因基线缺少
  `scripts/research/dual_loop_segmentation_model_selection/README.md` 仍失败，非本次变更引入；剩余风险：新任务仍需正确判断模式，边界不确定时按较高风险
  路由并记录升级理由。
- 时间：2026-07-31（Asia/Hong_Kong）；执行者：violjjet。切换主线至
  `DUAL_LOOP_SEGMENTATION_MODEL_SELECTION_R1`：冻结 DDRNet-23-Slim 与 SegFormer-B0
  的同 split、同输入、同增强/损失/optimizer-step、同 YOLO trace/fusion operator
  比较协议；保留 SANPO INT8 reference 为 rejected baseline，不重新调参。原 120-frame
  blind holdout 降级为 Development/regression-only，并按官方 test split 顺序冻结四条
  未消费 chest/left session 作为 fresh source-native pixel-truth formal identity。
  当前仅完成 protocol、dataset role ledger 和训练 configs；尚未读取 fresh mask 像素、
  训练候选或授权 Android/QNN/风险事件/主动提醒。详细合同见
  `docs/research/dual-loop/DUAL_LOOP_SEGMENTATION_MODEL_SELECTION_R1_PROTOCOL_2026-07-31.json`。
- 后续核验：按冻结 identity 从 SANPO official-test 下载 4 sessions × 50 frames；freezer
  只做 object metadata、GCS MD5、文件 SHA256 与图像 header 尺寸检查，不解析 semantic
  mask 像素。fresh manifest SHA256 为 `eaad2a32640dfa1a64c30fc53a6c10818a99c74b7eacc4c8718bd50515ff879d`，
  receipt SHA256 为 `90214d93c2eaa02a1355bf341bf890358c442219535c51939290e957a52ece3e`，
  200 行/4 session 已冻结为 fresh formal role。
- 时间：2026-07-31（Asia/Hong_Kong）；执行者：violjjet。冻结并执行
  `DUAL_LOOP_SEGMENTATION_CANDIDATE_UTILITY_R0`：以 SANPO-Real v0 canonical R3
  source-native pixel truth 完成 dev calibration 与 120-frame blind formal；实现
  YOLO-only、segmentation-only、union 三臂的 pixel/component、candidate outside-box、
  raw/motion-warped temporal 字段、runtime P50/P95、独立 validator 与 fail-closed
  输入/身份/SHA 回归。formal validator 为 `VALID`，唯一终态为
  `CURRENT_SEGMENTATION_REFERENCE_REJECTED`：C-minus-A recall `0.073670`、
  candidate component recall `0.688129`，但 false activation `13.833/帧 > 3.0`
  且 total incremental host P95 `138.444 ms > 30 ms`。当前 reference 关闭；不接
  Android、QNN、风险事件或主动提醒。新增 module、protocol、结果快照与证据路径见
  `scripts/research/dual_loop_segmentation_candidate_utility/`、
  `docs/research/dual-loop/DUAL_LOOP_SEGMENTATION_CANDIDATE_UTILITY_R0_RESULT.md`。
  同时将 host trace manifest 的 timestamp 检查修正为 per-source，并补齐 Kotlin
  fixed-tensor parity、numpy scalar JSON、finalize-existing、manifest duplicate/
  descending/SHA fail-closed 回归。
- 时间：2026-07-31（Asia/Hong_Kong）；执行者：violjjet。用户授权后执行并独立复核
  `DUAL_LOOP_SEGMENTATION_COMPLEMENTARITY_R1`：4,891 个 matched Shiraz Development frame，
  固定 INT8 segmentation reference，A/B/C 只输出 image-space class-wise uncovered fraction、
  temporal IoU、component count 与 host cost。pairing/finite/hash/union validator 为 `VALID`；
  `unknown_nonwalkable` temporal IoU median `0.725020`、`obstacle` `0.249790`、
  `boundary_step_curb` `0.080014`，稳定性按 class 混合。终态为
  `IMAGE_SPACE_SIGNAL_OBSERVED / STABILITY_MIXED_BY_CLASS / NO_FUSION_EFFECT_AUTHORITY`；
  不产生风险、事件、Android 或生产结论。结果快照见
  `docs/research/dual-loop/DUAL_LOOP_SEGMENTATION_COMPLEMENTARITY_R1_RESULT_2026-07-31.md`。
- 时间：2026-07-31（Asia/Hong_Kong）；执行者：violjjet。将用户提出的后续资源纪律固化到
  双环研究入口：失败路线最多一个 successor；fresh 双路语义失败后不再用第三 Agent、
  prompt、slot 或数据选择制造一致；下一阶段必须直接产生算法对照、端侧性能或路线关闭，
  不再增加只有 readiness 名称变化的多层阶段。该规则不改变已消费的 D0-A 结果。
- 时间：2026-07-31（Asia/Hong_Kong）；执行者：violjjet。修正双环研究入口标题，明确当前
  主线为 `YOLO + 语义分割 + 融合 C`；中央图像阻塞仅作为已经关闭的辅助观测审计，Q0
  semantic-refresh 仅作为独立封存旁路线保留。未改变任何历史结果、研究授权或实验状态。
- 时间：2026-07-31（Asia/Hong_Kong）；执行者：violjjet。冻结
  [segmentation complementarity Development design R0](docs/research/dual-loop/DUAL_LOOP_SEGMENTATION_COMPLEMENTARITY_DEVELOPMENT_DESIGN_R0.md)：
  基本单位为同一 `source/frame/image_sha256` 的 YOLO box 与 segmentation mask 配对，
  主 estimand 为 `segmentation mask - YOLO box union` 的 per-class uncovered fraction，
  session 先聚合，显式处理时间依赖、缺失配对和 burned Development 角色。A/B/C 只定义
  image-space coverage/union 输出，不读取 risk、feedback、event 或中央阻塞标签，也不
  产生可通行性、风险或融合效果主张。当前仅 `DESIGN_ONLY / NOT_EXECUTED`，因为 D0-B
  效果与融合仍未授权；没有把 4,891 frame matched Shiraz trace 冒充 held-out evidence。
- 时间：2026-07-31（Asia/Hong_Kong）；执行者：violjjet。按用户纠正后的主线，把
  YOLO + semantic segmentation 与 Q0 semantic-refresh 分开；保留中央阻塞 D0-A successor
  的不可变终态 `CENTRAL_OBSTRUCTION_AUXILIARY_FEATURE_ONLY`。新增独立
  `dual_loop_segmentation_technical_smoke` Module、contract tests 与
  `DUAL_LOOP_SEGMENTATION_TECHNICAL_SMOKE_R0` 结果快照。runner 只接受一个已声明
  reference model、隐藏 candidate/prior-review 的 fixed RGB slot，不读取中央阻塞标签、
  YOLO、风险、反馈或融合，不提供模型比较/阈值/拓扑接口。
  在 24 slot / 6 fixed clip / 3 source 上完成 smoke：tensor 合同和有限值通过；argmax
  像素 `walkable=100%`，其余三类为 `0%`。主机端 TFLite P50/P95/MAX 为
  `5.2386/8.1098/12.2758 ms`，未作手机/Snapdragon 结论；contact sheet 与 JSON 报告
  保存在 `artifacts.local/evidence/dual-loop-segmentation-technical-smoke-r0/`。
  终点为 `TECHNICAL_ONLY / NO_EFFECT_AUTHORITY`：不授权 D0-B、融合、Android 或生产；
  语义分割正式选型、客观互补单位和 A/B/C 仍未完成。验证：5 项 focused tests、Python
  compile、runner smoke 和可视化检查通过；详细结果见
  [technical smoke R0 result](docs/research/dual-loop/DUAL_LOOP_SEGMENTATION_TECHNICAL_SMOKE_R0_RESULT_2026-07-31.md)。
- 时间：2026-07-31（Asia/Hong_Kong）；执行者：violjjet。完成
  [中央图像阻塞 D0-A successor R0](docs/research/dual-loop/CENTRAL_OBSTRUCTION_AGENT_LABEL_READINESS_D0_A_SUCCESSOR_RESULT_2026-07-31.md)：
  保留 observation-level Agent 标签，新增冻结的 1 秒 fixed-clip/四 slot 转换函数、
  content-blind input freezer、双 isolated-review validator 与 4 项 unit tests；fresh
  calibration 使用 3 个未进入 burned 11 clips 的 session、6 个 clip、24 个 slot。
  固定边界复现 `1.0`，但两路 Agent observation agreement `16/24=0.6667`、unresolved
  `8/24=0.3333`、unit-state match `4/6=0.6667`，正式终态为
  `CENTRAL_OBSTRUCTION_AUXILIARY_FEATURE_ONLY / D0_A3_A4_STOPPED`。D0-A1 的
  `0.8545/0.8298` 保持不变；D0-A2/D0-AT/D0-B、模型效果、Android 与默认行为均未授权。
  冻结、validator、JSON schema、单测与文档索引/卫生检查见 successor evidence root 和
  结果快照。
- 时间：2026-07-31（Asia/Hong_Kong）；执行者：violjjet。完成
  [中央图像阻塞 D0-A1 R2](docs/research/dual-loop/CENTRAL_OBSTRUCTION_AGENT_LABEL_READINESS_D0_A1_RESULT_2026-07-31.md)
  的 fresh isolated second pass、8 项 third-Agent adjudication 与最终 readiness：fork-none
  second Agent 在 primary/aggregate/model output 不可见时覆盖 11/11 clip、55/55
  observation，raw SHA `47049587...d930`；两遍 observation/claim-critical agreement
  `0.8545/0.8298` 与 boundary P95 `1` 过门，但 parent-event match
  `12/19=0.6316 < 0.75`。8 个分歧全部裁决，7 个 adjudicated、1 个隔离为
  `NOT_EVALUABLE`；裁决未覆盖 raw review/event，最终 canonical `34/9/12`、19 event，
  终态 `AGENT_LABEL_PROTOCOL_NOT_RELIABLE / VALID`。新增 isolated/agreement 与
  adjudication/final readiness 与 post-output recomputation validator，D0-A1 15 项
  focused tests、模块合计 22 项测试及 compile/protocol/docs/structure/hygiene 门通过。D0-A2、D0-AT、
  D0-B、模型效果与 Android 均未授权；下一边界只能另立 D0-A 版本，在 burned
  calibration stress cases 上重设计 observation/event workflow，不得调 R2 门救援。
- 时间：2026-07-31（Asia/Hong_Kong）；执行者：violjjet。正式进入
  [中央图像阻塞 D0-A1](docs/research/dual-loop/CENTRAL_OBSTRUCTION_AGENT_LABEL_READINESS_D0_A1_ENTRY_2026-07-31.md)：
  从 D0-A0 的 calibration-only 角色中冻结 JRDB/Ulm/Alicante/Burwell 4 source、
  11 clip、55 observation，production overlap 与 candidate-output access 均为 0；
  ROI、三态 prompt、parent-event/matching、claim-critical 双 pass、low-risk 20%
  audit 与 readiness 数值门均已锁定。R0 source-only inspection 发现“任意 scene
  element 占 ROI”会把背景建筑误当阻塞，故在任何 raw label 前立 R1，将 positive
  收紧为前/中景实体实际遮挡后景或终止中央视线，并修正连续
  `NOT_EVALUABLE` event 合并。R1 输入 producer/独立 validator 为 `VALID`，但原始
  primary 的 submission time 晚于 validator，已按
  `INVALID_REVIEW_TIMESTAMP_ORDER` 保留。R2 只修复 evidence identity/output root
  与时间戳，显式披露前序访问并原样转录 55 个标签；当前三态为 `28/12/15`、
  18 parent event，coverage precondition 全过，但仍是非隔离 context，
  agreement/readiness 未评价，D0-A2/D0-B 均未授权。D0-A1 7 项 focused tests、
  模块合计 14 项测试及 Python compile、协议/文档/structure/hygiene 门通过后提交。
  下一动作只允许 fresh isolated second pass，不得读取 primary label 或修改 R2 lock。
- 时间：2026-07-31（Asia/Hong_Kong）；执行者：violjjet。完成
  [中央图像阻塞 D0-A0 输入宇宙冻结](docs/research/dual-loop/CENTRAL_OBSTRUCTION_AGENT_LABEL_READINESS_D0_A0_RESULT_2026-07-31.md)：
  reuse-first 审计后冻结 6 个完整 production-labeling session、34,279 帧、5 个
  ancestry group，逐帧 payload 独立复算为 `VALID`；107 行角色账本另记录 61 个
  calibration-only 与 40 个当前问题不可评价单元。保留 REveL 46 帧非主尺寸和其中
  9 帧极窄边 burden，D0-A1 必须 fail closed/`NOT_EVALUABLE`，不得删帧救援。
  R0 因遗漏 mandatory role ledger 封存为 implementation-incomplete；R1 因 producer
  后协议哈希并发漂移封存为 invalid predecessor；R2 因把 primary task 的 source-only
  review 错写成 isolated context 而封存。write-once 输出均未覆盖；R3 如实披露
  `isolated_context=false / source_only_view=true` 后通过独立 validator。7 项
  focused mutation/regression tests 通过。
  本阶段未生成标签、未读 candidate/truth/review 输出、未启动 D0-A2 或 D0-B；当前
  唯一下一动作是 D0-A1 排除式 calibration 与标签合同锁。
- 时间：2026-07-31（Asia/Hong_Kong）；执行者：violjjet。前向采用
  `RISK_TIERED_RESEARCH_GOVERNANCE_R3`，建立 `CANARY_LITE /
  DEVELOPMENT_STANDARD / CONFIRMATION_STRICT` 三档执行配置；R1/R2 policy 与历史
  receipt 保持不可变。Canary 默认不再要求穷尽数据、全量双 Agent、完整 hash chain
  或 one-shot；Development 允许在 burned 数据上按固定预算比较至多 3 个有因果差异
  的候选并在 held-out 前冻结一个；Confirmation 保留完整冻结、独立 validator 和
  禁止结果后救援。Agent review 改为确定性校验、低风险单 Agent 加冻结抽样审计、
  关键/歧义双 pass、材料分歧才第三 Agent；单次控制面错误可用轻量 incident receipt。
  guarded host preflight 改为正式 one-shot/不可逆 claim、预计超过 15 分钟或高资源
  风险触发，3–15 分钟可逆任务采用轻量运行合同。D0-A 已继承 `CANARY_LITE`：按适配度
  排序现有 RGB 并满足充分性即停止，先做标签 pilot；允许在排除数据上运行一个不产生
  效果证据、对标签 Agent 隐藏的 reference model-B 技术 smoke；正式标注采用风险分层
  Agent 审计，后继 D0-B 改为 bounded Development shortlist 后再冻结 held-out 候选。
  未运行标注、模型效果、融合、Android 或设备实验，默认 App 与历史终态不变。
- 时间：2026-07-31（Asia/Hong_Kong）；执行者：violjjet。小范围修订
  `CENTRAL_OBSTRUCTION_AGENT_LABEL_READINESS_D0_A`：将
  `REUSE_FIRST / FITNESS_FIRST` 固化为 D0-A0 准入原则，禁止仅因数据集曾被其他
  算法、主线或实验使用而整体排除；逐 session 记录当前适配度、实际历史访问、
  claim overlap、当前角色和局部排除原因。受污染单元优先降级为 calibration、
  Canary、Development、诊断、回归或压力样本，而不是丢弃；未增加数据、模型 B、
  融合、调度、Android、Confirmation 或产品权限。
- 时间：2026-07-31（Asia/Hong_Kong）；执行者：violjjet。按用户澄清将
  `HETEROGENEOUS_PLATFORM_P0A_R0` 的评估对象从未连接的 A568 收窄为当前真实连接的
  手机 `SM-S9280 / SM8650`（serial `R5CX10M8Y8X`）。完成手机 P0A 预检并达到
  `PLATFORM_ADMITTED_FOR_CANARY`，范围仅为 LiteRT/TFLite CPU 4-thread 与 GPU
  delegate；A568 历史报告仍保持 `HOLD_NOT_EVALUABLE`，没有用手机替代 A568。
  固定 10 帧/30 次 detector run 的 CPU 路径为 `53/53 ms` P50/P95、失败 `0`；
  `PRODUCTION_ROUTE -> CPU_XNNPACK` 60 秒为 `587` 帧、`9.776 FPS`、`54/57 ms`
  total P50/P95，GPU 60 秒为 `590` 帧、`9.820 FPS`、`47/65 ms`，两者失败均为 `0`，
  thermal status 均为 `0`。QNN 2.47.0 虽通过 HTP capability probe，但实际初始化因
  设备端缺少 `libQnnHtpV75Skel.so` 失败；生产路由已记录原因并回退 CPU，QNN 子路径
  保持 HOLD。仅给 device benchmark 增加 `BaselineOnly` 和无候选报告的安全默认值，
  未改变默认模型、生产路由或任何研究结论。证据见
  [手机准入报告](docs/HETEROGENEOUS_PLATFORM_PHONE_P0A_R0_2026-07-31.md) 与
  `artifacts.local/evidence/phone-admission/20260731-161634/`。没有外部功率仪表，
  功耗/能耗不作结论；T4 因缺少第二平台未开始，T5 继续关闭。
- 时间：2026-07-31（Asia/Hong_Kong）；执行者：violjjet。采用
  `CENTRAL_OBSTRUCTION_AGENT_LABEL_READINESS_D0_A` 为当前论文系统研究主线：
  D0-A 按 `CANARY` 只冻结连续 RGB 输入宇宙，并以两路隔离 Agent、分歧时 fresh
  第三 Agent 建立不可覆盖的中央图像阻塞 parent-event 账本；不设置人工队列。
  合同、AI workflow、项目/文档索引和权限表已同步，模型 B、拓扑算子、融合、调度、
  Android、A568、可通行性与安全结论保持关闭。验证使用 research protocol
  validator、JSON parse、文档索引、项目结构、repo hygiene 与差异检查。
- 时间：2026-07-31（Asia/Hong_Kong）；执行者：violjjet。完成
  `CI_RESOURCE_ISOLATION_R1` 的第一轮资源与门禁修复：CI run
  [30609736963](https://github.com/violetljj/blind-assist/actions/runs/30609736963)
  在同一 Gradle 大调用中让 `:app:packageUstrfExperiment` 与
  `:npu-candidate:mergeExtDexDebug` 同时触发 D8 `Java heap space`；现将原有
  结构/卫生、单测/lint、正式 App、USTRF、NPU/benchmark、研究合同/模型检查拆为
  独立 job，所有原有任务和失败门禁保留。Gradle job 固定使用现有 `2 GiB` heap、
  `--max-workers=2`，NPU 与 device benchmark 在同一 job 内顺序执行，未使用
  allow-failure 或无限增大 heap。
- 分组本地 warm/incremental 验证使用 Temurin 17.0.19、Gradle 8.10.2：
  unit/lint `236 actionable tasks` 成功；正式 App debug APK、androidTest、bundle
  与 release assets 成功；USTRF、shadow benchmark、NPU candidate、device benchmark
  均成功。NPU candidate 另补齐共享 `MainActivity` 所需的两个恒 false
  `BuildConfig` 字段；这是隔离后暴露的编译阻塞，不改变 candidate 行为。
- 原有研究合同套件首次暴露 U0 合同的 official config、truth-validator 和 runner
  implementation SHA 漂移；仅刷新当前文件的身份绑定，依赖清单、门槛和 U0/训练/生产
  authority 不变，完整 `13/13` contract files、`failure_count=0` 通过。
- 正式 App 的严格 16KB 门禁还证明 `useLegacyPackaging=true` 会产生压缩 native
  libraries；将 App 改为不压缩 JNI，APK 与 AAB 分别通过 `PAGE_ALIGNMENT_16K`。
  这会增加本地 debug 包体并留下 runtime/package-size parity 风险，NPU candidate
  仍保持其独立 legacy packaging，后续不能把静态门禁当作真机兼容性证明。
- 本轮不运行研究实验、不改变默认模型或研究结论。提交
  `63ea3bcfc72d3f483c1039ee17614b277048e540` 的完整 workflow 首轮与 rerun attempt 2
  均通过（run `30613616160`），USTRF/NPU 无 D8/package OOM；随后主线由并发研究
  提交推进至 `3b8c52ce9a56e8bf9f28872872cea5270ff23e26`，其 CI run `30613683882`
  亦通过。
- 时间：2026-07-31（Asia/Hong_Kong）；执行者：violjjet。完成
  `STACKED_PR_CLOSEOUT_R0`：相对当前 master，PR #3 与 #1 的 head 均为
  `ahead_by=0`（分别落后 173/62 commits），没有独有提交或文件需要迁移。已在
  两个 stacked PR 留下 obsolete 原因并关闭，`mergedAt=null`；未合并旧分支，未带入
  历史研究结论。
- 时间：2026-07-31（Asia/Hong_Kong）；执行者：violjjet。完成
  [DUAL_LOOP_STAGE_CLOSURE_R0](docs/research/dual-loop/DUAL_LOOP_STAGE_CLOSURE_R0_2026-07-31.md)：
  只整理既有机制、工程和事件级证据，并从 dual-loop README 链接；报告包含
  baseline/shadow/isolated-active/default-off、三来源事件结果和 retained-false
  五类分解三张教师可见 Mermaid 图，不运行新实验、不授权调度或 active R2。
- 时间：2026-07-31（Asia/Hong_Kong）；执行者：violjjet。完成
  `HETEROGENEOUS_PLATFORM_P0A_R0` 设备准入预检：ADB 健康检查为 ready，但唯一可访问
  设备为 `SM-S9280 / SM8650`，不是 A568；A568 硬件、runtime、模型加载、固定帧结果、
  温度/功耗/可复算日志均缺失。按门禁直接终止为 `HOLD_NOT_EVALUABLE`，不进入 T4/T5。
- 时间：2026-07-31（Asia/Hong_Kong）；执行者：violjjet。按用户明确授权完成
  [TARGET_LOCAL_BACKGROUND_WARP_RESIDUAL_R0 B Development 实现与单次执行](docs/research/dual-loop/TARGET_LOCAL_BACKGROUND_WARP_RESIDUAL_R0_IMPLEMENTATION_REVIEW_RESULT_2026-07-31.md)：
  在隔离 offline Module 中完成 truth-blind producer、独立 truth-late evaluator、
  synthetic fixtures、burned-input 准备器、实现锁及 root adapter；合同测试 `15/15 PASS`，
  `12,876` 个 REveL burned pair 已冻结（含 `32` 个 shape mismatch pair），producer 输出
  `51,504` 行且 receipt/hash 校验一致，truth-late join 为 `6,538` 行。R1 唯一选择的
  paired event gain 为 `-34/960`，覆盖率下降，终点为
  `NO_DEVELOPMENT_INCREMENT / CLOSE_CANDIDATE`；不重跑、不调参，不启动 C1/C2、Android
  或产品行为。
## 2026-07-31
- 时间：2026-07-31（Asia/Hong_Kong）；执行者：violjjet。完成
  [DUAL_LOOP_R1_EVENT_FAILURE_DECOMPOSITION_R0](docs/research/dual-loop/DUAL_LOOP_R1_EVENT_FAILURE_DECOMPOSITION_R0_RESULT_2026-07-31.md)：
  只读消费已关闭的 CrowdBot、Matoaka、Shiraz Development trace、truth ledger 与
  receipt，覆盖 49 个 ledger 窗口（24 个正例、25 个负窗；47 个 closed-scored，2 个
  CrowdBot 正例为 `TEMPORAL_SCORING_NOT_EVALUABLE`）。逐窗口评分范围内的
  baseline/candidate feedback rows 为 `206/202`，负窗分别为 `7/7`、`7/7`、`5/5`；
  retained-false 分类为
  `A=1/B=2/C=10/D=4/E=2/MIXED=0`。顶层终点为
  `POLICY_GRANULARITY_MISMATCH_SUPPORTED`：只在内存 upper-bound audit 中发现
  CrowdBot `49.241 ms`、Matoaka `900 ms` 的有限 hold witness，均保留 baseline-hit
  正例、induced negative window 为 0、正例新增首反馈时延为 `0 ms`；Shiraz 在预冻结
  `250 ms` 上限内无 witness。该 witness 需要新 runtime state，不是新的 R1 效果主张；
  不设计或实现 R2，建议关闭 scene-scale active 路线。逐窗口 CSV/JSON/Markdown、
  upper-bound JSON、确定性测试与 LF 字节测试已交付。
- 时间：2026-07-31（Asia/Hong_Kong）；执行者：violjjet。完成
  [DUAL_LOOP_SEMANTIC_REFRESH_Q0](docs/research/dual-loop/DUAL_LOOP_SEMANTIC_REFRESH_Q0_PROTOCOL_2026-07-31.json)
  的独立离线 R0 实现与单次 Development 回放：固定模型全频参考在 4,422 帧、两
  session 上通过逐帧 parity；33/66/100/167/267 ms fixed-time arms 的 detector
  calls 为 `3309/2793/2430/1560/1077`，Level-3 divergence 为
  `122/201/262/404/533`，event-window 命中未改变。feature-rule 与 Logistic arms
  因缺少独立 current-frame-only fast-feature trace 保持 `NOT_EVALUABLE`；本轮只支持
  Development-only reference-preservation 筛查，不授权 Android、能效、产品或安全结论。
- 时间：2026-07-31（Asia/Hong_Kong）；执行者：violjjet。完成
  [DUAL_LOOP_SEMANTIC_REFRESH_Q0_R0_1](docs/research/dual-loop/DUAL_LOOP_SEMANTIC_REFRESH_Q0_R0_1_EVALUATION_PROTOCOL_2026-07-31.json)：
  不重跑 detector，消费既有 Q0 trace 补齐 risk-episode segmentation/matching、独立
  active event ID、signed feedback delay、P50/P90/P95、feedback count delta 和
  zero-order-hold stale duration。8/8 Q0 单元测试、真实 4,422 帧 replay 与 parity 通过；
  raw nondominated set 为 6 个 VALID arms，预声明门下 admissible 为
  `FULL_RATE_REFERENCE / FIXED_TIME_33MS`，constrained best 为 `FIXED_TIME_33MS`。
  该 operating point 仅是两 session Development 评测诊断，不授权 learned scheduler、
  Android、能效、产品或安全结论。
- 时间：2026-07-31（Asia/Hong_Kong）；执行者：violjjet。完成
  [双环 R1 未见事件 R0 rank-1](docs/research/dual-loop/DUAL_LOOP_R1_UNSEEN_NATURAL_EVENT_R0_RANK1_RESULT_2026-07-31.md)
  的 truth-first 终点。下载的 480p payload SHA-256 为 `589711...f49`，生成
  566 帧 1 Hz review bundle 与 5,662 帧 10 Hz 固定 replay input。两路隔离、
  hash-bound canonical-prompt RGB 复核均得到 `0` 个高置信正例；另有新上下文
  对早期 `0 vs 7` 分歧逐段裁决，同样拒绝全部争议段。预冻结最低正例为 3，
  因此在 baseline 前有效终止为 `FIRST_UNSEEN_SOURCE_NOT_EVALUABLE`；
  `039757b` candidate 未打开、未调参，不声称算法失败。保留 6 个两路一致负窗为
  source-characterization/regression。输入准备与 selector 专项测试 `4/4 PASS`；
  下一步只允许按原 registry 排序启动 rank-2 新 evidence instance。
- 时间：2026-07-31（Asia/Hong_Kong）；执行者：violjjet。按用户确认的
  “效果线优先”路线冻结
  [双环 R1 未见自然来源事件评价 R0](docs/research/dual-loop/DUAL_LOOP_R1_UNSEEN_NATURAL_EVENT_R0_PROTOCOL_2026-07-31.json)。
  在任何视频 payload、baseline 或 candidate 输出访问前，以 Commons
  `First-person videos on foot` 的 57 项 metadata snapshot、固定 eligibility 和
  Unicode title 顺序选中 566.228 秒上海夜间步行视频；registry SHA-256 为
  `0a34051f...ca8d`。评价单位固定为预冻结正例事件/负例窗口，baseline adequacy
  只决定可评价性；同 ID retention、逐事件 250 ms 时延、induced-negative 与
  absolute recall 是 guardrail，反馈证据层级仅为设备回放的 simulated controller
  acceptance。单来源最高只到 event-level Development canary，不调 R1、不把 row
  下降当 event effect。selector 专项测试 `2/2 PASS`；视频尚未下载或查看。
## 2026-07-30
- 时间：2026-07-30（Asia/Hong_Kong）；执行者：violjjet。完成
  [双环隔离主动纠错 R1](docs/research/dual-loop/DUAL_LOOP_ACTIVE_CORRECTION_R1_RESULT_2026-07-30.md)：
  先以 4,422 帧完整 production detections 否决没有负例反证命中的 multitrack R0，
  再落地最小 scene-scale `ACTIVE_CONTRADICT_ONLY`。设备 Kotlin 回放与 host
  evaluator 达到 `4422/4422` 逐帧一致；CrowdBot 全序列触发行 `373 -> 357`、
  可评分负例行 `27 -> 25`，Matoaka 10,724 帧 strict QNN HTP 回放为
  `255 -> 247`、可评分负例行 `51 -> 49`，两者负例提醒窗口均 `7 -> 7`，已命中
  正例无新增延迟。100 ms hold 没有事件收益，200 ms 起损害 CrowdBot 正例召回，
  因而拒绝 latch/新状态机。独立 `dualLoopActive` APK 已安装冷启动并显示开发禁用
  警示；普通构建默认关闭，raw/stable risk 与事件规则不变。Python 合同测试
  `4/4`、`core:assist` `161/161`、普通/影子/主动 APK 与 device-benchmark 构建、
  repository hygiene 和 docs index 均通过。终点为
  `ISOLATED_ACTIVE_MECHANISM_LANDED / CROSS_SOURCE_ROW_SIGNAL_REPLICATED /
  NO_EVENT_LEVEL_EFFECT`，不构成默认生产、产品或安全主张。
- 时间：2026-07-30（Asia/Hong_Kong）；执行者：Codex。封存
  [D0 ego-motion error attribution R2 正式执行](docs/research/dual-loop/DUAL_LOOP_D0_EGOMOTION_ERROR_ATTRIBUTION_R2_EXECUTION_RESULT_2026-07-30.md)：
  R2 已修复 R1 的 `rosbags` 缺失，并通过冻结首条 Vicon message probe；正式 marker
  后在 calibration parser 的动态 `import yaml` 处失败。终态为
  `EXECUTION_INVALID / CONSUMED / NO_RERUN / NO_SCIENTIFIC_EXIT`，进度
  `0 / 469`，无 event table、analysis 或科学出口。R2 不补包重跑；只有新的 R3
  runtime-recovery identity 可在显式 PyYAML、全 reachable-import smoke、真实
  calibration output-blind parser smoke 与独立复核后继续，科学合同不变。
- 时间：2026-07-30（Asia/Hong_Kong）；执行者：Codex。冻结并通过
  [D0 ego-motion error attribution R1 设计复核](docs/research/dual-loop/DUAL_LOOP_D0_EGOMOTION_ERROR_ATTRIBUTION_R1_DESIGN_REVIEW_RESULT_2026-07-30.md)。
  R1 将 R0 不可识别的 `*_DOMINANT` 改为
  `EGO_CANARY_PRIORITY / TEMPORAL_TREND_PRIORITY / NO_PRIORITY_IDENTIFIED`，
  只作 burned single-capture operational routing。dependency preflight 复算
  469 个 primary events、159 个跨 target overlap pairs、310 个 transitive
  components 与六个固定 60 秒块；真实 golden 与 mutation tests `6/6 PASS`。
  经三轮独立统计/实现复核，闭合 A/B trace firewall、exact join、Vicon/ROI 时间基、
  source/share/quality support、component/block weighted Cliff、missingness、
  person competing、互斥出口、one-shot、canonical receipt 与独立 validator。
  协议 SHA-256 为
  `87931369f912fdd054783db9decb2a1813080d0a961c3526b83ce686d1a48183`；
  当前只授权实现和测试，正式 D0、后继 canary、Confirmation、Android、产品与安全
  仍未授权。
- 时间：2026-07-30（Asia/Hong_Kong）；执行者：Codex。完成
  [production temporal geometry factorial A/B R0 正式执行](docs/research/dual-loop/DUAL_LOOP_PRODUCTION_TEMPORAL_GEOMETRY_FACTORIAL_AB_R0_EXECUTION_RESULT_2026-07-30.md)：
  指定 `SM-S9280 / SM8650` 以 strict QNN HTP 对 4,422 帧逐帧一次 detector，
  生成 8,844 行隔离 A/B trace，producer `COMPLETE`、failure `0`。truth-blind
  validator 逐帧核对后发布 `SEALED`，随后 evaluator 才连接冻结的 17 项 truth。
  两臂在 8/8 可评分正例、7/7 负窗与两个 session 上的实际提醒完全一致：
  首次提醒增益全部为 0、paired correctness delta 为 0，且共同出现 6/8
  premature events 与 7/7 false-alert windows；终点为 `VALID / NO_INCREMENT`，
  one-shot authority 已消费，Confirmation 不授权。seal 后描述性定位发现 temporal
  geometry 改变 3,285 帧 approach trend 和 973 帧 risk score，却从未改变 level、
  proximity、feedback 或 event；后继转入统计修复后的 D0 R1 operational routing，
  不再继续同构 score-boost A/B。
- 时间：2026-07-30（Asia/Hong_Kong）；执行者：Codex。完成
  [production temporal geometry factorial A/B R0 实现复核](docs/research/dual-loop/DUAL_LOOP_PRODUCTION_TEMPORAL_GEOMETRY_FACTORIAL_AB_R0_IMPLEMENTATION_REVIEW_RESULT_2026-07-30.md)：
  生产 `TemporalRiskTracker` 中和因子、共享单次 QNN detections 的双臂隔离决策链、
  truth-blind device producer、implementation lock、activation gate、逐帧独立
  validator、原子 seal 与 seal-only truth evaluator 均已实现。独立审计提出的
  timestamp/truth/hash 绑定、marker/并发/TOCTOU、锁定源码覆盖和终点测试缺口均已
  修复，最终为 `PASS`。核心 clean test `135/135`、Python mutation tests `6/6`、
  Android build、真机 strict-QNN prestart、PowerShell/结构检查均通过；无授权
  正式入口在 marker 前按预期失败且 formal namespace 仍为空。实现 commit 为
  `2c53e89a67ec7848a7d2290ebf9e627f6bc96ff6`，implementation lock SHA-256 为
  `d7383b9339d46935599d1f0da9bd163b78dd159050e8409a0578969ef9bb23de`。
  当前只授权绑定后的唯一正式运行；候选输出、truth join、Confirmation、生产行为、
  产品与安全结论仍未执行或授权。
- 时间：2026-07-30（Asia/Hong_Kong）；执行者：Codex。将神经—几何双环主线从
  LITE R2 负结果后的 D0 单一路线，改为优先验证既有生产 `TemporalRiskTracker` 的
  object-detector temporal geometry contribution。冻结并通过
  [production temporal geometry factorial A/B R0 设计复核](docs/research/dual-loop/DUAL_LOOP_PRODUCTION_TEMPORAL_GEOMETRY_FACTORIAL_AB_R0_DESIGN_REVIEW_RESULT_2026-07-30.md)：
  A 只中和 object-detector temporal output，B 保持当前生产行为；两臂共享一次 QNN
  detections 但隔离全部有状态链。新增 outcome-blind input preflight，复算
  `4422/4422` PNG 哈希/尺寸/时序；独立 truth-membership preflight 冻结 17 项原始
  truth 为 8 positive + 7 negative，候选输出前排除 `P007/P009` 两个零帧正例。
  两项独立复核均为 `PASS`，当前仅授权 factorization 实现、合成 mutation tests、
  implementation lock 和 activation review；正式 A/B、truth join、Confirmation、
  生产行为、产品与安全结论仍未授权。另补
  [D0 独立统计复核](docs/research/dual-loop/DUAL_LOOP_D0_EGOMOTION_ERROR_ATTRIBUTION_R0_STATISTICAL_REVIEW_RESULT_2026-07-30.md)，
  将单 capture 下的 `*_DOMINANT` 因果解释判为 `REPAIR_NEEDED`，D0 转为 A/B
  `NO_INCREMENT` 后的后备 operational routing。
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

## 2026-07-28

### RCLE R2 generator geometry P1 R0
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

## 2026-07-30：生产 TemporalRiskTracker 因子 A/B R0 实现与真机预启动

- 执行者：violjjet
- 在默认生产行为不变的前提下，为 `TemporalRiskTracker` 增加仅中和
  `DetectionSource.OBJECT_DETECTOR` temporal output 的显式模式；A/B 各自持有隔离的
  tracker、stabilizer、event、feedback、cooldown 与 fatigue 状态，单帧 QNN detector
  输出只生成一次并以 canonical hash 绑定两臂。
- 新增 truth-blind Android producer、正式 one-shot marker、逐帧冻结 RGB SHA
  再校验、独立 producer validator、post-seal truth evaluator、implementation lock
  与 activation gate。validator 不接受 truth，逐帧对照冻结 frame ledger 的 ID 与
  timestamp，并原子发布绑定 trace/producer/lock/activation/validation 全部哈希的
  seal；evaluator 不接受裸 trace 或自报 validation，只接受该 seal 与 lock-bound
  truth-membership receipt。
- 核心 tracker mutation/order/segmentation parity tests、Python evaluator/validator
  fixtures 与 Android debug build 通过。生产 app 与 instrumentation APK 已装入指定
  `SM-S9280 / SM8650`；prestart 复核 `4422/4422` 帧、`2,612,679,375` bytes、
  canonical inventory SHA
  `45621b226b4f6286962ec39c548234f92c3a34331cc4a1b2c413ef0bd3f7dd3b`，并以
  `qualcomm_qnn_htp / QNN 2.47.0` 完成 synthetic live probe。
- prestart 明确记录 `decision_rgb_decoded=false / candidate_output_written=false`；
  完整 kernel + feedback 链的 A/B 调用顺序 mutation 也通过。一次缺少设备端
  authorization 的 formal-entry 失败注入在首帧 decode 和 marker 前被拒绝，随后复核
  marker、temporary trace 与 output namespace 均不存在。正式 A/B 尚未 activation，
  truth join 与 Confirmation 尚未执行，不能据此形成增量、产品或安全结论。

## 2026-07-30：D0 ego-motion error attribution R3 最终恢复实现

- 执行者：violjjet
- R1 在 marker 后因冻结环境缺少 `rosbags`、R2 在 marker 后因缺少 `PyYAML`
  分别关闭为 `EXECUTION_INVALID / CONSUMED / NO_RERUN /
  NO_SCIENTIFIC_EXIT`；两次均为 `0/469`，没有 event table、analysis 或科学出口。
- R3 只恢复运行时和控制面，不改变科学合同；`analysis.py`、`bindings.py`、
  `producer.py` 在 R1/R2/R3 byte-identical，23 个科学字段类型和值精确一致。
- 为防止再次消耗 one-shot，冻结独立 R3 venv、八项 distribution tree、AST import
  closure、PyYAML module provenance、继承的 R2 operational probe，以及 producer
  与独立 validator 双 parser synthetic calibration smoke。
- Marker 防火墙改为强制显式 scope：review、activation、CLI 与 runner premarker
  不得打开 predecessor/current scientific inputs；`formal_start` 与初始 progress
  持久化后才运行完整 scientific-input validation、bundle、calibration 与 tracks。
- R1/R2 current/archive 和 exact inventory 均纳入两个 validator；VALID progress
  与 receipt 同一失败闭包，已有 terminal receipt 时零写拒绝重入。冻结解释器
  `56/56 PASS`，runtime、项目结构、live R1/R2 gate 与三路独立复审均通过。
- 共享 guarded host launcher 新增显式 Python 前置参数，使 formal child 与
  preflight validator 都能在 `-I -B` 下运行；R3 progress 同时满足既有科学状态和
  host guard 的 phase/units/throughput/ETA/time/status 合同。集成测试验证前置参数、
  worker 注入、成功、失败、stale progress 与 invalid preflight 均 fail-closed。
- 当前仅为 `DESIGN_PASS / IMPLEMENTATION_PASS / ROUTE_PASS / NOT_RUN`；协议 SHA
  为 `4412390fcfb4b4588600c368d3cb36a6ece875ec3f97ea7ef8bd051886f11064`，
  `run-r3/` 不存在。实现提交推送并生成 exact lock/review/activation 前不授权正式
  执行；marker 后失败永久 `NO_R4`。详见
  [R3 review](docs/research/dual-loop/DUAL_LOOP_D0_EGOMOTION_ERROR_ATTRIBUTION_R3_DESIGN_REVIEW_RESULT_2026-07-30.md)。

## 2026-07-30：D0 R3 唯一正式执行与不可变关闭

- 执行者：violjjet
- R3 exact lock/review/activation 与 guarded host preflight 完成；初始 8 GiB 门被
  证明是 `4 GiB worker estimate + 4 GiB host reserve` 的过度估算。基于 R2 同机
  完整顺序解码、约 86,033 条 Vicon pose、约 35 MiB 其余冻结 JSON、流式 bag/hash
  读取，将 worker 工程预算独立复核为 1.5 GiB，4 GiB host reserve 不变。
- 唯一 formal producer 创建 `formal_start` 后，调用侧前台工具超时中断外层
  monitor，但未终止 exact `python -I -B ... produce` 子进程；该既有进程被只读
  跟踪到原子 failure receipt，没有重启或重跑。
- 终点为
  `EXECUTION_INVALID / CONSUMED / NO_RERUN / NO_R4 / NO_SCIENTIFIC_EXIT`：
  `0/469`，错误 `BBOX log-area closure mismatch`，没有 event table、analysis、
  producer/execution receipt 或 D0 三出口。
- 静态根因确认：LITE 冻结 BBOX 字段是
  `0.5 * delta(log(area)) / dt`，D0 R1/R2/R3 却按
  `delta(log(area)) / dt` 闭合并要求 `1e-12/s` 一致，有限非零行必然系统性相差
  两倍。独立 validator 复制了同一错误语义，synthetic tests 只验证 D0 内部自洽。
- R3 保持不可变，不生成 R4，不据此选择 ego/temporal。详见
  [R3 execution result](docs/research/dual-loop/DUAL_LOOP_D0_EGOMOTION_ERROR_ATTRIBUTION_R3_EXECUTION_RESULT_2026-07-30.md)。

## 2026-07-30：双环正交 shadow-only 工程落地

- 执行者：violjjet
- 在不重跑 D0、不选择科学优先级且不改变默认提醒的边界下，冻结
  [shadow wiring contract](docs/research/dual-loop/DUAL_LOOP_SHADOW_WIRING_R0_CONTRACT_2026-07-30.json)。
- `core:assist` 新增 target/frame/track-epoch/availability/TTL/quality 绑定的
  `DualLoopGeometryEvidence` 与 `DualLoopShadowAdmitter`；缺失、未准入、来源弃权、
  帧/时间/目标/质量异常全部显式 abstain。生产 source allowlist 为空。
- 模式仅有 `OFF` 与 `SHADOW_ABSTAIN_ONLY`，没有 active/actuate。
  `AssistDecisionKernel` 仍是唯一 event/feedback seam；即使 synthetic source
  通过准入，risk、event、feedback、trace 与 gateway call count 都必须保持 baseline
  frame-exact。
- `feature:assist` 只透传无 source 的 shadow observer；`app` 新增隔离
  `dualLoopShadow` build type（独立 application id suffix），默认/debug/release
  flag 均为 false，隔离变体为 true，USTRF flag 为 false。
- Temurin JDK 17.0.19 验证：`:core:assist:test` 146/146、
  `:feature:assist:testDebugUnitTest` 66/66；`:app:testDebugUnitTest` 为
  NO-SOURCE 且成功；`:app:assembleDebug` 与
  `:app:assembleDualLoopShadow` 均成功。
- 工程终点为
  `MECHANISM_SEAM_IMPLEMENTED / DEFAULT_OFF / SHADOW_ABSTAIN_ONLY /
  SYNTHETIC_BASELINE_NONINTERFERENCE_VERIFIED /
  NO_GEOMETRY_SOURCE_ADMITTED / NO_EFFECT_CLAIM`。
  它不证明双环准确、有效、提前提醒、产品改善、安全或独立助行。详见
  [implementation result](docs/research/dual-loop/DUAL_LOOP_SHADOW_WIRING_R0_IMPLEMENTATION_RESULT_2026-07-30.md)。

## 2026-07-30：真实几何双环工程闭环与算法后继诊断

- 执行者：violjjet
- 新增独立的 `DualLoopTargetProvenance.REPLAY_ANNOTATION` 留痕并只接受
  `REPLAY_TIMELINE` 的 `DualLoopJrdbReplayAdapter`；回放 detection 行为源保持
  `OBJECT_DETECTOR`，生产 allowlist 仍为空。host-only JavaExec 先硬校验 producer
  receipt 的 identity、精确 implementation/input SHA 和 outcome firewall，再从
  实际 TSV 重算逐 sequence 的 10,786 行、476 个决策帧与 8,836 个 eligible rows，
  然后对 4 个 JRDB 真实场景调用真实 `AssistDecisionKernel`，其中 474 个
  `ADMITTED_SHADOW`、2 个 `EVIDENCE_ABSENT`、0 adapter abstention，risk/event/
  feedback/explanation/session/gateway call count 逐帧差异为 0。
- JRDB producer 从不可变 packet/ledger 得到 10,786 个 exact joined target rows，
  其中 8,836 个两端真实 LiDAR sensor-supported；全景越界框确定性 clamp 并留痕。
  工程终点为 `ENGINEERING_SHADOW_CYCLE_VALID / DIAGNOSTIC_ONLY /
  NO_EFFECT_CLAIM`。
- Depth Anything V2 Small 在 REveL 512-frame/770 ROI Discovery 上，target depth
  与物理 range 的 Spearman 约 `-0.75`，但 temporal direction 仅 `49.0%` 正确、
  `29.4%` wrong-signed；关闭直接 depth derivative 候选。
- 新增 background homography residual target-flow：truth-blind producer
  `13,014/13,014`，11,381 行通过 0.50 质量门；469-event evaluator 得到
  `233/469` 正确、`91/469` wrong-signed、`452/469` evaluable。相对 LITE R2
  描述性减少反号，但总正确率 `49.7%`、quasi-static `25.0%`，且同一 burned
  capture 的 469 个事件含 159 个跨 target overlap pairs / 310 个 components，
  未做 dependence-aware inference；终点为 `SOURCE_READINESS_NOT_MET /
  INDEPENDENT_INFORMATION_NOT_EVALUATED / DEVELOPMENT_ONLY`。
- 不开放 active mode、不接 Android、不事后搜索 deadband。下一候选只允许在新
  source/session 上组合显式旋转补偿、静态 depth layer 与多帧同号 abstention，
  再进入 harm=0 的独立信息 screen。详见
  [result](docs/research/dual-loop/DUAL_LOOP_REAL_GEOMETRY_SHADOW_CYCLE_R0_RESULT_2026-07-30.md)。

## 2026-07-31：未见自然 rank-2 真值冻结与顺序设备门禁

- 执行者：violjjet
- 按 rank-1 正式关闭后的固定 fallback 顺序启动 Shiraz rank-2；冻结 480p payload、
  1 Hz review bundle 与 4,891 帧 10 Hz replay 输入，baseline/candidate 输出保持
  unopened。
- 两路隔离 AI reviewer 对正负事件集合存在分歧，第三路 fresh adjudicator 读取两份
  哈希绑定 review 后裁决 7 个正例和 6 个负窗；`finalize_rank2_truth.py` 验证
  prompt/input/身份/可见性与最低真值门，终点为 `TRUTH_FROZEN_ADEQUATE`。
- 新增 baseline-only 与 candidate-only Android 入口。baseline 使用 strict QNN HTP
  生成完整 detections/metrics/risk/feedback trace；host 至少观察到 1 个正例命中与
  1 个负窗误触发后才生成 candidate authorization。
- candidate 不重跑 QNN，只重放 baseline 的 byte-equivalent detections/metrics，
  并逐帧要求 raw/stable risk hash 相等；最终 evaluator 预冻结 250 ms 延迟门、
  exact baseline-hit retention、absolute recall 与四类负窗 pairing。
- 两路独立实现复核在提交前拦截了可伪造 authorization 与 baseline/candidate APK
  未绑定问题；修复后 candidate 同时验证 baseline assessment、evaluator SHA、
  rank-2 protocol/source activation SHA 及 app/test APK SHA，host 也从当前 truth
  与 baseline trace 重新计算 adequacy。合法 veto 派生的 `wasAlerted/cooldown/event
  snapshot` 可分叉，但第二环 event mutation permission 必须始终为 false。
- 第二轮复核又要求 truth producer 自身在发布前执行结构门；因此在 baseline 尚未
  打开时，以同一 review/adjudication 生成 canonical `truth-freeze-r2`。r1/r2 ledger
  字节一致，r2 receipt 额外绑定发布前的有限 confidence、视频边界、唯一 ID 与闭区间
  互斥校验；后继只消费 r2。
- 首次 baseline 设备运行本身完成 `4,891/4,891`，但 candidate 前的只读授权复算
  发现 assessment 实际 CRLF 字节与 evaluator 预计算 LF hash 不一致；门禁正确拒绝
  放行，candidate 保持 unopened。根因是 Windows `Path.write_text()` 默认换行转换。
  evaluator 改为显式 `newline="\n"` 并新增字节级回归；旧 baseline/evaluation 标记
  为 serialization-invalid 后保留，更新 evaluator/APK identity 再原样重跑。
- 修复后在同一 4,891 帧输入上完成 strict QNN HTP baseline 与 hash-authorized
  candidate replay。baseline/candidate 正例均为 `7/7`，exact/timely retention
  均为 `1.0`；3 个事件新增 100 ms，其余 0，risk mutation 与 event-mutation
  permission 均为 0。
- 5 个 baseline-false 负窗全部 retained，`corrected=0 / induced=0`；全序列反馈行
  `508 -> 494`。终点为 `FIRST_UNSEEN_SOURCE_NO_EVENT_LEVEL_EFFECT /
  DENSITY_SIGNAL_ONLY`，active R1 默认关闭，不在该来源上增加 latch 或调参。
- producer receipt 的旧字段 `vetoed_feedback_opportunity_count=633` 实为
  `DUAL_LOOP_CONTRADICTED` reason rows。trace 重算的同帧实际 veto 为 89，净反馈
  减少为 14；详细结果明确纠正命名，避免把内部 reason row 当用户少收提醒。
- 明确数据复用规则：“已使用”只取消 exact session 对同一候选的 unseen claim，
  不全局封存数据集。缺原生提醒标签的数据允许在输出盲条件下由多模型复核补齐，
  但 capture/session 独立单位不得用帧或滑窗扩张。

## 2026-07-30：最小因果三态双环来源确认与 Android 影子落地

- 执行者：violjjet
- 方法论纠偏：不再把 ego/target 责任归因、精确 TTC、pose、IMU、depth 或完整三维
  恢复作为基础提醒前置；第二环只输出
  `CONFIRM_APPROACH / CONTRADICT_APPROACH / ABSTAIN`。
- 冻结候选仅用同一 track 连续 7 帧 `log(bbox height)`：6 次相邻变化严格同号且
  OLS slope 绝对值 `>=0.2/s` 才表态，否则弃权。8 个 burned Development 会话先
  复现后，不再改规则。
- 独立 Confirmation 在任何选中 payload 打开前排除 13 个 outcome-open sequence，
  metadata-only hash 冻结 3 个新 JRDB sequence × 360 帧；先取得 2D source 并封存
  43,429 行输出，producer 明确记录 truth 不存在/未打开，之后才取得 3D truth。
- 非弃权 1,017 行中 1,008 正确，总精度 `99.12%`；confirm
  `377/385=97.92%`，contradict `631/632=99.84%`，coverage `2.391%`，
  43 个 distinct tracks，三个 session 均过预声明门。终点为
  `ANNOTATION_TRACK_SOURCE_CONFIRMATION_PASS`。
- `core:assist` 新增 `CausalTrackTristateGeometryProducer`，在
  `DUAL_LOOP_SHADOW` 内使用 production-selected detection、capture timestamp 与
  轻量 track continuity 生成同一三态证据。kernel 显式准入该 source；普通
  admitter 仍 fail closed。
- 新 source 仍完全 observational。Kotlin 回归覆盖增长/缩小/混合趋势、target/gap
  reset 与七帧 kernel admission，并逐帧验证 baseline/shadow 的 risk、event、
  feedback、session 和 gateway 调用完全相同。
- 当前终点：
  `TRISTATE_SOURCE_CONFIRMATION_PASS / END_TO_END_ANDROID_SHADOW_LANDED /
  DEFAULT_OFF / NON_ACTUATING / NO_EFFECT_CLAIM`。JRDB 使用 annotation track，
  Android 使用轻量 detection continuity；下一门只收集真机 live parity、三态分布、
  reset 与延迟，不增加算法复杂度。详见
  [result](docs/research/dual-loop/DUAL_LOOP_CAUSAL_TRACK_TRISTATE_R0_RESULT_2026-07-30.md)。
- SM-S9280 真机安装独立 `com.linnan.blindassist.dualloop.shadow` 后冷启动与
  CameraX smoke 成功；连续 frame ID 进入 `BlindAssistDualLoop`，短观测约 24 FPS。
  当时镜头 `count=0`，故全部 `EVIDENCE_ABSENT`，没有伪造 live 非弃权样本。
  隔离包因 `libcdsprpc.so` 不可见回退 `cpu_xnnpack`，不形成 NPU/性能结论；smoke
  后 force-stop 隔离进程，APK 保留安装，正式包和数据未动。

## 2026-07-30：多目标连续性反事实筛选入口

- 执行者：violjjet
- 现有 production-selected 单目标轨迹在冻结的 CrowdBot 基线中，27 个负窗触发帧
  只有 1 个 `CONFIRM`、0 个 `CONTRADICT`；若直接启用反证抑制，收益为零，若强制
  `CONFIRM` 则会破坏正例召回。因此不开放 active mode。
- 新增 Development-only 真机完整 detection dump 入口，复用既有 4,422 帧与同一
  strict QNN HTP detector，但写入独立命名空间，既不读取 truth，也不覆盖已完成的
  production temporal A/B 正式输出。
- 冻结
  [多目标连续性反事实 R0 协议](docs/research/dual-loop/DUAL_LOOP_MULTITRACK_COUNTERFACTUAL_R0_PROTOCOL_2026-07-30.json)：
  只有 `CONTRADICT_APPROACH` 可提议抑制已有提醒；必须至少消除 7 个负窗中的 2 个，
  同时保留 8/8 正例且单事件延迟不超过一帧，否则拒绝该 active 路线并换路。
- 当前只完成 outcome-blind 数据出口；尚未运行真机 dump 或 truth join，不形成效果
  结论，也不改变默认、风险、事件或反馈行为。

## 2026-07-31：YOLO + 语义分割图像空间互补性跨来源 Development 诊断

- 执行者：violjjet
- 用户明确授权后，沿用已冻结的 image-space estimand，不重开中央阻塞 Agent 标签、
  不增加第三 Agent/提示词/slot readiness 层。先实现固定 YOLO11n host trace adapter：
  同一 `yolo11n_fp16_320.tflite`、80 类标签、320 输入、`0.35` confidence、
  `0.45` class-wise NMS；trace 明确标记为 `DEVELOPMENT_HOST_REFERENCE_ONLY`，不冒充
  QNN/device parity。
- Shiraz 与 Shanghai 两个 RGB source 分别完成 `4,891/4,891` 与 `5,662/5,662`
  host detector frame，模型 SHA 为
  `00edb41a528b0a7e709c4af8ce3e685491492c4539274804e5cfc17a1a867cd2`；随后使用同一
  segmentation reference 运行 A/B/C image-space diagnostic。两个 source 的独立
  validator 均返回 `VALID`，`NOT_EVALUABLE=0`，风险/反馈/事件字段均未读入。
- 同 host backend 下，两个 source 都观察到非零的 class-wise YOLO-uncovered mask；
  `walkable`/`unknown_nonwalkable` 相邻 IoU 相对较高，`boundary_step_curb`/
  `obstacle` 稳定性偏低。YOLO coverage 与 obstacle uncovered magnitude 随 source
  改变，不能包装成来源不变的障碍增量。
- 终点为
  `CROSS_SOURCE_IMAGE_SPACE_SIGNAL_REPLICATED / CLASS_STABILITY_MIXED_AND_SOURCE_DEPENDENT /
  NO_OBJECTIVE_OBSTACLE_TRUTH / NO_FUSION_EFFECT_AUTHORITY`。保留 host/QNN 差异说明；
  不进入 Android、主动提醒、风险真值或生产路径。详见
  [cross-source result](docs/research/dual-loop/DUAL_LOOP_SEGMENTATION_COMPLEMENTARITY_R2_CROSS_SOURCE_RESULT_2026-07-31.md)。

## 2026-08-01：HFTF F0.1 SANPO body/head teacher opportunity

- 执行者：violjjet
- 在首个 F0.1 teacher geometry outcome 前，先把 same-train-split heldout 加强为
  official-test heldout，固定 6 train / 3 dev / 3 heldout parent sessions；完成
  300 RGB、300 masks、300 metric depths 的统一 transport/hash/split 审计，以及
  12/12 source-specific pose/canonical-transform/local-ground proxy authority。
- outcome 前另行冻结 future observation union：obstacle support 逐 cell 取两帧
  最大值，known 对同一 world probes 做跨观测 OR 后应用 5/9；future origin 只由
  history-to-anchor tangent velocity 推进，future pose 不得选择 origin、方向、
  anchor 或 sample。
- 专用 opportunity runner 在 12/12 source 上通过全部门。最弱 known coverage
  约 0.23（门 0.10），最弱 future positive-known=6（门 5），最弱 future
  negative-known=182（门 20）；每个 role 的 body/head positive-source count
  分别为 train 6、dev 3、heldout 3。
- 两次独立进程完整报告 SHA-256 均为
  `9db97892ae93267856e1388bccf808deb8947311e25cc5b39a1c362b4bb348b5`，
  且每个进程内部两遍 canonical payload byte exact。HFTF 单元测试为 151/151。
- 终态为 `F0_1_SANPO_TEACHER_OPPORTUNITY_READY_FOR_CORPUS`。只开放 train
  candidate corpus 与 dev reference targets 的下一步物化；official-test heldout
  targets 继续封闭到 checkpoint 冻结后，heldout training corpus 永不授权。
- 该结果只是 synthetic body/head geometry-proxy opportunity，不是 student effect、
  完整 HFTF、人体/事件/安全证据；研究主线、默认 App、Android 与生产均不变。详见
  [result](docs/research/hftf/HFTF_STAGE_C_SANPO_TEACHER_OPPORTUNITY_RESULT_F0_1_2026-08-01.md)。

## 2026-08-01：HFTF F0.1 train/dev corpus 与学生训练执行冻结

- 执行者：violjjet
- 按 outcome 前冻结的 corpus contract 物化 90 个 train candidate 与 39 个 dev
  reference student records；official-test heldout 为 0。student record 只含精确
  五帧历史 RGB、哈希和 nullable current/future labels，future teacher modalities
  保持在隔离 receipt 中。
- 两次独立物化的 student samples、teacher receipts 和 dataset spec 分别 byte exact；
  独立磁盘 validator 复核 129/129 records、129/129 receipts 与 645/645 历史 RGB
  引用，终态为 `F0_1_SANPO_TRAIN_DEV_CORPUS_VALIDATED`。
- 首个学生优化步前冻结 3 arms × 3 seeds 的 MobileNetV3-Small 时序学生训练合同：
  30 epochs、固定增强/损失/优化器、dev known-cell micro risk F1 选择且阈值固定
  0.5，九个 checkpoint 冻结前不得物化 heldout target。
- 新训练器逐臂在 CUDA/模型初始化前校验所有实际使用 RGB 的当前文件 SHA-256，
  绑定 torch/torchvision、预训练权重、语料、执行合同及所有父协议哈希；单元测试
  9/9、HFTF 全集 168/168。此提交只冻结可执行实现，尚未产生任何训练或 heldout
  outcome；主线、默认 App、Android、生产与安全主张均不变。

## 2026-08-01：HFTF F0.1 九个 train/dev checkpoint 完成

- 执行者：violjjet
- 严格按 seed-major 顺序完成 seeds 17/29/43 ×
  `SF_CURRENT / SF_FUTURE / HIST_FUTURE` 九个 run；全部 30/30 epochs、参数量
  1,022,448、loss/gradient/parameter 有限，且各自产生唯一 checkpoint SHA-256。
- checkpoint 只由固定阈值 0.5 下的 dev teacher-known cell micro risk F1 选择；未读取
  heldout RGB、teacher target 或 student output。三个 seed 的
  `HIST_FUTURE - SF_FUTURE` dev F1 差分别为
  `-0.007279 / -0.017149 / -0.010633`，一致为负，但 dev 仅选择 checkpoint，
  不能据此调参、停掉 frozen heldout 或作 student effect 结论。
- 新增独立九产物 validator：重算每个 earliest-best epoch，核验目录全集、报告与
  checkpoint 哈希、父协议/实现哈希、模型 strict load、优化器、有限性、参数量一致
  和 heldout 防火墙。定向测试 6/6、HFTF 全集 174/174；正式 validator outcome
  将在实现提交后运行。
- 当前仍不授权 heldout target 物化或推理；九 checkpoint gate 通过后只允许先冻结
  heldout 执行合同。主线、默认 App、Android、生产与安全主张均不变。

## 2026-08-01：HFTF F0.1 九 checkpoint gate 与 heldout 合同冻结

- 执行者：violjjet
- 已提交的独立 validator 对九个 run 复核 exact 目录集合、30 epochs、
  earliest-best dev checkpoint、唯一 SHA、1,022,448 参数、模型/optimizer
  keys/shapes/dtypes/完整 state、所有 finite 值、class weights、父哈希、实现 receipt
  和 exact heldout firewall，正式终态为 `F0_1_SANPO_NINE_CHECKPOINTS_FROZEN`；
  validation SHA-256 为
  `5a3c73968213d046f7e48ba82e356f08d46468bc90798be077600152e1e8d824`。
- 在任何 heldout target 物化或 student forward 前冻结 official-test 一次性执行合同：
  3 个固定 parent sessions × 13 reference anchors、九 checkpoint seed-major、
  threshold 0.5、39 条 inference inputs 与隔离 truth/teacher receipts、351 条
  truth-free predictions 先冻结 SHA 后才允许 join。
- effect gates 原样继承 F0：median micro-F1 delta 至少 0.03、每 seed 为正、recall/FPR、
  body/head、worst-source 和 `SF_CURRENT` median-seed learnability 0.6 全部门同时通过；
  任一失败即 no-gain stop，不允许 after-outcome rescue。
- 合同把 materializer、package validator、prediction-only runner、truth join 与独立
  terminal validator 的最终实现 SHA-256 全部 byte-bound；machine contract SHA-256
  为 `cdc05f52f3d10ce8479025a0a0137f6d8c8a4d5d6faf320245dd0295c3b39462`。
  HFTF 全域新增 LF checkout 约束，避免 Windows `autocrlf` 使实现 receipt 漂移。
- prediction 在首次 forward 前写固定全局 consumption ledger；truth join 与 terminal
  validation 分别在首次开 truth 前独占创建 canonical root、原子写入并 fsync receipt。
  任一异常都原子持久化 `NOT_EVALUABLE` 且不得重试。两种 truth 进程不导入 predictor、
  `torch` 或模型代码，并独立核验 351 条输出的 ordered join-key SHA。
- 定向 heldout execution 测试 10/10、HFTF 全集 184/184，五个 implementation
  receipt 均与磁盘字节 exact match；未物化 heldout package、未创建 consumption
  ledger、未运行 student heldout forward。
- 当前仅合同冻结；heldout materialization、student output 与 effect terminal 尚未产生。
  即使未来 signal-supported，也只形成 synthetic geometry-proxy 支线证据，不直接授权
  替换主线、Android、生产或安全主张。

## 2026-08-01：HFTF F0.1 official-test heldout 负终态

- 执行者：violjjet
- 按冻结顺序一次性物化并验证 3 个 official-test parent sessions × 13 anchors：
  package validation 为 `F0_1_SANPO_HELDOUT_PACKAGE_VALIDATED`，SHA-256
  `864504876a28ed16bce6a6f2a9ac525b61d84af3103e9298db8a6309c8b54a8e`。
- 唯一 prediction-only 进程在固定 RTX 5060 Laptop GPU、torch 2.11.0+cu128 /
  torchvision 0.26.0+cu128 上完成 9 × 39 = 351 条输出；predictions SHA-256 为
  `1a62a45412caf9582fb6d92fc037c84f8e3cef78069c200d32575e8eb83c3d1e`。
  全局 ledger 已永久消费，任何第二次 model forward 均不授权。
- truth join 得到
  `F0_1_SANPO_CROSS_SPLIT_BODY_HEAD_TEMPORAL_STUDENT_SIGNAL_NOT_SUPPORTED_STOP`；
  独立 validator 从冻结文件完整复算后返回
  `F0_1_SANPO_HELDOUT_EFFECT_TERMINAL_VALIDATED`，validation SHA-256 为
  `32d9d956cd162644696d96ed4476719bfa49e0f4156b41f6d7b66a5f5029bb33`。
- temporal micro-F1 delta 按 seed 17/29/43 为
  `-0.007233 / +0.015577 / -0.025393`，median `-0.007233`；head median delta
  `-0.008473`。更强的 blocker 是 `SF_CURRENT` median-seed F1 仅 `0.173267`
  （门 `0.6`），说明直接 RGB→geometry-proxy risk 的跨 split learnability 本身不足。
- F0.1 永久关闭，不允许换 checkpoint、阈值、来源、指标、gate 或 rerun rescue。
  三个 official-test sessions 已 burned，不能充当 successor 的 fresh validation。
  研究主线、默认 App、Android、生产与安全主张均未改变。
- 该负结果不证明所有 temporal factorization 都无效。若继续支线，只能新立机制不同、
  outcome-before 冻结且使用 fresh parent sources 的 successor；其首要 falsifier
  应先验证可跨来源迁移的物理中间表示能否解决 current learnability，再检验显式
  causal transport 的 future 增量。

## 2026-08-01：HFTF-G0 support-equivalent clearance 机制合同冻结

- 执行者：violjjet
- F0.1 负终态保持永久关闭。新 successor 不换 temporal backbone，而先检验
  `current RGB -> continuous clearance proxy -> frozen envelope risk` 是否能解决
  直接 binary-risk student 的跨来源 learnability blocker；G0-D0 只审计 consumed
  synthetic geometry-proxy mechanics，不计算 student output，也不给 fresh evidence。
- 半开 prism 不再误称真正连续 SDF。实现逐项复用原 teacher 的 reference
  stride-4/offset-2 obstacle point set、anchor basis、semantic filter、
  `searchsorted(side=right)`、末端 `8 m isclose(atol=1e-12, rtol=0)` 与 height
  membership；closed-box SDF 只提供距离大小，membership 强制符号，精确零值用
  float64 `nextafter` 打破平局。第二小 proxy 严格 `<0` 与 support count `>=2`
  完全等价；少于两个点先为 `+inf`，再 clip 至 `+1 m`。
- D0 对全部 12 个已 consumed F0.1 parent sessions × 25 current frames，逐
  `source × height` 要求二值等价、UNKNOWN null/never-safe、正负 known、
  clipped 毫米 bins、近边界数量及 risk/safe 非全饱和；支持终态仅为
  `G0_SIGNED_CLEARANCE_MECHANICS_SUPPORTED_FOR_FRESH_LEARNABILITY_CANARY`，不声称
  student 已可学。
- 来源规划器闭合验证 11 个 parent 的 path/hash/status/terminal，以及 F0/F0.1
  metadata firewalls、acquisition、authority、teacher-opportunity、effect burn 和
  历史 burn。9 个 outcome-open 来源内部固定为 6 train + 3 model selection；F0
  ranks 10–12 的三个来源仅可在模型、loss、checkpoint、threshold 和 gates 全部冻结后
  做 one-shot fresh evaluation。另从 official-test 仅 metadata 预留三个 future
  heldout；本合同不授权获取或打开任一 fresh outcome。
- source planner SHA-256 为
  `ee5e84accf8a58370faf3d1813a8a0170f0331ed3a3c3914ce77145ca98ac244`；
  mechanics runner SHA-256 为
  `0a5f39bd71ab6a28a214cd30f8e15262288f4b17e916fd302c84289902aa9d38`；
  machine contract SHA-256 为
  `0aa8e5828665a869837a1aa9027601d45610c0f66696da737351c9ec361da383`。
  定向测试 18/18、HFTF 全集 202/202。canonical source-plan 与 mechanics result
  roots 在冻结时均不存在，尚未运行 metadata scan 或 D0 outcome。
- 只有 source plan READY 且 D0 支持，才允许另行冻结 fresh-evaluation acquisition
  与 D1 训练/one-shot 合同。主线、默认 App、Android、生产与安全主张均不变。

## 2026-08-01：HFTF-G0-D0 mechanics 支持并独立验证

- 执行者：violjjet
- metadata-only source planner 正式返回 `G0_SIGNED_CLEARANCE_SOURCE_PLAN_READY`，
  固定 9 development reuse、3 one-shot fresh evaluation 和 3 个仅预留的
  official-test future heldout；source-plan SHA-256 为
  `886271cd1546e2f3f4cd91991f39725ed39b12907e0d4294b980404d132648a4`。
  新 RGB/depth/mask/pose、geometry teacher 与 student outcome 均未打开。
- D0 在全部 12 个已 consumed sources × 25 current frames 上返回
  `G0_SIGNED_CLEARANCE_MECHANICS_SUPPORTED_FOR_FRESH_LEARNABILITY_CANARY`；
  result SHA-256 为
  `050670764e15a8b9059dc893edb71534d6112ab8931a4fb118668653f8b577bf`。
- 24 个 `source × height` 单元中，positive/negative known 最小值为 `5/148`，
  clipped 1 mm bins 最小 `55`，近边界 known 最小 `10`；最大 risk-min/safe-max
  clip saturation 为 `0/0.888889`，且每类每单元均有非饱和 target。binary
  equivalence、unknown-nonnull 与 unknown→safe 违规全为 `0`。
- 独立 validator 在实现先提交并推送后正式重算 source roles、firewalls、全部 D0
  gates 与终态，返回
  `G0_SIGNED_CLEARANCE_SOURCE_AND_MECHANICS_TERMINAL_VALIDATED`；validation SHA-256
  为 `4659e1fbb7938a637c157c6ceaad1186bc2b9ec919951fca6cb252b61acacd62`。
- D0 只授权冻结 D1 current learnability 合同，不证明 RGB student learnability，
  不授权打开 reserved heldout、future/temporal 实验、主线、App、Android、生产或
  安全主张。

## 2026-08-01：HFTF-G0-D1 current learnability scientific design 冻结

- 执行者：violjjet
- D1 只比较同一 F0.1 SF_CURRENT MobileNetV3/输入/temporal fusion/known head 下的
  `DIRECT_RISK_CURRENT` 与 `SIGNED_CLEARANCE_CURRENT`；clearance arm 唯一机制变化
  是每 cell 输出无 activation、无 clamp 的线性 meter value，并以 `<0 m` 导出 risk。
- D0 显示 6 train 到 3 model-selection sources 的 body/head positive 比例从
  `24.17%/12.21%` 降至 `6.12%/4.33%`，safe `+1 m` saturation 从
  `46.47%/54.54%` 升至 `68.61%/76.52%`。因此禁用 pooled-MAE 选择与 bounded
  activation；clearance loss 固定为 risk/safe + near-boundary 加权 SmoothL1
  `beta=.1 m`、`0.1 ×` fixed-temperature sign BCE、`0.25 ×` known BCE。
- Phase A 用 6 train 完成 30 epochs，旧 3 selection 仅按 source-macro F1、
  worst-source F1、micro F1、严格 tie-break 选 epoch；Phase B reset 后用全部 9
  outcome-open sources 仍完成 30 epochs，只冻结预选 epoch。六 checkpoint 与
  prediction contract 冻结前，不得获取三条 fixed fresh sessions。
- Fresh 机会门逐 `source × height` 固定为 25 frames、coverage `>=.1`、
  positive/negative `>=5/20`、UNKNOWN→SAFE `0`；不足即
  `NOT_EVALUABLE` 且不得换 source。机会充分后 prediction-only 先冻结并消费输出，
  truth join 只执行一次。
- 预声明 effect/MAE/firewall 全部门；raw prediction out-of-range 仅强制报告、
  不用 clamp 隐藏，但不另设 outcome 后的新 gate。任何预声明门失败都永久停止同
  cohort rescue。
  即使支持，也只允许另冻 causal-transport 合同，不打开 reserved official-test，
  不改变主线、App、Android、生产或安全权限。
- 当前只冻结 scientific design；implementations、corpus、训练与 fresh acquisition
  均未授权。

## 2026-08-01：HFTF-G0-D1 Development execution contract 冻结

- 执行者：violjjet
- 执行前独立复核发现原 scientific design 的
  `ALL_25_CURRENT_10FPS_FRAMES` 与冻结 source plan 不完全一致：9 个 Development
  来源中 7 个 target FPS 为 10、2 个为 5。由于 D1 corpus/student outcome 尚未
  打开，新增 metadata-only timeline amendment，将语义修订为
  `ALL_25_CURRENT_FRAMES_AT_EACH_SOURCE_PLAN_FROZEN_TARGET_FPS`；source identity、
  model、target、loss、selection 与 gates 均不改变。amendment SHA-256 为
  `3029dc7622cefa5491415e619f23bd7e0f080bef4c9b09407b019ac66938030c`。
- Development corpus 固定为 6 train + 3 model-selection sources、每源 25 个
  source-plan current frames。student loader 只可打开 current RGB；UNKNOWN 的
  risk/clearance 保持 null。独立 validator 重新绑定真实 manifest、RGB/depth/mask、
  pose/authority receipts，并从 authority inputs 重推全部 labels，拒绝 self-consistent
  forgery、fresh/reserved masquerade、role overlap 与任意额外 teacher/future 字段。
- 训练分 Phase A/B：两个 arms × 三 seeds 均完整训练 30 epochs；Phase A 用 6/3
  split 冻结 epoch，Phase B reset 后用全部 9 个来源且只冻结对应 epoch。Phase B
  开始前重新计算 Phase A selection 并 strict-load finite checkpoint；最终独立
  validator 要求完整 12-run tree 与六个 final checkpoint hashes。
- corpus materializer / validator / trainer / training validator SHA-256 分别为
  `da0523fe7a01064540b788d9e92f889c0a7e331ae6e71ba5683023c96a70c153`、
  `bdfb8eb15cee7232d681e96c30e4b3186331ddec4e68d5226f2b311ca743e39c`、
  `d0d668b509015f5c18e6e40f5cd4ccac17f1523ac8744c5f6c78e60c287ec716`、
  `68713284875550ee7c31d335ae6025333b21571d4092937bcd62b0b2da4749b5`。
  execution contract SHA-256 为
  `fa7cce1e634d535bb8ff57a658befc8daa17d15e0d482580140a1675d9d88df7`。
- 两路独立复核最终均为 CLEAR；定向测试 35/35、HFTF 全集 242/242。
  独立 authority/label 重推预检成功绑定 225 records / 9 sessions，耗时约
  `147.5 s`，未写正式 corpus。
- 本节点只冻结并授权提交后执行 Development corpus → validation → Phase A →
  Phase B → training validation。fresh acquisition/prediction/truth、reserved
  official-test、future/temporal、研究主线、默认 App、Android、生产与安全权限均
  继续关闭。

## 2026-08-01：HFTF-G0-D1 六个 Development checkpoints 冻结

- 执行者：violjjet
- 正式 corpus materializer 返回
  `G0_D1_CURRENT_CLEARANCE_DEVELOPMENT_CORPUS_READY`；9 来源、225 student records
  与 225 teacher receipts 精确一一对应。student/teacher SHA-256 分别为
  `d707613109878ed11e573429e39124b819264b3939a7989e3f22189379c7372f` /
  `86d04cc10f7f30b151e9eac508c5e4b7708bdbfaadbc343fa49d4fcd37b11f89`。
- 独立 corpus validator 重新读取 manifest、RGB/depth/mask、pose/authority 并重推
  全部 labels，返回 `G0_D1_DEVELOPMENT_CORPUS_VALIDATED`；11 项 checks 全 true，
  UNKNOWN→SAFE 为 0，validation SHA-256 为
  `d20b6afa10625ef5edbfb7823be2aaa32a0ef1847ce43ae9e3531c0071f8eb0b`。
- 12 个冻结 runs 全部完成 30 epochs。Phase A 选择的 direct epochs 为
  `24/22/21`，clearance epochs 为 `13/11/20`（seeds `17/29/43`）；Phase B
  reset 后在全部 9 个 Development 来源上冻结同一 epochs。
- 独立 training validator 重算 Phase A selection、核对 Phase B parents、相同 seed
  的 initial arrays/loss 参数，并 strict-load/finite-check 全部 checkpoints，返回
  `G0_D1_SIX_FINAL_CHECKPOINTS_FROZEN`。六个 final checkpoint SHA-256 为
  `c6256d5d…63cf3 / b5e9dbe4…4eed2 / 73514643…0f560 /
  248b9a32…2e415 / ce65905d…b6323 / d252f96f…320a`；training validation
  SHA-256 为
  `b1ed88a7f7a889035b2e47b5e4d0f38349505b1349ab16d6bdf3b44f52e62156`。
- 该终态只授权冻结 one-shot fresh execution contract；fresh acquisition、
  prediction 与 truth 均未执行。reserved official-test、future/temporal、主线、
  App、Android、生产与安全权限继续关闭。

## 2026-08-01：HFTF-G0-D1 one-shot fresh 执行合同冻结

- 执行者：violjjet
- 在三条预声明 fresh SANPO-Synthetic source 的任何本地媒体、geometry teacher
  outcome 或 student prediction 打开前，冻结
  `HFTF_STAGE_C_CURRENT_CLEARANCE_FRESH_EXECUTION_CONTRACT_D1_2026-08-01`。
  来源固定为 G0 source-plan ranks 10–12、每源 `0,2,...,48` 共 25 个 current
  frames；六个 Phase-B checkpoint hashes、三 seed、两个 arms、450 条 prediction
  的精确顺序和全部 D1 gates 均不可更改。
- source 未打开时不存在可诚实预填的本地 manifest/spec/pose/media/authority
  hashes，因此选择权威由既有 hash-bound G0 source plan 的 session/frame/remote
  metadata identity 固定；acquirer 与 authority verifier 实现同时 hash-freeze。
  打开后产生的本地 hashes 只可作为传输与权威收据封存，不得改变来源、checkpoint、
  阈值、gate 或执行顺序。
- package validator 原子发布完整 truth-aware validation 和独立 truth-free
  prediction authorization。predictor 对授权对象使用精确字段白名单，禁止间接读取
  truth path/hash、teacher receipts 或 opportunity counts。全局 predictions
  completion 落盘后，evaluator 必须先耐久写 truth-consumption receipt，再只读取
  truth bytes 一次并同时核验 hash 与解析内容。
- package materializer 也必须在首次读取本地 fresh package source/media 前写入
  独立 execution receipt；任一后续异常会写 consumed failure，禁止第二次
  materialization 或换源。package validator 必须绑定该 receipt、completion 与
  package manifest。
- opportunity 要求每 source×height cell 的 known coverage `>=0.10`、positive
  `>=5`、negative `>=20`、UNKNOWN→SAFE `=0`；不足即 `NOT_EVALUABLE` 且不换源。
  clearance MAE 按六个 source×height 等权 macro，并要求每个 seed 均通过，即使用
  三 seed 最大 MAE 对 frozen 阈值；这项解释在 fresh outcome 前冻结。
- 定向 fresh 实现测试当前 39/39、HFTF 全集 281/281 通过；合同父证据与 12 个
  实现 receipts 均逐项 hash 复核通过；fresh execution contract SHA-256 为
  `b13b27d0fd882ec7a9904c6e2dd595629e0b3ca093f9e238549e32fc3f655ae2`。
  此节点只授权提交推送后按固定顺序执行 source acquisition →
  authority → package/opportunity → truth-blind prediction → single truth join →
  independent terminal validation。reserved official-test、future/temporal、
  主线、App、Android、生产与安全权限继续关闭。

## 2026-08-01：HFTF-G0-D1 fresh 执行不可评价关闭

- 执行者：violjjet
- fresh 合同已由 commit `ab9a6cc5257bf20477a097d5aec6fe9cf2703874`
  推送并确认 `HEAD == origin/master` 后，才按 source order 启动第一次正式
  acquisition。第一个固定 session 为 `15bc9dde…e02bf`，只执行一次 CLI；脚本的
  frame-0 depth 下载内部三次 retry 均失败，最终返回 `ok=false`。
- 失败目标 `.float16.gz.tmp` 路径长度为 263 字符；同一输出根内较短的 metadata、
  frame-0 RGB 与 mask 已成功写入。该组合支持
  `WINDOWS_PATH_LENGTH_TRANSPORT_FAILURE` 强推断，但没有完整 dataset manifest，
  也没有 authority、teacher opportunity、student prediction 或 truth join。
  stdout SHA-256 为
  `4b738c7cfd9e81058d7021210a49d1ad7a69db1099522182140f3eb9564cc7ee`。
- 因 fresh 内容已经打开，严格执行预先冻结的 no-retry/no-rematerialization/
  no-source-replacement 条款：不改短路径重跑，不补全 partial root，不打开剩余两条
  D1 fresh source，不碰 reserved official-test。终态为
  `G0_D1_FRESH_EVALUATION_NOT_EVALUABLE_NO_SOURCE_REPLACEMENT`。
- tracked fresh execution result JSON SHA-256 为
  `8fae114c9208a823fc305c19b2776f0fd29f4e51b4d84422621b0400fafc477e`。
- 该终态既不支持也不拒绝 signed-clearance，不产生模型负证据，不改变主线、App、
  Android、生产或安全权限。partial 文件与 acquisition logs 保留为 consumed
  failure evidence，不删除、不恢复 fresh 身份。任何 HFTF successor 必须是新问题、
  新合同和新数据角色边界，并在打开新 fresh source 前先通过 synthetic path-length
  transport canary；不得写成 D1 同 cohort 的路径修补或救援。

## 2026-08-01：HFTF successor D2 科学边界与 T0 短路径合同冻结

- 执行者：violjjet
- D2 明确不救援 G0-D1。新问题是：仅用 history pose 的恒速 causal SE(2)，运输已经
  冻结的 current G0 signed-clearance point-field 到 `+.4/+.8 s`，能否胜过
  current-field persistence。它不使用 D1 checkpoints/loss/predictions 或完整三条
  fresh cohort，也不补 partial root、换源或解释 D1 传输失败为模型证据。
- D2 在任何新 source outcome 前冻结：全部 source 统一 5 Hz/13 frames，7 个 anchors，
  history `t-.4,t`，future offsets `+2/+4`；平移与 yaw-rate 都只由历史 pose 决定。
  current-only preprocessor 在 future truth 前为 persistence current-grid 与 advected
  predicted-grid 封存 exact G0 obstacle points、9-probe counts 与 known masks；两臂
  输出 `{known, clearance_m}`，UNKNOWN 必须 null。truth、common-known、24 个
  opportunity strata、MAE/F1 macro、zero-denominator、`1e-12` tolerance 与全部 effect
  gates 均机器可判。D2 JSON SHA-256 为
  `06a8ff9cbe9c4c9b98cceeb7a36c69ba098f6f7d53ab980adb747b987a1728d9`；
  独立科学复核最终 CLEAR。
- 既有计划内可直接用于 D2 的新 official-train parent 为 0；这不证明池耗尽。后续
  必须另冻 metadata-only scan，排除所有 burned/consumed、完整 D1 cohort 与
  official-test reservation，按绑定 split 顺序锁定 6 条全新 parent；不足即
  `STOP_NO_ELIGIBLE_NEW_DEVELOPMENT_COHORT`。本节点尚不授权 scan 或新媒体打开。
- 新 T0 acquirer 使用短 token root 和 `00..18` timeline aliases，在任何 GCS 请求前
  枚举 final/staging/downloader `.tmp` 的全部 340 条路径并要求 `<240`；每个对象必须
  generation/size/MD5 完整。network acquire 必须绑定 exact tracked contract、自身
  hash、outcome-open Development source、canonical consumed package 与 G0 source
  plan；任意 train/test/source/root/config 漂移均 fail closed。
- source-blind filesystem canary 用 537 字符 synthetic identity 得到最大路径 174；
  exact source/root preflight 最大 150。两者均未联网、未打开 source；当前实现复跑
  preflight 与原 evidence byte-exact。离线 equivalence validator 将逐帧核对 canonical
  与 candidate 的 remote identities、本地 SHA/MD5、metadata、transport receipt 和
  实际/`.tmp` 路径。candidate manifest/spec hash 只能是 post-open receipt，合同拒绝
  预填。
- T0 exact source 固定为已消费 Development session `12b65d2c…c93bb`；canonical
  manifest/spec SHA-256 为 `476b4e5f…9bdc8 / 04d0bc12…d38b3`。T0 contract
  SHA-256 为
  `bcf38a45b3d48cb8b82ed9ecd833de9db3ba25f8007ea4f5710b5d08e44152c6`。
  两个失败出口统一为
  `T0_SANPO_SHORT_PATH_TRANSPORT_NOT_EVALUABLE_NO_SOURCE_REPLACEMENT`，不重跑、
  不补 partial、不换源；成功终态为
  `T0_SANPO_SHORT_PATH_CONSUMED_PACKAGE_EQUIVALENT`。
- 定向 T0 测试 17/17、HFTF 全集 298/298 通过；独立 T0 审计重算 12 项
  parent/implementation/test/evidence/canonical hashes 全匹配并最终 CLEAR。本节点只
  授权合同提交推送并确认 `HEAD == origin/master` 后，执行一次 consumed Development
  acquisition → offline equivalence。fresh/reserved、D2 scan/media/mechanics、
  teacher/student、主线、App、Android、生产与安全权限继续关闭。

## 2026-08-01：HFTF T0 consumed-Development 短路径等价通过

- 执行者：violjjet
- T0 contract 与实现由 commit `f38dd5c2bec75e307d6d5a1cf9c314f171710f72`
  提交推送、确认 `HEAD == origin/master` 后，只执行一次合同固定的 already
  outcome-open Development source `12b65d2c…c93bb`。25 个 selected frames 的
  RGB/mask/depth 与 5 个 metadata/split objects 全部通过 generation、size、MD5；
  acquisition terminal 为 `T0_SANPO_SHORT_PATH_TRANSPORT_READY`，report SHA-256
  为 `a69e68f5362fef34bce10daa0932682ddd150a850b575bdc78dd451196d8aa27`。
- short-path package 共 85 files，manifest/spec SHA-256 为
  `a47ffe44…273d / b39f494d…06e7`。离线 validator 重算 25 rows、75 个本地
  SHA+MD5/size、75 个 remote generation/size/MD5 identities、5 个 metadata/split
  identities，全部 7 gates 为 true；final 与模拟 `.tmp` 最大路径为 146/150，无
  `.tmp` 残留，`network_opened=false`。
- 终态为 `T0_SANPO_SHORT_PATH_CONSUMED_PACKAGE_EQUIVALENT`，equivalence report
  SHA-256 为
  `9f4fb76b6637027e92ecad62c5b52792f2aeb08d63bcc445e4cfdbbd9238cc28`。
  tracked result JSON SHA-256 为
  `82c5bed9dc9210dadd615c36176174ad1043ed4860c54b04941f78083075ac7b`；
  独立结果审计重算全部声明 receipts、package contents 与权限边界后 CLEAR。
- 本结果只授权冻结 D2 metadata qualification implementation contract。D1 关闭终态
  不变；这不是 D1 补跑、换源或模型证据。D2 scan/new media/mechanics、fresh/reserved、
  teacher/student、主线、App、Android、生产与安全权限继续关闭。

## 2026-08-02：HFTF D2 metadata-only qualification 执行合同冻结

- 执行者：violjjet
- 新 planner 沿 D2 design → G0 source plan/protocol → F0 inventory 与
  F0/R4/R3.1 burn ledgers → F0.1 consumed official-test result 的 hash-bound 父链，
  构造 78 个互斥 parent exclusion union。它只允许 generation/SHA 绑定的
  official-train split、candidate description JSON 以及 description/pose/media
  对象 receipts/listings；RGB、mask、depth bytes 与 pose 内容均不读取。
- 选择规则在 outcome 前固定为 official-train `session_id` 升序的前 6 个
  metadata-eligible 新 parents。candidate 经 3 次内部 retry 后仍 404 或 metadata
  不合法，记为 ineligible 并继续固定顺序；完整 split 不足 6 个即
  `STOP_NO_ELIGIBLE_NEW_DEVELOPMENT_COHORT`。扫描完成后不重跑、不追加、不替换。
- 主审补严授权 blocker：network CLI 不接受 caller-supplied 裸 implementation hash，
  只能接受正式 execution contract；在首个网络请求前必须验证合同与 planner 均为
  tracked、staged/unstaged clean，并确认 `HEAD == origin/master`。独立预执行审计
  发现后，进一步把 planner test 纳入同一门禁、把 retries 锁死为合同值 3，并要求在
  首个网络请求前写入不可覆盖的 durable attempt marker。planner SHA-256 为
  `4d8b206c887352d92c15cb3fe375d357551861c5e0a6113073a7426f332da58a`；
  targeted tests 14/14、HFTF 全集 312/312 通过；第二轮独立只读复审重算 12 项
  parents/derived/implementation/test receipts 并核对首网前门禁后最终 CLEAR。
- execution contract SHA-256 为
  `b9cb978027cb9f7d02b88753b43fbb9511a4e0766a11c08821842dd7e2c0a085`。
  本节点只授权在合同与实现提交推送、再次确认远端一致后执行一次 metadata scan。
  media/pose content、teacher、student、D2 mechanics、reserved official-test、研究
  主线、App、Android、生产与安全权限继续关闭。

## 2026-08-02：HFTF D2 metadata-only cohort 锁定通过

- 执行者：violjjet
- exact metadata contract 由 commit `335eb2630b3debac07cea9c38448f0b1cb3a8f3d`
  提交推送并确认 `HEAD == origin/master` 后，只启动一次 CLI。首网前 durable
  attempt SHA-256 为
  `b3547bc02c2f1a8e4633596681200ccc652a8cef0fe872ad4f0f8b5cafac0dc7`。
  外层命令在 124 秒超时，但原 child process 未被终止；后续只监控该原 PID，没有
  启动第二次执行。原进程最终写出 qualification 并自然退出。
- 终态为 `D2_OFFICIAL_TRAIN_METADATA_COHORT_QUALIFIED`。1560 条 official-train
  split 中，按 ID 升序扫描 149 条后锁定前 6 个 eligible parents：4 条 5 Hz、2 条
  20 Hz。它们全部与 78-parent exclusion union 不相交，并从现在起固定为 D2
  one-shot Development mechanics cohort，不得重扫、追加或换源。
- ledger 的其余 143 条为：71 条已遇到的冻结 exclusions、69 条 candidate metadata
  三次 retry 后 404、2 条 invalid argument、1 条 fps 不合格。qualification JSON
  SHA-256 为
  `63a217c3e658bbe4fee9e351c5c9abf68379ec2ccb89a6c3449f1581e385ee47`。
- 主审全部离线 gates 通过；独立审计重算 13 项 bindings、900 个媒体对象 receipts、
  18 个 canonical modality receipt hashes、选择序列与权限防火墙后 `CLEAR`。
  tracked result JSON SHA-256 为
  `fae85191e9e7c77f4206c37b722899afacb539f30dc129a9f1fae32252858096`。
- 本结果只授权冻结下一份 D2 media/mechanics implementation contract。RGB/mask/depth
  bytes、pose 内容、teacher/student、D2 effect、reserved official-test、研究主线、
  App、Android、生产与安全权限继续关闭。

## 2026-08-02：HFTF D2.1 result-changing 定义冲突在媒体前澄清

- 执行者：violjjet
- mechanics 实现审查发现原 D2 同时要求 exact G0 field，又写了 predicted field-domain
  外点排除；两者会产生不同 clearance。exact G0 runner 实际让全部 finite、semantic
  admitted stride4/offset2 points 对每个 cell 产生 proxy，nonmember 是正
  closed-box SDF，仍参与 second-smallest。另一个 blocker 是 yaw 未冻结投影轴、
  `atan2` 符号与 predicted basis 公式。实现因此暂停，未新增 mechanics 文件。
- 新 D2.1 不回写旧协议，只在两个冲突上取得优先级：忠实 exact G0、不做全局
  theta/distance prefilter；history/current forward 都投影到 current ground tangent
  plane，以 `atan2(up·cross(history,current), dot)` 得到 `[-pi,pi)` 最短角，再用
  Rodrigues 绕 current up 延拓。predicted right 固定为
  `cross(predicted_forward,up)`，origin 使用 current ground projection 加 tangent
  translation velocity。
- current ground plane 精确绑定既有 verifier：stride16、下部 45% 像素、
  depth `[0.5,8.0]`、classes `{1,3,5,6,17,30}`、source-frame seed。每个 anchor
  只读自身 history/current pose 与 current mask/depth，并在后续 anchor 前 durable
  写出；全部 84 anchor-horizon records 后才允许 truth join。
- D2.1 JSON SHA-256 为
  `51ed1c0bc2a98481b4991f237d44979cf0c455624031c2c0ee41715ec0d6a8f0`。
  独立只读审计重算 3 个 parents 与 3 个 implementation receipts，并核对 exact
  G0、pose/ground authority、角度符号、per-anchor firewall 及 outcome 边界后
  `CLEAR`。冻结时仍只有 metadata outcome；媒体、pose 内容与 geometry/effect
  outcome 均未打开，因此这不是同 cohort outcome-driven retuning。
- 本节点仍不授权媒体、preprocessor、truth/effect 或 RGB student；只允许继续冻结
  hash-bound one-shot media/mechanics implementation contract。

## 2026-08-02：HFTF D2 六源短路径媒体获取执行合同冻结

- 执行者：violjjet
- 在 6 条 official-train Development parents 仅有 metadata outcome、尚未读取其
  RGB/mask/depth bytes 或 pose 内容时，冻结独立的一次性媒体获取合同。合同精确继承
  D2、D2.1、tracked metadata result、完整 qualification artifact 与 T0 short-path
  equivalence；6 个 source、fps、13-frame timeline 及全部 generation/size/MD5
  receipts 均不可替换。
- acquirer 在首网前验证 exact contract/acquirer/test 为 tracked、clean、hash-bound，
  实际 GCS metadata/download/retry/`.tmp` dependency 也必须为 exact
  hash、tracked/clean。随后确认 `HEAD == origin/master`、`--retries 3`、
  canonical root 不存在，并在首网回调前用 exclusive create + `flush + fsync`
  固化 durable attempt。任一 source 失败即
  `D2_MEDIA_ACQUISITION_NOT_EVALUABLE_NO_RETRY_NO_SOURCE_REPLACEMENT`；不得重跑、
  换源、追加或 partial fill。
- acquisition 只下载并校验固定对象；RGB/mask/depth 不解码。完整 pose CSV 校验后，
  仅把 13 个 selected READY rows 物化为独立 hash-bound pose slices，作为后续
  future-blind preprocessor 的最小读取接口，不计算 candidate、truth 或 effect。
- source-blind preflight 未联网、未打开媒体、未创建 acquisition root；覆盖 1510 个
  final/staging/downloader `.tmp` 内容路径，最大长度 173，receipt SHA-256 为
  `c41ee24cb13978ea8bf50b7df26063967bf651a508f9b715504505254e81fb95`。
  acquirer/test/network dependency SHA-256 为
  `31802d25db633265988c989136fc4d1a4ebbb4a0007ab7ecf1ff1cb7531b8668` /
  `aafdf107a3d4422836a850a21ca3124c3bd6058416ef019d9a214da423667322` /
  `9e8694f0474adc20ea65068e70e6b28e49a0431daff2ad4cdb868ad5332a8854`；
  targeted tests 12/12 通过。合同 JSON SHA-256 为
  `e4e457cfac3d1009866dc0832d22757744707a86938ff2eddc1f2771bbdc147f`。
- 本节点只授权在 exact files 提交推送、远端一致和独立首网前审计 CLEAR 后执行一次
  六源获取。成功只允许另冻 future-blind mechanics execution contract；teacher、
  truth/effect、student、reserved official-test、研究主线、App、Android、生产与
  safety 权限继续关闭。

## 2026-08-02：HFTF D2 六源短路径媒体获取成功并离线封存

- 执行者：violjjet
- exact contract/acquirer/test 由 commit
  `1f04af5bb77acee45ce3432c5d5ce0d5784f8c92` 提交推送并再次确认
  `HEAD == origin/master` 后，只启动一次正式 CLI。durable attempt 在首网前完成
  exclusive write、`flush + fsync`；原进程自然退出，没有重启。
- 终态为 `D2_SIX_SOURCE_SHORT_PATH_MEDIA_COHORT_ACQUIRED`。254/254 下载请求均在
  attempt 1 成功，retry line 为 0、stderr 为 0 bytes；6/6 sources 全部完成后，
  整个 staging 才原子发布为 cohort。没有 failure terminal、换源、追加或 partial
  fill。
- 独立离线审计闭合 378/378 files、0 missing、0 extra，共 300,811,962 bytes。
  234 个媒体对象共 299,513,891 bytes，全部 SHA、size、MD5、name、generation
  匹配；6 个 pose CSV 与 78 个 pose slices 的 selected row、READY、finite、
  xyzw quaternion、CSV SHA 和 index SHA 全部一致。final/假设 `.tmp` 最大路径
  168/172，无 `.tmp`、staging 或 failure 残留。
- attempt/acquisition receipt/cohort manifest/per-frame index SHA-256 分别为
  `8153156da811807e927c600ce12342b640eee8ae8f481587f4b08cc292cc3117` /
  `59c9677393b06809b160163b81c918c6635c0fe6db2e6c12ba13b027e39667a6` /
  `07b968e97c1a010c7d49beff6d09dc2fb8677826680be6ea4efc235aedd355c4` /
  `60e63e2df8b2813519e90a287b841dbcfa2b2c9a9b0765b1f10ebcf7c9c8b2a8`。
  offline validation SHA-256 为
  `62abd95c32926417b04986b1872c45951a64a307cb74f0549ac1f0f43ac186c4`；
  tracked result JSON SHA-256 为
  `dd877a90d7198445f32dc33c9310bbfdf1c3d5bf11ad7c6881bd623045add50e`。
- RGB/mask/depth 只做流式 hash，未解码或视觉打开；future pose/depth/mask truth、
  candidate、geometry teacher 与 effect 均未执行。本节点只授权冻结另一个
  hash-bound mechanics execution contract；student、reserved official-test、
  研究主线、App、Android、生产与 safety 权限继续关闭。

## 2026-08-02：HFTF D2 future-blind mechanics 执行合同冻结

- 执行者：violjjet
- 在六源媒体 acquisition outcome 已封存、但 candidate prediction 与 future truth
  均未打开时，冻结 D2 mechanics execution contract。合同精确绑定 D2/D2.1、
  tracked metadata/media results、完整 qualification 与 per-frame index、G0/swept
  mechanics，以及 common/preprocessor/evaluator/tests 的 exact bytes。6 parents ×
  7 anchors 固定生成 42 个 predictions 与 84 个 horizon records，不得换源、追加或
  同 cohort 调参。
- preprocessor 只读每个 anchor 的 history/current pose slices 与 current
  depth/mask；在首次输入读取前独占写入并 `fsync` attempt，每个 anchor 的 points 与
  prediction 在读取下一 anchor 前 durable。evaluator 只有在 completion 闭合 exact
  order/count/hashes 后才能启动，并在首个 future pose/depth/mask read 前独占写入并
  `fsync` truth-join receipt。既有 pretruth failure、truth receipt 或 effect failure
  都在任何第二次 completion/future read 前 fail-closed。
- common/preprocessor/evaluator/test SHA-256 分别为
  `7f2a4041c7275c94e27cb8a30b5107f6e0ed15a9b54193e5ec3409461ba62071` /
  `aec88988188f027878fb7951d696a4789b59f8478ddfb52cd5d0c0579557078f` /
  `166641e2b277d476628908d6c9d0d56f0f18df41970922cdf22f0918a7c0ab2e` /
  `1d8393be7e99626285263ce96aef2c9dc4ac24ca9872d13d30b73650b8b7c97e`。
  targeted tests 13/13、HFTF 全集 337/337 通过；合同 JSON SHA-256 为
  `2afb530400b157990474523f4157630f9bf1bc225f15e32bfe9a0ffd4f034c56`。
  独立最终 hash-after 审计核对两个入口、prior-failure rerun guard、全部 canonical
  outputs 仍不存在及权限布尔值后 `CLEAR`，且未打开或解码 D2 media/future truth。
- 本节点只授权 exact files 提交推送并确认 `HEAD == origin/master` 后，依次各执行
  一次 future-blind preprocessor 与 truth/effect evaluator。即使正终态也只授权另冻
  RGB student protocol，不授权 student training/execution；研究主线、默认 App、
  Android、reserved official-test、生产与 safety 权限继续关闭。

## 2026-08-02：HFTF D2 mechanics 因 opportunity 不足而 NOT_EVALUABLE

- 执行者：violjjet
- exact mechanics contract/implementation/tests 由 commit
  `ed56242178538cb2c83ee465615cf9073e78caad` 提交推送并确认
  `HEAD == origin/master` 后，preprocessor 与 evaluator 严格依序各启动一次并自然
  退出。preprocessor 先 durable attempt，再完成 42 个 points/prediction 对与 84 个
  horizon records；全部 prediction 的 `future_depth_mask_or_pose_read=false`。
- completion 离线闭合后，evaluator 在首个 future pose/depth/mask read 前排他写入并
  `fsync` truth-join receipt，然后精确产生 84 个 synthetic geometry-proxy truth
  records。没有 failure artifact、stderr、重跑、换源、追加、partial fill 或同 cohort
  retuning。
- attempt/completion/truth-receipt/result SHA-256 分别为
  `5203515259ac66fb63529efe24073d2f5304c484531364cb553ba73a0136ece0` /
  `da01d2abe5ba3f07e87f2f68d0862abbddd7a119cc67e76e00c91e231a158ca3` /
  `b6186923b1fdc051ae9af6984d973a07475c14c3e2ae1bba642d00661a15ef99` /
  `a6c34d28876c46b09b3507ab46468530c04ea9b409d5fdd3e0d0701b91356276`。
- 24 个 frozen `parent × height × horizon` opportunity strata 只有 8 个通过、16 个
  失败。16 个失败 strata 全部 known-risk cells 少于 5；其中 3 个还同时低于 0.10
  common-known coverage 与 20 个 known-safe cells。UNKNOWN→SAFE 为 0。独立只读
  审计重算 42 prediction hashes、84 truth keys/future offsets、24 strata 与完整
  hash chain，全部 0 mismatch，最终 `CLEAR`。
- 终态为
  `D2_NOT_EVALUABLE_OPPORTUNITY_INADEQUATE_NO_SOURCE_REPLACEMENT`。effect gates、
  MAE、F1 与 parent improvement 均未获得判定权限，所以既不能支持也不能否定
  transport 假设。该六源 cohort 已消费且不得定向救援；RGB student、reserved
  official-test、研究主线、默认 App、Android、生产与 safety 权限全部关闭。任何继续
  必须建立新的 protocol/data-role 边界，并在新 mechanics outcome 前独立冻结
  opportunity-adequate cohort 规则。

## 2026-08-02：HFTF D3-Q0 条件机会挑战集主协议冻结

- 执行者：violjjet
- D2 六源 cohort 与其 `NOT_EVALUABLE` 终态保持不可变。新 D3-Q0 只把数据角色改为
  `REFERENCE_AND_SUPPORT_ONLY` opportunity-qualified conditional challenge cohort；
  D2 signed-clearance field、SE(2) mechanics、estimand、5/20/.10 opportunity gates
  和全部 effect gates均不改，因此不是 D2 source replacement 或同 cohort 救援。
- 后续 implementation contract 必须在任何 D3 media/support/truth 前同时锁定
  metadata roster、qualifier、exact D2 primitives、effect skeleton、tests 与 transport
  dependencies。source pool 固定为排除全部历史 burned/consumed/closed/reserved 后的
  SANPO-Synthetic official train，按 session ID 字典序；最多 40 个 truth-screened
  slots，首 6 个四 strata 全合格即停。slot failure 也消耗预算，不得第 41 个、替换、
  跳序或依据 D2 parent/fps/scene/motion/risk deficit 选源。
- qualifier 只能使用三方 9-probe support/known 与 future truth sign 形成 exact
  common-known coverage/risk/safe counts；禁止计算或落盘两臂 clearance values、
  MAE、F1、confusion、delta 或 improvement。selector receipt 与 effect-only sealed
  truth payload 必须隔离；所有打开 support/truth 的 source 立即 burned。
- Q0 JSON SHA-256 为
  `42773cc9b0f27c187e97b7a03dfd96570e9178dbd742ee2b41759cce973d9b5f`。
  独立只读审计复算 6 个 parent hashes、JSON/MD、40-slot/first-6、sealed-truth
  firewall、effect skeleton 前置与所有授权后 `CLEAR`，未修改文件。
- 当前只授权另冻 hash-bound implementation execution contract。D3 metadata scan、
  media/pose、support/truth qualification、effect、RGB student、reserved official-test、
  研究主线、默认 App、Android、生产与 safety 权限全部仍为 false。

## 2026-08-02：HFTF D3-Q0 metadata-only 40-slot roster 合同冻结

- 执行者：violjjet
- 为避免在 qualifier 完成前打开任何新媒体或 truth，D3 实现拆成两级。第一级只从
  official train metadata 锁定 40-slot roster；成功后仍需另冻完整
  reference-and-support qualifier、sealed-truth firewall 与 outcome 前 effect skeleton，
  才能开始逐 slot screening。
- exclusion 从已封存 D2 qualification 机械派生：原 78 个互斥 parents 加完整 D2
  consumed six，共 84 个且禁止手工增删。剩余 source 按 session ID 升序；metadata
  eligibility 只检查 synthetic/chest-left/5或20 Hz/pose receipt 与 exact selected
  13-frame RGB/mask/depth receipts，不读取媒体或 pose 内容。
- 独立首轮审计发现 planner 误复用 D2 的 50-frame helper，会把“所需 13 帧完整但
  非选中帧缺失”的合法 D3 source 错误排除并改变 first-40 roster。正式执行前已改成
  D3-specific exact-13 qualifier：5 Hz 为 `0..12`，20 Hz 为 `0,4,...,48`；新增
  selected-only 13-frame canary，明确允许非选中帧不存在。没有通过修改合同来迎合旧
  实现。
- planner/test/contract SHA-256 分别为
  `d23d335e07b474b6a2f1edbd21df3377f033676f8f4e907f0bcb6ebe359b910d` /
  `8c4fafc2c8e2595628bb7a58242fa87130aab14617905a014b22f6107ddb7642` /
  `efc95ee82fa5bb31b4d26744841ef4a45df5ca56d3da88f944af3cc1a7991614`。
  targeted tests 15/15、HFTF 全集 352/352 通过；最终独立 hash-after 审计复算
  exact-13、84 exclusions、first-40、首网前 `fsync`、no-rerun 与全部 firewall 后
  `CLEAR`。
- 当前三个 canonical paths 均不存在。本节点只授权 exact files 提交推送、确认远端
  一致后执行一次 metadata-only scan。成功不授权媒体、pose content、support、truth、
  effect、student、reserved official-test、研究主线、默认 App、Android、生产或
  safety。

## 2026-08-02：HFTF D3-Q0 metadata-only 40-slot roster 锁定

- 执行者：violjjet
- exact contract 由 commit
  `f4b5b2581f6b56d6847148bc1ce5e829a3a0ef1f` 推送，执行前
  `HEAD == origin/master`。只启动一次 metadata CLI；首网前 exclusive
  `attempt.json` 已 flush/fsync，原进程被监控且未重启，约 928 秒后自然写出
  `D3_Q0_METADATA_ROSTER_40_SLOTS_LOCKED`。stderr 为 0，failure artifact 不存在。
- attempt/roster SHA-256 分别为
  `a2f1764b7af0f5a9f50d28e7e489be38f98a88947e606bda0712ef7dce409800` /
  `8720a68855e0ddcbee9ae174de69383dd6d596329d76f83d0798197e333ba7db`。
  official train 1560 个 session 中按 ID 升序检查到第 236 个账本条目时锁定前 40
  个 metadata-eligible slots：77 个 frozen exclusions、115 个三次重试后 404、
  4 个非 5/20 Hz，另有 16 个 5 Hz 与 24 个 20 Hz 合格 slots。
- 主审确认 40 IDs 升序、唯一、与完整 84-parent exclusion union 零重叠；复算
  120 个 modality receipt-list hashes、1560 个 selected-frame receipts、5/20 Hz
  exact-13 timelines、split/contract/attempt bindings 和全部 firewall，0 mismatch。
- 独立只读审计重新派生 78+6 的互斥 exclusion categories、前 40 个 eligible
  选择序列、120 组 modality hashes、1560 个对象 receipt 与完整 binding chain，
  结论 `CLEAR`，0 mismatch、0 blocker，未打开媒体或运行下游流程。
- 这只是 metadata roster terminal，不是 HFTF effect。当前只授权冻结独立的
  qualifier/sealed-truth/effect-skeleton execution contract；roster 不可重跑、替换、
  追加或重排，逐 slot 媒体/pose、support/truth、effect、RGB student、reserved
  official-test、研究主线、默认 App、Android、生产与 safety 权限全部仍关闭。

## 2026-08-02：HFTF D3-Q0 screening/sealed-effect 执行合同冻结

- 执行者：violjjet
- 在已封存 40-slot metadata roster 后、任何 D3 slot pose/media/support/truth 前，
  同时冻结 next-slot qualifier、selector-only aggregator、selected-six future-blind
  preprocessor、sealed-effect evaluator、D2/G0/mechanics dependencies 与 29 个专属
  tests；hard-interruption 回归补强后专属 tests 增至 35。canonical screening root
  在冻结与验证时不存在。
- 40 slots 保持原字典序，failure/interruption 也消耗 slot；首 6 个四 strata 合格
  source 立即停止，禁止 replacement、reorder、manual skip、budget expansion 与
  outcome 后改门。每个 body/head × `.4/.8 s` stratum 的 denominator 固定 252，
  common-known coverage/risk/safe 门为 `.10/5/20`，UNKNOWN→SAFE 必须为 0。
- qualifier 只下载 1 个 pose CSV 与 normalized `2..12` 的 11 depth/11 mask，RGB 为
  0；只计算 persistence/advected support known 与 future truth，不计算候选臂
  clearance 或任何 effect metric。sealed payload 先 durable，再从同一待写确定性
  bytes 直接传递 SHA-256 给 selector；selector/aggregator 不接收或重读 payload。
- future-blind preprocessor 只读 pose `0..8` 与 current/history depth/mask `2..8`。
  effect 只在 exact first-six selection 与 42 predictions durable 后创建 attempt 和
  open-once receipt，再各读 selected payload 一次。pretruth failure、truth-open 后
  interruption、qualification/support recompute mismatch 都有冻结 no-rerun/no-
  replacement terminal；过早调用 evaluator 不消费 attempt。
- 首轮独立科学审计先以 `NOT CLEAR` 发现 selector 为取 hash 重读 sealed payload，
  以及 effect pretruth/interruption 终点未完全闭合；没有绕过。修复 deterministic
  bytes digest handoff、durable failure writer 与 contract validator 后，第二轮独立
  只读审计复核原问题及授权边界为 `CLEAR`，未发现新科学 blocker。
- 独立工程审计随后以 `NOT CLEAR` 发现 slot receipt 未强制绑定 durable attempt、
  aggregator/preprocessor/effect 的 hard-interruption 孤儿状态、aggregate attempt
  顺序与 sealed-open count 缺口。修复后 state scanner 逐 receipt 重验 slot/global
  attempt hash，aggregator 在首个 receipt read 前 durable 写 attempt，所有 `.tmp` /
  `.orphan` 恢复只封存 failure 而不重开输入，opened payload receipt 在首次 read 后
  立即计数。最终独立工程审计结论 `CLEAR`，Windows final/tmp/orphan 最大路径实测
  140 字符，小于冻结的 240 上限。
- common/next-slot/aggregator/preprocessor/evaluator SHA-256 分别为
  `26bf520b7646b8f331c0fcd15fead1666b37370889adf150ab9444d402745356` /
  `3e4dbac02359261f5339c786935552941bdd3c143a912194faa34a61f238c4c2` /
  `2cd26d235a8bf3779dafbcb05622e6393d055ea48b2bc551d4931760461fae4c` /
  `9226bbe9cb1088890e4d24ec2e2bcc604ddb8476582f302ba0bc05736244fe8f` /
  `d98b0bea9555a40add2887917fae5b1360a91a14a8a5c9b16ef82a1c958a3bb2`；
  execution-contract JSON SHA-256 为
  `84f24a72c4640ca3ba66388ed9ec75a68aa55270c5e369b2b072a7b4d65354eb`。
  targeted 35/35、HFTF 全集 387/387 通过。
- 本节点只授权 exact files 提交推送、确认 `HEAD == origin/master`、formal
  `verify_git=True` 与最终工程审计通过后，执行唯一 next slot。它不授权 effect
  提前执行、RGB student、reserved official-test、研究主线、默认 App、Android、
  生产或 safety。

## 2026-08-02：HFTF D3-Q0 slot 1 closed-schema invalid stop

- 执行者：violjjet
- screening/effect contract 由 commit
  `306477105db033dbb805fc78bd8567c2afb29b34` 精确提交推送；执行前
  `HEAD == origin/master`、formal `verify_git=True`、35/35 targeted、387/387 HFTF
  与最终独立工程审计均通过。canonical root 原先不存在，只启动一次冻结 slot 1
  媒体/qualification 进程。
- global/slot attempt 均在首网前 durable。进程下载 1 pose CSV、11 mask、11 depth，
  RGB 为 0，并 durable 写出 content index、sealed payload 与 selector；随后
  state scanner 拒绝 selector：runner 把 `slot_attempt_sha256` 同时写在允许的
  `source_authority_and_content_hashes` 与禁止的 selector 顶层，closed schema 报
  `extra=['slot_attempt_sha256']`。
- 没有通过放宽 validator、修改 receipt 或重启媒体进程来救援。恢复调用只重读
  selector receipt 并写出 `D3_QUALIFICATION_INVALID_STOP`，没有重开 media 或
  sealed payload。selector 的 forensic terminal 为 not-qualified，但因 schema
  非法不得进入 screening state、统计或后继选源依据；slot 1 永久 burned。
- screening attempt / slot attempt / content index / sealed payload / invalid
  selector / invalid terminal SHA-256 分别为
  `137d0fa065c2eabd61fdc2ba158b12d9f586c1021fe2b0e64a292faf5492f364` /
  `bff9cc469a1b9571fa9e858eafe853e646fba8a935476bf9e2e225b7c08e44f8` /
  `7df2d5fbeab7483235f38b8fd9f2fa50007eab8c909ba55fa529a620b2610f6a` /
  `7a1271ffa876df453df38ea52ba3db4c14044631ef9dc70e44023ea5433d55ed` /
  `cbad78e83d3b3aca80a2a9faaa6d14bde2151ae08e10fc9e2f922d99a1814865` /
  `e1975e896b5d6a26f8a28ee7ee29b5a9d1d3f4cc53b0a183c3dd0aec658e962d`。
  selection、budget terminal、aggregate attempt 与 slot failure 均不存在。
- 同一 Q0 contract 已关闭，禁止 rerun、reopen、replacement 或 expansion。唯一
  新权限是冻结独立 Q0.1 schema-only successor：只删除重复顶层字段，slot 1 保持
  burned，从原 slot 2 开始最多 39 slots；`.10/5/20`、252 denominator、四 strata、
  UNKNOWN、effect gates、roster order 与 no-replacement/no-expansion 全部不变。
  新 contract/root 提交推送和独立审计前不得打开 slot 2。

## 2026-08-02：HFTF D3-Q0.1 schema-only successor 冻结

- 执行者：violjjet
- 以已封存的 Q0 execution contract 和
  `D3_QUALIFICATION_INVALID_STOP` result 为双重法源，冻结独立 Q0.1
  contract 与新 canonical root。Q0.1 不是新科学协议或新 cohort，只删除
  selector 顶层重复且被禁止的 `slot_attempt_sha256`；authority object 内合法
  attempt receipt 保留。
- 新 root 的 slot 1 只允许 outcome-free carry-forward burn receipt：原 slot 1
  计入 40-slot 总预算，但不计 qualified、not-qualified 或 execution failure；
  旧 content、sealed payload、selector 和日志不得重开或导入。原 index/order
  保持，首个 active slot 为 2，最多新开 39 个。
- 四 strata、252 denominator、`.10/5/20`、UNKNOWN、首六即停、failure consumes、
  no-replacement/no-expansion、future-blind 输入集合、42-prediction 前置、D2
  estimand 与 effect gates 全部保持不变。首个 runner 调用被实现为纯控制面
  初始化，第二个调用才可能触碰原 slot 2。
- 专属 state/pipeline tests 当前为 23/23 与 17/17；其中显式覆盖 synthetic
  duplicate-field regression、slot-1 任意非 carry artifact 拒绝、slot-2 起始、
  burn+39 budget、首调用零媒体和第二调用只指向原 slot 2。合同提交推送、
  `HEAD == origin/master`、formal `verify_git=True` 与最终独立审计前，slot 2
  仍不得打开。
- 最终独立科学审计与工程审计均为 `CLEAR`、0 blocker。工程复核全部 parent /
  implementation / test hashes、8319 个 canonical final/tmp/orphan paths（最长
  142 < 240）、py_compile 与 40/40 专属 tests；项目标准环境的 HFTF 全集
  392/392 通过。冻结 contract SHA-256 为
  `268f1491835fb8b4d365a24064eac94edc5046633fa7861b7fbd1588ded7225a`，
  审计时 canonical root 不存在。

## 2026-08-02：HFTF D3-Q0.1 screening 预算耗尽终点

- 执行者：violjjet
- Q0.1 contract 由 commit `ef248690e60a77ba5ab4f98443fefaa64fbc1b50`
  精确提交推送；执行前 `HEAD == origin/master`、formal `verify_git=True` 与
  双终审均通过。首次调用只创建控制面与 slot-1 outcome-free carry receipt，没有
  打开新媒体；随后原 slots 2–40 严格按原顺序各执行一次，没有重跑、重开、替换、
  重排、扩预算或同 cohort 调门。
- 40 个 consumed slots 闭合为 1 carry + 5 qualified + 32 not-qualified + 2
  execution failures。qualified 为原 slots `3/14/20/29/37`；原 slots `2/28`
  因 `D2 current ground sample is inadequate` 消耗槽位。只形成 5 个 qualified，
  未达到冻结的 first-six / 6-source formal cohort。
- aggregator 在首个 receipt read 前 durable 写 attempt，只运行一次且不读 sealed
  payload，终态为
  `D3_REFERENCE_SUPPORT_OPPORTUNITY_COHORT_NOT_EVALUABLE_BUDGET_EXHAUSTED_NO_EXPANSION`。
  budget terminal SHA-256 为
  `e992a8117184b2f97dbfd4ac81805cc665a003fbf6f85167fec1d213d2b9e89b`；
  `selection.json`、`screening_invalid.json` 与 `formal/` 均不存在。
- 独立科学审计用冻结 scanner 重建 terminal 逐字段一致；独立工程审计复核 40-slot
  receipts/hash/log/process 闭集，均为 `CLEAR`、0 blocker。该结果只说明当前冻结
  roster/order/budget/gates 未形成 effect cohort；不支持或否定 transport/HFTF，
  不授权任何 D2 effect metric、preprocessor/effect、RGB student、主线/App/Android、
  生产或 safety。后继只能另建独立 protocol/data-role 边界，不得救援本 cohort。

## 2026-08-02：HFTF D3-Q0.1 consumed-selector Failure Atlas R0

- 执行者：violjjet
- 对 37 个已消费合法 selector receipts 做只读描述性归因，不打开 sealed payload、
  媒体或 formal artifact。148 个 strata 中 93 个失败；89/93 包含 risk-count
  failure，68/93 仅 risk 不足，coverage/safe 分别失败 24/148、25/148，
  UNKNOWN→SAFE 为 0。head × `.8 s` 只有 7/37 通过，30/37 risk 少于 5。
- 该 Atlas 只形成
  `D3_Q0_1_CURRENT_REFERENCE_TRUTH_RISK_OPPORTUNITY_SCARCITY_DOMINANT_HYPOTHESIS_ONLY`：
  当前 reference/qualification 表示下 risk opportunity 是主导瓶颈，但不能区分
  自然稀有、teacher/reference blind spot 或 sampling mismatch。slots 4/7 的近失
  数值是 outcome 后诊断，不授权把 risk 门从 5 降到 3、追认样本或扩预算。
- 独立科学审计为 `CLEAR`，同时强调 37 个 parent 才是独立单位、2 个 failure source
  存在 selector 选择性缺失。首选后继改为
  `D4_OPPORTUNITY_ECOLOGY_AND_RECRUITABILITY`：在 fresh prospective source
  population 中先估计 source-level all-four opportunity 率与 pre-truth metadata
  招募成本；成功也只能授权另一批独立 sealed-effect cohort。受控 paired geometry
  intervention 降为第二候选。Atlas 只授权冻结设计，不授权执行；Q0/Q0.1、student、
  主线/App/Android、生产与 safety 均保持关闭。

## 2026-08-02：HFTF D4 opportunity ecology/recruitability R0 冻结

- 执行者：violjjet
- 设计与机器合同见
  [D4 R0](docs/research/hftf/HFTF_STAGE_C_D4_OPPORTUNITY_ECOLOGY_AND_RECRUITABILITY_R0_2026-08-02.md)
  和
  [D4 R0 JSON](docs/research/hftf/HFTF_STAGE_C_D4_OPPORTUNITY_ECOLOGY_AND_RECRUITABILITY_R0_2026-08-02.json)。
- 将后继问题从 effect 改为 fresh source recruitability：先估计前瞻 target
  subpopulation 的 all-four opportunity rate 与 pre-truth metadata 招募成本，再决定
  是否值得消耗另一批独立 effect pool。Q0.1 的 5/37 及 post-hoc 5 Hz `3/15`、
  20 Hz `2/22` 仅生成假设，不进入 fresh interval 或 effect。
- 第一执行级 M0 只允许为后续 metadata census 冻结合同：在 SANPO-Synthetic
  official train 1560 IDs 中排除原 84 + Q0 40 = 124 个互斥 consumed/reserved
  parents。实现前复核确认其中 6 个是 official-test parent、不在 train，因此保留
  global 124 exclusion authority，但 train 投影为 118 exclusions + 1442 个必须
  各尝试一次的 metadata candidates；完整账本仍覆盖 1560 IDs。M0 禁止读 pose
  content、RGB/mask/depth bytes、support、truth、clearance/effect 或 sealed payload。
- 首轮独立科学审计以 `NOT CLEAR` 拒绝“M0 后再人工决定 target/allocation/p_min”。
  修订后 target 固定为 fresh 5 Hz metadata-eligible parents，source-content 外生上限
  128；M0 只把 `N` 机械代入 `C=min(N,128)`、`n=floor(3C/8)`、`B=C-n`，单一
  hash rank 前 n/后 B 分给 ecology/effect。`N<64` 即停止，20 Hz 不 fallback。
- 推断预冻为有限总体无放回 exact hypergeometric：反演 ecology 的 source-level
  `x/n` 得到 `K_L/K_U`，并以 effect reserve 中获得至少 6 个 qualified 的概率
  `>=.90` 推出 `R_min`。lower bound 过门才 GO，upper bound 不足则 STOP，中间只能
  NOT_EVALUABLE；不得扩样、改 allocation、换频率或互相补位。
- 第二轮独立科学审计仍以 `NOT CLEAR` 指出无 seed hash 不等于 uniform random
  permutation，且 exact CI 未锁 alpha/离散尾。修订后 M0 必须在 eligible manifest
  fsync 后 one-shot 用 OS CSPRNG 生成 32-byte seed 并绑定 attempt/manifest；
  orphan/重抽直接 invalid。CI 固定 one-sided 95%、`alpha=.05` inclusive exact tail，
  ecology 跑满 n，所有 acquisition/execution/orphan failure 的 operational indicator
  为 0；`R_L/R_U` 三分终点与未分配 parent 禁用规则闭合。
- 最终独立科学与工程终审均为 `CLEAR`、0 blocker。工程复算 parent bindings、
  exact 124-parent exclusion union、单一 seeded rank、allocation/CI/failure 闭集与
  README/log 链接。后续实现前审计修正了 `1560-124=1436` 的错误使用：global
  124 中有 6 个 official-test IDs，故 official-train M0 的正确 candidate count
  是 `1560-118=1442`；修正后的 D4 JSON SHA-256 为
  `d7d26ac2267fe43c2a80d36cfe164a5544e34034c3b80509544be1591e3f0a68`，
  并将由 M0 execution contract 绑定。
- 当前仅授权冻结/审计 M0 execution contract，不授权执行 metadata census、ecology、
  effect、student、主线/App/Android、生产或 safety。

## 2026-08-02：HFTF D4-M0 metadata census execution contract 实现

- 执行者：violjjet。新增 D4 专用 M0 planner、15 项 focused tests 与机器合同；不修改
  已关闭的 Q0/Q0.1 实现。正式 ledger 固定为 1560 rows = 118 train exclusions（零
  candidate 请求）+ 1442 candidate attempts，保留 global 124 exclusion authority
  及其中 6 个 official-test IDs 的精确身份。
- M0 eligibility 只读 description bytes、pose object metadata 与 exact-13
  mask/depth listings；RGB listing、pose/media bytes、support、truth、clearance、
  effect、sealed payload 全部 fail closed。5 Hz pool 持久化后才写 allocation attempt，
  再 one-shot 调用 `secrets.token_bytes(32)`；rank、`C/n/B` 和三组分配完全机械化。
- Windows durable barrier 明确为 exclusive create + file fsync + close + exact-byte
  reopen verify，不虚构不受支持的 directory fsync。任何 partial/unknown root 后续
  只冻结 INVALID，不联网、续跑或重抽 seed。
- 审计后将 local drift 检查收紧为只读 exact `1+39` 个 slot `attempt.json`，完全不
  遍历 sealed payload/selector/truth；attempt 后、首网前另写 preflight。仅 HTTP 404
  或确定 schema failure 可记 ineligible，timeout/DNS/5xx 使 one-shot INVALID；已有
  terminal 必须通过 exact closed set、schema、terminal 与 hash chain。
- focused tests `21/21`、HFTF full suite `413/413`。valid locked/insufficient
  terminals 均覆盖全上游 hash chain，任一 preflight/attempt tamper 会被拒绝。当前仍只授权提交推送与独立审计；
  正式 M0 尚未执行，canonical root 必须保持不存在。机器合同 SHA-256 为
  `21a6de0e16e65998318aa83b549c3467eb9fe2b59193faa1fa44d72d1d891759`。
- 提交后 formal preflight 暴露 Git 门范围错误：ignored `artifacts.local` parents 被
  错要求 `ls-files`。修订为合同/设计/implementations/helpers/tests 必须
  tracked-clean-pushed；ignored evidence parents 仍由 exact path/SHA/schema/terminal
  约束，不改变任何数据或实验规则。

## 2026-08-02：HFTF D4-M0 metadata census invalid stop

- 执行者：violjjet。Git 门修复提交 `72af4c7` 推送且 formal preflight 通过后，只启动
  一次 D4-M0 CLI。`05:48:33 +08:00` durable 写入 attempt/preflight；preflight 仅读
  exact 40 个历史 slot attempts，全部 IDs 位于 frozen global-124 union。
- 外层 wrapper 一小时 timeout 后原 Python PID 仍存活，随后只监控原进程，没有重启、
  resume 或新 seed。原进程在 `06:51:30` 因
  `OSError: [Errno 22] Invalid argument` 自行写入 failure 并退出，终态
  `D4_M0_FRESH_METADATA_RECRUITABILITY_POOL_INVALID_STOP`。上述时间/PID 过程是未绑定
  process receipt 的 operator observation，不进入 canonical terminal 或 claim。
- attempt/preflight/failure SHA-256 分别为
  `7ba7f6a6bc9404fbe43dfee2955ad853929b32a7d7a310dcba4a38ccf404feb8`、
  `52735837a65f52603c31c4a3e6a2d76986d63e4cebb322904aadea34182efeb4`、
  `b9fb61cd33cd820113b246aaf9cf36ac58379dc37a916b5a03ff47fbafba96f5`。
  census/pool/allocation-attempt/seed/result 全部不存在；fresh pose/media/support/truth/
  effect 均未打开。
- 该终态不支持 5 Hz pool 不足、opportunity prevalence 或 HFTF effect 结论。同一
  canonical root、同一 1442-candidate census、transport patch 后重跑均关闭；任何
  后继必须是新 protocol 与新 source population。
- 机器结果 SHA-256 为
  `bba56892cd579b2e278705070ad6f42cbb6db1bc1264ec99de3132f9d888c993`。

## 2026-08-02：HFTF D5 TartanGround differential-drive S0 冻结

- 执行者：violjjet。D4-R0 invalid 后不修 transport、不复用 SANPO 1442 parents；
  新 source population 改为 TartanGround `Data_diff/P1xxx` natural trajectories。
  官方候选资料称全数据有 63 environments、878 trajectories、1.44M samples，并提供
  front RGB、metric depth、semantic segmentation、6-DoF pose、robot-height metadata
  与 semantic occupancy；这些只是 publisher claim，尚不是本地 inventory evidence。
- S0 提交推送后只允许 exact-commit toolkit clone 与 metadata catalog/list/dry-run，
  的原表述经审计收紧：本文件提交后仍不授权 clone/list，必须另冻 exact-commit
  execution contract、call allowlist、attempt-first、payload sentinel 与 failure closure。
  该合同最多允许 ZIP central-directory、exact metadata JSON，以及只作 SHA/行数的
  pose member stream；禁止解析 pose 值或任何 scene payload。
- feasible 门为至少 64 个 diff trajectories、8 environments、每 parent 同时绑定 robot
  height + extrinsic、`lcam_front` image/depth/seg/pose 共同至少 25 个 10 Hz raw
  frames，再以 `0,2,…,24` 规范成 13×5 Hz / 2.4 s；不可观察 authority 时整体
  NOT_EVALUABLE，不能伪装 pool insufficient。64/8 只表示容量/覆盖，同环境是 cluster；
  未来 ecology/effect 必须 environment-disjoint 且使用 environment 独立单位或预冻
  cluster-aware inference。
- 成功也只授权另冻 D5-M0 allocation/acquisition contract。TartanGround 始终是 synthetic
  ground-robot proxy，不是人体步态、盲人路线或 safety truth；主线/App/Android/生产
  权限保持关闭。
- D5-S0 机器设计 SHA-256 为
  `122eccb74d0eb83e231c4e1fa02a36284bab9e6b5df7d251845a7284eeff6b2d`。

## 2026-08-02：HFTF D5-S0A TartanGround exact-commit catalog 合同

- 执行者：violjjet。将 D5-S0 拆为不接触数据托管端的 S0A：只允许一次 fetch
  官方 `castacks/tartanairpy` 精确提交
  `158a6844d782942110967325ca3082f50ab2bfc7`，读取该提交中的
  `.gitmodules` 与 `tartanair/download_ground_files.txt` 两个 Git blob，并核对
  三个冻结 gitlink。fetch 显式使用 `--recurse-submodules=no`；禁止
  submodule checkout/read、数据 ZIP 请求、central-directory/member、pose、metadata、
  scene payload、opportunity/effect 或 student 输出。
- 清单正则只验证实际列出的 `environment/Data_diff/P1xxx`，不生成父体。目录完整
  父体必须列出 front image/depth/seg 与 metadata 四个 ZIP，清单行必须是
  `<path> <positive-decimal> G`；至少 64 trajectories / 8 environments 仍只是
  capacity/coverage 门，同环境轨迹保持 clustered。达标终点只能是
  `D5_S0A_TARTANGROUND_DIFF_CATALOG_LOCKED_REQUIRES_S0B_STRUCTURAL_AUTHORITY`，
  不能宣称 source feasibility。
- attempt 与 preflight 均在首个 Git 网络请求前用 exclusive-create + file fsync +
  close + exact-byte reopen 验证。任何 transport/object/format/gitlink/local-binding
  异常只写 INVALID；partial root 后续只冻结，不续跑或重试。已有 terminal 必须把
  frozen contract、attempt、preflight、catalog、result 与本地 `FETCH_HEAD` 的
  schema、hash chain、repository/commit/manifest/gitmodules identity 全部交叉核验。
- focused tests `22/22`、项目标准 HFTF full suite `435/435`。独立科学与工程终审
  均为 `CLEAR`、0 blocker。机器合同 SHA-256 为
  `49c104bec55324dd42454f8db88042216be30d9e796ea36db276eca18238a66f`。
  当前只授权提交推送；正式 canonical root 在提交推送与 formal git gate 前必须不存在。

## 2026-08-02：HFTF D5-S0A catalog invalid stop

- 执行者：violjjet。合同提交 `b65c0d916c7359a91e6854c6ffe7697728fdef6e`
  推送、formal Git/hash gate 通过且 canonical root 不存在后，只调用一次 S0A CLI。
  attempt/preflight 在首个 Git 网络请求前 durable；随后 manifest parser 在第 978 行
  以 `ValueError: Unexpected declared size format at row 978` 关闭为
  `D5_S0A_TARTANGROUND_DIFF_CATALOG_INVALID_STOP`。
- attempt/preflight/failure SHA-256 分别为
  `4a5b65a2a53ecfb343c50bff4929f03e8c0f109695df509098d3b2d499cf3ac8`、
  `0a5b9514e9a7332249c44169757551f051f79d128fe5cd4a392abbe4c6ed9652`、
  `28f4c0337935a0778d1a9ea58c89de559779d85d59d919347a948140d6dd7fd5`。
  `catalog.json` 与 `result.json` 均不存在；没有数据托管端请求、ZIP、submodule
  checkout/read、scene payload、pose value、structural authority、opportunity 或 effect。
- 终态后只做本地控制面核对：`FETCH_HEAD` 是冻结提交，`.git/modules` 与工作树
  `.gitmodules` 不存在；没有重开失败行或 manifest。该观察不在 canonical failure
  哈希链中，不参与 claim。本终态既不是目录容量不足，也不是 source/HFTF 负结果。
- 同一 root、同一合同、原合同内 parser patch/retry 全部关闭。若继续，只能另冻新版本
  控制面协议、新 canonical root，并在执行前独立审计；不会自动授权 S0B 或 payload。
  机器 invalid result SHA-256 为
  `f86153427117ed8542cb892204a693805b80b0f4eac87cdf18c26e9d2aad4961`。

## 2026-08-02：HFTF D5-S0A.1 opaque-suffix repair 设计

- 双审确认 S0A invalid 只消费控制面清单语法，没有产生 catalog/count，也没有打开
  scene/payload/outcome；因此不必换 source population，但禁止原 root/contract 内
  patch/retry。S0A.1 必须是新版本合同、新 canonical root、新 attempt/preflight 与
  新的一次 exact-commit fetch，且旧 toolkit/manifest 不复制、不复用、不读取。
- 唯一 parser 修订是：每个非空行只取首个 whitespace token 作为 path，余下 suffix
  全部丢弃且 opaque/non-gating；不要求 suffix 存在，不解释或验证 size/数字/单位/
  正负，不保留衍生指标，也不得针对第 978 行写特殊规则。path safety、manifest-only
  identity、四 archive、64 trajectories / 8 environments 与 cluster 边界不变。
- 本设计只授权提交推送及后续冻结 hash-bound execution contract；当前不授权新 fetch、
  manifest read、S0A.1/S0B、dataset ZIP、payload、ecology/effect/student 或任何
  主线/App/Android/生产/safety 变更。机器设计 SHA-256 为
  `10d1ed5085ea1978973fa6afd57a1cb4a737a8bec8b88f1c74806be93a90d0ee`。

## 2026-08-02：HFTF D5-S0A.1 execution contract 实现

- 执行者：violjjet。新增独立 S0A.1 planner/test/机器合同；旧 S0A root 只作为
  禁止使用的 path 常量参与 canonical-root 排除，没有 load/open/copy/git I/O。
  lineage 只绑定 tracked immutable invalid-result JSON。新 root 必须在 attempt 与
  preflight durable 后才单次 exact-commit fetch；fetch 固定 no-tags/depth-1/
  recurse-submodules=no，只读两个 root-repo blobs 与三个 gitlinks。
- parser 只以 LF/CRLF 分行，并把 TAB/FF/VT 保留为同行 ASCII whitespace；每行只取
  首 token path，suffix 不要求、不验证、不保留、不生成 manifest byte/hash/metric。
  执行级测试证明不同 suffix 形状及 LF/CRLF 得到完全相同 catalog observation。
- terminal validator 从 hash-bound catalog rows 机械重算 eligible parent/environment、
  64/8 passes、terminal、完整 gate 与 next authority，并强制 catalog/result 的旧-root、
  suffix、structural/source、S0B/payload/effect、主线/App/safety firewall 全 false。
  failure terminal 同时校验 attempt/preflight schema/status/hash chain 与可观察
  `FETCH_HEAD`；任何非空 commit 必须等于冻结提交。
- focused tests `24/24`、项目标准 HFTF full suite `459/459`。独立科学和工程终审均
  `CLEAR`、0 blocker。机器合同 SHA-256 为
  `84b9a2efbd9363ccf1fb2231a332dc96d63cfdd1d78219802f3e7a91397ee4d4`。
  当前只授权精确提交推送；formal 新 root 仍必须不存在。

## 2026-08-02：HFTF D5-S0A.1 catalog locked

- 执行者：violjjet。合同提交 `de088fb6be115769aaaaabeb1aed73d7ebc19002`
  推送、formal Git/hash gate 通过且新 canonical root 不存在后，只调用一次 S0A.1
  CLI。终态为
  `D5_S0A1_TARTANGROUND_DIFF_CATALOG_LOCKED_REQUIRES_S0B_STRUCTURAL_AUTHORITY`；
  canonical terminal validator 通过，failure 不存在。
- exact-commit manifest 有 34671 个非空且 unique path tokens，其中 7722 个
  `Data_diff/P1xxx` archive paths，形成 198 个 target parents / 42 environments；
  198/198 parents 全部列出 front image/depth/seg 与 metadata 四 archive，超过
  64/8 catalog capacity/coverage 门。suffix 未验证、保留或用于门，也未保留完整
  manifest byte/hash。
- attempt/preflight/catalog/result SHA-256 分别为
  `5f6b2fe547b43df54e87da4c675df7bc3e02c0177f79b657cbbcfd94f33daf0c`、
  `4a2d5fb59021df43f82ab71ab965db7febee603ffaf6520c435b9faf4186126d`、
  `a8a4c33aa4f57cc6ffdf882f030cac3374e6b381c4aea2d36fd32bfba92c46f4`、
  `10ab1e74d44753296c5dee58a3bd4bcdaa0c9f4e27cbe96ef59d59200f76cd73`。
- 本结果没有请求数据托管端或 ZIP，没有检验 pose、height、extrinsic 或共同时间线；
  不是 structural authority/source feasibility/opportunity/effect 证据。只允许另冻
  D5-S0B contract，不自动授权 S0B 或 payload。机器结果 SHA-256 为
  `8b2aeb086dcdfd18a675d281a887dbea3cc63a23b2f3b7cac1bd375e613a4a2f`。

## 2026-08-02：HFTF D5-S0B structural authority 设计

- S0B 将问题从 catalog capacity 提升为受限结构权威。先把 catalog 字典序首 parent
  机械保留为永久退出 pool/payload/effect 的 schema sentinel，再对其余 197 parents
  完整 census，逐个建立 finite positive metric robot height、带方向/frame/convention/unit
  的 robot→front-camera rigid extrinsic、至少 25 行的 exact dynamic front pose，
  以及 image/depth/seg/pose 共同连续 25 个 10 Hz raw frames；仍固定规范化为
  数值最早共同窗口内相对 offsets `0,2,…,24` 的 13×5 Hz / 2.4 s。
- 在任何数据托管端请求前，P0 source-only contract 必须从 exact toolkit commit
  解析并锁定 provider URL derivation/198-parent mapping，不得猜 URL；P1 再仅用永久
  排除的 sentinel 冻结 image/depth/seg member/index、metadata/extrinsic/pose 与 ZIP
  schema；R0 绑定 P0/P1、catalog hash/198 order 和 exact 197 census order。
- ZIP 闭集补齐为 EOCD/ZIP64/central-directory，以及仅 metadata/pose 的 bounded local
  header + compressed range；只允许 stored/deflate，并冻结 encryption/data-descriptor/
  ZIP64-extra/CRC/size/decompression-ratio budgets 与 fail-closed fixtures。
  允许的 member payload 只有 exact metadata JSON 和仅作 SHA/bytes/line-count 的
  front pose；image/depth/seg bytes、pose values 和其他 scene payload 均禁止。
- 若 provider/schema/field/unit/frame/index/member 的全局 authority 缺失或歧义，
  整体必须 `SOURCE_AUTHORITY_NOT_EVALUABLE`；transport/range/budget/hash/implementation/
  protocol-parser/partial failure 才是 INVALID；只有 authority/parser 全局有效且读取成功，
  但单 parent 缺项、malformed、height/extrinsic/pose/timeline 确定违规时才 ineligible。
  197 全部完成后，至少 64 parents / 8 environments 只过
  structural capacity/coverage 门；同环境仍 clustered。
- 本设计当前只授权提交及后续冻结 provider-resolution/S0B execution contract，不授权
  toolkit/provider read、dataset host/ZIP、S0B、payload、ecology/effect/student 或
  主线/App/Android/生产/safety。机器设计 SHA-256 为
  `87641ff8585dc5fe112d77cfacd3e5ce8c23b402b1396768473733f1c716aef1`。

## 2026-08-02：HFTF D5-S0B-P0A toolkit source closure contract

- 执行者：violjjet。P0A 只从 exact toolkit commit 的
  `tartanair/__init__.py` 出发，按 AST 追踪相对 import 与 `tartanair.*` import；
  tree 只读 names，只有 seed 可达 Python blobs 能读取。每个 blob 先查询 Git
  object-size，再在内容读取前检查 128 blobs / 4 MiB 总预算。
- 动态 import 不执行。直接 `__import__` / `importlib.import_module`、模块对象和
  callable 的简单别名会计数；subscription、container escape、`getattr`、`exec`、
  `eval` 等另计 indirect evidence。未来 P0B 遇到任一非零计数必须
  `NOT_EVALUABLE`；零计数只表示冻结检测器未命中，不是运行时完整性证明。
- terminal validator 重算 tree/closure hash chain、contract/status/bindings、预算、
  非负动态计数及从 seed 出发的 exact graph reachability；seed 缺失严格配为
  `D5_S0B_P0A_TOOLKIT_SOURCE_CLOSURE_NOT_EVALUABLE`。partial/failure 同时绑定
  attempt、preflight 与可观察 `FETCH_HEAD`，不得 resume/retry。
- focused tests `27/27`、HFTF full suite `486/486`。最终科学与工程双审均
  `CLEAR`、0 blocker。planner/test/叙述合同/机器合同 SHA-256 分别为
  `5a09da4d548775d0442e6cef327f0d50a003e60484cf90a9d2656f77d3c285d6`、
  `2f4fe44d8684920fc17ce77c55e8954c20226fdca8014193bb407cc0d2afcf7a`、
  `3f04ce0d02273bec82602171507064440e576b8c6bdd0df10e69508fc45ab1b2`、
  `0da2a0ca485435b5ad458895f2dbc1cb7c929794c69d888ab2f664dcad5bfb93`。
  当前只授权精确提交推送；formal root 仍不存在，未发生 toolkit 或 dataset-host
  请求。

## 2026-08-02：HFTF D5-S0B-P0A toolkit source closure locked

- 合同提交 `3789d3b1ed7c01f2a1bb2fc93a414df48ddfc2fc` 推送且 formal gate
  通过后，只调用一次 P0A CLI。终态为
  `D5_S0B_P0A_TOOLKIT_SOURCE_CLOSURE_LOCKED_REQUIRES_P0B_PROVIDER_RESOLUTION`；
  canonical terminal validator 接受，failure 不存在。
- exact toolkit tree 含 25 paths / 19 Python paths；从
  `tartanair/__init__.py` 出发的冻结静态 import closure 为 18 blobs /
  250569 bytes。direct dynamic 与 indirect dynamic/exec evidence 均为 0；
  这只表示冻结检测器未命中，不是 runtime import 完整性证明。
- attempt/preflight/tree/closure/result SHA-256 分别为
  `9107aaf0a82b0dc9538a46f09184958d2af22bd4d98c43490ad8d0004e1d01ee`、
  `c0ae093d5722c379614417bac0cb56887df550aa166d38475d2a7b4d4627fcd2`、
  `8b74807e5584297d0875e345ec47301208f78c48102cc90cd0646f94f2b20f0e`、
  `ef0b07fca57004c59d0bd659133e3cc7155705f26ed1d93f864c989b7eb78881`、
  `72dcebb4f8ca69518a8d86bc3982f5e2f5691faf7aafbb60351533dc132f7582`。
- 本终态没有解释 provider 控制流、提取 URL、建立 mapping 或请求 dataset host。
  它只允许另冻 hash-bound P0B source-semantic evidence contract；不自动授权
  P0B、P1、S0B census、payload、主线/App/生产或 safety claim。机器 locked
  result SHA-256 为
  `15f0bc4c96a1adea45aaa1ee1d1dddba4341f3390500147c165a4c343b523137`。

## 2026-08-02：HFTF D5-S0B-P0B provider semantic evidence 设计

- P0B 是 syntactic/source-evidence extractor，不是 provider resolver。它只允许从
  既有 P0A local object store 按 hash-bound 18 rows 全量复核 commit:path OID、
  object type/size、raw bytes 与 SHA-256；禁止 refetch、checkout、unresolved/
  unreachable source、外部 txt/config、dataset host 或 ZIP。
- 18 个 object receipts 必须在 AST extraction 前全部成立。随后只用冻结 encoding
  detector 与 Python AST 记录全部 string literals、import aliases、functions、
  calls、assignments 和 bounded expression graph；禁止 compile/import/exec/eval、
  模块初始化、CFG/dataflow/runtime reachability 或字符串模板求值。
- URL-like、单/多候选、docstring、logging/error/help/example、dead branch、
  assignment 与 call spelling 均不能升级为 provider/runtime authority。JoinedStr、
  BinOp、`%`、`.format`、`urljoin` 只保留结构。零 URL 或零 dynamic evidence
  也不证明 provider 缺失或 runtime closure 完整。
- LOCKED 只允许另冻 P0C provider-resolution contract；任何 cap/receipt/OID/hash/
  FETCH_HEAD/partial/implementation failure 为 INVALID。当前设计不授权 source blob
  read、P0B/P0C、dataset host、P1/S0B census、payload 或主线/App/生产/safety。
  机器设计 SHA-256 为
  `a15ed80b6f64f339b1a3c4ee6376de38ce50802e61094f92de51712db35b9324`。

## 2026-08-02：HFTF D5-S0B-P0B provider semantic evidence execution contract

- 执行者：violjjet。P0B 只复用 P0A local Git object store；18 个 hash-bound
  source blobs 必须按 closure 顺序逐个完成 commit:path OID、type、size、bytes 与
  SHA-256 收据，并且所有 object receipts 在首个 AST extraction 前完成。禁止新
  fetch/checkout/network、unresolved source、外部 txt/config、dataset host 或 ZIP。
- runtime 冻结为 CPython 3.11.9，并绑定 launcher/base executable、parser DLL、
  `ast.py` 与 `tokenize.py`。每个 AST occurrence（包括复用的 operator/context
  singleton）单独编号；child-first canonical shallow dump 构成 Merkle 式 node ID。
  validator 从每个 Module 根按 runtime `_fields` 与 list index 顺序重走完整 DFS，
  并核对每 path preorder/count/depth、双向 parent/child edge 与全部 node IDs。
- strings/calls/assignments/expressions 必须与 all-node AST exact one-to-one 覆盖；
  literal role/docstring、call/assignment links、function arguments 与 import aliases
  均从 canonical dump graph 回算。每个成功 blob 后立即执行 global AST/record caps；
  syntax NOT_EVALUABLE 使用 exact schema，携带 parse prefix AST/record cap usage，
  因而后续 SyntaxError 不能掩盖已发生的 cap overflow。
- focused tests `16/16`、项目标准 HFTF full suite `502/502`。最终科学与工程独立
  复审均为 `CLEAR`、0 blocker；formal canonical root 仍不存在。planner/test/
  叙述合同/机器合同 SHA-256 分别为
  `93a31d1f45b399d7e3fb43519e70c483322ade285fe627f3fe0cdec231c1abcd`、
  `cceb16f8587bd4f80e27655f4a97a8ed637ae701a25e9d2a976fb5498efbf038`、
  `8f2af2ee97ab50df049368e11f43d5b9eba57089a9453fc58775c0fb34cd3f52`、
  `dce5c3b07350cf52e0b2dcbe8e34868b8dcd734fe0118f1ebccba964dc782187`。
- 当前只授权精确提交推送；正式 P0B 必须在推送后再次通过 tracked/clean、
  `HEAD == origin/master`、合同/实现/测试 hash 与 root-absent gate，且只允许调用
  一次 canonical CLI。LOCKED 只允许另冻 P0C contract，不自动授权 P0C、host、
  P1/S0B census、payload、主线/App/生产或 safety claim。

## 2026-08-02：HFTF D5-S0B-P0B evidence-cap INVALID

- 合同提交 `2d8420dfab65310f682d3b1c53631855d0dcd029` 推送、formal
  tracked/clean/hash/HEAD/root-absent gate 通过后，只启动一次 canonical P0B。
  wrapper 的短观察窗超时后没有重跑；原进程继续运行并自行关闭为
  `D5_S0B_P0B_PROVIDER_SEMANTIC_EVIDENCE_INVALID_STOP`。
- 失败原因为 `ValueError: P0B total evidence JSON byte cap exceeded`，即完整
  AST evidence 超过冻结的 8 MiB cap。attempt/preflight/failure SHA-256 分别为
  `3fff64c50ebece11909aaa288e7ba599bf98821461769fff29ad5f3c031c8560`、
  `8f3260460f57677e3788b6df4e07d0a7c727d1b09850052df47a5ed700f4fa61`、
  `61dd13e081352410e6059c304b86db8470467b3498f069eef88953af99da8ec9`；
  canonical failure validator 接受，evidence/result 均不存在。
- 这是 evidence representation capacity failure，不是 provider/source 的正负结果。
  同 18-source semantic population 视为 consumed recovery population；禁止旧 root、
  旧合同、cap-only retry 或从进程内中间态推 provider 结论。没有 network/host/
  ZIP/payload/source execution/P0C。机器 INVALID result SHA-256 为
  `357ea359b7346253c8916d79809dd636e098c047063321fba2d02518fba00164`。

## 2026-08-02：HFTF D5-S0B-P0B.1 sharded evidence repair 设计

- P0B.1 只做内容无关的表示修复：每 source path 一个 manifest-index shard；
  canonical AST object 每 node 只存一次，expression 以 node receipt 引用。generic
  expression 省略文本但保留 segment SHA/UTF-8 length/encoding/span；string/call/
  assignment lexeme 仍 durable。claim ceiling 是 AST-semantic completeness，不是
  generic exact-lexeme parity；P0C 若需要原文必须另冻 source-reread 权限。
- attempt/preflight 后先在内存完成 18/18 receipts、parse/extract/serialize/caps。
  正常 NE 为 0 shard/index；LOCKED 才按 000..017 exclusive-fsync shards，再写
  index、result。任何 shard/NE/index/result 写入中断只可 INVALID，failure 绑定除
  自身外所有 present artifact 的 exact name/bytes/SHA，禁止 resume/reread。
- 每 shard cap 固定为 `max(1 MiB, 512 × P0A blob bytes)`；18 项预冻 cap 总和
  129690624 bytes，ordered cap manifest SHA-256 为
  `a7e3203057f17467dfe50e5671ab51fa578b832d439305764895a7c845f0a9f8`。
  科学与工程设计终审均 `CLEAR`；这只允许冻结 execution contract，不授权 source
  reread/P0B.1/P0C/network/host/payload/mainline/App/production/safety。机器设计
  JSON/MD SHA-256 分别为
  `6b2523091a967b2a64e2062c9314d1cc4d6eaf37b99de204f4fd9ccf953f5d9d`、
  `363bba692465f0cf7c7fed6b35cf14c43fd4312ec1a52bfa576d01e1f18b4408`。

## 2026-08-02：HFTF D5-S0B-P0B.1 fail-closed draft checkpoint

- 新增明确 `DRAFT_NOT_EXECUTABLE` 的 execution-contract schema/MD、planner skeleton
  与 tests。CLI 只读指定 contract，丢弃 output-root 参数，不访问 source、Git、
  network 或 canonical root；implementation/test receipts 必须保持 exact nested
  `UNBOUND_TODO`，关键 authorization 全 false，随后无条件拒绝执行。
- draft 固定 18 个 `shard_000..017.json`、node/expression/call/assignment/keyword
  schemas、LOCKED/NE/INVALID closed sets、NE durable receipts/prefix caps、18 项 cap
  manifest 与 failure partial binding。loader 递归拒绝 top-level/nested duplicate
  JSON keys，并核对 design/auth/caps/schemas/closed sets。
- focused tests `7/7`、HFTF full suite `509/509`。科学与工程复审均确认
  `CHECKPOINT CLEAR`，明确非 execution CLEAR；new formal root 不存在。planner/
  test/JSON/MD SHA-256 分别为
  `51cd7b7ee6678204e47e83377fef9b9f2024e527dbdf2fb655b29b5bf8788fda`、
  `05921dd875576a13397e8eb7ac55df1920c5347be69863c1eee8e1c634a66449`、
  `acf8b1239d12091870e940c3403d9e69fa945f63bc666f3fd39a59d949b6e70b`、
  `487c31c5f3121e5e1c8ac89baba1342c53dd500d85f96a59ec13fbfcfe9963c7`。

## 2026-08-02：HFTF D5-S0B-P0B.1 semantic/durability implementation checkpoint

- P0B.1 已有独立 sharded extractor；只复用 P0B 的纯 AST indexing/visitor kernel，
  不调用旧 P0B monolithic `extract_evidence`、execute、terminal validator 或 writer。
  exact 18-row 测试证明顺序为全部 blob/OID/type/size/SHA receipts 完成并冻结 set hash，
  再检测全部 18 个 encoding，最后才开始首次 parse；每个 blob 只读一次且无 network。
- 新增 fail-closed terminal validator：首个 shard 写入前验证 18 个完整序列化 payload 的
  exact schema、node ID、parent/edge、canonical DFS、same-shard references、one-to-one
  expression/string/call/assignment/function/import coverage、record/global/shard/aggregate
  caps；写完后从 durable shards 重验，index 写后再重验 hash/count/depth chain。
- 两个正常 result 均固定 consumed-source recovery role、8-key 全 false claim ceiling
  与 `p0c_execution_authorized_automatically=false`。只有 LOCKED 可要求另冻独立
  hash-bound P0C contract；NOT_EVALUABLE 必须 stop。syntax/encoding terminal 绑定
  exact failed manifest row、encoding state、18/18 object receipts 与 parse-prefix；
  dynamic reason 保持 0 source reads/receipts。
- 所有 control artifacts 有独立 1 MiB serialization cap；四个规定 write/fsync
  interruption 点均生成 raw-byte/hash-bound INVALID terminal，不 resume、不 source
  reread。raw execution core 无 validated gate 会拒绝，test gate 也拒绝 repo/canonical
  路径。focused tests `18/18`、绑定运行时下 HFTF full suite `520/520`；正式
  canonical P0B.1 root 仍不存在，本 checkpoint 不授权 source reread 或正式执行。

## 2026-08-02：HFTF D5-S0B-P0B.1 executable candidate

- P0B.1 已收缩为最小执行门：exact input/runtime/code/test hashes、单次 canonical
  root、durable terminal validation。复用的纯 AST helper 也已 hash-bind；P0A
  locked-result、closure 与 toolkit commit 完成交叉绑定。test-only source path
  仅接受 synthetic authority，formal source read 还要求 durable attempt/preflight。
- focused `18/18`、绑定运行时 HFTF full `520/520`，科学边界复审 `CLEAR`；
  executable contract 的 semantic self-hash 为
  `765946ab06afe8f8d6856b04a7ebd647036e6c74a169c0d7c59cd80e599599b0`。
  canonical root 仍不存在。后续不再扩展治理；推送后直接单次执行并按终态进入 P0C
  或停止。

## 2026-08-02：HFTF D5-P0C 回到 Development 科学实验

- P0B.1 已成功锁定 18 个 source shards；随后不再新建 P0C one-shot 合同。可修复
  resolver 直接确认 official Hugging Face revision
  `388faf9c800568cfc6828fa47e063f8369397eb3` 覆盖 catalog 的
  `198/198 parents`、`7,722/7,722 archive paths`，缺失为 0。
- 三个 outcome-open sentinel 的 metadata、12 路 pose 与 RGB/depth/seg indices
  完全对齐。27-anchor `.4/.8 s` pilot 在 2,555 个 common-known future cells 中
  观察到 54 个 risk state changes（30 onset、24 clearance）和 43 个 newly-known
  cells；pose-depth 重投影 pair-median relative error 为 `.00068–.00144`。
- 当前只结论为 aligned geometry teacher 可构造且 future label 在 Development
  窗口非完全冗余。未训练 student，也没有系统、主线、App 或 safety 结论。后续直接
  扩展 environment-clustered Development corpus；工程故障在关键 held-out outcome
  未观察前允许修复重跑，不烧毁 source。

## 2026-08-02：HFTF D5 environment expansion 与空间结构增量

- outcome-open expansion 增加 `WaterMillDay` diagnostic counterpart 和六个固定
  哈希顺序的未使用 environments，共 231 samples / 518 PNG；samples SHA-256
  `fad64102b9c1bcbeb5a93662f0f8c5acb30ea615668daf22f4d851ac3f958049`。
  原 staged-history 三个 checkpoint 在七环境上均未超过 pooled single。
- pooled head 在 expansion 上 macro F1 `0.3444`，且 head label 的 AUROC 约
  `0.491/0.472`。保留水平方向轴的 directional head 参数更少，在七环境上达到
  `0.3905`、6/7 environments 胜出，但在原两环境 dev 上较弱，因此只进入跨折复核。
- 15 environments / 495 samples 的三折 environment-held-out Development 中，
  directional 相对 pooled 的 environment-macro F1 delta 为
  `+0.0058/+0.0112/+0.0806`，折均 `+0.0326`；15 环境 11 胜 4 负。折均
  aggregate macro/micro/AUROC/AP/FPR delta 为
  `+0.0327/+0.0411/+0.0459/+0.0587/-0.0098`。最差 `GreatMarsh`
  为 `-0.1788`，不声称每环境支配。
- 当前 representation 终态为
  `DIRECTIONAL_SPATIAL_STRUCTURE_CROSS_ENVIRONMENT_INCREMENT_SUPPORTED_IN_DEVELOPMENT`。
  它将 directional single 设为 HFTF Development reference，不建立事件级系统效用、
  主线晋级、App 或 safety 结论。

## 2026-08-02：HFTF D5 无对齐 history fusion 负终态

- 原 single 使用五份重复当前帧，导致 5-tap temporal convolution 只约束权重之和；
  换成真实 history 会先产生未识别的时间权重扰动。新增 current baseline 加
  zero-initialized temporal residual，使三折 epoch 0 与 directional single 精确相同。
- joint history 相对 directional single 三折 delta 为
  `-0.0140/-0.0123/+0.0017`。zero-init residual 全模型微调三折均选 epoch 0；
  冻结基线只训练 2,304 个 1×1 residual 参数也三折均选 epoch 0；20,736 参数的
  3×3 spatial residual 仅 fold 2 为 `+0.0029`，fold 0/1 仍选 epoch 0。
- 当前精确负终态为 `UNALIGNED_HISTORY_FUSION_INCREMENT_NOT_SUPPORTED`，只关闭
  当前 joint/逐点/局部 spatial、无显式对齐的 history fusion。它不证明历史 RGB
  没有信息；只有显式 feature alignment、flow 或 ego-motion compensation 才值得
  重开。当前停止更多无对齐结构和学习率搜索。
- 结果层级改为显式分离：teacher、representation、decision kernel、research
  mainline、App/safety。后一层未完成只限制 claim，不抹掉前一层正结果；路径、parser、
  network、serialization 和 interruption 失败属于可修复工程故障，不产生科学终态。

## 2026-08-02：HFTF D5 directional paired multi-seed replication

- 在同一三折上增加 paired seed 29/43，每个 seed 同时重训 pooled/directional。
  9 个 fold×seed 单元的 environment-macro F1 为 8 胜 1 负，mean/median delta
  `+0.0351/+0.0385`，range `-0.0046..+0.0806`。seed 17/29/43 的三折
  mean delta 分别为 `+0.0326/+0.0424/+0.0304`。
- aggregate macro/micro/AUROC/AP mean delta 为
  `+0.0357/+0.0375/+0.0395/+0.0448`，各 8/9 改善；45 个
  environment×seed 比较为 30 胜、15 负。当前 representation 终态提升为
  `DIRECTIONAL_SPATIAL_STRUCTURE_MULTI_SEED_CROSS_ENVIRONMENT_INCREMENT_SUPPORTED_IN_DEVELOPMENT`。
- threshold behavior 尚不稳健：recall mean delta `+0.0797`，FPR mean delta
  `+0.0229`，FPR 6/9 变差。唯一 environment-macro 反向单元 seed43/fold1 为
  `-0.0046`，其 aggregate macro/micro 仍为 `+0.0153/+0.0200`。
- `GreatMarsh` 的精确 shift 是 future body/head positive rate：fold0 train
  `48.9%/15.3%`，GreatMarsh `93.1%/0.97%`。directional seed17 显著降低 FPR，
  但把 body recall 压到约 `0.27`。下一步先做 train-side height-aware calibration，
  以 dev folds 分别检查 body recall 与 head false-alert tradeoff，再进入事件级
  decision kernel；不把 F1/排序正结果写成系统提醒改善。

## 2026-08-02：HFTF D5 calibration 与 synthetic event transfer

- loss-derived `w/(1+w)` threshold 显著降低 head FPR，但几乎清空 head recall；
  seed17 三折 macro F1 全降。按 10 个 train environments 的 environment-macro
  F1 选择 horizon×height threshold，也只改善 fold1，fold0/2 下降。停止
  post-processing threshold search。
- 新增 synthetic teacher-derived continuous-event proxy：每条
  environment×horizon×height×direction lane 中，任一 teacher-known risk cell
  为 positive，六个 distance cells 全 known 且均非风险才为 negative，其余
  unknown；candidate 需同时 predicted-known 与 predicted-risk。它只评估连续
  hit/miss、negative false-active、clearance，不是 human truth、route 或 App kernel。
- 9 个 paired fold×seed 单元的 event recall delta 4 正 5 负，mean/median
  `+0.0102/-0.0069`；false-active rate 3 正 6 负，mean/median
  `+0.0207/-0.0182`；clearance median 为 0。三个 folds 完整负 exposures 只有
  `55/114/187` lane-frames，seed 重复不增加 truth exposure。
- height 分解显示 body recall/false-active mean delta
  `-0.0482/-0.0565`，head recall/false-active
  `+0.0820/+0.1544`。directional 的 cell-level 正结果主要重分配 body/head
  行为，没有稳定穿过最小事件代理。终态为
  `UNCALIBRATED_SYNTHETIC_EVENT_TRANSFER_NOT_SUPPORTED`。
- 下一步不再调阈值；修改训练目标或采样，分别控制 body critical recall 与 head
  false-active，再运行同一个 proxy。只有代理稳定改善才进入真实 parent-event
  decision kernel。

## 2026-08-02：HFTF D5 known-loss intervention 与研究边界纠偏

- 先审计 known gate：risk-only body 激活仍高，主要召回损失来自
  `predicted-known AND predicted-risk`。新增 train-only known positive
  reweighting，保留同一数据、directional 架构、0.5 threshold 和 synthetic event
  proxy。
- 完全 inverse-frequency balanced 的 3 seeds × 3 folds 相对 directional
  reference，environment-macro F1 mean delta 仅 `+0.0010`；event/body recall
  mean delta 为 `+0.0941/+0.1492`，但 false-active/body false-active 同时为
  `+0.0435/+0.0578`，没有建立事件级改进。
- 有界追加 seed17 三折 sqrt-balanced。event recall delta 为
  `-0.0688/+0.0928/+0.0759`，false-active 三折全部恶化
  `+0.0182/+0.0263/+0.0214`，clearance 为
  `0/-0.0714/-0.0571`。不扩 seed29/43，不继续调标量权重，终态为
  `KNOWN_LOSS_REWEIGHTING_EVENT_INCREMENT_NOT_SUPPORTED`。
- 该终态是有效算法权衡负结果，只关闭 plain/balanced/sqrt-balanced known 正类
  重加权；不抹掉 directional representation 正结果，也不关闭显式
  observability/alert 解耦。
- 研究状态重新分为科学负结果、可修复工程无效和主张边界。Windows 长路径、
  scanner `OSError`、manifest/parser、单文件尺寸、网络与 interruption 只能触发
  修复重跑，不能关闭科学问题或烧毁 cohort。teacher、representation、
  decision-kernel、research-mainline、App/safety 是逐层证据；后一层未完成只限制
  主张，不把前一层正结果改写成失败。

## 2026-08-02：HFTF D5 height-temporal selective decision kernel

- 静态移除/放宽 predicted-known 硬门在 9/9 paired 单元提高召回，却一致增加
  false-active，说明 observability/alert 解耦还需要因果时间选择性。
- 新增可审计 decision-kernel evaluator。v1 的 body 使用 risk≥0.5 连续 3 个
  anchor，不让 predicted-known 单帧否决风险；head 使用 known-and-risk 连续 2 个
  anchor，但 risk≥0.8 可立即高置信 override。
- directional checkpoint 上，v1 相对 hard-known-and-risk 的 event recall mean
  delta `+0.1705`，8/9 改善；false-active rate `-0.0245`，7/9 改善；
  response-delay median 9/9 不变。body recall 9/9 提高、mean `+0.3569`；
  head recall mean `+0.0038`，head false-active mean `-0.2286`。Development
  正结果为
  `HEIGHT_TEMPORAL_SELECTIVE_DECISION_KERNEL_SIGNAL_SUPPORTED_IN_DEVELOPMENT`。
- 边界：clearance mean delta `-0.0503`，false-alert event count mean
  `+0.78`，负 exposure 每折仅 `55/114/187` 且 seed 不增加 truth exposure。
  v1 是候选，不是 human-event 或系统效用证据。
- 同一 v1 下 directional 相对 pooled 的 event recall/false-active mean delta
  只有 `+0.0144/-0.0006`，分别 5/4 与 4/3/2；表示增量仍未稳健穿过事件层。
  当前保留 directional 为 representation reference、v1 为 decision-kernel
  candidate，下一步只处理 risk-coverage、clearance 和 false-alert fragmentation。

## 2026-08-02：HFTF D5 spatial-support v2 与 selective event transfer

- v1 的 body lane 可被任一 distance cell 触发。拒绝使用最大 alert duration 等会在
  持续危险中静音的指标投机规则；v2 要求至少 `3/6` cells risk≥0.5，或任一 cell
  risk≥0.8，再连续 3 个 anchor。head 保持 v1。
- directional checkpoint 上，v2 相对 hard 的 event recall/false-active/clearance
  mean delta 为 `+0.1352/-0.1091/+0.0566`；recall 8/9 改善，false-active
  7/9 改善，body recall 9/9 提高。false-alert event count mean `-0.78`。
- 相对 v1，v2 的 false-active 8/9 下降或不变、clearance 9/9 提高或不变、
  false-alert events 9/9 减少或不变；event recall 9/9 小幅回退但相对 hard
  仍保持多数正向。v2 取代 v1，终态为
  `HEIGHT_SPATIOTEMPORAL_SELECTIVE_DECISION_KERNEL_SIGNAL_SUPPORTED_IN_DEVELOPMENT`。
- 同一 v2 下 directional 相对 pooled 的 event recall/false-active/clearance mean
  delta 为 `+0.0810/-0.0739/+0.1958`；event recall 8/9 正，body recall 9/9
  正，body false-active 8/9 降低或不变。首次建立
  `DIRECTIONAL_SPATIAL_STRUCTURE_SELECTIVE_EVENT_TRANSFER_SIGNAL_SUPPORTED_IN_DEVELOPMENT`。
- 该正结果保留在 Development 层。v2 在这些 outcome-open folds 上选择，head
  false-active mean 仍 `+0.0341`，完整 negative exposure 每折只有
  `55/114/187`。下一步停止当前-fold kernel search，直接做 outcome-unseen
  TartanGround environment transfer。

## 2026-08-02：HFTF D5 outcome-unseen TartanGround transfer

- v2 固定后，排除此前 15 个 environments，从具有 metadata/front RGB/front
  depth 的未使用 P1000 parents 中按
  `sha256(HFTF_D5_OUTCOME_UNSEEN_TRANSFER_V0:environment)` 升序选择 6 个：
  `ModularNeighborhoodIntExt / Fantasy / GothicIsland / OldIndustrialCity /
  Hospital / OldTownFall`。共 198 transfer samples、444 PNG；selected 与 used
  environment sets 无交集，全部路径、PNG 解码和 teacher labels 验证通过。
- corpus 中有 266 个 positive lane events、1,608 个 positive lane frames、
  130 个 complete negative lane frames 和 20 个 clearance-eligible events；
  六环境均同时有正、负机会。engineering failure 可修复重跑，不是 one-shot。
- 不经 kernel 的 field comparison 中，directional 相对 pooled 的
  environment-macro F1 mean delta `+0.0473`，7/9 单元为正，54 个 environment
  cells 为 37 胜 17 负；但裸 threshold FPR mean `+0.1420` 且 9/9 恶化，
  表示正结果不能直接当作提醒行为改善。
- 固定 v2 下，directional 相对 pooled 的 event recall/false-active/clearance
  mean delta 为 `+0.1809/-0.0727/+0.0444`；recall 9/9 正、false-active
  7/9 改善、clearance 6/9 改善、response-delay median 9/9 不变。终态提升为
  `DIRECTIONAL_SPATIAL_STRUCTURE_SELECTIVE_EVENT_TRANSFER_REPLICATED_ON_OUTCOME_UNSEEN_TARTANGROUND_ENVIRONMENTS_IN_DEVELOPMENT`。
- 反例不隐藏：false-alert event count mean `+3.56`，主要是 head 短事件碎片；
  54 个 environment cells 的 false-active 等权 mean delta 为 `+0.0129`，
  31 恶化、16 改善、7 不变。v2 相对 directional hard 虽改善 recall、clearance
  与 fragmentation，却使 false-active mean `+0.0359`。因此当前只建立合成新环境
  selective-event 迁移，不建立逐环境 guardrail、human-event、主线、App 或 safety
  主张。下一步进入真实 parent-event cohort 或预先固定
  environment-balanced head/negative fragmentation guardrail。

## 2026-08-02：HFTF D6 SANPO real parent-event transfer

- 直接复用已消费的 RISKSEG-R0 30-session / 1,920-frame SANPO event view：
  16 个人工审阅正事件、14 个负事件，同口径当前 YOLO reference 为
  `13/16 hits、6/14 false alerts、5/16 cleared`。不把 consumed cohort 重新包装成
  fresh/held-out，也不新增协议 ceremony。
- HFTF checkpoints 从未用 SANPO 训练。outcome 前固定 adapter：current-only
  single input、v2 原 5 Hz confirmation、中央 direction indices 2/3
  （`-15°..+15°`）、near/far × body/head 任一 lane active 即提醒，5 Hz state
  causal hold 到原 10 Hz timeline；评分完全复用 RISKSEG event hit/negative
  false-alert/passed-clearance 口径。
- 9 个 directional checkpoints 全部 `16/16 hits、0 critical misses`，
  blocking/boundary 各 8/8；真实事件 recall positive terminal 为
  `REAL_EVENT_RECALL_SIGNAL_SUPPORTED_ACROSS_NINE_CHECKPOINTS_IN_DEVELOPMENT`。
- 同一模型 false-alert 为 `13–14/14`、cleared 为 `0–2/16`，all-frame active
  fraction mean `87.08%`；0/9 对当前 YOLO 形成 hits/false-alert/cleared
  Pareto 支配。效用终态为
  `FIXED_KERNEL_REAL_EVENT_SPECIFICITY_AND_CLEARANCE_NOT_SUPPORTED`。
- paired pooled 中，directional hit count 4/9 更高、5/9 相同、0/9 更低，
  mean `+2.22 events`；false-alert 2 改善/3 恶化/4 相同、mean `+0.33`，
  cleared 2 提高/3 降低/4 相同、mean `-1.89`。因此
  `DIRECTIONAL_REAL_EVENT_PARETO_INCREMENT_NOT_SUPPORTED`；它不抹掉真实 recall
  正信号或合成表示正结果。
- 下一步不在同一 outcomes 上搜索绝对 threshold。先做跨 9 checkpoints 的
  central-minus-lateral direction-profile 诊断，检验前方侵入与 parallel-curb
  是否有同向相对结构；有则进入 session-held-out weak event calibration，无则转向
  真实负例/actionability relation supervision。

## 2026-08-02：HFTF D6 central-vs-lateral profile diagnostic

- 诊断不改变 alert output、不搜索 threshold。对 9 个 directional checkpoints，
  分别把每个 positive alertable interval 和完整 negative event 压成 event-level
  median，比较 `risk_mean / risk_max / body_k3 / head_known_risk / known_mean`
  的中央绝对值、中央减侧向均值与中央峰值减侧向峰值。
- 最佳绝对中央 `risk_mean` 的 all-negative AUC mean/median 为
  `0.5893/0.6071`，parallel-curb 为 `0.5327/0.5625`；只有弱排序信号。
- `risk_mean central-minus-lateral` 的 all-negative AUC 为 `0.5501`，
  parallel-curb 为 `0.4990`；body_k3 相对 profile 为 `0.5060/0.4772`，
  head known-risk 相对 profile 为 `0.5025/0.4633`。关键关系分离接近或低于随机。
- 终态为 `CENTRAL_VS_LATERAL_ACTIONABILITY_PROFILE_NOT_SUPPORTED`。它不否定真实
  16/16 recall signal；它把高 recall 归因为普遍高激活，而不是已经学会
  “前方侵入 vs 平行但不阻塞”。
- 下一步停止 absolute threshold、手工 relative formula 和更多 v2 搜索。固定 HFTF
  backbone，用 30 consumed sessions 做严格 source-session-held-out、低容量 weak
  actionability relation head；若仍不能同时守住 recall/specificity/clearance，再
  进入真实 RGB backbone fine-tune。

## 2026-08-02：HFTF D6 source-session-held-out weak relation head

- 30 events 按 bucket 内固定 hash 分 5 folds。每折 test sessions 不参与标准化、
  拟合或 threshold；train labels 只取 positive alertable、positive passed 与完整
  negatives 的 5 Hz frames，transition gap 排除。输入为五类 HFTF profile ×
  6 directions 共 30 features；event 与 class 均衡，L2 logistic、0.5 threshold、
  两步因果确认全部固定。
- 9 个 directional backbones 的 out-of-fold hits 为 `11–16/16`、mean `13.22`；
  false-alert events `8–13/14`、mean `11.22`；cleared `4–11/16`、mean `7.22`。
- 相对 fixed v2，false alerts 9/9 减少、mean `-2.56`，cleared 9/9 增加、mean
  `+6.67`；guardrail 正信号为
  `WEAK_RELATION_HEAD_SPECIFICITY_CLEARANCE_SIGNAL_SUPPORTED_IN_DEVELOPMENT`。
- hit count 同时 mean `-2.78`，8/9 下降。0/9 checkpoint 同时非劣于当前 YOLO
  `13 hits / 6 false alerts / 5 cleared`，终态为
  `WEAK_RELATION_HEAD_REAL_EVENT_PARETO_INCREMENT_NOT_SUPPORTED`。
- 关系监督不是完全无效，但 output field 压缩后的 30 features 不足以兼顾 recall 与
  specificity。停止搜索 L2/threshold/confirmation/fold；保持相同 held-out 口径，
  下一候选把低容量 relation head 接到固定 encoder spatial feature map，之后才考虑
  解冻 backbone。

## 2026-08-02：HFTF D6 fixed-encoder spatial relation head

- 保持 weak relation head 的 5-fold source-session split、labels、event/class
  weights、0.5 threshold 与两步 5 Hz 确认不变；固定 HFTF encoder/backbone，只把
  输入前移到 pointwise fused `128×3×6` spatial feature。test sessions 不参与
  标准化或拟合；L2 strength 预先固定为 1.0。
- 9 个 directional backbones 的 OOF hits 为 `12–14/16`、mean `13.00`；
  false-alert events `7–11/14`、mean `9.00`；cleared `6–12/16`、mean `9.00`。
- 相对 output-field head，false alerts 9/9 减少、mean `-2.22`；cleared 6/9
  增加、1/9 同、mean `+1.78`；hits mean `-0.22`。模型层正终态为
  `FIXED_ENCODER_SPATIAL_RELATION_HEAD_OVER_OUTPUT_FIELD_GUARDRAIL_INCREMENT_SUPPORTED_IN_DEVELOPMENT`。
- 相对当前 YOLO，mean hits 相同、cleared `+4`，但 false alerts `+3` 且 9/9
  更差；0/9 Pareto，系统比较终态为
  `FIXED_ENCODER_SPATIAL_RELATION_HEAD_REAL_EVENT_PARETO_INCREMENT_NOT_SUPPORTED`。
- 不以系统负终态撤销表示层正结果。同一 consumed cohort 停止 grid/L2/threshold/
  confirmation 搜索；下一候选固定空间头，只解冻靠近输出端的最小 backbone 子集。

## 2026-08-02：HFTF D6 complementarity 与 fusion

- 9 个空间头平均补回 YOLO `2.56/3` misses，同时丢失 `2.56` 个 YOLO hits；
  event-level OR mean 为 `15.56 hits / 11 false alerts`，AND 为
  `10.44 hits / 4 false alerts`。保留
  `YOLO_HFTF_EVENT_COMPLEMENTARITY_SIGNAL_SUPPORTED_IN_DEVELOPMENT`，但简单
  OR/AND 不构成候选 policy。
- 固定 30 HFTF profiles + 7 causal-200ms YOLO features 的静态融合，复用相同
  source-held-out folds/weights/L2/threshold/confirmation。9-checkpoint OOF
  mean 为 `12.89 hits / 9.78 false alerts / 6.89 cleared`；0/9 YOLO Pareto，
  终态 `STATIC_YOLO_HFTF_FUSION_PARETO_INCREMENT_NOT_SUPPORTED`。
- 2,305-parameter spatial head 的 train loss 已约 0.002。rank-2/293-parameter
  canary 为 `13/10/7`，弱于同 backbone 完整空间头 `13/8/7`；终态
  `LOW_RANK_SPATIAL_RELATION_HEAD_CANARY_INCREMENT_NOT_SUPPORTED`。
- current + 1 s delta + 1 s prefix mean causal fusion canary 为 `11/9/9`；
  clearance 改善但 recall 降到 68.75%，终态
  `CAUSAL_TRANSITION_FUSION_CANARY_INCREMENT_NOT_SUPPORTED`。
- 当前 30-event cohort 对更多 head 结构已 information-limited。停止 rank/history/
  fusion-feature/L2/threshold 变化；下一步扩充与这 30 sessions 隔离的真实关系监督，
  未新增监督前不解冻 backbone。

## 2026-08-02：HFTF D6 cross-source provisional relation transfer

- 训练只使用与 SANPO evaluation sessions 隔离的外部 provisional supervision：
  初始为 14 episodes、7 sources、49 frames（8 alert / 6 no-alert）；SANPO labels
  不参与 fit。30-feature output-field head 的 9-checkpoint mean 为
  `14.44 hits / 13.33 false alerts / 3.78 cleared`，7/9 hits 高于 YOLO、2/9
  相同；保留
  `CROSS_SOURCE_PROVISIONAL_RELATION_RECALL_SIGNAL_SUPPORTED_IN_DEVELOPMENT`，
  但 specificity/clearance 不成立。
- 加入两个经复核 normal-passage source episodes 后，训练 inventory 为
  16 episodes、9 sources、611 frames、8/8 类平衡。output-field canary
  `15/14/3→13/14/5`，未改善误报。
- 相同外部训练集改用固定 encoder `128×3×6` spatial feature，9-checkpoint
  mean 为 `12.33 hits / 8.78 false alerts / 7.22 cleared`。相对外部
  output-field head，false alerts 9/9 改善、mean `-4.56`；cleared 8/9 改善、
  mean `+3.44`；hits mean `-2.11`。终态为
  `CROSS_SOURCE_SPATIAL_RELATION_OVER_OUTPUT_FIELD_GUARDRAIL_INCREMENT_SUPPORTED_IN_DEVELOPMENT`。
- consumed Development threshold sweep `0.30–0.80` 没有稳健 YOLO Pareto；
  threshold 0.35 的 mean 为 `13.00/9.22/6.44`。0/9 checkpoints 超过 YOLO，
  系统终态为
  `CROSS_SOURCE_SPATIAL_RELATION_REAL_EVENT_PARETO_INCREMENT_NOT_SUPPORTED`。
- 只执行一次 AI-abstained/quarantined parallel-curb weak-negative canary。加入
  82 个新增去重帧后 seed17/fold0 为 `12/8/7`，相对 normal-negative spatial
  `14/10/7` 用召回换误报；threshold 0.30 为 `14/9/7`。终态
  `QUARANTINED_PARALLEL_CURB_WEAK_NEGATIVE_CANARY_INCREMENT_NOT_SUPPORTED`。
  quarantine 不升级为真值或训练 reference。下一步需要新的、人工确认且 source
  isolated 的 parallel-curb / obstacle-approach 关系监督，不再搜索当前 cohort 的
  threshold、L2、集成或更多低置信负例。
# 2026-08-02 — HFTF D6 多源关系监督 canary

- 复用 r789 的 16 个人工 actionability events，按
  `clear/context → intervention → clear` 状态转移切成 28 个 public-video
  segments，并以 2 Hz 从 11 个本地源视频直接解码。
- 与既有 provisional supervision 合并后固定为
  `42 segments / 18 sources / 485 frames`；不更换 backbone、L2 或确认逻辑。
- seed17/fold0 SANPO canary 为 `12 hits / 11 false alerts / 9 cleared`，
  未超过 reviewed-normal-negative reference `14/10/7`，完整阈值曲线也没有
  YOLO Pareto 点，因此没有扩到 9 checkpoints。
- public-video 逐来源留一诊断对 intervention 的 frame/segment recall 都为
  `0`，balanced accuracy 分别为 `0.4962/0.5000`。结论是 fixed HFTF spatial
  representation 缺少跨来源 actionability relation 可迁移性，而不是 parser、
  path、scanner 或 output-size 工程 invalid。
- 即使使用 held-out source 的全部人工 no-alert segments 构造 episode-balanced
  baseline oracle，intervention frame/segment recall 仍为 `0`，frame balanced
  accuracy 为 `0.4899`；因此 source-centering fixed-feature rescue 也不支持。
- 固定 `delta + abs(delta)` 的 13,137-parameter 3×6 convolutional relation
  encoder 仍为 `0` intervention recall。再加入 30 个 consumed SANPO sources、
  46 个 phase episodes、711 帧作训练 support 后，public-video held-out frame BA
  反降到 `0.4394`；各 fold train loss 接近 `0`，确认是记忆而非迁移。
- paired-RGB backbone canary 解冻 `encoder[9:] + pointwise` 的 810,472 个参数，
  只评价 Bangkok/Ulm/Edmonton 三个 intervention-bearing held-out sources。早期
  CUDA adaptive-pool 两次不一致被归类为 engineering invalid 并允许修复重跑；
  deterministic bilinear 版本 repeat A/B 逐分数完全一致。有效结果为 frame
  alert recall `0`、AUROC `0.5034`、segment alert recall `0`、AUROC `0.3377`。
- 另用 TartanGround current-body 中央 clear/risk 配对训练 6 个 parents，并直接
  迁移到 2 个 outcome-unseen parents：frame BA/AUROC `0.7098/0.7124`，
  episode BA/AUROC 均为 `1.0`。因此配对任务的 synthetic learnability 是正结果；
  同一状态直接 synthetic→public 的 frame alert recall/AUROC 只有
  `0.025/0.4053`，只否定跨真实域迁移。
- SANPO-only 46 episodes / 30 sources / 711 frames 训练到 Bangkok/Ulm/Edmonton
  的 18 segments / 272 frames，public 参数更新帧数为 0。两次运行除时间戳外逐字段
  一致；pooled frame/episode AUROC 为 `0.5811/0.5844`。逐来源 Edmonton 为
  `0.7958/0.75`、Bangkok `0.5527/0.50`、Ulm `0.0326/0`，source-macro 为
  `0.4604/0.4167`。保留 Edmonton source-local 正信号，不升级为 source-general。
- 固定的 TartanGround→SANPO→public 课程把 pooled frame AUROC 降到 `0.4920`，
  source-macro episode AUROC 降到 `0.3278`，无增量。下一候选需在 backbone 内
  联合比较 frame pair，或直接学习人体包络短时未来风险场；不再继续
  encode-then-difference 的预训练、tail/head 或 threshold 搜索。
- 新增 28,313-parameter early joint-pair stem，直接联合编码
  `current/baseline/signed RGB delta/abs delta`，冻结 HFTF current context，
  SANPO-only 训练并零 public 参数更新。两次运行除时间戳外逐字段一致；相对
  encode-then-difference，pooled frame alert recall 从 `0.275` 升到 `0.375`，
  no-alert recall 保持 `0.8621`，BA/AUROC 从 `0.5685/0.5811` 升到
  `0.6185/0.6978`。但 Bangkok/Ulm/Edmonton frame AUROC 为
  `0.1836/0.2582/0.8134`，source-macro `0.4184`。保留 pooled frame 与 Edmonton
  局部表示增量，不建立 source-general transfer；下一实验把 early interaction
  转入 structured HFTF cell/lane future-risk teacher task。
- 保留此前 spatial-over-output-field 的正结果；只关闭“增加关系监督即可救固定
  backbone”的窄假设，也关闭当前 paired-RGB tail fine-tune recipe。下一步必须
  改变 pair interaction 或风险场表示，并先通过 source-heldout actionability
  recall。

# 2026-08-02 — HFTF D7 public-real review pilot

- 为 52,216 个候选窗口保留当前 `NOT_COMPLETE` 终态：候选目标已达到，
  但 admitted parent events 为 `0/10000`，`HOLD_ROLE_REVIEW` 的两个 ancestry
  冲突仍未授权 split、training 或 Confirmation。
- 用 public extracted EgoWalk RGB 固化 5 个 model-blind 窗口，并为
  RGB A/B/C、source-native geometry、counterexample 建立彼此隔离的输入包；
  5 个角色各完成 5 条独立记录。RGB A/B/C 的 5/5 bucket 观察一致，但严格
  phase contract 未满足；geometry 全部因 pose-only、缺少 obstacle geometry/
  depth/tracks 而为 `NOT_EVALUABLE`。
- 最终 adjudicator 消费全部 5 类 raw review 后输出 `5 NOT_EVALUABLE / 0 ADMITTED`。
  没有把 RGB negative 观察、缺失 geometry 或候选发现信号升级为 event truth；
  training/Confirmation/production authority 仍为 false。
- 新增 `materialize_review_bundle.py`、`ingest_review_outputs.py`、
  `materialize_adjudication_bundle.py`、`ingest_adjudications.py` 及对应防泄漏、
  phase、原子合并测试；每次合并均保留 backup 和 sha256 receipt。
- SANPO-Real 另完成一个明确 session/camera/view 的 bounded canary：20 RGB、
  20 depth、20 segmentation mask、2 pose CSV，对 62 个公开 GCS 对象逐一做
  provider MD5 校验；由于该 pose CSV 没有时间戳，暂不把这个媒体 canary 擅自
  变成带 phase contract 的 D7 event candidate，保留为 source-intake evidence。

# 2026-08-02 — HFTF D7 public-real source expansion receipts

- SANPO-Real canary expanded to 60 RGB/depth/mask frames plus session intrinsics
  and raw/fixed pose CSVs.  The official 15 FPS value is recorded only as
  `DERIVED_RELATIVE_NOMINAL`; `timestamp_ns` remains null, capture timestamps
  are not authoritative, and pose-row/frame binding remains `NOT_EVALUABLE`.
- THÖR-MAGNI public Zenodo ZIP central-directory inspection fetched 135,487
  bytes of metadata for a 22,259,767,649-byte archive: 122 videos, 581 point
  clouds, and 185 tabular/JSON members were inventoried without full-archive
  download.
- A bounded six-member THÖR-MAGNI canary materialized and CRC/SHA-256 verified
  151,725,897 bytes, including one scene video, one synchronized scenario CSV,
  synchronization metadata, Tobii raw eye-tracking, goals, and camera
  intrinsics.  QTM `Frame`/`Time` windows (100 Hz, 400 QTM rows/window) retain
  24,057 source rows including one duplicate QTM frame, 6,104 unique scene
  frames, and 60 four-second source windows for one `Visitors-Alone` Pupil
  run; only 58 have complete SceneFNr coverage and 3 have complete camera
  centroid coverage.  These remain intake-only and
  `NOT_EVALUABLE`; no top-level event label or authority was created.
- 时间：2026-08-03（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D28
  THOR-MAGNI kinematic field distillation。D27 current-static 与
  history-kinematic distance fields 分别监督等容量 current/history RGB students；
  530 anchors、19 sources、5 paired folds、10 training runs 均完整，stderr
  为空，非工程 invalid。history-current source-macro AUROC/AP 为
  `-0.02350/-0.02159`（均 2/5 正折），safest-choice `+0.01347`
  （2/5），pooled AUROC/AP `-0.01708/-0.00856`；冻结 gate 2/12，
  终态 `D28_THOR_MAGNI_KINEMATIC_FIELD_DISTILLATION_INCREMENT_NOT_SUPPORTED`。
  fold2/4 三项真实未来指标同时为正，但不足以建立 source-general increment；
  teacher MAE delta `+0.03538 m` 通过非劣且与 future ranking 不稳定对齐。保留
  D27 强 information ceiling，只关闭 direct whole-frame teacher-distillation
  recipe；下一步转向显式 object-centric detection/correspondence/velocity
  bottleneck。report SHA-256
  `2f359f12b04a15fa9de7f109e87231bc7c738de2dac95fb134762f18e119e29c`。
- 时间：2026-08-03（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D29
  THOR-MAGNI object-slot motion residual。冻结 YOLO11n current-person boxes 与
  box 内 backward-RAFT 形成 8×34 slots，530 anchors 中 393 有检测，
  coverage `74.15%`，1,161 selected slots 的 mean warp-valid `90.95%`。
  14,104 参数 paired DeepSets 在五折完整训练；history-current source-macro
  AUROC/AP `-0.04125/-0.02507`（均 2/5 正折），safest-choice `+0.00560`
  （2/5），pooled AUROC/AP `-0.04875/-0.02593`。teacher MAE delta
  `+0.14027m` 与 monotonicity 通过，其余 gate 失败，终态
  `D29_THOR_MAGNI_OBJECT_SLOT_MOTION_RESIDUAL_INCREMENT_NOT_SUPPORTED`。
  这关闭 low-resolution current-box + within-box-flow residual recipe，不撤销
  D27 information ceiling；也不支持只用全分辨率 detector 重跑。下一步先检验
  2D box 与 source-native bearing/distance 的 measurement correspondence。
  object-slot/report SHA-256 分别为
  `aa9d0f28b1e050105086fee3078002862fd0d21d06e5bd4aa12ecc950ec451f7` /
  `22b910c1500beb7683241ea69fc0f5a3a5fa88747ebed06d28a1a10100ba1206`。
- 时间：2026-08-03（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D30
  THOR-MAGNI current box-to-world measurement diagnostic；不训练、不读取 future
  outcome，并明确只评价 `Helmet_*` Visitor/Carrier，排除 DARKO/LO1 非 person。
  289 anchors 同时有 box/visible person，501 assigned、310 accepted，accepted
  fraction `61.88%`。source-macro box-x/bearing Pearson `0.7089`、bearing MAE
  `14.12°`，17/19 sources 可评，均通过；pooled distance Spearman `0.4246`
  且 5/5 folds 为正。nearest-body coverage `46.51%`、source-macro distance
  Spearman `0.2867` 与 anchor opportunity 未过，整体 5/8，终态
  `D30_THOR_MAGNI_BOX_WORLD_MEASUREMENT_RELATION_NOT_SUPPORTED`。保留 bearing
  measurement 与跨折 distance-rank 正信号，不升级为完整 state estimator。下一步
  只做 hash-bound 原视频 full-resolution current measurement replication，完全
  复用 D30 assignment/gates。report SHA-256
  `245e3625f8ea80cecdb629be9c6cd5498433ac3ae6fa58875488c95f80604c95`。
- 时间：2026-08-03（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D31
  THOR-MAGNI full-resolution current measurement replication。19/19 原视频 hash
  校验、530 anchors 解码完整；相对 D30，person detection coverage
  `74.15%→87.36%`、共同 anchors `289→322`、accepted fraction
  `61.88%→67.74%`、nearest coverage `46.51%→58.43%`，source-macro
  bearing Pearson `0.7089→0.7847`、MAE `14.12°→11.23°`。但 source-macro
  distance Spearman `0.2867→0.2485`，pooled `0.4246→0.3410`；overall
  6/8，终态
  `D31_THOR_MAGNI_FULL_RESOLUTION_MEASUREMENT_RELATION_NOT_SUPPORTED`。
  保留跨分辨率 person-bearing 正结果；停止 THOR box-height distance fitting，
  下一步转入原生 2D/3D identity-bound person trajectories。boxes/report SHA-256：
  `ecc30d0106372245c26cae6e5bece1b051036a2037ddb8e5908a4d75ff27701f` /
  `bb8f68214cb617729ca289fc4762ab700b4e452e04fa386403e442dc4c0bb860`。
- 时间：2026-08-03（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D32
  JRDB causal-track future-range canary。直接复用四个既有 native multisensor
  observation packets，不重新下载或扫描 metadata；source 原样继承冻结的同一
  identity 七帧 `log(box_height)` OLS tri-state，truth 改为同一 identity 在
  `+15 frames`（约一秒）的 `center_base_link_m` range change。8,766 个 future
  opportunities 中产生 480 条 non-abstain evidence、25 条 sequence-bound tracks，
  coverage `5.48%`。pooled precision `97.50%`；confirm `209/216=96.76%`，
  contradict `259/264=98.11%`，相对对应 prevalence lift
  `+45.24/+49.62 pp`。Clark/Gates/STLC 三条证据充分序列分别
  `97.00%/100.00%/96.60%`；Meyer 仅 9 rows，保留 9/9 正观察但不计正式
  sequence pass。全部可判定与 effect gates 通过，终态
  `D32_JRDB_CAUSAL_TRACK_FUTURE_RANGE_SUPPORTED`。首次执行在读取任何 packet
  前因 cwd-relative path 触发 `FileNotFoundError`；仅修复 repo-root resolution
  后按同一协议重跑，归类为 engineering failure，不烧毁 cohort。该结果建立
  `JRDB_ANNOTATION_TRACK_SHORT_FUTURE_MECHANISM_SUPPORTED`，证明
  same-identity causal trajectory state 是有效的短未来变量；不升级为 live
  detector/tracker、事件效用、App 或安全主张。下一步只替换 source measurement
  为冻结 detector + causal tracker。report SHA-256
  `81761e24b2098d9f585d8c8fd9a786eea0e21fff22c9b99b55dfb017dd07c2ec`；
  删除 wall-clock 字段后连续两次重建 SHA 一致。
- 时间：2026-08-03（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D33
  JRDB detector-track future-range replication。只把 D32 annotation box/native
  identity source 替换为真实 stitched RGB 上冻结的五 tile YOLO11n + ByteTrack；
  七帧 `log(box_height)` tri-state、`+15 frames` future range、deadband 与 gates
  不变。按既有 packet member/CRC/SHA 从官方 ZIP range 恢复 480/480 JPEG，
  network `197,136,580` bytes，未下载完整 archive。source producer 产生
  8,665 raw detections、5,366 tracked occurrences、165 tracks；4,772 个
  detector/native matches 的 IoU median/P10 为 `0.770/0.528`。3,392 个七帧+
  future opportunities 中有 283 条 non-abstain evidence、25 个 native identities；
  pooled precision `274/283=96.82%`，confirm `128/133=96.24%`，contradict
  `146/150=97.33%`，相对对应 prevalence lift `+65.70/+27.88 pp`，七帧
  native-ID 全一致率 `96.47%`。Clark/Gates/STLC precision
  `96.36%/100%/95.31%` 全部通过；Meyer 有 161 个 opportunities 但 0 个严格单调
  non-abstain，按 gate 为 sequence-level insufficient，不改写成错误方向负结果。
  全部 evaluability/effect gates 通过，终态
  `D33_JRDB_DETECTOR_TRACK_FUTURE_RANGE_SUPPORTED`。相对 D32 precision 仅下降
  `0.68 pp`，建立
  `JRDB_DETECTOR_TRACK_SHORT_FUTURE_MECHANISM_SUPPORTED`；下一步进入不驱动提醒的
  Android shadow state parity/runtime canary，主线与默认 App 不变。tracks/report
  SHA-256 分别
  `efa249fdfe8114dfeb1da419ffdb359189e3d4e6b1f406fabad04a31a39a0fa1` /
  `fa2b403328428bbe596833a670970785964ae197e992b39cc47f878b3013984a`，
  连续重建一致。
- 时间：2026-08-03（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D34
  Kotlin shadow-state parity/runtime canary。把 D33 全部 5,366 个 source-only
  detector-track occurrences、165 tracks 与 packet timestamps 物化为 deterministic
  TSV，不携带 annotation association、native identity、3D range 或 future truth；
  input SHA-256
  `d1f24dc7c61890e912d2a4a1cbca23e4b729dfceb1ef76b435cd573c97e6021e`。
  直接调用 production `CausalTrackTristateGeometryProducer`，第一遍 warm-up、
  第二遍逐 occurrence parity/计时。decision mismatch `0/5,366`，slope presence
  mismatch `0`，最大 absolute slope error `8.44e-7/s`。host JVM producer-call
  P50/P95/P99 `0.0014/0.0022/0.0044 ms`，P95 远低于冻结 `0.10 ms` gate；
  `core:assist` 全量测试通过。终态
  `D34_KOTLIN_SHADOW_STATE_PARITY_RUNTIME_SUPPORTED`，建立
  `PRODUCTION_KOTLIN_CAUSAL_TRACK_STATE_PARITY_AND_HOST_RUNTIME_SUPPORTED`。
  execution 未进入 decision/event/feedback seam，non-actuating、future-truth-free，
  主线与默认 App 不变。首次 Gradle 验证命令因 PowerShell 未引用 `-D` 参数，在
  编译前被误读为 task；修正后成功，归类为工程命令错误，不产生科学终态。
  report SHA-256
  `c6ac570f19cf5d06f00dc159b920f75dbbd44be1d2808949bc894620631a9247`。
  下一步进入 isolated `.dualloop.shadow` 物理设备 parity/runtime/non-interference
  canary。
- 时间：2026-08-03（Asia/Hong_Kong）；执行者：violjjet。冻结并实现 HFTF D35
  Android device shadow parity/runtime/non-interference canary，但尚未执行科学
  measurement。新增独立 `:hftf-device-canary` `com.android.test` 模块，以同名
  test build type 绑定 target App `dualLoopShadow`；`aapt` 确认 target package
  `com.linnan.blindassist.dualloop.shadow`，production BuildConfig 为
  `DUAL_LOOP_SHADOW=true`、`DUAL_LOOP_ACTIVE=false`。D34 的 5,366-row
  source-only corpus 以 gzip payload 内嵌，APK 内 payload SHA-256
  `91039be8a9d6282d89a8a9dc3e6200a8e8e09cc6f4fc43aa80c9ae935aeecfec`，
  device report 改用 Android `AtomicFile` interruption-safe 写入。
  `:hftf-device-canary:assembleDualLoopShadow` 通过；target/test APK SHA-256
  分别为
  `e28e5c996174adef706f43ad6267a44e1c2ab017261ad99643b4efd4016a9557` /
  `adffd1be8c401a65070c25b2e51263394311951d1f9986ef1693f812d8e695c3`。
  构建过程中修复 Kotlin named-argument 与 Android 自动解包 `.gz` asset 两项
  engineering failure，均发生在设备 measurement 前，不烧毁 corpus。当前
  `adb devices -l` 为空且本机无 AVD，终态保持 `NOT_EVALUATED` /
  `READY_FOR_DEVICE_EXECUTION`，不解释为科学负结果。
- 时间：2026-08-03（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D36
  THOR-MAGNI production track veto event replay。复用 D12/D31 的 19 个
  outcome-open Development sessions、530 个 proximity-eligible anchors 与冻结
  YOLO11n source；3,710 个 unique source frames 产生 14,364 个 person
  detections。D31 anchor raw-count mismatch、selected-mask mismatch 与最大
  selected-box error 为 `0 / 0 / 0.0`。paired production kernel 的 baseline
  使用 `OFF`，candidate 只注入 production
  `CausalTrackTristateGeometryProducer` evidence；raw/stable risk mismatch
  为 0。baseline/candidate positive event hits 均为 `79/107`，negative false
  alerts 均为 `251/373`，candidate-only windows 与 positive losses 均为 0。
  但完整 cohort 只有 2 个 admitted contradiction frames、来自 2 个 sessions，
  低于冻结的 `>=10 anchors / >=5 sessions` opportunity gate；终态
  `D36_THOR_MAGNI_PRODUCTION_TRACK_VETO_EVENT_NOT_EVALUABLE`，具体瓶颈为
  `SELECTED_TARGET_STRICT_CONTRADICT_COVERAGE_INADEQUATE_FOR_EVENT_VETO`，不解释
  为算法负结果。错误 Python 环境、顺序 decode 过慢与首版 seek batching parity
  漂移均在 truth/outcome join 前停止并修复，不烧毁 cohort。detections、
  producer receipt、kernel replay、report SHA-256 分别为
  `5083db4c86ff20c01d12a47aa9b419d3cd1727b8f8ed1383020b33601ad6f731`、
  `26f2991f4f836c2611224bc9ea1c9066d00b6bf03b600155736fafcbf8ab5ade`、
  `9401307d5b4a5bce766a94b54f0890031d733cf44144b70d2aca41748a25f25d`、
  `a3c7861a4b2a1297c6deae1dc9e3464a30043037f003eb533160bec4115ab5d3`。
  下一步保持同一输入与 event gates，只替换为 production scene-scale producer，
  不在同一 outcome 上调 track threshold/history/monotonicity，也不改变主线或
  默认 App。
- 时间：2026-08-03（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D37
  THOR-MAGNI production scene-scale veto event replay。保持 D36 的 19 sessions、
  530 anchors、truth-free detections、production kernel 与 event gates 不变，
  唯一变量为 kernel 内 production
  `CausalSceneScaleTristateGeometryProducer`。全部 evaluability gates 通过：
  admitted contradiction 覆盖 `351 anchors / 19 sessions`、共 682 frames，
  raw/stable risk mismatch 与 non-scene source observations 均为 0。candidate
  造成 508 次逐帧 feedback suppression，positive anchor/event 均零损失，
  candidate-only frames/windows 均为 0；但 negative triggered windows 仅从
  `251/373` 降到 `250/373`，绝对减少 1、relative reduction `0.398%`，且只有
  1/5 folds 出现任何 reduction，未通过冻结的 `>=10`、`>=20%`、`>=3/5 folds`
  gates。终态
  `D37_THOR_MAGNI_PRODUCTION_SCENE_SCALE_VETO_EVENT_NOT_SUPPORTED`。它保留
  `PRODUCTION_SCENE_SCALE_CONTRADICTION_HAS_REAL_EVENT_OPPORTUNITY` 正机制，但
  `FRAME_LOCAL_SCENE_SCALE_VETO_EVENT_UTILITY_NOT_SUPPORTED`；断点位于逐帧
  suppression 与 window/event terminal 之间，下一变量应是 bounded
  temporal/event-scoped veto semantics，而不是调 scene threshold。首次报告
  虽正确写出 `NOT_SUPPORTED`，但后缀判断错误地序列化
  `supported=true`；改为 exact status equality、增加回归测试并原样重跑后为
  `supported=false`，归类为可修复 control-plane bug，不烧毁 cohort。
  kernel/report SHA-256 分别为
  `390fa479ce1bedec904d6b22ff70fa97b32288e89a3cc26d1d1695e37856622e` /
  `875d2b092cd110d9dae60bdf94490c8dd61a150e8a48604709d37730d23309bb`；
  重复 replay 一致，`core:assist` 全量测试与 D36+D37 evaluator 8 tests 通过。
  主线、默认 App、D35 真机终态均不变。
- 时间：2026-08-03（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D38
  bounded temporal veto event replay。该实验明确标记为看到 D37 后的
  `POST_D37_ADAPTIVE_OUTCOME_OPEN_DEVELOPMENT`；保持同一 19 sessions / 530
  anchors、detector、scene producer、risk/event/planner 与 gates，只新增独立
  `ACTIVE_CONTRADICT_TTL` mode，将 admitted contradiction 的 feedback-only veto
  按 production evidence TTL 固定延续 250 ms，未搜索 duration，D37 原 mode
  不变且重复 replay SHA 仍为
  `390fa479ce1bedec904d6b22ff70fa97b32288e89a3cc26d1d1695e37856622e`。
  D38 产生 492 次 latch-only suppressions，覆盖 `231 anchors / 19 sessions`，
  全部 evaluability gates 通过；negative windows 从 `251/373` 降至
  `217/373`，绝对减少 34、relative reduction `13.55%`，4/5 folds 改善，建立
  `BOUNDED_TEMPORAL_VETO_CHANGES_EVENT_TERMINALS_DEVELOPMENT_ONLY`。但 positive
  anchors 从 114 降至 98，positive events 从 79 降至 73，损失 16 anchors /
  6 events，同时 relative negative reduction 未达冻结 20% gate；终态
  `D38_THOR_MAGNI_BOUNDED_TEMPORAL_VETO_EVENT_NOT_SUPPORTED`，拒绝
  `FIXED_250MS_UNCONDITIONAL_SCENE_VETO_PERSISTENCE`。不得在同一 outcome 上
  搜索其他 hold duration；若继续，变量必须是可解释的 event/target continuity
  与解除条件，并以新鲜独立 outcome evidence 评价。kernel/report SHA-256 为
  `8cf20b345f30fa757307c430e5eeeb63a2859450d238c06a50ad5fbd22394930` /
  `af97a203f06208f6256a1e1bee45191908c46bda41a5dc45793216f4a4ef09d7`；
  `core:assist` 全量测试及 D36+D37+D38 evaluator 10 tests 通过。一次合并验证
  命令因从脚本 cwd 使用 repo-relative path 而失败，未改写 report；回到 repo
  root 后原样成功，归类为可修复 path error。主线、默认 App 与 D35 均不变。
- 时间：2026-08-03（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D39
  confirm-release veto event replay。该实验明确标记为 D38 后的 adaptive
  outcome-open Development；新增独立 source
  `CAUSAL_SCENE_SCALE_BIDIRECTIONAL_R1`，保持 scene association/median/quality
  不变，以严格对称 `-0.05/+0.05/s` 输出 contradict/confirm；新增独立
  `ACTIVE_CONTRADICT_TTL_CONFIRM_RELEASE` mode，让 admitted confirm 立即解除
  250 ms hard-cap latch。D37/D38 artifact SHA 保持
  `390fa479ce1bedec904d6b22ff70fa97b32288e89a3cc26d1d1695e37856622e` /
  `8cf20b345f30fa757307c430e5eeeb63a2859450d238c06a50ad5fbd22394930`。
  D39 有 1,247 admitted confirm frames；406 anchors / 19 sessions 有 confirm，
  272 / 19 实际解除 live latch，331 次 release；latch-only suppressions 降至
  73。全部 evaluability gates 通过。positive event losses 从 D38 的 6 恢复为
  0，但仍损失 2 positive anchors；negative windows 仅从 `251/373` 降至
  `250/373`，relative reduction `0.398%`，仅 1/5 folds 改善。终态
  `D39_THOR_MAGNI_CONFIRM_RELEASE_VETO_EVENT_NOT_SUPPORTED`。D38 无条件保持
  过强、D39 单帧对称 confirm release 过弱，按冻结 stop rule 建立
  `HFTF_SCENE_SCALE_PERSISTENCE_FAMILY_STOP`；不得在同一 outcome 上继续搜索
  threshold、confirm count 或 duration。下一变量必须来自新的 target/event
  continuity evidence 或新鲜 event cohort。kernel/report SHA-256 为
  `3b3a3d7a587a95baa5942b3b343ad9bd31a3cf788f5ef3c6929f4d25216ea832` /
  `bfad01a931d169178e5060e13e2fcb4f40aefccf612e57c8bf03158cd5e7abb7`。
  重复 D37-D39 replay、`core:assist` 全量测试及 D36-D39 evaluator 12 tests
  通过。首次 combined patch 因 context mismatch 整体未应用，拆分后成功；
  发生在 source replay/truth join 前，不烧毁 cohort。主线、默认 App 与 D35
  均不变。
- 时间：2026-08-03（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D40
  continuous-track projected-risk replay。该实验在 outcome 前冻结并离开
  `HFTF_SCENE_SCALE_PERSISTENCE_FAMILY_STOP`：复用 D36 的 19 sessions / 530
  anchors / 3,710 unique frames / 14,364 detections，从 production
  `CausalTrackTristateGeometryProducer` 读取连续 `signedApproachRatePerS`，
  固定 `1.0 s` horizon、`scale=exp(slope*horizon)`，保持 selected box
  bottom-center 后运行独立 production risk kernel。205 forecast windows
  覆盖全部 19 sessions（136 positive-slope / 69 negative-slope），全部
  evaluability gates 通过；但 candidate 与 baseline 的 positive anchors
  `114/157`、positive events `79/107`、negative alerts `251/373` 完全一致，
  五 folds 均无 gain/loss。终态
  `D40_THOR_MAGNI_CONTINUOUS_TRACK_PROJECTED_RISK_NOT_SUPPORTED`，建立
  `D40_SELECTED_TARGET_BOX_SCALE_PROJECTION_RECIPE_STOP`；不在已消费 outcome
  上搜索 horizon、clamp 或 threshold。该结果不撤销 D32/D33 future-range
  mechanism；下一候选必须使用新的 geometry teacher/field evidence 直接表达
  future traversability，并绑定新数据角色或新鲜 outcome cohort。重复 Kotlin
  replay SHA 为
  `fae215ddebfcb774c15e5ef18934fca36a85b1481d63905762fb70ac435884e4`，
  report SHA 为
  `c4716729c69de435f40eee3717c5bdada2e20ee6f49fb79f0dfec8d4869d0d06`；
  `core:assist` 全量测试与 D36-D40 evaluator 14 tests 通过。source-only
  阶段修复 deduplicated-frame ordinal 比较，truth join 前完成，不烧毁 cohort；
  一次 Gradle JVM property 被 PowerShell 误解析为 task，改用 `GRADLE_OPTS`
  后原样成功。主线、默认 App 与 D35 均不变。
- 时间：2026-08-03（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D41
  JRDB causal future-box field。实验在 outcome 前冻结，以 D33 detector tracks
  的连续 7 帧分别对 center x/y 与 log width/height 做 timestamp-aware OLS，
  固定外推到 `+15 frames`；forecast 不读取 annotation，评价才联接 current
  Hungarian match 与 future same-identity native box。3,392 opportunities /
  54 identities、4 sequences 的 evaluability gates 全部通过。candidate mean
  future-box IoU 从 current-box baseline 的 `0.36434` 升至 `0.40926`
  （`+0.04491`），3/4 sequences mean delta 为正；但 median delta 为 0，
  candidate better fraction 仅 `47.995%`，center error 只降低 `6.887%`，
  absolute log-area error 从 `0.29466` 恶化到 `0.41313`，终态
  `D41_JRDB_CAUSAL_FUTURE_BOX_FIELD_NOT_SUPPORTED`。保留
  `D41_TRANSLATION_LOCAL_SIGNAL_RETAINED_DEVELOPMENT_ONLY`，建立
  `D41_CONSTANT_VELOCITY_LOG_SCALE_RECIPE_STOP`；不得在已消费 outcome 上删除
  scale、搜索 state subset/horizon/regression。下一变量需使用新鲜 evidence 或
  ego-motion/metric-geometry teacher。R0.1 在任何聚合 outcome 前修复
  20/3,692 fully-off-frame forecast 语义，保留 raw box 并原样惩罚；R0.2 修复
  `478 non-empty frames` 被误当作 source census，改绑定 D33 COMPLETE receipt
  的 `480/480`，不改变任何 effect metric且不可能翻转已失败的四项 support gate。
  D32/D33/D41 evaluator 11 tests 通过；report 连续重建 SHA 稳定为
  `73418b3308a259e63a2c413105d907f6ea416297628568f1d80f0d0d0db71ba3`。
  主线、默认 App 与 D35 均不变。
- 时间：2026-08-03（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D42 JRDB
  ego-object metric teacher。D42 在 outcome 前冻结，复用 D41 detector-matched
  opportunities 与 7-frame history，以 packet 的 `odom <- base_link` pose、
  `center_base_link_m`、`center_odom_m` 构造 current-static、ego-only、
  ego+person-world 三臂，固定预测 `+15 frames`。3,384 opportunities /
  53 identities、4 sequences 全部 evaluable；transform parity maximum error
  `1.1368683772161603e-13 m`。完整 teacher 相对 current-static 使 mean
  horizontal error `0.80935 -> 0.34757 m`（-57.06%）、median
  `0.74938 -> 0.14080 m`（-81.21%），`79.994%` opportunities 改善，
  range/bearing error 分别降低 `81.80%/53.79%`，四 sequences 全部改善，
  7/7 support gates 通过，终态
  `D42_JRDB_EGO_OBJECT_METRIC_TEACHER_SUPPORTED_DEVELOPMENT_ONLY`。
  ego-only 仅改善 `9.65%`，加入 person world motion 后相对 ego-only 再改善
  `52.47%`，建立 `D42_PERSON_WORLD_MOTION_DOMINANT_INCREMENT_SUPPORTED`。
  该正结果只授权冻结 D43 的 phone-causal 2D track/RGB/IMU student contract，
  不授权 inference 使用 native identity/pose/3D/future truth，也不建立 event、
  Android、主线、产品或安全主张。D32/D33/D41/D42 evaluator 13 tests 通过；
  report 连续重建 SHA 稳定为
  `1b8a8b9458edb2dd7b5f34eca95b5c0bdd9b0715efa8881cbbf8a43d5e1f5dfb`。
  主线、默认 App 与 D35 均不变。
- 时间：2026-08-03（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D43
  track-IMU metric residual student source gate。D43 在训练/held-out outcome
  前发现 IMU coverage 不满足冻结四折合同：Clark/Meyer 各 120/120
  IMU-complete frames 与 1,304/194 complete track histories；Gates/STLC
  complete frames/histories 均为 0。未填零、插值、删 sequence 或降为两折；
  `model_training_executed=false`、`future_outcome_evaluated=false`，终态
  `D43_JRDB_TRACK_IMU_METRIC_RESIDUAL_STUDENT_NOT_EVALUABLE`，不产生 IMU
  learnability 结论。原协议的四折 `TRACK_ONLY` arm 输入完整，因此独立冻结
  D43.1，保持原 10 features、D42 teacher target、Ridge alpha、zero baseline
  与 effect floors，不回填 D43 IMU 主张。主线、默认 App 与 D35 均不变。
- 时间：2026-08-03（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D43.1
  track-only metric residual student。固定十个 current/7-frame detector-track
  state/slopes/confidence features、population standardization、closed-form
  multi-output `Ridge(alpha=1.0)` 与四折 leave-one-sequence-out。3,384
  opportunities / 53 identities 全部 evaluable。相对 zero residual，pooled
  teacher vector error `0.80238 -> 1.10533 m`（恶化 37.76%），actual future
  error `0.80935 -> 1.11648 m`（恶化 37.95%），actual better fraction
  `22.370%`；仅 Meyer 1/4 folds 改善，STLC actual error 恶化 `169.30%`。
  终态 `D43_1_JRDB_TRACK_ONLY_METRIC_RESIDUAL_STUDENT_NOT_SUPPORTED`，建立
  `D43_1_FIRST_ORDER_2D_TRACK_METRIC_MAPPING_STOP`。不得在同一 outcome 上改
  alpha、feature subset、target normalization、加非线性或删 STLC。D42 teacher
  ceiling 保持；下一 source 必须增加完整 IMU 或 causal metric-depth/ground
  measurement。D42/D43/D43.1 focused tests 4 PASS；report 连续重建 SHA 稳定为
  `d104279a42a8089a171ca4fcab4db7c85e0004f1f201ee51f1667bd9dbadcd23`。
  主线、默认 App 与 D35 均不变。
- 时间：2026-08-03（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D44 JRDB
  causal relative metric track。D44 在 outcome 前冻结，复用 exact 3,384
  opportunities / 53 identities，对 same-target 连续 7 帧
  `center_base_link_m` x/y/z 做 timestamp-aware OLS，固定预测 `+15 frames`，
  不显式使用 future、world pose 或 ego/object decomposition。相对 current-static，
  mean horizontal error `0.80935 -> 0.35324 m`（-56.36%），median
  `0.74938 -> 0.13948 m`（-81.39%），`79.787%` opportunities 改善，
  range/bearing error 分别降低 `79.10%/54.20%`，四 sequences 全部改善，
  7/7 gates 通过，终态
  `D44_JRDB_CAUSAL_RELATIVE_METRIC_TRACK_SUPPORTED_DEVELOPMENT_ONLY`，建立
  `D44_RELATIVE_METRIC_HISTORY_SUFFICIENCY_SUPPORTED`。D44 mean error
  `0.35324 m` 几乎达到 D42 full world teacher 的 `0.34757 m`，将下一瓶颈定位为
  phone-causal same-target metric-depth measurement，而不是更大模型。只授权
  source-only depth measurement/quality/latency shadow canary，不接入 event 或
  production seam。D42/D43.1/D44 focused tests 5 PASS；report 连续重建 SHA
  稳定为
  `c96c37fca85f8a52fb37d372a8290a564982e241352e8d7a173e4b5a4ad03f09`。
  主线、默认 App 与 D35 均不变。
- 时间：2026-08-03（Asia/Hong_Kong）；执行者：violjjet。冻结并实现 HFTF D45
  phone metric-depth source canary readiness。D45 不读取 event/alert outcome，
  固定 person-box center-60% metric sampler、coverage/confidence/IQR/staleness
  gates 与 exact same-target 7-point OLS `+1.0 s` solver。实现置于独立
  `:hftf-metric-depth-canary-core`，7 个 focused JVM tests 通过；ARCore 1.33.0
  仅加入专用 `:hftf-device-canary` test APK，capability probe 不 resume session、
  不打开 camera、不请求安装，并以 `AtomicFile` 写 canonical receipt。相对冻结
  commit `9f47a7d`，`app/core/feature/gradle` production tree 零 diff；default App
  runtime classpath/merged manifest 均不含 ARCore 或 D45 module。target/test APK
  SHA-256 分别为
  `afa7a774b9f47074b2bf2e59755e712e92421484140789513578b32b68f0f149` /
  `1b0142c94abd19a5b0702f67c3c7a38115251f51bd04a25411d6867a570a64ca`。
  R0.1 在任何 device outcome 前确认 only raw depth 暴露对应 confidence image，
  将 raw+confidence 设为唯一 measurement-ready source，automatic-only 不伪造
  confidence。当前 ADB 无设备，终态保持
  `D45_NOT_EVALUATED_NO_READY_DEVICE`，不是 source
  负结果；设备 capability/measurement 尚未执行，主线、默认 App 与 D35 均不变。
- 时间：2026-08-03（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D45 raw
  source decoder readiness R0.2，不新增结果门。实现 stride-safe unsigned
  16-bit raw depth + 8-bit confidence decoder，并用类型边界固定为
  `SOURCE_REGISTRATION_UNVERIFIED`，不能进入 person sampler；10/10 focused JVM
  tests 通过。isolated `:ustrf-shadow-benchmark` device canary 只聚合 acquisition
  failure、timestamp、valid-pixel coverage 与 acquisition+decode P50/P95，单
  `AtomicFile` receipt 上限 256 KiB，不保存 raster、不产生人物/事件结果。既有
  SM-S9280 source-class prior 在 moving runs 中取得 585/813 次 raw depth，而
  两个 autonomous frame-bound runs 均为 0/150 tracking/depth，故零观测固定为
  `NOT_EVALUABLE_*`，不作为算法负结果或 cohort burning。benchmark/test APK
  SHA-256 为
  `4b316a5895da000023f24ba19e118d5c1aa97024f8702c0f2e6e9904aa3b3087` /
  `d4b90e06c1d0430885dcb9498f305a747555653c078e4d3733dcbf1b67d5f83c`。
  default App production runtime classpath/manifest 仍不含 ARCore/D45；当前 ADB
  无设备，科学终态仍为 `D45_NOT_EVALUATED_NO_READY_DEVICE`，主线、默认 App
  与 D35 均不变。
- 时间：2026-08-03（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D45
  coordinate registration readiness R0.3，不读取 device/person outcome。官方
  ARCore 语义确认 raw depth 是 GPU-aspect/native-orientation camera crop，原
  detector-box→depth 的简单宽高 scale 不能成立。实现改为显式组合 CameraX
  detector rotation 与同帧 ARCore `IMAGE_PIXELS -> TEXTURE_NORMALIZED`
  9-point affine receipt；sampler inverse-map native raw-depth pixel center，
  不 upsample sparse depth、不重复计数。registration 与 exact source frame
  id/timestamp 绑定，跨帧不能解锁；transform id 对微小 float noise canonicalize，
  depth uint16 使用 Android native byte order；depth crop 外目标显式报告
  `NO_REGISTERED_PIXELS`，不混入 depth-quality failure。18/18 focused JVM tests 与
  benchmark/test APK 编译通过；APK SHA-256 为
  `3e99937243b7014a8cdaf27dfa00343d0f4a5666d41d295dadd1ab82e15639b4` /
  `bd364997988853474d71d6825bfa40787698da5f04cee521f1b2857e6c27ad6b`。
  device receipt 即使得到 `AFFINE_REGISTRATION_OBSERVED_DEVICE_ONLY`，仍固定
  `external_alignment_verified=false`、`person_registration_verified=false`；
  当前无 ADB 设备，科学终态不变，主线、默认 App 与 D35 均不变。
- 时间：2026-08-03（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D45
  physical person measurement runner readiness R0.4，不读取人物测量 outcome、
  不新增 gate。isolated `:ustrf-shadow-benchmark` 只读复用 exact production
  YOLO asset（5,359,428 bytes，SHA-256
  `00edb41a528b0a7e709c4af8ce3e685491492c4539274804e5cfc17a1a867cd2`），
  在同一 ARCore frame 内串联 fresh raw depth/confidence、frame-bound
  registration、stride-safe YUV_420_888→RGBA、CPU person detector 与 native
  depth sampler。controlled scene 固定 exactly one person，1/2/3/5 m 分别运行；
  measurement latency 包含 conversion、detector 和 sampling。24/24 focused JVM
  tests 通过，benchmark/test APK 编译成功；不保存 camera/depth/box，只写至多
  1,800 个 depth/latency 标量和 aggregate metrics 的 `<=256 KiB` AtomicFile
  receipt。缺 reference 参数时 test SKIP；source/detector/registration 不可用
  仍为 `NOT_EVALUABLE_*`，四距离完成前不产生总终态。默认 App 与主线均不接
  ARCore/D45，当前无 ADB 设备，D35 不变。
- 时间：2026-08-03（Asia/Hong_Kong）；执行者：violjjet。完成 HFTF D45
  recoverable four-distance aggregation R0.5，仍未读取 device/person outcome、
  未新增 gate。measurement runner receipt 现在记录 target/test APK 内容哈希与
  `risk_feedback_invocation_count=0`；host reader 只接受显式的 1/2/3/5 m 小
  receipt，strict 校验 UTF-8 JSON、duplicate key、finite scalar、size ceiling、
  bounded arrays、recomputed error 及同 device/build/camera/detector binding。
  overall error/latency 固定 pooled accepted observations，coverage/history
  固定 ratio-of-sums。10/10 host tests 与 24/24 focused JVM tests 通过，
  benchmark/test APK 编译成功。缺距离、malformed input、跨构建和 baseline
  mismatch 均固定 `scientific_terminal=null` 且不创建最终 output；修复后可
  重跑，不能烧毁 D45。default App APK hash 仍为
  `afa7a774b9f47074b2bf2e59755e712e92421484140789513578b32b68f0f149`；
  当前无 ADB 设备，科学终态、主线、默认 App 与 D35 均不变。
- 时间：2026-08-04（Asia/Hong_Kong）；执行者：Codex。完成 ARKitScenes visit
  `484248` scale-free 反例机制审计与 sealed camera-conditioned scale student R0
  离线压力测试，全程未调 margin/percentile/window、未重训或 refit。反例审计确认
  150/150 帧按官方 pose 均为需顺时针 90 度矫正的 `left` orientation；132/148
  帧含大平面，source confidence-2 覆盖中位数 `.9549`、跨 band 最近邻借点中位数
  仅 `.0057`，而 DA/sensor 完整 band 排序仅 `.4797`。解释性 upright 反事实把
  coverage/方向一致率从 `.2241/.3846` 提到 `.8261/.8947`，但原 R2
  `NOT_EVALUABLE` 保持。scale-free 不作为辅助输出或 fallback，仅在 orientation
  receipt 后保留 Development disagreement detector；未矫正输入关闭。冻结学生
  在 330 帧 × 47 cached-depth 场景中能抵消 DA 全局尺度 `±40%`，但 20% bandwise
  局部形变 accepted-bad 最高 `.5152`；全宽 lower ROI 遮挡 50% 仍有 `.9758`
  coverage，却达到 `.3211 m/.1231` MAE/false-clear。50 帧 × 25 RGB→冻结 DA
  场景中 Gaussian `sigma=3` coverage `.86`、MAE `.3780 m`、false-clear `.1611`，
  证明现 plane residual/支持不能代替独立 blur 与 ground-support gate。产出 provisional
  phone capture contract：高度不确定度 `<=5 cm`、严格 camera/intrinsics/crop/rotation/
  mount identity、至少 75% 全宽 lower-ROI support、独立 blur quality gate，否则
  `UNKNOWN`；pitch 数值范围仍待真实手机确认。20 个 focused tests、语法、JSON/
  ledger 数量与 protocol hash 复核通过；默认 App、生产与安全权限不变。
- 时间：2026-08-04（Asia/Hong_Kong）；执行者：Codex。完成
  `QNN_NATIVE_CACHED_CONTEXT_R0` USB 真机闭环。基于 QAIRT SampleApp 补齐
  backend/device/cached-context/graph 生命周期、复用 FP16 direct input/output buffer，
  并从 `QAIRT_ROOT` 生成 APK JNI runtime，不提交 proprietary binary。相同 runtime 与
  相同 FP16 tensor 下 App/CLI 深度逐元素误差为 0；10 次 graph execute
  P50/P95 `74.45/74.69 ms`，Native preprocess+execute `79.64/94.29 ms`，thermal
  `0 -> 0`。遗漏 `deviceCreate` 的诊断臂约 `274 ms` 且 RPC polling 不可用，已修正。
  Kotlin half conversion 改为 IEEE ties-to-even 后官方 FP16 parity 恢复；但 fused Native
  FP16 深度输出 mean/P95/max 为 `1.99/7.81/46.88 mm`，严格 `2/5/20 mm` 门失败，
  即使下游 status/height/scale 门通过也不救活。终态分别为
  `QNN_NATIVE_CACHED_CONTEXT_R0_SUPPORTED_DEVICE_ONLY` 与
  `FP16_FUSED_PREPROCESS_STRICT_DEPTH_PARITY_NOT_SUPPORTED`；CameraX、持续能耗、生产和
  安全 authority 仍未建立。
- 时间：2026-08-04（Asia/Hong_Kong）；执行者：Codex。完成
  `CAMERAX_LATEST_ONLY_R0` USB 真机短跑。隔离 canary 固定真实 `YUV_420_888`
  640x480、CameraX `KEEP_ONLY_LATEST`、rotation 后居中 4:3 crop、三槽循环复用、
  单运行任务加可替换 pending、750 ms TTL 与 severe-thermal fail closed。20 秒内
  291/291 个 `ImageProxy` 关闭，三个槽全部归还；5 秒压力段提交 64 次并发生 6 次
  pending 替换，随后 2 Hz 段提交 29 次，最大深度并发严格为 1。YUV copy P50/P95
  `5.47/18.64 ms`，YUV->FP16->QNN P50/P95 `75.93/84.44 ms`，结果年龄 P95
  `141.14 ms`；无 stale、异常或 thermal fail closed。终态仅为
  `CAMERAX_LATEST_ONLY_R0_SUPPORTED_DEVICE_CANARY_ONLY`；新增相机 crop/rotation/color
  合同已冻结，但尚无准确率、10 分钟持续、生产或安全 authority。
- 时间：2026-08-04（Asia/Hong_Kong）；执行者：Codex。完成
  `GEOMETRY_EQUIVALENT_OPTIMIZATION_R0`。保持 stride=4、确定性 5000 点 cap、
  `Random(1729)`、240 次 RANSAC、全部门限/特征/拒答不变，仅缓存像素射线、改用
  reusable SoA/inlier/residual/finite-depth buffer 与精确 order statistic。冻结 clean
  HTP depth 真机 100 次 reference/optimized P50 为 `119.87/64.04 ms`，每帧分配
  `23,655,998.4/3,276.8 bytes`，GC `90/0`，逐字段最大误差 `6.94e-18`，JVM
  synthetic noisy/invalid parity 同样通过。终态为
  `GEOMETRY_EQUIVALENT_OPTIMIZATION_R0_SUPPORTED_DEVICE_ONLY`；稀疏采样、减少迭代、
  提前停止均未执行，生产与安全 authority 不变。
- 时间：2026-08-04（Asia/Hong_Kong）；执行者：Codex。完成 600 秒亮屏
  `CAMERAX_FULL_PIPELINE_SUSTAINED_R0`。真实 YUV CameraX 持续 8993 帧且 8993 个
  `ImageProxy` 全关闭；2 Hz 精确链路完成 1144 次 YUV->FP16->cached QNN->depth
  decode/resize->等价几何，最大并发 1，三槽全归还，geometry `1144 VALID/0 UNKNOWN`。
  全链路 P50/P95/max `174.70/202.69/220.58 ms`，结果年龄 P95 `218.47 ms`；
  thermal before/max/after `0/0/0`，非亮屏观察 0，过期结果显式
  `UNKNOWN(EXPIRED)`。PSS endpoint 增加约 19.2 MiB，ART 全程 14 次 GC/321 ms；
  endpoint 不单独证明无 leak slope，但无 owned resource 泄漏或延迟/温控门失败。
  终态 `CAMERAX_FULL_PIPELINE_SUSTAINED_R0_PERFORMANCE_SUPPORTED_DEVICE_ONLY`；因使用
  strict depth parity 已失败的 fused FP16 臂，准确率、生产和安全 promotion 继续拒绝，
  GPU 前处理 gate 未触发。
- 时间：2026-08-04（Asia/Hong_Kong）；执行者：Codex。完成独立可启动的
  `:hftf-depth-demo-app` 设备体验版，不改默认 BlindAssist App 或安全决策路径。
  Demo 复用 canary 的 canonical Native FP32 OpenCV/NEON、严格 IEEE ties-to-even
  FP32→FP16 与 QNN cached-context 源码，构建时打包本地 SM8650 DLC 和 QNN runtime；
  CameraX 使用真实后置 `640x480 YUV_420_888`、`KEEP_ONLY_LATEST`、单任务与 nominal
  2 Hz 节流，并展示 `343x259`、按帧内有效深度 5th/95th percentile 动态着色的
  红→黄→青→深蓝热力图；默认采用左 RGB/右纯热力图对照，并可切换全屏叠加或
  RGB-only，同时显示中心/近处深度、全链路延迟、刷新率和 thermal 状态。
  `SM-S9280 / SM8650` 上 debug APK 构建、安装、授权与 cold start 成功；实拍画面显示
  中心约 `1.67 m`、近处约 `0.70 m`、全链路 `92.8 ms`、刷新 `2.1 Hz`、thermal `0`，
  Activity 保持 resumed 且无 fatal exception。终态仅为
  `DEPTH_EXPERIENCE_APP_R0_AVAILABLE_DEVICE_ONLY`；跨设备、场景准确率、无障碍、发布
  签名、生产与安全 authority 均未建立。
- 时间：2026-08-04（Asia/Hong_Kong）；执行者：Codex。为
  `:hftf-depth-demo-app` 增加 R0.1 短时平滑显示：保留 nominal 2 Hz 真实 QNN 深度，
  每张已完成真实深度图只在 View 显示层执行 `110 ms` 线性交叉渐变。曾诊断性尝试
  8 Hz target，但低照 CameraX source 只有约 3–4 FPS，且用户决定暂不保留高频模式；
  最终版本不强制 Camera2 FPS、不提高推理频率。渐变像素不回写 metric 数值、QNN
  tensor、thermal gate 或任何下游判断；状态面板只报告真实完成帧。CameraX 仍为
  `KEEP_ONLY_LATEST`、单任务 in-flight、`ImageProxy` finally-close，severe thermal
  仍 fail closed。该改动只改善设备体验观感，不新增准确率、生产或安全 authority。
- 时间：2026-08-04（Asia/Hong_Kong）；执行者：Codex。记录并首次执行
  `FRESH-TF R0` consumed diagnostic。该候选明确继承 HFTF 已有 foot/body/head swept
  envelope，不把人体分层重新申报为创新；本次只比较 2 Hz zero-order hold、750 ms TTL、
  uniform age freshness 与 selective RGB-change freshness。唯一已消费 Bonn
  parent sequence 提供 30 帧、每臂 1,530 cells 和 30 个 anchor-to-current state
  transitions。selective 臂把 false-clear `3 -> 0`，但 known coverage
  `100% -> 21.24%`，未过预冻结 65% 门，终态
  `FRESH_TF_R0_CONSUMED_DIAGNOSTIC_NOT_SUPPORTED`。失败限定为 whole-frame grayscale
  MAD 乘全局 age decay；不在同一片段调 scale/tau/threshold 救援。下一可评价问题需
  新 parent/session-disjoint 数据、motion-compensated local-cell support 与分层硬
  validity gates；默认 App、NPU scheduler、提醒、生产和安全 authority 不变。6 个
  focused tests 通过。
- 时间：2026-08-04（Asia/Hong_Kong）；执行者：Codex。收紧上述 R0 的结论边界：
  正式记录 `GLOBAL_FRAME_FRESHNESS_PROXY_REJECTED` 与
  `LOCAL_GEOMETRIC_VALIDITY_NOT_YET_EVALUATED`，不把全局 MAD 的负结果外推为
  FRESH-TF 总概念失败。冻结 R1-A 只评价 motion-compensated local-cell support，
  foot/body/head 分层延后到 R1-B；NPU 调度、语义、ToF 与学习模型均不进入本轮。
  在媒体 outcome 未打开的条件下，从 TUM 旧官方端点取得预锁定的 `freiburg1_rpy`、
  `freiburg1_desk`、`freiburg3_sitting_static` 三个 archive；SHA-256、archive 根目录及
  `rgb.txt/depth.txt/groundtruth.txt` 均已封存。仅解析时间戳元数据后，三个序列分别
  接纳 721/596/688 个 RGB frame，均通过每 session 300 帧与 15 秒的来源 admission。
  当前终态仅为 `FRESH_TF_R1A_SOURCE_TRANSPORT_AND_METADATA_ADMISSION_SUPPORTED`；
  尚未打开图像/深度 outcome、实现 C1 或运行四臂。每机制仍只有一个 session，正式
  效果评价继续 `NOT_YET_ADMISSIBLE`。
- 时间：2026-08-04（Asia/Hong_Kong）；执行者：Codex。执行 R1-A C1 mechanics /
  opportunity canary。媒体打开前补齐并冻结 10 Hz sampling、TUM 内参、depth scale、
  full-resolution Farneback 参数、3 px geometry-flow residual 和 cell 状态优先级，
  最终 protocol SHA-256 为
  `2379D50E497ED417C6EF8BF6D9CFDD793AF64709B22AD494061E861687D345F9`。
  9 个 focused tests 通过；三序列共评价 676 帧、64,896 cells。C1 cell support
  coverage macro `28.91%`、worst-session `19.12%`；rpy/desk/sitting-static 分别为
  `19.12% / 23.61% / 44.01%`。状态中识别到 12,890 个 occluded、8,956 个
  newly-exposed 和 6,329 个 out-of-frame cell opportunity；硬状态赋值保证这些 cell
  不继承 supported，但这不是 false-clear 或遮挡检测准确率证据。冻结 C1 mechanics
  终态为 `FRESH_TF_R1A_C1_FROZEN_MECHANICS_NOT_SUPPORTED_CANARY_ONLY`；正式四臂
  gate 未运行，因为 direction/traversability truth 缺失且每机制只有一个 session。
  `LOCAL_GEOMETRIC_VALIDITY_EFFECT_NOT_EVALUATED` 保持；不得在已打开三序列上调参救援。
- 时间：2026-08-05（Asia/Hong_Kong）；执行者：Codex。基于 FRESH-TF R0 的全局
  freshness coverage collapse 与 R1-A C1 的 local-cell support collapse，正式登记
  `DENSE_OR_FIXED_CELL_DEPTH_PROPAGATION_FAMILY_STOP`。停止 whole-frame RGB validity、
  dense pixel/fixed image-cell 深度传播和二维光流维持完整可通行场；不关闭 fresh
  metric depth、异步双环、继承的 foot/body/head swept envelope 或未来决策层周期
  米制锚定。明确禁止在三条 consumed R1-A sequence 上换网格、点数、光流或 residual
  阈值救援，也不把三层人体包络重新申报为首次创新。
- 同日冻结 `HFTF_FRESH_METRIC_SNAPSHOT_LAYERED_INTRUSION_R0`，状态
  `FROZEN_BEFORE_SOURCE_COLLECTION_OR_QNN_OUTCOME`。新 formal cohort 固定为一台
  SM-S9280、CameraX 同帧 QNN depth、18 个完全重置 parent sessions、六类受控物理
  场景和 180 个 session 内重复 snapshot；独立卷尺/激光/fiducial 真值必须在第一份
  QNN output 前封存。固定比较 ground-only 2D、height-collapsed 3D 与继承的
  foot/body/head 三层表示；禁止 propagation、Track、语义、ToF/ARCore depth、调度和
  alert。新增 fail-closed source validator，强制 roster/order、真实镜头高度、同帧
  timestamp、calibration/truth/media SHA-256、18/180 exact counts，并拒绝任何 arm-output
  key；6/6 tests 与 py_compile 通过。当前只授权新数据采集和来源 admission，不授权
  outcome、PMAF Track、App、生产或安全结论。
- 时间：2026-08-05（Asia/Hong_Kong）；用户明确要求暂停 FRESH-TF 及本轮开拓出的
  后继路线。新增
  `FRESH_TF_AND_OPENED_SUCCESSORS_PAUSED_BY_USER / PAUSED_NO_ACTIVE_EXECUTION`
  覆盖记录，暂停 R1-A successor、dense/fixed-cell propagation successor、fresh
  metric snapshot 18-session collection/evaluation、PMAF/HSTF-PMA、periodic metric
  anchoring、stable Track metric anchoring 和相关 NPU scheduling/App integration。
  暂停不是失败，不改写既有终态；HFTF 明确保留为用户与本项目已经建立的原创贡献，
  CameraX/QNN/NPU 工程结果、depth demo、默认 App 和无关路线保持不变。fresh-snapshot
  protocol 在正式采集和 QNN outcome 前暂停，仍为 unconsumed design。只有用户以后明确
  指定 route/scope 并完成 repository/source/hardware/authority 复核后才可恢复。
- 时间：2026-08-05（Asia/Hong_Kong）；执行者：Codex。继续执行 DA V2 端侧完整链路
  R0-R3 工程优化，冻结 `518x686 FP16` cached DLC、前处理、5000 candidates、240 次
  RANSAC、seed 1729、阈值和几何语义。两级 latest-only pipeline 在 45 秒饱和 A/B 中
  从 5.700 提高到 9.175 Hz（+61.0%）；phase-locked 2/3/4/5 Hz 矩阵通过，首轮 5 Hz
  cadence drift 失败被原样保留。Native FP16 decode 对全部 65,536 half patterns raw-bit
  parity mismatch 0；Native C++ geometry 在真实深度及缩放、缺失、微扰共 8 cases 中状态
  一致，最大字段误差 `2.22e-16`。固定 APK 的 10 分钟 R3c 达到 5.00 Hz、3024/3024
  `VALID`，QNN/geometry/full P95 分别为 `96.01/17.80/123.18 ms`，thermal 0/2/2，
  两类 pool 全归还、runtime failure 0。一次 R3b 因运行中本地 APK 被重建而产生收据哈希
  漂移，已写 `INVALID_RECEIPT.md` 并禁止作为正式证据；runner 已改为安装前锁定 APK
  哈希。R3c result/gate SHA-256 为
  `3F9FFCE6B424E44356F0A16D312DE37715CAA3161D346D26373A12C4D0E87311` /
  `33225988C60C0F45CE90A3F384FD9473EE7A3A0A036C90D448289516B9535DBF`。
  当前只支持单设备部署与性能诊断；accuracy、false-clear、产品和 safety authority 不变。
- 同日对同一 cached DLC 完成冷态 QNN `detailed` 与 HTP `linting` profiling。detailed 24 次
  execution、每次 470 ops，算子/root cycle closure error `5.8e-14%`；Transformer encoder
  占 88.24%，其中 Softmax-attributed attention composite 56.15%、MatMul 16.17%、
  LayerNorm 6.58%，reshape+transpose 仅 3.68%。linting 11 次 execution 的关键路径
  mean/P95 `117.14M/117.35M cycles`，Transformer 87.76%、MatMul 70.81%；73.57% 的
  summed-op cycles 同时标记 HVX+HMX+DMA，DMA inclusive 96.24%。日志无 DramToTcm、
  TcmToDram、SystemService 或 BlockZapOp，故只确认广泛 DMA 参与，不声称已证明 VTCM
  spill 或 DDR bytes 瓶颈。两种 profile attribution 层级不同，但共同否定 JNI/layout 为
  图内首要瓶颈。独立 accuracy/false-clear 门仍缺失，混合精度、小尺寸和 student 保持 HOLD。
- 同日完成 R4 direct-depth bridge：QNN FP16 direct output 在 Native thread-local workspace
  bit-exact decode，并按冻结 align-corners 映射写入 owned direct 640x480 depth slot；独立 Native
  geometry executor 直接消费该 slot。真实 QNN output 加全 65,536 half-pattern tiled fixture 共
  检查 614,400 aligned outputs，finite raw-bit/non-finite class mismatch 均为 0、最大误差 0，
  geometry 字段与拒绝 reason 严格一致。固定 APK 10 分钟 R4 达到 5.00 Hz、3026/3026
  `VALID`，QNN/direct-bridge/geometry/full P95 为 `94.82/12.83/17.88/120.16 ms`，fresh age
  P95 `132.67 ms`，thermal 0/2/2、pool 3/3、runtime failure 0；device-installed app/test
  APK 哈希与安装前收据一致。result/gate SHA-256 为
  `F04760F3F3F7970DEA729D88B714D357FFDC21102C79D7FBB33A8C2198EB37FD` /
  `D0E8C3CB330C1F4F5F5F85AB841B430822BBFC4BD22E651CD2DF44847FA601A4`。
  该路径只消除 Java raw/aligned 两份深度数组；Native decoded workspace、owned direct buffer
  及 backend 内部未知搬运仍保留，不称 zero-copy，不新增 accuracy、false-clear、产品或 safety authority。
- 时间：2026-08-06（Asia/Hong_Kong）；执行者：violjjet。将硬件、Android、延迟、视频流和稳定性迭代的默认测试节奏写入 `AGENTS.md`：先运行 10 秒 smoke，日常有指标回归优先运行 1 分钟短测；5 分钟仅用于阶段性正式基线、重大架构或固件变更、候选交付或用户明确要求。30–60 分钟压力测试不再作为默认步骤，仅在候选发布、重大稳定性变更或用户明确授权时执行。该规则只调整测试成本与默认时长，不降低结果身份、设备绑定、异常记录或证据边界要求。验证：人工核对规则位置和 Git diff；无代码或构建变更，未运行 Gradle。
- 时间：2026-08-08（Asia/Hong_Kong）；执行者：violjjet。完成 DepthART SelectiveScan G4-A/G4-B 真机里程碑：ADB 确认当前设备为 `SM-S9280 / SM8650 / arm64-v8a`（HTP v75），因此 v73 保留为 compile-only 工件并新增同源 v75 build。修复 HTP 双侧同名 package 注册（CPU prepare/validator + HTP DSP）、BOOL 参数 `Int32 Const` 合同与低维 tensor 左侧 4D backfill 后，单算子 `B=1/C=48/G=4/N=8/L=196` graph 在 QAIRT 2.47 完成 register、compose、finalize 与真实 HTP execute。nominal/accumulation/softplus-extremes 三组 primitive-oracle parity 全部在 `rtol=3e-5 / atol=3e-6` 内，max abs 分别 `7.45e-9 / 5.74e-7 / 2.98e-8`。当前签署 `G4-A_PACKAGE_REGISTRATION_PASS / G4-B_OPERATOR_PARITY_PASS_SM8650_V75`；完整 5-op graph、partition/fallback、性能和 Android/生产 authority 仍未评价。
- 时间：2026-08-09（Asia/Hong_Kong）；执行者：violjjet。Assistive Geometry A0 正式训练 runner 与 host 性能前门完成：runner 只读冻结 TRAIN，按 orientation bucket 执行 4×4 梯度累积，发布 guarded progress，并逐 epoch 原子写可恢复 checkpoint。真实 20-step `workers=0/1/4` pilot 的优化吞吐为 `0.4854/0.5453/0.4788 step/s`，选择 workers 1；三档前 8 个输入 batch SHA 一致，mean TRAIN loss 跨档跨度 `0.0003923`，但 CUDA 权重不签署 bit-exact。每 seed 6,000 steps 外推 `3.06h`，诊断上界 `4h`。当前只授权 seed 17 guarded TRAIN-only execution；Development/Confirmation、A1–A4、teacher、部署、默认 App 和 safety 均保持关闭。
- 同日修复 guarded host preflight 对 `artifacts.local` 治理 junction 的误判：路径门现在先要求逻辑路径进入仓库内 `artifacts.local/`，再验证物理解析仍位于该 junction target 内；继续拒绝 `..` 和非 artifact 终态路径。新增回归后 8 个 validator tests 与 guarded launcher integration 均覆盖该边界。
- 同日完成 Assistive Geometry A0 训练后 evaluator 的合成 dry-run：严格验证 seed `17/29/43` 共 12 个 epoch `5/10/15/20` checkpoint 及 `1499/2999/4499/6000` 累计步数、外部/内部 SHA、状态与 RNG；九格/parent/orientation 指标保留 UNKNOWN 和全局零分母语义，三 seed 只做全量统计与每项 2/3 门，不选择 best seed。通过路径与 checkpoint 缺失、协议漂移、缺 horizon、零分母、coverage 塌缩、best-seed 企图共 7 个场景均命中预期终态并生成 JSON/短报告/失败相邻日志；全程未打开 Development/Confirmation outcome，不产生模型质量或默认 App authority。
- 同日完成 Assistive Geometry forward hypothesis 的 `WILD_LAB / CANARY_LITE` 数学审查与纯合成 CPU canary。保留 censored robust-contact survival、profile-conditioned swept configuration clearance、maximum-bottleneck corridor loss 与 cluster-level one-sided conformal/CRC 四个可证伪方向，优先级 H1>H2>H3>H4；否决 crop/K equivariance、普通 SDF sweep、普通 topology loss 和 vanilla conformal 作为 standalone novelty。canary 证明 hazard occupancy nesting、body-profile clearance monotonicity 与 equal-band/different-topology 反例，并显示 5% iid conformal miscoverage `.05275` 在 shift 下升至 `.31995`；当前 4 个 calibration parents 的最佳 finite term 为 `.20`，无法支持 8% CRC，至少需 12 个独立 parents。4/4 focused tests 通过；全程未读模型、checkpoint 或任何数据 role outcome，不改变 A0–A4、seed 17 successor、默认 App 或 safety authority。
- 同日 seed 17 A0 guarded TRAIN-only 正式训练以 `COMPLETE` 收口：20 epochs、6000 optimizer steps、最终 carry 清空，epoch `5/10/15/20` 四个留存 checkpoint 均通过独立 SHA 与 CPU load；最终模型状态 SHA-256 为 `DD6C4D3F5DA1C88978CA52CFCD648A5DAA4F02940D9B069B83505CF1F8C78868`。训练未打开 Development/Confirmation，也未导入 teacher。随后在 seed 29 执行期间冻结并实现真实 Development evaluator v2：只允许三 seed 完整后物化四个 `DEVELOPMENT_SELECTION` parent，A0 走 predicted dense depth 加冻结 gravity/geometry reader，truth/pred clearance validity 独立，任务门补齐 ground recovery、clearance coverage、valid-to-UNKNOWN 与 geometry transition；10 个 focused tests 和 protocol binding 验证通过，Development outcome 仍未打开。
- 同日新增 A1–A4 outcome-blind additive-arm 通用训练 mechanics。四个 arm 均从同一 DepthART 初始化、相同 TRAIN roster、seed、增强、步数与调度独立训练，不串行继承前一 arm checkpoint；A1 只开放 ground modules，A2/A3 开放 ground/clearance/occupancy，A4 最后开放 confidence，并按冻结 loss 集严格累加。4 个 focused tests 与 py_compile 通过；该实现尚无训练激活权限，A0 Development outcome 前不得运行。
- 同日新增 Assistive Geometry 移动导出 mechanics：选定 checkpoint 可导出固定 portrait `608x448` 或 landscape `448x608` ONNX，并显式输出 dense depth、ground logits、clearance、occupancy logits 与 confidence logits；相机 prompt 由 host 按动态 K 计算，gravity/UNKNOWN/task postprocess 保留在图外，避免把缺失几何填成 clear。wrapper shape/parity 3 个 focused tests 与 py_compile 通过；尚未选模、未运行 QAIRT/HTP，不改变 strict G4-D 负终态，也不产生部署/性能/产品 authority。
- 同日冻结 Assistive Geometry C0 异质教师互补性 kill gate mechanics：未来仅在另行授权的 truth-bound cohort 上比较 metric teacher 与 temporal geometry teacher 的单体、oracle、独占正确 parent、分歧错误浓度和 temporal clearance delta。oracle clearance 相对增益不足 5% 且 false-clear 绝对改善不足 1 个百分点、任一教师独占正确少于 2 个 parent、分歧区错误率超额不足 10 个百分点或时序教师 delta MAE 优势不足 0.01 m 时，任一条件均停止 C1。3 个 focused tests 与 py_compile 通过；当前教师 identity、cohort 与输出均未授权，不得执行真实评价或蒸馏。
- 同日冻结 Assistive Geometry D0 时序消融 mechanics：在统一 8-frame GeometryState、48 hidden 和 50k 参数预算下比较 GRU、因果 TCN 与不作 Mamba 主张的 diagonal SSM，统一输出 future-clearance delta、TTC 和 raw compute-gate logit，最终三态与 UNKNOWN 权限保留在 host。未来扰动因果测试曾发现 GroupNorm 跨时间泄漏，原样失败后改为逐时间点 LayerNorm，最终 4 个 focused tests 与 py_compile 通过。当前稳定单帧候选、新 TEMPORAL_DEVELOPMENT/Confirmation cohort 和 truth materializer 均缺失，不授权打开 outcome、训练、部署或时序收益主张。
- 同日 seed 29 A0 Attempt 01 在 epoch 7 backward、2097 optimizer steps 处收到 CUDA OOM；guard 写出 `FAILED_WITH_RECEIPT`，Development/Confirmation firewall 仍为 false。epoch 6 / 1800 steps 的原子 `latest.pt` 可 CPU load，未落盘的 297 steps 不作为结果。因原 runner 未冻结 partial-epoch resume，Attempt 02 保留同一 seed、TRAIN roster、DepthART 初始化、模型、optimizer 与 schedule，从共同初始化完整重跑；只新增 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` allocator 防碎片环境，输出转入新 r2 根，旧失败目录保持不可覆盖。
- 同日冻结 Assistive Geometry M0 任务保持型移动部署 mechanics：选定模型后须双 shape ONNX 外部 camera parity、五 SelectiveScan、单 fixed-mixed QAIRT 2.47 recipe、SM8650 HTP v75 全图无 CPU fallback；现有 DepthART D1 roster 因明确排除 Assistive Geometry 不得复用，必须建立新的 8-primary + 8-reserve MOBILE_DEVELOPMENT cohort。raw parity 与 coverage/clearance/false-clear/false-block/temporal/transition/UNKNOWN 全部门通过后才允许测 `QNN P95 <=150 ms`、full GeometryState `P95 <=180 ms` 和 `>=5 Hz`；当前无选定模型、转换、设备或任务保持 authority，strict G4-D negative 不变。
