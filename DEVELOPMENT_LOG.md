# Development Log

Current window: 2026-08. Historical entries: [2026-07](docs/history/development-log/2026-07.md).

- 时间：2026-08-21（Asia/Hong_Kong）；执行者：violjjet。完成固定五个 `NO_CANDIDATE` 窗口的
  `ADT1_REAPPEARANCE_OBSERVABILITY_DIAGNOSTIC_R3`。五窗分别独立注入 GT-derived proposal，正式 RGB
  evaluator 对 oracle output fail closed；原 TargetMemory/verifier/2-of-3 最终重捕获 `4/5`，延迟为
  10/42/20/17 帧。唯一失败窗只有 1 帧、3×3 px、visibility 0.10，2-of-3 在定义上不可能完成。
  contact sheet 与逐帧统计把五窗分为 2 个不可见/重遮挡、3 个太小；640 observer 输入上的 bbox 最短
  边最大值均小于 10 px，无尺度足够 model miss 或同 prototype identity ambiguity 证据。五窗合计
  GT-invisible 780 帧、GT-visible 但低于尺度-可见性 proxy 177 帧、detectable-but-missed 0 帧；该合计
  不与全局 longest-dropout 直接相减。停止 DINOv2/SAM/Sky，唯一 successor 为只改变搜索尺度并使用新
  duration 分解的 `ADT1_SMALL_TARGET_SEARCH_SCALE_R4`。

- 时间：2026-08-21（Asia/Hong_Kong）；执行者：violjjet。完成 R1 failure accounting 与单变量
  YOLOE-26n visual-prompt candidate canary。R1 的 5 个失败 opportunity 全为 `NO_CANDIDATE`，
  `CANDIDATE_REJECTED/CONFIRMATION_FAILED=0/0`，GT-visible LOST candidate recall 为 `34/405=0.0840`。
  保持 TargetMemory、2-of-3、弱先验、quarantine、flow5 和 evaluator 不变，仅替换 LOST proposal 后，
  candidate recall 降为 `29/423=0.0686`、@30 从 `0.4` 降至 `0.2`、longest dropout 从 159 增至 164，
  5 个失败仍全为 `NO_CANDIDATE` 且 wrong-instance 为 0。终态为
  `YOLOE_26N_VISUAL_PROMPT_NOT_SUPPORTED`；不进入 DINOv2/SAM/Sky，唯一 successor 是只诊断目标尺度、
  遮挡与 RGB 可辨识性的 `ADT1_REAPPEARANCE_OBSERVABILITY_DIAGNOSTIC_R3`。

- 时间：2026-08-21（Asia/Hong_Kong）；执行者：violjjet。完成
  `BA-ADT-INSTANCE-REDETECTION-1` Development canary：在同一已消费 `seq136 / Carrot_A`
  上，为 YOLO11n + flow5 observer 增加最多 5 个 trusted crop 的 RGB appearance memory、弱尺度/
  空间先验、2-of-3 确认与重接后 8 帧记忆隔离；evaluator v3 新增 @90/@180、median delay、
  correct/wrong/unresolved instance redetection 与 ID-switch 指标。held-forward recall/mean IoU 从
  `0.5808/0.4469` 提至 `0.6203/0.4743`，false-visible 保持 `0.0073`，13 次实例重检测为
  `13 correct / 0 wrong / 0 unresolved`；但 @30/@90/@180 仍为 `0.4/0.5/0.5`，最长 dropout
  仅从 162 降至 159。终态为 bounded Development utility，长时重捕获仍未解决；唯一 successor
  是保持状态机/evaluator、只升级 YOLOE visual-prompt candidates 与 DINOv2 identity embedding 的
  `ADT1_LEARNED_INSTANCE_REDETECTION_R2`。Sky、冻结 Goal Copilot、默认 App 与产品/安全权限不变。

- 时间：2026-08-20（Asia/Hong_Kong）；执行者：violjjet。将 `BA-ADT-REAL-EVIDENCE` 的主线地位从状态
  标签提升为仓库级执行约束：它是 BlindAssist 唯一 active 产品与研究主线；当前 successor 只由 Goal
  Copilot current 与算法研究入口共同声明。D-ORACLE、SVRF、Assistive Geometry、TARO、SATOM、
  DepthART、旧 GC/Sky search 和 Android/default-App promotion 全部只保留为历史/关闭/暂停上下文，未经
  用户显式改变主线并先更新两份 current，不得自行恢复或占用执行预算。同时清除算法入口中残留的
  “SVRF 唯一算法主线”“D-ORACLE 唯一 active P0”和 DepthART `R1_RESEARCH_MAINLINE` 冲突标记；Sky
  仍只允许在真实 RGB failure 明确归因到 policy 层后另立任务。

- 时间：2026-08-20（Asia/Hong_Kong）；执行者：violjjet。将 BlindAssist 当前 successor 切换为
  `BA-ADT-REAL-EVIDENCE`：系统侧只允许真实 ADT RGB，ADT bbox/trajectory/depth/visibility GT 只允许
  进入隔离的 episode mining/evaluator；recorded ADT 不得解释为用户受引导后的 closed-loop navigation。
  当前只激活 ADT-0，并新增有界 sample acquisition（RGB preview + main GT 小于 32 MiB）与 GT-only
  Goal Episode Miner。ADT-1 RGB Observation、ADT-2 prerecorded guidance、ADT-3 policy failure benchmark
  依次后置；Sky、GC2-C、held-out、Android/default-App、产品和安全权限均保持关闭。
  官方 10 秒 sample 实跑得到 300 个 GT RGB timestamps、106 个 bbox target 和 102 个持续跟踪候选；
  search/acquire/track/lost/reacquire/approach 候选数分别为 18/102/102/35/35/2，但没有单一目标覆盖
  完整六阶段。结果因此为 `ADT0_SAMPLE_EPISODES_MINED / PARTIAL_EVENT_COVERAGE`，下一步固定门槛
  选择少量完整 sequence；不把 visibility gap 冒充已确认 tracker failure，也不为凑结果调门。
  随后复用已披露 consumed GT geometry prescreen 仅作 Development 优先级，固定门槛挖掘
  `clean_seq134/136`，分别得到 172/134 个六阶段候选；选中 `seq136 / Carrot_A`（1,502 visible
  frames，GT center-range proxy 约 4.59→1.77 m）作为 ADT-1 demo 目标。后续修正早期 evaluator 的
  ADT→preview 坐标错误：正确变换为 90° clockwise，而不是 y-axis flip；sample `WoodenBowl`
  held-forward recall 更新为 0.2488，仍暴露多实例 target grounding/association failure。官方 manifest
  transport 恢复后，按 manifest SHA-1 下载 114,143,011-byte `seq136` preview RGB；RGB-only YOLO11n
  `carrot` observer 处理 3,824 帧。仅用前 25% timeline 选择固定 506-frame offset、其余 2,160 帧评价：
  localization recall 0.4041、false-visible 0、最长 dropout 177 帧、normalized bearing MAE 0.01234、
  bbox-scale correlation 0.9681、approach-direction accuracy 0.9091，10 次 eligible reacquisition 中 30 帧内
  成功率 0.4。定位成功时 evidence 明确有效，但 visibility/tracking/reacquisition 尚不足，失败不在 policy，
  不授权 Sky。已将同一 observations 接到 SHA-256 固定的 GC1 winner，生成 72 秒 prerecorded ADT-2
  Development demo、guidance timeline 和 evaluator-only GT overlay；bearing/nearness 仅为 proxy，clearance
  与 completion fail closed，不能称为闭环导航。随后在同一 consumed Development sequence 加入 RGB-only
  sparse optical flow：30-frame persistence 虽把 recall 提至 0.6767，却使 GT-invisible false-visible 升至
  0.0940，故拒绝；原生 5-frame candidate 达到 recall 0.5808、mean IoU 0.4469、false-visible 0.0073，
  准入 Development demo。它仍有 162-frame longest dropout，30-frame reacquisition success 仍为 0.4，
  因此 M1 tracking 尚未建立。唯一下一步为 `ADT1_INSTANCE_CONDITIONED_REDETECTION`，Sky 继续关闭。

- 时间：2026-08-20（Asia/Hong_Kong）；执行者：Codex。完成 Goal Copilot 2 的零模型 observability、
  failure-autopsy、counterfactual 与 reality audit；只使用 consumed GC2 dev scenarios、冻结 simulator、
  GC2-B 锁定公开 winner 和既有 device evidence，未读取 held-out。Moderate `0/12` 的首次偏离分散为
  stale evidence `4`、tracking collapse `3`、dropout `2`、bearing-jitter alignment bypass `2` 和
  false-target reverse search `1`；逐项
  去掉任一 corruption 最多只恢复到 `1/12`。Hidden oracle、完整 noisy-history lookup 与当前六函数面
  的 consumed memorization upper bound 均为 `12/12`，后两者只排除有限场景上的绝对语法不可达，
  不建立可迁移 policy/search signal。现有 4,422 帧真实世界 RGB Android device replay 不是 phone
  capture，且缺 target identity、tracking、bearing、nearness 与 capture-time mapping，故 real-phone
  grounding 为 `NOT_EVALUABLE`。决策选择 A：停止 synthetic moderate optimization，保持 GC2-C、
  held-out、新 Sky/模型调用、扩预算和 representation ladder 关闭；real-phone evidence capture/audit
  须另冻结 source/truth/timing/privacy/roster contract，当前不授权执行。默认 App、产品和 safety 不变。

- 时间：2026-08-20（Asia/Hong_Kong）；执行者：violjjet。正式将 Goal-Driven Visual Copilot 建立为
  BlindAssist 上位产品/研究主线，并建立独立新 lineage `GOAL-COPILOT-1`。V0 只做冻结 symbolic
  observation 的零模型 mock roundtrip，覆盖 FIND_AND_REACH、TRACK_AND_REACQUIRE、
  FIND_ALIGN_INTERACT；BA 独占 task/evaluator/sealed truth/safety/acceptance authority，SkyDiscover
  只有 proposal authority。SearchTaskBundle/CandidateBundle 均 content-addressed、逐文件 SHA-256、
  exact allowlist 和 source/protocol 绑定；BA 以完整 evaluator vector 独立输出 ACCEPT/REJECT/
  NOT_EVALUABLE，Sky score 仅作 provenance。状态为 `GOAL_COPILOT_SKY_BRIDGE_V0_MECHANICS_READY /
  GOAL_COPILOT_1_MODEL_SEARCH_NOT_STARTED`；现有 L10M 仅作先导机制背景，所有封存结果、run 根、
  terminal 与 claim ceiling 未修改。默认 App、真实用户、安全效果和产品权限不变。

- 时间：2026-08-17（Asia/Hong_Kong）；执行者：Codex。将 `D-ORACLE-1` 冻结为 Failure Synthesis
  后的唯一 P0，状态为 `PROTOCOL_FROZEN / NOT_EXECUTED`。协议只含三个正式 arm：independent direct-action
  oracle、perfect source geometry→current policy、estimated representation→同一 policy；B/C 的 policy、
  threshold、feature contract、coverage rule、evaluator 与 parent denominator 完全相同。归因只使用 matched-
  coverage 的 `U(A)-U(B)` 与 `U(B)-U(C)`；另加 parent-local geometry permutation 作为非竞争机制 control。
  action truth 与 event-evaluation truth 分账盲化，统计单位为 parent，并冻结 native/matched coverage、paired
  delta、median、worst-parent 与 stratified bootstrap CI。SVRF 和其他 representation/search successor 暂停；
  只有观察到 `A` materially 高于 `B` 后才允许另行设计 H3/H4 分离的 D-ORACLE-2，本轮不增加 arm、
  不执行 cohort，也不改变默认 App、产品或 safety authority。

- 时间：2026-08-17（Asia/Hong_Kong）；执行者：violjjet。完成 BlindAssist Failure Synthesis /
  Global Reckoning，只读汇总既有 Git 历史、current、冻结 result/protocol 与 artifact bindings，未重跑
  benchmark、未修改算法或历史终态。正式签署 `BLINDASSIST_FAILURE_STRUCTURE_IDENTIFIED`、
  `SEARCH_CONCENTRATION / WRONG_LEVEL_OPTIMIZATION` 与 `PROXY_TARGET_ALIGNMENT_NOT_ESTABLISHED`。
  根因排序为 H3 target/supervision、H4 downstream objective/policy、H2 representation、H1 input/
  observability；H3/H4 的单一主因仍待 oracle decomposition。最高价值 successor 不是新模型，而是
  `D-ORACLE-1 ACTIONABILITY→GEOMETRY→REPRESENTATION LADDER`：同一 fresh parent cohort 比较 direct
  action oracle、source-native geometry→冻结 current policy、estimated representation→同 policy。
  在该诊断前停止为新 encoder/loss/fusion/depth/selector、B1/A0、Q-Plane、TARO rescue 或 proxy-only
  win 分配预算。默认 App、产品与安全 authority 不变。

- 时间：2026-08-13（Asia/Hong_Kong）；执行者：violjjet。完成 Assistive Geometry factor-wise no-regret
  Development 回合。11-parent frozen replay 中 perfect signed-advantage oracle coverage `80.30%`，确认
  correction 存在安全 headroom，但旧 learned selector 的 Bonn/TUM 外部结果保持失败。新增 source-diverse
  pixel candidate（显式 `0.05`，不篡改 checkpoint 的 `1.001` fallback）以及 neural quantile ensemble +
  cross-parent kNN LCB frame veto；14-parent calibration coverage `44.45%`、MAE delta `-0.7496 m`、bad-rate
  delta `-0.1120`。两组各 3-parent consumed TUM 诊断都把 harmful parent 降为 0，但均只有 1/3 parent
  非零 coverage，严格门失败。随后把全部 14 个 consumed TUM parent 降级重分为 11 fit / 3 calibration
  训练 TUM-only gate；oracle 仍有 `39.27%` coverage，但所有 kNN-LCB 候选均只能全回退，终态
  `AG_FACTORWISE_NO_REGRET_ORACLE_HEADROOM_RUNTIME_OBSERVABILITY_FAIL_STOP`。因此停止用同一 observable
  重训，唯一 successor 改为 `AG_RUNTIME_CORRECTION_GAIN_OBSERVABILITY_CANARY_R0`：先检验 temporal
  reprojection/model uncertainty 的 leave-one-parent-out 可观测性。三份此前未引用 TUM identity 已在任何
  payload/model/outcome read 前冻结；因当前候选全回退未消费。R21 boundary、support/UNKNOWN、reducer、
  ETH3D Confirmation、Android/default App、产品与安全权限全部不变；17 个 focused tests PASS。

- 时间：2026-08-13（Asia/Hong_Kong）；执行者：violjjet。完成 TARO task-directed observability 的 Bonn
  positive-oracle R1 可执行 canary 与数据分母收口。runner 先只读 timestamp/path/pose，以官方
  `T_ROS * T_groundtruth * T_ROS * T_marker` 坐标链做 outcome-blind source audit；26 个 parents 中 25 个
  具备合法 pair，从全序列均匀选择 100 references，selection image reads=0，pair capability PASS。五臂保持
  static/passive/fixed `6±2 cm, <=5 deg` micro/generic max-parallax/task oracle 的同一额外帧预算，输出仅
  `OCCUPIED/UNKNOWN`。实际评价 56 references / 504 queries，44 geometry abstentions；source-derived truth
  为 `404 OCCUPIED / 2 CLEAR / 98 UNKNOWN`，static-miss positive opportunity 与 CLEAR denominator 都只覆盖
  1 parent，低于冻结的 4/4 parent 门。虽然 passive、micro、task oracle 表面均恢复 2 个 positive，passive/
  micro 同时 false-occupy 2/2 CLEAR，所有臂间 decision 因分母不足保持 null；终态
  `NOT_EVALUABLE_DATA_OBSERVABILITY_DENOMINATOR`，不授权 learned scorer、Android、产品或默认 App。
  TartanAir JapaneseAlley 本机只有 128-byte Hugging Face metadata、原 archive 已按 cleanup record 删除，
  未冒充可运行 payload。
  唯一 successor 收紧为 `TARO_TASK_OBSERVABILITY_BALANCED_POSE_SOURCE_FRONTDOOR_R0`：先冻结 pose/depth/
  intrinsics/label contract 并满足 48 references、4 recovery parents、4 CLEAR parents，再允许另立五臂 R2。

- 时间：2026-08-13（Asia/Hong_Kong）；执行者：violjjet。完成 DepthART-S D3R3→D3R6
  source-support 与 selective-risk Development 回合，并按用户要求暂停。D3R3 fresh census 全量完成
  exact 64/64 GET（`5,580,879,686` bytes），exact 9,600 stems 中 9,597 个 depth+confidence paired；
  3 个缺失 stem 保持 `SOURCE_UNAVAILABLE_UNKNOWN`。旧 per-parent all-horizon 双向支持为 `0/32`，
  继续扩 21 个同分布 parent 仍未增加 far-CLEAR diversity，排除了“只加同分布身份即可解决”假设。
  D3R4 578-parameter 双头 router 在 parent-disjoint Development 上把 false-clear `34.39%→8.92%`，
  却把 false-block `8.62%→46.78%`；D3R5 加入 parent×band×horizon 相对秩并将 veto 对齐
  baseline-CLEAR 后，在首组 8 个 fresh parent 仍把 false-block `10.60%→27.92%`，两项 direct-veto
  机制均判负并保留。D3R6 只保留该风险排序，把动作改成每 parent 最多 `54/2700=2%`
  baseline-CLEAR cell 转 UNKNOWN，永不直接输出 CLEAR/OCCUPIED；budget 仅由 TRAIN 冻结。第二组 8 个
  此前未读 DepthART 输出的 fresh parent 上，432 个 deferral 中 335 个 truth OCCUPIED、0 个 CLEAR、
  97 个 truth UNKNOWN；false-clear `31.73%→29.92%`，false-block `18.92%→18.92%`，coverage
  下降恰好 `2.00%`，签署 `D3R6_BUDGETED_UNKNOWN_DEFERRAL_FRESH_CONFIRMATION_PASS`。结果
  `14,408 bytes / SHA-256 B089A050...55EDC`。D3R6 仅为 Development candidate；R2 outcome 未读，
  不产生设备、性能、默认 App、产品或安全 authority；当前 successor 为 `NONE / USER_PAUSED`。

- 时间：2026-08-13（Asia/Hong_Kong）；执行者：violjjet。按用户确认把新增算法预算收敛为两条可由
  Codex 并行、无固定天数的 evidence-gated Development 支线。Assistive Geometry 激活
  `AG_FACTORWISE_NO_REGRET_ORACLE_AND_PARENT_GATE_CANARY_R0`：先比较冻结 DepthART prior、correction
  expert、perfect signed-advantage oracle 与现有 selector，只有 oracle 有安全 headroom 才继续训练
  advantage-LCB router；selector admission 已收紧为每个 parent 的 MAE/`>0.10 m` error 双指标 no-regret，
  且至少一半 parent 具有非零 coverage，macro 改善不得掩盖单 parent 受伤。TARO 激活
  `TARO_TASK_DIRECTED_OBSERVABILITY_POSITIVE_ORACLE_CANARY_R0`：只做同一额外帧预算的 static/passive/
  fixed-micro/generic/task-directed positive-occupancy oracle，不开放 `CLEAR`。新 source-only pair audit 对
  R7 170 帧与 R10 710 帧确认 pose 完整，但相邻最小间隔均为 2 秒，1 秒内合法 pair 均为 0；因此停止在
  这两个 cohort 上训练时序模型，下一步改用 outcome-blind 选择的 pose-rich consumed Development source。
  R11 sealed top24 与 protected outcomes 保持不读不改；FARO Phase-B implementation lock 已完成，但正式
  outcome 未读，Formal execution 暂停且不阻塞 Development；默认
  App、产品和安全主张不变。AG/TARO 11 个 focused tests PASS，pair audit 的 depth/FARO/truth/model/network
  reads 均为 0。

- 时间：2026-08-13（Asia/Hong_Kong）；执行者：Codex。按用户授权正式消费 AG R2 cross-sensor
  calibration-control R1 producer 与 independent replay，各恰好一次。execution lock 在任何 archive access
  前以 master `3a4247dfa022323ca4f36f574bc607c4ff252b05` 固定；producer 先消费新 root，验证 archive hash、
  枚举 5 个成员并读取 2 个 YAML / 7,236 bytes，随后以 `F2_R1_KALIBR_ROSTOPIC` fail closed。两份 YAML
  已读但解析未完成，因此 matrix discovery 与 `/uvc_camera/cam_2` match count 都保持 `null/UNKNOWN`，未选择
  member、camera node 或 first/best。独立 validator 先消费 replay receipt，再且仅再 replay archive 一次，
  以 `CALIBRATION_CONTROL_R1_INDEPENDENT_REPLAY_CONFIRMED_PRODUCER_FAILURE` PASS；离线六文件 hash-chain
  验签通过。R1 root 为 6 files / 6,462 bytes，producer failure SHA-256 `35F125C2...D40D38`，validator result
  SHA-256 `57802146...ED15D`。session RGB-D/IMU、模型、truth、factor scoring、Confirmation、训练、reducer、
  网络、设备与默认 App 全部为 0，Confirmation root 不存在，科学状态为 `NOT_RUN`。R1 永久 consumed，
  不得 rerun/resume/replace；当前没有 active successor，未来恢复必须另立基于官方 archive-format evidence 的
  新版本 non-executing protocol、fresh root 并单独授权。

- 时间：2026-08-13（Asia/Hong_Kong）；执行者：Codex。TARO R11 Phase-A repaired independent audit 对同一
  immutable 5,219-file root 完整 PASS：48 parents / 1,043 frames / 9,387 queries，原 validator 的 root set、
  5,218 prior hashes、64 execution-lock bindings、source containers/payloads、candidate arrays、lineage、counts、
  read ledger、runtime/resource 全部一致；独立 source payload replay 为 4,172 次。repaired audit 原子 root 恰好
  1 file / 3,035 bytes，SHA-256 `2D80268D...78D19C`。numeric repair 仅把独立重建 pose/gravity 按 producer
  frozen canonical JSON round-12 后作精确比较，无 epsilon。模型未重跑，highres/FARO/truth/label/outcome、
  R9 scoring/top-24、training/network 均为 0。Phase-A independent-validation blocker 与 pipeline hold 正式关闭；
  唯一 successor 为 non-executing `TARO_O1R_R11_SOURCE_ONLY_TOP24_IMPLEMENTATION_LOCK`，正式评分仍须独立 execution lock。

- 时间：2026-08-13（Asia/Hong_Kong）；执行者：violjjet。D3R2 coverage census r0 在第45个资产
  short body 后保持不可变，另立 D3R3 transport-recovery version。新 scope 保持 exact-32 identity
  顺序、每身份 exact-300 stems、9600-stem plan 和 64 URL 顺序，但不继承 D3R2 activation、44 bodies、
  checkpoints、failure/temp 或 partial coverage。fresh HEAD 64/64 PASS（声明正文 `5,580,879,686` bytes），
  Content-Length/ETag/Last-Modified 相对旧 snapshot 全部零漂移、redirect/body read 均为0；producer-free
  validator PASS。因此把 D3R2 的失败假设收窄为 premature EOF，而不是源对象版本变化。D3R3 census
  只增加一个可证伪变化：HTTP 200 且 headers 匹配但正文短于 Content-Length 时记
  `TRANSIENT_BODY_SHORT_READ`，删除 partial 并从 byte 0 完整重试，最多3次；15/15 targeted tests PASS。
  exact-64 fresh-root census 已激活，但仍禁止 Range、member payload、pixel/truth、selection、RGB/model/R2。

- 时间：2026-08-13（Asia/Hong_Kong）；执行者：Codex。主机重启后 RTX 5060 / CUDA 12.8 恢复，针对同一
  immutable TARO R11 Phase-A 5,219-file root 重跑原 independent validator。原 validator 通过 CUDA 前检并
  重验 terminal/source，但在首帧 `466160/44796584/17383.777` 以
  `R11_PHASE_A_VALIDATION_SOURCE_BINDING` 停止：producer 通过冻结 canonical JSON 把 pose/gravity 序列化为
  12 位小数，validator 则把重建的 float64 值在序列化前作 Python exact equality；例如
  `-0.403235180695` 对 `-0.4032351806954706`、`-0.077485681602` 对
  `-0.07748568160183153`。两组独立重建值按冻结 round-12 规范化后与 stored 数值的 canonical SHA 均为
  `B3CCB272...49574C7`，证明当前故障只属于 numeric representation，不是 source/evidence corruption。
  原 validator、execution lock、terminal 和 Phase-A root 全部保持字节不变，模型/FARO/highres/truth/label/
  outcome/scoring 均未重跑或读取。新增并冻结 protocol-only repair：只在原 validator 的 independent trajectory
  重建后规范化 `camera_to_world_4x4` 与 `gravity_up_camera_xyz`，随后仍作精确比较；无 epsilon、无 schema/hash
  bypass，原有 5,219-file/root/source/candidate/lineage/ledger/resource 检查全部保留。正式结果先在同卷 sibling
  partial root 完整写入并回读 seal/bytes/SHA/单文件集，再原子发布整个 root；注入 fsync 故障时 formal root 保持
  absent 且 partial 清理。10/10 focused tests 与
  py_compile PASS。Attempt 01 推送后首次调用在创建 output root 或读取 Phase-A frame payload 前以
  `R11_PHASE_A_REPAIR_PATH` fail closed：内部 absent output 保留 repo lexical `artifacts.local` spelling，CLI exact
  path 则 resolve 到授权 junction target，二者被误判为不同。正式/partial root、模型、source-frame payload、
  FARO/highres/truth/label/outcome 均为 0。Attempt 01 原字节保留；Attempt 02 只把 exact CLI 与 exact authorized
  path 两侧 resolve 后作 exact equality，alternate target 仍拒绝。当前唯一 successor 是推送 Attempt 02 后对同一
  sealed root 执行只读 post-terminal audit；只有该结果 PASS 才可另立 source-only top-24 lock。

- 时间：2026-08-13（Asia/Hong_Kong）；执行者：Codex。消费用户明确授权的 D3R2 Phase-B
  exact-64 coverage-only census。activation 先通过 PR #34 的 11 项检查并合并至受保护 master；正式 r0
  随后完成连续 `44/64` asset checkpoints / 22 paired identities，保留 44 bodies / `4,223,537,610`
  bytes。第45个 `44796744/lowres_depth.zip` 首试 HTTP 200，但流式正文长度与冻结 Content-Length
  不一致；producer 封存 terminal `DownloadFailure <- ValueError: download length mismatch`、failure
  sidecar 与 non-resumable temp marker 后停止。metadata-only auditor 验证 attempt、001..044 checkpoint
  seals、failure、HEAD URL/header/length 与 source 名称/长度，body read/hash、ZIP open/member read 均为0。
  无 manifest/validation，partial coverage 不发布，truth/selection 未打开；终态
  `D3R2_PHASE_B_COVERAGE_CENSUS_EXECUTION_INVALID_INCOMPLETE / scientific_terminal=null / next_gate=null`。
  当前 r0 不可 resume、修补、复用 partial assets 或同版本重跑；未来恢复须再次授权新版本/协议/root。
- 时间：2026-08-13（Asia/Hong_Kong）；执行者：Codex。收到 R1 calibration-control one-shot 授权后，
  在真实 archive/root 访问前完成 formal pre-execution hardening。独立审计发现旧 replay 只声明一次、未先消费
  receipt，且未完整验签 R1 lock/start/terminal/manifest、非 namespace failure 与 downstream Confirmation
  binding；本次只修复这些证据链，不改变 `/uvc_camera/cam_2` 选择、数据 identity、budget 或科学门。
  replay 现在先原子写入独占 start receipt，再且仅再打开 calibration archive 一次，并封存独立 terminal/manifest；
  所有 producer fail-closed 类均可按相同 ZIP/parser 合约重放，reseal mutation 与二次 replay 均 fail closed。
  最终 Confirmation execution-lock validator 只接受 R1 producer PASS 加 independent replay PASS 的完整 hash chain。
  69/69 cross-sensor tests、专项 ruff、compile 与 R1 repair-lock validator PASS；真实 archive/member、session、
  model/checkpoint、truth、factor scoring、Confirmation 与 Confirmation root 仍为 0，科学状态仍 `NOT_RUN`。
  用户授权的当前唯一动作是先提交该 implementation boundary，再签发并各消费一次 R1 producer/replay；R0
  继续永久 consumed，禁止 rerun/resume/replace。

- 时间：2026-08-13（Asia/Hong_Kong）；执行者：violjjet。按用户对唯一 successor 的授权，完成 AG R2
  cross-sensor calibration-control R0 failure audit 与 R1 protocol repair implementation lock；本步无 archive
  执行权限。只读复核 R0 start/failure/manifest 三文件及 hash chain，确认终态有效但 failure 未保存 YAML
  candidate、matrix discovery 或 target-match count，因而零个/多个不可恢复且不得事后重开补看。官方 ETH3D
  文档把 `imu.txt` 绑定到 RGB/depth 同视点右 RGB 相机及 `/uvc_camera/cam_2/imu`；Kalibr YAML 将每个
  camera node 的 image `rostopic` 与 `T_cam_imu` 同节点绑定。R1 因此不猜 `cam0/cam1`，而要求所有有界
  YAML discovery 中恰好一个 same-node rostopic namespace 为 `/uvc_camera/cam_2`；失败 evidence 保存完整
  candidate/read/discovery/target-match counts 与摘要，仍拒绝 first/best。producer-free validator 可独立重放 PASS
  与 failure；58/58 focused tests、ruff、compile、repair-lock validator 均 PASS。本步真实 archive/member、session、
  model/checkpoint、truth、factor scoring 与 Confirmation 访问全部为 0，科学状态仍 `NOT_RUN`。唯一 successor
  是另行授权、另立 hash-bound lock 与新 R1 root 的
  `BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_CONFIRMATION_CALIBRATION_CONTROL_R1_ONE_SHOT_EXECUTION_LOCK`；
  R0 不得 rerun/resume/replace，R1 仍不授权 session/model/Confirmation、产品或 safety。

- 时间：2026-08-13（Asia/Hong_Kong）；执行者：violjjet。按用户对唯一 successor 的单独授权，正式消费
  Assistive Geometry R2 cross-sensor factor Confirmation 的 calibration-control R0 one-shot。execution lock
  在任何 archive access 前固定为提交 `35f80eeac8c0c78f2576ef98a578ceacf0dc3fad`，仅授权 camera-IMU
  calibration archive 的哈希、成员枚举与最多 32 个、每个最多 4 MiB 的 YAML candidate 读取；session
  RGB-D/IMU、checkpoint/model、source truth、factor scoring、Confirmation、网络、设备、默认 App、产品和
  safety 全部无权限。exclusive control root 先于 archive 读取永久消费；archive bytes/SHA 匹配，枚举后终态为
  `F2_CALIBRATION_CONTROL_AMBIGUOUS_OR_MISSING_MATRIX`。failure evidence 未保存 exact candidate/discovery count，
  只能确定合法 discovery 不是恰好一个，不能区分零个与多个，因而没有绑定 exact member/camera node，也不得
  选择 first/best 或重跑补看。start/failure/manifest 三文件 hash chain 已复核；R0 one-shot 不得 rerun/resume/
  replace。三个 session archive、模型、truth、评分、Confirmation 与 Confirmation root 仍为 0，科学状态保持
  `NOT_RUN`，本终态不是 AG factor 的科学 PASS/FAIL。唯一 successor 为另行授权、无执行权限的
  `BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_CONFIRMATION_CALIBRATION_CONTROL_R0_FAILURE_AUDIT_AND_R1_PROTOCOL_REPAIR_LOCK`；
  只允许审计 sealed R0 failure、修复 failure observability，并在不重开 archive 前冻结披露既有 control access
  的 R1 protocol 与 official-evidence-backed camera-node selection contract。

- 时间：2026-08-13（Asia/Hong_Kong）；执行者：violjjet。正式消费 TARO O1R R11 all-48 source-only
  Phase A one-shot。producer 原子终态为 `TARO_O1R_R11_FRESH_POOL_PHASE_A_SOURCE_ONLY_SEALED_PASS`，封存 exact
  48 parents / 1,043 frames / 9,387 queries、1,043 次 DepthART inference 与对应 R7/R11 factors。R7 state 为
  `7,315 OCCUPIED / 2,072 UNKNOWN / 0 CLEAR`，R11 state 为 `7,313 OCCUPIED / 2,074 UNKNOWN / 0 CLEAR`，即冻结
  abstention 将两个 R7 positive 改为 UNKNOWN。evidence root 恰好 5,219 files / 959,553,693 bytes，terminal file
  SHA-256 为 `C4084BDB...73186`；wall 7,983.922 s、OS peak RSS 1,342,758,912 bytes、CUDA peak allocated
  140,934,144 bytes 与 terminal 前 evidence 958,520,288 bytes 均在冻结上限内。允许的 color/intrinsics/lowres/
  confidence payload 各读 1,043 次；highres、FARO、truth、label、outcome、training、network 与 R9 parent scoring/
  top-24 均为 0。独立 validator 随后只读重建 exact 5,219-file root set、重哈希全部 5,218 个 terminal binding，
  并验证 control seals、64 个 execution-lock binding 与 authority；但正式运行后 NVIDIA RTX 5060 进入 Windows
  `Code 43 / CM_PROB_FAILED_POST_START`，CUDA runtime identity 前检以 `R11_PHASE_A_VALIDATION_CUDA` 环境阻断，
  完整 1,043-frame lineage 与 4,172 次允许 payload decode replay 尚未完成。正式 root 未被 validator 改写且
  one-shot 不得重跑。当前唯一 successor 是重启主机恢复 CUDA 后对同一 root 只读重验；验签 PASS 前不得进入
  source-only top-24 或 selected-only FARO。本结果不产生 task、路线、部署、设备、产品或 safety 晋级。

- 时间：2026-08-12（Asia/Hong_Kong）；执行者：Codex。将 GitHub CodeQL 从无法完成 Kotlin
  autobuild 的默认配置改为仓库内可审查的高级配置：Actions、C/C++ 与 Python 使用无构建分析，
  Java/Kotlin 使用 JDK 17 和 `assembleDebug` 手动编译，覆盖 push、PR 与每周定时扫描。CodeQL action
  固定到完整 commit SHA；open-source readiness gate 现在要求该工作流存在、Java/Kotlin 保持 manual
  build，并用负向 canary 拒绝浮动的 `@v4` action 引用。此变更只建立静态安全扫描证据，不把扫描
  通过提升为设备、产品或安全结论。

- 时间：2026-08-12（Asia/Hong_Kong）；执行者：violjjet。建立面向外部贡献者的社区增长入口：新增英文
  三分钟 Quick Start、技术发布包、稳定 Android 架构图和“代码证据 → 设备证据 → 产品权限”边界图；发布包
  明确要求真实设备连续录制与元数据，缺少设备时不得伪造演示或把截图/benchmark 包装成用户效果。治理规则
  冻结 Contributor、Regular contributor、Triager/reviewer、Area maintainer、Core maintainer 的客观晋升阶梯，
  仓库权限遵循个人账户 2FA、保护分支、PR、required checks 与禁止 force-push。社区招募目标为 Android
  accessibility、on-device ML、Linux 构建复现和 reproducible evaluation 贡献者，不组织统一点 star；项目
  公共利益意图对应 SDG 10，但不据此声称已经取得社会、用户或安全结果。

- 时间：2026-08-12（Asia/Hong_Kong）；执行者：violjjet。依据用户对当前唯一 successor 的授权，完成
  Assistive Geometry R2 独立跨传感器 factor-level Confirmation 的非执行
  `CONTROL_FORMAT_AND_RUNTIME_BINDING_REPAIR_IMPLEMENTATION_LOCK`，科学状态保持 `NOT_RUN`。上一步
  activation blocker 不改写；execution schema 升级为 v2，必须 exact-bind Kalibr camchain YAML 的 camera
  node/`T_cam_imu` nested `4×4`、IMU→camera 方向、ETH3D mocap/time keys、IMU column/frame 与 stationary
  specific-force sign，并同时绑定 tracked official control evidence 与未来 calibration-control result。inline
  16-float、重复 matrix path、非正交旋转或无法唯一定位 camera node 均 fail closed。新增 calibration-control-only
  preflight：未来须另锁独立 one-shot/control root 后才可 hash/枚举唯一 camera calibration archive，最多读取
  32 个、每个 4 MiB 的 YAML candidate；没有或多于一个合法 matrix 均终止，不能选 first/best。producer-free
  validator 独立重哈希 archive、重枚举并重算 selection。另冻结 DepthART 实际可导入的 29 个 metric/
  selective-scan Python 文件，共 `160,284 bytes`，独立逐文件 bytes/SHA 复核通过；checkpoint/extension 保持
  分离 binding。51/51 focused tests、ruff、compile、source-manifest 与 repair-lock validator 全部 PASS。本步
  真实 archive file/member、RGB/depth/IMU/trajectory/calibration payload、checkpoint、model inference、source
  truth、factor scoring、Confirmation 与 Confirmation evidence root 均为 0。唯一 successor 为另行授权、
  另行 hash-bound 的 `BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_CONFIRMATION_CALIBRATION_CONTROL_PREFLIGHT_ONE_SHOT_EXECUTION_LOCK`；
  它只允许 camera calibration control，不授权三个 session archive、模型、reducer、设备、默认 App、产品或 safety。

- 时间：2026-08-12（Asia/Hong_Kong）；执行者：violjjet。用户授权只创建并冻结 Assistive Geometry R2
  独立跨传感器 factor-level Confirmation 的唯一 one-shot execution lock；activation preflight 在任何
  archive member access、model inference、source truth、scoring 或 evidence-root 创建前 fail closed，lock 未签发且
  one-shot 未消费。官方 ETH3D/Kalibr 控制证据表明 camera-IMU calibration 是 YAML `T_cam_imu` 嵌套
  `4×4` IMU→camera 矩阵，冻结 parser 却只接受同一文本行的 `<key> + 16 floats`；official-shaped synthetic
  control 稳定命中 `F2_IMU_CALIBRATION_MATRIX`。当前 exact-key execution schema 也不能表达 calibration
  encoding、transform direction、IMU column/frame/specific-force sign 或证据 binding，且 exact calibration
  member 因未枚举 archive 仍未知。11 个 runtime role 中 10 个本地候选完成 bytes/SHA preflight，必需的
  `blindassist.depthart.source_manifest.v1` 不存在；七个 archive direct-child 名称/bytes 仍与 data identity
  一致但未重哈希。正式状态为 `EXECUTION_LOCK_NOT_ISSUED_PRE_ACCESS_IMPLEMENTATION_REPAIR_REQUIRED`、
  `SCIENTIFIC_NOT_RUN`、`CONFIRMATION_OUTCOMES_UNOPENED`。唯一 successor 是非执行的
  `BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_CONFIRMATION_CONTROL_FORMAT_AND_RUNTIME_BINDING_REPAIR_IMPLEMENTATION_LOCK`；
  只允许修订 schema/parser、合成验证 Kalibr YAML、生成并独立复核 source manifest，以及实现未来另行授权的
  calibration-control-only preflight，不授权真实 member、模型、Confirmation、reducer、设备、默认 App、产品或 safety。

- 时间：2026-08-12（Asia/Hong_Kong）；执行者：Codex。冻结 TARO O1R R11 all-48 source-only Phase A
  one-shot execution lock，状态 `AUTHORIZED_UNCONSUMED / FORMAL_PHASE_A_NOT_RUN`。锁精确绑定已推送 implementation
  commit `79c976f8f5c919d4fa70563781eb6db498f69c6a`、64 个完整 repository/import/artifact bindings、48 parents /
  1,043 frames / 9,387 queries、正式 Python 3.11.9 + Torch 2.11.0 CUDA 12.8 + RTX 5060 runtime、DepthART
  source/checkpoint/preprocess/postprocess、R9 successor selector、用户 exact R11 authority、16 h / 16 GiB RSS /
  12 GiB CUDA / 2 GiB evidence ceilings，以及 `5,218 prior files + atomic terminal = 5,219` 的终态合同。runner
  static preflight 与不导入 producer 的 independent lock validator 均 PASS，content SHA-256 为
  `384B956CE67DF23812EA31DFD48B7D94CA2885CA83FA77C2E481BD90EC70B832`；正式 root 仍不存在，未读 source
  member、未运行模型、FARO/truth/training/network 均为 0。锁推送到 `origin/master` 前不得激活；推送后唯一动作
  是 formal CUDA 环境下 exact module argv，root 创建即消费，不覆盖、不续跑、不重跑。

- 时间：2026-08-12（Asia/Hong_Kong）；执行者：Codex。完成 TARO O1R R11 all-48 source-only Phase A
  implementation lock，状态 `LOCKED_NON_EXECUTING / SCIENTIFIC_NOT_RUN / FORMAL_PHASE_A_NOT_RUN`。独立 runner
  冻结 exact `48 parents / 1,043 frames / 9,387 queries`、`5F+3=5,218` terminal 前文件数与最终 5,219-file root；专用
  `PhaseAFrameRef` 只含 color/intrinsics/lowres/confidence 四种 capability，highres metadata 与 sealed index
  对齐后即丢弃。reader 在 ZIP lookup 前拒绝 highres 与跨 phase role，正式 success ledger 要求四种允许 role
  各 1,043 attempts/completed、highres attempts/completed=0。全部 candidates 先封存，随后逐帧封存 source、
  prospective、public reducer、R7 source/base factor 与 R11 abstention，并重载验证 R11 positive 是 R7 positive
  子集及 abstention 恒等式。独立 blocker 复核后又关闭四个 seam：runner 的全部 repository 传递 import/package
  closure 均由 implementation commit hash-bind，正式 runtime/candidate identity 精确冻结；trajectory SHA 与 parse
  复用同一 payload、正式 ledger 为 48 次；RSS 用 OS `peak_wset` 且 terminal 前重验 wall/RSS/CUDA/evidence；成功或
  失败只允许最后一个原子 `terminal.json`，并以 4 MiB 实体+逻辑 reserve 防止 result/manifest/failure 并存或末端
  预算耗尽。VRAM probe/failure sealing 不得静默降级。
  R9 selector/rule 只绑定为下一阶段 identity，本阶段 scoring/top-24=false。独立 validator 不导入 producer，
  将在正式执行后重建 exact file set、逐帧 source/inventory binding 并重算全部 candidate/lineage/count/ledger/
  resource seals；其独立 4,172 次 allowed-payload byte/decode replay 与 producer ledger 分开，highres/FARO=0。
  19/19 focused tests 与
  py_compile PASS；未读取 source member、未运行模型或创建 formal root，FARO/truth/training/network 均为 0。唯一 successor
  是另提交并推送 `TARO_O1R_R11_FRESH_48_PARENT_SOURCE_ONLY_PHASE_A_ONE_SHOT_EXECUTION_LOCK`。

- 时间：2026-08-12（Asia/Hong_Kong）；执行者：Codex。正式消费 TARO O1R R11 source inventory
  one-shot。exact 48/48 parents 均至少接纳 1 个 pose-bounded exact frame，共 `1,043` frames；compressed
  source `2,960,390,828 bytes`，central-directory 声明展开量 `3,540,113,101 bytes`，无 parent replacement。
  exclusive evidence root 恰好 4 files / 95,681 bytes。独立复核不导入 producer，重算 content seals、manifest
  bindings、48-row roster、exact-ns token 排序/唯一性、count/byte sums，并核对 144 container bindings 与 sealed
  download receipts，全部一致。正式 ZIP/highres member payload reads 均为 0，pixel/source-frame/model/FARO
  value/truth/training/network 均为 0；declared CRC 未冒充 payload CRC。正式终态
  `TARO_O1R_R11_FRESH_POOL_INVENTORY_AND_FRAME_PLAN_PASS`，scientific outcome 仍 `NOT_RUN`。唯一 successor 是
  `TARO_O1R_R11_FRESH_48_PARENT_SOURCE_ONLY_PHASE_A_IMPLEMENTATION_LOCK`，只实现 all-48 source/DepthART Phase A
  与 FARO=0 firewall，不授权直接执行 Phase A、selection、Phase B、设备、部署、产品或安全动作。

- 时间：2026-08-12（Asia/Hong_Kong）；执行者：Codex。冻结 TARO O1R R11 source inventory one-shot
  execution lock，状态 `AUTHORIZED_UNCONSUMED`。锁绑定已推送 implementation commit
  `9c659c3087e9b7d64bd931bf6291ce8d47ce512f`、11 份代码/协议/授权/download evidence、exact
  pool/request/download seals、central-directory-only policy、30 GiB declared-materialized ceiling、64 MiB
  evidence ceiling、2 小时 wall ceiling与用户 exact R11 authority。preflight 仅允许 lock/small sealed evidence；
  exclusive root 与 start receipt 创建后才可重验 144 source files、读取 ZIP central directory 与 trajectory。
  不预冻结真实 frame count、declared bytes 或 inventory hash；content seal 为 `244D5E16...BB10`，runner 在
  output root absent 状态完整复验通过。当前 ZIP member payload、pixel、DepthART、FARO value、truth、training
  均为 0；唯一 successor 是该 lock 推送并确认 `HEAD == origin/master` 后严格消费固定 module argv 一次。

- 时间：2026-08-12（Asia/Hong_Kong）；执行者：Codex。完成 TARO O1R R11 source inventory
  implementation lock，正式 inventory 与科学结果仍 `NOT_RUN`。独立审计指出旧 materializer 的 `testzip()`
  会解压读取包括 highres depth 在内的 member payload，不满足 sealed top-24 前 FARO payload read=0；正式实现
  因此改为独立 central-directory-only API，只验证 ZIP 路径/symlink/encryption/compression/duplicate、声明尺寸、
  声明 CRC、video/timestamp identity，并以 exact Decimal timestamp、intrinsics stem 和 source trajectory 形成
  pose-bounded frame plan。测试将 `ZipFile.testzip/open/read` 设为调用即失败，8/8 inventory focused tests、compile 与
  download-evidence record replay 通过。implementation 期间曾有一次无输出探针调用旧 `testzip()`；它未创建
  formal root、未解码/返回/解释像素、未运行模型或产生 score/selection，但 member payload read 非零且未精确
  计数，因此不作为正式 evidence，所得计数/hash 不进入 execution lock。唯一 successor 是另提交
  `TARO_O1R_R11_FRESH_48_PARENT_SOURCE_INVENTORY_ONE_SHOT_EXECUTION_LOCK`；其 root 创建后才重验 144 source
  files 并执行正式 inventory，member payload、DepthART、FARO value、truth、training、device、deployment、
  product 与 safety 继续关闭。

- 时间：2026-08-12（Asia/Hong_Kong）；执行者：Codex。正式消费 TARO O1R R11 bounded source download
  Attempt 02。exact 48 Training parents × 3 assets 的 144/144 GET 全部首试成功，network GET/recovered
  asset 为 `144/0`；source 恰好 144 files / `2,960,390,828 bytes`，`.partial=0`，三类分项与 zero-body
  HEAD 完全一致。独立复核重新验证 144 个 URL/identity/path、Content-Length/ETag/Last-Modified、SHA-256/
  CRC32，并重哈希 manifest 的 147 个文件 binding；evidence 为 148 files / 239,136 bytes，manifest content
  seal `FCE3E06D...B7A21`。正式终态 `TARO_O1R_R11_FRESH_POOL_SOURCE_DOWNLOAD_INTEGRITY_PASS`，one-shot
  已消费且不得重跑。archive member/source frame、DepthART、FARO、truth、training 均未打开，scientific
  outcome 仍 `NOT_RUN`。唯一 successor 是 `TARO_O1R_R11_FRESH_48_PARENT_SOURCE_INVENTORY_IMPLEMENTATION_LOCK`。

- 时间：2026-08-12（Asia/Hong_Kong）；执行者：Codex。冻结 TARO O1R R11 bounded source download Attempt 02
  execution lock，状态 `AUTHORIZED_UNCONSUMED`。锁绑定已推送 module-entry implementation commit
  `ad609b35444430d526d8d5531976dff2b67ab961` 与 argv
  `-m scripts.research.taro_o1r_r11_abstention_runtime.run_pool_download`；11 份 binding、144-row request plan、
  `2,960,390,828` bytes、deadline/retry/evidence budgets 与用户 authority 均不变。锁同时引用 Attempt 01
  incident 并冻结其 GET/root/consumed 为 `0/0/false`。content seal 为
  `F31AA068937AB3533BC26011ACF406A363161FC92B10771658F714E168D094F9`，runner 完整复验通过；source/
  evidence roots 仍 absent。本步 GET/source body/DepthART/FARO/truth/training 均为 0；唯一 successor 是锁
  推送并确认 `HEAD == origin/master` 后消费固定 argv 一次。
- 时间：2026-08-12（Asia/Hong_Kong）；执行者：Codex。用户在 D3R1 Phase-B r0
  `INVALID_INCOMPLETE` 收口后授权另立恢复版本；已冻结 D3R2 source-coverage scope、protocol、
  stdlib producer、独立 validator 与 14 项 synthetic tests。D3R2 保持原 exact-32 顺序和 exact-9,600
  stems，未来 census 仍为 exact-64 全新 bodies / `5,580,879,686 bytes`，但只解析 ZIP central
  directory/member names：member payload read、`testzip`/CRC、pixel decode、truth/support 与 selection
  全为 false。官方 RAW README 已 SHA 绑定其“同步 60FPS family 不保证每 timestamp 全资产存在”事实；
  推荐的 fixed-exact-300 + `SOURCE_UNAVAILABLE_UNKNOWN` 仍只是未激活候选，coverage evaluability gate
  等完整 census 后另行登记。当前正式 D3R2 root 保持 absent，没有 HEAD/GET/Range、旧 r0 body read/hash
  或 archive access；唯一 successor 为
  `EXPLICIT_D3R2_PHASE_B_EXACT64_COVERAGE_ONLY_CENSUS_ACTIVATION`。

- 时间：2026-08-12（Asia/Hong_Kong）；执行者：Codex。TARO O1R R11 source download Attempt 01 在正式
  root/GET 前的无副作用入口检查中停止：锁冻结 direct script，但绑定 Python 导入时报
  `ModuleNotFoundError: No module named 'scripts'`。未设置未冻结 `PYTHONPATH`，也未以不同 argv 绕过；
  `GET/source-body/archive-decode/source-frame-decode/DepthART/FARO/truth/training = 0`，source/evidence
  双 root 均不存在，one-shot 未消费。Attempt 01 锁保留并 supersede，不得原地修改或重跑。runner 现仅把
  formal argv 改为项目稳定 `-m scripts.research.taro_o1r_r11_abstention_runtime.run_pool_download`，新增入口
  回归；roster、HEAD evidence、144-row plan、预算和 authority 不变。唯一 successor 是先提交此实现，再另立
  `TARO_O1R_R11_FRESH_48_PARENT_BOUNDED_SOURCE_DOWNLOAD_ONE_SHOT_EXECUTION_LOCK_ATTEMPT_02`。

- 时间：2026-08-12（Asia/Hong_Kong）；执行者：Codex。冻结 TARO O1R R11 exact bounded source download
  one-shot execution lock，状态 `AUTHORIZED_UNCONSUMED`。锁绑定已推送 implementation commit
  `399b53ec9cb28efca9512f5e541aee865a7a7e1a`、11 份代码/authorization/HEAD evidence、144-row
  request-plan SHA、`2,960,390,828` source bytes、300 秒共享 asset deadline、14,400 秒全局 deadline、
  最多 432 GET 与 64 MiB evidence ceiling；source/evidence 双 root 均保持 absent。内容 seal
  `8C68922A667BEAFA457ECC1F73A8B8B109DB47FBAA7221CF811618E28AFB56B2` 已由 runner 完整复验。
  本步 GET/source body/archive decode/source-frame decode/DepthART/FARO/truth/training 均为 0；唯一 successor
  是在本锁提交推送且 `HEAD == origin/master` 后消费固定 argv 一次。

- 时间：2026-08-12（Asia/Hong_Kong）；执行者：violjjet。正式启动 DepthART-S D3R1 Phase-B
  exact-64 body 与 source-truth-support 后，在第 `2/32` 身份、checkpoint 002 前按冻结 exact-frame
  coverage gate 停止：固定 stem `42898216_694900.389` 不在 source depth inventory。只形成 1 个完整
  checkpoint；其 support 未过，但 partial `0/1` 不得升级成整体资格或科学终态。当前 attempt 保留
  4 个 source bodies 共 `841,796,127 bytes`，无 temp、scientific manifest/validation 或 selection。
  metadata-only auditor 只复验 attempt/checkpoint seals、root inventory 与 body length=HEAD，不读取/哈希
  body 或打开 archive，结果 PASS；双 modality 各 16,106 PNG、目标 stem 均缺而邻帧存在的 post-stop
  观察没有绑定 probe/log/failure receipt 或 identity-2 body SHA，只记为 unbound operator diagnostic，
  不冒充独立科学复验。正式状态为 `INVALID_INCOMPLETE`，不是科学
  PASS/FAIL 或 `D3_DATA_SUPPORT_NOT_EVALUABLE`；`scientific_terminal=null / next_gate=null`。当前 r0
  不得 resume、repair、覆盖、换帧、换身份或同版本重跑，RGB、模型、角色、训练、Development、R2、
  性能、默认 App、production 与 safety 均未授权。

- 时间：2026-08-12（Asia/Hong_Kong）；执行者：Codex。完成 TARO O1R R11 bounded source download
  implementation，尚未执行 GET。runner 只接受另行提交且 hash-bound 的 one-shot lock，绑定 consumed 144-row
  HEAD evidence、exact request plan 与 `2,960,390,828 bytes`；逐文件复核 Content-Length、ETag/Last-Modified、
  SHA-256、CRC32，并限制每 asset 最多 3 次、全局最多 432 次 GET，只有 transient transport/指定暂态 HTTP
  状态可重试，三次重试共享同一个 300 秒 asset deadline，全局 deadline 在 success seal 前复核，最终 manifest
  也计入 64 MiB evidence ceiling。独立审计发现 evidence/source 双 root 之间的竞争失败原先可能无终态；现改为 evidence root
  reservation 后先封存 start receipt，第二 root 失败也写 sealed failure 与 manifest，并增加故障注入回归。
  8/8 focused tests、compile 与 diff check 通过。本步 GET/source body/archive decode/source frame decode/
  DepthART/FARO/truth/training 均为 0；唯一 successor 是提交独立 download execution lock 后消费一次固定 argv。

- 时间：2026-08-12（Asia/Hong_Kong）；执行者：Codex。正式消费 TARO O1R R11 exact 48-parent
  zero-body HEAD one-shot。144/144 冻结 URL 全部首试返回 200 + positive Content-Length，redirect/error 为
  0，ETag/Last-Modified 为 144/144；总声明正文 `2,960,390,828 bytes`，分项为 upsampling
  `2,907,505,248`、intrinsics `48,933,412`、trajectory `3,952,168`。response/media body、GET、source
  decode、DepthART、FARO、truth scoring、training 均为 0。exclusive root 已消费且不得覆盖、替换或重跑；
  receipt 144 行 attempt→final、request identity、三个 premanifest 文件与 manifest 的 size/SHA 已独立重算
  一致。正式状态 `TARO_O1R_R11_FRESH_POOL_ASSET_HEADERS_AVAILABLE_MEDIA_UNOPENED`，scientific outcome 仍
  `NOT_RUN`。唯一 successor 是以真实字节数冻结 bounded source download implementation；不得直接发送 GET。
- 时间：2026-08-12（Asia/Hong_Kong）；执行者：violjjet。完成 Assistive Geometry R2 独立跨传感器
  factor-level Confirmation executor implementation lock，科学状态保持 `NOT_RUN`。实现严格 opaque
  archive byte/SHA/direct-child binding 与受预算 ZIP preflight、metadata-only 12 calibration + 12 score
  roster、只接收 RGB+K 的 model-only extractor、raw prediction 全量 seal/reload 后才开放 calibration source、
  conditioned factor 全量 seal/reload 后才开放 score truth 的 phase firewall，以及 source geometry、27-gate
  factor-only metrics、exclusive evidence writer 和独立重算 validator。相机/IMU 整数纳秒只转 camera seconds，
  仅 pose 查询应用 mocap scale/anchor/offset，IMU 不应用该变换；support uncertainty 对 signed
  point-to-frozen-support-plane residual 米制误差评分。独立 validator 不导入 producer/source adapter/recipe/
  metrics/reducer，并对 key/type/float64/UNKNOWN/terminal/manifest mutation fail closed。45/45 synthetic/metadata
  focused tests、ruff、compile、implementation-lock validator 与 diff check 通过。真实 ETH3D archive bytes/member、
  RGB/depth/IMU/trajectory/calibration payload、checkpoint、model inference、evidence root 和 Confirmation run
  均为 0；实现锁不构成 accuracy PASS/FAIL。唯一 successor 是另行授权、另行 hash-bound 的
  `BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_CONFIRMATION_ONE_SHOT_EXECUTION_LOCK`；本步未创建
  或授权它。执行前仍须冻结 exact camera-IMU calibration member/encoding/keys 和经官方或独立验证的 IMU
  column/axis/specific-force sign convention，任何不匹配一律 fail closed。

- 时间：2026-08-12（Asia/Hong_Kong）；执行者：Codex。记录 TARO O1R R11 exact 48-parent 数据使用授权，
  并完成 zero-body HEAD implementation。授权 receipt 精确绑定 Training 48 parents、`upsampling.zip /
  lowres_wide_intrinsics.zip / lowres_wide.traj` 三类资产、144 URL、pool/request-plan SHA 与 frozen source-first
  顺序；每阶段仍要求独立 hash-bound one-shot lock，不授予训练、设备、部署、产品、安全或再分发权限。
  新 R11 runner 复用 no-redirect HEAD transport，但使用独立 schema/root/terminal；逐 attempt 校验 retry index、
  success 后不得重试、final row 必须等于 last attempt、Content-Length 总和与 12 GiB ceiling，并在 exclusive
  root 创建时消费 one-shot。5/5 focused synthetic/mutation tests、protocol validator、compile 与 diff check
  通过；独立审计发现 authorization request/authority 字段可重封篡改后，已补 exact text 校验与对应
  mutation tests，最终 6/6 PASS。implementation commit `aebab93d` 与 7 份文件随后被 one-shot lock 精确
  绑定；本步 HEAD/GET/body/model/FARO/training 均为 0，R11 scientific status 仍为 `NOT_RUN`。唯一
  successor 是消费一次 exact 144-request zero-body HEAD。

- 时间：2026-08-12（Asia/Hong_Kong）；执行者：violjjet。将 Codex 开源维护用途从申请叙事收口为
  可执行治理：新增 maintainer automation current contract，明确 issue/PR/dependency/release/provenance/
  security review 的输入输出、模型仅 advisory、人工 write/merge/release 权限、fork prompt-injection 与
  API key 边界，并用现有 CI、Dependabot、asset identity 和 release manifest 作为确定性证据；同时新增
  Android camera、AtomS3R 明文局域网例外、模型/原生库、Actions、debug release、本地产物与 AI 维护的
  threat model，保留 cleartext、单维护者、无独立安全审计和不可 bit-for-bit 重现 YOLO export 等公开风险。
  两份 current 文档已进入 docs index、README、SECURITY 与 open-source readiness 门禁；不自动提交表单、
  不制造外部贡献，也不把模型输出提升为许可证、设备测量、security acceptance 或 safety 事实。

- 时间：2026-08-12（Asia/Hong_Kong）；执行者：violjjet。冻结并激活 DepthART-S D3R1
  Phase-B exact-64 body 与 source-truth-support gate。协议固定 exact-32 × 300 Phase-A stems、
  `5,580,879,686 bytes`、NoRedirect/无 Range、HEAD→GET Content-Length/ETag/Last-Modified
  精确绑定、全 ZIP CRC/member safety、3 位 identity checkpoint 与连续前缀 resume；任何传输、
  CRC、schema 或 decode 错误均为 execution invalid，不可伪装为 support failure。support 聚合显式
  分离九格 clear/occupied，UNKNOWN 不作 negative；只有完整处理 32 后才可按原顺序锁 first-16，
  少于 16 必须空 selection 并落 `D3_DATA_SUPPORT_NOT_EVALUABLE`。本记录仅代表协议/执行器已冻结，
  不提前声明正式 body 或科学终态；RGB、模型、角色、训练、Development、R2、性能、默认 App、
  production 与 safety 继续禁止。

- 时间：2026-08-12（Asia/Hong_Kong）；执行者：Codex。完成 Assistive Geometry R2 独立跨传感器
  factor-level Confirmation 的 F2 protocol/data/statistics lock，科学状态仍为 `NOT_RUN`。先拒绝
  `rgbd_bonn_person_tracking / rgbd_bonn_moving_obstructing_box`：二者虽未进入 current recipe，却被既有
  historical exclusion 明确标为 future formal cohort `DENY`。随后仅按官方文字 metadata 与冻结前
  repository 全历史 exact-ID 零命中，锁定 ETH3D SLAM custom global-shutter active-stereo RGB-D/IMU rig
  的 `plant_scene_2 / motion_1 / mannequin_5` 三个 capture session。六个 RGB-D/IMU archive 加一个
  camera-IMU calibration archive 共 `721,072,411 bytes`，已按 official URL、Content-Length 与 SHA-256
  绑定；下载只作 opaque byte/hash receipt，未枚举 ZIP member、未解压，也未打开 RGB、depth、IMU、
  trajectory 或 calibration payload。每 session 的 metadata-only hash rank 固定 `12 calibration + 12 score`
  identity，角色 overlap=0；一次性 source-native session geometry factor 只能从 calibration role 产生，
  raw score prediction 与 conditioned factors 必须分别在 calibration/score truth 打开前封存。协议冻结 learned
  depth/support/obstacle/boundary 的 absolute parent-macro + worst-parent accuracy、known coverage/UNKNOWN 与
  1σ/2σ calibration/rank-order gates，禁止 baseline 胜负、reducer/task state、walking_xyz/sitting_rpy 调参和
  partial-factor rescue；K/gravity/support normal/camera height 明确为 source context，不冒充 learned target。
  data identity / contract SHA-256 为 `E7552882...B345B` / `8BA036E6...D709F`；通用 R4 validator
  `VALID / 0 warning`，docs index、project structure 与 diff check 通过。当前 model inference、source outcome、
  factor result 和 Confirmation metric 均未打开；唯一 successor 为
  `BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_CONFIRMATION_EXECUTOR_IMPLEMENTATION_LOCK`，
  只允许以 synthetic/metadata fixture 实现并验证 exact source adapter、roster、firewall、factor-only scorer
  和独立 validator，不授权执行 Confirmation、训练、reducer、设备、默认 App、产品或 safety。

- 时间：2026-08-12（Asia/Hong_Kong）；执行者：violjjet。完成 DepthART-S D3R1 Phase-B
  exact-32 depth/confidence HEAD-only preflight。冻结 exact-64 request-plan SHA、NoRedirect、HEAD-only、
  三必需响应头、transient-only retry、fresh/exclusive attempt root 与同时覆盖 PASS/UNAVAILABLE/INCOMPLETE
  的离线 validator；18 项 synthetic tests 与 frozen binding replay PASS。正式结果 64/64 HTTP 200、
  Content-Length/ETag/Last-Modified 完整，声明总大小 `5,580,879,686 bytes`（depth
  `5,329,635,728`，confidence `251,243,958`），全部一次成功，redirect/recovered/unresolved error
  均为 0。原独立 validator PASS 后审计发现其对 attempt→row 矛盾字段的拒绝不完整；保持冻结 result
  与 validation 不变，新增纯离线 post-result repair auditor，独立重算 status/redirect/availability/
  recovered flags，真实 64 行四类 mismatch 均为 0，科学 PASS 保持不变且未重发 HEAD。response/media
  body 均为 0，没有 GET/Range、archive/decode、source-truth-support、first-16 selection、RGB、角色、模型、
  Development、R2、性能、默认 App、production 或 safety authority。唯一 successor 为
  `EXPLICIT_D3R1_PHASE_B_DEPTH_CONFIDENCE_BODY_AND_SOURCE_TRUTH_SUPPORT_ACTIVATION`；必须另行冻结约
  `5.58 GB` body、checkpoint/resume、全 32 support audit 与 fail-closed validator。

- 时间：2026-08-12（Asia/Hong_Kong）；执行者：Codex。完成 TARO O1R R11 non-execution
  abstention development 与 fresh dual-class confirmation protocol lock。consumed R10 的唯一 definite-clear
  false positive 只用于形成 source-only 候选：R7 2-pixel / 0.08 m / 2.0 m positive 还必须命中既有
  16-pixel、0.15 m height 或 1.5 m forward 相邻强度 cell 之一，否则 `OCCUPIED → UNKNOWN`；候选不输出
  `CLEAR`、不接受 identity/truth/result 字段并保留 R6 prior occupied。R10 只读 replay 从 1,769 TP / 1 FP
  变为 1,768 TP / 0 FP，但严格为 development-only，R10 `NOT_EVALUABLE` 不变。新协议按固定 exclusion
  commit 和 R10 全 32-parent pool 冻结 48 个 fresh Training parents / 144 URL，未来只允许 source-only
  排名 top 24 后读取 selected FARO，并新增 definite-clear physical-frame 与 parent-aware gates；18/18 focused
  tests、协议 validator 与逐文件 compile 均通过。本步 HEAD/GET/source/model/FARO/training 均为 0，R11
  scientific status 为 `NOT_RUN`、execution=false。唯一外部前门是 exact 48-parent × 3-asset 数据使用授权；
  获授权后才可另锁 zero-body HEAD one-shot，默认 App、产品与 safety authority 不变。

- 时间：2026-08-12（Asia/Hong_Kong）；执行者：Codex。完成 TARO O1R R10 exact 32-parent
  fresh clear-enriched confirmation 全链并封存正式结果。zero-body HEAD 与 source download 均为
  96/96，source 为 `1,945,902,515 bytes`；inventory 冻结 710 frames。Phase A R0 在 inference
  前因 `timm` 缺失消费为无 candidate 的 implementation stop，R1 在新 root 完整重跑 710 次
  DepthART、封存 6,390 个 source-only query features 且 FARO=0。top-eight R0 又在任何 score/selection
  产生前因 round-12 validator parity 停止，R1 在新 root 完整重算并先封存 32 scores、再封存 8 个
  parents。Phase B 只读取 selected 260/260 `highres_depth`，unselected FARO=0、训练=0、UNKNOWN
  从未作 negative。2,340 labels 为 1,786 OCCUPIED / 13 CLEAR / 541 UNKNOWN；正占用 precision
  `0.999435`、Wilson lower `0.997472`、recall `0.990482`、8-parent macro coverage gain
  `+0.984977` 全过门，但 CLEAR 只跨 3 parents（门为 4），且唯一 false occupied 使 12/13 clear
  specificity 的 Wilson lower 为 `0.717742 < 0.8`。正式终态为
  `TARO_O1R_R10_FRESH_CLEAR_ENRICHED_NOT_EVALUABLE_DUAL_CLASS_COVERAGE`，无路线、部署、设备、产品或
  safety 晋级。260 labels、2,340-query summary 与 manifest 的 263 个 pre-manifest 文件已逐文件重算
  一致。R10 不得改门、重跑或救活；唯一 successor 为 non-execution R11 abstention + fresh dual-class
  protocol lock，任何新 source/FARO/model 执行仍需单独授权。

- 时间：2026-08-12（Asia/Hong_Kong）；执行者：violjjet。登记 DepthART-S D3R1 exact-32
  Phase-B depth/confidence source scope。新 receipt 精确绑定 Phase-A governed PASS、32 个 visit/session
  的顺序与 9,600 个冻结 frame stems、D3/D3R1 protocols、manifest/offline validation 及 reviewed
  ARKitScenes LICENSE SHA；仅覆盖 `lowres_depth.zip` 与 `confidence.zip` 共 64 assets 的 future
  transport/source-integrity 与 source-truth-support admission。阈值保持 known `>=1800`、clear
  `>=270`、occupied `>=900`、每 band-horizon clear/occupied `>=30`、valid band clearances `>=450`；
  全 32 完整审计前不得形成 first-16 lock。本步只登记 scope、核验本地 hashes 与 selection digest，
  没有发送 HEAD/GET/Range、没有读取 archive/depth/confidence、truth/model、RGB、角色、Development、
  R2 或 outcome。唯一 successor 为
  `EXPLICIT_D3R1_PHASE_B_DEPTH_CONFIDENCE_HEAD_ONLY_PREFLIGHT_ACTIVATION`；仍需单独激活 64 个零正文 HEAD。

- 时间：2026-08-12（Asia/Hong_Kong）；执行者：violjjet。完成 DepthART-S D3R1 exact-127
  Phase-A intrinsics/trajectory body 与 label-blind continuity。新 D3R1 materializer 使用 NoRedirect
  GET、HEAD/GET 三头精确绑定、全池不可早停、3 位 checkpoint、连续前缀 resume 与 retained-source
  containment；预冻结 validator 从一开始同时覆盖合法 PASS/FAIL。正式下载 254 bodies 共
  `133,734,849 bytes`，全部一次成功；完整处理 127 identities，校验 `603,634` 个 `.pincam`
  payload 与 `99,155` trajectory rows。producer 得到 `53/127` eligible，按 frozen pool order 锁定
  exact first-32；离线 validator 重哈希/重解析全部 bodies 后同结论 PASS。没有 RGB/depth/confidence、
  task truth/model、TRAIN/DEVELOPMENT、R2、性能、默认 App、production 或 safety authority。因原 source
  receipt 明确不含 Phase-B assets，唯一 successor 为
  `EXPLICIT_D3R1_PHASE_B_DEPTH_CONFIDENCE_SOURCE_SCOPE_REGISTRATION_FOR_EXACT_32_PHASE_A_SELECTION`，
  本步不直接发送 Phase-B HEAD/GET。21 项 focused tests、frozen binding/selected-roster replay、JSON、
  diff、docs-index、repository hygiene 与最新 `origin/master` integration structure checks 全部 PASS；
  structure gate 覆盖 81 个 research Modules，且本步未改写并行 TARO 工作。

- 时间：2026-08-12（Asia/Hong_Kong）；执行者：violjjet。完成 DepthART-S D3R1 exact-127
  Phase-A intrinsics/trajectory HEAD-only preflight。新执行器禁止 redirect（避免 urllib 将 HEAD
  自动改为 GET），冻结 max-attempts=3 与 transient-only retry，并在任何网络前独占新 attempt root；
  response `read/readinto/peek` 由定向测试强制不可调用。正式执行 254/254 assets 均为 HTTP 200、
  正整数 Content-Length、非空 ETag/Last-Modified，声明总大小 `133,734,849 bytes`；maximum attempts
  为 1，redirect/recovered/unresolved error 均为 0。离线 validator 逐行复验 request order、headers、
  attempt history 与 authority 后 PASS。response/media body 为 0，没有 GET/Range、archive/pose、
  selection、角色、Phase-B、truth/model、R2、性能、默认 App、production 或 safety 权限。唯一 successor
  为 `EXPLICIT_D3R1_PHASE_A_INTRINSICS_TRAJECTORY_BODY_AND_LABEL_BLIND_CONTINUITY_ACTIVATION`，必须
  另行冻结并激活 body protocol。13 项 D3R1 focused tests、binding replay、JSON、diff 与 docs-index
  checks PASS；全局 structure gate 仅被既有 TARO 的 7 个未登记/contract 不完整 runtime module、
  module count 及 TARO current-status/successor 漂移阻塞，未报告 D3R1/DepthART 缺陷，本步未改写该并行工作。

- 时间：2026-08-12（Asia/Hong_Kong）；执行者：violjjet。登记 DepthART-S D3R1 exact-127
  Phase-A source scope。receipt 绑定 frozen recovery protocol、127-parent/session roster 与 reviewed
  ARKitScenes LICENSE SHA，只覆盖 `lowres_wide_intrinsics.zip` 和 `lowres_wide.traj` 的 future
  source-integrity、schema 与 label-blind continuity use；不包含 depth/confidence、RGB、身份扩展或
  redistribution。本步只生成 scope receipt 并复核本地 hashes，没有发 254 个 HEAD、没有 GET、
  archive/pose content、truth/model、selection、TRAIN/DEVELOPMENT、R2、性能、默认 App、production
  或 safety 权限。唯一 successor 为
  `EXPLICIT_D3R1_PHASE_A_INTRINSICS_TRAJECTORY_HEAD_ONLY_PREFLIGHT_ACTIVATION`；它仍需单独激活。

- 时间：2026-08-12（Asia/Hong_Kong）；执行者：violjjet。另立 DepthART-S D3R1 pre-outcome
  Phase-A recovery version，并完成 metadata-only fresh roster lock。旧 D3 exact-48 及其中 21 个
  continuity-qualified identity 全部禁止 carry-over，`300 frames / adjacent <=0.5s / pose bracket
  <=0.25s / portrait {1,3}` 门保持不变。旧 `21/48` 只用于 one-sided 95% Clopper-Pearson
  resource-sizing heuristic：127 是 `P(X>=32)>=0.95` 的最小 planning pool；该假设不构成质量证据
  或成功保证。planner 绑定 published pre-recovery commit `8d17a053...` 的 immutable
  `docs/research` tree（490 official tokens），并以 AST literal、文件 SHA 与 ordered tuple digest
  额外排除 TARO R10 scripts-only 64 tokens；effective firewall 554。最终锁定 127 个 unique
  Training visit/session，与 workspace、TARO R10、D1/D2/R2 及旧 D3 overlap 全为 0；eligible
  capacity 为 3,724 rows / 1,233 unique visits。write-once generation、independent replay 以及
  roster 写入 docs 后 replay 均 PASS，后两者 byte-identical。没有发 HEAD/GET，没有 source-scope、
  body、truth/model、selection、TRAIN/DEVELOPMENT、R2、性能、默认 App、production 或 safety 权限。
  唯一 successor 为 `EXPLICIT_D3R1_SOURCE_SCOPE_REGISTRATION_FOR_EXACT_127_METADATA_ROSTER`。

- 时间：2026-08-12（Asia/Hong_Kong）；执行者：violjjet。完成 DepthART-S D3 Phase-A exact-48
  intrinsics/trajectory body 与 label-blind continuity execution。96/96 GET 共 `41,979,912 bytes`，
  Content-Length/ETag/Last-Modified 全部匹配冻结 HEAD；48/48 checkpoint、`190,028` 个 `.pincam`
  payload 和 `31,185` 个 trajectory row 经保留源离线复算通过。固定 300-frame、相邻
  `0 < gap <= 0.5s`、pose bracket `<=0.25s`、portrait index `{1,3}` 门仅有 `21/48` identity
  合格，少于所需 32，故正式 selection lock、TRAIN/DEVELOPMENT role 与 Phase-B authority 全为空，
  终态 `D3_PHASE_A_FAIL_FEWER_THAN_32_ELIGIBLE_IDENTITIES`。预冻结 validator 只接受 PASS manifest，
  合法 FAIL 后才暴露覆盖缺口；保留原协议/validator/manifest/hash，新增 post-terminal read-only repair
  auditor 对全部 96 bodies 和 48 continuity decisions 独立复核，结果
  `VALID_WITH_POST_TERMINAL_VALIDATOR_COVERAGE_REPAIR`，未重跑下载、未改 pool/门限/选择。
  RGB/depth/confidence/truth/model、训练、R2、性能、默认 App、production 与 safety 均未打开；当前
  D3 version 无 successor，恢复必须另立 pre-outcome fresh roster/source-scope/protocol 版本。16 项
  D3 focused tests、binding preflight 与 docs index PASS；全局 structure gate 仅被当前分支并行 TARO
  的 7 个未登记 Module 及 TARO current-status/successor 漂移阻塞，未报告任何 D3/DepthART 缺陷，
  本次未改写该并行工作。

- 时间：2026-08-12（Asia/Hong_Kong）；执行者：violjjet。完成 Assistive Geometry R2 的
  SuperTeacher 数据支线与 AG research-pipeline 落地。先保留无锚 `walking_xyz` 的冻结负结果：
  仅 1/12 帧结构有效、108/108 cells UNKNOWN；naive post-walking global ERM 又因旧 selection
  退化而不晋级。随后只用 factor depth、六个预声明物理候选和 camera-height geometry 选择
  `weighted q75` session scale anchor，已消费 walking_xyz 的 mean log-RMSE 从 `0.4126` 降到
  `0.2823`（31.57%），未读取 task/reducer outcome。该算法原样冻结到 current-recipe-checkpoint-unseen
  `sitting_rpy`：12 份 source-native metric depth + geometry-anchored support/boundary labels 全部通过，
  factor-only 推理 `targets_loaded=false`，12/12 adapter V2 frames 与 reducer replay 有效且确定，108 cells
  得到 `CLEAR=18 / UNKNOWN=90`，11/11 gates PASS；48/48 代码、模型、校准和逐帧 artifact 哈希一致。
  最终 result SHA-256 为 `A0D15EC9278E2BB766A5ECCD596F6EDE2793CBA8E1B7BFECB1D0853D11817886`。
  该结果只证明单个 TUM parent 的 SuperTeacher→learned factors→deterministic AG mechanics，不建立全局
  freshness、跨传感器泛化、导航效用、HTP/Android、默认 App、产品或 safety authority；Attempt17 等
  历史终态不改写。下一步只允许先冻结独立跨传感器 factor-accuracy confirmation，不在已消费
  walking_xyz/sitting_rpy 上调参，也不要求任意 baseline 胜负。
- 时间：2026-08-12（Asia/Hong_Kong）；执行者：violjjet。完成开源申请就绪与默认分支 CI 修复：根 README 收敛为中英双语公共入口，复用真实 v10.9.0 App 截图并明确非安全认证边界；新增贡献、安全、行为准则和 Issue/PR 表单。CI 根因分别修复为 QAIRT Windows 默认路径不再经 Gradle URL 解析、两个既有 TARO Module 补齐 76/76 合同/索引/唯一分类、TARO current 与 R4 导航恢复一致、local-only calibration PDF 不再伪装为公开链接。自有 `libblindassist_vision.so` 按 Android 官方 r27-or-lower CMake linker 合同设置 16KB max/common page size，四 ABI 与最终 APK 的 41 个 native entries 通过仓库静态门禁；这不是实际 16KB 设备运行证明。验证包括 14/14 research contract files、文档 674 链接、完整无设备 Gradle 483 tasks、模型 shape 断言、APK package/version/debug signature 与 16KB 对齐。默认模型、权限和运行逻辑未改变。
- 时间：2026-08-10（Asia/Hong_Kong）；执行者：violjjet。将 Windows/Codex 本地 Android Gradle 调用收口到 `scripts/run_android_gradle.ps1`：入口从自身路径锁定仓库根目录，从 version catalog、wrapper 与 `local.properties` 读取并校验 JDK/compile SDK/Gradle 合同，统一 machine-local Gradle/Android state，connected test 自动执行 10 秒 ADB 预检，多设备要求显式 serial；环境失败固定返回 `ENV_BLOCKED`，通过后才启动 wrapper。新增 `.editorconfig`，仓库 hygiene 直接执行 Git whitespace check 并覆盖“末尾多一空行”回归；AGENTS 同时固定单层 PowerShell、结构化路径和短锚点补丁规则，避免嵌套 shell、相对脚本路径及转义后的反斜杠继续污染正式验证。
- 时间：2026-08-10（Asia/Hong_Kong）；执行者：Codex。正式消费 TARO O0R Content-Length HEAD Attempt 02，冻结 72 个 HEAD target 中 `71/72` 返回 200 + positive Content-Length，可用总长度 `1,105,086,109 bytes`；唯一失败是 ADAPTER_FIT video `47333152` 的 `lowres_wide.traj`，3/3 attempts 均为 HTTP 403、无 Content-Length。全程 response body=0、redirect=0，HEAD receipt/start/result/manifest 的 bytes/SHA 与 72-row plan identity 复核为 `VALID_NEGATIVE_TERMINAL`；HEAD root 已消费，source/work/truth/factor 四 root 仍不存在，GET/source body/truth/uncertainty/DepthART/factorial 均为 0，truth one-shot 未消费。按预冻结 no-replacement/no-rerun 规则，终态 `TARO_O0R_ASSET_HEADERS_NOT_AVAILABLE_NO_REPLACEMENT / O0R_NOT_EVALUABLE_SOURCE_ASSET_UNAVAILABLE / NO_ACTIVE_EXECUTION`，当前 successor 为无；不得替换 `47333152` 或继续 source/truth。
- 时间：2026-08-10（Asia/Hong_Kong）；执行者：Codex。冻结 TARO O0R Content-Length HEAD Attempt 02 execution lock，状态 `AUTHORIZED_UNCONSUMED`。该锁保留 exact 24 parent × 3 asset、72-URL request-plan SHA、zero-body、8 workers、20 秒 × 3 attempts、20 GiB compressed ceiling 与同一 exclusive root；新增绑定 junction-aware implementation commit `2c0fdef8`、刷新后的 implementation lock/hash 和 Attempt 01 incident。Attempt 01 的 request/root/consumption 均为 0/false，旧锁不得重跑；Attempt 02 root 创建后才消费。该锁提交前仍未发送 HEAD，GET/source/truth/DepthART/factorial 均为 0/禁止；提交后唯一 successor 为执行一次冻结 Attempt 02 argv。
- 时间：2026-08-10（Asia/Hong_Kong）；执行者：Codex。TARO O0R Content-Length HEAD Attempt 01 在任何网络请求和 root 创建前以 `PATH_ESCAPE` 停止：旧 `safe_join` 将仓库规范明确授权的 `E:/linnan/linnan/artifacts.local` junction 解析到 `F:/ba-data/blindassist-artifacts-20260805` 后误判为越界。逐项复核 `HEAD/GET/body/source/truth/model-output=0/0/0/0/0/0`，五个正式 root 均不存在，HEAD/truth one-shot 未消费；Attempt 01 锁保留且不得原地重跑。最小修正仅允许 `artifacts.local` 首段进入受信任 junction containment，同时验证所有更深 parent 仍在同一 junction target 内，并保留仓库 lexical path 供 receipt 使用；新增专项回归后 materializer 为 25/25、implementation validator 6/6。当前仍 `HEAD_NOT_RUN / SOURCE_UNOPENED / SCIENTIFIC_NOT_RUN`，唯一 successor 为另冻并提交 Attempt 02 HEAD-only execution lock。
- 时间：2026-08-10（Asia/Hong_Kong）；执行者：Codex。基于已推送实现提交 `9c525103` 和用户的 exact 24-parent × 3-asset data-use authorization，冻结 `TARO_O0R_ARKITSCENES_CONTENT_LENGTH_HEAD_EXECUTION_LOCK / AUTHORIZED_UNCONSUMED`。锁精确绑定 72-URL request-plan SHA、5 份实现/协议文件、固定 argv、Python 环境、8 workers、20 秒 × 最多 3 attempts、20 GiB compressed 上限、zero response body、禁止 redirect、exclusive HEAD root 与四个必须保持 absent 的 source/work/truth/factor root。HEAD root 创建即消费 evidence version；transport/write/timeout 失败也必须写为 `ONE_SHOT_CONSUMED`，不得覆盖、重跑或换 parent。该锁提交前仍未发送 HEAD；GET、source/truth、uncertainty fit、DepthART、factorial、训练、设备、产品与 safety 均未授权。提交后唯一 successor 是执行冻结的 HEAD-only argv。
- 时间：2026-08-10（Asia/Hong_Kong）；执行者：violjjet。依据用户对锁定 24 个 ARKitScenes Training video 及每个 `upsampling.zip / lowres_wide_intrinsics.zip / lowres_wide.traj` 的 HEAD 与 source/truth-only WILD_LAB 使用授权，完成 `TARO_O0R_ARKITSCENES_TRUTH_MATERIALIZER_IMPLEMENTATION_LOCK_PASS`。授权 receipt 逐字绑定 72-URL request plan，但不自行激活 execution；outcome-blind amendment 同时冻结 all-exact frame denominator、8-parent 全帧 uncertainty fit-before-eval、每 query 独立 FARO-at-Apple-centers/confidence corridor lookup、official `{video_id}_{timestamp}` original-member envelope 与 content-addressed ndarray hydrate gate。新增 HEAD-only 与 source/truth-only fail-closed runner、bounded GET/ZIP/hash/CRC、atomic root-consumes-one-shot writer及 uncertainty/factor array 持久化；终检补齐 HEAD transport/write failure、truth root-creation 窄窗口与 `artifacts.local` junction containment，exclusive root 创建后统一记为 one-shot consumed 并尽力写 failure/manifest。25/25 focused tests、6/6 implementation-lock mutation tests与专项 validator `VALID`。全程 HEAD/GET/source body/root/truth/真实 uncertainty/DepthART/factorial 均为 0，五个 future roots保持不存在；当前为 `HEAD_NOT_RUN / SOURCE_UNOPENED / ONE_SHOT_UNCONSUMED / SCIENTIFIC_NOT_RUN / EXECUTION_NOT_AUTHORIZED`。唯一 successor 是另冻并提交 `TARO_O0R_ARKITSCENES_CONTENT_LENGTH_HEAD_EXECUTION_LOCK`（execution=false）；在该锁存在前不得调用 HEAD runner，更不得下载或物化 truth，默认 App、产品与 safety authority 不变。
- 时间：2026-08-10（Asia/Hong_Kong）；执行者：violjjet。完成 TARO `TARO_O0R_ARKITSCENES_TRUTH_ONLY_ONE_SHOT_PREFLIGHT_LOCK_PASS`，状态严格为 `PREFLIGHT_LOCKED / HEAD_NOT_RUN / ONE_SHOT_UNCONSUMED / EXECUTION_NOT_AUTHORIZED`。从 hash-bound O0R contract 重算 8 ADAPTER_FIT + 16 O0R_EVAL_CANDIDATE parent，以三个 frozen URL template 展开 72 个唯一 Training HEAD target，并冻结 canonical request-plan SHA、离线 validator argv、Python 3.11.9/NumPy 2.1.3/SciPy 1.17.1/Pillow 12.2.0、20 GiB compressed/50 GiB materialized/12 h/12 GiB/2 GiB budget、failure scope 与四个 absent roots。专项 validator VALID，8/8 static/mutation tests PASS；未发 HEAD/GET/Range，未读取 Content-Length/source body，未创建或消费 one-shot root，未物化 truth、拟合 selected-source uncertainty、运行 DepthART/O0R。锁同时 fail-close 记录：旧 B0 receipt 只覆盖 6 个旧视频且缺 trajectory，不能授权 TARO 24 × 3 body access；ADAPTER_FIT `47333152` 在官方 downloader missing-3dod list 中，未来 trajectory HEAD 非 `200 + Content-Length` 即 R0 NOT_EVALUABLE、不得换 parent。唯一 successor 为 `TARO_O0R_ARKITSCENES_TRUTH_ONLY_MATERIALIZER_IMPLEMENTATION_LOCK`（execution=false），且必须先有新的 TARO-specific signed 24-parent asset receipt；默认 App、产品与 safety authority 不变。
- 时间：2026-08-10（Asia/Hong_Kong）；执行者：violjjet。按用户要求不等待完整真值，完成 Assistive Geometry AG-ST R0 SuperTeacher factor-label factory。基于既有 16 TRAIN parent × 3 帧 MapAnything Stage 0A，source-first 融合 confidence、observed-anchor residual 与 multi-view reprojection residual，物化 48 份、105,090,692 bytes 的 A/B/C/UNKNOWN 分级 pseudo-label NPZ；metric depth / dense-normal diagnostic / conservative support / obstacle-boundary evidence 有效覆盖为 `96.45% / 94.66% / 25.05% / 71.60%`，support plane 为 `36/48` 帧。约 50% coverage 下，仅 confidence 的 MAE/`>0.10 m` 为 `0.03021 m / 5.12%`，combined gate 为 `0.01607 m / 0.85%`，相对下降 `46.8% / 83.4%` 且 16/16 parent 可评，证明 multi-view residual 是关键独立筛选信号。8/8 focused tests 与 48/48 NPZ invariant validation PASS；连续 80° 解析斜面不再被 raw depth gradient 误判，真实 depth step 仍检出，派生 factor 继承局部最弱 provenance。结论为 `GRADED_PSEUDO_LABEL_FACTORY_MATERIALIZED_MULTIVIEW_GATE_SUPPORTED`，下一步可直接进行 per-factor-validity/tier-weighted masked student WILD_LAB 训练；这些输出不冒充完整 truth、uncertainty truth、跨源泛化、F1、产品或 safety evidence。
- 时间：2026-08-10（Asia/Hong_Kong）；执行者：Codex。按用户要求跳过继续扩写规则，直接完成 Assistive Geometry AG-ST R0 的真实 Stage 0A `WILD_LAB`。新增 factor-only B0 raw adapter/MapAnything runner 与 5 项 focused tests；只读 RGB/K/pose/partial source depth/confidence，未读取 clearance/occupancy。最终覆盖全部 16 TRAIN parent × 3 帧与 `1,009,190` 个确定性隐藏参考像素：64 px 大孔洞的 source-only nearest baseline 为 `MAE 0.04933 m / >0.10 m 9.41%`，逐 view observed-anchor 校准后的 MapAnything 为 `0.03351 m / 6.78%`，完整覆盖分别改善 `32.1% / 28.0%`；50.1% confidence coverage 为 `0.03021 m / 5.12%`。未校准 Teacher 为 `0.05699 m / 15.93%`，证明 source metric anchor 必需；confidence 的 parent-macro risk 随筛选改善，但 global MAE 非严格单调且 parent coverage 异质，故终态为 `DEPTH_LABELABILITY_SIGNAL_SUPPORTED_CONFIDENCE_GATE_NOT_TRANSFER_READY`。全部 TRAIN parent 已为此问题消费，不构成 unseen canary；未物化 canonical pseudo-label，support/boundary/sigma、F1、跨源泛化、部署、产品与 safety authority 均未建立。
- 时间：2026-08-10（Asia/Hong_Kong）；执行者：violjjet。把 TARO 从 `O0R_NOT_EVALUABLE_DATA_AND_INTERFACE / PAUSED_NO_ACTIVE_EXECUTION` 推进到 ARKitScenes source-and-adapter **协议层**前门锁。基于 pinned Git exclusion snapshot 与官方 upsampling split metadata，outcome-blind 冻结 `8 ADAPTER_FIT + 16 O0R_EVAL_CANDIDATE` 个 fresh、visit-disjoint TRAIN parent；同时冻结 FARO-only factor/query truth、exact timestamp/K/pose receipt、fit-only real-residual uncertainty、9 query/frame、S/P/B 八臂 deterministic injection、parent-level 统计/非劣门、资源预算与 failure scope。implementation-lock 前的独立只读审查进一步收口 model-free SCALE truth、右侧 pose bracket 水位、9-query receipt 绑定、source-specific receipt 不冒充完整 P0 receipt、TARO 专用 query reducer、registration/boundary 与 uncertainty fallback；静态 validator 和 21 项 mutation tests、roster/exclusion/hash/权限/future-root absence 重算均 PASS，result 为 `VALID`。24 个 source body 均未下载或打开，truth materialization、DepthART inference、O0R scientific execution、G0/G1/A0/A1/J0、默认 App、产品与 safety authority 均未发生；当前终态为 `O0R_SOURCE_AND_ADAPTER_CONTRACT_LOCK_PASS / O0R_TRUTH_PREFLIGHT_NOT_RUN / O0R_SCIENTIFIC_NOT_RUN / EXECUTION_NOT_AUTHORIZED`，唯一 successor 为 `TARO_O0R_ARKITSCENES_SOURCE_ADAPTER_IMPLEMENTATION_LOCK`。
- 时间：2026-08-10（Asia/Hong_Kong）；执行者：Codex。完成项目结构与治理收口：根 `README.md` 只保留产品状态并委托研究总入口，`ALGORITHM_RESEARCH_CURRENT.md` 的状态/successor 与 current 路线 README 由结构门机器对齐，`SYSTEM_RESEARCH_CURRENT.md` 对 DepthART/HFTF 只做分类委托；研究 Module 数量收敛到 `MODULE_INDEX.md` 的 Git-visible `N-of-N` 真源。文档门扩展到非归档 current、路线 README 和 protocol 本地链接，CI 改为同时运行 live hygiene+structure、live docs 与三组负向回归。根日志的 176 个 2026-07 日期块按原文/原顺序迁入月度归档，根日志从 5,023 行降至 3,846 行；3 个根目录 native 编译产物按字节数/SHA-256 迁入 `artifacts.local/work/governance-root-native-artifacts-20260810/`，module-local `.cxx/` 保留现场但明确忽略并禁止提交。
- 时间：2026-08-10（Asia/Hong_Kong）；执行者：violjjet。按用户确认将 TARO 的论文定位从防御性“非模块拼装”优化为可证伪的**组合式创新**：允许显式复用/替换 factor encoder、residual posterior、observable-subspace solver、选择性校准、foundation geometry teacher 与 action scorer，但以共同 body/path clearance functional、条件式 GaugeFix→PARA 调用、预冻结非对称 false-clear risk + 人机成本、deterministic reducer authority 和 plug-compatible/equal-budget factorial attribution 作为联合 claim 的成立条件。路线指南新增 task-relevant information / sensory-value / conformal stopping / foundation geometry 前作去重、future-only conformal/CRC 边界、observation-withholding 与 sensing regret、geometry-anchored minimal pairs、组件替换消融，以及 Aria Digital Twin/ScanNet++/TwinScene 的公开文档级候选能力映射；未申请或打开数据 payload，未创建 runner/model/materializer/checkpoint，未读取 outcome、执行 O0M/O0R 或改变任何算法状态。TARO 仍为 `P0_PASS / O0M_SYNTHETIC_ANALYTIC_MECHANICS_PASS / O0M_ONE_SHOT_CONSUMED / O0R_NOT_EVALUABLE_DATA_AND_INTERFACE / PAUSED_NO_ACTIVE_EXECUTION`，无 successor，默认 App、产品与 safety authority 不变。
- 时间：2026-08-10（Asia/Hong_Kong）；执行者：Codex。完成 Assistive Geometry R2 F1 零参数 `FactorTensorAdapter` implementation lock 与首次 synthetic canary。实现、runner、10-test focused suite 和 F0 reducer 在 commit `c8ec23fe` 先行入库并推送后，才消费 exclusive evidence root；8/8 frozen cases、`A01..A10=10/10`、8/8 双独立进程 replay、`14/14` prediction fields、`28/28` F0 fields 与 7/7 sigma-only non-strengthening mutations 全部 PASS。receipt mismatch/support invalid 均全 UNKNOWN，局部缺深度不产生 OCCUPIED，portrait/landscape state map 等价，8-connected split/bridge 与 canonical order 正确，final-task shortcut 被拒绝，F0 reducer SHA 保持 `2D6C26AD...2092`。CPU-only 用时 `1.583935 s`，evidence `17,158 bytes`；network/GPU/device/real data/task outcome/model/training 均为 0。终态 `R2_F1_FACTORTENSOR_ADAPTER_SYNTHETIC_CANARY_PASS` 只关闭 tensor-to-frame synthetic mechanics blocker，不建立 real learnability、headroom 或 task utility。supervision frontdoor 仍未满足，F1 execution=false；唯一 successor 回到 `BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_SUPERVISION_SOURCE_AND_LABEL_CONTRACT_LOCK`，当前仅允许另锁 pre-outcome contract，不允许物化标签或训练。
- 时间：2026-08-10（Asia/Hong_Kong）；执行者：violjjet。完成 Assistive Geometry R2 F1 `FactorTensorAdapter` CANARY_LITE non-execution protocol lock。基于 byte-frozen F1 14-field factor schema 与 F0 reducer，冻结 `14/14` prediction-field consumers、全部 reducer-field producers、17 个确定性 outside-graph operation、receipt/K/gravity/frame binding、scale/support/boundary uncertainty、8-connected dense→obstacle semantics、canonical ordering、8 个 tiny synthetic cases 与 `A01..A10`。缺 receipt/support/local depth/uncertainty 一律 fail-closed，UNKNOWN 不得被当 negative 或随 uncertainty 增大变得更 clear/occupied；adapter 保持零参数、无 task outcome、无 final-state shortcut。通用 R4 与专项 validator 均 `VALID / 0 error`，13/13 mutation tests PASS。终态 `R2_F1_FACTORTENSOR_ADAPTER_PROTOCOL_LOCK_PASS / CANARY_NOT_RUN`；唯一 successor 只允许实现冻结 adapter 并运行 synthetic canary。真实数据、标签物化、模型、训练、F1/F2、设备、产品与 safety 仍未授权，continuous boundary/complete factor-schema supervision 前门仍是独立 blocker。
- 时间：2026-08-10（Asia/Hong_Kong）；执行者：violjjet。完成 TARO minimal-first 路线的唯一 O0M one-shot 并签署收口。冻结的独立 NumPy runtime 在预去重、预白化 synthetic analytic family 上得到 `TARO_O0M_SYNTHETIC_ANALYTIC_MECHANICS_PASS`：10 个 identifiability、80 个 factorial 与 2 个 action-filter records 合计 `92/92` 匹配，`O0M_G01..G10=10/10 PASS`，两个独立 worker replay 字节一致（SHA `66BA201D...C41D`）。执行耗时 `0.368618 s`、peak RSS `35,930,112 bytes`、evidence `126,603 bytes`，均低于 30 s / 256 MiB / 1 MiB 锁定预算；network/GPU/device/real-data 均未使用。exclusive root 已原子创建且 one-shot 已消费，result/records/receipt/manifest 均由签署结果逐字节绑定，不得覆盖、删除或重跑。Claim ceiling 仅为 synthetic analytic mechanics，不建立真实 factor causal headroom、真实 dedup/whitening、模型、被动/主动视角、设备、产品或 safety 结论。真实 O0R 因 complete factor/query truth、truth-clear bundle、continuous boundary/uncertainty truth、target timestamp/pose、deterministic injection adapter 与 fresh paired outcome 缺失，终态保持 `TARO_O0R_NOT_EVALUABLE_DATA_AND_INTERFACE`（非负证据）；`G0/G1/A0/A1/J0` 均未授权，路线转为 `PAUSED_NO_ACTIVE_EXECUTION` 且无 successor，只有新的 pre-outcome source-and-adapter contract 满足全部前门后才可另立版本。
- 时间：2026-08-10（Asia/Hong_Kong）；执行者：violjjet。执行 AG-DUE R1 SANPO-Synthetic exact metadata/object-inventory preflight 并 fail-close 收口。执行锁固定 `gresearch` host/bucket、official-TRAIN session `17c7d6bc...179cb`、`camera_chest/left`、4 个 metadata objects、3 个 frame prefixes、请求/字节/分页预算与 frame-body byte=0；20/20 runner mutation 与 8/8 governed-result mutation PASS。description 与 global labelmap body 只在内存读取且未持久化 raw bytes；冻结的 `.../left/frame_segmentation_annotation_type.json` 在两次 attempt 均 HEAD=404，协议禁止猜替代路径、换 session/camera/lens 或扩大 LIST，因此 pose table 未读、frame-prefix LIST 未开始、numeric-index intersection/lowest-25 未产生，RGB/mask/depth body request/read/bytes 全为 0。两次 attempt 共 12 个实际 request；tracked result 显式更正 retry receipt 的 prior-count=2 错记并保留原 receipt。终态 `NOT_EVALUABLE / STOP_SOURCE_OBJECT_INVENTORY_OR_SCHEMA_INCOMPLETE`，`source_data_support=false`、DCA/F1/body-canary/pose/timestamp/support/boundary/training 权限全 false；当前唯一 successor 为 `NONE_STOP_AT_PREFLIGHT_TERMINAL`，新 source/session/path 只能从另行版本化 R0 manifest 与 source-specific protocol 重新准入。
- 时间：2026-08-10（Asia/Hong_Kong）；执行者：violjjet。完成 TARO O0M non-execution protocol lock，终态 `TARO_O0M_PROTOCOL_LOCK_PASS / SCIENTIFIC_STATUS_NOT_RUN`。冻结与 P0 execution ID 分离的 10 个 identifiability cases、5 个 factorial scenes × 8 arms × 2 modes = 80 条逐臂 payload/output/common-support/intervention-guard records，以及 2 个 body-motion action filters；`R_weak` 与 `H_meas` 分轴、VALUE_ONLY common-support 与 FULL_BLOCK validity/uncertainty diagnostic 分离。Factorial solver 输入改为独立 `observed_base_mean_m` 加 patch delta，truth 只供 verifier 对照；multiplier `1.0` 的 halfwidth 明确是 deterministic budget，而非 Gaussian `1σ` 或 95% coverage。完整 protocol semantic core、fixture canonical digest、exact binding roles、静态 Module allowlist 与 absent exclusive artifact root 均 fail-closed。未来 one-shot 预算固定 CPU-only/30 s/256 MiB/1 MiB、seed 1729 但 rng 未使用，无 network/GPU/device/real data。通用/专项 validator VALID，33/33 mutation tests PASS；当前没有 implementation、runner 或 scientific artifact，execution authority=false。唯一 successor 为 `TARO_O0M_IMPLEMENTATION_LOCK`，只允许在独立 runtime Module 创建并静态测试纯解析 mechanics；真实 O0R 继续 `TARO_O0R_NOT_EVALUABLE_DATA_AND_INTERFACE`，默认 App、产品与 safety authority 不变。
- 时间：2026-08-10（Asia/Hong_Kong）；执行者：violjjet。完成 DepthART task-preserving D1 的 fresh-device pre-outcome gate。用 Android 设备健康检查与 fresh ADB identity 锁定 `R5CX10M8Y8X / SM-S9280 / e3q / SM8650 / Android 16 API36 / 2026-07-05 / DZG1 fingerprint`，不继承旧 DZF2 附件；扩展 deterministic full-graph canary 到 `1×3×608×448`，冻结 DLC、QAIRT 2.47 runtime、AArch64/v75 combined SelectiveScan+LayerNorm package 和全部 input SHA。Attempt 1 因 HTP package path 误用 `arm64/` 相对路径 exit 13 并保留；未改 candidate/runtime/input/tolerance，Attempt 2 用同名 LD/ADSP 双解析后 CPU/HTP registration PASS、graph finalize `0x0`，生成 22,552,576-byte context（SHA `FA5DC9DC...FC3D`）。direct 与 saved-context 各执行一次，finite float32 `[1,608,448]` 输出 SHA 均为 `0A464746...8EF9` 且 bit-exact。PyTorch↔HTP raw-depth diagnostic 仍 FAIL（max/mean/RMSE `1.42328/1.06983/1.08726m`），因此 strict G4-D 负终态不变；D1 只允许后续 frozen task-quality screen 判定。未读 D1 Development task truth/model outcome 或 R2 cohort，未测性能，DA2/default App/product/safety 不变。唯一 successor 为先冻结 runner/checkpoint-resume/8×300 exact activation receipt，再显式启动 D1 Development task-quality screen。
- 时间：2026-08-10（Asia/Hong_Kong）；执行者：violjjet。推进 DepthART task-preserving D1 到设备硬门。首先执行冻结的 16-identity ARKitScenes label-blind body preflight：4 primary 与 4 frozen-order reserve replacement 满足每 session 300 个连续 pose-derived portrait RGB/depth/confidence/K 帧，最终 8-session Development roster 锁定；只为 integrity 解码媒体，未视觉选样、未读 task/model outcome 或 R2 cohort。随后从 canonical checkpoint 重建 full-FOV `1×3×608×448` fixed-mixed 单候选：PyTorch shape/finite PASS，camera externalization bit-exact，ONNX checker PASS，QAIRT `2.47.0.260601114230` host converter 成功写出 exact-shape 850-op DLC；candidate ONNX/DLC、reference checkpoint、task postprocess 与 roster SHA 已一并冻结。执行时 ADB 设备数为 0，因此 SM8650/v75 context、HTP full-graph parity、partition、性能和 Development outcome 均未运行；host conversion 不作设备执行或质量证据。strict G4-D 负终态、DA2 baseline、R2 sealed cohort 与默认 App 不变。唯一 successor 为 `DEPTHART_TASK_PRESERVING_D1_SM8650_HTP_CONTEXT_AND_OUTCOME_ACTIVATION_PREFLIGHT`：设备恢复后只对 exact DLC 建 context、关闭 synthetic parity 并做无 outcome activation validation，通过后才能另行激活 D1 quality screen。
- 时间：2026-08-10（Asia/Hong_Kong）；执行者：violjjet。完成 AG-DUE R1 SANPO-Synthetic source-specific integrity/capability audit 的非执行协议锁。协议只绑定 exact official-TRAIN session `17c7d6bc...179cb`、`camera_chest/left`、四个 metadata object 和 RGB/panoptic/depth exact prefix；当前 network、source listing、metadata bytes、local payload、RGB/depth/mask body、Teacher、derivation、materialization、训练与 Development/Confirmation authority 全为 false。下一步仅允许 metadata/object inventory preflight，并须在读取 frame body 前冻结最低 25 个完整 aligned numeric index；不足 25 个即 `NOT_EVALUABLE`，不得换 session/camera/lens 或按内容挑帧。协议明确 depth 不是 support truth、panoptic presence 不是 continuous boundary truth、pose row/row count 不是 frame binding、`index/fps` 不是 explicit timestamp，quaternion/handedness/transform direction 与 unknown/void policy 均须另行验证；单 session 只算 1 parent，永远不能满足 F1 的 12-joint-parent 门。13/13 mutation tests、R1 validator 与全 AG-DUE 42/42 tests PASS；未联网或打开任何 source payload。唯一 successor 为 `BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R1_SANPO_SYNTHETIC_METADATA_AND_OBJECT_INVENTORY_PREFLIGHT_EXECUTION`，仅允许 exact object HEAD/LIST 和 description/labelmap/annotation-type/pose-table metadata，frame body、source support、DCA/F1 PASS、默认 App、产品与 safety 继续未授权。
- 时间：2026-08-10（Asia/Hong_Kong）；执行者：violjjet。完成 TARO 唯一 successor `P0` 非执行协议锁，终态 `TARO_P0_PROTOCOL_AND_SCHEMA_LOCK_PASS / SCIENTIFIC_STATUS_NOT_RUN`。新增四个 machine-readable schema（FrameReceipt、TaskQuery、FactorPosterior、ObservationCandidate）、十个 identifiability 解析期望、两项 body-motion action filter、六个 factor-oracle mechanics case 与 S/P/B 八臂 factorial；validator 从 measurement-only Jacobian 重算强/弱子空间、finite task ambiguity 与非光滑分支，并重算 `8 arms × 2 modes × 6 cases=96` 份 payload/output/common-support hash。`R_weak` 与强子空间测量区间 `H_meas` 分字段，只有前者进入 2 cm identifiability gate，后者只扩宽最终 decision interval。ObservationCandidate 的 predicted/realized baseline、frame/query/cutoff/provenance、body-motion/realized receipt 交叉语义与 posterior covariance 对称半正定均机器冻结；FrameReceipt/TaskQuery/FactorPosterior/ObservationCandidate 的 frame/query/body/path、factor identity、timestamp cutoff 与 provenance 跨对象一致性全部 fail-closed，FrameReceipt max-source timestamp 是 anchor/posterior/candidate 的唯一因果水位；future timestamp、重复 evidence、anchor/outcome 共享、矛盾 posterior、非有限 truth、空/非空漂移 gate、causal-factor 标签/数值不一致、改名 runtime 与未支持 schema keyword 均 fail-closed，33/33 mutation tests PASS。通用治理 validator 为 `VALID / 0 error / 2 disclosed early-stage sealed-future-partition warnings`。科学审查后把 prior/LM/regularizer 排除在 measurement-only rank 之外，将 `Null(F) subset Null(J_C)` 降为极限诊断并以 finite weak-subspace task ambiguity 为主 gate；K corruption 与 factorial 分离，value-only common-support 与 full-block oracle 分离，active 失败后的 passive continuation 也必须另立版本而不得隐式进入 joint J0。当前未创建 solver/runner/artifact，未读真实 outcome，O0M implementation/execution authority=false；real O0R 因 complete factor truth、truth-clear factor bundle、continuous boundary/uncertainty truth、target timestamp/pose、deterministic injection adapter 和 fresh paired outcome 缺失而固定 `TARO_O0R_NOT_EVALUABLE_DATA_AND_INTERFACE`。唯一 successor 为 non-execution `TARO_O0M_SYNTHETIC_IDENTIFIABILITY_AND_FACTORIAL_MECHANICS_PROTOCOL_LOCK`；默认 App、产品与 safety authority 不变。
- 时间：2026-08-10（Asia/Hong_Kong）；执行者：violjjet。完成 AG-DUE 冻结 successor 的两份 SANPO metadata-only static prescreen。对 byte/hash-locked Real/Synthetic manifest 逐一调用冻结 R0 evaluator，未联网刷新 metadata、未下载或打开 RGB/depth/mask/pose payload；两者均确定为 `PARTIAL`，hard rejection 均为空，完整 screening match 均为 0。相关性只出现在 R2 F1 supervision 与 temporal presence，且只有 R2 F1 标为 upgradeable；QSF right-censor、corridor 与 FCI truth bundle 继续 absent。新增专用 exact-path governed-result replay validator 与 7 项 mutation tests，结果固定 `source_data_support_established=false`、`supported_for_protocol_lock=false`、`execution_authorized=false`。Synthetic 仅因锁定 inventory 中 metric depth/panoptic source-native candidate 较明确而成为下一窄审计对象，不构成 source admission 或数据质量排名。终态 `AG_DUE_R0_SANPO_INITIAL_STATIC_PRESCREEN_COMPLETE_BOTH_PARTIAL`；唯一 successor 为 `BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R1_SANPO_SYNTHETIC_SOURCE_SPECIFIC_INTEGRITY_AND_CAPABILITY_AUDIT_PROTOCOL_LOCK`，只允许另锁协议，不授权执行审计、payload、derivation、Teacher、物化、训练、Development/Confirmation、默认 App、产品或 safety。
- 时间：2026-08-10（Asia/Hong_Kong）；执行者：violjjet。按用户明确立项建立独立并行 `TARO / Task-directed Active Risk Observability` WILD_LAB 算法路线，并把 GaugeFix、PARA、TwinScene、AC4D 的方向族、去重边界和依赖关系登记到 `idea.md`。TARO 只吸收 GaugeFix+PARA：在有效 frame receipt 与独立米制锚之后，对低维 residual gauge 的可观测子空间建立 posterior，并在 task query 仍不可识别时比较被动历史与站定相机微基线；TwinScene 保持可选离线因果 factor 数据/蒸馏想法，AC4D 保持需先超过 D44+Kalman/IMM 困难分层 oracle 的独立未来想法。新增 TARO current 与 R0 详细路线指南，冻结主张/非命题、local task-query identifiability、schema、阶段 ladder、数据角色、强基线、时间/因果负控、拟议 kill gates、端侧预算和 UNKNOWN/reducer 边界。当前状态为 `DESIGN_AUTHORIZED / EXECUTION_NOT_AUTHORIZED / DEFAULT_APP_UNCHANGED`；唯一 successor 是 `TARO_P0_TASK_QUERY_IDENTIFIABILITY_AND_FACTOR_ORACLE_CANARY_PROTOCOL_LOCK`，只允许另锁非执行 CANARY_LITE 协议，不授权实现、数据读取、oracle、训练、用户提示、TwinScene/AC4D、Android/HTP、产品或 safety。Assistive Geometry、DepthART、AG-QSF/AG-CBF 与默认 App 的 current、终态和 successor 均不变。
- 时间：2026-08-10（Asia/Hong_Kong）；执行者：violjjet。完成 AG-DUE 唯一 successor 的 SANPO initial source manifest lock（protocol SHA `1C4F8577...7ECA`），未执行正式 prescreen。披露一次 diagnostic metadata bootstrap：只读取 official repo HEAD、pinned README 与 Real/Synthetic train split text，未下载/打开 RGB、depth、mask、pose 或其他 payload；official-train identity 数为 `560/1,560`，各按 deterministic last-ID 锁一个未命中 tracked workspace/ledger 的 discovery-fresh session。两份 manifest 的 capability frame/orientation/parent count 均为 0，quality 全为 `CHARACTERIZED_NOT_VALIDATED`、camera/upright basis 全为 `UNKNOWN`；Real 的 ZED/CREStereo depth 不冒充 oracle，Synthetic metric depth/panoptic/pose 也不冒充 support/boundary/frame-bound transform truth。manifest lock validator 与 7/7 mutation tests PASS；正式 `PRESCREEN_ADMIT/PARTIAL/REJECT` 尚未产出，payload integrity/count audit、derivation、Teacher、materialization、训练与 outcome access 均未授权。唯一 successor 为 `BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R0_SANPO_INITIAL_MANIFEST_STATIC_PRESCREEN_EXECUTION`；R2 adapter/supervision blocker、默认 App、产品与 safety authority 不变。
- 时间：2026-08-10（Asia/Hong_Kong）；执行者：violjjet。冻结非算法数据工程 `BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R0` 的 gap-driven source prescreen（protocol SHA `3F7B72A9...AD5F`）：把 AG-DCA 已版本化 QSF/CBF/FCI 门与 R2 F1 三 factor supervision/temporal presence screen 绑定为 gap contract，并冻结 source identity/license/privacy/access、ancestry/independence、claim-bound evidence receipt、source-object/field mapping、alignment/registration/units/coordinate verifier、parent namespace/orientation basis及 B1 protected-role roster 防火墙。metadata-only 输出仅为 `PRESCREEN_ADMIT/PARTIAL/REJECT`；即使正向也固定 `source_data_support_established=false`、`supported_for_protocol_lock=false`、`execution_authorized=false`，只能另锁 source-specific integrity/payload-audit protocol。multi-Teacher/single-Teacher/heuristic/VLM 均不能填充冻结 truth gate，B1 consumed Development/Calibration/Confirmation 不能包装为 fresh FCI。15/15 mutation tests 与静态 validator PASS；未锁真实 source manifest、未打开/下载 payload、未调用 Teacher、未生成 pseudo-label、未物化或训练。R2 F1 的 `FactorTensorAdapter` ABI 与 supervision source/label contract 仍是独立 blocker。唯一 successor 为 `BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R0_INITIAL_SOURCE_MANIFEST_LOCK`，默认 App、产品与 safety authority 不变。
- 时间：2026-08-10（Asia/Hong_Kong）；执行者：Codex。完成 Assistive Geometry R2 F1-P→F0 byte-frozen 接口静态审计，终态 `R2_F1_EXECUTION_BLOCKED_FACTORTENSOR_ADAPTER_ABSENT`。保持 F1 factor schema SHA `8016430D...5F7E`、F1-P lock result SHA `55BD2566...5F67` 与 F0 reducer SHA `2D6C26AD...2092` 不变，确认当前不存在 hash-bound、learned-graph 外的 deterministic `FactorTensorAdapter`：dense `depth_log_sigma_hw` 没有变成 scalar `scale_sigma_m` 的校准/聚合；support 缺 `normal_sigma_rad` 和 `height_sigma_m`；dense depth/boundary/evidence tensors 没有变成 ordered metric obstacle list 的 component/split-merge/interval/evidence-sigma 规则；camera receipt 也没有 K/transform/gravity/orientation/coverage 的逐字段 frame binding。新增静态审计器与 7/7 mutation tests；未实现或执行 adapter、未运行 reducer canary、未物化、未定义模型、optimizer step 为 0。唯一 successor 改为 `BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTORTENSOR_ADAPTER_PROTOCOL_SCHEMA_AND_MUTATION_CANARY_LOCK`，execution authority=false；原 supervision source/label contract 仍是 adapter 前门之后的独立必要 blocker，F1/F2、teacher、temporal、mobile、默认 App、产品与 safety 继续未授权。
- 时间：2026-08-10（Asia/Hong_Kong）；执行者：Codex。完成且仅完成 Assistive Geometry R2 `F1-P` protocol lock，终态 `R2_F1_PROTOCOL_FROZEN_EXECUTION_NOT_AUTHORIZED_SUPERVISION_FRONTDOOR_UNSATISFIED`。冻结 14 个 factor prediction 字段、显式 validity/provenance/UNKNOWN 语义、13 个独立 loss、`FIT/CHECKPOINT_SELECTION/TRAIN_CANARY=8/2/2` parent-disjoint 最低角色、factor-only checkpoint Pareto/tie rule、三 factor conjunctive learnability/uncertainty proper-score 标准、8 项 Kill Gate 与 F2 admission；aggregate loss 与 reducer/task metric 均不得选 checkpoint 或拯救 factor failure。绑定 AG-DCA 当前能力矩阵：metric depth `4,767/16 parents`，support `320/11`，crisp obstacle `1,557/11`，但 continuous boundary truth 与 complete R2 factor-schema truth 均为 `0/0`；直接 uncertainty truth 为 0，只允许未来冻结的 parent-disjoint residual proper score，不得伪造 constant sigma。通用 protocol validator 与专用 static validator 均 0 error，9/9 mutation tests PASS。未创建 label materializer、factor model、trainer、optimizer step、checkpoint 或 training artifact root。唯一 successor 为 `BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_SUPERVISION_SOURCE_AND_LABEL_CONTRACT_LOCK`，execution authority=false；F1/F2、teacher、temporal、mobile/device、Calibration、Confirmation、默认 App、产品与 safety 均未授权。
- 时间：2026-08-10（Asia/Hong_Kong）；执行者：Codex。完成非算法基础设施 `BLINDASSIST_ASSISTIVE_GEOMETRY_DCA_R0` 的首次且唯一 full-TRAIN atlas execution：逐 bytes/SHA 扫描 16 parent × 300 = 4,800 target，41.002 s 产出 23,817-byte atlas（SHA `12EB3B92...8DC7`）与 6,881-byte decisions（SHA `3BA0445C...4BFD`）。QSF H1 reopen 因 right-censor 仅 59 帧/1 parent/portrait-only，冻结 `NOT_SUPPORTED_DATA`；CBF R0-style grid 因 full grid 仅 196 帧/4 parent、portrait/landscape 158/38，冻结 `NOT_SUPPORTED_DATA`；FCI-for-R2-decision 因 complete factor schema 与 truth-clear factor bundle 均为 0、joint parent 为 0，且 oracle injection interface 与 fresh selection-eligible paired outcome 均未冻结，冻结 `NOT_SUPPORTED_DATA_AND_AUTHORITY`。因此未创建或启动 AG-FCI。R0 终态 `AG_DCA_R0_COMPLETE_THREE_HYPOTHESES_NOT_SUPPORTED`，无活动 successor；checker/atlas schema 保留供新的版本化 requirements 重放。未读 RGB、模型、feature、checkpoint、outcome、Development/Calibration/Confirmation，UNKNOWN 未当 negative；不产生因果归因、算法执行、R2 选择、默认 App、产品或 safety authority。
- 时间：2026-08-10（Asia/Hong_Kong）；执行者：Codex。冻结非算法基础设施 `BLINDASSIST_ASSISTIVE_GEOMETRY_DCA_R0`：hash-bind B1 TRAIN target manifest `A6F809C7...A7C2`、B1 全角色 protocol、16 unique visit/video、4,800 target 精确 path/bytes/SHA/keyset/dtype/shape、parent 内时间顺序、protected Development/Confirmation identity 与 CPU-only 输出。atlas 将按 frame/parent/orientation 汇总 clearance event/right-censor、ground、forward/lateral coverage、full grid、occupancy、source-order temporal pair、camera geometry、truth clear/occupied 与 factor truth；机器 checker 同时冻结 QSF、CBF、FCI 的 joint parent-disjoint requirements。R2 F0 reducer 已在 `ee3c9d8d` 通过并作为 synthetic mechanics authority 绑定，但 real oracle-injection、完整 uncertainty/boundary factor truth 与 fresh paired outcome 仍为 false；B1 consumed Development 继续禁止作为 R2 selection。当前唯一 successor 是首次 full-TRAIN atlas execution；任何 PASS 只允许另锁协议，不授权算法、训练、F2、默认 App、产品或 safety。
- 时间：2026-08-10（Asia/Hong_Kong）；执行者：Codex。只解锁并完成 Assistive Geometry R2 F0 synthetic factor geometry kill gate，未解锁 F1。先将 parent R2 hypothesis、零参数 `geometry_r2_interval_reducer_f0_v1`、formal runner、9 项 focused tests 与 23-case analytic fixture suite 逐 SHA 冻结并通过 research protocol validator（0 error / 0 warning），再首次执行 formal canary。22 个 normal case 的 depth/scale、support、boundary、orientation 与 tri-state 真值逐项 exact，1 个 learned `final_state` shortcut 负控被 `FORBIDDEN_FACTOR_FIELD` 拒绝；22 次 replay deterministic，10/10 gate PASS。depth noise、scale uncertainty、support uncertainty、boundary blur 四条 degradation ladder 均只保持原状态或退化到 `UNKNOWN`，没有 uncertainty-only `CLEAR→OCCUPIED`；12 个宽阔平地、侧障碍中路开放、远障碍、局部缺深度、support/边界退化及不足 scale evidence 的反 A0 counterexample 均无 unsupported occupancy。终态为 `BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F0_SYNTHETIC_FACTOR_GEOMETRY_CANARY_PASS`，claim ceiling 仅为 synthetic reducer mechanics；零模型、零真实数据、零训练。唯一 successor 改为 `BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_TRAIN_ONLY_FACTOR_LEARNABILITY_PROTOCOL_LOCK`，只允许另行冻结 F1 协议；F1 数据物化、初始化、训练、F2 Development、teacher、temporal、mobile/device、Calibration、Confirmation、默认 App、产品与 safety authority 仍全部为 false。
- 时间：2026-08-09（Asia/Hong_Kong）；执行者：Codex。执行冻结的 AG-CBF R0 TRAIN-only grid data-support audit 并在 oracle 前 fail-close。逐 bytes/SHA 复核 16 parent × source-order-even 64 target，只读取八项 source geometry 字段；1,024 帧仅 `44` 帧满足 `32×31`、forward `0.2–5.0 m`、lateral `-2.0–2.0 m` 的 ground/observation 纵横覆盖门，portrait/landscape 为 `36/8`，0/16 parent 达到 `32/64`。重叠失败中 `974` 帧缺 longitudinal quartile ground support、`816` 帧缺 lateral-third observation、`288` 帧 ground-plane/geometry contract 无效，冻结 terminal `AG_CBF_R0_DATA_SUPPORT_NOT_EVALUABLE_ROUTE_CLOSE`。未打开 RGB、模型、feature、A0 consumed Development、Calibration/Confirmation，未计算 oracle outcome，未建模或训练，UNKNOWN 未当 negative。本 R0 无 successor；这不反证 corridor bottleneck 数学，未来只能以新的 pre-outcome source-geometry/target contract 与独立版本从 DATA SUPPORT 重启。
- 时间：2026-08-09（Asia/Hong_Kong）；执行者：Codex。建立独立 `BLINDASSIST_ASSISTIVE_GEOMETRY_CBF_R0` 并行 WILD_LAB 路线，只冻结 Phase 0 的 TRAIN-only grid data-support audit，保持 `DATA SUPPORT → ORACLE CEILING → REPRESENTATION VALUE → TRAIN` 强顺序。协议 hash-bind B1 TRAIN target manifest `A6F809C7...A7C2`、16 parent、source-order-even 64 帧/parent、只读八项 source geometry 字段、`32×31` ground-aligned 网格与 parent/orientation/纵横覆盖 gate；逐 target 将复核 bytes/SHA，并硬拒绝 Development/Confirmation、RGB、模型、feature 与自定义协议扩权。现有 H3 仅作为 synthetic mechanics predecessor，不继承真实 oracle authority；UNKNOWN 不得当 free/occupied/negative。当前唯一 successor 是执行数据支撑审计；oracle outcome、body inflation、representation-value、模型、训练、Android/HTP、默认 App、产品与 safety 均未授权。
- 时间：2026-08-09（Asia/Hong_Kong）；执行者：Codex。按三层结案永久关闭 Assistive Geometry B1-A0，并建立独立 R2 因子化几何假设。第一层冻结 `B1_A0_PROGRAM_PERMANENTLY_CLOSED_NEGATIVE_TERMINAL`：保留三 seed / 12 checkpoint、seed 29 OOM+full restart、1,200-frame Selection manifest、3,600-frame observations、正式 evaluation 与所有失败/被 supersede receipts；禁止 reseed、reloss、加 epoch、改 threshold、best-seed 或激活 A1–A4。第二层只读 failure anatomy 在同一 SHA-bound consumed Selection 上完成，终态 `B1_A0_FAILURE_ANATOMY_COMPLETE_NOT_ELIGIBLE_FOR_PROMOTION`：false-block 为 `841/852/870 of 1,139`，全部与 predicted-clearance threshold crossing 内部一致；signed clearance bias `-0.216/-0.226/-0.256 m`，跨 seed false-block Jaccard `0.924–0.936`，transition failure 的 `797/804/818` 为持续 truth-clear / predicted-occupied，非主要由 flip/jitter 引起。全部 truth-clear 支持集中 parent `464241`，因此不作全场景外推，也不因果指定 depth scale 或 ground/support 单一罪因。第三层 outcome-blind 冻结 R2：learned graph 只输出 metric-ish depth、support surface、obstacle boundary/evidence 与 uncertainty，最终 tri-state 由 deterministic `GeometryR2Reducer` 独占；顺序门为 F0 synthetic mechanics → F1 TRAIN-only factor learnability → F2 全新至少 8-parent Development。B1 Selection/threshold 不继承，teacher 仅 factor-level upper bound，temporal/mobile 仅接口。唯一 successor 为 `BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F0_SYNTHETIC_FACTOR_GEOMETRY_CANARY_PROTOCOL_AND_FIXTURES`，当前 execution authority=false；所有 Calibration/Confirmation、训练、teacher、移动、时序、默认 App、产品与 safety 均未授权。
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
- 时间：2026-08-10（Asia/Hong_Kong）；执行者：violjjet。完成 TARO `TARO_O0M_IMPLEMENTATION_LOCK_PASS / SCIENTIFIC_STATUS_NOT_RUN`：新增独立 `taro_o0m_runtime`，以 NumPy SVD 重算 measurement-only strong/weak projector、finite task ambiguity 与 measurement interval，并实现 truth-blind factorial patch、action filter、严格输入白名单、非轴对齐重参数化、uncertainty 单调性和 one-shot runner。13/13 disjoint `impl_unit_*` tests PASS，未调用 P0/O0M static evaluator，未运行正式 10+80+2 fixture，exclusive artifact root 仍不存在。唯一 successor 为另提交 `TARO_O0M_ONE_SHOT_EXECUTION_LOCK`；当前 execution、real data、training、device、默认 App、产品与 safety authority 均为 false，O0R 继续 `TARO_O0R_NOT_EVALUABLE_DATA_AND_INTERFACE`。
- 时间：2026-08-10（Asia/Hong_Kong）；执行者：violjjet。提交 TARO `TARO_O0M_ONE_SHOT_EXECUTION_LOCK / AUTHORIZED_UNCONSUMED`：绑定 protocol、fixture、implementation lock、两个独立 static oracle、mechanics、runner、tests 的 exact SHA/bytes，冻结 argv、Python/NumPy 线程环境、30 s/256 MiB/1 MiB、四文件 artifact set 与 G01–G10。锁定时 exclusive root 不存在；当前科学状态仍 NOT_RUN。下一动作只能按 exact argv 运行一次，root 创建即消费，不得覆盖、删除或重跑；real data、training、active prompt、device、默认 App、产品与 safety 均未授权，O0R 继续 `TARO_O0R_NOT_EVALUABLE_DATA_AND_INTERFACE`。
- 时间：2026-08-10（Asia/Hong_Kong）；执行者：violjjet。完成 TARO `TARO_O0R_ARKITSCENES_SOURCE_ADAPTER_IMPLEMENTATION_LOCK_PASS / SYNTHETIC_ONLY / SCIENTIFIC_STATUS_NOT_RUN`：新增纯内存 `taro_o0r_source_adapter_runtime`，冻结 exact Decimal timestamp/right-pose-bracket watermark、`8 ADAPTER_FIT + 16 O0R_EVAL_CANDIDATE` role roster、asset SHA/CRC 与 decoded-payload content receipt、内部 FARO/AppleDepth/confidence residual 及 private factory-bound/fingerprint-locked uncertainty model、eval-only deep-read-only FARO whole-geometry hash、由 support plane 重算的 9 query receipts、immutable common-support base、固定 DepthART model/checkpoint/metric-zero output receipt、TARO-specific reducer，以及带逐组件 lineage/parent context 的 8 arms × 2 modes injection。44/44 synthetic focused tests PASS；implementation lock JSON SHA-256 为 `6A040C040B45F1DE27700CBDE11DEA1B2FED1487BFDD718F8D62435ACBA1397E`。24 个 selected source body 未打开，future dataset/work/evidence roots 均不存在，真实 truth/uncertainty、DepthART 与 scientific O0R 仍 `NOT_RUN`。唯一 successor 为 `TARO_O0R_ARKITSCENES_TRUTH_ONLY_ONE_SHOT_PREFLIGHT_LOCK`（execution=false）；当前 candidate-relative SCALE correction、下载、物化、factorial execution、训练、G0/G1/A0/A1/J0、设备、默认 App、产品与 safety authority 均为 false。
- 时间：2026-08-15（Asia/Hong_Kong）；执行者：violjjet。按用户明确选择新开
  `SATOM-A / ACTIVE_REVERSIBLE_LANES=1`，保持 TARO、Assistive Geometry、Q-Plane、
  RCLE、USTRF 与 DepthART D3R6 的既有关闭/暂停终态不变。新增 SATOM-R0 deterministic
  host harness：VL53L1X 风格三 ROI range/noise/missing 模拟、ToF cone 到 frozen prior
  surface 的关联、metric-pose warp 的三带 polar evidential memory、center/random/
  round-robin/max-entropy/task-weighted information-gain 五策略、单帧/ToF-only/uniform
  fusion comparator、shuffled timestamp/wrong extrinsic/wrong ROI 负控，以及 pooled/
  parent-macro/worst-parent/false-clear/false-block/coverage/clearance MAE/ECE。当前只授权
  合成 mechanics canary；其 prior 明确为 truth-derived synthetic，不冒充 DepthART 或
  real utility。唯一 successor 是在现有 Bonn RGB-D+pose Development parent 上物化冻结
  DepthART dense prior 后执行最小 real E0；deterministic headroom 不成立即关闭，不训练、
  不接 Android、不改变默认 App 或 safety authority。
- 时间：2026-08-20（Asia/Hong_Kong）；执行者：violjjet。完成 L10M B5-A fresh
  generalization replication。Balanced/Control、8-generation budget、evaluator 与严格
  incumbent selection 均冻结自 B4-A；三张 fresh finite landscapes 在零模型调用下预先
  穷举合格，均需至少 5 个 strict-improvement steps，随后 9 个新 paired identities 完成
  144/144 调用。终态为 `B5A_EVALUABLE_COMPLETE /
  B5A_GENERALIZATION_NOT_REPLICATED`：Balanced 为 4 胜、1 平、4 负，median paired
  normalized-progress delta `0.0`，Control/Balanced mean progress 分别为
  `0.4804232804232803/0.513227513227513`，global optimum 均 `0/9`；unsafe、invalid
  均 `0/0`，operator integrity 通过，调用成本 `72/72`。因此 B4-A 的 cohort-relative
  search value 不被改写，但跨 cohort 泛化未建立，不授予
  `ADMITTED_L10M_SEARCH_OPERATOR`；以 B5-A pass 为前提的 B5-B 保持未授权，不重跑或
  事后修改 Balanced。
- 同日完成零模型调用、只读 consumed-evidence 的 B5-C Balanced effect heterogeneity
  autopsy。分析实现先以提交 `91a97137` 冻结，再验证 B4-A/B5-A exact result、manifest、
  event ledger 与 landscape certificate hashes 后运行。13 个 outcome-blind landscape
  特征没有产生跨 B4/B5 的 conditional-domain hypothesis；预声明的 finite-horizon
  exploration waste 与 nonproductive coverage projection 在 4 个 loss 中均为 `0/4`，终态
  为 `NO_REPRODUCIBLE_HETEROGENEITY_EXPLANATION_CLOSE_OPERATOR_ADMISSION_ROUTE`。
  描述性分解显示 B4→B5 的 Balanced mean progress 变化 `-0.071022`、Control 变化
  `+0.155637`、相对优势变化 `-0.226659`；18/18 首次分叉均是
  `MODEL_UNTRIED_DIRECTION`，无 `COVERAGE_PROJECTION`。结果后汇总观察到 Balanced
  terminal strict-path reachability 在 `3/4` losses、`0/14` non-losses 中更差，且 4/4
  losses 的首次分叉即时分数不低；该线索与长期 basin 可达性问题一致，但不在冻结终态
  签名内，只能保留为未准入 retrospective clue，不能授权 V2、conditional domain 或新
  fresh budget。Balanced operator-admission route 正式关闭，B4 局部结果与 B5 泛化负终态
  均保持不变，B5-B 继续未授权。
