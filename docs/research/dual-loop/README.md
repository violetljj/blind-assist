# BlindAssist YOLO + 语义分割双环研究主线

状态：`DG_SRF_F0_COMPLETE / VALID /
STRUCTURAL_SIGNAL_NOT_SUPPORTED_STOP /
INFORMATION_CEILING_THREE_ARM_D0_VALID_MIXED_GAPS /
RISKSEG_R0_TASK_AND_DATA_CONTRACT_FROZEN /
FULL_SEQUENTIAL_EXECUTION_AUTHORIZED /
EVENT_EVAL_FROZEN_ADEQUATE /
PIDNET_S_TECHNICAL_PREFLIGHT_PASS / TRAINING_IMPLEMENTATION_LOCKED /
RISKSEG_R0_COMPLETE / VALID_NEGATIVE_DEVELOPMENT_RESULT /
RISKSEG_R0_EVENT_QUALITY_OR_STABILITY_FAIL /
PIDNET_S_TRAINED_FINAL_DEVICE_PASS /
RISKSEG_R0_TRAINED_NOT_PROMOTABLE_KEEP_YOLO /
YOLO_RULE_PATCHING_STOPPED /
RISKSEG_R1_P0_COMPLETE_VALID_NEGATIVE /
TRUTH_MASK_SOFT_ADAPTER_FAIL_CHANGE_ACTIONABILITY_LABELS /
RISKSEG_R1_P1_NOT_AUTHORIZED /
RISKSEG_ACT_A0_COMPLETE_VALID_NEGATIVE /
STOP_ACTIONABILITY_REFERENCE_CONSTRUCT_UNSTABLE /
ACTIONABILITY_FULL_FRAME_AND_TRAINING_NOT_AUTHORIZED /
OBJECTIVE_IMAGE_SPACE_CANDIDATE_INCREMENT_D0_COMPLETE_VALID_NEGATIVE /
STOP_FIXED_PIDNET_OBJECTIVE_CANDIDATE_NO_ROBUST_INCREMENT /
CONSUMED_SESSION_DISJOINT_THESIS_DEVELOPMENT /
TIMING_NOT_EVALUABLE_ONSET_INCOMPLETE /
SEGMENTATION_FAILURE_ATLAS_R1_TARGETED_EXPANSION_COMPLETE /
MECHANISMS_REPRODUCED / GATING_PARTIAL / RESIDUAL_WEAKLY_LABELABLE /
CONDITIONAL_GATING_R0_PRIMARY_VALID_NOT_SUPPORTED /
R0_1_SHADOW_VALID_NO_MATERIAL_NO_HETEROGENEITY /
BOUNDED_STATIC_HANDCRAFTED_GATING_FAMILY_STOP /
FP_WEIGHTED_SAMPLING_NOT_SUPPORTED /
SINGLE_FP_AWARE_SUCCESSOR_STOP /
LEARNED_COMPONENT_VALIDATOR_R0_NOT_SUPPORTED_AND_GATING_STOP /
VISUAL_ONLY_SIDECAR_R0_AVAILABLE /
THESIS_DEVELOPMENT_DEFAULT /
FINAL_CONFIRMATION_NOT_ACTIVATED / DEFAULT_APP_UNCHANGED`

最后核验：2026-08-04（Asia/Hong_Kong）

```text
FORWARD_GOVERNANCE: THESIS_FIRST_RESEARCH_GOVERNANCE_R4
DEFAULT_NEW_WORK_LANE: THESIS_DEVELOPMENT
DEVELOPMENT_REQUIRES_LEGACY_FORMAL_GATES: false
HISTORICAL_TERMINALS_IMMUTABLE: true
```

## 评价器有效性审计：EVAL-VALIDITY R0

在继续比较任何新视觉候选前，已冻结
[EVAL-VALIDITY R0 合同](EVAL_VALIDITY_R0_CONTRACT_2026-08-02.md)：它不训练模型，要求一个
新的 session-disjoint cohort 先把 native scene facts 与 output-blind 的
`reminder_now / cleared / knownness` event facts 分开；后者的双 reviewer 一致性必须先于
YOLO、truth box、truth mask 和 synthetic oracle 的输出访问。四臂使用同一 decision chain，
并按 oracle 单调性检验“更丰富输入不得产生更差事件结果”。当前为
`PRE_OUTPUT_LOCKED / DATA_ADMISSION_NOT_STARTED`；旧 RISKSEG 30-event cohort 保持
`CONSUMED`，不得重算、调阈值或用于此审计。

## 独立未来潜力支线：HFTF

HFTF 是独立于当前 RISKSEG/YOLO 主线的未来潜力支线；只有同口径证据持续超过主线，
才另立主线切换合同。最新
[MetricTraversabilityField shadow/demo R0](../hftf/METRIC_TRAVERSABILITY_FIELD_SHADOW_DEMO_R0_IMPLEMENTATION_RESULT_2026-08-04.md)
已把丰富米制观测场、身体扫掠包络、`UNKNOWN` 与末端提醒映射在独立侧车中解耦；
它是 `DEVELOPMENT_ONLY / SHADOW_DEMO` 软件能力，不是新效果证据，`MAINLINE_UNCHANGED / DEFAULT_APP_UNCHANGED`。
此前
[D8 THOR-MAGNI 局部路线监督](../hftf/HFTF_STAGE_C_D8_THOR_MAGNI_LOCAL_ROUTE_SUPERVISION_2026-08-02.md)
已从 19 个 Pupil/QTM sessions 物化 1,078 个 source-native local-actionability
样本。冻结 MobileNet 的五折筛查中，history 相对 current-only 的近距/走廊 AUROC
均 5/5 折提升，mean delta 为 `+.0559/+.0511`；但 48-cell occupancy AP 为
0/5 折提升，连续距离排序也不稳定。随后相同 4,610 参数与训练预算的等容量
temporal-head 对照未复制该 coarse 增量：近距 AUROC/AP 仅 2/5 折为正，走廊 AP
也仅 2/5。较高维 separability signal 保留为观察，但不能升级为 history 独立增量；
当前 pooled-feature 路线关闭，不继续硬救完整 field，主线与默认 App 不变。
最后保留 `4×7` spatial map 的等容量对照只在走廊 AUROC/AP 上产生
5/5 fold 的小增量（mean `+.0040/+.0038`），近距仍失败，走廊 AP 也只有
9/15 fold×seed units 为正。该 corridor-specific weak signal 不改变总终态；
THOR frozen-backbone 搜索停止，下一变量必须来自独立来源。
随后
[D9 JRDB 独立数据集复现](../hftf/HFTF_STAGE_C_D9_JRDB_CORRIDOR_REPLICATION_2026-08-02.md)
使用四个本地 RGB360+3D-person sequences、104 个 samples 和两个完整 source-pair
folds 复现该 corridor signal。history-minus-current corridor AUROC/AP 在 0/2
folds 为正，mean `-.00235/-.00152`；六个 fold×seed units 中 AP 为 0/6 正。
终态 `D9_JRDB_TEMPORAL_SPATIAL_CORRIDOR_REPLICATION_NOT_SUPPORTED`，当前 HFTF
frozen-feature history route 停止。该科学负结果不删除 D8 局部监督资产，也不改变
研究主线或默认 App。为区分 frozen-feature ceiling 与 history 本身，后续又执行
[D10 trainable-tail temporal canary](../hftf/HFTF_STAGE_C_D10_TRAINABLE_TAIL_TEMPORAL_CANARY_2026-08-02.md)：
解冻 MobileNet blocks `9..12`，current/history 使用相同 765,386 个 trainable
parameters。近距与走廊 AUROC/AP 四项 history-minus-current 都只有 2/5 folds
为正，mean delta 分别为 `-.000235/+.000004/-.000403/-.000546`。冻结扩展门失败，
终态 `D10_TRAINABLE_TAIL_TEMPORAL_INCREMENT_NOT_SUPPORTED_STOP`；不扩 seed、
不启动 JRDB zero-shot、不调 recipe 救援。该终态只关闭当前 late-tail temporal
residual successor，不关闭未来实质不同的时空表示研究。随后
[D11–D13 future-onset 任务修正](../hftf/HFTF_STAGE_C_D11_D13_TRUE_FUTURE_ONSET_REPAIR_2026-08-02.md)
发现原 0–2 秒 future-ever 标签包含 `t=0`，current-static QTM geometry 五折 AUROC
已约 `.89–.97`，history kinematic oracle 没有稳定增量。改为只预测当前安全样本的
未来 onset 后，近距有 157/530 positive/eligible、走廊 148/616，五折均可评价。
等容量 frozen-spatial history baseline 的四项 median delta 均为正，正折数为
`4/3/5/4`，达到冻结门；但效应只有约 `+.0008–+.0020`，走廊 AP mean 仍略负。
当前只记录 `D13_FUTURE_ONSET_TEMPORAL_SPATIAL_INCREMENT_SUPPORTED` 的弱
representation signal；下一变量是显式 motion，而不是退回 current-dominated 标签
或升级主线/App 主张。后续
[D14 显式 RAFT motion canary](../hftf/HFTF_STAGE_C_D14_EXPLICIT_MOTION_FUTURE_ONSET_CANARY_2026-08-02.md)
完成 4,312 个相邻 frame pairs。走廊 AUROC/AP mean delta 为
`+.0219/+.0240`，但 AP median 为负、仅 2/5 folds 正；近距 AP mean
`-.0103`。终态 `D14_EXPLICIT_MOTION_FUTURE_ONSET_INCREMENT_NOT_SUPPORTED`。
folds 0/1 的走廊局部正信号保留，但不按结果切换 target 或调 RAFT/grid/head；
true-onset estimand 保留，主线与默认 App 不变。随后
[D15 JRDB true-onset 独立复现](../hftf/HFTF_STAGE_C_D15_JRDB_TRUE_FUTURE_ONSET_REPLICATION_2026-08-02.md)
物化 proximity 14/102、corridor 10/71 onset-positive/eligible，两个固定
source-pair folds 均含正负例。等容量 history replication 的 corridor AUROC/AP
两折 seed-mean 都为负，aggregate `-.00618/-.03098`，终态
`D15_JRDB_FUTURE_ONSET_HISTORY_REPLICATION_NOT_SUPPORTED`。D13 弱正信号因此只保留
在 THOR source-local representation 层，不升级为跨数据集效应。
随后
[D16 TartanGround true-onset baseline](../hftf/HFTF_STAGE_C_D16_TARTANGROUND_TRUE_FUTURE_ONSET_BASELINE_2026-08-02.md)
在 15 environments、495 samples 上物化 19,478 eligible cells 与 1,652 onset
cells，四个 near/far × body/head targets 在三折都有正负。相同 frozen-spatial
history recipe 的 near 增量仅千分位，far body/head 多数为负，终态
`D16_TARTANGROUND_FUTURE_ONSET_HISTORY_INCREMENT_NOT_SUPPORTED`。这关闭 frozen
single-frame feature + post-hoc temporal residual family；true-onset task 保留，
下一表示必须在预训练阶段共同编码时序。

此前 D5/D6 Development 已推进到
[outcome-unseen TartanGround 与 SANPO real-event transfer](../hftf/HFTF_STAGE_C_D5_TARTANGROUND_DEVELOPMENT_PILOT_2026-08-02.md)：
directional + fixed v2 在 6 个未参与开发的合成 environments 上 event recall
9/9 提高、false-active 7/9 降低；随后在已消费的 30-session SANPO 人工审阅事件集上，
9 个 directional checkpoints 全部命中 16/16 正事件，但误报 13–14/14 负事件、
只清除 0–2/16。真实 recall signal 已成立，specificity/clearance 与相对当前 YOLO 的
Pareto 增量尚未成立，因此研究主线和默认 App 不变。下一步只做相对方向 profile
诊断，不继续盲调绝对阈值。该诊断现已发现 central-minus-lateral 对 parallel-curb
的 9-checkpoint AUC mean 只有 `0.499`，终态为
`CENTRAL_VS_LATERAL_ACTIONABILITY_PROFILE_NOT_SUPPORTED`；后继转为
source-session-held-out 的低容量 weak relation head。该 head 相对 fixed v2
在 9/9 backbones 上减少 false alerts 并增加 clearance，但 recall 8/9 下降且
0/9 Pareto-dominate YOLO；保留 guardrail signal，下一步把关系监督前移到固定
encoder spatial feature map。该 fixed-encoder spatial head 已完成：相对
output-field head，false alerts 9/9 减少、mean `-2.22`，cleared mean `+1.78`，
hits mean 仅 `-0.22`，证明 spatial representation placement 的 Development
增量；但绝对 mean 为 `13 hits / 9 false alerts / 9 cleared`，对当前 YOLO
`13/6/5` 仍是 0/9 Pareto。表示层正结果保留，研究主线与默认 App 不变；下一步
先审计互补性与样本容量。后续事件配对发现 HFTF 平均补回 YOLO `2.56/3` misses，
event-level OR hits 达 `15.56/16`，建立了
YOLO-HFTF 互补性正信号；但 9-checkpoint 静态 fusion mean 只有
`12.89 hits / 9.78 false alerts / 6.89 cleared`，rank-2 与 1 秒 causal
canaries 也未改善 Pareto。当前 30-event cohort 的 head search 停止；下一步先扩充
隔离的真实 relation supervision，未新增监督前不解冻 backbone。该跨源实验现已
使用 14 个外部 provisional episodes 建立高召回信号：output-field 9-checkpoint
mean 为 `14.44 hits / 13.33 false alerts / 3.78 cleared`。加入经复核 normal
passage negatives 并改用固定 spatial feature 后，mean 为
`12.33/8.78/7.22`；相对外部 output-field head 的 false alerts 9/9 改善，
但召回下降且 0/9 超过 YOLO。统一阈值和低置信 parallel-curb canary 都只移动
trade-off。跨源空间关系正结果保留在 representation/guardrail 层，研究主线和
默认 App 继续不变。随后
[D6 多源关系监督 canary](../hftf/HFTF_STAGE_C_D6_RELATION_SUPERVISION_CANARY_2026-08-02.md)
加入 Luna merged relation pool 与 r789 人工状态转移片段；两者都没有超过上述
reviewed-normal-negative reference。更关键的是，11-source public-video LOSO
对 intervention 的 frame/segment recall 都为 `0`，表明现有 fixed HFTF spatial
feature 不能把 actionability relation 迁移到新来源；即使使用 held-out source
人工 no-alert baseline 的 centering oracle，intervention recall 仍为 `0`。
固定 13,137-parameter nonlinear 3×6 relation encoder 再加入 30 个 consumed
SANPO support sources 后，public-video held-out intervention recall 仍为 `0`，
frame BA 只有 `0.4394`。进一步解冻 810,472 个 HFTF tail parameters 做 paired-RGB
训练，在 deterministic repeat A/B 中仍为 `0` intervention recall，frame/segment
AUROC 为 `0.5034/0.3377`。该负结果不撤销
spatial-over-output-field 正结果，只关闭“增加关系监督即可救固定 backbone”的窄
假设以及当前 paired-RGB tail recipe；下一阶段必须先新增独立正来源或新的预训练
任务，再通过 source-heldout actionability recall。后续配对预训练已进一步拆开
任务可学习性与真实域迁移：TartanGround 的 6-parent→2 outcome-unseen parent
取得 frame BA/AUROC `0.7098/0.7124`、episode BA/AUROC `1.0/1.0`，证明 synthetic
paired relation 可学习；但直接 synthetic→public 不支持。SANPO-only、public
零训练帧的直接迁移在 Edmonton 保留 frame/episode AUROC `0.7958/0.75` 的局部正
信号，Bangkok 近 chance、Ulm 反向，3-source macro AUROC 只有
`0.4604/0.4167`。TartanGround→SANPO 课程又降到 pooled frame AUROC `0.4920`，
没有增量。正结果分别保留在 synthetic learnability 与 Edmonton source-local
ranking 层；三来源 source-general transfer 未建立。下一 representation 必须在
backbone 内联合比较 frame pair 或直接学习人体包络未来风险场，不再继续
encode-then-difference 的 tail/head/threshold 搜索。
该 joint-pair canary 随后已用一个 28,313-parameter raw-pair stem 完成：
`current/baseline/signed delta/abs delta` 在 backbone 外早期联合编码，再与冻结
HFTF context 拼接。相对 encode-then-difference，pooled frame alert recall 从
`0.275` 升到 `0.375`、no-alert recall 保持 `0.8621`、BA/AUROC 从
`0.5685/0.5811` 升到 `0.6185/0.6978`；重复运行逐字段一致。但 Edmonton frame
AUROC `0.8134` 的同时，Bangkok/Ulm 只有 `0.1836/0.2582`，source-macro AUROC
`0.4184`。因此 early interaction 的局部表示增量保留，source-general transfer
仍不成立；下一步把 early interaction 放回 structured HFTF future-risk field
teacher task，而不是继续优化 binary actionability head。
该 structured-field 后继现已完成 3 seeds × 3 folds。31,560 个可训练参数的
zero-initialized early-pair residual 相对各自 directional-single reference，
environment-macro future body/head F1 为 6 正/3 个精确 epoch-0 中性/0 负，
mean `+0.00565`；但 dev fixed-event recall/FPR mean delta 为
`+0.00761/+0.04808`，没有形成行动层增量。在 6 个 outcome-unseen synthetic
environments 上，AUROC/AP/FPR mean delta 为
`+0.00757/+0.01065/-0.01387`，false-alert events 平均减少 `2.0`，但 macro F1
`-0.00135`、event recall `-0.00919`。因此保留 early interaction 的
ranking/specificity 表示信号，不升级为 event utility。下一候选不调 threshold：
冻结 directional baseline，让 pair module 只学习 selective residual/veto，并对
teacher-positive future body/head cells 显式施加 recall-preservation constraint。
该 ranker 后续虽在 synthetic outcome-unseen 上建立 AUROC/AP 增量，但真实 SANPO
迁移未成立。四段 3/3 model-blind RGB reject 的 150-frame 困难负例上，冻结
zero-training-true-alert threshold 只 veto `48/24,046` active model-cells，
中央方向只 veto `19/11,019`，仅一个模型清除一个窗口、无多数模型复现；全方向
field 清除为 `0/1,308` model-windows。30-session 人审 cohort 的 candidate/comparator
event-phase p95 AUROC mean 为 `.4613/.5714`，仅 2/9 单元增量为正。当前终态为
`D6_CONSERVATIVE_REAL_HARD_NEGATIVE_EXECUTION_NOT_SUPPORTED /
D6_SYNTHETIC_VETO_RANKING_REAL_TRANSFER_NOT_SUPPORTED`；synthetic ranking 正结果
保留。后续 source-session-held-out baseline-only vs candidate-aware Logistic
calibration 的 AUROC delta mean 虽为 `+0.01704`，但 median `-0.00833`、仅
`3/9` 为正，AP 与 paired direction 也仅 `3/9` 为正，终态为
`D6_CANDIDATE_AWARE_REAL_CALIBRATION_INCREMENT_NOT_SUPPORTED`。不再校准当前
candidate score；下一步只允许一个表示级 canary，把 real-phase supervision 放回
early-pair RGB interaction/structured field。该 canary 训练拟合成功，但在 7 个
held-out sessions 上 AUROC/AP 从 baseline `.7500/.6389` 降到
`.4167/.4444`，positive paired direction 从 `4/4` 降到 `1/4`，终态为
`D6_REAL_PHASE_SUPERVISED_EARLY_PAIR_CANARY_INCREMENT_NOT_SUPPORTED_STOP`。不扩展
seed/fold；下一变量必须改成 motion-aligned correspondence/flow representation，
不再改当前 head/loss/threshold。后续相同 54 维/L2 projection 的 sparse-LK
partial-affine residual 在 folds 0/1 双增量、fold4 双负，fold2/3 因局部 phase
alignment coverage 不足保持 `NOT_EVALUABLE`。终态为
`D6_MOTION_ALIGNED_PAIR_SEPARABILITY_SIGNAL_MIXED_NOT_READY_TO_TRAIN`；显式
ego-motion removal 信号保留，但当前 classical alignment 不进入 field training。
进一步以固定 pretrained RAFT-small 将 flow coverage 提到 `100%` 后，residual-flow
相对 raw pixel 只有 fold2 的 AUROC/AP 双增量，其余 4 folds 双负；raw flow 也没有
fold 双增量。终态为 `D6_RAFT_RESIDUAL_FLOW_SEPARABILITY_NOT_STABLE`。当前
30-session global phase cohort 停止更多模型变化，下一需求是新真实来源的 local
route/actionability correspondence。

此前路线进行到
[G0-D1 scientific design](../hftf/HFTF_STAGE_C_CURRENT_CLEARANCE_LEARNABILITY_D1_2026-08-01.md)
之后的
[timeline amendment](../hftf/HFTF_STAGE_C_CURRENT_CLEARANCE_LEARNABILITY_D1_TIMELINE_AMENDMENT_2026-08-01.md)
与
[Development execution contract](../hftf/HFTF_STAGE_C_CURRENT_CLEARANCE_LEARNABILITY_EXECUTION_CONTRACT_D1_2026-08-01.md)：
只比较 current RGB 下 direct-risk 与 signed-clearance 两种输出机制。9 个已
outcome-open 来源固定为 6 train + 3 model selection；三条 one-shot fresh 与三条
reserved official-test 尚未打开。[Development training result](../hftf/HFTF_STAGE_C_CURRENT_CLEARANCE_LEARNABILITY_TRAINING_RESULT_D1_2026-08-01.md)
现已通过独立 authority/label 重推、12 个 Phase A/B runs 与六个最终 checkpoint
hash freeze，终态为 `G0_D1_SIX_FINAL_CHECKPOINTS_FROZEN`。这只允许另立 fresh
一次性评估合同。该
[fresh execution contract](../hftf/HFTF_STAGE_C_CURRENT_CLEARANCE_FRESH_EXECUTION_CONTRACT_D1_2026-08-01.md)
现已在任何 fresh 媒体、teacher outcome 或 student prediction 打开前冻结：三条
session、75 个 current samples、六个 checkpoint、450 条 prediction、truth-free
prediction authorization、先 receipt 后单次 truth read、全部效果 gate 与失败后
不换源/不重跑均固定。该提交节点仍只是合同冻结，fresh source 当时尚未打开；
主线、默认 App、Android、reserved official-test、生产与安全权限均未改变。

合同推送后的一次性执行已由
[fresh execution result](../hftf/HFTF_STAGE_C_CURRENT_CLEARANCE_FRESH_EXECUTION_RESULT_D1_2026-08-01.md)
关闭为 `G0_D1_FRESH_EVALUATION_NOT_EVALUABLE_NO_SOURCE_REPLACEMENT`。第一个
固定 source 在 frame-0 depth 临时文件创建时因 263 字符 Windows 路径触发传输
失败；此前 metadata、首帧 RGB/mask 已打开，因此不允许改短路径重跑、补全 partial
root、继续另两条 source 或换源。该终态没有产生完整 package、opportunity、
prediction 或 truth join，既不支持也不拒绝 signed-clearance；主线与 App 仍不变。

其后的独立 D2 mechanics 问题没有救援 D1，而是比较 history-only 恒速因果 transport
与 current-field persistence。六源 one-shot 已由
[D2 result](../hftf/HFTF_STAGE_C_D2_CAUSAL_SIGNED_CLEARANCE_TRANSPORT_RESULT_2026-08-02.md)
封存为 `D2_NOT_EVALUABLE_OPPORTUNITY_INADEQUATE_NO_SOURCE_REPLACEMENT`：24 个
`parent × height × horizon` strata 仅 8 个通过，effect gates 未获判定，因此既不支持
也不否定 transport。D2 cohort 不重跑、不换源、不调参。

唯一新后继是
[D3-Q0](../hftf/HFTF_STAGE_C_D3_REFERENCE_AND_SUPPORT_ONLY_CHALLENGE_QUALIFICATION_Q0_2026-08-02.md)
条件机会挑战集设计：保留 D2 mechanics、opportunity/effect gates 全部不变，只允许
另冻 reference-and-support-only qualifier 与 outcome 前 effect skeleton。当前尚未
授权 D3 metadata scan、媒体、truth、effect 或 RGB student；研究主线与默认 App
继续不变。

2026-08-04 的已知相机高度地面尺度后继先尝试 fresh ARKit source gate，但固定的
4 个 parents 只有 2 个达到 pre-DA 高度代理资格，故以
`HOLD_SOURCE_AUTHORITY_NO_REPLACEMENT` 关闭 fresh effect。用户随后明确允许消费数据
用于开发；[消费开发结果](../hftf/KNOWN_CAMERA_HEIGHT_GROUND_SCALE_CONSUMED_RESULT_2026-08-04.md)
在 5 个 TartanGround parents、165 个冻结 anchors 上显示：R0 相对 raw DA 将
parent-macro clearance MAE 从 `1.0424 m` 降至 `0.4710 m`，false-clear 从
`.4672` 降至 `.1814`；后验、只用过去帧的 9-valid-scale 因果中值 R1 又改善到
`0.3949 m/.1510`。但 MAE、agreement、false-clear、temporal 四个绝对 gate 仍失败，
终态 `POSTHOC_CONSUMED_R1_ABSOLUTE_GATES_FAIL_STOP_OPTIMIZATION`。这些结果只能作为
同一消费 synthetic Development 信号，不授权继续 outcome-conditioned 搜索、购买 ToF、
主线切换或默认 App 变更。

同日又按目标允许的激进软件后继冻结了
[相机条件化轻量尺度学生](../hftf/CAMERA_CONDITIONED_SCALE_STUDENT_R0_RESULT_2026-08-04.md)。
固定十特征 ridge 在 5 个 parents 的 leave-one-parent-out Development 中通过全部
parent-macro gates；随后不改模型，在 10 个训练未见 parents、330 个 anchors 的外部
复现中再次全部通过：coverage `.9909`、clearance MAE `.1046 m`、agreement `.9899`、
false-clear `.0101`、temporal delta MAE `.0518 m`，且 10/10 parents 均优于 raw DA。
该信号只把路线提升为 `PROMOTE_TO_FRESH_REAL_PHONE_MECHANISM_CANDIDATE_NOT_DEFAULT_APP`：
数据仍是历史已消费 synthetic，精确 robot height/intrinsics 属于 oracle-quality receipt，
个别 parent 仍超局部门槛，真实手机 exact-timestamp depth cohort 也尚不可评价。因此不授权
ToF 采购、默认 App 接入、实时提醒或安全主张；下一门只能是另冻的固定高度/内参真机 shadow。

同日完成的[离线压力测试](../hftf/CAMERA_CONDITIONED_SCALE_STUDENT_OFFLINE_STRESS_R0_RESULT_2026-08-04.md)
没有重训或搜索。330 帧 cached-depth 结果显示学生能把单一 DA 全局尺度漂移 `±40%`
基本抵消，但不能修复局部 geometry：20% bandwise 形变的 accepted-bad 最高 `51.5%`。
50 帧 frozen-DA RGB 重放又显示 Gaussian `sigma=3` 时 coverage 仍有 `.86`，却产生
`.378 m` MAE 与 `.161` false-clear；下部 ROI 全宽遮挡 50% 时 coverage 仍有 `.976`，
MAE/false-clear 已到 `.321 m/.123`。因此真机合同必须在学生外绑定 camera/intrinsics、
CameraX crop/rotation、mount/gravity、ground support 与独立 blur quality gate；当前只保留
严格外部门控的 mechanism candidate，不授权 App 或安全输出。

[ARKitScenes 484248 反例审计](../hftf/SCALE_FREE_TRAVERSABILITY_R2_ARKITSCENES_COUNTEREXAMPLE_AUDIT_R0_RESULT_2026-08-04.md)
确认 R2 失败不是 global scale 或 nearest-fill 主导：150/150 帧按官方 pose 都需顺时针
90 度矫正，132/148 帧有大平面，完整 band 排序仅 `.480` 一致。解释性 orientation
反事实把 coverage/方向一致率从 `.224/.385` 提到 `.826/.895`，但不改写原 R2
`NOT_EVALUABLE`。scale-free 不作为辅助输出或 fallback；仅在 orientation receipt 后保留为
Development disagreement detector，未矫正输入直接关闭。

## 当前决定

2026-08-01，当前唯一算法主线切换为
[RISKSEG-R0](RISKSEG_R0_TASK_DATA_AND_EXECUTION_CONTRACT_2026-08-01.md)：停止继续修补
当前 YOLO 决策规则，训练一个四类轻量风险/可通行性分割模型。任务 ID 固定为
`0 walkable / 1 blocking_obstacle / 2 boundary_level_change /
3 unknown_nonwalkable`；旧 520-frame canonical mask 的 ID 1/2 含义相反，必须先按
`0->0,1->2,2->1,3->3` 物化带 hash 的重编码视图。520 帧按 source session 固定为
320-frame train / 200-frame dev；既有 90-frame / 3-event 集因两个 source-session
重叠且含 22 张相同 RGB，只做 contaminated non-gating regression smoke。
新 event-eval 必须与二者 session-disjoint，至少 30 个 parent events 并覆盖障碍、
台阶/落差、平行路沿和正常通行负例。历史
[数据门结果](RISKSEG_R0_EVENT_EVAL_DATA_GATE_RESULT_2026-08-01.md) 的
`HOLD_EVENT_EVAL_DATA` 保持不可变；随后新增 output-blind review bundle 与裁决，
已冻结 30 parent events / 30 source sessions，四桶精确为 `8/8/7/7`，successor
数据门为 `EVENT_EVAL_FROZEN_ADEQUATE`。

唯一模型候选是 PIDNet-S，技术预检固定 `512x288 / W8A8 / four-class`，要求 TFLite
和 QNN 都能编译、输出有限且尺寸正确、SM-S9280 冻结链 total P95 `<=100 ms`，并通过
10 分钟持续运行退化门。当前
[技术预检结果](RISKSEG_R0_PIDNET_S_TECHNICAL_PREFLIGHT_RESULT_2026-08-01.md)
已经 `PIDNET_S_TECHNICAL_PREFLIGHT_PASS`：SM-S9280 上 QNN HTP 完整接管
`163/163` 节点；7,619 次全链路 P95 `75.739 ms`，末/初 2 分钟 P95 比
`1.00255x`，failure 和 thermal status 均为 0。训练 recipe 已由
[implementation lock](RISKSEG_R0_PIDNET_S_TRAINING_IMPLEMENTATION_LOCK_2026-08-01.json)
绑定。完整执行结果现已由
[RISKSEG-R0 最终结果](RISKSEG_R0_FINAL_RESULT_2026-08-01.md)
关闭为 `RISKSEG_R0_TRAINED_NOT_PROMOTABLE_KEEP_YOLO`。三个固定 seed 均完成训练、
full W8A8 导出和 SM-S9280 三臂事件评价，但事件质量门 `0/3` 通过：YOLO 的正事件召回
为 `13/16`、false-alert event 为 `6/14`；learned 三 seed 分别为
`13/16、14/16、13/16` 和 `13/14、13/14、14/14`，共同命中中位延迟为
`+3/+5/+3` 帧。固定决策 seed 的最终 10 分钟 QNN/HTP 性能门本身通过：
7,727 样本、total P95 `77.374 ms`、inference P95 `5.198 ms`、末/初 P95 比
`1.07624x`、thermal/failure 均为 0，并两次完整委派 `173/173` 节点到 1 partition。
性能 PASS 不覆盖事件质量否决；不改默认 App，不接 learned segmentation，不在已消费
30-event cohort 上调参或增加规则。

其后显式冻结的
[RISKSEG-R1 P0 soft dense adapter audit](RISKSEG_R1_P0_SOFT_DENSE_ADAPTER_AUDIT_CONTRACT_2026-08-01.md)
只把该已消费 cohort 作为 post-consumption nested Development diagnostic，不恢复
fresh/held-out 身份。P0 已由
[结果](RISKSEG_R1_P0_SOFT_DENSE_ADAPTER_AUDIT_RESULT_2026-08-01.md)
关闭为 `TRUTH_MASK_SOFT_ADAPTER_FAIL_CHANGE_ACTIONABILITY_LABELS`：保留完整四通道
INT8 soft evidence 后，truth-mask family reference 为 `14/16` hits、`12/14`
false alerts、`4/16` cleared，未守住 YOLO 的 `13/16、6/14、5/16`。learned 三
seed 分别只有 `11/16、12/16、7/16` hits，完整相对 guardrails 为 `0/3`。因此
不进入当前四类目标的 P1 训练；下一步必须先改变 actionability/event supervision，
再建立新的 session-disjoint cohort。默认 App 继续保持 YOLO。

该后继已由
[RISKSEG-ACT A0 标签可行性审计](RISKSEG_ACT_A0_ACTIONABILITY_LABEL_READINESS_RESULT_2026-08-01.md)
关闭为有效负终态 `STOP_ACTIONABILITY_REFERENCE_CONSTRUCT_UNSTABLE`。A0 在已消费
30-event cohort 上冻结 120 个 content-blind anchors，并完成
`CURRENT_ONLY / CAUSAL_HISTORY / HINDSIGHT_REFERENCE` 三种信息条件各两路、共六路
fresh 隔离审阅；独立 validator 从六份原始 review 复算为 `VALID`。CURRENT/CAUSAL
的 alertable exact 均为 `.85`，但 boundary relation exact 只有
`.173/.241`、intrusion F1 `.632/.649`、parent-event sequence match
`.533/.600`；HINDSIGHT 也未形成稳定 reference，derived exact `.658`、
intrusion F1 `.553`、sequence match `.400`。因此不是只缺历史信息，而是当前路线
关系/空间侵入/事件相位合同本身不稳定。按预冻结 fail-fast 顺序，不进入 1,920-frame
A0-P1，不授权新 matched-pair cohort 合同或 actionability-aware 模型训练，不通过
prompt、第三 Agent、UNKNOWN 转负类、网格调优或专用 parallel-curb 规则修补结果。
默认 App 与 YOLO baseline 保持不变。

行动标签路线关闭后，当前主线转为
[Objective image-space candidate increment D0](OBJECTIVE_IMAGE_SPACE_CANDIDATE_INCREMENT_D0_PROTOCOL_2026-08-02.md)。
它不训练新模型，而把固定 PIDNet-S seed-20260801 仅作为 YOLO 框外客观区域候选
生成器，在剥离所有 `positive/bucket/alertable/passed/event` 字段的
30-session / 1,920-frame objective-only view 上评价 source-mask 像素/组件覆盖、
false area、碎片、面积变化、持续性和低成本图像空间算子。该 view 与 PIDNet
train/dev session-disjoint，但 cohort 已被前序研究消费，因此只提供
`THESIS_DEVELOPMENT` 机制证据。主集只有 4 个 onset-complete sessions，低于冻结的
12-session timing 门；本 D0 无论结果如何都不得声称“更早覆盖”，也不修改默认 App。
协议、门、exact model/data/implementation hashes 与负终态已在任何新 YOLO box trace
或 PIDNet pixel output 前冻结。当前
[D0 result](OBJECTIVE_IMAGE_SPACE_CANDIDATE_INCREMENT_D0_RESULT_2026-08-02.md)
已独立复算为有效负终态
`STOP_FIXED_PIDNET_OBJECTIVE_CANDIDATE_NO_ROBUST_INCREMENT`：`C-A` recall
`+.069380` 和固定预算通过，但 residual component recall 仅 `.285436`，
false components/frame `7.867188`，session P90 added FP `.127410`，boundary
recall gain 仅 `+.004039`。因此关闭 exact raw PIDNet candidate operator，不换 seed、
不加 gate/latch/组件分类器、不修改默认 App。

此前 2026-08-01 的候选算法主线曾切换为
[DG-SRF image-space structural complementarity F0](DG_SRF_IMAGE_SPACE_STRUCTURAL_COMPLEMENTARITY_F0_PROTOCOL_2026-08-01.md)。
它不继续救援已关闭的 segmentation gating，而是检验固定 Depth Anything V2 Small
相对逆深度中的 `N/E/R+/R-` 结构信号，能否在实际 YOLO coverage 外，以低于冻结 raw
DDRNet residual 的假激活代价，对 `boundary_step_curb / obstacle` canonical pixels
提供跨 source-session 稳定互补。

该 [F0 result](DG_SRF_IMAGE_SPACE_STRUCTURAL_COMPLEMENTARITY_F0_RESULT_2026-08-01.md)
现已 `COMPLETE / VALID / STRUCTURAL_SIGNAL_NOT_SUPPORTED_STOP`：520/520 q 健康，
但 D1-D4 均未形成跨组 stable signal；D4 macro AUPRC `.309456` 低于 frozen binary
DDRNet B 的 `.362109`，只在 1/10 组优于最佳单信号。LOSO 九门只通过 4/9，
overall/minimum-group/obstacle recall retention 为
`.254913 / .000019 / .139797`，false components/frame 为 `6.823077`。独立
validator 通过 29,031 项检查。当前精确定义的 DG-SRF F0 已关闭，不用同一 520 帧调参
或引入 Video Depth/时序救援；F1-F5、Android/QNN/A568、risk/feedback、提醒和默认
App 均未授权。

随后完成的
[信息上限三臂审计 D0](INFORMATION_CEILING_THREE_ARM_D0_RESULT_2026-08-01.md)
在同一 90-frame / 3-parent-event SANPO consumed Development cohort 与当前
`AssistDecisionKernel` 上得到有效混合终态：当前 YOLO 正事件 `0/2`、关键漏报 1；
mask-derived 真值风险框恢复为 `2/2`、漏报 0，但产生 53 个误提醒帧、负事件误报和
`0/2` passed 清除；source-native mask 经当前 adapter/source policy 后为 `2/2`、
0 漏报、0 误提醒、`2/2` 清除。独立 validator 从原始 mask/manifest 与逐帧输入账本
复算并 `PASS`。这支持停止继续为同一 YOLO 失败模式增加后规则，并在“只推进一个主模型
候选”时把下一 Development 候选设为轻量风险/可通行性分割；YOLO 仍冻结为 baseline，
默认 App 不变。由于 mask adapter 最终只转发一个框且 B/C source policy 不同，该结果
不单独证明 bbox 几何上限或 learned segmentation 效果。

2026-08-01 起，后续双环工作采用论文优先的 `DEVELOPMENT_STANDARD`，不再把旧 formal
R1 的 one-shot、fresh holdout、逐项 SHA 和全量独立复算要求复制到新的 Development。
这是一项只向前生效的执行降级：

- R1 的 `BLOCKED / NOT_EVALUABLE`、R2-P0 的 readiness 终态和全部旧 receipt 保持不可变；
- 新 Development 可在明确标注的 development/consumed 数据上修复输入映射、decoder、
  schema 或 runner，并以新 evidence version 重跑；
- Discovery 和算法早期默认使用 Development/consumed 数据，不再为每次小试验消费
  fresh holdout；mapping、decoder 与 tensor/schema 适配先用 synthetic canary 全覆盖；
- 可在最终选模前提前做 host/device runtime benchmark，但必须区分可参与候选排序的
  `ALGORITHM_SELECTION_BENCHMARK` 与只验证 backend/build/operator/memory/thermal 的
  `PLATFORM_ENGINEERING_BENCHMARK`；后者不得反向排序算法；
- 每轮至多推进一条主路线；本轮已选择 segmentation failure atlas、有限 gating probe
  与 residual labelability，模型比较、可视化平台和设备 sidecar 均不并行推进；
- 每轮最多 3 个候选，并必须交付新增召回、false components/frame、mask 碎片、
  组件稳定性、P95 推理/总链路成本中的适用指标，以及表、图、失败案例或 demo；
- 不接提醒、不改默认 App，不把 Development 结果写成安全、产品或最终确认结论。

该 Atlas 轮次本身完成了 200-frame consumed Development pilot 与固定 320-frame
定向扩展，没有训练模型、运行真机 benchmark 或融合实验。其后的 FP-aware DDRNet
单候选和本次轻量 Logistic component validator 均已按各自协议执行并到达负终态，详见
下文；二者都没有授权真机、融合或 Confirmation。最终 Confirmation 仍未激活；只有
Development 候选先通过冻结 utility/engineering 门时，才可另行冻结确认问题、独立数据、
实现和统计。

2026-07-31，用户明确要求以 Agent-only 标注检验“中央图像阻塞互补性”的可行边界；该
探索路线现已完成 successor calibration 并关闭。唯一获授权阶段曾是
[CENTRAL_OBSTRUCTION_AGENT_LABEL_READINESS_D0_A](CENTRAL_OBSTRUCTION_AGENT_LABEL_READINESS_D0_A_PROTOCOL_2026-07-31.json)：
按 `CANARY_LITE` 先对现有连续 RGB 做适配度排序，以冻结充分性规则选出最小 admitted
bundle；在排除式 calibration 上完成标签 pilot 和抽样审计规则，再用一个 primary
Agent 标注 admitted bundle。正例、`NOT_EVALUABLE`、歧义/terminal-changing item 和
冻结抽样进入第二路隔离 Agent；只有材料分歧才由 fresh 第三 Agent 裁决。本路线不设
人工队列；标注、抽样复核、裁决与验收均由隔离 Agent 自主完成。

D0-A0 采用 `REUSE_FIRST / FITNESS_FIRST / STOP_ON_ADEQUACY`：先按当时的中央阻塞
问题的可观察性、身份完整性、事件多样性、实际历史访问、claim overlap 和处理成本
排序现有可访问连续 RGB；满足冻结充分性规则后停止扩充，不要求为完整性遍历全部存量。
数据曾被其他算法、主线或实验使用，不构成按数据集名称排除的理由；相关历史
访问最多使受影响的最小身份单元降级为 calibration、Canary、Development、诊断、
回归或压力样本。只有当前问题所需信息确实缺失时才局部记为
`NOT_EVALUABLE_FOR_CURRENT_QUESTION`，不得全局封存优秀数据。

[D0-A0 输入宇宙冻结](CENTRAL_OBSTRUCTION_AGENT_LABEL_READINESS_D0_A0_RESULT_2026-07-31.md)
现已 `COMPLETE / VALID`：6 个完整 production session、34,279 帧、5 个 ancestry
group 已冻结；107 行 reuse-role ledger 将其余本地资产分别降级为 calibration-only
或当前问题不可评价。D0-A0 不再扩数据。

[D0-A1 已完成](CENTRAL_OBSTRUCTION_AGENT_LABEL_READINESS_D0_A1_RESULT_2026-07-31.md)：
R2 在 4 个 calibration-only source 上冻结 11 clip / 55 observation，并完成 fresh
isolated second pass、8 项 third-Agent adjudication 与最终复算。overall/critical
agreement 为 `0.8545/0.8298`，但 parent-event match `0.6316 < 0.75`，终态
`AGENT_LABEL_PROTOCOL_NOT_RELIABLE`。D0-A2、D0-AT、D0-B 均未运行或获授权；唯一
successor boundary 是在新的 D0-A 版本中用 burned calibration stress cases 重设计
observation/event workflow，不能调 R2 门或覆盖 raw review。

[D0-A successor R0 已完成](CENTRAL_OBSTRUCTION_AGENT_LABEL_READINESS_D0_A_SUCCESSOR_RESULT_2026-07-31.md)：
保留 observation-level Agent 标签，把分析单位改为程序生成的 1 秒 fixed clip，3 个
新 session、6 个 clip、24 个 slot 只做 fresh calibration。固定边界复现率为 `1.0`，
但两路 Agent observation agreement 为 `0.6667`、unresolved 为 `0.3333`，终态为
`CENTRAL_OBSTRUCTION_AUXILIARY_FEATURE_ONLY`。因此不启动 D0-A3/A4，不扩展中央阻塞
主线；D0-A1 的 `0.8545/0.8298` 有效信息保持不变。

D0-A 的 predecessor 只回答中央图像可观察阻塞能否形成稳定的、带风险分层独立审计的
Agent-labeled parent-natural-event 账本；successor R0 改为固定 clip unit，边界由冻结
时间窗和 slot 程序生成，不再从 label 推断 event。标签仍固定为
`VISIBLE_CENTRAL_OBSTRUCTION_PRESENT`、
`NO_VISIBLE_CENTRAL_OBSTRUCTION_EVIDENCE` 与 `NOT_EVALUABLE`；第二项不表示画面无
障碍、可以通行或安全。D0-A 属于 `CANARY_LITE / CAPABILITY_DISCOVERY`。唯一允许的
模型 B 工作是 `D0-AT`：在排除于 readiness 和后续效果评价的数据上，用一个声明的
reference implementation 检查加载、输出 schema、空间分辨率、有限值和有界运行；
它不选择或调优模型/类别/拓扑，不计算事件效果，且输出对标签 Agent 保持关闭。
D0-A 不做融合、调度、Android 或 A568。只有
`READY_FOR_D0_B_AGENT_LABELED_DEVELOPMENT` 才允许另行设计
`DEVELOPMENT_STANDARD` D0-B，且不自动授权 D0-B 执行。

旧神经—几何双环、Q0 语义刷新与 target-local warp residual 的终态均保持不可变，
作为已关闭前序、Development 诊断或旁路线保留；它们不自动进入新主线。

## 用户纠正后的检测 + 语义分割双环主线

本页的 D0-A 终态没有改变，但“下一条主线”不再指向 Q0 semantic-refresh。当前新
双环的概念边界是：环 A 使用 YOLO/QNN 输出已知类别目标，环 B 使用语义分割输出像素级
区域，融合 C 再检验 B 是否为 A 提供可重复的互补信息。D0-A 的中央阻塞 Agent 标签
只用于审计并已关闭，不能作为 B 的主真值或 C 的主要效果指标。

独立的 [segmentation technical smoke R0 result](DUAL_LOOP_SEGMENTATION_TECHNICAL_SMOKE_R0_RESULT_2026-07-31.md)
已完成：接口与有限值通过，但当前 reference 在 24 个 RGB slot 上 argmax 全部塌缩为
`walkable`。该结果只保留为 `TECHNICAL_ONLY / NO_EFFECT_AUTHORITY` 诊断，不代表分割
方向或其他模型候选已被否定。

[Segmentation complementarity Development design R0](DUAL_LOOP_SEGMENTATION_COMPLEMENTARITY_DEVELOPMENT_DESIGN_R0.md)
已冻结并执行为 R1：主量是 `segmentation mask − YOLO box union`
的 image-space uncovered fraction，按 session 聚合并保留时间相关性；它不使用中央阻塞、
risk、feedback 或事件真值，也不把非零区域解释为可通行性或风险。

[Segmentation complementarity R1 result](DUAL_LOOP_SEGMENTATION_COMPLEMENTARITY_R1_RESULT_2026-07-31.md)
已完成用户授权的 4,891-frame Development mechanism diagnostic：配对、finite output 和
独立 validator 均通过；class-wise image-space 增量可观察，但 temporal stability 按 class
混合，且只有一个 burned source。因此 R1 不授权融合、事件效果、Android 或生产结论。

[Segmentation complementarity cross-source R2 result](DUAL_LOOP_SEGMENTATION_COMPLEMENTARITY_R2_CROSS_SOURCE_RESULT_2026-07-31.md)
随后在 Shiraz 与 Shanghai 两个 source 上，用同一 YOLO11n 资产、同一 host 解码合同和
同一 segmentation reference 完成 `4,891 + 5,662` 个配对 frame；两份独立 validator
均为 `VALID`。class-wise image-space signal 在两个 source 上重复出现，但量级随来源和
YOLO coverage 改变，`obstacle`/`boundary_step_curb` 稳定性仍偏低。因此只升级为
`CROSS_SOURCE_IMAGE_SPACE_SIGNAL_REPLICATED / CLASS_STABILITY_MIXED`，仍不授权风险
融合、事件效果、QNN/device parity、Android 或生产结论。

[Segmentation candidate utility R0 result](DUAL_LOOP_SEGMENTATION_CANDIDATE_UTILITY_R0_RESULT.md)
已在 SANPO-Real v0 source-native pixel truth 上完成 calibration 与 120-frame blind
formal。A/B/C 三臂的 pixel 增量和 candidate component recall 达到部分门，但
false activation 为 `13.833/帧`、total incremental host P95 为
`138.444 ms`，分别超过冻结的 `3.0/帧` 与 `30 ms` 门；
独立 validator 为 `VALID`，唯一终态
`CURRENT_SEGMENTATION_REFERENCE_REJECTED`。因此关闭当前 segmentation
reference，不接 Android、QNN、风险事件或主动提醒。

[Segmentation model selection R1](DUAL_LOOP_SEGMENTATION_MODEL_SELECTION_R1_RESULT_2026-07-31.md)
已进入不可恢复终态 `SEGMENTATION_MODEL_SELECTION_R1_BLOCKED /
MODEL_SELECTION_NOT_EVALUABLE`。R1 formal 在首次 source-native mask decode contract
不匹配时 fail closed，正式报告为零行；不得修复、重跑或从已消费 fresh holdout 恢复
model-selection 结论。原 R1 result、failure receipt、closeout validator、formal freeze
receipt 和全部身份收据保持不可变。

R1 四个已消费 session
`GxMb4zhAvoM5jbF54kfcs8wxTL4fqNnT`、
`972O8sd5HpUbGeEE_UAb1g0z1OZUtfHl`、
`ic_BpoiSOIW-7_mffGenT6yissRNiPzT`、
`eHxtA669WpN381O4ZjVAmG3-3ZUewuXr`
现按
[consumed-role amendment](DUAL_LOOP_SEGMENTATION_MODEL_SELECTION_R1_CONSUMED_ROLE_AMENDMENT_2026-08-01.json)
永久降级为 regression/rehearsal/validator/canonicalizer-canary only；复制、重命名、
重新映射、重新打包或 manifest alias 均不能恢复 fresh/unseen 身份。

[R2-P0 readiness result](DUAL_LOOP_SEGMENTATION_R2_P0_RESULT_2026-08-01.md)
已在不选择、下载或读取任何新 fresh mask truth 的条件下完成。独立 canonical view、
synthetic/consumed rehearsal 和逐帧逐阶段 runtime validators 均为 `VALID`；一次冻结的
36 点 DDRNet 后处理搜索没有候选通过全部 readiness margin。DDRNet baseline 的 false
activation 为 `7.885/frame`，SegFormer baseline 的 total incremental P95 为
`74.139 ms`，最接近 DDRNet refinement 仍以 delta FP area `0.072513` 失败。唯一终态为
`R2_NOT_WORTH_BURNING_FRESH_HOLDOUT`，因此 `R2_NOT_AUTHORIZED /
DEVICE_BENCHMARK_NOT_AUTHORIZED / DEFAULT_APP_UNCHANGED`。新增的
[R2 protocol draft](DUAL_LOOP_SEGMENTATION_R2_P0_PROTOCOL_DRAFT_2026-07-31.json)
只定义未来单候选 qualification 问题，状态仍为 `DRAFT_NOT_AUTHORIZED_FOR_FORMAL`。
这里的 device 禁令是 R2-P0 formal 协议的历史边界，不再禁止新
`DEVELOPMENT_STANDARD` 在最终选模前采集明确标注的工程 runtime 证据。

[Segmentation Failure Atlas 与 residual 可标注性 R0 pilot](DUAL_LOOP_SEGMENTATION_FAILURE_ATLAS_AND_RESIDUAL_LABELABILITY_R0_RESULT_2026-08-01.md)
及其
[固定 320-frame 定向扩展](DUAL_LOOP_SEGMENTATION_FAILURE_ATLAS_TARGETED_EXPANSION_R1_RESULT_2026-08-01.md)
现为唯一科学主线。扩展严格复用 6 个既有 dev/consumed session、同一 DDRNet INT8、
同一未过滤 postprocess 与原 9 个非组合 probe。五类机制均跨两种角色复现，pilot 与
expansion 的 aggregate 排序 Spearman 为 `0.90`；residual pixel proxy 仍为
`LABELABLE`，三态归因仍为 `WEAKLY_LABELABLE`。

简单 gating 失败没有完整复现：causal 2-of-3 与 median confidence ≥ `0.65` 达到既有
`PARTIAL` overall 门，但最低 session recall retention 只有 `47.29% / 40.87%`，没有
`SUFFICIENT` gate。当前冻结终态为 `GATING_PARTIAL`，所以不启动 residual-aware DDRNet
训练，也不在同一轮组合或选择 gate。

后继 [conditional gating R0 配置](../../../configs/dual_loop_segmentation_conditional_gating_r0/default.json)
现已在任何新候选 outcome 前冻结为独立 Module：只执行一个
`CLASS_CONDITIONED_MULTI_NEGATIVE`，不做 3 选 1。obstacle 仅在 raw component 低置信且
属于小碎片或与纯几何 upper band 相交时，删除其中缺少同类别 causal 2-of-3 支持的
pixels；boundary/step/curb 只整组件删除低置信小碎片。Atlas 中依赖
`dominant_truth_class` 的 background proxy 明确禁止进入 gate。固定 520-frame、10 个
burned Development source session 只做 fit-free held-out stress，不称 LOSO
cross-validation 或独立确认。

[Conditional gating R0 result](DUAL_LOOP_SEGMENTATION_CONDITIONAL_GATING_R0_RESULT_2026-08-01.md)
现已在冻结 Git `2e46d76057becb1f85c22bf0c9ea4e8b59d26c31` 上完成一次执行并由独立
validator 复算为 `VALID`。候选 overall recall retention 为 `0.942399`，但
false-positive reduction 只有 `0.092572 < 0.30`，最低 source-session recall
retention 为 `0.774580 < 0.80`；它不支配任何 predecessor reference，也不是新的
Pareto improvement。终态为
`CONDITIONAL_GATING_NO_ROBUST_INCREMENT_STOP_GATING_ROUTE`。本轮停止 gating，下一
边界原记为 residual-aware DDRNet Development 设计，尚未授权或执行训练。

用户随后纠正了这个 terminal 的解释范围：一个 primary 失败不能证明全部条件门失败。
R0 machine terminal 与全部 evidence 保持不可变，但 `STOP_GATING_ROUTE` 只表示 R0
不以未执行候选救援 primary。前向
[R0.1 post-primary shadow protocol](DUAL_LOOP_SEGMENTATION_CONDITIONAL_GATING_R0_1_SHADOW_PROTOCOL_2026-08-01.md)
现冻结 `CLASS_CONDITIONAL_TEMPORAL` 与 `MULTI_NEGATIVE` 为 diagnostic-only shadows；
两者在 R0 outcome 前已被概念性提出，但当时未进入 repo hash 或执行授权。R0.1 一次
全量报告、不选优、不改 primary terminal；bounded family terminal 与 residual-aware
训练顺序等 shadow closeout 后再定。

R0.1 初始 implementation activation 在读取 raw shadow input 前因 list/single-binding
loader 类型错误停止；没有 output、mask 或指标。V2 仅修复 input-list 路由并把完整
membership 检查前移到 preflight，科学合同不变。V2 随后完成一次 shadow execution；
初始 validator 又因 primary 摘要 schema 不一致而在 0 项 aggregation checks 后停止。
recovery 只修摘要字段匹配，不修改既有 result/frame/component evidence，最终通过
`167,327` 项检查、错误数 `0`，独立复算的两个核心 JSONL 逐字节一致。

[R0.1 shadow result](DUAL_LOOP_SEGMENTATION_CONDITIONAL_GATING_R0_1_SHADOW_RESULT_2026-08-01.md)
显示：`CLASS_CONDITIONAL_TEMPORAL` 的 FP reduction / overall recall retention 为
`0.284667 / 0.781123`，`MULTI_NEGATIVE` 为 `0.109286 / 0.922445`；两者最低
session retention 仅 `0.612024 / 0.629324`，后者 boundary retention 也只有
`0.612015`。两个 shadow 均无 material，`H_min/H_cross` 均为 false。因此停止这三个
精确定义的静态手工门家族，并把下一主边界转入 FP-aware DDRNet Development；不得扩大
为所有 conditional/learned gating、postprocess 或分割路线失败。

[FP-aware DDRNet R0 result](DUAL_LOOP_SEGMENTATION_FP_AWARE_DDRNET_R0_RESULT_2026-08-01.md)
已完成唯一 successor `FP_WEIGHTED_UNGUIDED_FULL_FRAME`。validator 重新推理六个
checkpoints、逐像素核对 1,920 个 prediction masks，并通过 28,861 项检查、错误数 0。
三个 same-seed pair 均未通过全部九门：FP reduction 为
`.198713 / -.138991 / -.043984`，false components/frame 为
`4.41875 / 7.81875 / 5.61875`。正式终态为
`FP_WEIGHTED_SAMPLING_NOT_SUPPORTED`。这只关闭按 train-only baseline FP pixels
重加权 30% full-frame 抽样的单一候选，不扩大为所有 residual-aware/FP-aware training
失败；不在已消费 outcome 上换 seed、改 crop、loss 或 target 救援。

[Failure-Aware Causal Component Validator R0](DUAL_LOOP_SEGMENTATION_LEARNED_COMPONENT_VALIDATOR_R0_RESULT_2026-08-01.md)
现已完成并由独立 validator 判为 `VALID`。它只用 current/past runtime component 特征、唯一
`StandardScaler + L2 Logistic Regression`、10 个 source-session 的 nested LOSO
cross-fit 与训练上下文内阈值选择，比较 raw、causal 2-of-3、confidence `>=.65`、历史
primary conditional gate 和 learned validator。全部 520 帧/11,757 components 已烧为
Development；每个 scored session 从自身 fold 的 scaler、weight、模型和阈值中排除，
但结果仍只可称 consumed Development robustness，不能称 fresh、unseen、independent
validation 或 Confirmation。协议冻结九项 utility 门、host P95 `<3 ms` 与有界内存门；
失败后不换分类器或 feature subset 救援。实际只通过 4/9 utility 门：FP reduction /
overall / minimum-session / boundary retention 为
`.177920 / .855661 / .466375 / .207740`，`C-A` FP area 为 `.087407`；host P95
`9.376145 ms` 也未过 `<3 ms`。终态
`NOT_SUPPORTED_AND_GATING_STOP`：关闭当前 reference 上的 active learned gating，
不授权 component-aware loss、设备 benchmark、Android 或 Confirmation；visual
sidecar / coverage diagnostic 保留。

独立的 host-only visual sidecar R0 已可用，只显示 YOLO boxes、
raw heatmap、候选、gate pass/reject/abstain 与原因，固定水印且
`drives_alerts=false`；它不获得 Android、QNN/A568、risk/feedback、TTS、振动或默认
App 权限。

Failure Atlas 的批量失败案例图册维护入口见
[`scripts/research/failure_case_atlas/`](../../../scripts/research/failure_case_atlas/)。
运行 `batch_album.py` 可按当前输入批量生成原图、YOLO 框、segmentation/truth mask、
depth、可选 risk heatmap、前后帧、错误类型、置信度、事件阶段、来源信息、分类
contact sheet 与 `index.html`。类别规则由同目录的 `category_rules.json` 管理，
也可用 `--category-config` 注入另一份版本化规则；该图册仍是 Development-only、
host-side visual diagnostic，不改变 alert authority。

## 已关闭前序与保留证据

2026-07-31 的 rank-2 Shiraz 设备评价已完成：baseline/candidate 均命中 `7/7`
正例，5 个 baseline-false 负窗全部保留，反馈行由 `508 -> 494`；终点为
`FIRST_UNSEEN_SOURCE_NO_EVENT_LEVEL_EFFECT / DENSITY_SIGNAL_ONLY`，不是事件级
误提醒改善。随后完成
[R1 事件失败分解](DUAL_LOOP_R1_EVENT_FAILURE_DECOMPOSITION_R0_RESULT_2026-07-31.md)：
只消费三来源已关闭 Development trace、truth ledger 与 receipt，输出逐窗口字段、
retained-false 分类和不写 candidate trace 的 upper-bound audit。该 post-terminal
分析的唯一 top-level terminal 为 `POLICY_GRANULARITY_MISMATCH_SUPPORTED`；它不重写
R1 evidence、不改阈值、不授权或实现 R2。普通生产行为仍不变。

教师可读的阶段收口见
[DUAL_LOOP_STAGE_CLOSURE_R0_2026-07-31](DUAL_LOOP_STAGE_CLOSURE_R0_2026-07-31.md)。
该报告只整理既有证据，不授权新实验或调度实现。

同日将“事件保持型语义刷新调度”作为独立后继路线建立了
[Q0 协议](DUAL_LOOP_SEMANTIC_REFRESH_Q0_PROTOCOL_2026-07-31.json)。它不修复或升级
旧几何双环，而是只在固定模型全频参考下审计“何时需要刷新语义结果”。Q0 R0 使用
独立 arm 状态与 `ZERO_ORDER_HOLD_SEMANTIC_PROPAGATION_R0`，完成 4,422 帧、两 session 的
固定时间 baseline：33/66/100/167/267 ms 分别调用 3,309/2,793/2,430/1,560/1,077
次，Level-3 event-or-feedback divergence 为 122/201/262/404/533 帧，其中纯
feedback decision divergence 为 121/200/261/403/527 帧；当前 8 个正例与 7 个
负窗的 event-window 命中数均未改变。该结果只是 Development-only reference-preservation
筛查，说明 event-window 粒度尚不足以替代逐帧 divergence；没有 learned scheduler、
真实 tracker、能效、Android、产品或安全结论。

随后在不重跑 detector、不读取旧 R2/R3 的前提下完成了
[Q0 R0.1 评测修订](DUAL_LOOP_SEMANTIC_REFRESH_Q0_R0_1_EVALUATION_PROTOCOL_2026-07-31.json)：
将风险 episode 定义为连续 active risk-signature run，补齐独立 active event ID、
episode onset/offset、temporal IoU、风险字段一致率、episode feedback count delta、
最长 stale duration 与 signed feedback delay 的 P50/P90/P95。原始 nondominated set
仍包含所有 6 个 VALID operating points；在预声明的零 reference miss、episode coverage
≥0.95、平均 IoU ≥0.80、onset-delay P95 ≤150 ms、risk-signature match ≥0.95 门下，
admissible set 为 `FULL_RATE_REFERENCE / FIXED_TIME_33MS`，约束型最低调用点为
`FIXED_TIME_33MS`。这只是两 session 的 Development 评测诊断，不代表自适应策略优于固定周期。

rank-1 仍按其自身终点 `FIRST_UNSEEN_SOURCE_NOT_EVALUABLE / VALID` 封存；它说明
truth-first 门在无足够正例时停止，不与 rank-2 的结果混为一谈。

2026-07-30 已完成
[隔离主动纠错 R1](DUAL_LOOP_ACTIVE_CORRECTION_R1_RESULT_2026-07-30.md)。
双环现已从真实 shadow 推进到独立 application id 的
`ACTIVE_CONTRADICT_ONLY`：几何环只在至少两个唯一关联框共同明显缩小时否决当前帧
提醒，其余全部弃权；raw/stable risk 与事件身份/生命周期规则不变，普通 debug、
release 和 shadow 仍默认不干预。该日的 CrowdBot/Matoaka 结果复现了触发行小幅下降
且已命中正例无新增延迟，但两个来源的负例提醒窗口均为 `7 -> 7`；当日 R1 终点是
`CROSS_SOURCE_ROW_SIGNAL_REPLICATED / NO_EVENT_LEVEL_EFFECT`。随后 Shiraz rank-2
与本次 failure decomposition 已将当前状态收口为顶部所列 frozen conclusion，均不构成
误提醒下降结论。

此前已完成
[因果框尺度三态源 R0](DUAL_LOOP_CAUSAL_TRACK_TRISTATE_R0_RESULT_2026-07-30.md)。
主线不再把 ego/target 运动责任归因、精确米制 TTC、pose、IMU、depth 或完整三维
恢复作为基础提醒的前置条件。当前第二环只对 production semantic loop 选中的目标，
根据连续 7 帧 `log(bbox height)` 的严格同号趋势输出
`CONFIRM_APPROACH / CONTRADICT_APPROACH / ABSTAIN`。

固定规则先在 8 个已烧毁 Development 会话上复现，再在任何选中 payload 打开前以
metadata-only hash 冻结 3 个全新 JRDB 会话、每个 360 帧。2D source 在 3D truth
取得前封存。独立 Confirmation 得到 1,017 个非弃权判断、1,008 个正确，总精度
`99.12%`；confirm `377/385 = 97.92%`，contradict
`631/632 = 99.84%`，coverage `2.391%`，三个会话均过预声明门。终点为
`ANNOTATION_TRACK_SOURCE_CONFIRMATION_PASS`，claim ceiling 仅为
`ANNOTATION_TRACK_MECHANISM_CONFIRMATION_ONLY`。

同一数学规则随后已进入 Android `AssistDecisionKernel` 的真实
`DUAL_LOOP_SHADOW` 路径：不再传入恒定 null，而是使用当前 production-selected
detection 与轻量 track continuity 生成三态 evidence，再由 frame/time/target/TTL
绑定的 admitter 准入。逐帧回归确认 risk、event、feedback、session 与 gateway
调用保持 baseline 完全一致。该 shadow 结果仍是主动纠错的工程前序，不被后继改写。

此前的
[真实几何 shadow cycle R0](DUAL_LOOP_REAL_GEOMETRY_SHADOW_CYCLE_R0_RESULT_2026-07-30.md)
仍作为工程前序保留：JRDB LiDAR 回放证明接缝可运行；Depth Anything temporal
derivative 与 homography residual flow 的负结果则解释了为何转向更简单、选择性
更强的框尺度三态源。

旧 Sparse LK F-1B 路线仍以 `NO_INCREMENT / VALID` 关闭；该结论没有被重写。
LITE R2 两臂均不达 readiness floor 后，主线不再优先把失败的 radial-flow 候选
接入 Android。当前最短可证伪路线改为对现有生产 `TemporalRiskTracker` 做
[factorial A/B R0](DUAL_LOOP_PRODUCTION_TEMPORAL_GEOMETRY_FACTORIAL_AB_R0_PROTOCOL_2026-07-30.json)：
A 只中和 object-detector temporal geometry output，B 保持当前完整生产链；每帧
QNN 只推理一次，两臂使用相同 detections 和完全隔离的决策/反馈状态。

[独立设计复核](DUAL_LOOP_PRODUCTION_TEMPORAL_GEOMETRY_FACTORIAL_AB_R0_DESIGN_REVIEW_RESULT_2026-07-30.md)
为 `PASS`。outcome-blind 预检已验证 `4422/4422` 个冻结 RGB；truth-membership
预检将 17 项原始 truth 冻结为 8 个可评分正例 + 7 个负窗，并在候选输出前排除
两个零有效帧正例。A/B factorization、truth-blind device producer、独立
validator/evaluator 已实现；核心回归、合成 mutation tests、Android build 均通过。
真机 prestart 在 `SM-S9280 / SM8650` 上复核 `4422/4422` 帧身份，并以 strict
QNN HTP 完成 synthetic probe，未解码 decision RGB、未写候选输出。hash-bound
implementation lock 与
[独立实现复核](DUAL_LOOP_PRODUCTION_TEMPORAL_GEOMETRY_FACTORIAL_AB_R0_IMPLEMENTATION_REVIEW_RESULT_2026-07-30.md)
均已完成且为 `PASS`。唯一正式 producer、truth-blind validation/seal 与后续
truth evaluator 已完成；[执行结果](DUAL_LOOP_PRODUCTION_TEMPORAL_GEOMETRY_FACTORIAL_AB_R0_EXECUTION_RESULT_2026-07-30.md)
为 `VALID / NO_INCREMENT`。两臂在 8 个可评分正例、7 个负窗和两个 session 上的
实际提醒完全相同；正式 authority 已消费，Confirmation 不授权。

实现审计发现并已修复逐帧 timestamp 未绑定、truth receipt 未硬绑定、缺 post-validator
seal、review 未绑定当前 lock、host 并发启动窗口、producer marker 可覆盖、外部复制
pre-temporal hash 等正式前缺口。现在设备 producer 自身要求 lock + activation
authorization；validator 逐帧对照冻结 ledger 并发布哈希闭合 seal，evaluator 只接受
该 seal。修复后的独立 implementation 复审仍是 activation 的前置条件。

用户将双环设为新的研究主线后，2026-07-30 完成了独立 successor Discovery：
[可归因区域级接近证据源 Discovery R0](DUAL_LOOP_ATTRIBUTABLE_REGIONAL_APPROACH_SOURCE_DISCOVERY_R0_2026-07-30.md)。

LITE R2 负结果之后，用户已授权第一步冻结
[D0 ego-motion error attribution R0](DUAL_LOOP_D0_EGOMOTION_ERROR_ATTRIBUTION_R0_PROTOCOL_2026-07-30.json)。
它只允许使用已经烧毁的 R2/REveL evidence，以 469 个 parent natural event 为
分析单位，描述性检查 person/sensor 径向分量、相机运动、ROI 抖动、事件长度、
flow MAD 与轨迹支持度。早先工程设计审查为
[`PASS / NOT_RUN`](DUAL_LOOP_D0_EGOMOTION_ERROR_ATTRIBUTION_R0_DESIGN_REVIEW_RESULT_2026-07-30.md)，
但后续
[独立统计复核](DUAL_LOOP_D0_EGOMOTION_ERROR_ATTRIBUTION_R0_STATISTICAL_REVIEW_RESULT_2026-07-30.md)
指出单 capture 依赖与“dominant mechanism”不可识别，终点为
`REPAIR_NEEDED / NOT_IMPLEMENTATION_READY / NOT_RUN`。D0 现作为生产 A/B
`NO_INCREMENT` 后的后备诊断，不再是唯一前瞻工作；旧 F-1B decision 继续密封。

本地只读连接复算确认，REveL Dynamic 的 RGB 人框、green/yellow 目标身份与
person/sensor Vicon 径向轨迹能在 LEFT/CENTER/RIGHT 全部区域输出目标可归因的
approaching/quasi-static/receding 开发真值。随后
[LITE R0 设计评审](DUAL_LOOP_TARGET_TRACK_CAUSAL_RADIAL_GEOMETRY_LITE_R0_DESIGN_REVIEW_RESULT_2026-07-30.md)
冻结完整连续 capture、两条最小 arm、输出/TTL/abstention、parent-event 分母与
停止规则并通过独立评审。两臂 producer、post-hash evaluator 和 24 个 synthetic
fixtures 随后完成并通过
[implementation review](DUAL_LOOP_TARGET_TRACK_CAUSAL_RADIAL_GEOMETRY_LITE_R0_IMPLEMENTATION_REVIEW_RESULT_2026-07-30.md)；
一次性 activation 也通过独立复核。但唯一 R0 full producer attempt 在同目标相邻
RGB 尺寸从 `260×346` 变为 `258×346` 时触发 OpenCV LK 前提失败。按冻结规则
[LITE R0 execution result](DUAL_LOOP_TARGET_TRACK_CAUSAL_RADIAL_GEOMETRY_LITE_R0_EXECUTION_RESULT_2026-07-30.md)
为 `EXECUTION_INVALID_STOP_NO_RERUN / NOT_EVALUABLE`，未进入 truth join。

独立 R1 冻结跨尺寸处理后，正式 producer 完成，但共享 host guard 将 JSON 中 UTC `Z`
时间戳误解释为本地时间；R1 因执行包络门失败而同样停止，完整输出不作科学救援。
独立 R2 仅修复该执行包络、绑定新 identity/namespace，并通过设计、实现、pilot、
preflight 与 activation 评审。R2 的唯一 producer 和条件 evaluator 均有效完成。
[LITE R2 execution result](DUAL_LOOP_TARGET_TRACK_CAUSAL_RADIAL_GEOMETRY_LITE_R2_EXECUTION_RESULT_2026-07-30.md)
在冻结的 469 个 primary 自然事件上得到：

- box 面积增长：204/469 正确，153/469 wrong-signed；
- ROI 稀疏径向光流：188/469 正确，161/469 wrong-signed；
- flow 相对 box 的正确事件增量为 `-16`，两个 target 与三个区域增量均为负；
- 两臂均未达到正确率 `>=0.60`、wrong-signed `<=0.20` 的 readiness floor。

```text
PREDECESSOR_F1B: COMPLETE / NO_INCREMENT / VALID
SUCCESSOR_SOURCE_DISCOVERY: COMPLETE / SOURCE_FOUND_FOR_DEVELOPMENT
SUCCESSOR_LITE_DESIGN: DESIGN_REVIEW_PASS / F1_INTERFACE_FROZEN
RUNTIME_GEOMETRY_SOURCE: OFFLINE_IMPLEMENTED / ONE_SHOT_EVALUATED
SUCCESSOR_LITE_DEVELOPMENT: BOTH_NOT_READY_FOR_CONFIRMATION
CONFIRMATION: NOT_AUTHORIZED
EXECUTION_AUTHORITY: CONSUMED / NO_RERUN
CLAIM_CEILING: SINGLE_CAPTURE_ORACLE_ROI_CONDITIONED_DEVELOPMENT_ONLY
```

BlindAssist 的 predecessor 路线已按“先准入、后实现”顺序走完神经—几何双环
阶段−1，并在科学生死门停止：

```text
DATA_STATUS: READY
TIMING_STATUS: READY
SCIENCE_STATUS: NO_INCREMENT
RUNTIME_STATUS: NOT_RUN
ROUTE_CONTRACT_STATUS: MAINLINE_STOPPED
DATA_PROTOCOL_STATUS: VALID
TIMING_PROTOCOL_STATUS: VALID
SCIENCE_PROTOCOL_STATUS: VALID
RUNTIME_PROTOCOL_STATUS: NOT_RUN
EXECUTION_AUTHORITY: NONE
CLAIM_CEILING: DEVELOPMENT_ROUTE_REJECTION_ONLY
```

顺序固定为：

```text
F-1A 数据能否评价
  ↓
F-1B0 当前语义与几何结果何时真实可用
  ↓
F-1B 几何证据是否产生事件级增量
  ↓
F-1C 指定手机是否承载得住
  ↓
才决定是否实现正式双环
```

2026-07-30 的初始
[DUAL_LOOP_DATA_READINESS_R0](DUAL_LOOP_DATA_READINESS_R0_2026-07-30.md)
终点为 `HOLD_DATA`。经用户授权的固定既有 RGB 标签修复 R0 保持一次性终点不变；
独立后继 R1 只补缺失负类，最终达到
[F-1A `READY / VALID`](DUAL_LOOP_F1A_NEGATIVE_CATEGORY_SUPPLEMENT_R1_RESULT_2026-07-30.md)。

随后 [F-1B0 真机时序基线](DUAL_LOOP_F1B0_TIMING_BASELINE_R0_RESULT_2026-07-30.md)
在 `SM-S9280 / SM8650` 上形成生产 QNN 与隔离 Sparse LK 的完整
capture→available→consume 因果账本，终点为 `READY / VALID`。

F-1B 在 decision 候选输出仍为零访问时，对 hash-bound 的现有几何接口与生产提醒
状态机完成结构可达性检查。现有 Sparse LK 五通道只有全局中心走廊残差和质量，没有
目标身份、LEFT/CENTER/RIGHT 区域、接近方向、径向扩张或 TTC。在不伪造这些语义的
最薄融合下：

- 中心 `NEAR/CRITICAL` 在 A 分支已是 `HIGH`，一帧立即确认；
- 唯一有两帧确认延迟的可提醒分支是侧向 `NEAR/MEDIUM`，但全局中心走廊几何不能归因
  到 LEFT/RIGHT，必须 abstain；
- `MID/FAR/NO_CANDIDATE` 不得由残差升级距离、风险或创建提醒；
- 几何不得绕过既有 cooldown、fatigue 或实际交付语义。

因此 B 相对 A 的 `PAIRED_FIRST_DELIVERABLE_ALERT_LEAD` 理论上界为 `0 frame`。
[F-1B 结果](DUAL_LOOP_F1B_STRUCTURAL_REACHABILITY_PROTOCOL_REPAIR_R2_RESULT_2026-07-30.md) 为
`NO_INCREMENT / VALID`，按冻结合同停止论文双环主张，不消费 decision 集，也不进入
F-1C。

详细输入、状态、判定和停止门以
[双环阶段−1准入合同 R0](BLINDASSIST_DUAL_LOOP_PHASE_MINUS1_ADMISSION_CONTRACT_R0_2026-07-30.md)
为准。本轮执行来自用户连续推进授权；合同本身仍不构成未来新实验权限。

## 研究候选

- 语义证据：现有 YOLO11n 与现有生产检测/风险接口；目标设备若已准入，可使用其真实
  QNN 路由。
- predecessor 几何证据：Sparse LK 五通道已经证实不具备目标、区域和接近语义，
  保留为负结果与 regression fixture，不再作为 successor 的主候选。
- successor 几何证据：annotation-track `causal track tri-state` 保留为机制
  Confirmation；主动 R1 使用近期帧至少两个唯一关联检测框的共同缩小作为场景级
  `CONTRADICT`，否则弃权。两者均不使用运行时 truth、pose、IMU、depth 或米制 TTC。
- 汇合位置：同一事件/区域、真实时间戳、质量、时效和失效原因进入既有统一决策接缝；
  两个环不得分别提醒。
- 论文候选贡献：不是“双环”这个框图本身，而是双环相对 YOLO-only 是否产生可重复的
  首次有效提醒提前、风险判别改善或风险连续性改善。

运行时三态 source 已接入非干预 shadow；最小 scene-scale contradiction 已接入隔离
`dualLoopActive` 构建。当前不实现生产自适应调度、深度、分割、ARCore、新风险场、
latch、新状态机或第二套反馈系统；Q0 的 hold/cache/event state 只存在于独立离线
反事实模拟器中。

## 与 RCLE、USTRF 和 Project Guideline 的关系

- [RCLE](../rcle/README.md) 已由用户于 2026-07-30 暂停。既有科学终态、协议终态、
  one-shot 消费状态和未消费的 `480+16` 全部保留；双环不重跑或救援 RCLE。
- Sparse LK 是阶段−1的默认轻量候选。RCLE 若未来恢复，只能通过新的、独立的准入决定
  成为可选证据源，不能自动替换 Sparse LK。
- 已关闭的
  [USTRF route-conditioned program](../ustrf-sc/USTRF_ROUTE_CONDITIONED_PROGRAM_CLOSURE_R1_2026-07-25.md)
  不因“双环”名称重启；不恢复旧 dense risk field、route、lifecycle 或旧数据门。
- [Project Guideline 适配审计](../../PROJECT_GUIDELINE_COMPONENT_ADAPTATION_AUDIT_2026-07-30.md)
  只提供失效语义、时间戳、最小证据账本和可重算原则；仍是
  `REFERENCE_ONLY / NO_IMPLEMENTATION_AUTHORITY`。

## 当前证据边界

本轮最终证据边界如下：

- 正式 CameraX 把 ImageAnalysis 配置为 `640×480 / KEEP_ONLY_LATEST`，Preview 请求
  `24 FPS`；该请求不等于真实 analysis 或结果频率。
- SM-S9280 / SM8650 的现有 QNN HTP 路由已经晋升，完整检测已有同机延迟与十分钟
  持续观测；能耗优势没有证据，SM8550 也没有被该结论覆盖。
- `LatestOnlySidecar` 与 `RgbaLumaSidecar` 已实现单槽替换、拥有的 luma 副本和过期
  结果拒绝，但不包含视觉算法、风险或提醒语义。
- 旧 CPU-era Sparse LK 回放与真机 shadow 结果相互提示：并发形态可能可承载，但真实
  CameraX 组合路径曾超过旧 `70 ms` 门，且没有 matched live YOLO-only 因果对照。
- 当前已有独立 annotation-track 三态方向证据、真实 Android shadow source、设备端
  4,422 + 10,724 + 4,891 帧 active replay 和隔离 APK smoke。CrowdBot、Matoaka 与
  Shiraz 三个 Development 来源只出现行级下降或持平，没有负例提醒窗口减少；普通
  生产行为因此保持不变。

因此 predecessor 只允许写成 `DEVELOPMENT_ROUTE_REJECTED / NO_INCREMENT`；
successor 当前只允许写成 `ANNOTATION_TRACK_SOURCE_CONFIRMATION_PASS /
ISOLATED_ACTIVE_MECHANISM_LANDED / DEFAULT_OFF /
FIRST_UNSEEN_SOURCE_NO_EVENT_LEVEL_EFFECT / DENSITY_SIGNAL_ONLY`。
不得写成 live 算法有效、误提醒事件已下降、默认生产改善或安全结论。

## 当前权限

| 能力 | authority |
| --- | --- |
| D0-A0 适配度排序、最小 admitted bundle 与 reuse-role freeze | `COMPLETE / VALID / CANARY_LITE_ONLY` |
| D0-A1 排除式 calibration、标签 pilot 与 readiness lock | `COMPLETE / VALID / AGENT_LABEL_PROTOCOL_NOT_RELIABLE` |
| D0-A successor R0 fixed-clip calibration | `COMPLETE / VALID / AUXILIARY_FEATURE_ONLY` |
| D0-A2–A4 Agent 标注、裁决与 readiness 审计 | `NOT_AUTHORIZED / STOPPED_BY_D0_A_SUCCESSOR_R0` |
| D0-AT 排除数据 reference model-B 接口 smoke | `NOT_RUN / STOPPED_BY_LABELABILITY_TERMINAL` |
| 独立 segmentation technical smoke R0 | `COMPLETE / VALID / TECHNICAL_ONLY / NO_EFFECT_AUTHORITY` |
| image-space segmentation complementarity R0 design | `COMPLETE / DESIGN_FROZEN / EXECUTED_AS_R1` |
| D0-A 人工标注、人工复核或人工验收队列 | `NOT_REQUIRED / MUST_NOT_BLOCK` |
| D0-B 模型 B、主阻塞算子与 A-vs-C 设计 | `R1_IMAGE_SPACE_DESIGN_EXECUTED / NO_RISK_OPERATOR` |
| D0-B image-space A/B/C mechanism diagnostic R1 | `COMPLETE / DEVELOPMENT_ONLY / IMAGE_SPACE_ONLY / NO_EFFECT_AUTHORITY` |
| DUAL_LOOP_SEGMENTATION_CANDIDATE_UTILITY_R0 | `COMPLETE / VALID / CURRENT_SEGMENTATION_REFERENCE_REJECTED` |
| D0-B 风险融合、事件增量、Android 或生产评价 | `NOT_AUTHORIZED / NOT_EVALUATED` |
| 编写和维护阶段−1准入合同 | `AUTHORIZED` |
| F-1A 数据审计与既有 RGB 标签修复 | `COMPLETED / READY / VALID` |
| F-1B0 双源时间基线补测 | `COMPLETED / READY / VALID` |
| F-1B 几何增量评价 | `COMPLETED / NO_INCREMENT / VALID` |
| F-1B decision 输出执行 | `NOT_RUN / NOT_NEEDED / SEALED` |
| F-1C 手机双环 A/B | `STOPPED_BY_F-1B / NOT_AUTHORIZED` |
| successor 几何真值源 Discovery | `COMPLETED / SOURCE_FOUND_FOR_DEVELOPMENT` |
| successor causal runtime geometry | `ANDROID_SHADOW_IMPLEMENTED / ISOLATED_ACTIVE_IMPLEMENTED` |
| successor selective tri-state Development | `COMPLETE / CROSS_SESSION_REPLICATED` |
| production temporal geometry factorial A/B R0 | `COMPLETE / VALID / NO_INCREMENT / ONE_SHOT_CONSUMED` |
| D0 ego-motion error attribution | `R0_REPAIR_NEEDED / R1-R3_EXECUTION_INVALID / CONSUMED / NO_RERUN / NO_R4` |
| successor 独立 Confirmation | `COMPLETE / ANNOTATION_TRACK_SOURCE_CONFIRMATION_PASS` |
| 一次有限修复 | `NOT_APPLICABLE_TO_MISSING_INFORMATION_SEMANTICS` |
| 第二环输入合同、准入门与生产影子接线 | `IMPLEMENTED / DEFAULT_OFF / SHADOW_NON_ACTUATING` |
| 隔离 active contradiction-only 构建 | `IMPLEMENTED / DEVELOPMENT_ONLY / NO_EVENT_EFFECT_CLAIM` |
| R1 event failure decomposition | `COMPLETE / POLICY_GRANULARITY_MISMATCH_SUPPORTED / DEVELOPMENT_ONLY` |
| 事件保持型语义刷新调度 Q0 离线 R0 | `COMPLETE / DEVELOPMENT_ONLY / ZERO_ORDER_HOLD / NO_ANDROID_AUTHORITY` |
| 事件保持型语义刷新调度 Q0 离线 R0.1 评测修订 | `COMPLETE / DEVELOPMENT_ONLY / EPISODE_ALIGNMENT / NO_ANDROID_AUTHORITY` |
| scene-scale active successor / single-variable R2 | `CLOSED / NOT_WORTH_DESIGNING / NOT_IMPLEMENTED` |
| 默认生产 active/actuating 行为变更 | `NOT_AUTHORIZED` |
| 自适应调度、深度、ARCore | `NOT_AUTHORIZED` |
| RISKSEG-R0 任务/数据/顺序执行合同 | `FROZEN / FULL_SEQUENTIAL_EXECUTION_AUTHORIZED` |
| RISKSEG-R0 新 event-eval | `FROZEN_ADEQUATE / 30_EVENTS / 30_SESSIONS / BUCKETS_8_8_7_7` |
| PIDNet-S TFLite/QNN/SM-S9280 技术预检 | `COMPLETE / PASS / 163_OF_163 / P95_75.739_MS` |
| PIDNet-S 三 seed 训练与三臂事件评价 | `COMPLETE / VALID / EVENT_QUALITY_OR_STABILITY_FAIL / 0_OF_3_PASS` |
| PIDNet-S trained-final SM-S9280 性能门 | `PASS / 173_OF_173 / P95_77.374_MS / 1.07624X / THERMAL_0` |
| RISKSEG-R0 默认 App 替换 | `NOT_RUN / NOT_PROMOTABLE / KEEP_YOLO` |
| RISKSEG-R1 P0 soft dense adapter audit | `COMPLETE / VALID_NEGATIVE / TRUTH_MASK_SOFT_ADAPTER_FAIL_CHANGE_ACTIONABILITY_LABELS / P1_NOT_AUTHORIZED` |
| DUAL_LOOP_SEGMENTATION_CONDITIONAL_GATING_R0 | `PRIMARY COMPLETE / VALID / NOT_SUPPORTED / HISTORICAL TERMINAL IMMUTABLE / DEVELOPMENT_ONLY` |
| DUAL_LOOP_SEGMENTATION_CONDITIONAL_GATING_R0.1 SHADOW | `COMPLETE / VALID / NO_MATERIAL / NO_HETEROGENEITY / BOUNDED_STATIC_HANDCRAFTED_GATING_FAMILY_STOP / POST_PRIMARY_DIAGNOSTIC_ONLY` |
| DUAL_LOOP_SEGMENTATION_FP_AWARE_DDRNET_R0 | `COMPLETE / VALID / FP_WEIGHTED_SAMPLING_NOT_SUPPORTED / SINGLE_SUCCESSOR_STOP / THREE_PAIRED_SEEDS / DEVELOPMENT_ONLY` |
| 默认模型、提醒、反馈或产品行为变更 | `NOT_AUTHORIZED` |
| 真人、独立助行、安全、产品或跨设备结论 | `NOT_AUTHORIZED` |

## 下一步

D0-A successor R0 已经回答了本轮唯一允许的问题：固定 clip 转换函数能够把事件边界
变成确定性、可单测的程序规则，但 fresh observation semantics 仍未过门。当前动作
固定为：

1. 保留 D0-A1 的 `0.8545/0.8298` observation evidence 与全部 raw review，不覆盖、不
   重跑、不重切 burned 11 clips。
2. 中央阻塞只保留为 `AUXILIARY_FEATURE_ONLY` 的 observation-level、非生产候选；不将
   `MIXED_OBSERVATION` 或 `NOT_EVALUABLE` 重标为事件。
3. 停止 D0-A3/A4，不启动 D0-A2、中央阻塞标签续作、默认生产或安全工作；独立
   technical smoke R0 与 R1 image-space diagnostic 均不产生风险/Android 权限。
4. 如未来重新提出中央阻塞研究，必须另立独立问题、证据版本和评审，不得把它称为本
   successor 的自动延续。

### 新双环的下一条短链

该短链已全部执行并到达负终态：

1. 30 parent events / 30 source sessions 的 session-disjoint event-eval 已冻结；
2. PIDNet-S `512x288 / W8A8 / four-class` 技术预检通过；
3. 官方结构/预训练权重的三个固定 seed 已训练并导出，不含 gate、FP sampler 或组件分类器；
4. `YOLO-only / learned segmentation-only / truth-mask oracle` 三臂评价已完成并独立复算；
5. trained-final SM-S9280 10 分钟性能门通过，但事件质量和跨 seed 稳定性失败，因此
   `RISKSEG_R0_TRAINED_NOT_PROMOTABLE_KEEP_YOLO`。

RISKSEG-R0 不再有自动下一阶段。禁止用已消费 event-eval 调参、挑 seed、改 taxonomy、
增加规则或重开默认 App 集成。若未来提出 successor，必须具有新的因果假设、独立冻结
合同和新的 session-disjoint parent-event cohort。

显式授权并独立冻结的 RISKSEG-R1 P0 已作为该 successor 的机制诊断执行，但它没有把
已消费 cohort 重新称为确认集。truth-mask soft adapter 的误报与清除门失败，learned
seed 稳定性为 `0/3`，所以当前四类目标的 P1 训练不获准。新的 successor 必须先重建
actionability/event supervision；不能继续扩 adapter grid、增加规则或换更大分割模型。

conditional gating R0 已完成 520 帧执行、逐帧/逐组件独立复算与 held-out/direct
等价检查；五项门中的 false-positive reduction 和 minimum-session recall retention
失败，只能判定 primary 不受支持。R0.1 又以前向冻结、一次全量、全部公开且无选择权限
的方式执行两个 diagnostic-only shadows；两臂均无 material signal，且没有冻结定义的
minimum-session-only 或跨 session winner inversion。由此关闭的是这个精确三臂、
固定阈值、静态手工门家族。后继 FP-aware DDRNet 也已经到达负终态；它们现在只作为
历史 comparator，不再是下一主边界。RISKSEG-R0 不增加手工阈值、latch、类别规则或
oracle session routing。

该训练边界现已到达有效负终态。三个 seed 都没有通过冻结的 relative 五门与 absolute
四门；其中两个 seed 的 FP 反而增加，三个 seed 的 absolute FP-area 和 false-component
门全部失败。因此停止这个单一 FP-weighted sampler，不挑最好 seed，也不在相同
consumed outcome 上改成 crop、加 loss 或调 weight。若未来继续训练研究，必须另立具有
不同因果变量和明确数据角色的新 Development 协议；该后继现已由 RISKSEG-R0 独立合同
给出新的四类任务、数据角色、INT8/runtime/device 路径与顺序授权，不恢复旧 DDRNet
candidate identity。

### 后续资源纪律

对中央阻塞路线之后的新研究，冻结以下三条规则：

1. 每条失败路线最多允许一个 successor；successor 必须回答预先声明的关键不确定性，
   不得把同一失败拆成连续 readiness 子阶段；
2. fresh 双路结果若显示语义构念不稳定，不再引入第三 Agent 将分歧裁决成表面一致，
   也不再通过改 prompt、增加 slot 或寻找更容易的数据挽救标签；
3. 下一阶段必须直接产出算法对照指标、端侧性能数据或明确关闭具体路线；不得再产生
   只有 readiness 名称变化而没有新决策信息的多层阶段。

### 已关闭几何与调度路线的保留规则

LITE R0/R1/R2、production temporal geometry factorial A/B R0 与 D0 R1/R2/R3
保持各自已消费的关闭终态，不重跑、不调阈值救援。隔离 active R1 已完成
[事件失败分解](DUAL_LOOP_R1_EVENT_FAILURE_DECOMPOSITION_R0_RESULT_2026-07-31.md)：
row-density 仍可作为 Development diagnostic，但 25 个负窗中没有一个被完整消除。

upper-bound audit 只在内存中使用已记录的 R1 candidate opportunities：CrowdBot 与
Matoaka 各有一个满足既有正例命中、零 induced negative window 和预冻结新增时延上限的
有限 hold witness；它们都需要新的 runtime state，Shiraz 在 `250 ms` 上限内没有
witness。因此该 terminal 只支持解释 policy granularity，不授权任何 R2。

当前推荐关闭 scene-scale active 路线；不实现该路线的 hold、latch、事件状态、阈值
调整或单变量 R2。Q0 的 zero-order-hold/cache/event state 仅用于独立离线模拟，且 feature-rule
与 Logistic arms 因没有独立 current-frame-only fast-feature trace 而为
`NOT_EVALUABLE`。保留默认关闭的机制、receipt、回归夹具、逐窗口分解和失败分类；
默认、debug、release、产品、真人助行与安全行为均不改变。

Q0 R0.1 评测修订现已封存；若继续推进，只允许先冻结独立的 Q1A fast-feature trace
合同（当前 RGB/IMU、该 arm 自己的历史状态、session 身份和 feature provenance），再在
相同调用预算下比较 `FIXED_TIME_66MS / 100MS / 167MS` 与 `MAX_AGE + TRACK_FAILURE +
SCENE_CHANGE` 规则。没有跨 session 的独立 fast-feature evidence 前，不训练 Logistic；
没有离线规则增量前，不进入 Snapdragon、Android shadow、能效或热量测量。

### D0 R1/R2/R3 执行终态

[D0 R1 执行](DUAL_LOOP_D0_EGOMOTION_ERROR_ATTRIBUTION_R1_EXECUTION_RESULT_2026-07-30.md)
因冻结解释器缺少 `rosbags`，在读取任何 bag message 前以
`EXECUTION_INVALID / CONSUMED / NO_SCIENTIFIC_EXIT` 关闭。

[D0 R2](DUAL_LOOP_D0_EGOMOTION_ERROR_ATTRIBUTION_R2_PROTOCOL_2026-07-30.json)
随后只修复运行时包络，并通过设计复核、40 项测试、实现锁、独立实现复核与
activation。正式运行通过首条 Vicon message probe 后，在冻结 calibration parser
动态导入 `yaml` 时发现环境只有 `ruamel.yaml`。因此
[R2 执行结果](DUAL_LOOP_D0_EGOMOTION_ERROR_ATTRIBUTION_R2_EXECUTION_RESULT_2026-07-30.md)
同样为 `EXECUTION_INVALID / CONSUMED / NO_RERUN / NO_SCIENTIFIC_EXIT`：
`0 / 469` event 完成、没有 event table、没有 D0 指标、没有科学出口。

R2 不能补包重跑。R3 以新的独立环境与 namespace 补齐 PyYAML 后，唯一正式运行
越过 runtime 问题，却暴露了冻结 D0 合同与上游 BBOX 字段之间的二倍语义冲突。
该 authority 已消费，协议与失败回执保持不可变；不生成 R4，不修补或重跑 R3。

## 2026-07-31：未见自然来源 rank-2 结果与 R1 收口

上海 rank-1 因 0 个正例在 baseline 前以 `FIRST_UNSEEN_SOURCE_NOT_EVALUABLE`
关闭后，按预冻结顺序启动 Shiraz rank-2。两路输出盲 RGB review 与第三路分歧裁决
冻结 7 个正例、6 个负窗，状态为 `TRUTH_FROZEN_ADEQUATE`。固定 4,891 帧 10 Hz
输入的 baseline-only adequacy 已通过，candidate 随后按一次性授权完成同一输入回放。

设备入口已拆为物理独立的 baseline-only 与 candidate-only。host 必须先确认
baseline 至少命中一个正例且误触发一个负窗，才生成绑定 input/truth/baseline SHA
的 candidate authorization；candidate 只重放同一 detections/metrics 并逐帧校验
raw/stable risk 不变。详见
[rank-2 protocol](DUAL_LOOP_R1_UNSEEN_NATURAL_EVENT_R0_RANK2_PROTOCOL_2026-07-31.json)
和
[rank-2 truth result](DUAL_LOOP_R1_UNSEEN_NATURAL_EVENT_R0_RANK2_TRUTH_RESULT_2026-07-31.md)。

“已使用”从此只限制同一候选的 unseen/independent claim，不对数据集作全局封存。
旧数据仍可作为 Development、回归或新问题来源；缺原生提醒标签的数据可在算法输出
打开前由隔离多模型复核补齐，但不能虚增独立 session。

rank-2 设备评价随后完成：baseline 和 candidate 都命中 7/7 正例，timely retention
为 7/7；但 5 个 baseline-false 负窗全部保留，`corrected=0`。全序列反馈行只从
508 降到 494。当前终点为
[`FIRST_UNSEEN_SOURCE_NO_EVENT_LEVEL_EFFECT / DENSITY_SIGNAL_ONLY`](DUAL_LOOP_R1_UNSEEN_NATURAL_EVENT_R0_RANK2_EFFECT_RESULT_2026-07-31.md)。
这关闭 active R1 的事件效果主张；默认生产保持关闭，shadow、机制结果、隔离 Android
工程和
[post-terminal failure decomposition](DUAL_LOOP_R1_EVENT_FAILURE_DECOMPOSITION_R0_RESULT_2026-07-31.md)
保留。后续不自动实现或设计 scene-scale active R2。

## 独立 successor 提案（proposal-only）

[TARGET_LOCAL_BACKGROUND_WARP_RESIDUAL_R0 设计合同](TARGET_LOCAL_BACKGROUND_WARP_RESIDUAL_R0_DESIGN_CONTRACT.md)
是一个不继承 R1 或 D0 权限的独立目标局部背景 warp residual 假设；经
[独立设计复核](TARGET_LOCAL_BACKGROUND_WARP_RESIDUAL_R0_DESIGN_REVIEW_RESULT_2026-07-31.md)
修订后，用户已明确授权 B Development；[B 实现复核](TARGET_LOCAL_BACKGROUND_WARP_RESIDUAL_R0_IMPLEMENTATION_REVIEW_RESULT_2026-07-31.md)
现记录为 `B_TERMINAL_CLOSED / NO_DEVELOPMENT_INCREMENT / C1_C2_NOT_AUTHORIZED`。
已冻结的 burned REveL 输入完成一次 truth-blind producer 和 truth-late evaluator，R1
唯一选择未通过 Development gate；因此不重跑、不调参、不自动进入 C1/C2、Android 或产品行为。
