# BlindAssist Assistive Geometry

状态：`current / R2_F0_SYNTHETIC_REDUCER_PASS / F1_SUPERVISION_FRONTDOOR_SATISFIED / AG_R2_SUPERTEACHER_TO_AG_FINAL_V2_SEAM_PASS / FACTORWISE_ORACLE_HEADROOM_PASS / CORRECTION_GAIN_LOPO_FAIL_STOP / ANGULAR_BOUNDARY_FAIL_CLOSED_SAFE_BUT_TASK_INERT / SUPPORT_VALIDITY_FAIL_OPEN / OBSTACLE_RGB_INTERACTION_FAIL_STOP / POSE_ANALYTIC_FAIL_STOP / QPLANE_O0A_REPRESENTATION_HEADROOM_FAIL_CLOSE_NO_TRAINING / CURRENT_OBSTACLE_TASK_ROUTE_CLOSED / SCIENTIFIC_NOT_RUN / CONFIRMATION_OUTCOMES_UNOPENED / DEFAULT_APP_UNCHANGED`

本页是路线日常操作真源。较早完整叙事保存在
[14d8ad7e 历史快照](archive/README_FULL_HISTORY_2026-08-13.md)，不能从中恢复旧权限。

## 当前主张

可替换轻量视觉 encoder 学习 metric-ish depth、support surface 与 obstacle boundary 连续因子；不同
factor 只在各自证据成立时选择性组合，再由确定性 body-swept reducer 构造 Clearance、Occupancy 与
`UNKNOWN`。DepthART-S 是 frozen depth prior/initialization，不是算法终点。

## 当前结论

- R2 F0 reducer、F1 supervision frontdoor 与 SuperTeacher → AG final V2 seam 已落地，但不证明真实
  跨传感器精度。
- consumed Development 已出现可复现的 angular-boundary 正信号；相反，ARKit-trained depth correction
  在 Bonn 有明显 negative transfer。下一步必须按 factor 分路，并让 correction 可 abstain/fallback，不能
  继续追求一个无条件 unified checkpoint。
- ETH3D calibration-control R0/R1 均 fail-closed consumed；R1 independent replay 只证明失败可复现，
  不把它改写为科学 PASS/FAIL。Confirmation outcomes 仍未打开。
- factor-wise oracle 在 11 个 consumed calibration parent 上确认安全 correction headroom；但 learned
  selector 的 Bonn/TUM 外部结果失败。新增的 pixel candidate + neural/kNN frame veto 在两组 consumed
  TUM 诊断上均做到零受伤，却都只覆盖 `1/3` parent，不能以 safe fallback 冒充机制成功。
- 在 14 个 consumed TUM parent 上新增 flip equivariance、temporal reprojection 与二者合取后，三种
  leave-one-parent-out 候选仍全部为 `0` coverage / `0` nonzero parent；当前 correction expert/router
  按停止条件关闭，fresh3 未打开。
- R21 angular boundary 在冻结 ICL source-exact depth/support/obstacle 的任务干预中，把相对 source-reference
  的 cell 一致数由 R20 的 `54/108` 提高到 `72/108`，但 naive probability conjunction 仍有 `3` 个
  false-clear 和 `30` 个 reference-UNKNOWN 越权确定状态。改为“边界缺失不否定 obstacle、只增大定位
  sigma”后两者均降为 `0`，但 R20/R21 都变成 `108/108` 相同，未形成严格任务增益。boundary 组件保留，
  boundary-to-task mapping 停止；默认 App、Android/HTP、产品和安全权限不变。
- 随后的 positive-factor substitution audit 固定 source-exact depth/boundary：learned support 保住全部
  `33/33` reference-known cells，却把 `18` 个 reference-UNKNOWN 升成 definite（其中 `12` 个越权 CLEAR）；
  learned obstacle 无危险放行，但把 `23/33` reference-known 退成 UNKNOWN、`OCCUPIED=0`。二者合取与
  obstacle-only 完全相同，因此当前主瓶颈是 obstacle positive coverage，同时 support 必须降为 veto-only，
  不得自行创建 validity。
- 冻结 obstacle logit 的六父样本 leave-one-parent-out 三态校准随后 `0/6` fold 获得完整阈值对：低分尾部
  无法在逐 parent `FNR<=1%` 下形成 verified-negative，高分尾部也只有 `1/6` fold 能在训练父样本上同时
  满足 `FPR<=5%` 与覆盖要求。当前标量 obstacle 排序不能进入 reducer；下一步只检验一次小型因子交互
  selective head，且仍须嵌套 parent-disjoint 校准。
- 小型 obstacle/support/boundary/depth interaction head 随后用嵌套 parent holdout 执行：每个 inner
  calibration parent 都由排除自身的四父样本模型预测，每个 outer parent 再由其余五父样本模型一次评分；
  结果仍为 `0/6` 完整阈值对。当前 R21 RGB factor observable 家族不再授权新 selector。唯一剩余候选
  改用手机运行时可获得的 pose/gravity，把 frozen metric depth 接回解析高度几何。
- 最后的 depth+pose analytic canary 在 `3/6` parent 恢复了完整合理的跨帧 support-height geometry，说明
  pose/gravity 确实增加了可观测性；但六个 leave-one-parent-out fold 仍全部没有安全双阈值，且最保守的
  low=`0.01`、high=`0.99` 在每个训练 roster 都违反逐 parent error cap。当前瓶颈落到 predicted metric-depth
  scale/plane fidelity，现有 obstacle-to-task 路线关闭，不再授权 threshold/selector 变体。
- 作为 materially different representation 的唯一重开，Q-Plane O0-A 在 120 帧、4 parent consumed
  cohort 上比较 A0–A5 与四类负控。query-local ray-plane 的 parent-macro clearance MAE `0.17284 m`，
  差于 global scale `0.14590 m`、global affine `0.15373 m` 和 global ray-plane `0.15405 m`；gap closure
  `-12.20%`，仅 `2/4` parent 的 false-block、`1/3` horizon 改善，且逐 parent false-clear 门失败。
  Q-Plane 因而关闭，不授权 O0-B 或训练。

## 当前证据入口

- [R2 factorized hypothesis](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_FACTORIZED_GEOMETRY_HYPOTHESIS_PROTOCOL_2026-08-09.md)
- [F1 source-native supervision result](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_SOURCE_NATIVE_LABEL_MATERIALIZATION_AND_FRONTDOOR_RESULT_2026-08-11.json)
- [SuperTeacher → AG final V2 result](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_SUPERTEACHER_TO_AG_LANDING_RESULT_2026-08-12.json)
- [Calibration-control R0 terminal](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_CONFIRMATION_CALIBRATION_CONTROL_PREFLIGHT_ONE_SHOT_RESULT_2026-08-13.json)
- [R1 terminal](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_CONFIRMATION_CALIBRATION_CONTROL_R1_ONE_SHOT_RESULT_2026-08-13.json)
- [Factor-wise no-regret + runtime observability result](BLINDASSIST_AG_FACTORWISE_NO_REGRET_AND_RUNTIME_OBSERVABILITY_RESULT_2026-08-13.json)
- [Runtime observability + boundary task-route result](BLINDASSIST_AG_RUNTIME_AND_BOUNDARY_TASK_ROUTE_RESULT_2026-08-13.json)
- [Positive obstacle/support task-effect audit](BLINDASSIST_AG_POSITIVE_OBSTACLE_SUPPORT_TASK_EFFECT_AUDIT_RESULT_2026-08-13.json)
- [Obstacle evidence tri-state calibration terminal](BLINDASSIST_AG_OBSTACLE_EVIDENCE_TRISTATE_CALIBRATION_RESULT_2026-08-13.json)
- [Obstacle selective-interaction terminal](BLINDASSIST_AG_OBSTACLE_SELECTIVE_INTERACTION_HEAD_RESULT_2026-08-13.json)
- [Depth+pose analytic obstacle terminal](BLINDASSIST_AG_DEPTH_POSE_ANALYTIC_OBSTACLE_RESULT_2026-08-13.json)
- [Q-Plane O0-A representation-headroom terminal](BLINDASSIST_BA_CLEAR_QPLANE_O0A_REPRESENTATION_HEADROOM_RESULT_2026-08-14.md)
- [Unopened fresh-TUM source lock](BLINDASSIST_AG_FACTORWISE_NO_REGRET_FRESH_TUM_SOURCE_LOCK_R0_2026-08-13.json)
- [算法路线总表](../ALGORITHM_RESEARCH_CURRENT.md) · [研究脚本 Module](../../../scripts/research/assistive_geometry/README.md)

## 唯一 successor

无 active successor。只有获得真正新增的 source-native obstacle supervision，或提出不属于
global/query-local inverse-depth ray-plane residual 家族的 materially different representation，才可另立
新协议并重新分配 consumed/fresh role；不得以新名字重调本轮失败候选或 Q-Plane。

## 当前允许

- 只读复核本轮 obstacle scalar、nested interaction 与 depth+pose analytic 三个冻结终态；
- 只读复核 Q-Plane O0-A 的 frozen candidate plan、结果、负控与隔离回执；
- 维护 fresh3 identity/source lock，但在候选全回退时不读取其 RGB/depth/model outcome；
- 只读复核已签署 evidence、失败链和 frozen negative terminals；
- 重放不访问真实 archive payload 的 focused tests。

## 当前禁止

- 重跑、resume、替换、覆盖或重新解释 R0/R1，或把 null 当 0/negative；
- 读取 ETH3D session/Confirmation、在已看 EVAL 调 threshold，或用 macro gain 掩盖任一 parent 退化；
- 用相同 observable 继续堆 correction selector/ensemble，或打开 fresh3 拯救全 fallback；
- 继续尝试 boundary probability/sigma 映射，或把 fail-closed 后 R20/R21 task-identical 写成 task gain；
- 让 learned support 创建 validity，把低 obstacle probability 直接解释成 clear/negative，继续调当前
  scalar obstacle 阈值，或在同一 R21 RGB observable 上增加 selector；
- 重跑/调参 depth+pose analytic canary，或在没有新增 supervision/representation 时创建 obstacle successor；
- 结果后重调/重跑 Q-Plane 的 horizon weight、epsilon、ridge、query mask、负控或 gate，创建 O0-B 或
  learned Q-Plane head；
- 让 depth selector 改写 boundary/support/`UNKNOWN`，或把全 fallback 写成机制成功；
- 把 Development 写成 Confirmation、默认 App、产品或安全证明。

## Claim ceiling

当前只证明 factor/reducer/supervision seam、组件级 angular-boundary 信号、factor-wise correction oracle
headroom、当前 correction-gain observable 不足，以及 boundary 到 task 的 naive 语义不安全、fail-closed
语义安全但无严格增益；positive-factor audit 进一步定位了 support fail-open 与 obstacle undercoverage，
六父样本三态校准排除了当前 scalar obstacle logit 的安全双侧解释，嵌套交互头又排除了同一 RGB factor
observable 家族上的浅层组合；depth+pose analytic 进一步表明 pose/gravity 能恢复一半父样本的几何，
但不能修复预测深度与 obstacle source factor 的跨父可靠性；Q-Plane O0-A 又排除了当前 query-local
inverse-depth ray-plane oracle representation 的整体 headroom，当前 obstacle-to-task landing route 已关闭。
它不证明 learned correction selector、boundary task landing、跨传感器统一泛化、
完整 task superiority、部署可行性或助行安全。
