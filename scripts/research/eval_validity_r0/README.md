# eval_validity_r0

状态：`PRE_OUTPUT_LOCKED / OUTPUT_BLIND_SOURCE_SCREENING_FROZEN / NATIVE_MATERIALIZATION_IN_PROGRESS`

## 研究问题与版本

`EVAL_VALIDITY_R0` 是无训练的评价有效性审计：在严格更丰富的 oracle 输入下，当前
scene-fact → event-fact → feedback 评价链是否保持单调，因而可以公平比较未来模型？允许的
claim 仅为 Development evaluator-construct evidence；不产生模型优劣、独立泛化、产品、
Android、默认 App 或安全结论。完整合同是
[EVAL_VALIDITY_R0_CONTRACT_2026-08-02.md](../../../docs/research/dual-loop/EVAL_VALIDITY_R0_CONTRACT_2026-08-02.md)。

## 稳定 Interface

先建立不能重叠的 source-session 注册表：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m scripts.research.eval_validity_r0.build_exclusion_registry `
  --riskseg-ledger docs/research/dual-loop/RISKSEG_R0_DATA_ROLE_LEDGER_2026-08-01.json `
  --consumed-truth-ledger artifacts.local/evidence/riskseg-r0/event-eval/frozen-cohort-v1/truth_ledger.jsonl `
  --output artifacts.local/evidence/eval-validity-r0/exclusion-registry-v1.json
```

将任何 source-mask-only SANPO discovery 与该注册表相交，得到可继续作 RGB 盲审的候选（仍不是
event truth）：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m scripts.research.eval_validity_r0.select_unseen_discovery `
  --discovery artifacts.local/evidence/eval-validity-r0/sanpo-train-sparse-discovery-v2.json `
  --discovery artifacts.local/evidence/eval-validity-r0/sanpo-train-sparse-discovery-v3.json `
  --exclusion-registry artifacts.local/evidence/eval-validity-r0/exclusion-registry-v1.json `
  --output artifacts.local/evidence/eval-validity-r0/unseen-discovery-selection-v1.json
```

严格 normal-walkable 的 source-mask shortlist 需要独立扫描；它同样不产生负例真值：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m scripts.research.eval_validity_r0.discover_normal_candidates `
  --exclusion-registry artifacts.local/evidence/eval-validity-r0/exclusion-registry-v1.json `
  --max-sessions 96 `
  --output artifacts.local/evidence/eval-validity-r0/sanpo-train-normal-discovery-v1.json
```

冻结 48 个 source-session 的 output-blind screening universe 后，先以 GCS 元数据确认每个
窗口有 60 个连续的 native RGB/mask frame。这个步骤不能将 source-mask profile 写成
event bucket；它只修正缺帧/越界窗口，并保留原 source-mask reference frame。

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m scripts.research.eval_validity_r0.freeze_screening_cohort `
  --hazard-selection artifacts.local/evidence/eval-validity-r0/unseen-discovery-selection-v3.json `
  --normal-discovery artifacts.local/evidence/eval-validity-r0/sanpo-train-normal-discovery-v1.json `
  --exclusion-registry artifacts.local/evidence/eval-validity-r0/exclusion-registry-v1.json `
  --output artifacts.local/evidence/eval-validity-r0/screening-cohort-v2.json

E:\codex-tools\bin\blindassist-python.cmd -m scripts.research.eval_validity_r0.reconcile_screening_windows `
  --screening-cohort artifacts.local/evidence/eval-validity-r0/screening-cohort-v2.json `
  --output artifacts.local/evidence/eval-validity-r0/screening-cohort-v4.json

E:\codex-tools\bin\blindassist-python.cmd -m scripts.research.eval_validity_r0.materialize_screening_inputs `
  --mode plan `
  --screening-cohort artifacts.local/evidence/eval-validity-r0/screening-cohort-v4.json `
  --output artifacts.local/evidence/eval-validity-r0/continuous-native-input-plan-v2.json

E:\codex-tools\bin\blindassist-python.cmd -m scripts.research.eval_validity_r0.materialize_screening_inputs `
  --mode fetch `
  --screening-cohort artifacts.local/evidence/eval-validity-r0/screening-cohort-v4.json `
  --plan artifacts.local/evidence/eval-validity-r0/continuous-native-input-plan-v2.json `
  --output artifacts.local/evidence/eval-validity-r0/continuous-native-inputs-v1
```

materialization 完成后必须先做全量准入审计。它核验原生 payload hash、旧 truth ledger、
41 个排除 session、decoded RGB hash 和 pHash/crop/mirror 候选；清空候选时全量检查，发现
候选后仅保留至多 200 条证据并报告计数下界，仍立即输出 `HOLD_EVAL_VALIDITY_DATA`。

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m scripts.research.eval_validity_r0.audit_data_admission `
  --screening-cohort artifacts.local/evidence/eval-validity-r0/screening-cohort-v4.json `
  --materialized-root artifacts.local/evidence/eval-validity-r0/continuous-native-inputs-v1 `
  --exclusion-registry artifacts.local/evidence/eval-validity-r0/exclusion-registry-v1.json `
  --old-truth-ledger artifacts.local/evidence/riskseg-r0/event-eval/frozen-cohort-v1/truth_ledger.jsonl `
  --prior-image-feature-cache artifacts.local/evidence/data-contamination-audit-r0/image_features_cache.jsonl `
  --output artifacts.local/evidence/eval-validity-r0/data-admission-v1.json
```

若 HOLD 的唯一原因是**完整枚举的** pHash 候选（其余 exact/session/decoded/mask 检查均为
PASS），先用下面的 RGB-only 双人独立包判定每个固定候选是同一自然采集、不同自然采集或未知。
packet 不显示 session/event 身份、pHash 阈值、mask、action fact 或任何模型/oracle 输出；private
map 不得交给 reviewer。两位 reviewer 必须对全部 opaque pair 独立提交，且每一个 pair 都必须
exact agreement 为 `DISTINCT_CAPTURE`。`SAME_CAPTURE`、`UNKNOWN`、不一致、缺项或候选枚举不完整
都维持 `HOLD_EVAL_VALIDITY_DATA`；绝不修改候选、样本、阈值或旧 30-event 队列。

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m scripts.research.eval_validity_r0.prepare_phash_manual_review `
  --admission-receipt artifacts.local/evidence/eval-validity-r0/data-admission-v1.json `
  --materialized-root artifacts.local/evidence/eval-validity-r0/continuous-native-inputs-v1 `
  --workspace-root E:\linnan\linnan `
  --prior-image-feature-cache artifacts.local/evidence/data-contamination-audit-r0/image_features_cache.jsonl `
  --reviewer-a-root artifacts.local/evidence/eval-validity-r0/phash-reviewer-a-v2 `
  --reviewer-b-root artifacts.local/evidence/eval-validity-r0/phash-reviewer-b-v2 `
  --private-root artifacts.local/evidence/eval-validity-r0/phash-private-map-v2

E:\codex-tools\bin\blindassist-python.cmd -m scripts.research.eval_validity_r0.finalize_phash_manual_review `
  --admission-receipt artifacts.local/evidence/eval-validity-r0/data-admission-v1.json `
  --private-map artifacts.local/evidence/eval-validity-r0/phash-private-map-v2/private-review-map.json `
  --packet-a artifacts.local/evidence/eval-validity-r0/phash-reviewer-a-v2/packet.json `
  --packet-b artifacts.local/evidence/eval-validity-r0/phash-reviewer-b-v2/packet.json `
  --review-a artifacts.local/evidence/eval-validity-r0/phash-review-a-v2.json `
  --review-b artifacts.local/evidence/eval-validity-r0/phash-review-b-v2.json `
  --output artifacts.local/evidence/eval-validity-r0/phash-resolution-v2.json

E:\codex-tools\bin\blindassist-python.cmd -m scripts.research.eval_validity_r0.reconcile_phash_data_admission `
  --held-admission-receipt artifacts.local/evidence/eval-validity-r0/data-admission-v1.json `
  --phash-resolution artifacts.local/evidence/eval-validity-r0/phash-resolution-v2.json `
  --output artifacts.local/evidence/eval-validity-r0/data-admission-after-phash-v2.json
```

只有直接 PASS，或上述固定双人 pHash 审阅生成
`EVAL_VALIDITY_DATA_ADMISSION_PASSED_AFTER_PHASH_MANUAL_REVIEW`，才可产生两份物理分离的 P0 packet。每个 packet
有 192 个随机排序的 opaque items（48 event × 4 anchor），每项仅含结束于该 anchor 的 RGB；
private map 绝不能交给任何 reviewer。

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m scripts.research.eval_validity_r0.prepare_p0_review_packets `
  --screening-cohort artifacts.local/evidence/eval-validity-r0/screening-cohort-v4.json `
  --admission-receipt artifacts.local/evidence/eval-validity-r0/data-admission-v1.json `
  --materialized-root artifacts.local/evidence/eval-validity-r0/continuous-native-inputs-v1 `
  --reviewer-a-root artifacts.local/evidence/eval-validity-r0/p0-reviewer-a-v1 `
  --reviewer-b-root artifacts.local/evidence/eval-validity-r0/p0-reviewer-b-v1 `
  --private-root artifacts.local/evidence/eval-validity-r0/p0-private-map-v1
```

两位 reviewer 独立提交后，custodian 只能使用 private map 与原 packet 做下面的 JSON-only
归档；它不读取 RGB/mask，更不读取任何模型、truth 或 oracle trace。任一 `UNKNOWN` 或不一致会
生成 `STOP_EVENT_FACT_CONSISTENCY_NOT_ESTABLISHED`；漏项、packet 替换或隔离声明失效会被拒绝归档。
两种情形都不允许第三人仲裁或抽换 event：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m scripts.research.eval_validity_r0.finalize_p0_anchor_agreement `
  --screening-cohort artifacts.local/evidence/eval-validity-r0/screening-cohort-v4.json `
  --admission-receipt artifacts.local/evidence/eval-validity-r0/data-admission-v1.json `
  --private-map artifacts.local/evidence/eval-validity-r0/p0-private-map-v1/private-review-map.json `
  --packet-a artifacts.local/evidence/eval-validity-r0/p0-reviewer-a-v1/packet.json `
  --packet-b artifacts.local/evidence/eval-validity-r0/p0-reviewer-b-v1/packet.json `
  --review-a artifacts.local/evidence/eval-validity-r0/p0-action-review-a-v1.json `
  --review-b artifacts.local/evidence/eval-validity-r0/p0-action-review-b-v1.json `
  --output artifacts.local/evidence/eval-validity-r0/p0-anchor-agreement-v1.json
```

只有 `P0_ANCHOR_CONSISTENCY_PASSED` 才能生成两份新的、与 P0 reviewer 不同的全事件 RGB
packet；每份只包含 48 个随机排序的 opaque event，并包含全窗口的因果 RGB 序列：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m scripts.research.eval_validity_r0.prepare_p1_review_packets `
  --screening-cohort artifacts.local/evidence/eval-validity-r0/screening-cohort-v4.json `
  --admission-receipt artifacts.local/evidence/eval-validity-r0/data-admission-v1.json `
  --p0-agreement artifacts.local/evidence/eval-validity-r0/p0-anchor-agreement-v1.json `
  --materialized-root artifacts.local/evidence/eval-validity-r0/continuous-native-inputs-v1 `
  --reviewer-a-root artifacts.local/evidence/eval-validity-r0/p1-reviewer-a-v1 `
  --reviewer-b-root artifacts.local/evidence/eval-validity-r0/p1-reviewer-b-v1 `
  --private-root artifacts.local/evidence/eval-validity-r0/p1-private-map-v1
```

两位新的 reviewer 独立提交后，custodian 运行同样 JSON-only 的 P1 归档。两人须对 knownness 与
两类区间完全一致，而且每个 P1 区间必须逐点包含/排除 P0 已冻结 anchor；任一不一致、UNKNOWN 或
与 P0 冲突都会 `STOP_FULL_EVENT_FACT_CONSISTENCY_NOT_ESTABLISHED`：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m scripts.research.eval_validity_r0.finalize_p1_action_facts `
  --screening-cohort artifacts.local/evidence/eval-validity-r0/screening-cohort-v4.json `
  --admission-receipt artifacts.local/evidence/eval-validity-r0/data-admission-v1.json `
  --p0-agreement artifacts.local/evidence/eval-validity-r0/p0-anchor-agreement-v1.json `
  --private-map artifacts.local/evidence/eval-validity-r0/p1-private-map-v1/private-review-map.json `
  --packet-a artifacts.local/evidence/eval-validity-r0/p1-reviewer-a-v1/packet.json `
  --packet-b artifacts.local/evidence/eval-validity-r0/p1-reviewer-b-v1/packet.json `
  --review-a artifacts.local/evidence/eval-validity-r0/p1-action-review-a-v1.json `
  --review-b artifacts.local/evidence/eval-validity-r0/p1-action-review-b-v1.json `
  --output artifacts.local/evidence/eval-validity-r0/p1-action-facts-v1.json
```

通过后，P1 action facts 才能与独立冻结的 source-mask scene facts 合并为
`full-event-facts`；它含 P0 agreement hash、两份 full-review hash、零 unresolved units，以及
每个正例完整的 alertable/passed interval。只有它、scene-frame ledger、trace manifest 和全帧
feedback traces 都冻结后，运行：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m scripts.research.eval_validity_r0.finalize_scene_facts `
  --screening-cohort artifacts.local/evidence/eval-validity-r0/screening-cohort-v4.json `
  --admission-receipt artifacts.local/evidence/eval-validity-r0/data-admission-v1.json `
  --p0-agreement artifacts.local/evidence/eval-validity-r0/p0-anchor-agreement-v1.json `
  --p1-action-facts artifacts.local/evidence/eval-validity-r0/p1-action-facts-v1.json `
  --materialized-root artifacts.local/evidence/eval-validity-r0/continuous-native-inputs-v1 `
  --output-root artifacts.local/evidence/eval-validity-r0/frozen-scene-event-facts-v1
```

该步骤把原生 SANPO panoptic mask 的固定 `R` class-ID、`G/B` instance-ID 映射为可复算的
source scene mask/box（不是“应提醒”标签），再按合同中已冻结的 source-mask stratum × P1 action
interval 规则形成 12-12-12-12 bucket。任一 profile/action 不匹配或桶数量不符，都只产生
`HOLD_EVAL_VALIDITY_DATA` receipt；不得重抽事件、重标或调阈值。它输出的 `cohort-v1.json`、
`full-event-facts-v1.json` 和 `scene-facts.jsonl` 才是之后 trace runner 的输入。

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m scripts.research.eval_validity_r0.evaluate `
  --contract docs/research/dual-loop/EVAL_VALIDITY_R0_CONTRACT_2026-08-02.json `
  --exclusion-registry artifacts.local/evidence/eval-validity-r0/exclusion-registry-v1.json `
  --cohort artifacts.local/evidence/eval-validity-r0/frozen-scene-event-facts-v1/cohort-v1.json `
  --p0-agreement artifacts.local/evidence/eval-validity-r0/p0-anchor-agreement-v1.json `
  --p1-action-facts artifacts.local/evidence/eval-validity-r0/p1-action-facts-v1.json `
  --full-event-facts artifacts.local/evidence/eval-validity-r0/frozen-scene-event-facts-v1/full-event-facts-v1.json `
  --scene-frames artifacts.local/evidence/eval-validity-r0/scene-frames-v1.jsonl `
  --trace-manifest artifacts.local/evidence/eval-validity-r0/trace-manifest-v1.json `
  --feedback-traces artifacts.local/evidence/eval-validity-r0/feedback-traces-v1.jsonl `
  --output artifacts.local/evidence/eval-validity-r0/result-v1.json
```

输入不变量：cohort 必须与 registry 的每一个 native source session 不重叠；review A/B 需要
独立、opaque 且在 trace access 前提交；P1 的 full-event facts 同样必须先冻结；`scene-frames`
只允许 `current_yolo`、`truth_box`、`truth_mask`，且每行必须绑定冻结的 source scene-fact manifest；
`feedback-traces` 必须有四臂，并覆盖每个冻结 event 的每一帧 delivered-feedback 布尔值。输出目录不可覆盖。

P0 review 的顶层字段为 `schema_version`、`protocol_id`、`reviewer_role`、`screening_cohort_sha256`、
`isolated_context=true`、`other_review_visible_before_submission=false`、
`model_or_oracle_output_visible=false` 与 `items`；每个 item 只含一个 opaque `review_item_id`
及其唯一的 `anchor.frame_index / reminder_now / cleared / knownness`，不能把同一 event 的多个
anchor 放进一个 item。P1 `full-event-facts`
还必须绑定 P0 agreement SHA、两份独立 full-review SHA，以及每个正例的完整 interval；审计器
不会接受只带汇总结论或只带 anchor interval 的替代文件。

在审计器产生 `result-v1.json` 后，可以渲染面向研究决策的双层 Markdown report：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m scripts.research.eval_validity_r0.render_report `
  --result artifacts.local/evidence/eval-validity-r0/result-v1.json `
  --output artifacts.local/evidence/eval-validity-r0/result-v1.md
```

## 输出

只写入 `artifacts.local/evidence/eval-validity-r0/` 的显式新目录。registry 记录历史角色的
hash 和 41 个当前排除 session；最终 `result-v1.json` 分开报告 representation 与 event
quality，并保存全部输入 SHA-256。输入 materializer 只下载冻结的原生 RGB/mask，绝不下载或
运行模型/oracle/Android；evaluator 本身不训练模型或运行 Android。

## 裁判审计 R0

四个裁判测试的独立合同见
[JUDGE_AUDIT_R0_CONTRACT_2026-08-02.md](../../../docs/research/dual-loop/JUDGE_AUDIT_R0_CONTRACT_2026-08-02.md)。裁判员只提交原子观察，不提交
`ACTIONABLE/NON_ACTIONABLE`；审计器按冻结规则派生动作标签。review map 使用不含语义的
opaque ID，且 causal packet 不暴露模型输出、候选类别或选择理由。每个 event 还绑定一个仅供
审计分析的 `discovery_arm`；正式 cohort 至少包含 source-mask 与一个独立 arm。先只做候选
preflight：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m scripts.research.eval_validity_r0.judge_audit_preflight `
  --discovery artifacts.local/evidence/eval-validity-r0/unseen-discovery-selection-v3.json `
  --discovery-arm source_mask `
  --discovery artifacts.local/evidence/eval-validity-r0/sanpo-train-normal-discovery-v1.json `
  --discovery-arm source_mask `
  --exclusion-registry artifacts.local/evidence/eval-validity-r0/exclusion-registry-v1.json `
  --output artifacts.local/evidence/eval-validity-r0/judge-audit-preflight-v6.json
```

该输出只回答“是否有足够独立候选开始下一阶段”，不会把 source-mask profile 变成 event truth；
当前 discovery mix 仍未建立，正式分母关闭。pilot 的时序固定为：先 output-blind 冻结 8–12
个事件，完成两份 causal 与一份 retrospective primitive review 并封存；之后才让 YOLO 以
`SELECTION_ONLY` 读取框、尺度、位置和标签无关的 `selection_time_slot`，deterministic 枚举
并按冻结排序取前 3–4 对；pair-builder 明确不能读取 `reviewed_event_phase` 或
`reviewed_motion_relation`。最后生成 native/system-chain oracle trace，再运行
`judge_audit`。若 pair 不足，保持 `NOT_EVALUABLE/HOLD`，不得回头按标签挑样本。
系统链仍沿用统一 decision kernel；同时必须提供不经过 YOLO-shaped kernel 的 native
information-ceiling 路径。缺失 depth/geometry/trajectory 或出现 UNKNOWN 时保持
`NOT_EVALUABLE`，primitive 与 derived stability 分开报告；不得用 retrospective 结果替代
causal 裁决。

当前第一轮 burned pilot 的失败结果保留在
`artifacts.local/evidence/eval-validity-r0/judge-burned-pilot-v1/`，不可与新边界下的 review
混用。修订后的 v2 packet 位于
`artifacts.local/evidence/eval-validity-r0/judge-burned-pilot-v2/`：选取未消费的 009–016 八个
事件、两份 causal 前缀可见 RGB packet 和一份 retrospective 全序列 RGB packet。v2 将
`visibility` 固定为 `CURRENT_RGB_FRAME_ONLY`，每个 frame card 同时提供
`current_rgb_frame`、`visibility_rgb_frames` 与字段专用 `temporal_rgb_frames`；模糊、旋转和
关系不确定仍由 `evidence_quality`/对应 primitive 记录。packet 只要求六个 primitive，不含
mask、模型输出、YOLO、oracle 或 action label。三个 reviewer 提交后，先用
`seal_judge_review_bundle` 生成只暴露 bundle hash 的 seal，再用
`select_judge_counterfactual_pairs` 读取完整的 selection-only candidate universe，按冻结排序
生成 pair manifest；若 pre-label eligible 数不足，保留 `NOT_EVALUABLE`，不回挑样本。
v2 的新 plan binding 与 RGB-only staging 分别由
`rebind_judge_native_plan` 和 `stage_judge_rgb_subset` 生成，并各自写入 calibration receipt；
它们不改变既有 RGB frame receipt，不带 source mask，也不打开 formal denominator。

v4 在 v2 之后进一步冻结 `primitive_observability_v4`：所有几何 primitive 共享
`route_anchor`（当前承载相机的行人支撑面及其正前方连续延伸），`path_relation`、
`route_certainty` 和 `evidence_quality` 只能看当前 RGB 帧；只有 `motion_relation` 与
`phase` 才按 causal past prefix / retrospective full event 使用 temporal frames。`phase` 明确定义为
当前 path 占用相对于允许前缀的时相：当前 non-blocking 且此前没有 blocking 为
`BEFORE_INTRUSION`，当前 blocking 为 `CURRENT_INTRUSION`，当前 non-blocking 且此前出现
blocking 为 `PASSED_CLEAR`，只有 path/锚点/证据/允许前缀不足才为 `UNKNOWN`；causal 不看未来，
retrospective 不改写 current-only 几何字段。几何字段的
`AMBIGUOUS`、`UNKNOWN` 和 `INSUFFICIENT` 语义不再由“是否提醒”反推，review packet 给出
独立 current-only frame cards 与字段级版本/窗口。v3 曾计划使用 017–028 十二个事件，但因
reviewer 未提交而不形成审阅证据；这些事件不再复用。v4 使用未消费的 029–040 十二个事件重新
烧录；v2/v3 review 与 v4 review、seal、pair manifest 不得混用。

## 安全边界

- 旧 RISKSEG 30-event cohort 是 consumed Development evidence，禁止作为任何输入或调参来源。
- scene mask/box 仅是 source-grounded scene fact，不能成为“应提醒”或用户安全真值。
- `synthetic_oracle` 是 host-only evaluator-integrity control，永不成为模型 target 或运行时输入。
- 任何 action review 的 `UNKNOWN` 或 disagreement 都 fail closed，绝不能转为 no-alert/safe。

## 停止条件

`STOP_EVENT_FACT_CONSISTENCY_NOT_ESTABLISHED` 时不读取 trace；
`STOP_EVALUATOR_INTEGRITY_NOT_ESTABLISHED` 时不解释真实 arm；
`STOP_ORACLE_MONOTONICITY_NOT_ESTABLISHED` 时修正 evaluator/adapter，而不是训练、挑 seed、
调阈值或修改风险规则。新 cohort 未达到 `48 events / 48 sessions / 12-12-12-12` 或污染审计
有未解决候选时，终态为 `HOLD_EVAL_VALIDITY_DATA`。

## 假设与规则质疑

可证伪假设是：固定同一 decision chain 后，`YOLO → truth box → truth mask → synthetic oracle`
不会降低 hit、clearance 或时效性，也不会提高 false-alert event。任一逆向即表明评价器或
fact-to-evidence adapter 不能支撑模型排序。若 synthetic oracle 自身失败，优先怀疑评分/反馈
链而不是感知算法。

## 失败资产复用

未通过 action-review 一致性的 RGB 包只能作为 ontology counterexample；未通过单调性的
trace 只能作为 evaluator regression fixture；未通过数据隔离或 pHash 人工判定的候选保持
`HOLD`，不得改名为新 event-eval。任何产物都不得包装为模型训练、选模或 App 晋级证据。
