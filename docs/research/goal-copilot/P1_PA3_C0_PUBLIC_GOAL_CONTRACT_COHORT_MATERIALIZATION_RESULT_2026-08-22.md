# P1-PA3-C0 public Goal Contract cohort materialization

日期：2026-08-22（Asia/Hong_Kong）

终态：`P1_PA3_C0_PROSPECTIVE_INTAKE_READY_EXISTING_ELIGIBLE_EPISODES_ZERO_PA3_INFERENCE_NOT_AUTHORIZED`

## 最窄答案

PA2 后的待验证解释改称 `Proposal–Identity Responsibility Mismatch`，但当前 7-case PA0–PA2 cohort 不能直接
进入 goal-semantic proposal：它的 public input 只有 current frame、initial exemplar 与 referent ID；`door / wall
artwork / wall clock / wine rack / pot / smart home video` 只存在 private evaluator metadata。把这些类别冻结为 text
prompt 仍是 GT leakage。

C0 对既有候选来源做了 provenance 审计，合格 episode 为 `0`：

- P1-D0 / PA0–PA2：没有 provider-public 原始 goal text；
- Silver-B：`goal_text` 随 truth-bearing reviewed annotation / target name 生成，不证明 goal 先于 truth；
- Last-10m：`goal_text` 由 `goal_name` 模板生成，没有原始 user-task-before-truth receipt。

因此没有回填历史 goal，也没有创建伪 prospective episode、private truth、RGB roster、provider output 或 PA3 result。

## 已建立的最小 prospective surface

[`p1_pa3_c0`](../../../scripts/research/goal_copilot_bridge/p1_pa3_c0/README.md) 现在提供：

- 空的 prospective intake template；空 roster 会被 materializer 拒绝，不能伪称 cohort ready；
- 全局 exact `goal_type -> canonical_prompt` 映射；当前只登记产品主线已有的
  `NAMED_BUILDING_ENTRANCE -> building entrance`，没有 synonym set 或逐 episode override；
- `USER_TASK_INPUT / PRODUCT_TASK_INPUT` source authority、capture `NOT_STARTED`、truth `NOT_CREATED` 前序约束；
- provider-public truth-key firewall，拒绝 target/category/instance/bbox/mask/referent/evaluator 与手填
  `canonical_prompt`；
- deterministic intake、prompt-map 与 materialized receipt SHA-256；输出仍显式
  `created_before_truth=PENDING_FUTURE_TRUTH_BINDING_TO_THIS_RECEIPT` 与 `pa3_inference_authorized=false`，不靠
  materializer 自我证明时间先后。未来 private truth 必须反向绑定该 immutable goal-receipt body SHA-256，admission 才能
  确认 precedence。

只有未来真实 prospective goal intake materialized 后，才能另行冻结 RGB/truth roster并决定是否授权 PA3。
Goal receipt 本身不授权采集、标注、模型调用、semantic proposal、identity、AMRM、verifier 或 App。

Claim ceiling：`PUBLIC_GOAL_CONTRACT_PROVENANCE_MECHANICS_ONLY_NO_PROPOSAL_IDENTITY_MODEL_PRODUCT_OR_SAFETY_CLAIM`。
