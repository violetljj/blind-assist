# P1-PA3 goal-semantic proposal availability implementation

日期：2026-08-22（Asia/Hong_Kong）

状态：`SUPERSEDED_BY_EXECUTED_DEVELOPMENT_RESULT`

执行结果见 [`P1-PA3 + FRG1 result`](P1_PA3_GOAL_SEMANTIC_AND_FUNCTIONAL_REGION_RESULT_2026-08-22.md)。本页只保留
执行前实现边界，不再代表当前动态状态。

## 已落地的算法执行面

PA3 只回答：合法、先于 capture/truth 的用户 Goal Contract 通过全局 canonical text prompt 驱动同一
YOLOE-26n-seg 时，legal target 是否进入 bounded candidate pool。它不选择 identity，也不恢复 AMRM/verifier。

实现位于 [`p1_proposal_availability`](../../../scripts/research/goal_copilot_bridge/p1_proposal_availability/README.md)：

1. `materialize_pa3_inputs.py` 从 immutable C0 goal receipt、capture manifest 与后生成 private truth 自动生成
   public input、private evaluator wrapper 和逐 case precedence receipt；机械验证
   `goal_time < capture_time < truth_time`、图像/hash、C0 body hash、prompt-map hash、reference mode 与 legal target set。
2. `run_yoloe_semantic_prompt.py` 只读 public input，以 `model.set_classes([canonical_prompt])` 使用冻结 text-prompt
   interface；保持 `imgsz=640 / conf=0.001 / max_det=100 / bounded K=10`，不接收 private path，不使用 visual exemplar，
   不执行 identity selection 或 prompt/config sweep。runner 与 evaluator 都 fail-closed 固定 PA0–PA2 的
   checkpoint SHA-256 `1741c1f8...c8c1b` 和 Ultralytics `8.4.52`，确保唯一变量确实只是 conditioning interface。
3. `evaluate_pa3.py` 在 prediction 完成后私开 truth，报告 IoU >= 0.30 的 Recall@1/3/5/10、逐 legal target
   Recall@10 与 best IoU。`UNIQUE` 要求恰好一个 legal target；`SET_VALUED` 以任一合法目标进入池为 primary success，
   并另报 legal-set coverage；`AMBIGUOUS` 不进入 specific-referent primary denominator，只作 proposal-set diagnostic。

public input 会递归拒绝 bbox/mask/category/instance/object UID/referent/evaluator truth 和 per-episode canonical prompt
override。precedence receipt 绑定 canonical private-truth body hash，private wrapper 再绑定最终 public-input file hash，
避免循环哈希或运行时 GT 泄漏。

## 尚未发生

现有 C0 eligible episode 仍为 `0`；没有 prospective goal receipt、capture roster 或 private truth，因此没有创建正式
PA3 run directory、prediction、evaluation、模型调用或 performance terminal。当前实现不授权从 PA0–PA2 private
category 回填 prompt，也不允许把未来 PA3 与旧 7-case 进行不配对的数值胜负比较。

未来合法 cohort 到位后，唯一算法执行顺序为：C0 receipt → capture → private truth → PA3 input materialization →
一次 GT-blind semantic proposal → 一次 private evaluation。结果分支只描述该 cohort 的 zero/partial/full bounded
availability；identity、Contrastive Verifier、functional/region grounding 与 App 均需依据结果另行授权。

验证：PA3 8 项 contract tests、PA0–PA2 相邻回归合计 15 项通过。

Claim ceiling：`PROSPECTIVE_GOAL_SEMANTIC_PROPOSAL_MECHANICS_ONLY_NO_PERFORMANCE_IDENTITY_GENERALIZATION_PRODUCT_OR_SAFETY_CLAIM`。
