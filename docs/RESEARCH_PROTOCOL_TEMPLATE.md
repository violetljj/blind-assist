# 渐进式研究协议模板

状态：current template

上位规则：[BlindAssist 渐进式研究治理](RESEARCH_GOVERNANCE.md)

新研究协议或旧协议的实质修订至少包含以下内容。日期化结果文档不需要反向套用。
模板和上位规则也允许被 challenge；若字段本身造成低信息增益负担，应提出精简版本，
而不是为了填表而填表。

## 1. 问题与阶段

```text
protocol_id:
version:
stage: DISCOVERY | CANARY | DEVELOPMENT | CONFIRMATION | DEPLOYMENT
question:
non_goals:
claims_allowed:
```

明确本阶段回答什么，不回答什么。不要在 discovery 协议里写 confirmation 或产品
结论。

## 2. 数据分区与访问

| ID | source/content identity | independence group / ancestry | Role | Outcome access | 后续用途 |
| --- | --- | --- | --- | --- | --- |
|  | 稳定来源 ID、内容/manifest 身份与依据 | 独立单元及其父级 |  | `NONE/METADATA_ONLY/GEOMETRY_ONLY/FULL` |  |

看过算法输出的 canary/development 数据不得再进入同一命题的 confirmation。
仅换别名不构成独立数据；机器合同必须写 `source_identity`、`content_identity`、
`identity_basis`、`independence_group` 和 `ancestry`。
`CONFIRMATION/DEPLOYMENT` 还必须写仓库相对路径 `identity_manifest_ref` 和实际文件
`identity_sha256`；validator 会读取 JSON 并复算，不接受不可解析的自我声明。

## 3. 约束表

| ID | Class | 描述 | 单位/值 | 依据 | 敏感性 | 修改规则 | Failure scope |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | `INVARIANT/GATE/GUARDRAIL/DIAGNOSTIC/ASSUMPTION` |  |  |  |  |  |  |

每个数值约束必须交代单位、依据、calibration source、敏感性与 revision policy。
探索阶段允许这些字段暂为 warning，但 confirmation 前必须补齐。
`GATE` 还必须是可执行判据，包含 `metric`、`operator`、`threshold` 和 `unit`；
纯文字标题不能充当 confirmation gate。

## 4. 冻结和修改

```text
freeze_level: F0 | F1 | F2 | F3
outcome_access_started: false
amendment_mode: IN_PLACE_BEFORE_OUTCOME | NEW_VERSION_ONLY
```

写明哪些字段当前冻结、哪些仍可迭代。outcome access 后只能新建版本。

## 5. 两轴结果

```text
execution_validity: VALID | INVALID | NOT_RUN
scientific_outcome: NOT_RUN | NOT_EVALUABLE_DUE_TO_EXECUTION | 按 stage 的允许值
invalid_execution_effect: CLOSE_EVIDENCE_VERSION_ONLY
terminal_scope: ITEM | WINDOW | SEQUENCE | BRANCH |
  IMPLEMENTATION_VERSION | EVIDENCE_VERSION | RESEARCH_QUESTION | PRODUCT
```

INVALID 与科学结果分开。terminal 采用能解释错误的最小范围。
`NOT_RUN` 只能配 `scientific_outcome=NOT_RUN`，不得在执行前预写结果。
若关闭/退役科学问题，独立证据 registry 的每项必须绑定仓库 JSON `evidence_ref`、
实际内容哈希及与引用对象一致的 protocol/source/independence identity。

## 6. 合法后继

明确：

- PASS 后可进入什么；
- FAIL、NOT_EVALUABLE 和 INVALID 各关闭什么；
- 哪些数据已经烧掉；
- 哪些新假设、独立数据或实现修复允许另立版本；
- 哪些权限永远不会由本协议自动产生。

## 7. 假设选择与失败学习

每个主要候选写明：

```text
hypothesis_id:
theoretical_or_empirical_basis:
causal_difference:
expected_information_gain:
minimal_test:
evaluation_metric:
falsifier:
cost:
resource_budget:
stop_condition:
selection_reason:
```

不得只因为 Agent 可以并行就穷举所有组合。网格搜索必须说明为何比假设驱动的少量
试验更有效。

实验设计还必须写明：

```text
search_strategy:
minimal_discriminating_experiment:
resource_budget:
stop_conditions:
```

可选但建议记录 `efficiency_strategy`：本轮复用什么、并行什么、为什么选择当前最小
验证集，以及哪些风险会触发升级。不要为了填写模板新增无信息量流程。

若重开旧失败：

```text
reopens_prior_failure: true
prior_failure_id:
material_change_dimensions:
  DATA | INPUT_SIGNAL | COMPENSATION | SYSTEM_ROLE |
  EVALUATION_TARGET | DEPLOYMENT_CONDITION
difference_from_previous:
```

相同前提下不得重复运行已经验证失败的方案。

失败后追加：

```text
failure_class:
observation:
inference:
alternative_explanations:
constraint_challenges:
next_hypotheses:
reuse_candidates:
information_gain:
```

任何 `AGENTS.md`、current 协议或数值门都可进入 `constraint_challenges`；修改
必须走版本化 amendment，不得静默绕过。

## 8. 每轮沉淀

```text
new_facts_and_evidence:
weakened_or_rejected_hypotheses:
unresolved_questions:
reusable_assets:
next_high_information_experiments:
governance_changes_needed:
```

同时判断低价值算法/治理模块是否应继续、合并、降级或删除。

## 9. 机器合同

为协议准备 JSON contract，字段结构可参考：

机器合同必须绑定 `governance_policy_id` 与 canonical policy SHA-256。调整策略时应
显式升级并重绑合同；不能用 `--policy` 临时换一份更宽松的文件取得 `VALID`。

```json
{
  "schema_version": "blindassist.research_protocol.v1",
  "governance_policy_id": "PROGRESSIVE_RESEARCH_GOVERNANCE_R1",
  "governance_policy_sha256": "<canonical-policy-sha256>",
  "protocol_id": "EXAMPLE_R0",
  "version": "R0",
  "stage": "DISCOVERY",
  "question": "What source-native conditions exist?",
  "claims_allowed": ["CANDIDATE_FOUND", "CANDIDATE_NOT_FOUND"],
  "data_partitions": [
    {
      "id": "discovery-a",
      "source_identity": "PUBLIC-SOURCE-A",
      "content_identity": "PUBLIC-SOURCE-A-MANIFEST-R0",
      "identity_basis": "Frozen source manifest and content digest.",
      "independence_group": "PUBLIC-SOURCE-A-DISCOVERY",
      "ancestry": [],
      "role": "DISCOVERY",
      "outcome_access": "METADATA_ONLY",
      "reuse_policy": "NOT_CONFIRMATION_IF_USED_FOR_SELECTION"
    }
  ],
  "constraints": [
    {
      "id": "source-authority",
      "class": "INVARIANT",
      "description": "Source identity is recorded.",
      "failure_scope": "ITEM"
    }
  ],
  "freeze": {
    "level": "F0",
    "outcome_access_started": false,
    "amendment_mode": "IN_PLACE_BEFORE_OUTCOME"
  },
  "result_model": {
    "execution_validity": "NOT_RUN",
    "scientific_outcome": "NOT_RUN",
    "invalid_execution_effect": "CLOSE_EVIDENCE_VERSION_ONLY",
    "terminal_scope": "ITEM"
  },
  "successor_policy": {
    "new_version_allowed": true,
    "preserve_previous_evidence": true
  },
  "experiment_design": {
    "search_strategy": "SINGLE_VARIABLE_COUNTERFACTUAL",
    "minimal_discriminating_experiment": "One test that separates proxy mismatch from source absence.",
    "resource_budget": "One sequence and one deterministic replay.",
    "stop_conditions": "Stop this hypothesis if the direct measure fails on the calibration fixture."
  },
  "hypotheses": [
    {
      "hypothesis_id": "H1",
      "theoretical_or_empirical_basis": "Source-native geometry is closer to the causal confound.",
      "causal_difference": "Uses source-native geometry rather than camera speed.",
      "expected_information_gain": "Separates proxy failure from source absence.",
      "minimal_test": "One controlled counterfactual fixture.",
      "evaluation_metric": "Separation of translation-induced radial expansion.",
      "falsifier": "Geometry-derived expansion remains above the guardrail.",
      "cost": "LOW",
      "resource_budget": "One fixture replay.",
      "stop_condition": "Stop H1 if physical calibration fails.",
      "selection_reason": "Directly tests the challenged proxy."
    }
  ],
  "failure_learning": {
    "failure_class": "NOT_RUN",
    "observation": "No execution yet.",
    "inference": "None.",
    "alternative_explanations": [],
    "constraint_challenges": [],
    "next_hypotheses": [],
    "reuse_candidates": [],
    "information_gain": "None before execution."
  }
}
```

验证：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/validate_research_protocol.py `
  --contract <path-to-contract.json>
```

validator 的 `errors` 是证据诚信或阶段越权问题；`warnings` 是尚未补齐的方法说明。
在 discovery/canary 中 warning 不阻断学习，在 confirmation/deployment 中关键门槛
说明缺失会成为 error。
