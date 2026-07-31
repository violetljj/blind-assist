# Failure-Aware Causal Component Validator R0 Protocol

状态：`PROTOCOL_AND_IMPLEMENTATION_FROZEN / RESULT_NOT_RUN /
DEVELOPMENT_STANDARD / FINAL_CONFIRMATION_NOT_ACTIVATED`

日期：2026-08-01（Asia/Hong_Kong）

协议：`DUAL_LOOP_SEGMENTATION_LEARNED_COMPONENT_VALIDATOR_R0`

## 结论先行

本轮只允许一个轻量 learned component validator：固定 21 列因果 allowlist、
`StandardScaler + L2 Logistic Regression`、source-session grouped nested cross-fit、
训练上下文内阈值选择、九项 utility 门与两项工程门。不会再训练 DDRNet、换分类器、
搜索 feature subset、访问 fresh holdout 或接主动融合。

## 证据角色与前向规则质疑

11,757 个 raw components 来自 520 帧、10 个 source-session；全部 outcome 已被查看，
且缺 participant、route、parent-capture ledger。它们不能产生 unseen、independent
validation 或 Confirmation。

R1 的四个 consumed-fresh session 历史 amendment 及 R1 terminal 保持不可变。本协议
依据当前 `DEVELOPMENT_STANDARD` 的“可复用 declared consumed data”前向规则，为一个
不同因果变量的 learned-postprocess 新问题增加：

```text
CONSUMED_DEVELOPMENT_CROSSFIT_CONTEXT_ONLY
```

这不是恢复 fresh。每个 source-session 的被评分 prediction 都来自完全排除该 session
的 scaler、sample weight、模型和 inner threshold；但 feature/hypothesis 设计已受全体
burned evidence 影响，因此结论仍只是内部 Development robustness。旧 R1 不 repair、
不 rerun，fresh/unseen/formal/Confirmation 角色仍永久禁止。

## 标签、feature 与禁止项

标签固定为：

```text
KEEP = raw component intersects same-class canonical residual truth by >=1 pixel
REJECT = otherwise
```

这是 class-strict image-space residual proxy，不是实例对应、现实障碍、安全或可通行
真值。feature 只能来自 current raw component、最多 5 个过去 materialized observation
与 current YOLO-union geometry。history 在 session/sequence 边界清空。

现有 Atlas 的完整未来 persistence、false run length、next IoU 与 truth/mechanism 字段
只作 target 或事后失败归因，严禁进入模型。track age 被替换为“不假设实例对应”的
同足迹 causal age。entropy 没有保存完整 probability tensor，明确不伪造。raw component
已是 `segmentation AND NOT A`，所以与 YOLO mask overlap 结构性为零；COCO 与两类
segmentation 也没有合法 same-class 映射，这两项不进入 R0。

机器 feature、缺失值和范围合同由
[`default.json`](../../../configs/dual_loop_segmentation_learned_component_validator_r0/default.json)
与
[`component_table.schema.json`](../../../configs/dual_loop_segmentation_learned_component_validator_r0/component_table.schema.json)
共同拥有。

## Grouped 模型与 operating point

外层为 10 个 source-session 的 leave-one-out。每个外层训练上下文的 9 个 session 再
进行 grouped inner OOF；scaler、class/session weights 与 Logistic 系数都只从相应
训练 session 学习。阈值只在 inner OOF 上从固定 19 点 grid 选择，绝不读取 outer-heldout
label 后修改。

模型合同固定：

```text
LogisticRegression
penalty=L2
C=1.0
solver=liblinear
max_iter=1000
tol=1e-6
random_state=20260801
```

阈值选择最大化九门的最小规范化 margin；并列依次选择更高 minimum-session retention、
更高 FP reduction、更低 threshold。若 inner 没有全部过门点，仍按同一 least-shortfall
规则选择一个预声明 operating procedure 供外层 falsification；不得看 outer outcome
救援。

## 对照、九门与工程门

对照固定为 raw DDRNet、union causal 2-of-3、confidence `>=.65`、历史预冻结 primary
`CLASS_CONDITIONED_MULTI_NEGATIVE` 和 learned validator。R0.1 shadows 全报历史但没有
selection authority，不事后挑“best”。

九门保持：

```text
FP reduction >= 0.30
overall recall retention >= 0.90
minimum-session recall retention >= 0.80
boundary retention >= 0.80
obstacle retention >= 0.80
C-A recall >= 0.05
C-A FP-area increment <= 0.05
component recall >= 0.50
false components/frame <= 3.0
```

host 工程门固定：

```text
feature extraction + Logistic + keep/reject + mask reconstruction P95 < 3 ms
serialized model + scaler <= 64 KiB
bounded causal state + maximum feature buffer <= 1 MiB
```

延迟排除 DDRNet/YOLO inference、truth、文件 I/O 和 fit；包含从既有 raw component/mask
计算因果特征、标准化、sigmoid、整组件决策与 candidate mask 重建。

## 三态 terminal

`SUPPORTED` 要求九门和两工程门全部通过。

`NEAR_MISS_SINGLE_TRAINING_SUCCESSOR` 只在工程门全过、最多一项 utility 门失败且落在
预冻结容差内，并且 retained false area 至少 50% 属于 Atlas
`STABLE_HIGH_CONFIDENCE_ERROR`、覆盖至少 5 个 session 时成立。它只授权另立
component-aware loss 协议设计，不自动授权训练，且不得从 consumed stress outcome
构造 loss target。

其他有效负结果统一为 `NOT_SUPPORTED_AND_GATING_STOP`：关闭当前 reference 上的 active
learned component gating，不换 XGBoost/MLP/Transformer、不调 grid、不接 Android；
visual sidecar/coverage diagnostic 可保留。

执行合同异常单列 `NOT_EVALUABLE` protocol axis，不占用或伪造三个科学终态。

## 停止条件

protocol/implementation freeze 后只执行一次 prepare、grouped Logistic、benchmark 与
独立 validator。没有 operating procedure 同时通过即按冻结映射结束；不访问新 fresh、
不选择少数 fold/seed、不改标签、feature、C、solver 或 threshold grid。
