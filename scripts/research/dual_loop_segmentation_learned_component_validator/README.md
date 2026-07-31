# dual_loop_segmentation_learned_component_validator

状态：development；`COMPLETE / VALID / NOT_SUPPORTED_AND_GATING_STOP`

正式结果见
[`DUAL_LOOP_SEGMENTATION_LEARNED_COMPONENT_VALIDATOR_R0_RESULT_2026-08-01.md`](../../../docs/research/dual-loop/DUAL_LOOP_SEGMENTATION_LEARNED_COMPONENT_VALIDATOR_R0_RESULT_2026-08-01.md)。
10-session nested LOSO 只通过 4/9 utility 门；host P95 为 `9.376145 ms`，因此关闭
当前 reference 上的 active learned component gating，不授权训练、设备或 Android
后继。下文保留冻结协议与可复算入口。

## 研究问题与版本

`DUAL_LOOP_SEGMENTATION_LEARNED_COMPONENT_VALIDATOR_R0` 只回答：多个弱但运行时可得的
component 特征，经唯一一类 Logistic Regression 组合后，能否在 source-session
grouped cross-fit 的 consumed Development 上同时通过既有九项 utility 门与两项工程门。

现有 520 帧、11,757 个 raw components、10 个 source-session 全部已经烧为 Development。
每个外层 fold 的被留出 session 不得进入该 fold 的 scaler、sample weight、模型或内层
阈值选择；但所有 session 的 outcome 过去均已被研究流程查看，因此结果只叫
`CONSUMED_DEVELOPMENT_CROSSFIT_ROBUSTNESS`，不叫 unseen、independent validation 或
Confirmation。participant、route 与 parent-capture 独立性仍不可评价。

历史 R1 role amendment 与 R1 terminal 不回写；本协议只为这个新 learned-postprocess
问题前向增加 `CONSUMED_DEVELOPMENT_CROSSFIT_CONTEXT_ONLY`。它不恢复 fresh 身份，也不
授权 R1 repair/rerun、模型选择、Confirmation 或产品行为。

## 稳定 Interface

使用专用 Python 3.11 环境：

```powershell
$py = "E:\codex-tools\projects\blindassist\toolchain\venvs\learned-component-validator-py311\Scripts\python.exe"

& $py -m scripts.research.dual_loop_segmentation_learned_component_validator.prepare `
  --repo-root . `
  --config configs/dual_loop_segmentation_learned_component_validator_r0/default.json `
  --output-root artifacts.local/evidence/dual-loop-segmentation-learned-component-validator-r0/prepared `
  --preflight-only
```

实现冻结并推送后，才按顺序执行：

```powershell
& $py -m scripts.research.dual_loop_segmentation_learned_component_validator.prepare `
  --repo-root . --config configs/dual_loop_segmentation_learned_component_validator_r0/default.json `
  --output-root artifacts.local/evidence/dual-loop-segmentation-learned-component-validator-r0/prepared

& $py -m scripts.research.dual_loop_segmentation_learned_component_validator.evaluate `
  --repo-root . --config configs/dual_loop_segmentation_learned_component_validator_r0/default.json `
  --prepared-root artifacts.local/evidence/dual-loop-segmentation-learned-component-validator-r0/prepared `
  --output-root artifacts.local/evidence/dual-loop-segmentation-learned-component-validator-r0/evaluation

& $py -m scripts.research.dual_loop_segmentation_learned_component_validator.benchmark `
  --repo-root . --config configs/dual_loop_segmentation_learned_component_validator_r0/default.json `
  --prepared-root artifacts.local/evidence/dual-loop-segmentation-learned-component-validator-r0/prepared `
  --evaluation-root artifacts.local/evidence/dual-loop-segmentation-learned-component-validator-r0/evaluation `
  --output-root artifacts.local/evidence/dual-loop-segmentation-learned-component-validator-r0/benchmark

& $py -m scripts.research.dual_loop_segmentation_learned_component_validator.validate `
  --repo-root . --config configs/dual_loop_segmentation_learned_component_validator_r0/default.json `
  --prepared-root artifacts.local/evidence/dual-loop-segmentation-learned-component-validator-r0/prepared `
  --evaluation-root artifacts.local/evidence/dual-loop-segmentation-learned-component-validator-r0/evaluation `
  --benchmark-root artifacts.local/evidence/dual-loop-segmentation-learned-component-validator-r0/benchmark `
  --output artifacts.local/evidence/dual-loop-segmentation-learned-component-validator-r0/validation.json
```

## 因果 feature 与 truth firewall

模型矩阵只接受 config 中精确 21 列 allowlist：class、面积/框/质心、纯几何 upper/central
indicator、confidence/margin、过去 observation 的 IoU/同足迹 age/flicker，以及既有
YOLO-union bbox gap。`session/scene/role` 只用于分组、权重和 history reset，不进入
feature matrix。

Atlas 的 `persistence_observations`、`false_activation_run_observations`、
`next_observation_iou`、truth histogram、mechanism tag 与 terminal outcome 均禁止作为
feature。机制 tag 只允许在结果后判断 near-miss 的残余失败结构。现有 component 已经
减去 YOLO A mask，所以 pixel overlap 结构性为零；COCO 与 segmentation 也没有冻结的
same-class taxonomy。R0 不伪造这两项。完整 probability tensor 未保存，因此 entropy
明确记为 `NOT_AVAILABLE_NOT_FABRICATED`。

## 模型、split 与阈值

只运行 `StandardScaler + L2 LogisticRegression(C=1, liblinear)`。外层 10-fold
leave-one-source-session-out；每个外层训练上下文内再做 9-fold grouped OOF，阈值只从
固定 `0.05..0.95 / step .05` 选择。选择规则最大化九门最小规范化 margin；并列时依次
偏向更高 minimum-session retention、更高 FP reduction、更低阈值。被留出的外层 session
不得传入任何 fit 或 threshold API。

历史对照固定为 raw DDRNet、union causal 2-of-3、confidence `>=.65` 和原预冻结 primary
`CLASS_CONDITIONED_MULTI_NEGATIVE`。R0.1 shadows 没有 selection authority，不事后称
“最佳历史 gate”。

## 输出

全部输出只写独立的
`artifacts.local/evidence/dual-loop-segmentation-learned-component-validator-r0/`：

- `prepared/component_table.jsonl`：identity / exact runtime features / target /
  diagnostic 四个 namespace；
- `evaluation/fold_models.jsonl`：scaler、系数、intercept、inner threshold receipt，
  不保存 pickle；
- `evaluation/held_out_predictions.jsonl`、`frame_metrics.jsonl`、`result.json`；
- `benchmark/runtime_rows.jsonl`、`benchmark/report.json`；
- `validation.json`：独立重算 membership、feature、fold、纯 NumPy sigmoid、九门、工程门
  与 terminal。

## 安全边界

不访问新 fresh holdout，不做随机 component/frame split，不搜索模型族、C、solver、
feature subset 或 seed，不接 Android/QNN/A568、risk/feedback、TTS、振动或默认 App。
正结果最多授权另立 unseen-source Confirmation 设计；设备平台工程不得反向选算法。

## 停止条件

有效执行只允许：

```text
SUPPORTED
NEAR_MISS_SINGLE_TRAINING_SUCCESSOR
NOT_SUPPORTED_AND_GATING_STOP
```

`NEAR_MISS` 的 failed-gate 数、数值容差、stable-high-confidence 面积占比和 session
覆盖在 config 中预先数值化；它只授权另立训练协议设计，不授权训练。schema、membership、
leakage、fold、runtime 或 validator 异常进入独立的 `NOT_EVALUABLE` protocol axis，
不得伪装成三个科学终态之一。

## 假设与规则质疑

候选的因果差异是把多个弱 component 证据交给一个线性、可解释的学习器，而不是继续
堆固定手工门或训练完整 DDRNet。falsifier 是 cross-fit 后没有 operating procedure
同时通过十一门，或跨 session 不稳。R1 consumed-role 的前向挑战只降低到新的 consumed
Development fit context；旧 terminal、anti-recovery invariant 与 Confirmation 隔离均
保留。

## 失败资产复用

负结果可作为 learned-gating counterexample、component-aware loss 的设计约束、回归与
visual-only sidecar 诊断；不得再换分类器、阈值或 feature subset 救援，不得恢复 unseen
身份。
