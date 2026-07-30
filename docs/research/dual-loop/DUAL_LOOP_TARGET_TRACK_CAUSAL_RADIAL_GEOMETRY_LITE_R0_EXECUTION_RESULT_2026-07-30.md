# Target/track-conditioned causal radial geometry LITE R0 execution result

日期：2026-07-30（Asia/Hong_Kong）

```text
DESIGN_REVIEW: PASS
IMPLEMENTATION_REVIEW: PASS
ACTIVATION_REVIEW: PASS
CLOSEOUT_REVIEW: PASS
PRODUCER_ATTEMPT: 1 / 1
PRODUCER_TERMINAL: EXECUTION_INVALID_STOP_NO_RERUN
PRODUCER_OUTPUT: NOT_CREATED
PRETRUTH_RECEIPT: NOT_CREATED
EVALUATOR: NOT_INVOKED
TRUTH_JOIN: NOT_RUN
SCIENTIFIC_OUTCOME: NOT_EVALUABLE
OLD_F1B_DECISION_ACCESS: SEALED / DECLARED_ZERO
CONFIRMATION: NOT_AUTHORIZED
CLAIM_CEILING: EXECUTION_FAILURE_DIAGNOSTIC_ONLY
```

## 结论

LITE R0 在唯一获准的 full producer attempt 中于首次出现的同目标、同 track epoch、
相邻帧尺寸变化处失败。预注册 activation 对 producer failure 的终点是
`EXECUTION_INVALID_STOP_NO_RERUN`；因此本轮没有第二次尝试、没有实现修补、没有
truth/event join，也没有两臂效果结论。

这不是 `BBOX_LOG_AREA_GROWTH` 或 `ROI_SPARSE_RADIAL_FLOW` 的输赢结果。两臂尚未形成
完整的 26,028 行 pre-truth ledger，固定的 469 个 primary natural events 从未进入
评价器，故科学终点只能是 `NOT_EVALUABLE`。

## 冻结执行身份

- implementation commit：`64fbc5aa546faa3d787ce014df55a7dd8ee8ab9f`
- implementation lock SHA-256：
  `9e2173d0b0cea2959fa35d2e52c7f13378aae0e5c62642a13c1524cbd5971807`
- activation SHA-256：
  `605fb2c7299bcd65eb53873224b17dfc966c930713932a7b54f3e6e9bc890cd3`
- replay input SHA-256：
  `14f1f7f0f330d8b01146e37c31505240f3f0e8d301846ebcad44a628948e6440`
- failure receipt SHA-256：
  `e00c6a5d8a7eb2db239fbf3a26ff950d0e02bb326ca255c4ac1222797ec92696`

执行前重新确认 activation、implementation lock 和 replay identity 均未漂移，且
`producer_output.jsonl`、pre-truth receipt 与 `evaluation.json` 均不存在。

## 失败证据

唯一 producer 在 `cv2.calcOpticalFlowPyrLK` 处收到 OpenCV assertion：
previous/next LK pyramid level size 不一致。source-only 诊断随后只读取 frozen
replay ledger 与 RGB：

- replay rows：13,014；
- unique replay images：8,363；
- decoded row shapes：`260×346` 12,965 行，另有 `258×346` 16 行、
  `260×259` 15 行、`260×344` 12 行、`250×346` 6 行；
- 首个同目标、同 epoch 尺寸变化位于 replay line 1,728：
  `track-000 / epoch-0009` 从 `revel-dynamic:01364` 的 `260×346`
  进入 `revel-dynamic:01365` 的 `258×346`。

失败发生在 producer 将内存结果写盘之前。复核确认：

- `producer_output.jsonl` 不存在；
- `producer_output.jsonl.receipt.json` 不存在；
- `evaluation.json` 不存在；
- 持久化 candidate rows 为 0；
- evaluator 未调用，truth/event ledger 未由本次执行读取；
- 旧 F-1B decision 输出未访问。

完整 source-only 失败凭据保存在 ignored
`artifacts.local/evidence/dual-loop/target-track-causal-radial-geometry-lite-r0/run-r0/execution_failure_receipt.json`。
独立 closeout review 又从 replay 与 RGB metadata 复核了首次 mismatch、两张图像
SHA-256、全量 shape 计数、目录文件集合和 activation stop rule，终点为
`CLOSEOUT_REVIEW_PASS`。

## 为什么必须停止

设计锁允许的唯一 implementation-only repair 已在 truth join 前用于闭合独立实现评审
发现；activation 又明确规定 producer attempt limit=1、partial/failure 后 no rerun。
为尺寸变化新增 resize/pad、跨尺寸 abstention/reset 或预处理会改变冻结 producer
行为和可能的 coverage，不是本轮可静默采用的工程修补。

若未来另行授权，只能创建新的 evidence version：在任何 candidate/truth outcome
访问前冻结跨尺寸语义、重做 synthetic/source fixtures、重新锁定实现并重新独立评审。
R0 的失败凭据和 `NOT_EVALUABLE` 终点必须保留，不得用后继版本覆盖或改写。

## 仓库核验说明

专项 implementation-lock validator 为 `VALID / errors=[]`，24 个 synthetic tests
通过，research protocol validator 与 documentation index check 通过。全仓
project-structure/repo-hygiene 门仍报告三项：

- 既有 `scripts/research/dual_loop/` 缺少 module README；
- 两份冻结历史凭据为精确 identity/command provenance 直接记录了 research
  implementation path，触发“应使用 stable root adapter”的结构规则。

第一项不属于本轮；后两项来自本轮已消费的 implementation lock 与 activation。
在执行终点后修改它们会破坏已评审 SHA 和审计链，因此按原样保留并显式披露，不能把
全仓结构门写成 PASS。

## 权限边界

本结果不支持算法有效、flow 优于 bbox、目标/区域复制、早提醒、实时可行、产品改善或
安全结论。它也不授权第二次 Development replay、Confirmation、Android、正式融合器、
提醒状态机或默认模型变更。REveL 仍只承担 Development truth；旧 F-1B decision 集
继续密封。
