# HFTF F0.1 SANPO official-test heldout 执行合同

状态：`FROZEN_AFTER_NINE_CHECKPOINT_GATE_BEFORE_HELDOUT_TARGET_MATERIALIZATION_OR_STUDENT_OUTPUT`

机器合同 SHA-256：
`cdc05f52f3d10ce8479025a0a0137f6d8c8a4d5d6faf320245dd0295c3b39462`

## 结论

九个 train/dev checkpoint 已通过独立完整性 gate，因此现在只开放一次性
official-test heldout canary。此合同在任何 heldout target 物化或学生输出前冻结，
且不因三个 dev temporal delta 为负而改变来源、模型、checkpoint、阈值、指标或门。

## 固定评估对象

- 来源：source lock 中固定的 3 个 official-test parent sessions，顺序不变。
- 样本：每个来源 13 个 reference anchors，共 39 个；teacher 只用
  stride-4/offset-2 reference view。
- 模型：seeds 17/29/43 × `SF_CURRENT / SF_FUTURE / HIST_FUTURE` 的九个
  byte-bound checkpoints，按 seed-major 顺序运行。
- 推理：CUDA float32、无 AMP、eval、batch 8、320×192、ImageNet normalize、
  risk threshold 0.5；known head 不遮蔽 risk metric。

## 真值/输出防火墙

heldout 包拆成三个文件：只含历史 RGB 的 `inference_inputs.jsonl`、隔离的
`heldout_truth.jsonl` 与 `teacher_receipts.jsonl`。预测进程只能打开前者、九个
checkpoint 和历史 RGB，先冻结不含真值的 351 条 predictions 及其 SHA-256；
随后独立 truth-join 进程才可读取 truth。这样 student forward 不接触 teacher
target、depth、mask、pose 或 future RGB。truth-join 与终态 validator 也不导入
predictor、训练模型或 `torch`，且所有 known/risk 二值真值都要求原始 JSON 整数
`0/1`，不接受 bool、字符串或浮点强制转换。

预测 one-shot 在首次 forward 前以固定全局 ledger 消耗；truth-join 和独立终态
validation 则分别在首次打开 truth 前独占创建 canonical output root，并写入、fsync
execution receipt。成功结果或 `NOT_EVALUABLE` failure 都原子落盘；receipt 一旦创建，
相应的一次授权即被消耗，异常或部分执行也不得重试。truth-join 与终态 validator
都会独立重算 predictions 的 canonical ordered join-key SHA-256。

## 冻结 effect gates

主比较为每个 seed 的 `HIST_FUTURE - SF_FUTURE`：

- median micro F1 delta ≥ 0.03，且每个 seed delta 严格大于 0；
- median recall delta ≥ -0.02；
- median false-positive-rate delta ≤ 0.02；
- body/head 各自的 median F1 delta ≥ 0；
- 三个 parent source 中最差的 median-seed F1 delta ≥ -0.02；
- `SF_CURRENT` 三 seed median current-label micro F1 ≥ 0.6。

所有比较使用未舍入值。全部门同时通过才得到 signal-supported；任何 effect gate
不通过都直接得到 no-gain stop，不允许 after-outcome rescue。

## 主张边界

即使通过，也只支持固定 SANPO-Synthetic body/head geometry-proxy 上的时间历史信号，
不等于人体、真实事件或安全效果，更不直接授权替换当前主线。若通过，下一步仍需与
主线做单独、同预算、同 heldout 规则的公平比较。
