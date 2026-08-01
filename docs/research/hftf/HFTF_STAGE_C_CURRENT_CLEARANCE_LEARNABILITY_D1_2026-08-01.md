# HFTF-G0-D1 current clearance learnability

状态：
`FROZEN_SCIENTIFIC_DESIGN_AFTER_G0_D0_BEFORE_D1_IMPLEMENTATION_CORPUS_OR_STUDENT_OUTCOME`

## 研究问题

D1 只问：

> 在 current RGB、encoder、输入、参数量和推理预算不变时，把每格直接 binary-risk
> logit 改成线性 meter-clearance scalar，能否改善 fresh source 上的 risk
> learnability？

它不加入 future/history 信息，也不换 backbone。两臂都保持 F0.1 `SF_CURRENT`：
anchor RGB 重复五次、MobileNetV3-Small、同 temporal fusion、known head、augmentation、
seeds 和 144 个输出。唯一目标机制变化是：

- direct arm 输出 risk logits；
- clearance arm 输出无 activation、无训练时 clamp 的线性 meter value，以 `<0 m`
  导出 risk。

## 为什么不能用普通 pooled MAE

D0 显示明确 source shift。6 个旧 train sources 的 body/head positive 比例约为
`24.17%/12.21%`，3 个旧 model-selection sources 降至 `6.12%/4.33%`；safe target
在 `+1 m` 的饱和比例又从 train 的 `46.47%/54.54%` 上升到 selection 的
`68.61%/76.52%`。普通 MSE、pooled MAE 或 bounded sigmoid/tanh 很容易奖励
safe-majority 常数输出。

clearance loss 因此冻结为：

```text
class-and-boundary-balanced SmoothL1(beta=0.1m)
+ 0.1 * BCEWithLogits(-clearance/0.1m, risk)
+ 0.25 * BCEWithLogits(known)
```

risk/safe 在每个 height 内按 6 个 train sources 静态等权，`|target|<=0.2 m` 再乘
2，之后重归一到均值 1。UNKNOWN 不填零或 `+1`，不进入 task loss。direct arm 使用
known-masked BCE 和同一 `0.25` known-loss 系数；positive weight 只由 6 train
sources 计算。

## 两阶段训练

Phase A 对每个 arm/seed 用旧 6 train 训练完整 30 epochs，在旧 3
model-selection sources 上按同一 risk 词典序选 epoch：

1. 最高 three-source macro F1；
2. 最高 worst-source F1；
3. 最高 micro F1；
4. clearance 只有在前三项精确相同时才看更低 source-macro MAE；
5. 最早 epoch。

Phase B 重置全部 RNG 和模型，在 9 个 outcome-open sources 上仍完整训练 30 epochs，
但只冻结同 arm/seed 的 Phase-A 预选 epoch，不再选择或 early-stop。这样两臂保持相同
最大训练预算，同时利用全部 9 个旧 sources。

## fresh opening 顺序

只有六个 final checkpoint hashes、训练 traces、代码与 prediction contract 全部冻结，
才能获取三个固定 fresh parents。获取后先做每个 `source × height` 的机会门：

- 25 个 current frames；
- known coverage 至少 `0.1`；
- positive/negative known 至少 `5/20`；
- UNKNOWN→SAFE 违规为零。

不足即
`G0_D1_FRESH_EVALUATION_NOT_EVALUABLE_NO_SOURCE_REPLACEMENT`，不得换来源。

机会充分后，prediction-only 进程只读 fresh RGB 与六 checkpoint；预测文件先冻结并
全局消费，随后 truth join 一次。不得第二次 forward、替换 checkpoint 或改变 threshold。
三个 parent sessions 才是独立单元，frames/cells 只是重复观测。

## 成功与停止

clearance arm 必须同时达到：

- median-seed micro F1 至少 `0.60`；
- 相对 direct median delta 至少 `+0.05`，且三个 seed 全为正；
- body/head、worst fresh parent delta 均不为负；
- worst fresh parent absolute median-seed F1 至少 `0.40`；
- recall delta 至少 `-0.02`，FPR delta 至多 `+0.02`；
- source-height macro overall/risk/safe/near-boundary MAE 分别不超过
  `0.10/0.15/0.15/0.10 m`；
- target UNKNOWN→SAFE 违规为零。

raw clearance 超出 `[-0.5,+1.0] m` 的比例必须完整报告，但只作线性 head
稳定性诊断，不是成功门，且不得用 prediction clamp 隐藏 MAE。

任一失败即
`SIGNED_CLEARANCE_CURRENT_CROSS_SOURCE_LEARNABILITY_NOT_SUPPORTED_STOP`，不得在同一
fresh cohort 上换 activation、loss、权重、checkpoint、threshold、margin、backbone、
source 或 metric。

即使全部通过，也只允许另行冻结 causal-transport 合同；不打开预留 official-test，
不直接支持 temporal value、主线、App、Android、生产或安全主张。

本文件只冻结 scientific design。implementation receipts、corpus、训练和 fresh
execution 均未授权，必须在后续执行合同中逐项 byte-bind。
