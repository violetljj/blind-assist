# HFTF Stage C D6 early-pair risk-veto canary

日期：2026-08-02

## 结论

终态：

> `EARLY_PAIR_RISK_VETO_SPECIFICITY_MICROSIGNAL_OBSERVED`
>
> `EARLY_PAIR_RISK_VETO_UTILITY_INCREMENT_NOT_SUPPORTED`
>
> `EARLY_PAIR_RISK_VETO_CROSS_SEED_EXPANSION_NOT_SUPPORTED`

相对冻结 directional-single reference，seed 17 三折的 future body/head FPR
全部下降，均值 `-0.000408`；known accuracy 三折精确不变；environment-macro
selection F1 三折均为极小正增量，均值 `+0.000103`。

但该效应不构成 utility increment：

- future body/head recall 为 0 正、2 零、1 负，均值 `-0.000549`；
- aggregate future body/head macro F1 均值 `-0.000151`；
- future body/head F1 均值 `-0.000164`；
- 15 个 environment unit 只有 3 正、10 零、2 负。

因此保留“结构化 veto 方向产生可测 specificity 微信号”的层级结论，但不扩展
seeds 29/43，不进入 outcome-unseen event proxy。

## 固定机制

本轮没有调整上一轮 recall-preservation loss 的权重。新增的唯一机制变量是把
pair 分支从共享 feature field 移到 risk-logit 输出层：

```text
candidate_risk_logit =
    frozen_reference_risk_logit
    + clamp_max(pair_risk_delta, 0)

candidate_known_logit = frozen_reference_known_logit
```

因此 candidate 在结构上不能提高任何 risk logit，也不能改变 known logit。它
只能选择性执行 veto。训练仍使用固定权重 `1.0` 的 teacher-positive future
body/head recall-preservation 项；其余数据、三折、seed、epoch、LR 和选择规则与
前两轮一致。

## 工程不变量

- veto head 的 weight 和 bias 均零初始化；
- epoch 0 candidate risk/known logits 与 reference 精确相等；
- 任意 pair 输出下，candidate risk logit 都不大于 reference；
- candidate known logits 与 reference 精确相等；
- 三折 checkpoint 中，reference 的全部已有 state tensors 逐 tensor 精确相等；
- candidate 只新增 `early_pair_stem.*` 和 `early_pair_veto_head.*`；
- 三折 trainable parameter count 均为 22,142；
- 三折 constraint loss 均为有限非零值。

## 相对 directional-single reference

| 指标 | 三折均值增量 | 中位数 | 正/零/负 |
|---|---:|---:|---:|
| environment-macro selection F1 | +0.000103 | +0.000086 | 3/0/0 |
| aggregate future body/head macro F1 | -0.000151 | +0.000083 | 2/0/1 |
| future body/head F1 | -0.000164 | +0.000078 | 2/0/1 |
| future body/head AUROC | -0.000057 | +0.000009 | 2/0/1 |
| future body/head AP | -0.000016 | +0.000005 | 2/0/1 |
| future body/head recall | -0.000549 | 0 | 0/2/1 |
| future body/head FPR | -0.000408 | -0.000253 | 0/0/3 |
| known accuracy | 0 | 0 | 0/3/0 |

选择 epoch 为 fold 0/1/2 的 `16/2/7`。三折 selection 增量分别只有
`+0.000079`、`+0.000144`、`+0.000086`，应视为接近 reference 的微小变化，
不能以“三折全正”包装成稳定 utility。

## 解释边界

本轮回答了两个此前混在一起的问题：

1. feature-level recall-preservation 为何 FPR 恶化：共享 residual 可以通过抬高
   risk 换取 recall；
2. 若结构上禁止抬高 risk，pair 是否能减少误激活：可以，但当前收益极小，并且
   仍会误 veto 少量 positive cells。

允许保留：

> 在当前 TartanGround Development 三折中，output-level one-sided pair veto
> 能以 known 完全不变为条件，产生方向一致但极小的 FPR 改善。

不允许声称：

- critical recall 已完全守住；
- aggregate field utility 已提高；
- event-level false alerts 已减少；
- outcome-unseen 或真实来源收益成立。

## 下一研究变量

不搜索 veto threshold、clamp 强度、loss 权重或更大 stem。下一步先隔离表示能力：

> 在冻结 baseline-active future body/head cells 上，把问题改写为
> `true alert` 对 `false alert` 的 pair-based veto-eligibility ranking。

先评价 environment-held-out AUROC/AP 和跨 fold 符号，而不把 classifier 接入
系统输出。只有 pair 表征能稳定区分 false-active cells，才冻结一个独立的
selective-veto 执行协议；否则关闭这条局部机制，而不是继续调控制面旋钮。

## 复现

单 fold：

```powershell
E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/research/hftf/train_stage_c_d5_tartanground_development_student.py `
  --samples artifacts.local/evidence/hftf/stage-c-d5-tartanground-cross-environment-v1/fold-0/samples.jsonl `
  --pretrained artifacts.local/models/hftf/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth `
  --output-root artifacts.local/evidence/hftf/stage-c-d6-early-pair-risk-veto-canary-v0/seed-17/fold-0 `
  --arm history --architecture directional `
  --temporal-mode early_pair_risk_veto `
  --optimization-mode early_pair_only `
  --pair-constraint-mode future_body_head_recall `
  --selection-mode environment_macro `
  --initial-checkpoint artifacts.local/evidence/hftf/stage-c-d5-tartanground-cross-environment-v1/training/fold-0/directional-single-seed17/checkpoint.pt `
  --seed 17 --epochs 20 --head-lr 3e-4
```

三折汇总：

```powershell
E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/research/hftf/summarize_stage_c_d6_early_pair_structured_field_canary.py `
  --candidate-root artifacts.local/evidence/hftf/stage-c-d6-early-pair-risk-veto-canary-v0 `
  --seeds 17 `
  --output artifacts.local/evidence/hftf/stage-c-d6-early-pair-risk-veto-canary-v0/summary.json
```
