# HFTF Stage C D6 early-pair recall-preserving field canary

日期：2026-08-02

## 结论

终态：

> `EARLY_PAIR_RECALL_PRESERVATION_IMPROVES_RECALL_BUT_NOT_SPECIFICITY`
>
> `EARLY_PAIR_RECALL_PRESERVING_FIELD_CROSS_SEED_EXPANSION_NOT_SUPPORTED`

固定权重的单向 recall-preservation loss 达到了它的局部目的：相对冻结的
directional-single reference，seed 17 三折的 future body/head recall 全部上升，
平均增量为 `+0.101213`。但 false-positive rate 也在三折全部上升，平均恶化
`+0.090506`。因此它没有同时保留上一轮在 outcome-unseen 上观察到的 specificity
信号，不进入 seeds 29/43，也不运行 outcome-unseen event proxy。

这不是协议失败。训练、选择、产物和冻结参数核验均完成；终止原因是固定科学变量
未满足预先声明的 recall-and-specificity 联合目标。

## 固定变量

除下述目标项外，沿用 early-pair structured-field canary：

- 同一 TartanGround 三折和 seed 17；
- 同一 directional-single 初始 checkpoint；
- 同一 31,560 个可训练 early-pair 参数；
- 同一 20 epochs、`3e-4` head LR；
- 同一 environment-macro future body/head F1 选择规则；
- 不搜索阈值、epoch 上限、LR、模型容量或 loss 权重。

新增的唯一科学变量是：

```text
base loss
+ 1.0 * mean(
    relu(reference_risk_logit - candidate_risk_logit)
    on teacher-positive future body/head cells
  )
```

reference 在 candidate pair residual 之前取得并停止梯度。candidate 与 reference
在训练态共享同一 dropout mask，避免把随机 dropout 差异误当成 pair-induced
logit decrease。

## 工程不变量

- 零初始化时，candidate 和 reference 的 risk/known logits 在训练态共享 dropout
  后精确相等；
- current horizon、foot height、teacher-negative 和 unknown cells 不进入约束；
- 三折 checkpoint 中，初始 checkpoint 已有的全部参数均逐 tensor 精确相等；
- candidate 只新增 `early_pair_stem.*` 和 `early_pair_output.*` 16 个 state keys；
- 三折报告的 trainable parameter count 均为 31,560；
- 三折 constraint loss 均为非零有限值。

首次前台执行被工具短超时中断，未写出 report，因此作为 engineering-invalid
execution repair 重新运行同一 fold；没有关闭 fold、改变配置或烧毁 cohort。

## 相对 directional-single reference

| 指标 | 三折均值增量 | 中位数 | 正/零/负 |
|---|---:|---:|---:|
| environment-macro selection F1 | +0.031171 | +0.034322 | 3/0/0 |
| aggregate future body/head macro F1 | +0.013296 | +0.015960 | 3/0/0 |
| future body/head F1 | +0.014724 | +0.016221 | 3/0/0 |
| future body/head AUROC | +0.003176 | +0.001583 | 2/0/1 |
| future body/head AP | -0.007953 | -0.005338 | 1/0/2 |
| future body/head recall | +0.101213 | +0.096003 | 3/0/0 |
| future body/head FPR | +0.090506 | +0.106983 | 3/0/0 |

15 个 environment unit 中，selection F1 为 12 正、0 零、3 负，均值
`+0.031171`。这个结果支持“recall-preservation 产生稳定 Development
selection signal”，但不能覆盖 FPR 的一致恶化。

## 相对无约束 early-pair

约束版相对上一轮无约束 early-pair 的 future body/head recall 在三折分别增加
`+0.089823`、`+0.067405`、`+0.052605`；FPR 同时分别恶化
`+0.037962`、`+0.082988`、`+0.032806`。

这排除了“约束没有实际生效”的解释，也显示当前 feature-level residual 可以通过
整体抬高 risk 换取 recall，而不是学习到 selective veto。

## 研究边界与下一变量

允许保留的主张：

> 在当前 TartanGround Development 三折中，单向 teacher-positive logit
> preservation 能稳定提高 recall 和 selection F1，但不能同时降低 false
> activation。

不允许声称：

- event utility 已提高；
- outcome-unseen specificity 已保留；
- 真实关系数据或公开真实来源已改善；
- 该 loss 可通过调权重、阈值或 seed 被“救回”。

不继续搜索本 loss。下一候选应改变机制而不是旋钮：

> 将 pair 分支放到输出 logit 层，只允许产生非正 risk delta，并保持 known logits
> 完全等于冻结 reference。这样 candidate 结构上只能执行 selective veto，不能靠
> 抬高全局 risk 购买 recall。

该候选必须从零初始化与 reference 精确等价开始；若 seed 17 三折不能在不降低
critical recall 的前提下减少 false activation，则同样以科学负结论终止。

## 复现

单 fold：

```powershell
E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/research/hftf/train_stage_c_d5_tartanground_development_student.py `
  --samples artifacts.local/evidence/hftf/stage-c-d5-tartanground-cross-environment-v1/fold-0/samples.jsonl `
  --pretrained artifacts.local/models/hftf/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth `
  --output-root artifacts.local/evidence/hftf/stage-c-d6-early-pair-recall-preserving-field-canary-v0/seed-17/fold-0 `
  --arm history --architecture directional `
  --temporal-mode early_pair --optimization-mode early_pair_only `
  --pair-constraint-mode future_body_head_recall `
  --selection-mode environment_macro `
  --initial-checkpoint artifacts.local/evidence/hftf/stage-c-d5-tartanground-cross-environment-v1/training/fold-0/directional-single-seed17/checkpoint.pt `
  --seed 17 --epochs 20 --head-lr 3e-4
```

三折汇总：

```powershell
E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/research/hftf/summarize_stage_c_d6_early_pair_structured_field_canary.py `
  --candidate-root artifacts.local/evidence/hftf/stage-c-d6-early-pair-recall-preserving-field-canary-v0 `
  --seeds 17 `
  --output artifacts.local/evidence/hftf/stage-c-d6-early-pair-recall-preserving-field-canary-v0/summary.json
```
