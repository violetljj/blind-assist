# HFTF Stage C D6 veto-eligibility ranking

日期：2026-08-02

## 结论

本轮得到两个不同层级的终态：

> `CONFIDENCE_ANCHORED_PAIR_VETO_RANKING_INCREMENT_SUPPORTED_IN_DEVELOPMENT`
>
> `CONFIDENCE_ANCHORED_PAIR_VETO_RANKING_INCREMENT_SUPPORTED_ON_OUTCOME_UNSEEN_SYNTHETIC_DEVELOPMENT`
>
> `CONSERVATIVE_PAIR_VETO_EXECUTION_UTILITY_NOT_SUPPORTED`

early-pair 表征能够在 frozen baseline 已激活的 critical cells 中，为
`true alert / false alert` 提供超过 baseline confidence 的增量排序信息。
该结果在 outcome-unseen synthetic Development 上比原 Development 更一致。

但使用“不误 veto 任何训练 true alert”的保守执行规则时，false-alert coverage
只有约 `0.31%`，与 outcome true-alert veto rate 约 `0.31%` 同量级。排序信号
尚不能转化为有意义的执行 utility，因此不接入系统输出、不冻结产品阈值。

## 科学问题

前两轮分别发现：

- feature-level recall preservation 可提高 recall，但同时抬高 FPR；
- output-level hard veto 可稳定微降 FPR，但收益接近零。

本轮不再直接优化系统输出，而先回答：

> 在 baseline 已经认为需要激活的 future body/head cells 中，early RGB pair
> 是否能区分其中的 true alert 与 false alert？

eligible cell 固定为：

```text
teacher known
AND future horizon
AND body/head height
AND frozen baseline risk probability >= 0.5
AND frozen baseline known probability >= 0.5
```

正类是 teacher-negative false alert。评价使用 AUROC/AP，不选择执行 threshold。

## 两个排序臂

### Pair-only

22,142 个参数的 early-pair stem 直接预测 false-alert score。seed 17 三折
environment-macro AUROC 为 `0.6704 / 0.5571 / 0.6399`，说明 pair 本身有信号；
但相对 `1 - baseline risk probability` 对照的 macro AUROC 增量为
`-0.0074 / -0.0145 / -0.0147`。

因此 pair-only 没有证明增量价值。

### Confidence-anchored pair residual

candidate score logit 固定为：

```text
-frozen_baseline_risk_logit + pair_residual_logit
```

pair head 无 bias 且零初始化，所以 epoch 0 与 `1 - baseline risk probability`
排序精确等价。pair 只学习对已有 confidence 排序的增量。

不搜索模型容量、LR、epoch 上限、threshold 或 comparator。

## Development 九单元

3 seeds × 3 folds，选择规则为 environment-macro AUROC。

| 指标 | 均值 | 中位数 | 正/零/负 |
|---|---:|---:|---:|
| candidate macro AUROC | 0.673656 | 0.718372 | 9/0/0 |
| macro AUROC 增量 | +0.062951 | +0.066259 | 8/0/1 |
| macro AP 增量 | +0.068730 | +0.073150 | 7/0/2 |
| pooled AUROC 增量 | +0.045444 | -0.022854 | 4/0/5 |
| pooled AP 增量 | +0.075126 | +0.001520 | 5/0/4 |

45 个 environment units：

- AUROC 增量：29 正、0 零、16 负；
- AP 增量：30 正、0 零、15 负。

这支持 environment-macro ranking increment，但 pooled 结果的符号异质性禁止把
它解释为统一的全域执行收益。

## Outcome-unseen synthetic Development

同一 9 个冻结 checkpoint 在 6 个 outcome-unseen environments 上评价。一个
`OldTownFall` unit 因 eligible cells 缺少正/负两类而标为 AUROC
`NOT_EVALUABLE`；其余单元保留，不关闭整个队列。

| 指标 | 均值 | 中位数 | 正/零/负 |
|---|---:|---:|---:|
| macro AUROC 增量 | +0.038723 | +0.029043 | 8/0/1 |
| macro AP 增量 | +0.027969 | +0.024772 | 9/0/0 |
| pooled AUROC 增量 | +0.064297 | +0.056823 | 9/0/0 |
| pooled AP 增量 | +0.036699 | +0.036338 | 9/0/0 |

53 个可评价 environment units：

- AUROC 增量：34 正、1 零、18 负；
- AP 增量：37 正、1 零、15 负。

该结果支持 outcome-unseen ranking increment，但仍是 synthetic
Development 表示证据，不是 event utility 或真实来源证据。

## 保守执行 canary

为了避免结果驱动 threshold 搜索，每个 unit 只用自己的 unaugmented training
records 校准：

```text
threshold =
    next representable score above
    max(score on all eligible training true alerts)
```

因此每个 unit 的 training true-alert veto count 严格为 0。threshold 不读取
outcome-unseen labels。

Outcome-unseen 九单元：

| 指标 | 均值 | 中位数 | 正/零/负 |
|---|---:|---:|---:|
| false-alert veto coverage | 0.003144 | 0.000741 | 7/2/0 |
| true-alert veto rate | 0.003146 | 0.001238 | 5/4/0 |
| net correct veto cells | 3.8889 | 0 | 4/5/0 |

54 个环境中，false-alert coverage 的中位数为 0；53 个含 true alerts 的环境中，
true-alert veto rate 的中位数也为 0。该协议几乎不动作，不能称为 utility 成功。

## 工程修复与科学边界

两次 engineering-invalid 均修复后原样重跑，没有烧毁科学单元：

1. outcome evaluator 起初要求每个环境 AUROC 都有定义；改为逐环境
   `NOT_EVALUABLE`，macro 只在 candidate/comparator 同时可评价的同一集合计算；
2. threshold 起初用 float64 `nextafter`，与 float32 score 比较时回落到原最大值；
   改用 score dtype 的下一个可表示值，训练 true-alert veto invariant 随即通过。

允许保留：

> Confidence-anchored early-pair residual 在当前 synthetic Development 中对
> baseline false-alert eligibility 提供可复现的 threshold-free ranking
> increment。

不允许声称：

- 已获得有效 selective-veto 执行策略；
- event-level false alerts 已减少；
- critical recall 已受系统级保护；
- 真实公开来源或手机端收益成立；
- 可以通过放宽 harm threshold 把当前结果晋级。

## 路线处置

当前 classifier 不接 system output。保留为两个候选用途：

1. hard-negative ranking：优先发现 baseline-active 的疑似 false alerts；
2. active review/annotation：为未来真实关系数据排序审核队列。

若继续执行型 veto，必须使用新的、未被本轮 ranking 或 execution 评价消费的
fresh cohort，并在结果前冻结 harm budget；不能在当前 outcome-unseen 数据上
调 threshold。

## 复现

训练单元：

```powershell
E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/research/hftf/train_stage_c_d6_veto_eligibility_ranking.py `
  --samples artifacts.local/evidence/hftf/stage-c-d5-tartanground-cross-environment-v1/fold-0/samples.jsonl `
  --pretrained artifacts.local/models/hftf/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth `
  --reference-checkpoint artifacts.local/evidence/hftf/stage-c-d5-tartanground-cross-environment-v1/training/fold-0/directional-single-seed17/checkpoint.pt `
  --output-root artifacts.local/evidence/hftf/stage-c-d6-veto-eligibility-confidence-residual-canary-v0/seed-17/fold-0 `
  --seed 17 --epochs 20 --learning-rate 3e-4 `
  --ranking-mode confidence_residual
```

汇总与评价：

```powershell
E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/research/hftf/summarize_stage_c_d6_veto_eligibility_ranking.py `
  --output artifacts.local/evidence/hftf/stage-c-d6-veto-eligibility-confidence-residual-canary-v0/development-summary.json

E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/research/hftf/evaluate_stage_c_d6_veto_eligibility_outcome_unseen.py `
  --output artifacts.local/evidence/hftf/stage-c-d6-veto-eligibility-confidence-residual-canary-v0/outcome-unseen-summary.json

E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/research/hftf/evaluate_stage_c_d6_conservative_veto_execution.py `
  --output artifacts.local/evidence/hftf/stage-c-d6-veto-eligibility-confidence-residual-canary-v0/conservative-veto-execution-summary.json
```
