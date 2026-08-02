# HFTF Stage C D6 early-pair structured-field canary

日期：2026-08-02

## 结论

把 early RGB-pair interaction 从单一 actionability 二分类接回结构化 HFTF
future-risk field 后，得到两个必须同时保留的结果：

1. 在 3 seeds × 3 environment-held-out folds 上，environment-macro future
   body/head F1 的 selected delta 为 6 正、3 个精确 epoch-0 中性、0 负，
   mean/median `+0.00565/+0.00396`：
   `EARLY_PAIR_STRUCTURED_FIELD_SELECTION_SIGNAL_SUPPORTED_IN_DEVELOPMENT`。
2. 该信号没有形成稳定事件效用。dev event proxy 的 recall/FPR mean delta 为
   `+0.00761/+0.04808`；outcome-unseen transfer 为
   `-0.00919/-0.00940`：
   `EARLY_PAIR_STRUCTURED_FIELD_EVENT_UTILITY_INCREMENT_NOT_SUPPORTED`。

在 6 个 outcome-unseen synthetic environments 上仍保留一层更窄的正信号：

- future body/head AUROC mean `+0.00757`；
- average precision mean `+0.01065`；
- false-positive rate mean `-0.01387`；
- false-alert event count 6 个实际训练单元全部不增，九单元 mean `-2.0`。

但 macro F1 mean `-0.00135`、event recall mean `-0.00919`。因此当前支持：

`EARLY_PAIR_OUTCOME_UNSEEN_RANKING_AND_SPECIFICITY_SIGNAL_OBSERVED_IN_DEVELOPMENT`

不支持把它称为整体 field、event、主线或系统增量。

## 科学变量

reference 是每个 `seed × fold` 已有的 directional-single checkpoint。candidate
不重训 backbone、field head 或 decision kernel，只新增一个零初始化的 early-pair
residual：

```text
过去四帧均值 baseline
current RGB
signed(current - baseline)
abs(current - baseline)
             ↓
12-channel lightweight pair stem
             ↓
128 × 8 × 14 pair feature
             ↓ bilinear resize
zero-initialized 1×1 residual
             +
原 directional-single 128 × 4 × 7 feature
             ↓
原 directional HFTF field head
```

固定条件：

- trainable parameters：31,560；
- reference 其余参数全部冻结；
- input：真实 5-frame history；
- structured target：
  `current/near/far × foot/body/head × 6 directions × 6 distances`；
- 20 epochs，AdamW，LR `3e-4`；
- selection：与 reference 相同的 environment-macro future body/head F1；
- seeds：17、29、43；
- folds：0、1、2；
- 不搜索结构、学习率、threshold 或 decision policy。

零初始化后，真实 fold-0 smoke 的 aggregate metrics、逐环境 metrics 和 selection
score 与对应 directional-single reference 完全一致。candidate 可以选择 epoch 0，
所以训练退化不会伪装成采用后的增量。

## 工程无效与修复

两处控制面错误均在读取科学结论前修复，未烧毁数据：

1. 初版 epoch-0 为提速把 repeated-current Conv3D 化简成 kernel sum，因浮点运算
   顺序使 selection F1 产生约 `-8.4e-5` 差异。恢复与 reference 完全相同的
   repeated-current 路径后，逐字段完全一致。
2. 初版 event wrapper 复用旧 predictor 的默认 `single` input，导致 early-pair
   没看到真实 history。predictor 增加显式 `input_arm`，旧调用仍默认 `single`，
   candidate 用 `history` 全量重推。错误输出不进入结论。

这两项都是工程 invalid，不是科学负结果。

## Environment-held-out Development

九单元 selected epoch 为：

| seed | fold 0 | fold 1 | fold 2 |
|---:|---:|---:|---:|
| 17 | 18 | 3 | 0 |
| 29 | 8 | 1 | 1 |
| 43 | 0 | 11 | 0 |

主要 delta：

| 指标 | mean | median | 正/中/负 |
|---|---:|---:|---:|
| environment-macro future body/head F1 | +0.00565 | +0.00396 | 6/3/0 |
| aggregate macro F1 | +0.00232 | +0.00161 | 5/3/1 |
| aggregate micro F1 | +0.00424 | +0.00089 | 5/3/1 |
| AUROC | +0.00292 | 0 | 4/3/2 |
| average precision | -0.00348 | 0 | 4/3/2 |
| recall | +0.01744 | +0.00151 | 6/3/0 |
| false-positive rate | +0.01181 | 0 | 3/3/3 |

45 个 `environment × seed` 单元的 macro F1 delta 为
`22 positive / 15 zero / 8 negative`，mean `+0.00565`。这建立小幅 cell-level
selection signal，但 threshold-free ranking 混合，并且 recall 增量伴随平均 FPR
恶化。

## Fixed synthetic event proxy

使用既有 `height_spatiotemporal_selective_v2`，不修改 threshold、确认长度或
空间支持：

| event 指标 | mean delta | 正/中/负 |
|---|---:|---:|
| event recall | +0.00761 | 3/3/3 |
| false-active lane-frame rate | +0.04808 | 3/3/3 |
| hit events | +1.33 | 3/3/3 |
| false-alert events | +0.11 | 3/3/3 |
| clearance rate | -0.08571 | 1/4/4 |

没有形成 recall、false activation 与 clearance 的一致联合改善。

## Outcome-unseen transfer

六个先前 outcome-unseen environments、198 samples 上复用九个 selected
checkpoints，不再训练：

| cell 指标 | mean delta | 正/中/负 |
|---|---:|---:|
| macro F1 | -0.00135 | 2/3/4 |
| micro F1 | +0.00348 | 4/3/2 |
| AUROC | +0.00757 | 4/3/2 |
| average precision | +0.01065 | 4/3/2 |
| recall | -0.00514 | 2/3/4 |
| false-positive rate | -0.01387 | 1/3/5 |

固定 event proxy：

| event 指标 | mean delta | 正/中/负 |
|---|---:|---:|
| event recall | -0.00919 | 2/3/4 |
| false-active lane-frame rate | -0.00940 | 1/3/5 |
| false-alert events | -2.00 | 0/3/6 |
| clearance rate | +0.02222 | 3/5/1 |

结果表明 early-pair residual 在新合成环境上更倾向于降低误激活和改善排序，但尚未
守住 critical recall。它不是“无信号”，也不是可晋级 event utility。

## 下一研究变量

不继续搜索本模型的 threshold、epoch、LR 或更大 pair stem。下一候选应利用已经
观察到的 specificity signal，同时显式保护 baseline recall：

> 冻结 directional reference，只允许 early-pair 产生 selective residual/veto；
> 对 teacher-positive future body/head cells 加入单向 recall-preservation
> constraint，对完整 teacher-negative cells学习抑制误激活。

这改变训练目标，而不是靠 post-hoc threshold 把 trade-off 包装成成功。只有它在
environment-held-out 与 outcome-unseen event proxy 上同时守住 recall 并减少
false alerts，才值得进入真实关系数据。

## 复现

单个训练单元：

```powershell
E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/research/hftf/train_stage_c_d5_tartanground_development_student.py `
  --samples artifacts.local/evidence/hftf/stage-c-d5-tartanground-cross-environment-v1/fold-0/samples.jsonl `
  --pretrained artifacts.local/models/hftf/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth `
  --output-root artifacts.local/evidence/hftf/stage-c-d6-early-pair-structured-field-canary-v0/seed-17/fold-0 `
  --arm history --architecture directional `
  --temporal-mode early_pair --optimization-mode early_pair_only `
  --selection-mode environment_macro `
  --initial-checkpoint artifacts.local/evidence/hftf/stage-c-d5-tartanground-cross-environment-v1/training/fold-0/directional-single-seed17/checkpoint.pt `
  --seed 17 --epochs 20 --head-lr 3e-4
```

汇总与评价：

```powershell
E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/research/hftf/summarize_stage_c_d6_early_pair_structured_field_canary.py `
  --output artifacts.local/evidence/hftf/stage-c-d6-early-pair-structured-field-canary-v0/summary.json

E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/research/hftf/evaluate_stage_c_d6_early_pair_structured_field_event_proxy.py `
  --output artifacts.local/evidence/hftf/stage-c-d6-early-pair-structured-field-canary-v0/event-proxy-history-summary.json

E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/research/hftf/evaluate_stage_c_d6_early_pair_outcome_unseen_transfer.py `
  --output artifacts.local/evidence/hftf/stage-c-d6-early-pair-structured-field-canary-v0/outcome-unseen-summary.json
```
