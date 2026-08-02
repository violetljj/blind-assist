# HFTF Stage C D6 SANPO real veto transfer

日期：2026-08-02

## 结论

本轮及其预声明校准后继得到三个有效但窄的科学负结果：

> `D6_CONSERVATIVE_REAL_HARD_NEGATIVE_EXECUTION_NOT_SUPPORTED`
>
> `D6_SYNTHETIC_VETO_RANKING_REAL_TRANSFER_NOT_SUPPORTED`
>
> `D6_CANDIDATE_AWARE_REAL_CALIBRATION_INCREMENT_NOT_SUPPORTED`
>
> `D6_REAL_PHASE_SUPERVISED_EARLY_PAIR_CANARY_INCREMENT_NOT_SUPPORTED_STOP`

此前 confidence-anchored pair residual 在 TartanGround
Development/outcome-unseen 上建立的 false-alert ranking signal 保留，不因本轮撤销。
本轮关闭的是两个更后的假设：

1. “零训练真警 veto”阈值能在真实困难负例上产生有意义的清除；
2. synthetic-trained veto score 能在 SANPO 真实人审正负事件上继续优于简单的
   `1 - baseline risk` 排序；
3. 当前 candidate score 经固定低容量真实域校准后能提供稳定增量；
4. raw RGB early-pair field residual 经真实 phase 监督后能 source-session-heldout
   泛化。

四者都不支持。当前问题主要在 real-domain representation/transfer，不只是阈值过严。

## 模式与问题

模式为 `REVERSIBLE_EXPLORATION / DEVELOPMENT_STANDARD`。

本轮没有重新训练、搜索阈值或修改 App。它只回答：

- 已冻结 veto score 在独立盲审的 SANPO 困难负例上是否实际抑制 baseline cells；
- 连续 veto score 在 30-session 人审正负事件上能否把“应抑制”排在“应保留”之前。

工程、科学和主张边界分开记录。运行或 schema 错误可以修复重跑；有效指标负结果才关闭
对应科学假设。

## 盲审困难负例重合

D7 SANPO review 使用匿名 `d7sess-*`，而模型导出使用 raw session ID。通过 candidate
index、review bundle temporal manifest 和逐帧 SHA-256 join，确认当前 raw session：

```text
-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG
```

与 D7 已完成 review 的五个候选属于同一来源。四个候选满足：

- RGB reviewer A/B/C 均完成；
- 三人均看不到模型输出；
- 三人决定均为 `REJECT`；
- 每个候选 60 帧。

四段区间为 `0–59 / 30–89 / 60–119 / 90–149`，240 次帧观察去重为 150 个唯一 RGB，
重叠帧 SHA 全部一致。它们连续覆盖 146 个五帧窗口，且每个完整窗口都落在至少一个
3/3 reject 区间内。

这支持的是：

> `UNANIMOUS_MODEL_BLIND_RGB_ACTIONABLE_NEGATIVE`

它不提供 cell localization truth。D7 final adjudicator 因 authoritative timestamp
和 phase binding 缺失仍将系统事件终态写为 `NOT_EVALUABLE`。该 claim ceiling
不应覆盖三位 RGB reviewer 已经成立的 clip-level 科学观察，但也不能被反向升级为
精确 phase/cell truth。

物化结果：

```text
artifacts.local/evidence/hftf/
stage-c-d6-sanpo-blind-negative-media-v0/
```

## 真实困难负例执行

9 个 frozen confidence-residual models 使用此前固定的阈值：

```text
nextafter(max training true-alert score, +inf)
```

该规则在 synthetic train 上保证零 training true-alert veto。SANPO 盲审负例结果：

| 指标 | 结果 |
|---|---:|
| windows | 146 |
| full-field baseline active model-cells | 24,046 |
| full-field vetoed model-cells | 48（0.1996%） |
| central baseline active model-cells | 11,019 |
| central vetoed model-cells | 19（0.1724%） |
| central windows with any model veto | 17 / 146 |
| central windows with majority model veto | 0 / 146 |
| central fully cleared model-windows | 1 / 1,285 |
| central windows fully cleared by any model | 1 / 146 |
| central windows fully cleared by majority models | 0 / 146 |

这不是有用的真实 hard-negative execution effect。模型能够产生 ranking scores，
但冻结安全阈值只让一个模型清除一个中央窗口，没有多数模型复现。全方向 field
本身为 `0/1,308` cleared model-windows；该诊断不再被误写成 central-action
estimand。

输出：

```text
artifacts.local/evidence/hftf/
stage-c-d6-sanpo-blind-negative-veto-v2/
```

## 30-event 真实正负排序

随后直接使用已消费的 SANPO 30-session / 1,920-frame human-reviewed Development
cohort，不调模型、不调阈值。每个 parent event 内构造连续五帧 history：

- 负事件全部 scored cells：false-alert target；
- 正事件 alertable interval：true-alert / 应保留；
- 正事件 passed interval：false-alert / 应抑制；
- 只评价 central directions `2/3` 的 near/far × body/head baseline-active cells；
- comparator 固定为 `1 - frozen baseline risk probability`；
- event primary statistic 为 phase 内 cell score 的 p95，max 仅作 sensitivity。

结果：

| 指标 | candidate mean | comparator mean | candidate delta |
|---|---:|---:|---:|
| pooled cell AUROC | 0.509579 | 0.519693 | -0.010114 |
| event-phase p95 AUROC | 0.461310 | 0.571429 | -0.110119 |

跨 3 seeds × 3 folds：

- cell AUROC delta：`3 positive / 6 negative`；
- event p95 AUROC delta：`2 positive / 7 negative`；
- event p95 delta median：`-0.102679`；
- 最差/最好 event delta：`-0.339286 / +0.111607`。

正事件内部 passed score 应高于 alertable score。143 个可评价 model×event pairs 中，
candidate 只有 56 个方向正确；candidate 的 passed-minus-alertable p95 mean 为
`-0.030058`，方向相反。comparator mean 为 `-0.003127`。

这说明 exact synthetic-trained pair score 在真实事件层接近 chance，且总体弱于
简单 baseline confidence。不能靠重新选择阈值把它写成可用真实 veto。

输出：

```text
artifacts.local/evidence/hftf/
stage-c-d6-sanpo-real-veto-ranking-v1/report.json
```

## Source-session-held-out 真实域校准

按预声明的唯一低成本 successor，直接在同一 consumed Development cohort 上运行
5-fold source-session-held-out ablation。30 个 source sessions 按正/负 strata 内
SHA-256 顺序轮转分配，fold session counts 为 `7/6/6/6/5`；同一正事件的
alertable/passed phases 始终留在同一 fold。

- arm B：frozen baseline risk/known 的 mean/p95/max、eligible-cell count 与固定空间
  统计；
- arm C：B 再增加 candidate mean/p95/max；
- 两臂均固定 `StandardScaler + L2 LogisticRegression(C=1, liblinear,
  class_weight=balanced)`；
- 没有 feature、C、model、fold 或 threshold search。

跨 3 seeds × 3 folds 的 OOF event-phase 结果：

| 指标 | baseline-only mean | candidate-aware mean | C - B mean | C - B median | 正/负 |
|---|---:|---:|---:|---:|---:|
| AUROC | 0.504550 | 0.521592 | +0.017042 | -0.008333 | 3 / 6 |
| average precision | 0.694038 | 0.697522 | +0.003484 | -0.003541 | 3 / 6 |
| passed-minus-alertable mean increment | — | — | +0.011967 | -0.003022 | 3 / 6 |

AUROC 的正 mean 由少数单元驱动，最大单元为 `+0.127083`；6/9 单元为负，median 也为
负。candidate-aware arm 因而没有稳定增量。这是有效的 representation-level 科学
负结果，不是工程失效，也不能被 absolute AUROC 略高误写成稳定支持：

> `D6_CANDIDATE_AWARE_REAL_CALIBRATION_INCREMENT_NOT_SUPPORTED`

输出与 SHA-256：

```text
artifacts.local/evidence/hftf/
stage-c-d6-sanpo-real-veto-calibration-v0/report.json
87681af73f56987e3ceb83d74d461591cfb9b0a51f7f41e6033dd960a017dc2a
```

## Real-phase-supervised early-pair canary

校准失败后按预声明只改变 representation/training signal，不再拟合旧 candidate score。
固定 `seed17/model-fold0/heldout-fold0`：

- 30 source sessions 仍使用相同 hash-stratified 5-fold split；
- 23 train sessions / 1,016 scored windows，7 held-out sessions / 286 windows；
- frozen directional-single inverse-risk comparator；
- candidate 为 `EarlyPairStem` 加 zero-initialized direction-preserving field residual；
- 只在 central baseline-active cells 上，以人审 `negative_event /
  positive_alertable / positive_passed` phase label 训练；
- class 与 source-session-phase group 平衡；
- 固定 20 epochs、AdamW `3e-4`，只评价 final epoch；held-out 不用于选模；
- AUROC 与 AP delta 必须同时大于 0 才扩展 seed/fold。

初始 residual 的 maximum absolute value 为精确 `0`。训练 loss 从 `0.777228` 降到
`0.111119`，所以优化确实工作；结果失败在 source-session-held-out 泛化：

| held-out event-phase p95 | baseline | candidate | delta |
|---|---:|---:|---:|
| AUROC | 0.750000 | 0.416667 | -0.333333 |
| average precision | 0.638889 | 0.444444 | -0.194444 |

四个 held-out positive sessions 中，baseline 的 passed score 均高于 alertable
(`4/4`)；candidate 仅 `1/4`，paired delta mean 为 `-0.093330`。因此不扩展剩余
seed/fold：

> `D6_REAL_PHASE_SUPERVISED_EARLY_PAIR_CANARY_INCREMENT_NOT_SUPPORTED_STOP`

第一次 launcher 因 5 秒命令超时在写入输出前中断，随后按完全相同配置重跑。该工程
中断没有被当成科学负结果，也没有关闭 cohort。

输出与 SHA-256：

```text
artifacts.local/evidence/hftf/
stage-c-d6-sanpo-real-phase-early-pair-canary-v0/report.json
ec611e75d57c9c0f9e28c53db449aae4c1a9964602677f85e6dadb45984f072d
```

## 保留与关闭

保留：

- synthetic Development/outcome-unseen false-alert ranking signal；
- early pair interaction 在 synthetic structured field 上的 ranking/specificity signal；
- D7 三人盲审 clip-level negative evidence；
- 本轮 150-frame hard-negative corpus 作为 real-domain regression asset。

关闭：

- 当前 zero-training-true-alert conservative threshold 的 real hard-negative utility；
- 当前 synthetic-trained confidence-residual veto representation 的 SANPO real transfer；
- 当前 candidate score 经固定低容量真实域校准后的稳定增量；
- 当前 global phase label 直接监督的 exact early-pair field residual recipe；
- 在同一表示上继续搜索 threshold、top-k、vote count 或确认长度。

未评价：

- 新表示是否可用；
- positive-event recall 下的任何新执行阈值；
- source-general real transfer；
- 主线、App、设备、生产或安全效用。

## 下一可证伪候选

当前结果说明“给 raw RGB early-pair 更直接的 event phase supervision”仍不足以
source-heldout 泛化。下一候选不得再改 loss/head/threshold；必须改变输入表示本身。
优先检验单一低成本 motion-alignment variable：冻结 baseline 与 phase split，只把
raw frame difference 替换为 ego-motion-compensated correspondence/flow residual。
先做无训练 representation separability audit；只有 train-only 固定投影在 held-out
AUROC/AP 同时优于 raw-pair reference，才允许训练新的 field residual。缺少可靠
alignment 时记为 `NOT_EVALUABLE`，不把工程缺失写成算法负结果。

## 复现

```powershell
E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/run_research_tool.py hftf materialize_stage_c_d6_sanpo_blind_negative_media.py `
  --candidate-index F:\ba-data\hftf-d7-public-real\candidates\sanpo_candidate_index_d7-r1-sanpo-candidates-chest-left-rgb-depth-20260802.jsonl `
  --rgb-review F:\ba-data\hftf-d7-public-real\reviews\review_a.jsonl `
  --rgb-review F:\ba-data\hftf-d7-public-real\reviews\review_b.jsonl `
  --rgb-review F:\ba-data\hftf-d7-public-real\reviews\review_c.jsonl `
  --staging-root F:\ba-data\hftf-d7-public-real\reviews\input_bundles\d7-r1-sanpo-review-5-parallel-20260802\staging `
  --output-root artifacts.local/evidence/hftf/stage-c-d6-sanpo-blind-negative-media-v0

E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/run_research_tool.py hftf export_stage_c_d6_veto_review_candidates.py `
  --media-manifest artifacts.local/evidence/hftf/stage-c-d6-sanpo-blind-negative-media-v0/media_manifest.jsonl `
  --threshold-report artifacts.local/evidence/hftf/stage-c-d6-veto-eligibility-confidence-residual-canary-v0/conservative-veto-execution-summary.json `
  --output-root artifacts.local/evidence/hftf/stage-c-d6-sanpo-blind-negative-veto-v2 `
  --top-k-windows 30

E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/run_research_tool.py hftf evaluate_stage_c_d6_sanpo_real_veto_ranking.py `
  --output artifacts.local/evidence/hftf/stage-c-d6-sanpo-real-veto-ranking-v1/report.json

E:\codex-tools\projects\blindassist\toolchain\venvs\learned-component-validator-py311\Scripts\python.exe `
  scripts/run_research_tool.py hftf evaluate_stage_c_d6_sanpo_real_veto_calibration.py `
  --ranking-report artifacts.local/evidence/hftf/stage-c-d6-sanpo-real-veto-ranking-v1/report.json `
  --output artifacts.local/evidence/hftf/stage-c-d6-sanpo-real-veto-calibration-v0/report.json

E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/run_research_tool.py hftf run_stage_c_d6_sanpo_real_phase_early_pair_canary.py `
  --output-root artifacts.local/evidence/hftf/stage-c-d6-sanpo-real-phase-early-pair-canary-v0
```
