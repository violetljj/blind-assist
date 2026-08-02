# HFTF Stage C D6 SANPO real veto transfer

日期：2026-08-02

## 结论

本轮得到两个有效但窄的科学负结果：

> `D6_CONSERVATIVE_REAL_HARD_NEGATIVE_EXECUTION_NOT_SUPPORTED`
>
> `D6_SYNTHETIC_VETO_RANKING_REAL_TRANSFER_NOT_SUPPORTED`

此前 confidence-anchored pair residual 在 TartanGround
Development/outcome-unseen 上建立的 false-alert ranking signal 保留，不因本轮撤销。
本轮关闭的是两个更后的假设：

1. “零训练真警 veto”阈值能在真实困难负例上产生有意义的清除；
2. synthetic-trained veto score 能在 SANPO 真实人审正负事件上继续优于简单的
   `1 - baseline risk` 排序。

两者都不支持。当前问题主要在 real-domain representation/transfer，不只是阈值过严。

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
stage-c-d6-sanpo-real-veto-ranking-v0/report.json
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
- 在同一表示上继续搜索 threshold、top-k、vote count 或确认长度。

未评价：

- 新表示或 real-domain calibration 是否可用；
- positive-event recall 下的任何新执行阈值；
- source-general real transfer；
- 主线、App、设备、生产或安全效用。

## 下一可证伪候选

只允许一个低成本 successor：在相同 30-session consumed Development 上做
source-session-held-out 的低容量 real-domain calibration ablation。

- arm B：只用 frozen baseline risk/known 与固定坐标；
- arm C：在 B 上增加当前 candidate score；
- 模型固定为 Logistic Regression；
- primary：held-out event-phase p95 AUROC 的 `C - B`；
- secondary：positive passed-minus-alertable paired direction；
- 不按结果搜索 threshold、feature family 或 fold。

若 C 不能稳定优于 B，则当前 veto representation 本身没有可复用的真实增量，停止该
exact pair-residual 路线，下一 representation 必须把 real-domain actionability
interaction 放回 backbone/field task，而不是继续校准输出。

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
  --output artifacts.local/evidence/hftf/stage-c-d6-sanpo-real-veto-ranking-v0/report.json
```
