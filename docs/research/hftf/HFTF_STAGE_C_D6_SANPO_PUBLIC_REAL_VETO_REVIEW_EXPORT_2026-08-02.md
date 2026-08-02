# HFTF Stage C D6 SANPO public-real veto review export

日期：2026-08-02

## 结论

终态：

> `PUBLIC_REAL_MODEL_AWARE_VETO_REVIEW_QUEUE_MATERIALIZED`
>
> `PUBLIC_REAL_FALSE_ALERT_TRUTH_NOT_EVALUABLE`

已将 synthetic Development 上验证过的 confidence-anchored pair ranking
实际运行到一段本地 SANPO-Real RGB canary，并生成确定性的 hard-negative
review queue。

这使该支线从“训练结果”推进为可使用的公开真实数据 discovery 资产，但没有把
ranking score 转换成 false-alert label，也没有接入 D7 confirmation blind
review 或系统输出。

## 输入

来源：

- dataset：SANPO-Real；
- source session：`-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG`；
- camera/view：`chest/left`；
- 本地 RGB：60 帧；
- 时间：显式 15 FPS 导出的 relative nominal time；
- capture timestamp：不权威；
- truth authority：无。

输入 manifest：

```text
F:\ba-data\hftf-d7-public-real\manifests\
sanpo_media_manifest_d7-r1-sanpo-media-4s-nominal15-intrinsics-20260802.jsonl
```

窗口只使用连续 5 帧，共形成 56 个 history windows。

## 排序规则

使用 3 seeds × 3 folds 的 9 个 confidence-anchored veto-eligibility models。
每个模型独立计算：

- frozen baseline risk probability；
- frozen baseline known probability；
- early-pair false-eligibility score。

只保留 baseline `risk>=0.5 AND known>=0.5` 的 future body/head cells。
排序为：

1. active model vote count 降序；
2. active models 的 mean false-eligibility score 降序；
3. 稳定 window/cell key。

多数票定义为 `active_vote_count >= 5/9`。它只是 review eligibility，不是
false-alert 决策 threshold。

## 输出

首次导出：

```text
artifacts.local/evidence/hftf/
stage-c-d6-sanpo-public-real-veto-review-v0/
```

统计：

| 项目 | 数量 |
|---|---:|
| contiguous windows | 56 |
| models | 9 |
| at-least-one-active cells | 3,599 |
| majority-consensus active cells | 536 |
| windows with consensus cells | 56 |
| bounded review queue | 20 |

文件：

- `active_cell_ranking.jsonl`：全部至少一票 active cell；
- `review_queue.jsonl`：按窗口去重后的前 20 个 review items；
- `report.json`：输入、模型和输出 SHA-256 receipts。

确定性重复导出：

| 文件 | SHA-256 | 重复是否一致 |
|---|---|---|
| active_cell_ranking.jsonl | `2174ba1985580b1a39ccb2c2092cddbc53b83a8f9d7af35b910944d906c08350` | exact |
| review_queue.jsonl | `e974e91db18a616a45035c121b7baabb2d786da84882a186359f01cf61c0ee19` | exact |

## Sanity check

最高排名窗口 anchor frame 12：

- active vote：9/9；
- best score：`0.6872`；
- cell：near/head，grid row 1、column 2；
- mean baseline risk：`0.6591`。

可视检查显示该序列包含脚手架、强阴影、高对比反光和路侧车辆，属于值得判断的
domain-shift case。anchor frame 40 可见远处行人，说明队列中也会混入可能的
true alerts。

因此：

> review queue 有实际审核价值，但 score 不是自动 false label。

没有根据这次视觉检查回写、调序、删选或调整模型。

## 权限边界

所有输出都声明：

```text
role = MODEL_AWARE_DEVELOPMENT_DISCOVERY
truth_status = NOT_EVALUABLE
confirmation_review_eligible = false
system_action_authority = false
```

本轮没有修改：

- `F:\ba-data\hftf-d7-public-real\reviews\*.jsonl`；
- D7 RGB reviewer/adjudicator bundles；
- D7 candidate selection；
- app、threshold 或 alert policy。

这些候选若进入人工审核，必须使用独立的 model-aware counterexample review，
不能混入对模型输出盲化的 Confirmation reviewer agreement。

## 下一步

该资产已经可以用于：

1. 让 model-aware reviewer 标记
   `CONFIRMED_FALSE_ALERT / PLAUSIBLE_TRUE_ALERT / AMBIGUOUS`；
2. 将 confirmed false alerts 作为 Development hard negatives；
3. 在新的 source sessions 上重复 consensus ranking，检查跨 session 稳定性。

当前只有一个 60-frame session，因此不能计算真实 false-alert precision，也不能
声称 public-real ranking utility。下一强证据需要至少一个独立 SANPO session
的本地 RGB，或完成这 20 个候选的 model-aware review。

## 复现

```powershell
E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/research/hftf/export_stage_c_d6_veto_review_candidates.py `
  --media-manifest F:\ba-data\hftf-d7-public-real\manifests\sanpo_media_manifest_d7-r1-sanpo-media-4s-nominal15-intrinsics-20260802.jsonl `
  --candidate-root artifacts.local/evidence/hftf/stage-c-d6-veto-eligibility-confidence-residual-canary-v0 `
  --output-root artifacts.local/evidence/hftf/stage-c-d6-sanpo-public-real-veto-review-v0 `
  --top-k-windows 20
```
