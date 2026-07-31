# dual_loop_segmentation_failure_atlas

状态：development

## 研究问题与版本

`DUAL_LOOP_SEGMENTATION_FAILURE_ATLAS_AND_RESIDUAL_LABELABILITY_R0` 在现有
200-frame consumed rehearsal 上回答两个问题：当前 false activation 由哪些可行动机制
构成；简单空间、因果时序或单一置信门是否有信息增量，以及 pixel-level residual 是否
可标。当前 evidence instance 为 `PILOT_200_R1_CONSUMED_FRESH_V1`，stage 为
`DEVELOPMENT_STANDARD`，不产生 Confirmation、产品或安全结论。

组件 false activation 固定为“候选类别未与同类别 pixel residual truth 相交”；任意
hazard 相交与 YOLO-overlapped hazard 另列字段，避免把类别混淆或 attribution uncertainty
悄悄改写成纯背景误激活。时间 persistence 的单位是相邻 materialized observation，
不是原视频所有未抽取 frame。

## 稳定 Interface

```powershell
python -m scripts.research.dual_loop_segmentation_failure_atlas.atlas `
  --repo-root . `
  --config configs/dual_loop_segmentation_failure_atlas_r0/pilot.json `
  --frames artifacts.local/evidence/dual-loop-segmentation-r2-p0/rehearsal-ddrnet-baseline-v2/frames.jsonl `
  --components artifacts.local/evidence/dual-loop-segmentation-r2-p0/rehearsal-ddrnet-baseline-v2/components.jsonl `
  --view-root artifacts.local/evidence/dual-loop-segmentation-r2-p0/canonical-view `
  --yolo-trace artifacts.local/evidence/dual-loop-segmentation-model-selection-r1/fresh_holdout/yolo_trace.jsonl `
  --output-root artifacts.local/evidence/dual-loop-segmentation-failure-atlas-r0/pilot-200
```

输入必须是 200-frame `r1_consumed_fresh` rehearsal、对应 canonical view 和冻结 YOLO
trace。packed mask、component ledger、truth identity 或 source/frame pairing 不一致时停止。

## 输出

仅写入 `artifacts.local/evidence/dual-loop-segmentation-failure-atlas-r0/pilot-200/`：

- `atlas_components.jsonl`：组件级真值类别、空间、时序、归因与错误机制；
- `frame_summary.jsonl`、`session_summary.json`：frame/session 聚合；
- `gating_probes.json`：非组合式空间、因果时序和置信 probe；
- `residual_labelability.json`：三态 residual 可标性与输入 availability；
- `result.json`：二维结论与是否值得定向扩展。

## 安全边界

只读使用 consumed Development 数据。当前没有 instance-level attribution truth，故
YOLO-overlapped hazard 像素记为 `ATTRIBUTION_UNCERTAIN`，不得伪装为
`A_EFFECTIVELY_COVERED`。不训练、不访问 fresh holdout、不接 Android、risk、TTS、振动或
默认 App。

## 停止条件

完成一次 200-frame pilot 即停止。只允许配置中的 4 个空间、3 个因果时序和 2 个置信
probe，禁止笛卡尔积搜索。只有至少一种可行动错误机制达到预声明面积占比并跨至少两个
session，才建议选择最多 12 个高信息 session 扩展；否则停止扩展。

## 假设与规则质疑

假设是当前失败并非单一碎片阈值问题，而包含可区分的大块混淆、边界膨胀或时间错误。
falsifier 是所有错误机制均低占比/单 session，且简单 probe 没有可重复 trade-off。probe
输出是 Development Pareto 描述，不选择“最优门”。

## 失败资产复用

旧 R1/R2-P0 终态保持不可变。Atlas 结果可作为诊断、训练任务设计、回归和 demo 选择依据，
不得恢复 consumed 数据的 unseen 身份。
