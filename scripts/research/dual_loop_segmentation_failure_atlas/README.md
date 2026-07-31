# dual_loop_segmentation_failure_atlas

状态：development；320-frame targeted expansion 已完成为 `GATING_PARTIAL`；
host-only visual sidecar R0 已可用

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

320-frame expansion 先分别以
`scripts.research.dual_loop_segmentation_r2_p0.run_rehearsal` 对 `dev` 与
`consumed_old_blind` 生成同协议 rehearsal，再通过重复参数一次聚合：

```powershell
python -m scripts.research.dual_loop_segmentation_failure_atlas.atlas `
  --repo-root . `
  --config configs/dual_loop_segmentation_failure_atlas_r0/expansion_320.json `
  --frames artifacts.local/evidence/dual-loop-segmentation-failure-atlas-r0/expansion-320-dev-rehearsal/frames.jsonl `
  --frames artifacts.local/evidence/dual-loop-segmentation-failure-atlas-r0/expansion-320-consumed-old-blind-rehearsal/frames.jsonl `
  --components artifacts.local/evidence/dual-loop-segmentation-failure-atlas-r0/expansion-320-dev-rehearsal/components.jsonl `
  --components artifacts.local/evidence/dual-loop-segmentation-failure-atlas-r0/expansion-320-consumed-old-blind-rehearsal/components.jsonl `
  --view-root artifacts.local/evidence/dual-loop-segmentation-r2-p0/canonical-view `
  --yolo-trace artifacts.local/evidence/dual-loop-segmentation-model-selection-r1/dev/yolo_trace.jsonl `
  --yolo-trace artifacts.local/evidence/dual-loop-segmentation-candidate-utility-r0/formal/yolo_trace.jsonl `
  --output-root artifacts.local/evidence/dual-loop-segmentation-failure-atlas-r0/expansion-320-v4
```

expansion config 固定 6 个 session/320 帧、五类机制、pilot reference hashes、排序与
source-dependence 判据、原 gating thresholds 及案例选择规则。不得借重复参数加入第七个
session；input contract 会拒绝额外或缺失 frame。

输入必须是 200-frame `r1_consumed_fresh` rehearsal、对应 canonical view 和冻结 YOLO
trace。packed mask、component ledger、truth identity 或 source/frame pairing 不一致时停止。

## 输出

仅写入 `artifacts.local/evidence/dual-loop-segmentation-failure-atlas-r0/pilot-200/`：

- `atlas_components.jsonl`：组件级真值类别、空间、时序、归因与错误机制；
- `frame_summary.jsonl`、`session_summary.json`：frame/session 聚合；
- `gating_probes.json`：非组合式空间、因果时序和置信 probe；
- `residual_labelability.json`：三态 residual 可标性与输入 availability；
- `result.json`：二维结论与是否值得定向扩展。

expansion 另写：

- `replication.json`：五类机制的 session/role 覆盖、pilot-vs-expansion Spearman、
  逐 session profile 与冻结决策树终态；
- `gate_cases.json`、`case_figures/`：每个非 baseline gate 的固定成功/失败案例选择与
  `DEVELOPMENT DIAGNOSTIC ONLY` 图。

## Visual-only sidecar R0

独立 host renderer 只消费已生成的 rehearsal、canonical manifest、冻结 YOLO trace 与
绑定 DDRNet INT8。它重新推理 softmax heatmap 仅用于显示，并从 rehearsal 的 packed mask
原样重建 visual candidate 与指定 Development probe：

```powershell
python -m scripts.research.dual_loop_segmentation_failure_atlas.visual_sidecar `
  --repo-root . `
  --config configs/dual_loop_visual_only_sidecar_r0/default.json `
  --frames artifacts.local/evidence/dual-loop-segmentation-failure-atlas-r0/expansion-320-consumed-old-blind-rehearsal/frames.jsonl `
  --components artifacts.local/evidence/dual-loop-segmentation-failure-atlas-r0/expansion-320-consumed-old-blind-rehearsal/components.jsonl `
  --manifest artifacts.local/evidence/dual-loop-segmentation-r2-p0/canonical-view/manifest.jsonl `
  --yolo-trace artifacts.local/evidence/dual-loop-segmentation-candidate-utility-r0/formal/yolo_trace.jsonl `
  --model artifacts.local/evidence/dual-loop-segmentation-model-selection-r1/ddrnet23_slim/int8/model_int8.tflite `
  --output-root artifacts.local/evidence/dual-loop-visual-only-sidecar-r0/example `
  --probe-id TEMPORAL:CAUSAL_2_OF_3 `
  --view-row-id <canonical-view-row-id>
```

每张图固定显示 YOLO known-object boxes、raw segmentation heatmap、visual candidates、
gate-passed candidates、rejected/abstained 与原因；顶部和底部固定写
`DEVELOPMENT VISUALIZATION ONLY / DOES NOT DRIVE ALERTS`。配置明确禁止
`confirmed danger`、`safe route`、`verified obstacle`，输出 manifest 固定
`authority=VISUAL_CANDIDATE_ONLY` 与 `drives_alerts=false`。它不连接 Android、risk、
feedback、TTS、振动或默认 App。

## 安全边界

Atlas 与 sidecar 只读使用 Development/consumed 数据。当前没有 instance-level
attribution truth，故
YOLO-overlapped hazard 像素记为 `ATTRIBUTION_UNCERTAIN`，不得伪装为
`A_EFFECTIVELY_COVERED`。不训练、不访问 fresh holdout、不接 Android、risk、TTS、振动或
默认 App。

## 停止条件

pilot 完成一次 200-frame 即停止；expansion 完成固定 320-frame 即停止。只允许配置中的
4 个空间、3 个因果时序和 2 个置信 probe，禁止笛卡尔积搜索。pilot 只有至少一种可行动
错误机制达到预声明面积占比并跨至少两个 session，才允许定向扩展；expansion 不再继续
补满 920 帧。当前终态为 `GATING_PARTIAL`，本 Module 不在同一结果上选择或组合 gate，
也不启动 residual-aware training。

## 假设与规则质疑

假设是当前失败并非单一碎片阈值问题，而包含可区分的大块混淆、边界膨胀或时间错误。
falsifier 是所有错误机制均低占比/单 session，且简单 probe 没有可重复 trade-off。probe
输出是 Development Pareto 描述，不选择“最优门”。

## 失败资产复用

旧 R1/R2-P0 终态保持不可变。Atlas 结果可作为诊断、训练任务设计、回归和 demo 选择依据，
不得恢复 consumed 数据的 unseen 身份。
