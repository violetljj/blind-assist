# Segmentation Failure Atlas 320-frame 定向扩展 R1 结果

状态：`TARGETED_EXPANSION_COMPLETE / MECHANISMS_REPRODUCED /
AGGREGATE_RANKING_STABLE / GATING_PARTIAL /
RESIDUAL_WEAKLY_LABELABLE / DEVELOPMENT_MECHANISM_DIAGNOSTIC_ONLY /
FINAL_CONFIRMATION_NOT_ACTIVATED`

日期：2026-08-01（Asia/Hong_Kong）

协议：`DUAL_LOOP_SEGMENTATION_FAILURE_ATLAS_AND_RESIDUAL_LABELABILITY_R0`

evidence instance：`TARGETED_EXPANSION_320_DEV_AND_CONSUMED_V1`

## 先给结论

固定的 6 个非 pilot session、320 帧已经全部执行。五类 pilot 机制均跨 dev 与
consumed old blind 两种角色复现，aggregate false-area 排序与 pilot 的 Spearman
相关性为 `0.90`；因此 pilot 不是四个原 session 才有的孤立结构。

但是“简单 gating 全部失败”没有完整复现。未改阈值、未新增门的前提下：

- `CAUSAL_2_OF_3` 达到既有 `PARTIAL`：FP 降低 `26.28%`，overall recall
  retention `79.30%`；
- `COMPONENT_MEDIAN_CONFIDENCE_GE_0_65` 也达到既有 `PARTIAL`：FP 降低
  `20.86%`，overall recall retention `85.28%`。

两者最差 session recall retention 只有 `47.29%` 与 `40.87%`，都没有达到
`SUFFICIENT`。因此终态严格按冻结决策树进入 `GATING_PARTIAL`，不是
`RESIDUAL_TASK_REFORMULATION_JUSTIFIED`：本轮不训练 residual-aware DDRNet，不选择
gate，不组合门，也不接提醒。下一条主线只能先冻结一个有限组合门的 Development
设计，并继续保留 visual/candidate 权限。

## 固定输入与执行完整性

| 角色 | session | 场景 | 帧数 |
|---|---|---|---:|
| consumed old blind | `5Llq…Kjb9` | center obstacle | 60 |
| consumed old blind | `i2jg…m4T3` | step/curb | 60 |
| dev | `CCG-…HlXk` | center obstacle | 50 |
| dev | `LRWT…5Ypp` | parallel boundary | 50 |
| dev | `lmkI…Oicv` | step/curb | 50 |
| dev | `yQ5I…13FO` | lateral pedestrian/e-bike | 50 |

两种角色均使用与 pilot 相同的 DDRNet INT8
`f76e280e…7a0f`、baseline postprocess `7e64e150…0cfc`、canonical evaluator 与冻结
YOLO trace。两次 rehearsal 分别产生 `200 / 4,089` 与 `120 / 2,625`
frame/component rows，并通过独立全量复算 validator。Atlas 汇总共消费
`320` 帧、`6,714` 个组件、`6` 个 session；没有访问 `r1_consumed_fresh`、train 或
synthetic role。

## Session-wise false positive

| session | FP pixels | FP area fraction | 组件数 | false components | false component area |
|---|---:|---:|---:|---:|---:|
| `5Llq…Kjb9` | 784,536 | 19.95% | 1,327 | 1,045 | 204,851 |
| `CCG-…HlXk` | 390,687 | 11.92% | 1,305 | 1,074 | 151,271 |
| `LRWT…5Ypp` | 422,714 | 12.90% | 824 | 422 | 24,794 |
| `i2jg…m4T3` | 300,693 | 7.65% | 1,298 | 865 | 96,570 |
| `lmkI…Oicv` | 62,121 | 1.90% | 778 | 374 | 88,082 |
| `yQ5I…13FO` | 349,627 | 10.67% | 1,182 | 685 | 81,712 |

总计 `4,465` 个同类 residual-truth false activation component，组件面积
`647,280` pixels。FP pixel area 与 false component area 是不同口径，不应相加或互相
替代。

## 五类机制复现与排序

| 机制 | expansion false-area share | session 覆盖 | 角色覆盖 |
|---|---:|---:|---:|
| `UPPER_FIELD_BACKGROUND_ACTIVATION_PROXY` | 47.43% | 6/6 | 2/2 |
| `TEMPORAL_FLICKER` | 43.77% | 6/6 | 2/2 |
| `STABLE_HIGH_CONFIDENCE_ERROR` | 21.37% | 5/6 | 2/2 |
| `YOLO_ATTRIBUTION_AMBIGUITY` | 11.83% | 6/6 | 2/2 |
| `SMALL_FRAGMENT_NOISE` | 8.24% | 6/6 | 2/2 |

pilot 排序为 `UPPER > FLICKER > YOLO_AMBIGUITY > STABLE_HIGH_CONFIDENCE >
SMALL_FRAGMENT`；expansion 只交换中间两项，Spearman `ρ=0.90`，高于冻结的
`0.60`。逐 session 对 pilot 的相关性为 `0.90 / 0.70 / 0.50 / 0.80 / 0.60 /
0.30`；没有两个 session 低于冻结的 `0.30` source-dependence 线，故不进入
`SOURCE_DEPENDENT_FAILURE_STRUCTURE`。

机制 tag 非互斥，share 不能求和。`SMALL_FRAGMENT_NOISE` 从 pilot 的 `10.41%` 降到
`8.24%`，但仍覆盖全部 6 个 session 与两种角色；这支持“机制继续出现”，不支持将其
称为 dominant mechanism。

## 原样复跑的 gating probes

baseline B 的 aggregate residual recall 为 `0.344974`，TP/FP pixels 为
`2,582,787 / 2,310,378`。没有新增或组合 probe。

| Probe | FP reduction | Overall recall retention | 最低 session retention | 最差 session | 判定 |
|---|---:|---:|---:|---|---|
| lower field | 79.27% | 23.67% | 1.53% | `CCG-…HlXk` | `INSUFFICIENT` |
| central body corridor | 90.20% | 22.07% | 0.00% | `5Llq…Kjb9` | `INSUFFICIENT` |
| upper/head band | 41.11% | 49.97% | 28.08% | `LRWT…5Ypp` | `INSUFFICIENT` |
| causal 2-of-3 | 26.28% | 79.30% | 47.29% | `lmkI…Oicv` | `PARTIAL` |
| causal 3 consecutive | 55.27% | 49.54% | 14.59% | `lmkI…Oicv` | `INSUFFICIENT` |
| median confidence ≥ 0.65 | 20.86% | 85.28% | 40.87% | `lmkI…Oicv` | `PARTIAL` |

`PARTIAL` 沿用 pilot 配置中的 overall 规则：FP reduction ≥ `0.10` 且 recall
retention ≥ `0.75`；它不等于 `SUFFICIENT`，也不代表可上线。`SUFFICIENT` 仍要求
overall retention ≥ `0.90`、minimum-session retention ≥ `0.80` 且 FP reduction ≥
`0.30`，本轮没有任何门满足。

## 成功与失败案例图

runner 按冻结规则为每个非 baseline probe 选择案例：成功例要求 frame-level recall
retention ≥ `0.90` 且确实移除 FP，再按移除 FP 数排序；失败例按最低 recall retention
排序。`lower field` 与 `central body corridor` 没有合法成功例，其余 probe 均生成成功
与失败图。图中固定区分 source、residual truth、baseline FP、gate-kept TP/FP 与
rejected pixels，并标注 `DEVELOPMENT DIAGNOSTIC ONLY`。

机器图与选择清单位于：

`artifacts.local/evidence/dual-loop-segmentation-failure-atlas-r0/expansion-320-v4/case_figures/`

`artifacts.local/evidence/dual-loop-segmentation-failure-atlas-r0/expansion-320-v4/gate_cases.json`

## Residual 可标注性是否改变

320 帧共有 `7,783,826` 个 canonical hazard pixels：

- pixel residual proxy：`7,486,907`，占 `96.19%`，仍为 `LABELABLE`；
- YOLO box 与 canonical hazard overlap：`296,919`，仍只能是
  `ATTRIBUTION_UNCERTAIN`；
- `A_EFFECTIVELY_COVERED`：仍缺 instance correspondence，不能合法计算。

所以 pilot 与 expansion 均为 pixel proxy `LABELABLE`、三态 attribution
`WEAKLY_LABELABLE`，状态未改变。高 residual share 不能解释成 YOLO 对真实 hazard
覆盖率很低；它仍只是 box-union 与 canonical semantic pixels 的可复算关系。

## 终态、边界与下一步

- Atlas expansion：`TARGETED_EXPANSION_COMPLETE`；
- mechanism：`REPRODUCED / AGGREGATE_RANKING_STABLE`；
- gating：`PARTIAL`，没有 `SUFFICIENT` gate；
- residual：`WEAKLY_LABELABLE`，未改变；
- decision：`GATING_PARTIAL`；
- residual-aware DDRNet：`DEFERRED_BY_DECISION_TREE / NOT_TRAINED`；
- Confirmation：`NOT_ACTIVATED`；
- Android、risk、feedback、TTS、振动、默认 App：`UNCHANGED / NOT_AUTHORIZED`。

本轮停止于结果解释，不从观察到的两个 partial gate 选择“最佳门”，也不立即组合。下一
条科学主线若继续，应先冻结一个有限组合门，检验是否能改善最差 session retention；
visual-only sidecar 已在主线暂停点以独立 host Development renderer 启动，固定
`VISUAL_CANDIDATE_ONLY / drives_alerts=false`，不得驱动 alerts。

机器结果位于
`artifacts.local/evidence/dual-loop-segmentation-failure-atlas-r0/expansion-320-v4/`。
`result.json` SHA256 为
`58ec2a8e9ae29f3eab4d66857a6a1ca77da38c3d0c514a8a54930a82aa5f7cf7`；
全量确定性复跑的 8 个核心输出与 10 张案例图逐文件一致。
