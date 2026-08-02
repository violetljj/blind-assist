# HFTF Stage C D40 THOR-MAGNI continuous-track projected-risk result

日期：2026-08-03（Asia/Hong_Kong）

## 结论

终态：

`D40_THOR_MAGNI_CONTINUOUS_TRACK_PROJECTED_RISK_NOT_SUPPORTED`

D40 离开已经停止的 scene-scale veto/persistence family，直接检验一个更接近
HFTF 原始问题的候选：用 production causal track 的连续尺度斜率，把当前选中目标框
投影到一秒后，再由独立 production risk kernel 计算 future-risk terminal。

该机制有充分机会且完整执行：

- 205 forecast windows / 19 sessions
- 136 positive-slope frames
- 69 negative-slope frames
- 530 anchors 全部映射
- 3,710 unique frames / 14,364 detections 与冻结 source receipt 一致

但 baseline 与 candidate 的窗口终态完全相同：

- positive anchors：`114/157 -> 114/157`
- positive events：`79/107 -> 79/107`
- negative alerts：`251/373 -> 251/373`
- 五个 folds 的 gains/losses 全部为 0

因此 D40 不是 `NOT_EVALUABLE`，也不是 control-plane invalidation；它是一个明确的
科学负结果：当前“连续 box-scale slope + 固定一秒 bottom-center-preserving
projection + 现有 risk kernel”没有产生任何 event/anchor utility 增量。

## 冻结实现

source-only replay 在任何 truth join 前完成：

1. 将 D36 重叠 sample windows 去重为 session-continuous frame stream；
2. session 切换或 gap `>500 ms` 时重置 baseline、candidate 与 track producer；
3. baseline 用当前 detections 运行独立 production kernel；
4. candidate 从 production `CausalTrackTristateGeometryProducer` 读取连续
   `signedApproachRatePerS`，不要求 tri-state admit；
5. 对 selected box 使用固定 horizon `1.0 s`：
   `scale = exp(slope * 1.0)`；
6. 保持 bottom-center，缩放 width/height，clamp 到 frame；非法输出 fail closed
   到当前 detections；
7. candidate 用投影后的 detections 运行另一独立 production kernel。

没有搜索 horizon、slope clamp、history、association 或 risk threshold。

## paired outcomes

| 指标 | baseline | D40 candidate | 差异 |
|---|---:|---:|---:|
| positive anchor alerts | 114/157 (72.611%) | 114/157 (72.611%) | 0 |
| positive event hits | 79/107 (73.832%) | 79/107 (73.832%) | 0 |
| negative anchor alerts | 251/373 (67.292%) | 251/373 (67.292%) | 0 |
| candidate-only negative windows | — | 0 | 0 |

fold outcomes：

| Fold | forecast windows | positive event delta | positive anchor delta | negative alert delta |
|---:|---:|---:|---:|---:|
| 0 | 51 | 0 | 0 | 0 |
| 1 | 62 | 0 | 0 | 0 |
| 2 | 42 | 0 | 0 | 0 |
| 3 | 5 | 0 | 0 | 0 |
| 4 | 45 | 0 | 0 | 0 |

## gates

evaluability：

- exact cohort/source census：PASS
- all anchor windows mapped：PASS
- detector/anchor parity：PASS
- finite forecast output：PASS
- forecast opportunity `>=50 windows / >=5 sessions`：PASS，`205 / 19`
- baseline positive/negative opportunity：PASS

support：

| Gate | 结果 |
|---|---|
| positive event noninferiority | PASS |
| positive anchor noninferiority | PASS |
| negative alert noninferiority | PASS |
| candidate-only negative bound | PASS |
| meaningful strict gain | **FAIL：0 event gains / 0 negative reduction** |
| strict gain in >=3/5 folds | **FAIL：0/5** |

## 科学边界

D40 不撤销 D32/D33 已支持的 causal track future-range mechanism。它缩小的是从
mechanism 到 utility 的缺口：

- 连续尺度斜率确实在 205 个窗口中可计算，D36 的严格 tri-state coverage 不是
  continuous estimator 的 coverage ceiling；
- 但仅用 selected target 的 box-scale projection，再进入当前离散 risk kernel，
  不足以改变 terminal；
- 因此当前 selected-target box-scale projection recipe 停止，不在已消费 outcome
  上搜索 horizon、clamp 或 threshold。

若继续推进 HFTF，下一候选必须引入新的、预先冻结的几何 teacher/field evidence，
直接表达 future traversability，而不是继续围绕现有 alert 做 veto、hold 或
selected-box risk 重参数化。它需要新的数据角色或新鲜 outcome cohort。

## 控制面记录

实现阶段在 truth join 前发现：同一去重 frame 会从不同 sample ordinal 再次出现。
replay equality 因此改为比较 observation fields、排除非语义 ordinal。该修复发生在
source-only 阶段，没有读取 label、没有生成 outcome，也没有烧毁 cohort。

一次复跑命令还被 PowerShell 将 JVM property 误解析为 Gradle task；改用
`GRADLE_OPTS` 后原样成功。它属于 verification command failure，不影响 artifact。

## 验证与 artifact

重复 production Kotlin replay：

`D40_KERNEL_REPLAY samples=530 unique_frames=3710 detections=14364 forecast_frames=205 forecast_windows=205 baseline_windows=365 candidate_windows=365`

- `core:assist` 全量测试：PASS
- D36-D40 evaluator：14 tests PASS

目录：

`artifacts.local/evidence/hftf/stage-c-d40-thor-magni-continuous-track-projected-risk-v0/`

- `kernel_replay.tsv`
  - SHA-256：
    `fae215ddebfcb774c15e5ef18934fca36a85b1481d63905762fb70ac435884e4`
- `report.json`
  - SHA-256：
    `c4716729c69de435f40eee3717c5bdada2e20ee6f49fb79f0dfec8d4869d0d06`

input bindings：

- D12 samples：
  `9a099a52d29da60f889d40cacc1a2e267e506c23dc4aafa7fba1764eb1d64a54`
- D36 detections：
  `5083db4c86ff20c01d12a47aa9b419d3cd1727b8f8ed1383020b33601ad6f731`
- D36 producer receipt：
  `26f2991f4f836c2611224bc9ea1c9066d00b6bf03b600155736fafcbf8ab5ade`

## claim ceiling

`RESEARCH_MAINLINE_UNCHANGED / DEFAULT_APP_UNCHANGED`

D40 不覆盖 D35 device gate，不建立 independent generalization、产品或安全主张。
