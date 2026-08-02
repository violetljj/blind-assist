# HFTF Stage C D41 JRDB causal future-box field result

日期：2026-08-03（Asia/Hong_Kong）

## 结论

终态：

`D41_JRDB_CAUSAL_FUTURE_BOX_FIELD_NOT_SUPPORTED`

D41 可评估，且出现一个不能被抹掉的窄正信号：

- 3,392 same-identity future opportunities / 54 identities
- candidate mean IoU：`0.40926`
- current-box baseline mean IoU：`0.36434`
- mean IoU delta：`+0.04491`
- 3/4 sequences 的 mean IoU delta 为正

但增量不广泛、不稳定，冻结 support gates 只通过 3/7：

- median IoU delta：`0.00000`
- candidate better fraction：`47.995%`
- normalized center error 只降低 `6.887%`，未达 10%
- absolute log-area error：`0.29466 -> 0.41313`，明显恶化
- STLC mean IoU delta：`-0.00376`

因此不能声称 causal future-box representation 整体优于 current box。当前证据更像：
image-space translation trend 在部分 source/样本上有效，但 constant-velocity
log-scale 外推缺乏跨样本稳定性。

## 冻结候选

对 D33 detector tracks 的连续 7 帧，以真实 timestamps 分别拟合 center x/y 与
log width/height，并外推到 `+15 frames`。forecast 不读取 annotation；current
Hungarian association 与 future same-identity native box 只在评价阶段使用。
没有 evidence threshold，也没有搜索 regression、history、horizon、state subset、
clamp、association 或 sequence exclusion。

## pooled paired metrics

| 指标 | current-box baseline | D41 candidate | 差异 |
|---|---:|---:|---:|
| mean future-box IoU | 0.36434 | 0.40926 | +0.04491 |
| median IoU delta | — | — | 0.00000 |
| candidate IoU better fraction | — | 47.995% | gate FAIL |
| mean normalized center error | 0.53249 | 0.49582 | -6.887% |
| mean absolute log-area error | 0.29466 | 0.41313 | +0.11847 |

## sequence 分解

| sequence | opportunities | mean IoU delta | median delta | better fraction | center reduction | area-error delta |
|---|---:|---:|---:|---:|---:|---:|
| Clark Center | 1,100 | +0.09509 | 0.00000 | 48.364% | 4.251% | +0.14182 |
| Gates Basement | 830 | +0.03229 | +0.01740 | 53.855% | 7.975% | +0.06623 |
| Meyer Green | 161 | +0.16057 | +0.12619 | 59.627% | 9.827% | +0.05111 |
| STLC 111 | 1,301 | -0.00376 | -0.00695 | 42.506% | 10.973% | +0.14039 |

## gates

evaluability 全部通过：

- producer receipt source frames：480/480
- frames with tracked occurrences：478（diagnostic）
- opportunities：3,392
- distinct identities：54
- sequences with >=50 opportunities：4
- finite metrics：PASS

support：

| Gate | 结果 |
|---|---|
| pooled mean IoU delta >=+0.02 | PASS |
| pooled median IoU delta >=+0.02 | **FAIL：0.00000** |
| candidate better fraction >=55% | **FAIL：47.995%** |
| center error reduction >=10% | **FAIL：6.887%** |
| log-area error noninferiority | **FAIL：+0.11847** |
| positive mean IoU delta in >=3 sequences | PASS：3/4 |
| no sequence mean IoU delta <-0.02 | PASS |

## 科学解释与下一变量

D33 的 scalar range-direction mechanism 与 D41 的局部 mean-IoU gain 保持成立；
D41 拒绝的是把四个 image-box state 都按同一 constant-velocity 假设外推，就足以
形成稳定 future spatial field。

所有四个 sequence 的 mean log-area error 都变差，而 center error 全部改善。这给出
一个清楚的断点：

- translation state 值得保留为候选 primitive；
- scale state 不能再用无几何约束的一秒 log-linear extrapolation；
- 不得在同一 outcome 上删掉 scale 后重跑来“救”D41。

下一实验必须使用新鲜 source/outcome，或引入预先冻结的 ego-motion/metric-geometry
teacher，分别建模 camera translation 与 object-relative scale；不能继续搜索
state subset、horizon 或 regression order。

## control-plane repairs

R0.1：首次 join 在任何聚合 outcome 前遇到 fully-off-frame forecast。source-only
census 为 3,692 forecasts，其中 20 fully outside、227 partially clipped。修复为
保留 raw projected box，让全部指标原样惩罚越界，不拉回画面、不排除样本。

R0.2：首次完整 report 把 478 个非空 track frames 误当作 producer source census，
而冻结 COMPLETE receipt 为 480/480。评估器改为绑定 receipt；478 保留为
non-empty diagnostic。该修复不改变任何 effect metric，而且当时已有四项 support
gate 失败，不可能把结果救成 supported。

这两项是可修复 control-plane defects，不烧毁或删除 cohort，也不把失败伪装成
算法负结果。

## 复现

- D32/D33/D41 evaluator tests：11 PASS
- report 连续重建 SHA 稳定：
  `73418b3308a259e63a2c413105d907f6ea416297628568f1d80f0d0d0db71ba3`
- report size：5,612 bytes
- tracks SHA：
  `efa249fdfe8114dfeb1da419ffdb359189e3d4e6b1f406fabad04a31a39a0fa1`
- producer receipt SHA：
  `fa91162274222b9fe2254ae675ccb95af3fcdd6dca50ab267d476d74764be318`

artifact：

`artifacts.local/evidence/hftf/stage-c-d41-jrdb-causal-future-box-field-v0/report.json`

## claim ceiling

`RESEARCH_MAINLINE_UNCHANGED / DEFAULT_APP_UNCHANGED`

D41 不建立 event utility、Android runtime、独立泛化、产品或安全主张，也不覆盖
D35 device gate。
