# HFTF Stage B reference metric pilot result D1

日期：2026-08-01

终态：`D1_REFERENCE_METRICS_READY_FOR_R3_GATE_FREEZE`

证据角色：`CONSUMED_METRIC_DESIGN_ONLY`

## 1. 结论

在四个 burned R2 sessions 上，stride-8 swept-envelope candidate 相对使用完全相同
points 的 angular point-support baseline，对不相交 stride-4 dense geometry proxy
reference 表现出稳定增量。四个预冻结 reference count thresholds 的 cohort micro-F1
增量均为 `+0.1587–+0.1720`，4/4 sessions 在每个 threshold 上均为正。

这足以冻结 fresh R3 的正式比较门，但仍是同一 source metric-depth/panoptic 派生的
geometry-proxy agreement，不是独立人类风险真值、用户效果或主线替换证据。

## 2. 绑定结果

报告：

`artifacts.local/evidence/hftf/stage-b-reference-metric-pilot-d1-20260801/reference_metrics.json`

SHA-256：

`d4eb37137f0c2502a7f860e29d7d2148c9dafb89dea261f1e31ca12b1c31e6cf`

执行实现 commit：`bd0bd42`

实现 SHA-256：

`66396b354127f7001cb152cfce946d29e6efb6a32359baa6d7c34a6de176fff2`

D1 protocol SHA-256：

`fbfef7fd0e7dde06fba29e93dcaafff1f428e40a3e13e78a7533479898c04b2b`

4/4 source binding、pixel-lattice disjointness、每 session/threshold 正负 reference
opportunity，以及 candidate/baseline disagreement readiness 全部通过。

## 3. Threshold sensitivity

| reference count | positive / negative | candidate P/R/F1 | baseline P/R/F1 | F1 delta | paired candidate-only / baseline-only |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 613 / 2,569 | .9983 / .9804 / .9893 | .7187 / .9837 / .8306 | +.1587 | 243 / 10 |
| 2 | 608 / 2,574 | .9967 / .9868 / .9917 | .7139 / .9852 / .8279 | +.1638 | 246 / 7 |
| 4 | 602 / 2,580 | .9917 / .9917 / .9917 | .7068 / .9850 / .8230 | +.1687 | 249 / 4 |
| 8 | 588 / 2,594 | .9734 / .9966 / .9849 | .6913 / .9864 / .8129 | +.1720 | 251 / 2 |

primary threshold 不能按最大增量选择。R3 固定 count `2`：它拒绝 reference 的单采样
点 speckle，同时保留 608/613 个 threshold-1 positives，且位于预冻结 sensitivity
范围内部。

threshold 2 的 height-layer F1：

| layer | candidate | baseline | delta |
| --- | ---: | ---: | ---: |
| foot | .9873 | .7824 | +.2049 |
| body | .9947 | .8839 | +.1108 |
| head | .9909 | .7895 | +.2014 |

## 4. Fresh R3 冻结门

R3 使用全新 sources，并在 outcome 前固定：

- primary threshold `2`；
- cohort micro-F1 delta `>= +.10`；
- cohort precision delta `>= +.10`；
- cohort recall delta `>= -.02`；
- 4/4 session micro-F1 delta 均 `>= +.05`；
- foot/body/head primary-threshold F1 均高于 baseline；
- thresholds `1/2/4/8` 的 cohort F1 均高于 baseline，且 paired
  candidate-only-correct 均多于 baseline-only-correct；
- 每 session 每 height obstacle-known coverage `>= .10`。

这些门明显低于 D1 observed delta，但仍要求稳定、非微小且不靠 recall 换 precision
的增量。

ground continuity 必须单列 ground-known/risk/UNKNOWN。若 fresh cohort 没有真实
step/drop opportunity，只能得到
`OBSTACLE_ENVELOPE_GAIN_SUPPORTED_GROUND_NOT_EVALUABLE`，不得把 synthetic fixture
外推为完整 Stage B 通过，也不得进入 future Stage C。

## 5. 权限

D1 只授权冻结并准备 fresh-source formal R3。R3 outcome、future Stage C、student/H2、
研究主线、Android、提醒、默认 App、生产与安全仍需后续独立终态。
