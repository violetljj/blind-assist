# HFTF Stage B reference metric pilot D1

日期：2026-08-01

状态：`FROZEN_DEVELOPMENT_PILOT_RESULT_NOT_RUN`

## 1. 目的

D0 证明 swept-envelope mechanics 能运行，但“比旧 point-support 多输出 collision”
是几何定义的直接结果，不能单独证明更准确。D1 在相同 burned R2 sessions 上设计
formal R3 的比较指标，不产生 fresh evidence。

## 2. 三个 arm

- candidate：stride 8、offset 4 的 swept human-envelope obstacle collision；
- baseline：使用与 candidate 完全相同的 stride-8 points，但按 R2 angular-cell
  point-support 分箱；
- reference：stride 4、offset 2 的 swept human-envelope collision。

candidate lattice 的坐标为 `4 mod 8`，reference lattice 为 `2 mod 4`，两者没有共享
pixel。reference 的采样密度为 candidate 的 4 倍，但仍来自同一 metric-depth 与
panoptic source，只能称为 disjoint dense geometry proxy reference，不是独立人类
风险真值。

## 3. 公平比较

candidate 与 baseline 在同一 swept-prism known mask 上评价，避免一个 arm 靠删除
UNKNOWN 获益。ground continuity 不混进 obstacle confusion；其覆盖和风险另行报告。

reference positive count threshold 不在 D1 内挑一个最好结果，而是同时冻结
`1/2/4/8` 四个 sensitivity points。每个 session、每个 height layer 与 cohort 都报告
precision、recall、F1、accuracy、confusion counts，以及 paired
candidate-only-correct / baseline-only-correct。

## 4. D1 readiness

只有以下各项全过，D1 才允许冻结 R3：

1. 4/4 burned sources 精确绑定；
2. candidate/reference pixel lattices 确认不相交；
3. 每个 threshold、每个 session 至少有一个 reference positive 和 negative；
4. 每个 session 至少一个 known cell 上 candidate 与 baseline 不同。

上述 readiness 不等于 candidate 胜出。R3 的 primary threshold、最低 effect margin、
session consistency 与停止条件必须阅读全部 D1 sensitivity 后冻结，并在任何 fresh
source acquisition 前提交。

## 5. 权限

D1 只允许消费 burned R2 sessions 进行 metric design。它不授权 fresh outcome、
future Stage C、student/H2、研究主线、Android、提醒、默认 App、生产或安全 claim。
