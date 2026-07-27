# RCLE low-reference false-trigger R1

状态：development complete / ready for external confirmation design

## 研究问题与版本

`RCLE_LOW_REFERENCE_FALSE_TRIGGER_ATTRIBUTION_R0` 在已烧过的四窗 development
cohort 上区分低参考触发来自旋转补偿、pair-level local flow 还是 observable
support-manager。允许声明旧实现是否值得进入唯一针对性修订，不产生 confirmation。

## 稳定 Interface

以模块方式调用 `attribution.py`，提供仓库内 contract、TUM source root 和全新输出
目录。输入 hash、967/598 分母、pair timestamp 或旧 threshold 漂移即 fail closed。

## 输出

只写入 `artifacts.local/evidence/rcle_low_reference_false_trigger_r1/` 下的新目录。
旧 A4 result/ledger、冻结 cohort 和 source-native geometry audit 永不回写。结果见
[`RCLE_LOW_REFERENCE_FALSE_TRIGGER_R1_RESULT_2026-07-27.md`](../../../../docs/research/rcle/RCLE_LOW_REFERENCE_FALSE_TRIGGER_R1_RESULT_2026-07-27.md)。

## 安全边界

只具 development 机制诊断权限；不换数据、不补窗、不调 `0.01/s`、不形成 Android、
真人、产品或安全结论。

## 停止条件

baseline-only support-manager 反事实与唯一 causal three-pair confirmation 均已
完成并独立复算。不得在当前四窗继续搜索确认长度、阈值或候选算法；下一合法边界
只是在新协议下设计未见 all-real cross-source 外部验证。

## 假设与规则质疑

主假设是短时 pair-level local-flow 波动主导误触发。反证是 support-manager 或
rotation-compensation 类别占 geometry-below 旧触发多数。成本为两个缓存 TUM 窗。

## 失败资产复用

失败或 INVALID 只关闭本 attribution evidence version；旧 ledger 仍可作 development
regression fixture，不能重新包装为 unseen confirmation。
