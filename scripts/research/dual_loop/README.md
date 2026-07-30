# Dual-loop phase-minus-one evidence utilities

状态：frozen historical development utilities

## 研究问题与版本

本 Module 保存 2026-07-30 已完成的 F-1A、F-1B0 与 F-1B phase-minus-one 数据、
时序和结构可达性工具。当前权威结论见 `docs/research/dual-loop/README.md`；这里不
授权重跑历史 decision 数据或启动 successor。

## 稳定 Interface

各脚本只按对应的 hash-bound historical contract 调用。它们不是新的跨域 stable
Interface；新的 successor 必须使用独立 Module 和已审查的 root Adapter。

## 输出

历史工具只写各自声明的 `artifacts.local/evidence/dual-loop/` 子目录。不得覆盖既有
receipt、review bundle、label ledger 或 structural-reachability result。

## 安全边界

- 旧 F-1B decision 输出保持密封，除非新的明确合同另行授权；
- 不把数据/时序/结构审计写成算法、提醒、产品或安全效果；
- 不修改 Android、YOLO、CameraX、风险或反馈行为。

## 停止条件

输入 identity、review receipt、固定分母、时间因果或历史 contract 不匹配即停止。
既有 terminal 不因工具仍在仓库而自动重开。

## 假设与规则质疑

这些脚本是可复算历史资产，不是当前算法假设。若结构门或接口治理变化，应通过前向
Adapter/policy 修订解决，不改写已消费 evidence。

## 失败资产复用

允许作为 regression fixture、source characterization 和 protocol negative
evidence；不得重新包装为 unseen Confirmation。
