# D0 ego-motion error attribution R0 独立统计复核

状态：`REPAIR_NEEDED / NOT_IMPLEMENTATION_READY / NOT_RUN`

日期：2026-07-30（Asia/Hong_Kong）

## 结论

该复核不改写已冻结 D0 合同，也未运行 D0。它纠正了早先工程设计
`PASS` 不能覆盖的统计与可识别性问题：

- 469 行是 observational parent-event units，不是 469 个独立样本；最高独立单位
  仍只有一个 REveL capture；
- wrong-signed 与 sensor-motion/temporal-instability 的富集不能识别“因果机制占主导”，
  因为 ego motion 本身也可能产生 temporal instability；
- 当前三个 `*_DOMINANT` 出口应改为 operational canary priority，不得写成机制证明；
- finite coverage、missingness、跨事件时间重叠组件、固定时间块敏感性、person competing
  explanation 和候选指标选择规则尚未闭合。

因此 D0 R0 保持 `CONTRACT_FROZEN / NOT_RUN`，但不再是当前唯一或立即可执行的前瞻
路线。只有当生产 temporal geometry A/B 得到有效 `NO_INCREMENT`，才考虑另立 D0 R1，
把出口改成 `EGO_CANARY_PRIORITY / TEMPORAL_TREND_PRIORITY / NO_PRIORITY_IDENTIFIED`
并完成依赖敏感性后再实现。

## 审计证据

- D0 protocol SHA-256：
  `f43add496d1dab53072bc9a27ddd28e716ec9480360a41cca6243ed4b326cf62`
- 早先工程设计审查 SHA-256：
  `2e51d47f59d5c4bece294e1628abd8fe3c0fd0f1e4dfb907e0c03afa26300cc7`
- 469 个 primary event ID 唯一，同一 target 内无时间重叠；
- 全部事件来自一个 capture；只读复核发现 159 个跨 target 时间重叠事件对，精确时间
  重叠组件为 310 个；
- R2 result、manifest、replay、truth、natural events、producer receipt 与 evaluation
  哈希均匹配，R2 仍是有效 Development 负结果。

本复核未打开旧 F-1B decision output、未访问 Confirmation、未修改候选阈值或重跑
REveL。
