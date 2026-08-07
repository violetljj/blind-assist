# DEPTHART_ADMISSION_R0 冻结协议

日期：2026-08-07

状态：`FROZEN / DEVELOPMENT / CANDIDATE_CORE_ONLY / DA2_BASELINE_RETAINED`

## 结论先行

本轮只回答一个问题：未经蒸馏、量化、阈值搜索或真值尺度拟合的官方
`DepthART-S metric indoor`，能否在现有 120 帧 TUM consumed Development 回归集上，
保持 DA2 的 clearance、false-clear、ground、状态和时序包络，从而值得继续消耗 ONNX
与 Snapdragon 可行性预算。即使质量门通过，也不授权替换 DA2。

## 固定身份与唯一主臂

- A0：当前 DA2 metric 518 canonical 缓存，SHA-256
  `9A7FC55D...95D3A34`，角色固定为 baseline/teacher/regression reference。
- A1：官方源码提交 `0384521b...84c`，官方
  `depthart_metric_indoor_s_448.pth`，SHA-256 `597631AC...667E65`。
- TUM 是室内 cohort，故不得用 outdoor checkpoint。官方 metric S 标签为 448，但其
  NYUD/室内推理入口按原图内参与 480x640 实际张量执行；报告必须同时写清二者。
- 官方加载后的实际参数计数由运行 receipt 记录，不用论文约数替代。

DepthART-B、outdoor S、relative 224/448 均不在首轮盲矩阵中。只有 A1 结果提出明确的
容量或场景域问题才打开对应单臂；relative 模型没有冻结的米制尺度合同前，不得进入
clearance 比较。

## 四道门

1. 官方资产与有限输出：源码/checkpoint/hash/大小一致，固定内参，输出全有限。
2. 任务质量：ground、clearance、collision agreement、false-clear、truth status、
   false-block 全部相对 DA2 通过；任何 null/NaN/inf 直接失败。
3. 时序质量：clearance delta、dense-depth delta 与逐帧 metric scale drift 全部通过。
4. 导出与真机：PyTorch→ONNX 数值一致和算子清单通过后，才允许做 QNN/LiteRT/QAIRT
   的 graph coverage、unsupported ops、P50/P95、RAM、thermal 与 quality drift。

本文件把附件所列 Gate 1/2 拆成任务与时序两门；导出为 Gate 3，Snapdragon 为 Gate 4。
全局 AbsRel、分距离 AbsRel、scale bias 等是诊断，不可抵消 false-clear/时序失败，
也不因略差而单独否决任务更好的候选。

## 数据与权威边界

120 帧已消费，只能产生 Development regression/Pareto 信息。它允许判断是否值得继续
工程投入，不是 Confirmation、产品、安全、真实助盲有效性或默认 App 替换证据。最终
摄像头结论仍需要新 session/parent-disjoint、独立 RGB-D 或量距真值。

旧 DA2 P1/P2 终点不改写；FRESH-TF 和已打开的 successors 保持用户暂停状态。本轮不改
Android、提醒、风险规则、QNN 默认路由，也不删除任何 DA2 资产。
