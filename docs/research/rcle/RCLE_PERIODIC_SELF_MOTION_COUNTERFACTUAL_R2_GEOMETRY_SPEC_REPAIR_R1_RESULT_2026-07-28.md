# RCLE periodic self-motion counterfactual R2 geometry repair R1 result

终态：`INTERVENTION_NOT_EVALUABLE / HOLD_P1 / EXECUTION_NOT_AUTHORIZED`

R1 保留 R0 的 80 条 MAIN 记录、全部 numeric seed、相机内参和轨迹，并只为
8 条 GUARD 记录物化固定的显式中深度渲染目标。独立回执实际消费的
angle/acos G13 实现对 8 条 `MONOTONIC_APPROACH_PLUS_PERIODIC` 报告
`2.98e-8–3.65e-8 rad > 1e-12 rad`，因此 13/14，合法 fail closed。

不可变身份：

- implementation lock SHA-256：
  `b49efb5ef2d267dbcb50a3ff85f1890b4026272d77188a914dca2e9a91cc624d`
- amendment SHA-256：
  `521fd5fe523e9970c437c82e0dd5f3091a283e57de78e953db48b5d0cb0bfe48`
- 失败 receipt SHA-256：
  `af00df05c115036ea31bb3d05addbebfcebad73122d2b354f7e52170c2277e9a`
- 实际消费的 validator 语义 SHA-256：
  `5be754efdcd04e4fcaa3fafc64a6b39ce92a5e36594e4b3fd2e141a62c5b9d8b`

## 并发 source-hash 竞态

R1 进程启动后，validator 文件从 angle/acos 版本替换为 scipy-magnitude 版本。
运行中的进程仍执行 `5be754…` 的 angle/acos 代码，但写 receipt 时从磁盘记录了
后来文件的 `fd80e5…` SHA-256。回执中的 `2.98e-8–3.65e-8 rad` 数值、可复现
G13 failure，以及 lock 内的 `5be754…` 共同证明实际消费语义。

该竞态是 R1 的历史缺陷，不能通过覆盖 receipt 或把 R1 改称 dry run 修复。R1
源码、lock、amendment 和 receipt 已恢复并冻结为上述历史身份；所有后续修复只能
放入新版本。

## 权限边界

R1 未读取或运行 RCLE output，未校准 blur/low-texture，未运行 P3/P4，也未修改
R3、阈值或三-pair；sequence16、CoTracker、Android 和 realtime 均未进入。
