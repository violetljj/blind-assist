# Dual-rate metric depth observer R1 protocol

日期：2026-08-03

状态：`FROZEN_BEFORE_D_ARM_EXECUTION`

## 研究问题

R1 只检验一个新系统假设：高频 DA V2 输出能否被低频、异步完成的 Metric3D
关键帧用固定的稳健仿射关系校正，并在锚点过期时显式输出 `UNKNOWN`。它不检验告警、
用户收益或安全，也不授权默认 App、硬件采购或取消 ToF。

## 数据和证据角色

开发回放只使用 A0 已消耗的四个 TUM RGB-D 窗口、共 120 帧。三个输入报告及 SHA-256
已写入同目录的机器合同。A/B/C 是冻结的既有结果，D 在查看其逐帧 outcome 前按本合同
固定。此处只能得到 `THESIS_DEVELOPMENT` 证据；即使 D 通过，也必须在新的
session-disjoint RGB-D/测距数据和目标设备上另做一次性确认。

## 四臂

- A：DA V2 Metric FP16 单独运行；
- B：Metric3D ViT-S FP16 单独运行；
- C：既有每五帧同步运行 Metric3D、按左中右偏移的回放；
- D：DA V2 每帧运行，Metric3D 单 worker 异步关键帧；只使用已经完成的锚点拟合
  `depth = a * DA + b`，向后传播到当前帧。

D 的关键帧请求周期固定为五帧；busy 时合并请求，不形成无界队列。最近三个已完成锚点
的左/中/右成对值进入 Theil-Sen 斜率和中位截距估计。最少 3 对、DA 值跨度至少
0.25 m、拟合中位绝对残差不超过 0.25 m、斜率必须在 `[0.25, 4.0]`。锚点年龄按
`当前捕获时刻 - 锚点源帧捕获时刻` 计算，超过 1.0 秒整帧 `UNKNOWN`；不得按完成时刻
伪装锚点新鲜度，也不得沿用旧决策。

这些常数不在本 cohort 上搜索。D 只实现仿射参数传播；稠密光流传播是不同复杂度和
失败模式的后继假设，不在 R1 中混入。

## 指标与门

任务门沿用 A0：paired-valid `>=90%`、clearance MAE `<=0.25 m`、三带包络一致率
`>=90%`、false-clear `<=5%`、temporal delta MAE `<=0.15 m`。系统门另要求已知输出
`>=90%`、anchor source-age P95 `<=1.0 s`、DA 中断比例 `<=5%`，并必须提供共驻峰值
内存、持续温度和 fresh session-disjoint 结果。缺失系统证据是 `NOT_EVALUABLE`，
不是通过。

## 两种执行边界

Windows CUDA 独立 worker 用冻结的 Metric3D 稳态 `142.33 ms` 做开发回放，DA 不因
锚点 worker 停止。手机资源审计使用已测得的 canonical Metric3D HTP `1500.794 ms`
与当前 DA HTP `174.319 ms`；两者共享 HTP 时把 Metric3D busy 区间计为 DA 中断。
但手机 DA 资产是 relative-only Qualcomm checkpoint，不是 PC 的 Hypersim metric
checkpoint，因此手机侧只能裁决资源时序，不能继承 PC 质量结果。

## 终局规则

只有 D 在新真实数据上过全部任务门，且目标设备的内存、温度和中断门也通过，才允许
继续讨论不增加 ToF。当前 consumed 回放无论正负都不可用于调 cadence、TTL、拟合器、
阈值或换 cohort 救援。若手机共享 HTP 的中断/老化失败，结论是当前实现边界失败，
不是 ToF 已被证明为唯一方案。
