# RCLE reference-track failure diagnosis R0

日期：2026-07-28

状态：`SELECTED_NEXT_EXPERIMENT / DESIGN_ONLY / NOT_STARTED`

## 决策

旋转补偿机制审计关闭后，不增加 bearing，也不接入路径走廊。下一独立实验只使用
一个参考模型：CoTracker3 offline，将其作为点轨迹诊断参照，而不是 oracle、
伪真值或端侧候选。

## 可证伪问题

> ADVIO 冻结高角速度窗中，补偿后 RCLE 仍高触发，主要来自 sparse LK / 局部
> 仿射提取误差，还是来自仅用姿态旋转无法解释自然行走图像运动？

该实验只区分失败来源，不评价碰撞检测性能。

## 输入冻结

- source：已经 `TUNED_ON` 的
  `ADVIO_OFFICE03_SEQUENCE15_IPHONE`；
- 窗口：保持机制审计的 high `343..462` 与 low `2..121`；
- 图像：官方去畸变、0.5 scale；
- pose：`wxyz` + official `T_cam_imu`；
- RCLE threshold 与三-pair：只读保留，不修改；
- future ADVIO sequence16：禁止访问；
- 不增加其他视频、滑窗或事后挑选失败片段。

## 唯一参考模型

CoTracker3 offline 只生成参考点轨迹。它是学习模型，可能受域偏移、遮挡和
伪标签训练影响，因此：

- 不把其轨迹写成 ground truth；
- 不用其输出训练或校准 RCLE；
- 不因其与 RCLE 一致而确认 RCLE；
- 不因其与 LK 不一致而自动判 LK 错误。

执行前必须固定模型版本、权重 hash、代码 commit、输入尺寸、query 生成规则和
可见性处理；若资产或运行成本不能有界闭合，终态为 `NOT_EVALUABLE`。

## 配对诊断

对两个冻结窗口逐 pair：

1. 读取 R2 ledger 中同一 previous-frame RCLE query 点；
2. 同时取得 LK next point 与 CoTracker3 next point；
3. 保留各自原生 validity/visibility，不互相填补缺失；
4. 在相同 cell 和相同 pair identity 上计算：
   - LK–reference endpoint disagreement；
   - pose-predicted flow 对 LK/reference 的 endpoint error；
   - raw 与 rotation-compensated local affine expansion；
   - support、空间覆盖和 fit residual；
5. 所有汇总固定按 window 与 cell 报告，不把相邻 point/pair 当独立样本量。

## 固定判读

| 结果模式 | 允许解释 |
| --- | --- |
| reference 与 pose 模型一致、且 reference-based compensated expansion 明显低于 LK | `TRACKER_LIMITED_DIAGNOSTIC_SUPPORT` |
| reference 与 LK 基本一致、两者对 pose 模型均保留同方向残差 | `ROTATION_ONLY_MODEL_LIMITED_DIAGNOSTIC_SUPPORT` |
| reference/LK 在 visibility、support 或方向上不稳定 | `FAILURE_SOURCE_NOT_EVALUABLE` |

“明显低于”和“基本一致”的数值容差必须在任何 reference 输出访问前另立小型
实现合同；本设计不事后填写阈值。

## 停止边界

- 两个冻结窗口完成一次即停止，不扩 session；
- 不运行 ScaleFlow++、Video Depth Anything、VGGT 或第二个点跟踪器救援结果；
- 不修改 RCLE、阈值、三-pair、support manager 或 cell；
- 不计算 AUROC/F1，不产生 performance/generalization；
- 不进入 bearing、路径权重、Android、人体、安全或产品。

## 后继权限

本设计本身不授权下载权重或执行。只有模型与代码身份、资源预算、query 规则、
缺失处理及判读容差预冻结后，才能另行启动一次 Development Diagnostic。
