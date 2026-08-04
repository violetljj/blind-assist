# FRESH-TF R1-A：运动补偿局部几何有效性协议

日期：2026-08-04
状态：`FRESH_TF_R1A_SOURCE_ACQUISITION_ONLY`
权限：`FROZEN_PROTOCOL_NO_OUTCOME_AUTHORITY`

## 研究问题

R1-A 只回答一个问题：相对于零阶保持、统一年龄和 R0 的全局 RGB MAD，经过运动补偿的
局部 cell support 能否在不把方向 coverage 压垮的前提下，减少 false-clear？

本轮不加入 foot/body/head、NPU 调度、语义、ToF 或学习模型。人体高度分层只有在
R1-A 通过后，才作为 R1-B 的独立消融进入。

## 固定四臂与状态

- `B0`：2 Hz zero-order hold。
- `B1`：uniform age freshness。
- `B2`：保持 R0 不变的 global RGB-change freshness。
- `C1`：motion-compensated local-cell support，加硬拒绝门。

每个 cell 只能处于 `SUPPORTED`、`OCCLUDED`、`OUT_OF_FRAME`、
`NEWLY_EXPOSED`、`LOW_FLOW_SUPPORT`、`HIGH_WARP_RESIDUAL` 或 `STALE`。
除 `SUPPORTED` 外均不得继承旧 `CLEAR`。

C1 使用 source-native 深度、内参和相对 ground-truth pose，将 anchor 点投影到当前视角
并做 z-buffer。pose 在这里只是离线机制 oracle，不代表手机已经具备可用 pose。
固定为 12×8 cell、500 ms anchor、750 ms hard TTL、每 cell 至少 32 个投影点且覆盖
至少 60%；forward-backward residual 上限 1.5 px；遮挡容差为
`max(0.10 m, 0.05 × current depth)`。任何出界、新暴露、遮挡、支持不足或残差越门都
硬拒答。

来源 metadata admission 后、RGB/depth 媒体打开前补全此前遗漏的执行定义：几何投影与
光流 endpoint 的中位残差上限固定为 3.0 px；cell 分母是当前视角有效深度像素；40%
以上反投影越过 2 px 边界记为 `OUT_OF_FRAME`；至少 32 个投影点被当前深度以前述容差
遮挡记为 `OCCLUDED`；其后 support fraction 低于 60% 记为 `NEWLY_EXPOSED`；光流至少
32 点且 60% 通过 forward/backward 门。固定优先级为 stale、出界、遮挡、新暴露、
低光流支持、高 warp residual、supported。该补全后重新绑定 protocol SHA-256，媒体
打开后不得再改。

执行采样固定为 10 Hz；深度单位固定为 uint16/5000 m；Freiburg 1 内参固定为
`517.3, 516.5, 318.6, 255.3`，Freiburg 3 为 `535.4, 539.2, 320.1, 247.6`。
双向光流固定为 full-resolution grayscale OpenCV Farneback：pyr_scale 0.5、levels 3、
winsize 15、iterations 3、poly_n 5、poly_sigma 1.2、flags 0。

## 执行前来源锁

媒体打开前锁定三个此前未进入仓库证据清单的 TUM RGB-D 序列：

| sequence | 固定角色 |
| --- | --- |
| `freiburg1_rpy` | 静态场景中的纯相机旋转 |
| `freiburg1_desk` | 静态办公室中的平移与旋转 |
| `freiburg3_sitting_static` | 近静态相机下的局部人物运动 |

必须存在 `rgb.txt`、`depth.txt`、`groundtruth.txt`；RGB-depth 和 RGB-pose 最近邻时间差
都不得超过 30 ms。每序列至少接纳 300 帧和 15 秒。传输失败终态为
`FRESH_TF_R1A_SOURCE_TRANSPORT_NOT_EVALUABLE`，媒体打开后不得换序列。

当前每种机制只有一个锁定 session，因此即使三条序列均可运行，也只形成 mechanics /
opportunity canary。正式效果主张要求每种机制至少两个独立 session，不能把单 session
结果升级为论文证据。

## 指标与晋级门

同时报告 false-clear 帧与事件、valid→invalid 和 invalid→valid 延迟、cell 与方向
coverage、macro-session 与 worst-session coverage、新暴露/遮挡误继承、左右相反方向
错误以及 UNKNOWN 原因构成。

晋级要求同时满足：C1 false-clear event 不高于 B1；macro direction coverage ≥65%；
worst-session direction coverage ≥50%；新暴露与遮挡误继承均为 0；valid→invalid P95
≤200 ms；invalid→valid P95 ≤300 ms；三种机制机会全部可评价。机会缺失返回
`NOT_EVALUABLE`，不能记为通过。

任何结果打开后不得调整阈值、网格、特征、来源、标签或会话权重。本协议只允许离线局部
有效性机制结论，不提供手机、NPU、App、导航、生产或安全权限。
