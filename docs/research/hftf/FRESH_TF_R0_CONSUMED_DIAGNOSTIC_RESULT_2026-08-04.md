# FRESH-TF R0 consumed diagnostic result

日期：2026-08-04
终态：`FRESH_TF_R0_CONSUMED_DIAGNOSTIC_NOT_SUPPORTED`
证据角色：`CONSUMED_DIAGNOSTIC_ONLY`

## 结论

第一次受控尝试否定了“整帧 RGB 变化直接乘入 decision freshness”这一具体实现。
selective RGB-change 臂把 false-clear 从 3 个降到 0，但 known coverage 从 100% 降到
`21.24%`，低于执行前冻结的 65% 门。零 false-clear 主要来自大面积拒答，不能记为
机制成功，也不能在本片段调整 RGB scale、quality threshold 或 tau 救援。

范围化终态为：

- `GLOBAL_FRAME_FRESHNESS_PROXY_REJECTED`
- `LOCAL_GEOMETRIC_VALIDITY_NOT_YET_EVALUATED`

这不关闭 FRESH-TF 总问题。它关闭的是：

`whole-frame grayscale MAD × global age decay × decision-selective tau`

在当前冻结参数和已消费 Bonn 单序列上的候选。下一次只有换成新的 parent/session-disjoint
数据，并将变化证据限定到 motion-compensated local cells、遮挡/边界支持和硬有效性门，
才是新的可评价问题。foot/body/head 分层不进入 R1-A；只有 R1-A 先通过，才在 R1-B
单独评价分层增量。

## 固定四臂结果

每臂评价 30 帧 × 3 horizons × 17 directions = 1,530 cells；anchor 与当前真值之间
出现 30 个 state transitions，因此不是机会为零的 `NOT_EVALUABLE`。

| arm | known coverage | known accuracy | false-clear | false-clear / predicted clear |
| --- | ---: | ---: | ---: | ---: |
| 2 Hz zero-order hold | 100% | 98.04% | 3 | 0.271% |
| 2 Hz + 750 ms TTL | 100% | 98.04% | 3 | 0.271% |
| uniform age freshness | 80.00% | 98.61% | 1 | 0.113% |
| selective RGB-change freshness | **21.24%** | 100% | 0 | 0% |

唯一失败 gate 是 selective known coverage。`UNKNOWN -> CLEAR` violation 为 0；实现仍
保持 fail closed。

## 失败归因

冻结 `64x48 grayscale MAD / 255` 在 30 帧中的中位数约 `0.0865`，已经大于固定
scene-change scale `0.08`。该信号混合了相机外观变化、人物运动、亮度与像素错位，
不能判断某个方向/距离单元的旧米制证据是否仍支撑 `CLEAR`。它把大部分 anchor 后帧
整体拒答，说明新鲜度必须是局部、坐标对齐且支持感知的场，而不能是一个全图乘数。

## 证据绑定与权限

- protocol SHA-256：`1A8574E8FC8758E3810F01E783BFB600952264552AAB5A7278304AFDA48012F6`
- evaluator SHA-256：`E750976075FD15BE6CB144AA85175CFD481548C81DF6EB40E724279450AE9085`
- ignored result SHA-256：`358220156DCC7A0F3779FF1891E773583EE09AB787489402A8E1E09E1A1DFD43`
- ignored trace SHA-256：`8D737F0F1AEAD7544BAE87FB77AA6166887518B0341316A09D6EFF1BCC27BB14`
- focused tests：6/6 通过。

本结果不评价 height-stratified 增量、光流/pose warp、NPU 主动调度、能耗、手机现实
深度准确率、App 提醒、导航或安全效果；默认 App 与研究主线不变。
