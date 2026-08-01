# HFTF F0.1 SANPO official-test heldout effect 结果

终态：
`F0_1_SANPO_CROSS_SPLIT_BODY_HEAD_TEMPORAL_STUDENT_SIGNAL_NOT_SUPPORTED_STOP`

独立复算：
`F0_1_SANPO_HELDOUT_EFFECT_TERMINAL_VALIDATED`

## 结论

这个具体方案已经失败并永久停止：五帧 history-only MobileNetV3 student 在固定
SANPO-Synthetic body/head geometry proxy 上没有显示可复现的 future 增量，而且
single-frame current arm 的跨 split 绝对 learnability 也远低于预声明门。不能换
checkpoint、阈值、来源、指标或 gate 来救回 F0.1，也不能重跑 official-test forward。

这不等于“时间历史在所有表示中都无用”。当前证据更直接支持一个有限诊断：
**端到端从 RGB 直接学习 geometry-proxy risk cells 的跨来源可学性不足，是当前主要
瓶颈；在这个瓶颈上叠加历史帧没有稳定收益。**

## 一次性执行完整性

- 固定 3 个 official-test parent sessions、39 个 anchors、9 个 frozen checkpoints；
- package validator 成功后才运行唯一一次 prediction-only 进程；
- 351/351 predictions 已冻结，SHA-256 为
  `1a62a45412caf9582fb6d92fc037c84f8e3cef78069c200d32575e8eb83c3d1e`；
- one-shot ledger 已永久消费，SHA-256 为
  `fd2260d5a50117006510f56c29f168d1e3abef7228955d380c3ef9796d6cdfb5`；
- truth join result SHA-256 为
  `fad451876e87d24160cee8fe9dc4f8f8fdc9917b69a175ad825f42a144e57336`；
- 独立 terminal validation SHA-256 为
  `32d9d956cd162644696d96ed4476719bfa49e0f4156b41f6d7b66a5f5029bb33`。

执行没有进入 `NOT_EVALUABLE`，负终态来自冻结 effect gates，而不是缺文件、失败进程、
零分母或 hash/schema 错误。

## Gate 结果

失败：

- median seed micro-F1 delta：`-0.007233`，门为 `>= 0.03`；
- each-seed positive：seed 17/29/43 分别为
  `-0.007233 / +0.015577 / -0.025393`，只有一个 seed 为正；
- head median F1 delta：`-0.008473`，门为 `>= 0`；
- `SF_CURRENT` median-seed micro-F1：`0.173267`，门为 `>= 0.6`。

通过但不足以改变终态：

- median recall delta：`-0.018127`；
- median FPR delta：`-0.021403`；
- body median F1 delta：`+0.000551`；
- worst-source median-seed F1 delta：`-0.014909`。

known-head accuracy 约为 `0.870–0.929`，但 risk micro-F1 只有
`0.093–0.176`。这进一步说明“知道哪些 cell 可判定”并没有转化成足够好的 risk
分类。

## Burn 与后续边界

三个 official-test parent sessions 已作为 F0.1 effect evidence 消耗，不能再充当任何
successor 的 fresh validation。F0.1 不授权 after-outcome rescue、第二次 prediction、
第二次 truth join、主线替换、Android、生产或安全主张。

后续若继续 HFTF，只能另立一个机制不同的新问题、使用 fresh parent sources 和新的
outcome-before contract。优先问题应从“再换一个时序 backbone”转为：是否把视觉表征
先约束到可跨来源迁移的物理中间量，再用显式因果 transport 形成 future field，能够
先解决 current learnability，再证明 history 的独立增量。
