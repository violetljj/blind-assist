# DA V2 选择性 W8A16 A5S R0

日期：2026-08-05（DLC 转换前冻结）

## 决策

只量化 12 个 Transformer block 中 qkv、projection、fc1、fc2 的 48 个静态 MatMul 权重。
权重采用 axis 1 per-output-channel symmetric INT8；激活、动态 attention MatMul、Softmax、
LayerNorm、残差、patch embedding、图输入输出和整个 depth head 保持 FP16。

这不是父协议中被禁止的全图 `A5_FULL_IMAGE_INT8`，因此命名为 `A5S`。它不使用 activation
calibration，也不能在转换失败后扩大 INT8 范围。

## 风险与门

该 arm 的精度风险低于 activation INT8，但 profile 表明最重的动态 attention/Softmax 仍为
FP16，因此性能收益可能低于 `1.15x`。转换后首先检查 DLC 是否精确保留 48 个 INT8 权重，
并确保激活与 IO 没有被量化；静默回退 FP16 或量化传播都直接停止。

只有 precision contract 通过，才物化 120 帧候选缓存并运行一次 P1 R1。质量 14/14 通过后
才能生成 SM8650 cached context 并测固定 APK；最终还必须同时达到 full-chain `>=1.15x`
和模型 `>=8 Hz`，否则不能充当高频 student。

该路径仍是 consumed Development evidence，不建立产品或安全 authority。
