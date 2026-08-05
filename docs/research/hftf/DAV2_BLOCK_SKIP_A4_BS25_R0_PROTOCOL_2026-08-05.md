# DA V2 A4-BS25 固定 block-skip 协议

日期：2026-08-05

状态：`FROZEN_BEFORE_A4_BS25_P1_CACHE`

## 唯一改动

保持 518 输入、FP16、原 checkpoint、patch/token 数、depth head、预处理和输出对齐不变，
仅将 12 个 ViT-S Transformer block 中第 4、8、12 个（零基索引 `3/7/11`）替换为
identity。四个 DPT feature tap 仍位于 `2/5/8/11`，候选共有 9 个活动 block。

这是一次固定的均匀 25% block-cost reduction，不在 P1 上搜索跳层数量或位置。现有 profile
给出 encoder 占全链路 88.24%，因此理想化全链路加速上限约 `1.283x`，高于冻结的
`1.15x` 性能门，足以先检验质量而不是继续做无效微优化。

## 执行和停止规则

1. 在不打开 sensor depth/label 的阶段只物化一个 120 帧缓存并锁定 SHA-256。
2. 按既有 P1-R1 协议只评价一次，不更改任何门限。
3. 任一质量门失败即停止，不做 Android 转换、QNN profile，也不以速度挽救失败。
4. 只有全部质量门通过，才允许导出完全相同的图并运行固定设备 QNN 与全链路 P95；
   实测 full-chain speedup 必须至少 `1.15x`。

证据上限始终是 consumed Development engineering regression，不是产品或安全授权。
