# HFTF Stage C SANPO F0.1 student-training execution contract

日期：2026-08-01

状态：`FROZEN_BEFORE_FIRST_F0_1_STUDENT_OPTIMIZATION_STEP`

## 1. 公平比较

三个 arms 使用完全相同的 MobileNetV3-Small encoder、5-frame depthwise temporal
Conv3d、pointwise projection 与双头输出。相同 seed 在每个 arm 开始前重置全部
RNG，因此新层初始化相同；arms 只允许改变输入序列与 target horizon：

- `SF_CURRENT`：anchor RGB×5，current target；
- `SF_FUTURE`：anchor RGB×5，future target；
- `HIST_FUTURE`：精确五帧历史 RGB，future target。

## 2. 输入与增强

输入 resize 为 320×192，使用 ImageNet normalization。train 的五帧共享同一组
seeded color jitter；水平翻转概率 0.5，并同步反转 target theta 轴。禁止 crop、
rotation、perspective 或其他会破坏冻结几何格的增强。dev 只 resize/normalize。

dataloader 只能打开 `student_samples.jsonl` 与其中五个 history RGB paths，不能打开
teacher receipts、future modality 或 heldout 文件。

## 3. 优化与 checkpoint

每个 run 固定 30 epochs、batch 8、AdamW，encoder LR `3e-5`、temporal/head LR
`3e-4`、weight decay `1e-4`、无 scheduler、global norm clip 5。risk loss 在
teacher-known cells 上按 train-only、height-specific 正负比平衡；known loss 为全部
cells 的普通 BCE，二者等权相加。

每 epoch 在 exact dev reference known cells 上以 risk threshold 0.5 计算 micro F1。
每 arm/seed 选择最高 dev micro F1；平分取最早 epoch。known head 不参与 risk metric
mask。所有 run 仍完整训练 30 epochs。

## 4. 开封边界

只有 3 arms×3 seeds 全部完成、loss/参数有限、参数量相同且 9 个 checkpoint hash
封口后，才允许物化 heldout reference targets 并进行一次 ordered evaluation。

dev metrics 只能选 checkpoint，不能改阈值、来源、增强、architecture 或训练参数。
本阶段不打开 heldout RGB/teacher targets/student outputs，也不形成 student effect、
主线晋级、Android、生产或安全结论。
