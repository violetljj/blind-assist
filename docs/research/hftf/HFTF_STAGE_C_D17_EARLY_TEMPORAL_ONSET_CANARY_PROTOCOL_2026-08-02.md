# HFTF Stage C D17：early-temporal true-onset canary protocol

日期：2026-08-02

证据角色：Development / source-diverse synthetic representation canary

研究主线：不变

默认 App：不变

## 待检验主张

D16 已关闭 frozen single-frame feature 加 post-hoc temporal residual family。D17
只检验一个新变量：在高分辨率低层视觉特征上先编码五帧有序运动，再让其进入后续
完整空间编码器，是否能改善 true-future-onset field。

它不是 D5 的 late fusion。五帧先共享 MobileNet block 0；四个相邻时刻的 signed
feature difference 经过 depthwise-separable 3D convolution，在 block 1 之前写回
current feature。其后所有 MobileNet blocks 都在 motion-conditioned representation
上运行。训练目标直接是 D16 的 near/far × body/head × 6×6 cell onset。

## 冻结比较

- 数据：D16 的 495 samples、15 environments 和原三折 environment assignments；
- candidate：真实 `[-.8,-.6,-.4,-.2,0] s` 五帧；
- comparator：同一网络把 current frame 重复五次；
- 两臂逐折、逐 seed 共享完全相同初始化、参数量、增强和 batch 顺序；
- comparator 的相邻 feature difference 严格为零；
- ImageNet MobileNet 全编码器可训练，encoder LR `2e-5`，temporal/head LR `2e-4`；
- 30 epochs、batch 8、AdamW、weight decay `1e-4`；
- cell-eligibility mask、train-fold positive weight、environment-balanced BCE；
- fixed final epoch，不在 held-out environments 选模。

## Seed-17 canary gate

主指标是 held-out `environment × target` macro cell AUROC/AP 的
history-minus-current：

1. 三折 mean AUROC 至少 `+0.010`，AP 至少 `+0.005`；
2. AUROC 与 AP 各至少 `2/3` folds 为正；
3. 四个 cell targets 中至少三个的 AUROC 和 AP mean 都为正；
4. sample-level macro AUROC 与 AP 均不得低于 `-0.005`。

全部通过才原样扩展 seeds `23/41`。任一失败即停止本 early unaligned temporal
convolution candidate；不得在本轮修改 epoch、LR、宽度、loss、聚合或门槛救援。

即使通过，它也只授权同设计多 seed 复现；不等于真实来源迁移、系统效用、主线替换
或 App/安全证据。
