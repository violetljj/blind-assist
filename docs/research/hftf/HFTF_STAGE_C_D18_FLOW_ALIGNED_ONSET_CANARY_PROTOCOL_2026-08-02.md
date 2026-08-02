# HFTF Stage C D18：flow-aligned true-onset canary protocol

日期：2026-08-02

证据角色：Development / source-diverse synthetic aligned-representation canary

研究主线：不变

默认 App：不变

## 待检验主张

D17 的 sample-level onset AUROC 三折均为正，但 cell localization 不稳定。D18
只检验这个分裂是否来自 feature correspondence 缺失：

> 把历史低层 feature 显式 warp 到 current coordinates 后，早期 temporal fusion
> 是否能在不牺牲 sample onset 的情况下恢复稳定 cell onset localization。

## 冻结改变

固定 pretrained torchvision RAFT-small `C_T_V2`，对 current→四个 history frames
计算 backward flow。在 MobileNet block-0 的 `64×112` resolution，使用该 flow
对 history feature 做 bilinear backward sampling；越界位置不进入 residual。
四个 aligned-history-minus-current residual 继续使用 D17 完全相同的 2,448 参数
3D temporal stem。

除 alignment 外全部继承 D17：

- D16 的 495 samples、15 environments、原三折与四个 6×6 onset targets；
- identical model、initialization 与 1,003,956 parameters；
- current comparator 重复 current 五次并使用 zero flow；
- 30 epochs、batch 8、相同两档 LR、AdamW 与 weight decay；
- 相同 environment-balanced masked cell BCE；
- fixed final epoch；
- 相同 environment-macro cell primary metrics 与 gate。

## Seed-17 gate

1. environment-macro cell AUROC mean 至少 `+0.010`；
2. environment-macro cell AP mean 至少 `+0.005`；
3. AUROC/AP 各至少 `2/3` folds 为正；
4. 至少 `3/4` cell targets 的 AUROC/AP mean 同时为正；
5. sample-macro AUROC/AP 均不低于 `-0.005`。

全部通过才原样扩展 seeds `23/41`。失败则关闭当前
`RGB_RAFT_ALIGNMENT_PLUS_EARLY_RESIDUAL_FUSION` 候选，转向 geometry-teacher
dynamics；不调 flow direction、RAFT weights、stem width、epoch、loss 或门槛救援。

RAFT cache 是可修复的工程产物。读取、路径、序列化或中断错误只修复重跑，不改变
科学 gate，也不烧毁 D16 cohort。
