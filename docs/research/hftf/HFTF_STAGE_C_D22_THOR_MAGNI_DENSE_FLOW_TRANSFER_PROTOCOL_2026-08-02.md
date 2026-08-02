# HFTF Stage C D22：THOR-MAGNI dense-flow dynamics transfer protocol

日期：2026-08-02

证据角色：Development / independent-source representation transfer

研究主线：不变

默认 App：不变

## 科学问题

D20 在 TartanGround 上建立了跨四个 target、pooled cell、sample 与
environment-macro AP 的广泛 true-onset 正信号，但完整 gate 的
environment-macro AUROC effect 未达到 `+.010`。D21 继续更换轻量 temporal
operator 没有解决环境异质性，因此停止同一 operator family。

D22 不再调 D20 模型，而回答更关键的问题：

> D20 的 dense-flow dynamics 增量能否在独立的 THOR-MAGNI source 与
> current-negative true-future-onset targets 上复现？

该结果只支持或否定 representation transfer，不直接回答真实助盲事件效用、研究
主线替换、App 或安全。

## 冻结输入

- D12 的 1,078 个 samples、19 个 source sessions 与五个 held-out folds；
- targets：
  - current-negative future proximity onset；
  - current-negative future corridor-intrusion onset；
- D10 已物化的五帧 `128×224` RGB cache；
- pretrained MobileNetV3-small；
- pretrained RAFT-small current→四个 history frames 的 dense backward flow，
  下采样为 `4×2×64×112` float16；
- seed17 canary。

路径、解析、缓存、落盘或中断失败属于可修复工程故障；在两个 arm 的 held-out
metrics 产生前修复并重跑，不烧毁 D12 samples。

## 冻结比较

两臂共享相同初始化、参数、训练预算和 target head：

- current：current RGB 重复五次，flow 精确为零，20-channel dynamics tensor
  精确为零；
- history：五帧 RGB，加 current→history warp 后的 16-channel feature residual、
  normalized flow x/y、magnitude 与 warp validity。

四步 dynamics 继续使用 D20 的 depthwise 3D collapse，不改为 ConvGRU。两臂都训练
完整 MobileNet encoder；encoder LR `2e-5`，temporal/head LR `2e-4`，
AdamW、weight decay `1e-4`、batch 8、固定 30 epochs。source-session × target
在训练损失中等权；只评价固定最后一轮，不做 held-out model selection。

## Seed-17 gate

history-minus-current 必须全部满足：

1. target-macro source-session-macro AUROC mean 至少 `+0.010`；
2. target-macro source-session-macro AP mean 至少 `+0.005`；
3. AUROC/AP 各至少 `3/5` folds 为正；
4. proximity 与 corridor 两个 target 的 source-macro AUROC/AP mean 均同时为正；
5. pooled target-macro AUROC/AP 均不低于 `-0.005`。

全部通过才扩 seeds `23/41`。失败不修改门槛、不切 target、不调整 epoch，也不把
TartanGround 的 D20 正结果追溯改写成无效；只把它限定为 source-local Development
signal，并停止 dense-flow transfer 主张。
