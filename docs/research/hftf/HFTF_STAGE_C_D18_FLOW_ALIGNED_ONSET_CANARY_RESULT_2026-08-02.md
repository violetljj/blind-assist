# HFTF Stage C D18：flow-aligned true-onset canary result

日期：2026-08-02

证据角色：Development / source-diverse synthetic aligned-representation canary

研究主线：不变

默认 App：不变

## 结论

D18 没有通过完整 seed-17 gate：

`D18_FLOW_ALIGNED_ONSET_CANARY_NOT_SUPPORTED`

但显式 dense alignment 建立了两个应独立保留的 Development representation 正结果：

- `D18_FLOW_ALIGNMENT_RESCUES_POOLED_CELL_LOCALIZATION_SIGNAL_DEVELOPMENT_ONLY`
- `D18_FAR_HEAD_ALIGNED_ONSET_SIGNAL_SUPPORTED_DEVELOPMENT_ONLY`

history-minus-current：

| metric | mean delta | 正折 |
|---|---:|---:|
| environment-macro cell AUROC | +0.00031 | 1/3 |
| environment-macro cell AP | +0.00550 | 2/3 |
| pooled target-macro cell AUROC | +0.00549 | 3/3 |
| pooled target-macro cell AP | +0.00886 | 3/3 |
| sample-macro AUROC | +0.02124 | 3/3 |
| sample-macro AP | +0.00469 | 2/3 |
| far-head cell AUROC | +0.01706 | 3/3 |
| far-head cell AP | +0.01802 | 3/3 |

相对 D17，pooled cell AUROC/AP 从 `-0.00789/-0.00116` 变为
`+0.00549/+0.00886`，sample AUROC 从 `+0.00580` 增至 `+0.02124`。
四个 targets 中三个的 mean AUROC/AP 同时为正，target breadth gate 通过。

因此不能把 D18 简化为“RAFT 又失败”。更准确的结论是：correspondence 是真实机制
变量，能把 coarse presence signal 转成 pooled cell localization；但效应在五个
held-out environments 内不够均匀，尚未达到 environment-robust general signal。

## 为什么完整 gate 仍失败

冻结 gate 的 environment×target macro cell AUROC 要求：

- mean 至少 `+0.010`；
- 至少 `2/3` folds 为正。

实际仅 `+0.00031`、1/3 folds 为正。AP effect、AP positive folds、3-target
breadth 和两项 sample non-inferiority 均通过。失败范围因此是跨环境 AUROC
幅度/一致性，不是 pooled localization、sample presence 或 far-head target。

不扩 seeds `23/41`，不把 far-head 改成新的主 target，也不通过改门槛救援。

## Alignment mechanics

固定 pretrained torchvision RAFT-small `C_T_V2` 物化全部 495×4 =
1,980 个 current→history backward flows：

- low-resolution shape：`495×4×2×64×112`；
- float16 bytes：56,770,688；
- flow SHA-256：
  `10be7dfe3f50b32a89d98fb48fdfb8f72900af078433ab578276cb3814ad13df`。

在固定每五个样本抽一个的 99-sample 方向检查中：

- valid sampling fraction mean：`0.87984`；
- warp 前 photometric L1：`0.11301`；
- warp 后 photometric L1：`0.05766`；
- relative reduction：`48.98%`。

这证明 current→history flow 被用于正确的 backward sampling 方向。

首次 extraction 的 1,980 对推理已完成，但 Windows 对只读文件句柄调用 `fsync`
产生 `OSError: [Errno 9] Bad file descriptor`。这是可修复工程失败：把句柄改为
`r+b` 后原协议重跑并成功物化，不烧毁 cohort，也不构成算法证据。

## 模型执行

- 与 D17 相同 1,003,956 trainable parameters 和 2,448 temporal parameters；
- history feature 先 warp 到 current coordinates；
- aligned-history-minus-current residual 再进入相同 early 3D stem；
- current comparator 使用 repeated current + zero flow，residual 精确为零；
- D16 原三折、seed17、30 fixed epochs、相同 loss/LR/gate；
- 三折 history final train loss 均低于 current，但不作为 effect gate。

## 可复现证据

```text
artifacts.local/evidence/hftf/
  stage-c-d18-tartanground-backward-raft-flow-v0/
  stage-c-d18-tartanground-flow-aligned-onset-canary-v0/
```

- canary report SHA-256：
  `203692578b678d4477f9ac2cbb24769b76335f83790c2a341fb99fb1c8aa31c7`

## 下一科学变量

D19 保留 D18 的 dense alignment，不再改变 correspondence；只增加
geometry-teacher dynamics pretraining：

1. 先用 current/near/far 的 body/head known-risk fields 训练对齐表征；
2. 再在相同 train folds 内用 D16 cell onset fine-tune；
3. 仍与 repeated-current 同容量对照比较；
4. 仍用 D18 的原 environment-robust gate。

这检验 D18 的跨环境幅度不足是否来自 sparse onset supervision，而不是继续搜索
RAFT、flow direction、stem width、seed 或 threshold。
