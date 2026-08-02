# HFTF Stage C D22：THOR-MAGNI dense-flow dynamics transfer result

日期：2026-08-02

证据角色：Development / independent-source representation transfer

研究主线：不变

默认 App：不变

## 结论

完整的 broad-transfer gate 未通过：

`D22_THOR_MAGNI_DENSE_FLOW_TRANSFER_CANARY_NOT_SUPPORTED`

但 D22 在独立 THOR-MAGNI source 上建立了一个明确、target-specific 的
Development 正结果：

`D22_PROXIMITY_SOURCE_MACRO_TRANSFER_SIGNAL_SUPPORTED_DEVELOPMENT_ONLY`

history-minus-current：

| metric | mean delta | 正折 |
|---|---:|---:|
| target-macro source-macro AUROC | +0.01638 | 3/5 |
| target-macro source-macro AP | +0.02376 | 3/5 |
| proximity source-macro AUROC | +0.03669 | 4/5 |
| proximity source-macro AP | +0.03660 | 4/5 |
| corridor source-macro AUROC | -0.00393 | 2/5 |
| corridor source-macro AP | +0.01091 | 3/5 |
| pooled target-macro AUROC | -0.01359 | 1/5 |
| pooled target-macro AP | +0.00409 | 1/5 |

完整 7 项 gate 通过 5 项。通过的是 source-macro AUROC/AP effect、两项
positive-fold consistency 与 pooled AP noninferiority；失败的是：

- 两个 targets 必须在 source-macro AUROC/AP 上同时为正，但 corridor AUROC 为负；
- pooled target-macro AUROC `-.01359`，低于 `-.005` noninferiority floor。

因此不能声称 D20 的 broad dense-flow dynamics 已跨 source 复现，也不扩展完整
D22 gate 到 seeds `23/41`。

## 正结果应保留在哪一层

proximity 并非结果后新增的 target，而是 D12/D22 预先冻结的两个 target 之一。
它在 source-session-macro AUROC/AP 上同时达到 `+.0367` 左右，且两项均 4/5 folds
为正；pooled proximity AUROC/AP 为 `-.00100/+.00723`，也满足 `-.005`
noninferiority。

这支持一个窄而有机制意义的解释：aligned dense flow 对“当前尚未近距、但短期内将
进入 1.25 m proximity”的运动接近信号具有跨 source-session Development 增量。
它没有支持更宽的 corridor intrusion；后者可能需要显式 ego-path/ground-plane
结构，而不是继续使用局部 image-flow dynamics。

这个 target-specific 正结果不是系统效用、主线晋级或安全证据，但也不能因为 broad
gate 失败而被改写成 `NOT_EVALUABLE` 或“模型完全无效”。

## 工程故障与科学结果分离

首次训练在一个两个 target 都无 eligible sample 的随机 mini-batch 上抛出
`ValueError`。这是 batch loss 对空有效集合处理错误：

- held-out metrics 尚未产生；
- samples、folds、seed、epoch、模型、loss 权重与 gate 均未改变；
- 修复为跳过空有效 batch 后从头重跑。

因此该失败是可修复工程无效，不烧毁 D12 cohort，也不进入科学终态。修复后的 10 个
training runs 全部完成。

## 流与复现证据

current→history RAFT cache：

- samples：1,078；
- pairs：4,312；
- shape：`[1078,4,2,64,112]` float16；
- bytes：123,633,792；
- SHA-256：
  `c959f81f6077fcd695a9a652b397dd9d72873da18474aa40b29d08a8f71d8557`。

固定每 5 个 sample 抽一个的 216-sample 方向诊断：

- valid warp fraction：`0.85275`；
- photometric L1 before：`0.15305`；
- after：`0.06485`；
- relative reduction：`57.63%`。

最终比较使用 1,078 samples、19 source sessions、5 folds、seed17、10 个 paired
training runs、相同 1,004,134 trainable parameters 与固定 30 epochs。

```text
artifacts.local/evidence/hftf/
  stage-c-d22-thor-magni-backward-raft-flow-v0/
  stage-c-d22-thor-magni-dense-flow-transfer-v0/
```

- report SHA-256：
  `7604ad77405ec5661abdfbe81ecd52020e7a2165687b998dc7d5deb062c2d03b`。

## 下一科学变量

不再继续 broad D22 gate，也不把 corridor 切成 proximity 后宣称“原协议通过”。
proximity 4/5-fold 双指标正信号只能作为新的 Development 假设。若继续，应另行冻结
同一数据、同一双 target 训练下的 proximity-only 多种子稳定性检查：

- seed17 作为已观察的 hypothesis-generating 结果；
- seeds23/41 只检验同一 proximity 指标是否保持方向与折一致性；
- 即使通过，也只建立 target-specific multi-seed Development robustness，随后才
  能进入真实事件层；
- corridor/broad-transfer 负结果保持不变。
