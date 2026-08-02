# HFTF Stage C D17：early-temporal true-onset canary result

日期：2026-08-02

证据角色：Development / source-diverse synthetic representation canary

研究主线：不变

默认 App：不变

## 结论

D17 没有通过冻结的 seed-17 canary gate：

`D17_EARLY_TEMPORAL_ONSET_CANARY_NOT_SUPPORTED`

early-temporal history 对“样本是否出现 onset”产生了三折一致的小幅 AUROC 正方向，
但没有把运动稳定定位到正确的 6×6 cell。最重要的分裂是：

| metric | mean delta | 正折 |
|---|---:|---:|
| environment-macro cell AUROC | -0.00128 | 1/3 |
| environment-macro cell AP | +0.00766 | 2/3 |
| pooled target-macro cell AUROC | -0.00789 | 2/3 |
| pooled target-macro cell AP | -0.00116 | 2/3 |
| sample-macro AUROC | +0.00580 | 3/3 |
| sample-macro AP | +0.00189 | 1/3 |

primary AP 的 effect 与正折门通过，sample non-inferiority 也通过；primary AUROC、
正折数和 target breadth 失败，因此不扩 seeds `23/41`。

这个结果不是“时序没有信息”。更准确的机制结论是：

`D17_COARSE_ONSET_PRESENCE_SIGNAL_WITHOUT_STABLE_CELL_LOCALIZATION`

它将下一变量收窄到 correspondence/alignment。D17 的低层 3D temporal convolution
仍直接比较未对齐的相邻 feature maps；fold 0 的四个 cell targets 大幅反向，而
fold 2 多数为正，符合 source-dependent ego-motion/parallax 把运动写入错误位置的
解释。下一候选应固定 D17 的 onset task、fold、loss 与 current comparator，只引入
显式 dense correspondence 后再融合，不再通过 width、epoch、seed 或门槛搜索救援。

## 冻结设计与执行

- 495 samples、15 environments、D16 原三折；
- 真实五帧 history 对 identical repeated-current comparator；
- 两臂每折共享相同初始化、1,003,956 trainable parameters；
- 2,448 temporal parameters；
- MobileNet block 0 后形成四个 signed adjacent differences；
- depthwise-separable 3D fusion 在其余 MobileNet blocks 之前写回 current feature；
- 直接监督 near/far × body/head × 6×6 cell onset；
- 30 fixed epochs，不做 held-out checkpoint selection；
- 三折 history 的最终训练 loss 均低于 current，但只作为优化诊断，不作为效果证据。

所有训练与评价完整完成，没有路径、parser、序列化、网络或中断造成的工程无效。

## Gate

冻结 gate 要求：

1. environment-macro cell AUROC mean 至少 `+0.010`；
2. cell AP mean 至少 `+0.005`；
3. 两项各至少 `2/3` folds 为正；
4. 至少 `3/4` targets 的 cell AUROC/AP mean 同时为正；
5. sample-macro AUROC/AP 不低于 `-0.005`。

实际仅 AP effect、AP positive folds 与 sample non-inferiority 通过；cell AUROC 与
target breadth 未通过。四个 target 没有一个在三折 mean 上同时得到正 AUROC 和正
AP。

## 可复现证据

```text
artifacts.local/evidence/hftf/
  stage-c-d17-tartanground-early-temporal-onset-canary-v0/
    report.json
    report.json.sha256
    checkpoints/fold-{0,1,2}/seed-17-history.pt
```

- report SHA-256：
  `c4f04bd47dcc968352a4d5c3816641b471ec9fbbc65d692be6723defd2502dda`
- fold 0 checkpoint：
  `b9e71934a66c6e322f99a8bd47d0c952d2dd9f5ad4e5dde885ae2598312a7a54`
- fold 1 checkpoint：
  `e5c9a46190eac02797bc8716387a06f244fe202c7c85897221b1b469549788b8`
- fold 2 checkpoint：
  `a45d062019f60ac362b5ed6cc12b6b5501958bfa3bcffd2b5c121a186c2c163d`

## 下一科学变量

D18 只测试 dense alignment：

- 对每个 current→history pair 用固定 pretrained RAFT-small 构造 backward flow；
- 在 MobileNet block-0 resolution 把 history feature warp 到 current coordinates；
- 保留 D17 的相同 onset targets、三折、优化预算与 repeated-current comparator；
- 对 aligned-history residual 做相同早期 temporal fusion。

D18 若仍不能恢复 cell localization，则停止“RGB optical-flow alignment +
early residual fusion”这一家族，转向 geometry-teacher distillation 或直接预测
future occupancy dynamics；不继续搜索 RAFT summary、卷积宽度或训练 seed。
