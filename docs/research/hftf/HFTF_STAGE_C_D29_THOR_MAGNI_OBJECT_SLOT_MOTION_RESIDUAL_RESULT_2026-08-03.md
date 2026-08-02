# HFTF Stage C D29：THOR-MAGNI object-slot motion residual result

日期：2026-08-03

证据角色：Development / explicit object-motion bottleneck canary

研究主线：不变

默认 App：不变

## 结论

D29 的 cache、5 个 source-heldout folds 和 5 个 paired models 全部有效完成。冻结
13 项 gate 只通过 teacher-MAE noninferiority 与 monotonicity 两项，终态为：

`D29_THOR_MAGNI_OBJECT_SLOT_MOTION_RESIDUAL_INCREMENT_NOT_SUPPORTED`

history motion residual 相对共享 current-static base：

| metric | mean delta | 正折 |
|---|---:|---:|
| source-macro direction×horizon AUROC | -0.04125 | 2/5 |
| source-macro direction×horizon AP | -0.02507 | 2/5 |
| source-macro safest-direction accuracy | +0.00560 | 2/5 |
| pooled direction×horizon AUROC | -0.04875 | 0/5 |
| pooled direction×horizon AP | -0.02593 | 1/5 |
| source-macro teacher MAE | +0.14027 m | 5/5 |

这不是控制面失败：模型确实学到较低训练 loss、输出合法并完整落盘，但显式
current-person box 内的 backward-flow residual 没有恢复 D27 的世界运动增量。

## object-slot opportunity

冻结 detector/cache 结果：

- 530 anchors 中 393 个有 current person detection；
- anchor coverage `74.15%`，低于冻结 `80%` gate；
- 1,165 个 raw detections，选择 1,161 个 slots；
- 每 anchor 平均 `2.19` slots；
- 只有 3 个 anchors 超过 8-slot 上限；
- selected slots 的四 lag mean warp-valid fraction 为 `90.95%`。

因此主要 opportunity 缺口是 current detector coverage，而不是 8-slot 截断或
backward-warp 大量越界。但 coverage 不是唯一失败：在保持未检测 sample
`history=current` 的结构约束下，pooled AUROC 五折全部为负，说明已检测 slots 的
motion residual 也没有形成稳定 ranking。

## 五折

| fold | AUROC delta | AP delta | safest-choice delta | teacher MAE delta |
|---:|---:|---:|---:|---:|
| 0 | +0.04521 | -0.01356 | -0.03333 | +0.24362 m |
| 1 | -0.00901 | +0.00615 | +0.04888 | +0.05967 m |
| 2 | -0.19003 | -0.10519 | -0.00123 | +0.07414 m |
| 3 | -0.05908 | -0.07618 | +0.06667 | +0.10082 m |
| 4 | +0.00665 | +0.06344 | -0.05298 | +0.22310 m |

没有一个 fold 的 AUROC、AP 与 safest-choice 三项同时为正。fold 2 的大幅负向也使
结果不能被描述为仅由 coverage 稀释的弱正信号。

## 三方向

| direction | source-macro horizon-macro AUROC delta | 正折 | AP delta | 正折 |
|---|---:|---:|---:|---:|
| left | -0.04549 | 3/5 | -0.02951 | 1/5 |
| center | -0.04307 | 1/5 | -0.00686 | 3/5 |
| right | -0.03519 | 1/5 | -0.03883 | 2/5 |

三个方向 mean 均为负，D29 不产生方向级正终态。D26 right-direction signal 与 D27
三方向 information ceiling 仍保留在各自证据层，不由 D29 撤销。

## D29 实际关闭什么

D29 关闭的是这一组冻结 recipe：

> 低分辨率 current YOLO-person boxes + box 内 current→history RAFT statistics +
> 14,104-parameter masked-DeepSets motion residual，可以直接恢复 D27 field
> increment。

它没有关闭：

- D27 source-native kinematic information ceiling；
- 全分辨率 detection 的 coverage opportunity；
- detector box 与 source-native body 的显式几何 correspondence；
- metric depth / calibrated bearing / track-conditioned state estimation；
- 独立 3D-person trajectory source 上的结构化运动模型。

但当前结果也不支持“只把 detector 换成全分辨率就重跑同一 residual”。在新增模型
前，必须先证明 2D box bearing/scale 与 source-native body bearing/distance 有稳定
的 source-heldout measurement relation。

## 工程与复现

- object-slot cache SHA-256：
  `aa9d0f28b1e050105086fee3078002862fd0d21d06e5bd4aa12ecc950ec451f7`；
- result report SHA-256：
  `22b910c1500beb7683241ea69fc0f5a3a5fa88747ebed06d28a1a10100ba1206`；
- 5 checkpoints 全部落盘；
- 14,104 trainable parameters，seed17、200 fixed epochs；
- current/history 共用 static base，motion head 零初始化；
- 无检测时结构上强制 history=current；
- monotonicity violations：0；
- 无 OOM、路径、parser、cache、serialization 或 fsync invalid；
- 未搜索 detector confidence、slot count、flow statistic、network capacity、
  epoch、seed 或 gate。

```text
artifacts.local/evidence/hftf/
  stage-c-d29-thor-magni-object-slots-v0/
    object_slots.npz
    object_slots.npz.json
    object_slots.npz.sha256
  stage-c-d29-thor-magni-object-slot-motion-residual-v0/
    report.json
    report.json.sha256
    checkpoints/
```

## 下一科学变量

下一步不是继续训练 field head，而是冻结 measurement correspondence diagnostic：

1. 从 source-native current body positions 与 wearer motion 得到每个人的相对
   bearing/distance；
2. 与 current YOLO boxes 的 x-center/height/area 做不读取 future 的匹配；
3. 按 source-heldout 披露 detection-to-body coverage、bearing error、distance-rank
   correlation 与 identity ambiguity；
4. 只有 measurement relation 稳定，才允许构建显式 world-state filter；
5. 若 relation 不稳定，则转向有原生 2D/3D identity binding 的独立 person
   trajectory source，不在 THOR 上继续靠 field loss 间接猜 correspondence。
