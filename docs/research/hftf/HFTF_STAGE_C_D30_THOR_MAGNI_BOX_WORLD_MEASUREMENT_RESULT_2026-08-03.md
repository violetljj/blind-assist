# HFTF Stage C D30：THOR-MAGNI box-to-world measurement result

日期：2026-08-03

证据角色：Development / current-frame measurement correspondence diagnostic

研究主线：不变

默认 App：不变

## 结论

D30 不读取 future outcome、不训练风险模型。冻结 gate 通过 5/8，整体终态为：

`D30_THOR_MAGNI_BOX_WORLD_MEASUREMENT_RELATION_NOT_SUPPORTED`

整体未通过，但不是“box 与 world geometry 无关系”。bearing measurement 已出现跨
source 的强正信号；失败集中在 opportunity、nearest-body coverage 与
source-general distance rank。

## 冻结指标

| metric | result | gate | terminal |
|---|---:|---:|---|
| detector + visible-person anchors | 289 | >=300 | FAIL |
| accepted / assigned pairs | 61.88% | >=60% | PASS |
| nearest visible person accepted coverage | 46.51% | >=60% | FAIL |
| source-macro box-x / predicted-x Pearson | 0.7089 | >=0.50 | PASS |
| source-macro bearing MAE | 14.12° | <=15° | PASS |
| source-macro height / inverse-distance Spearman | 0.2867 | >=0.30 | FAIL |
| positive distance folds | 5/5 | >=3/5 | PASS |
| sources with >=5 assigned pairs | 17/19 | >=15/19 | PASS |

pooled 上：

- 501 assigned pairs，310 accepted；
- box-x / predicted-x Pearson `0.7179`；
- bearing MAE `13.85°`；
- height / inverse-distance Spearman `0.4246`；
- visible-body assignment fraction `72.82%`；
- box assignment fraction `43.15%`。

## 保留的测量层正信号

固定 `100°` horizontal FOV、固定世界左正到图像左负映射、不拟合任何参数时：

- bearing Pearson 明确超过 gate；
- bearing MAE 通过；
- 17/19 sources 可评；
- distance rank 在 pooled 为 `0.4246`，且 5/5 folds 同向为正。

因此可以保留一个窄结论：

> current person box 的 x-center 对 source-native person bearing 有稳定测量价值；
> box height 对 inverse distance 也有跨折同向信号，但 source-macro 强度略低于
> 冻结门。

这不是完整 measurement relation，也不授权 world-state filter；但它首次把 D27 的
世界运动信息与可部署视觉 observation 之间建立了直接、可解释的桥。

## 为什么 overall 仍失败

三个失败不是同一种问题：

1. `289/300` opportunity 缺口主要来自 D29 低分辨率 current detector coverage；
2. nearest-body coverage 只有 `46.51%`，说明一对一 assignment 对最近人体仍常缺失
   或被其他 box/body 占用；
3. source-macro distance Spearman `0.2867` 距 `0.30` 很近，但不能四舍五入为通过；
   pooled `0.4246` 与 5/5 正折说明它更像 source calibration/分辨率异质性，而不是
   完全没有尺度信息。

## 非 person 语义边界

输入审计确认 THOR `others` 同时含：

- `Helmet_*` visitors/carriers；
- `DARKO_Robot`；
- `LO1` carried object。

D30 因 detector 为 COCO person，只评价 `Helmet_*` 且 role 为 Visitor/Carrier 的
人体。D26/D27 collision field 会计入非 person rigid bodies，因此 D29 的 person-only
student 与完整 teacher 还存在对象类别错配。D30 不把机器人/携带物缺检记作人体测量
失败，但后续完整 field 必须另设非 person object channel。

## 工程与复现

- report SHA-256：
  `245e3625f8ea80cecdb629be9c6cd5498433ac3ae6fa58875488c95f80604c95`；
- 530 anchors、19 sources 全部评价；
- 无训练、无 future outcome read；
- 无 FOV、distance cap、assignment cost 或 acceptance threshold 搜索；
- 无路径、parser、cache、serialization 或 fsync invalid。

```text
artifacts.local/evidence/hftf/
  stage-c-d30-thor-magni-box-world-measurement-v0/
    report.json
    report.json.sha256
```

## 下一科学变量

下一步冻结 D31 full-resolution current measurement replication：

- 从 hash-bound 原视频按同一 anchor frame 解码原分辨率 current RGB；
- 使用同一 YOLO11n weights、`imgsz=640/conf=0.10/NMS=0.50`；
- 完全复用 D30 的 person-role、FOV、assignment、threshold 与八项 gate；
- 不读取 D29/D26 future outcome，不训练 state/risk model。

若 D31 提升 opportunity、nearest coverage 与 source-macro distance rank并通过，
才进入显式 bearing-distance state filter；若仍失败，则迁移到具有原生 2D/3D
identity binding 的独立 person-trajectory source。
