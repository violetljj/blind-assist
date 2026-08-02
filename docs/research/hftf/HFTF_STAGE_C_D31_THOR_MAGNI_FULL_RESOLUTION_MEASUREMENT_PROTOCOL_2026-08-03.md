# HFTF Stage C D31：THOR-MAGNI full-resolution measurement replication

日期：2026-08-03

证据角色：Development / full-resolution current measurement replication

研究主线：不变

默认 App：不变

## 单一变量

D30 已建立强 bearing signal，但 overall 受低分辨率 detector opportunity、
nearest-person coverage 与 source-macro distance rank 限制。D31 只改变：

- D30/D29：从 `128×224` D10 cache 检测 current person；
- D31：从 hash-bound 原视频按同一 `anchor_scene_frame` 解码原分辨率 current RGB。

其余全部冻结：

- 同一 530 anchors、19 sources、五 folds；
- 同一 YOLO11n weights SHA、Ultralytics `8.4.102`；
- `imgsz=640/conf=0.10/NMS=0.50/class=person/max_det=30`；
- 同一 top-8 `confidence × sqrt(area)` selection；
- 同一 `Helmet_*` Visitor/Carrier person truth；
- 同一 `±50°` visible proxy、world-left→image-left 映射；
- 同一 Hungarian x-error cost、`0.25` acceptance；
- 同一八项 D30 gate；
- 不读取 future outcome，不训练 state/risk model；
- 不搜索 detector threshold、FOV、distance cap 或 assignment。

## 复现 gate

1. detector 与 visible person 同时存在 anchors 至少 300；
2. accepted / assigned pairs 至少 `0.60`；
3. nearest visible person accepted coverage 至少 `0.60`；
4. source-macro x Pearson 至少 `0.50`；
5. source-macro bearing MAE 不超过 `15°`；
6. source-macro height-vs-inverse-distance Spearman 至少 `0.30`；
7. distance Spearman 至少 3/5 folds 为正；
8. 至少 15/19 sources 有 5 对以上 assigned measurements。

通过终态：

`D31_THOR_MAGNI_FULL_RESOLUTION_MEASUREMENT_RELATION_SUPPORTED`

失败终态：

`D31_THOR_MAGNI_FULL_RESOLUTION_MEASUREMENT_RELATION_NOT_SUPPORTED`

## 边界

D31 通过只授权下一步冻结显式 person bearing-distance state filter canary。它不建立
identity truth、velocity、future collision、非 person objects、事件效用、App 或安全
主张。失败则停止 THOR current-box measurement route，转向原生 2D/3D identity
binding source。
