# HFTF Stage C D29：THOR-MAGNI object-slot motion residual protocol

日期：2026-08-03

证据角色：Development / explicit object-motion bottleneck canary

研究主线：不变

默认 App：不变

## 假设

D27 已证明 source-native body velocity 有强 information ceiling；D28 已否定现有
whole-frame network 直接蒸馏该 field 的 source-general increment。D29 检验更窄
的假设：

> 如果先把当前人体实例显式变成 object slots，再用 current→history flow 在每个
> slot 内建立历史对应，低容量 motion residual 能否恢复 D27 的增量？

这改变的是可识别变量与归纳偏置，不是换 backbone、调 D28 loss 或扩大 head。

## 冻结 object-slot cache

输入严格绑定 D28 使用的同一批 RGB/flow：

- 530 D26 eligible anchors、19 source sessions、五 folds；
- RGB：D10 `128×224` 五帧 cache；
- flow：D22 current→four-history RAFT-small，`64×112`；
- detector：本地 COCO-pretrained YOLO11n，
  SHA-256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`；
- Ultralytics `8.4.102`、`imgsz=640`、person class only、confidence `0.10`、
  NMS IoU `0.50`、`max_det=30`、无 augmentation。

只在 anchor current frame 做 detection。按
`confidence × sqrt(normalized box area)` 排序，最多保留 8 个 slots；不足补零并
显式 mask，不因缺失检测删除 sample。

每个 slot 固定 34 维：

- 6 个 current features：center-x、bottom-y、width、height、sqrt-area、
  confidence；
- 对四个历史 lag 各 7 个 causal features：
  - 框内 median raw flow x/y；
  - 相对全帧 median 的 residual flow x/y；
  - current pixels backward-warp 后的 log width/height ratio；
  - valid-warp fraction。

所有位置、尺度和 flow 均按 frame width/height 归一化；log ratio clip 到
`[-2,2]`。任何无框、无有效像素或越界只产生 mask/validity unknown，不删除
anchor、不改 label。

## 冻结 student

每个 fold 只训练一个 paired model：

- static slot encoder 只读 6 个 current features；
- motion slot encoder 读完整 34 维；
- 两个 encoder 均为共享 slot MLP + masked mean/max pooling；
- current field 由 static pooled feature 产生；
- history field 在同一 static logits 上加零初始化 motion residual logits；
- 两个输出都用 `-sigmoid` 约束到 `[-10m,0]`，再沿 horizon cumulative max；
- target 分别为 D27 current-static 与 history-kinematic fields；
- source-balanced paired Smooth L1，两个 target loss 等权；
- seed17、200 epochs、batch32、AdamW `lr=1e-3 / weight_decay=1e-4`、
  fixed final epoch；
- 确定性水平翻转同步变换 slot x/flow x，并交换 left/right teachers；
- 不搜索 detector threshold、slot count、features、pooling、loss、epoch、seed、
  residual scale 或 gate。

## 冻结评价与 gate

报告 detector/object-slot opportunity：

- current person-detection anchor coverage；
- slot count、8-slot saturation、各 lag valid fraction；
- 不以 coverage 低为科学负结果；只有 cache/input 破损才是 engineering invalid。

held-out 上评价 history-current：

1. source-macro/pooled direction×horizon AUROC/AP；
2. left/center/right source-macro horizon-macro AUROC/AP；
3. source-macro safest-choice accuracy；
4. 对各自 D27 teacher 的 source-macro/pooled MAE；
5. horizon monotonicity。

D29 仅在以下全部满足时支持：

1. detector anchor coverage 至少 `0.80`；
2. source-macro AUROC/AP mean delta 至少 `+0.010/+0.005`；
3. AUROC/AP 各至少 3/5 folds 为正；
4. AUROC/AP 各至少 2/3 directions 的五折 mean 为正；
5. safest-choice mean delta 至少 `+0.020`，且至少 3/5 folds 为正；
6. pooled AUROC/AP mean delta 均不低于 `-0.005`；
7. history teacher MAE 相对 current teacher MAE 的 mean 增量不超过 `+0.25m`；
8. monotonicity violations 为 0。

通过终态：

`D29_THOR_MAGNI_OBJECT_SLOT_MOTION_RESIDUAL_INCREMENT_SUPPORTED`

失败终态：

`D29_THOR_MAGNI_OBJECT_SLOT_MOTION_RESIDUAL_INCREMENT_NOT_SUPPORTED`

## 边界

D29 通过只建立 COCO-person + RAFT object-slot representation 在 THOR-MAGNI
source-heldout tracked-body proxy 上的 Development increment。它不建立检测真值、
人体身份真值、真实事件提醒效用、静态障碍/路缘/foot/head coverage、App 或安全
主张。

D29 失败只关闭这一组冻结的 current-box + within-box backward-flow + low-capacity
residual recipe。它不撤销 D27，也不授权在同一 outcomes 上搜索 detector confidence、
slot 数、flow statistic 或网络容量。
