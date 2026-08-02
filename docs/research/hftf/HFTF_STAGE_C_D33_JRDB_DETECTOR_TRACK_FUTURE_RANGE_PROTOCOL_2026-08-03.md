# HFTF Stage C D33：JRDB detector-track future-range replication

日期：2026-08-03

证据角色：Development / detector-bound short-future mechanism replication

研究主线：不变

默认 App：不变

## 单一科学变量

D32 已用 JRDB annotation box + native identity 建立约一秒 future-range 正机制。
D33 保持 future estimand、source rule 与 truth 不变，只替换 source measurement：

- D32 source：annotation stitched-2D box + native `label_id`；
- D33 source：stitched RGB 上冻结 YOLO11n person detector + ByteTrack identity。

source producer 不读取 2D/3D annotations；evaluation 才用 current-frame 2D IoU
把 detector track occurrence 关联到 native identity，并读取该 identity 的
`+15 frames` 3D range truth。

## 冻结 source producer

输入仍为 D32 的四个 120-frame JRDB observation packets。按 packet 已记录的
image member/CRC/SHA 恢复 480 张 stitched RGB；缓存缺失或网络中断是可修复工程
故障。

detector：

- weights：既有 `yolo11n.pt` 固定 SHA；
- Ultralytics：`8.4.102`；
- class：person only；
- `imgsz=640`、confidence `0.10`、NMS IoU `0.50`、`max_det=50`；
- 3760×480 panorama 固定切成五个 960×480 overlapping tiles，x starts：
  `0/700/1400/2100/2800`；
- tile detections 映射回 panorama x 后做一次全局 NMS `0.50`；
- 不使用 annotation 调 detector threshold、tile 或 NMS。

tracker：

- Ultralytics ByteTrack；
- `track_high_thresh=0.25`；
- `track_low_thresh=0.10`；
- `new_track_thresh=0.25`；
- `track_buffer=30`；
- `match_thresh=0.80`；
- `fuse_score=true`；
- 每个 sequence 独立 reset。

source output 只含 sequence/frame/timestamp、detector track id、bbox 与 confidence。

## 冻结 source decision 与 future truth

source rule 原样继承 D32：

- 同一 detector track 七个连续帧；
- `log(box_height)` causal OLS；
- slope `>=0.2/s` 且六次高度全增：
  `CONFIRM_APPROACH`；
- slope `<=-0.2/s` 且六次高度全减：
  `CONTRADICT_APPROACH`；
- 其他：`ABSTAIN`。

evaluation association：

- 每帧 detector bbox 与 native annotation bbox 做 one-to-one Hungarian IoU；
- minimum IoU `0.30`；
- 只用 current occurrence 的 match 决定 future native identity；
- history 内 native-ID purity 只做诊断，不筛选 evidence。

future truth 原样继承 D32：

- horizon：current frame `+15`；
- same native identity；
- `center_base_link_m` 三维 range；
- signed rate deadband：`±0.1 m/s`；
- confirm 预测 `APPROACHING`，contradict 预测 `NOT_APPROACHING`。

## 可判定性 gate

任一不足则终态
`D33_JRDB_DETECTOR_TRACK_FUTURE_RANGE_NOT_EVALUABLE`，不构成科学负结果：

1. 480/480 source frames 完成 detector/tracker；
2. detector-current/native 2D matches 至少 400；
3. 有七帧 history、current native match 和 future truth 的 opportunities 至少 400；
4. non-abstain evidence 至少 60；
5. evidence 至少覆盖 15 个 native sequence-bound identities；
6. 至少 3/4 sequences 各有至少 10 rows evidence；
7. confirm 与 contradict 各至少 15 rows。

## 支持 gate

在可判定前提下全部满足才支持 detector-bound mechanism：

1. pooled overall precision 至少 `0.85`；
2. confirm precision 至少 `0.80`；
3. contradict precision 至少 `0.80`；
4. 两方向相对 source-opportunity prevalence 的 lift 各至少 `0.10`；
5. 至少 3/4 evidence-sufficient sequences precision 至少 `0.75`。

通过：

`D33_JRDB_DETECTOR_TRACK_FUTURE_RANGE_SUPPORTED`

可判定但未通过：

`D33_JRDB_DETECTOR_TRACK_FUTURE_RANGE_NOT_SUPPORTED`

无论总终态如何，保留 detector coverage、association IoU、track history purity、
方向/序列 precision、prevalence 与 lift。

## 工程失败与重跑

下载、路径、JPEG、CUDA、OOM、dependency、serialization 或中断错误均非科学终态；
修复后在同一冻结配置下重跑。source images、detector tracks 与 report 都是可重建
Development artifacts，允许原子替换，不使用 one-shot/source burning/fsync
interruption 关闭 cohort。

## 主张边界

通过只建立 detector + offline ByteTrack 在四个 JRDB 短序列上的
short-future mechanism。它仍不建立 event utility、Android realtime、跨设备
泛化、默认 App、产品效果或 human safety。

D33 若支持，下一步进入 Android shadow state estimator canary；若不支持，则把
瓶颈明确定位为 person detection/association，而不是否定 D32 的 future-motion
hypothesis。
