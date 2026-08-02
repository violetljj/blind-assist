# HFTF Stage C D9：JRDB corridor 独立数据集复现

日期：2026-08-02

证据角色：Development / independent-dataset geometric proxy

研究主线：不变

默认 App：不变

## 结论

THOR-MAGNI D8 中保留空间 layout 后出现的 corridor-specific weak signal 没有在
JRDB 上复现。固定的两个 source-pair folds 中，history-minus-current 的 corridor
AUROC 与 AP 都为负；六个 fold×seed units 中 corridor AUROC 只有 1 个为正，AP
为 0 个。

终态：

- `D9_JRDB_LOCAL_ROUTE_REPLICATION_MATERIALIZED`
- `D9_JRDB_TEMPORAL_SPATIAL_CORRIDOR_REPLICATION_NOT_SUPPORTED`
- `HFTF_FROZEN_FEATURE_HISTORY_ROUTE_STOP`

这是科学负结果，不是协议、路径、parser、hash、环境或 claim-ceiling 失败。当前
frozen MobileNet history 路线关闭，不再在 THOR/JRDB 上更换 head、epoch、seed、
crop 或阈值。D8 的 19-session 局部监督资产继续保留；未来若重开，必须是实质不同的
end-to-end spatiotemporal representation，而不是本轮 head search 的救援。

## 独立来源与绑定

本地已有四个 JRDB `image_stitched` 序列，各 120 个连续 3760×480 RGB360 frames：

- `clark-center-2019-02-28_0`
- `gates-basement-elevators-2019-01-17_1`
- `meyer-green-2019-03-16_0`
- `stlc-111-2019-04-19_0`

四个同名序列都有逐帧 source-native `labels_3d`。其中三个还已有完整 observation
packet，明确绑定 frame stem、RGB timestamp、robot pose 和 person center；标签坐标
为 `jrdb_logical_rgb360_metric_frame`，`x` 前向、`y` 横向。D9 只使用 frame stem
与 3D label，不使用人工事件标签。

物化参数：

- source FPS：15；
- history offsets：`[-12, -9, -6, -3, 0]`，覆盖 0.8 秒；
- future：30 frames / 2 秒；
- anchor stride：3 frames / 5 Hz；
- 近距：未来最小平面距离 `<=1.25 m`；
- corridor：未来任一 person center 满足 `0<x<=4 m` 且 `|y|<=0.9 m`；
- 视觉输入：RGB360 中心 1,254-pixel、约 120° 前向 crop，再 resize 到 128×224。

共物化 104 个 samples，近距正例 16、corridor 正例 42。

## 为什么是两个 source-pair folds

geometry-only census 显示单序列类别不完整：

| sequence | samples | 近距正例 | corridor 正例 |
|---|---:|---:|---:|
| clark-center | 26 | 0 | 12 |
| gates-basement-elevators | 26 | 12 | 4 |
| meyer-green | 26 | 0 | 0 |
| stlc-111 | 26 | 4 | 26 |

四折 LOSO 会产生单类 held-out target，AUROC 不可评价。D9 因此在任何 RGB 模型运行前
固定两个完整 source-pair folds：

- fold 0 held out：clark + gates；
- fold 1 held out：meyer + stlc。

两折 train/test 都包含 corridor 与 proximity 正负例。这个配对发生在
geometry-only census 后，因此只属于 Development replication，不是预注册
Confirmation。

## 模型对照

D9 直接复用 D8 的等容量 temporal-spatial 对照：

- frozen MobileNet `5×576×4×7` maps；
- current arm 将 current map 重复五次；
- history arm 读取真实五帧；
- 两臂共享相同 13,586 参数；
- 120 epochs、source-balanced BCE、seeds `17/23/41`；
- held-out fold 不参与标准化、训练或模型选择。

主检验预先限定为 D8 唯一残留信号 `future_corridor_intrusion`：AUROC 与 AP 的
seed-mean delta 必须在两个 source-pair folds 都为正，且至少 4/6 fold×seed units
为正。近距是负对照，不用于反向选择主假设。

## 结果

| 指标（history - current） | mean | 正折 | 正 unit |
|---|---:|---:|---:|
| corridor AUROC | -0.00235 | 0/2 | 1/6 |
| corridor AP | -0.00152 | 0/2 | 0/6 |
| 近距 AUROC（负对照） | +0.00781 | 2/2 | 4/6 |
| 近距 AP（负对照） | +0.00236 | 2/2 | 4/6 |

两折 corridor absolute seed means：

| fold | AUROC current → history | AP current → history |
|---:|---:|---:|
| 0 | 0.67 → 0.67（微降） | 0.59 → 0.59（微降） |
| 1 | 0.93 → 0.93（微降） | 0.94 → 0.94（微降） |

近距负对照的小正值不能取代预先指定的 corridor replication。其正例只有 16 个，
而且 D8 的等容量 pooled/spatial 近距结果都不稳定。

## 可复现证据

物化：

```text
artifacts.local/evidence/hftf/
  stage-c-d9-jrdb-local-route-replication-v0/
```

- `samples.jsonl` SHA-256：
  `a233484d9744cfb910587d42362f70b982d8bc9a2d7492be836fdad50ec0f272`
- `report.json` SHA-256：
  `dac975f1bb19adc7734dc1d3655d3f9ca04ee667b925f6594282f7ad76f22722`

spatial features：

```text
artifacts.local/evidence/hftf/
  stage-c-d9-jrdb-spatial-features-v0/features.npz
```

- SHA-256：
  `530dedf0005f709eb0009f4ddbb79cbbe27d6b3b44047d916980da12d466a8c9`

模型结果：

```text
artifacts.local/evidence/hftf/
  stage-c-d9-jrdb-temporal-spatial-corridor-replication-v0/report.json
```

- SHA-256：
  `4f3ef861f045a2f57c0e6719920da8bef7d71723a99ed85d4adaae4cd1908c4a`

## 主张边界

JRDB 是移动机器人 360° 视觉，不是盲人佩戴相机；3D person boxes 是 source-native
几何，但 corridor/proximity 仍是 robot-local Development proxy，不是提醒真值、
用户意图或安全结果。D9 只用于检验 D8 weak representation signal 是否跨数据集。
