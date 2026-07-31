# YOLO + 语义分割图像空间互补性跨来源 R2 结果

状态：`DEVELOPMENT_COMPLETE / CROSS_SOURCE_IMAGE_SPACE_SIGNAL_REPLICATED /
CLASS_STABILITY_MIXED / NO_EFFECT_AUTHORITY`

日期：2026-07-31（Asia/Hong_Kong）
执行者：`violjjet`
授权：用户在本任务中明确授权继续执行第二来源与同后端复现

## 1. 研究问题与边界

本轮沿用已冻结的 [complementarity Development design R0](DUAL_LOOP_SEGMENTATION_COMPLEMENTARITY_DEVELOPMENT_DESIGN_R0.md)，
只检验：在两个不同 RGB source 上，使用同一 YOLO11n 模型资产、同一 host
preprocess/decode/NMS 合同和同一 segmentation reference，是否能重复观察到
`segmentation mask - YOLO box union` 的 class-wise image-space region 及其时间稳定性。

本轮不回答：

- 分割区域是否现实不可通行或是有效障碍；
- A+B 是否改善风险召回、误提醒、episode、feedback 或安全；
- 哪一个分割模型应被正式选中；
- host LiteRT 是否等价于手机 QNN HTP；
- 是否应进入 Android、默认模型或生产。

中央阻塞 Agent 标签、risk、feedback 和 event 字段均未进入输入面。

## 2. 输入身份与同后端控制

两个 source 都使用 `host_ai_edge_litert`、4 threads、同一模型资产和同一解码合同；
原 R1 的 Shiraz QNN trace 保留为独立设备参考，但不与本轮 host source 混合计算。

| 项目 | Shiraz host | Shanghai host |
| --- | --- | --- |
| RGB manifest | `artifacts.local/evidence/dual-loop-r1-unseen-natural-event-r0/rank2-shiraz/input-10hz-r1/manifest.jsonl` | `artifacts.local/evidence/dual-loop-r1-unseen-natural-event-r0/input-10hz-r1/manifest.jsonl` |
| RGB manifest SHA256 | `af0ab3c735d96737f451a6e64d1784681966345c7849131ad51bd46c9d7e6571` | `680d0e1594a99581eef3f517a831a5b91b66c72df4593421cc5316247017772c` |
| source | `commons_iran_shiraz_city_tour_2021_5` | `commons_shanghai_shopping_street_night_2024` |
| frame count | 4,891 | 5,662 |
| host YOLO trace SHA256 | `41f978f52b68d443f02e3e1a0b40a5bb2faeafcbed6d5a94bc7d47f1eaa6bc45` | `84ef428e6f94a896dbea94d21400cf52e9e7a1647ff19cd1a6fc6678c952a2b9` |
| post-NMS detections | 11,596 | 11,606 |
| YOLO model SHA256 | `00edb41a528b0a7e709c4af8ce3e685491492c4539274804e5cfc17a1a867cd2` | same |
| labels SHA256 | `bd17f1ee35d5f3c862a4894605855abbb9dda4b0621fdb0ac4c2c8c7bb7e730a` | same |
| segmentation reference SHA256 | `88f0184d2671230c1f1f43192758689d286b530d7490e1d1ca0671f83b50b50c` | same |

Host trace receipts：

- Shiraz：`artifacts.local/evidence/dual-loop-segmentation-complementarity-r2-shiraz-host-yolo/receipt.json`
  （SHA256 `6f69dba24ea140c4a348d53b2ed60b4c9f27a607a5a0e7f923a353f69136d277`）；
- Shanghai：`artifacts.local/evidence/dual-loop-segmentation-complementarity-r2-shanghai-host-yolo/receipt.json`
  （SHA256 `5cf75c95cdef468076fcb6daf50f13c984e461c80496bd6d391badd22c3cf983`）。

上海 trace 首次完整写入后只在 receipt 序列化阶段遇到 `numpy.int32` 类型错误；随后
以逐行 identity、模型/标签 SHA、框字段和计数复核的 `--finalize-existing` 封存，
`recovered_from_complete_trace=true`，没有重跑或修改任何 trace row。

## 3. 执行完整性

- Shiraz `4,891/4,891` 与 Shanghai `5,662/5,662` 均精确配对；两份独立 validator
  均返回 `VALID`，`NOT_EVALUABLE=0`；
- 两个 source 的 segmentation report 均通过 pairing、finite output、四类分区、
  union arithmetic、时间顺序和禁止字段检查；
- 两个 reference 均没有单类塌缩；没有读取中央阻塞、risk、feedback 或 event；
- segmentation 运行仍是 host-only：Shiraz P50/P95 total `11.696/17.737 ms`，
  Shanghai `11.812/18.020 ms`。这不是手机、Snapdragon、功耗或温度数据。
- 两份 report 的 `evidence_instance` 仍保留冻结设计标识
  `DUAL_LOOP_SEGMENTATION_COMPLEMENTARITY_R1`；这里的 R2 只表示同一 estimand 的
  第二来源执行，不是改变问题或新增 readiness successor。

产物目录：

- Shiraz report：`artifacts.local/evidence/dual-loop-segmentation-complementarity-r2-shiraz-host/`
  （report SHA256 `8eaa932e207b4905697b7c957b6229015e1fd1bd8aa69725af71f4fcde56ba6d`，
  frames SHA256 `56d636edb277a46b7a50b5c6603fb24a6877dfa8d48bc44e262a9fa1967ca1f5`，
  validation SHA256 `b2c7b3881cf90c52b73d08b815c0ff4c5b8fca588804177795d895ab2738e34c`）；
- Shanghai report：`artifacts.local/evidence/dual-loop-segmentation-complementarity-r2-shanghai-host/`
  （report SHA256 `2985080b352dbf13dca49a4824a9fdae3bb54887072daafb2c0bf7bfc32987d7`，
  frames SHA256 `45a97e4d450691d722db9653ba68c8b58f662fe9903e47979543a86de20784e8`，
  validation SHA256 `4df8d2b90b116fe2b076808832a3c504130181c5e0da0890b574ba5ce3865745`）。

## 4. 跨来源图像空间结果

以下是 session 内描述统计；两个 source 是重复观测，不把帧数当作独立重复，也不做
p-value 或 frame-independent confidence interval。

| 指标 | Shiraz host | Shanghai host |
| --- | ---: | ---: |
| YOLO box coverage mean | 0.122821 | 0.055703 |
| `walkable` uncovered median | 0.359589 | 0.412918 |
| `boundary_step_curb` uncovered median | 0.003860 | 0.002914 |
| `obstacle` uncovered median | 0.023270 | 0.012115 |
| `unknown_nonwalkable` uncovered median | 0.431198 | 0.496758 |
| `walkable` adjacent IoU median | 0.816846 | 0.777545 |
| `boundary_step_curb` adjacent IoU median | 0.080000 | 0.053571 |
| `obstacle` adjacent IoU median | 0.253715 | 0.174623 |
| `unknown_nonwalkable` adjacent IoU median | 0.723067 | 0.755459 |

两来源重复出现了同一组较窄的机制现象：四类 argmax mask 都有可计算的 YOLO 未覆盖
区域，`walkable`/`unknown_nonwalkable` 的相邻稳定性相对较高，而
`boundary_step_curb`/`obstacle` 的稳定性较低。因而可以把以下命题升级为
Development 级的跨来源图像空间观察：

```text
CROSS_SOURCE_IMAGE_SPACE_SIGNAL_REPLICATED
CLASS_STABILITY_MIXED_AND_SOURCE_DEPENDENT
```

但量级不是来源不变的：YOLO coverage 在两个 source 间约为 `0.123` 与 `0.056`，
`obstacle` uncovered median 约为 `0.0233` 与 `0.0121`。这说明分割输出的非零区域
可以复现，不说明某个区域就是有效障碍，更不说明 C 带来了风险效果。

## 5. 终态与下一步边界

本轮终态：

```text
CROSS_SOURCE_IMAGE_SPACE_SIGNAL_REPLICATED
CLASS_STABILITY_MIXED_AND_SOURCE_DEPENDENT
NO_OBJECTIVE_OBSTACLE_TRUTH
NO_FUSION_EFFECT_AUTHORITY
NO_ANDROID_OR_PRODUCTION_AUTHORITY
```

因此，中央阻塞 Agent 标签路线仍永久关闭，且不能用本轮结果恢复它。下一步若继续，
应另行冻结一个直接的 image-space fusion operator 和客观区域/像素评价协议，先回答
“A+B 是否产生可复算的几何增量、误激活和计算代价”；不得把本轮 `obstacle` 类名、
uncovered fraction 或 union increment 直接升级为风险事件真值。若没有明确的客观
评价单位和固定融合规则，则保留本轮机制证据并停止，不进入 Android 或主动提醒。
