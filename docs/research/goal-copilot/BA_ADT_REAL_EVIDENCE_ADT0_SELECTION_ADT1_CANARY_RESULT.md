# BA-ADT-REAL-EVIDENCE full-sequence selection, ADT-1 evaluation, and ADT-2 demo

状态：`VALID / ADT0_FULL_SEQUENCE_TARGET_SELECTED / ADT1_FLOW5_TEMPORAL_TRACKER_ADMITTED_FOR_DEVELOPMENT / ADT2_PRERECORDED_DEVELOPMENT_DEMO_RENDERED / LONG_DROPOUT_REACQUISITION_INSUFFICIENT / SKY_DISABLED`

## 结论

固定 ADT-0 miner 在两条完整 ADT sequence 中找到自然六阶段 episode，并选中
`Apartment_release_clean_seq136_M1292 / Carrot_A (uid 4917588638317799)`。官方 manifest 恢复后，
已按 manifest SHA-1 下载 114,143,011-byte preview RGB，RGB-only YOLO11n adapter 处理全部 3,824 帧；
GT 只进入后置 evaluator。

在仅用前 25% GT timeline 选择固定 frame offset、其余 75% held-forward 的 2,160 帧上，carrot 一旦
被定位，bearing、bbox-scale nearness 和 approach direction 都有明确有效信号。加入最多 5 帧、仅平移、
fail-closed 的 sparse optical-flow persistence 后，recall 从 0.4041 提升到 0.5808，GT-invisible 误报为
0.0073；因此该最小 tracker 准入 Development demo。长 dropout 与重捕获仍不足，失败仍定位在
perception/tracking，不在冻结 Goal Copilot policy，因此 Sky 继续关闭。

ADT-2 Development demo 已把同一 RGB observation 接到冻结 GC1 winner，并渲染 target track、bearing/
nearness proxy、state/guidance timeline 与 evaluator-only GT error。它是 prerecorded replay，不是闭环导航。

## ADT-0 完整 sequence 结果

两条 GT archive 共 43,092,014 bytes，下载后 SHA-1 与官方 manifest 一致。选择明确复用了已消费的
ADT GT geometry prescreen 作为 Development 优先级，因此不是 fresh/confirmation evidence；GT 不进入
RGB estimator。

| Sequence | GT frames | track candidates | six-phase candidates | search | lost/reacquire | approach |
|---|---:|---:|---:|---:|---:|---:|
| `clean_seq134` | 2,816 | 317 | 172 | 219 | 286 / 286 | 286 |
| `clean_seq136` | 2,879 bbox-union timestamps | 318 | 134 | 175 | 284 / 284 | 279 |

`Carrot_A` 在 `seq136` 有 1,502 个 visible frames、14 个 qualifying track segments，包含
`SEARCH / ACQUIRE / TRACK / LOST / REACQUIRE / APPROACH`；最强 GT center-range approach proxy 从约
`4.59 m` 降至 `1.77 m`。

## 坐标与时间对齐修正

早期 sample 诊断把 ADT bbox 到 preview 的旋转误写成 y-axis flip，导致已发布的 `0.1393` recall 不再
有效。修正后的坐标变换是 90° clockwise：`x' = H - y_max, y' = x_min`。Evaluator v2 还以
`aria_trajectory.csv` 为 2,880 帧 GT timeline，只在前四分之一选择固定 video offset，再在其余帧计分。

修正后 sample `WoodenBowl` 的 held-forward localization recall 是 `0.2488`，false-visible rate 是
`0.7895`，最长 dropout 是 85 帧，仍支持“多实例 target identity/grounding 不足”的 Development 诊断，
但不再引用错误变换下的旧数值。

## ADT-1 seq136 / Carrot_A RGB-only 结果

Detector-only baseline 的固定 alignment offset 为 506 video frames；5-frame tracker candidate 为 507；
两者 held-forward 区间均为 GT frame 720–2879。

| Metric | Result |
|---|---:|
| evaluated / GT-visible frames | 2,160 / 1,064 |
| localized recall at IoU ≥ 0.10 | 0.4041 |
| false-visible rate while GT-invisible | 0.0000 |
| longest localization dropout | 177 frames |
| mean IoU on GT-visible frames | 0.2821 |
| normalized bearing MAE | 0.01234 |
| predicted/GT bbox-scale correlation | 0.9681 |
| approach-direction accuracy, lag 15 | 0.9091 (231 comparisons) |
| eligible reacquisitions / success within 30 frames | 10 / 0.4000 |
| mean observation quality, localized / missed | 0.2928 / 0.0089 |

这建立的是条件性 evidence signal：定位成功时 bearing、relative nearness、approach 与 quality separation
有用；它没有建立稳定 visibility/tracking。

在同一已消费 Development sequence 上，先比较 30-frame persistence，发现 recall 虽升至 0.6767，但
GT-invisible 误报升至 0.0940，故拒绝。随后选择更保守的原生 5-frame candidate：

| Metric | Detector only | RGB flow, max 5 frames |
|---|---:|---:|
| localized recall at IoU ≥ 0.10 | 0.4041 | 0.5808 |
| false-visible rate while GT-invisible | 0.0000 | 0.0073 |
| longest localization dropout | 177 | 162 |
| mean IoU on GT-visible frames | 0.2821 | 0.4469 |
| normalized bearing MAE | 0.01234 | 0.00648 |
| bbox-scale correlation | 0.9681 | 0.9626 |
| 30-frame reacquisition success | 0.4000 | 0.4000 |

5-frame candidate 同时提高 recall/IoU、只引入 8/1,096 的 GT-invisible false-visible frames，因此仅准入
Development demo；reacquisition 没有改善，不能晋级为已解决的 M1 tracking。

## ADT-2 Development demo

最终 demo 使用 5-frame tracker，覆盖 held-forward 2,160 帧（72 秒），绿色/红色框来自 RGB observer，紫色框与误差只来自旁路 GT
evaluator。冻结 GC1 winner SHA-256 为
`24d4e57374dd99363700ae881d18db536e48ec5f79f39e95c5b873e96edbc3a1`。

Adapter 对 bearing 使用 `normalized_image_x × 45°` 的 policy proxy，明确不是相机标定角；nearness 是
预测 bbox 面积平方根，明确不是 metric distance。ADT preview 没有进入本 adapter 的 clearance evidence，
所以 left/right/forward clearance 全部 fail closed 为 false，completion claim 固定为 false。由此，demo
可以展示 search/acquire/track/uncertain/lost/reacquire/approach 理解和保守 guidance，但不能宣称安全移动、
到达或交互完成。

## Evidence identity

```text
selected_rgb preview MP4                              manifest SHA-1 76f2e6066ad7190b9a8e77a29e462466771f88e3
seq136 rgb_observations.json                         162cac8c090cd35101b661ee5ec0133bacc8830e78e79081dae9d713cab998f8
seq136 evaluation.json                               78cbb840dd42e78db181b6b061f5cb7bc0806ab30c9c187c72fd9b71a2714f11
seq136 rgb_observations_flow5_actual.json            9ecf7d1c787321baf11738aa3019b98db74b092777edf26259a5219bc57c117f
seq136 evaluation_flow5_actual.json                  3f5e7a26c99d278d446e249678091eed89af7a2f9be2a591b046e038bb91f100
seq136 guidance_timeline_flow5.json                  d64cb4ae25b1ba1881783c73201947e7ce6fc78d051db4105343c537fa16152e
seq136 offline_copilot_demo_flow5.mp4                e8d3fb71906207dfa33bc55ba307a15d74b50a2aa148e73296f15ca1e417868a
seq136 contact_sheet_flow5.png                       189d4dcd4e9e00feaede9bcd30affc2317aeda95934c33fed32dc1f678408733
sample corrected rgb_bowl_evaluation.json            716231cb779e722b97bb17668c1feb0eeffb8d847f82872bae3ad1a899482449
```

## Claim ceiling 与唯一下一步

已建立 ADT-0 Development 数据适配性、ADT-1 条件性真实 RGB evidence signal 和首个 ADT-2 prerecorded
Development demo。没有建立可靠 visibility/tracking、标定角度、metric range、clearance、completion、
interactive navigation、安全或默认 App 证据。

唯一 successor 是 `ADT1_INSTANCE_CONDITIONED_REDETECTION`：当前 5-frame tracker 只能跨越短漏检，下一步
应解决长达 162 帧的 dropout 与未改善的 reacquisition，而不是继续拉长会制造 false-visible 的 persistence。
只有 observation 充分而 guidance 仍错误时，才允许另立 policy failure benchmark；当前不授权 Sky。
