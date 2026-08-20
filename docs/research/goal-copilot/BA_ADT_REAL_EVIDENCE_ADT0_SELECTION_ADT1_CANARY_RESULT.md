# BA-ADT-REAL-EVIDENCE full-sequence selection and ADT-1 sample canary

状态：`VALID / ADT0_FULL_SEQUENCE_TARGET_SELECTED / ADT1_RGB_CANARY_PERCEPTION_IDENTITY_FAILURE / SEQ136_CARROT_RGB_NEXT`

## 结论

保持 sample miner 和全部门槛不变后，两条完整 ADT sequence 都自然提供大量六阶段 target episode。
ADT-0 数据适配性已经建立，首个 ADT-1 demo 目标选为
`Apartment_release_clean_seq136_M1292 / Carrot_A (uid 4917588638317799)`。

在等待该 sequence RGB 的同时，官方 sample 上的 RGB-only mechanical canary 已证明 observer/evaluator
防火墙和时序接口可运行，但 COCO `bowl` 类不能稳定维持 `WoodenBowl` 实例身份。该失败定位在
target grounding/association，不在 Goal Copilot policy；Sky 继续关闭。

## ADT-0 完整 sequence 结果

两条 GT archive 共 43,092,014 bytes，下载后 SHA-1 与官方 manifest 一致。选择明确复用了已消费的
ADT GT geometry prescreen 作为 Development 优先级，因此不是 fresh/confirmation evidence；RGB 下载数为
`0`，GT 不进入 estimator。

| Sequence | GT frames | track candidates | six-phase candidates | search | lost/reacquire | approach |
|---|---:|---:|---:|---:|---:|---:|
| `clean_seq134` | 2,816 | 317 | 172 | 219 | 286 / 286 | 286 |
| `clean_seq136` | 2,879 | 318 | 134 | 175 | 284 / 284 | 279 |

`Carrot_A` 在 `seq136` 有 1,502 个 visible frames（52.2%）、14 个 qualifying track segments，包含
`SEARCH / ACQUIRE / TRACK / LOST / REACQUIRE / APPROACH`；最强 approach segment 的 GT center-range
proxy 从约 `4.59 m` 降至 `1.77 m`。这些是 episode 候选，不把 visibility gap 预先解释为 tracker failure。

## ADT-1 sample RGB-only canary

Observer 只接收官方 sample preview MP4、本地 YOLO11n 权重和目标类 `bowl`；CLI 不支持 GT 参数。
它在 299 帧中报告 259 个 visible frames 和 29 个 acquire/lost/reacquire 转换。隔离 evaluator 再把输出
与 `WoodenBowl` GT 对齐，使用 ADT GT → preview 的 y-axis flip：

| Metric | Result |
|---|---:|
| GT-visible frames | 280 |
| localized recall at IoU ≥ 0.10 | 0.1393 |
| mean IoU on GT-visible frames | 0.0917 |
| longest localization dropout | 86 frames |
| false-visible rate while target GT-invisible | 0.7895 |
| normalized bearing MAE | 0.4412 |
| predicted/GT bbox-scale correlation | 0.3976 |

这不能解释为“YOLO 不会检测 bowl”。画面中存在多个 bowl-like object，通用类检测与贪心 IoU
association 会锁到错误实例；结果只说明当前 closed-vocabulary class-to-instance adapter 不足以支持
“帮我找那个木碗”。它为 open-vocabulary/instance-conditioned grounding 提供了真实失败基线。

## Evidence identity

```text
selected_gt/acquisition.json                                0c4b29c7c954f743b835cf26eb7eae022b4ef2cfc57d6bd1ae45e9aed0b9e6a6
clean_seq134 episodes.json                                 1b54d2a3454f123f9ea7f0a084726cb3d581afb41339af4ec8c50252dee98cc7
clean_seq136 episodes.json                                 e060336300267b665f6f3e7071e475f82aa0bc9386e7e8252aff9cc47da9aab9
sample rgb_bowl_observations.json                           5ab518e351160e1a1afa44b17244a5577971c0bd3712f80ae96079cc2512d672
sample rgb_bowl_evaluation.json                             f224efa76a4935db47df29fd2f9b740fa9fdb039c260de4fffb135313b300b95
```

## Claim ceiling 与唯一下一步

当前建立 ADT-0 Development 数据适配性、RGB-only observer mechanics 和一个 target-identity failure。
没有建立通用 detection/tracking/bearing/nearness 准确性，没有接 Goal Copilot，也没有离线副驾、导航、
安全或默认 App 证据。

唯一 successor 是 `ADT1_SEQ136_CARROT_RGB_CANARY`：官方 manifest 恢复后，hash-verified 下载
`seq136` preview RGB；以唯一 COCO carrot class 运行完全相同的 RGB-only adapter，再由 GT evaluator
评估。当前 manifest transport 的 TLS EOF 只阻塞该 RGB acquisition，不改变 ADT-0 结论。
