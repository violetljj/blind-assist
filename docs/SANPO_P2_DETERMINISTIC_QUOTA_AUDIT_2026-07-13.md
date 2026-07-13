# SANPO P2 确定性 quota sampler 审计

## 结论

P2 已完成严格 `25% × 4` quota sampler 的实现和五组 head-only 对照。工程合同闭合，但离线质量没有改善，因此该 sampler **保留为显式审计选项，不替换默认 sampler**。

- 四类 intent 为 boundary guided crop、obstacle guided crop、zero-boundary hard negative full-frame、per-session unknown-rich full-frame。
- 100 step、batch 6 共 600 draw，最终严格为 `150/150/150/150`，realized share 全部 `0.25`。
- 固定 model seed 时 sampler-seed selection range 从 P1-A 的 `0.0072` 降至 `0.0024`，采样顺序更加稳定。
- 但固定 sampler 时 model-seed 最差 selection 从 `0.1970` 降至 `0.1700`，range 从 `0.2951` 扩至 `0.3097`；最佳 mIoU 从 `0.4642` 降至 `0.4484`。
- P2 没有抬高弱初始化，下一步进入 P3 split/session 重构。

## 数据池设计

候选资格使用 canonical 原分辨率 mask；实际 crop 使用 384×384 nearest-resized mask。这样可以识别并排除 3 个因 resize 丢失极细 boundary 的帧，防止它们被误归为 hard negative。

| Quota | 候选帧 | Eligible sessions | 抽样方式 |
|---|---:|---:|---|
| boundary | 293 | 7 | eligible session 均匀轮转，目标 crop |
| obstacle | 398 | 8 | eligible session 均匀轮转，目标 crop |
| hard negative | 104 | 5 | 按候选数加权 session 轮转，完整帧 |
| unknown-rich | 104 | 8 | 每 session 内 q75，各 13 帧，完整帧 |

最初的全局 unknown q75 会只覆盖 100 帧、4 个 session，延续数据分布错位，因此在正式训练前被主动中止并改为 per-session q75。各 session 阈值写入训练报告，范围约 `3.97%–54.97%`。

## 调度与审计合同

- 全局样本周期固定为 `boundary → obstacle → hard_negative → unknown_full_frame`。
- batch 6 的第一批累计 `2/2/1/1`，第二批补齐为 `3/3/3/3`；每两批完成三个完整周期。
- 请求在 step 25/75 评估时，训练器自动对齐到 step 26/76；step 50/100 不变。所有 checkpoint 的 `quota_cycle_complete=true`。
- 报告包含 quota target/draw/share、候选帧/session、per-quota session draw、repeat factor、canonical/resized boundary 差异、quota×实际类 presence、cycle 状态与 sample trace SHA256。
- boundary/obstacle crop 后必须保留目标类；hard negative 必须在 canonical mask 中零 boundary；任一强制池为空时 fail closed。

## 五组结果

| Model seed | Sampler seed | Selection | mIoU | Boundary IoU | Unknown IoU | Macro-session | Worst-scene |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260711 | 20260711 | 0.4798 | 0.4484 | 0.5159 | 0.3879 | 0.3223 | 0.2611 |
| 20260712 | 20260711 | 0.1700 | 0.2031 | 0.1463 | 0.2963 | 0.1761 | 0.1052 |
| 20260713 | 20260711 | 0.2192 | 0.3046 | 0.1712 | 0.2357 | 0.2676 | 0.1751 |
| 20260711 | 20260712 | 0.4796 | 0.4473 | 0.5168 | 0.3993 | 0.3213 | 0.2605 |
| 20260711 | 20260713 | 0.4774 | 0.4449 | 0.5151 | 0.3913 | 0.3203 | 0.2594 |

与 P1-A 相比，quota sampler 降低了 sampler 顺序噪声，但对三个 model seed 的整体结果略有伤害。当前瓶颈不是“随机 sampler 没抽准”，而是可用 session/split 本身不能支撑稳定泛化。

## 验证与边界

- 训练器及训练/export/equivalence/quality-gate 相关 47 tests 通过。
- 正式报告：`test-artifacts.local/segmentation-candidate/p2-deterministic-quota-session-q75-20260713/training_report.json`（本地、不提交）。
- 报告 SHA256：`0534441fc6bb2841087d43da4a567bf023e110725085942897002163b8284140`。
- `blind_holdout_access=not_accessed_by_trainer`，`promotion=do_not_replace_default_model`。
- 未导出 INT8、未运行设备门、未修改 App。

默认 `--sampler-strategy` 继续为 `session_balanced_guided`；P2 可通过 `--sampler-strategy deterministic_quota --unknown-rich-quantile 0.75` 显式复核。
