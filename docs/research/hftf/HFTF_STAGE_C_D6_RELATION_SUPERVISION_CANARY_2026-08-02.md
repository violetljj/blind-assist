# HFTF Stage C D6 多源关系监督 canary

日期：2026-08-02

## 结论

固定 HFTF backbone 的 `128 × 3 × 6` 空间特征确实比 output-field profile
保留了更多可用信息，但**增加更多弱关系候选或既有人工 actionability 状态片段，
都没有把它变成可跨来源迁移的关系表征**。

本轮保留此前正结果：

`CROSS_SOURCE_SPATIAL_RELATION_OVER_OUTPUT_FIELD_GUARDRAIL_INCREMENT_SUPPORTED_IN_DEVELOPMENT`

同时记录新的科学负结果：

`FIXED_HFTF_SPATIAL_ACTIONABILITY_RELATION_TRANSFER_NOT_SUPPORTED`

并且最宽松的 held-out-source no-alert baseline oracle 也没有恢复 intervention：

`SOURCE_CENTERED_FIXED_FEATURE_RESCUE_NOT_SUPPORTED`

固定 `3 × 6` 网格上的非线性局部关系编码器以及额外 SANPO 正来源 support 也未
恢复迁移：

`FIXED_HFTF_GRID_NONLINEAR_RELATION_TRANSFER_NOT_SUPPORTED`

这不撤销空间特征相对 output-field 的增量，只关闭以下更窄假设：

> 保持现有 HFTF backbone 不变，仅通过扩大弱/人工关系监督和训练线性空间头，
> 即可获得超过当前 YOLO 的真实事件关系判断。

## 实验设计

所有结果只使用一个预先选定的 canary：

- checkpoint：`directional-seed17-fold0`；
- feature：固定 encoder 的 `128 × 3 × 6 = 2,304` 维空间特征；
- head：固定 L2 logistic；
- SANPO 30-event consumed Development 仅用于诊断；
- 当前 YOLO 参考：`13 hits / 6 false alerts / 5 cleared`；
- 既有 reviewed-normal-negative 空间头参考：
  `14 hits / 10 false alerts / 7 cleared`。

没有因为结果不佳而更换 checkpoint、L2、特征维度或确认逻辑。阈值曲线只用于
确认失败是否是 operating-point 偶然，不授权选择新阈值。

## Canary A：Luna merged relation pool

从 merged candidate pool 中只纳入精确复核类型：

- positive：`front_obstacle_approach`、`static_obstacle_approach`；
- negative：`normal_passage_negative`、`parallel_curb`；
- positive 同时派生严格 pre-trigger no-alert context；
- 同一来源在正负标签中重复出现的 frame SHA 全部移除。

训练库存为 `21 episodes / 11 sources / 771 frames`，默认阈值结果为：

| 监督 | hits | false alerts | cleared |
|---|---:|---:|---:|
| reviewed-normal-negative reference | 14/16 | 10/14 | 7/16 |
| merged relation pool | 13/16 | 11/14 | 3/16 |

`0.30–0.80` 的诊断阈值曲线没有一个点形成 YOLO Pareto 增量。因此不扩到其余
8 个 checkpoint：

`MERGED_WEAK_RELATION_SUPERVISION_INCREMENT_NOT_SUPPORTED`

## Canary B：人工 actionability 状态片段

复用 r789 已冻结的 16 个 public-video actionability events 和本地源视频。不是把
整个 positive window 粗标为 alert，而是按人工状态转移切分：

`route_clear_or_context → intervention_needed → route_clear`

以 2 Hz 直接从源视频解码，得到：

- public-video：`28 segments / 11 sources / 436 frames`；
- 与既有 provisional supervision 合并后：
  `42 segments / 18 sources / 485 frames`；
- label segments：`27 no-alert / 15 alert`。

SANPO canary 结果：

| 监督 | hits | false alerts | cleared |
|---|---:|---:|---:|
| reviewed-normal-negative reference | 14/16 | 10/14 | 7/16 |
| public-video actionability | 12/16 | 11/14 | 9/16 |

clearance 上升，但命中和误报同时退化；`0.30–0.80` 没有任何阈值形成 YOLO
Pareto 增量。

更关键的是，在 public-video 的 11 个来源上逐来源留一、复用同一固定空间特征：

| source-heldout 指标 | raw spatial | no-alert-centered oracle |
|---|---:|---:|
| frame alert recall | 0.0000 | 0.0000 |
| frame no-alert recall | 0.9924 | 0.9798 |
| frame balanced accuracy | 0.4962 | 0.4899 |
| segment alert recall | 0.0000 | 0.0000 |
| segment no-alert recall | 1.0000 | 1.0000 |
| segment balanced accuracy | 0.5000 | 0.5000 |

所有 7 个 held-out intervention segments 都被判为 no-alert。训练 loss 很低不能
改变这一点：当前固定特征可以拟合训练来源，却没有把 actionability relation
迁移到新来源。

no-alert-centered oracle 对每个 held-out source 使用该来源**全部人工 no-alert
segments** 的 episode-balanced 均值作为基线，再对其所有 frame 做差。这比未来
在线系统可获得的信息更宽松，仍然没有命中任何 intervention。因此失败不只是
absolute source appearance offset；在 fixed feature 上继续做 centering、线性
投影或阈值救援没有依据。

## Canary C：非线性局部关系编码器

为了排除“关系信号存在，但线性不可分”，固定一个不搜索超参数的小型卷积头：

- input：`current − source no-alert baseline` 与 `abs(delta)`；
- tensor：`256 × 3 × 6`；
- head：`1×1 conv 256→32`、`3×3 conv 32→16`、linear；
- parameters：13,137；
- training：200 epochs，AdamW，固定 threshold `0.5`；
- split：11 个 public-video source 逐来源留一；
- held-out source baseline 仍使用人工 no-alert oracle。

只使用 public-video 训练时：

| 指标 | 结果 |
|---|---:|
| frame alert recall | 0.0000 |
| frame no-alert recall | 0.9949 |
| frame balanced accuracy | 0.4975 |
| segment alert recall | 0.0000 |
| segment no-alert recall | 1.0000 |

随后加入已消费 SANPO 作为**训练 support，而不是评价**：

- 30 个 SANPO source；
- 46 个 phase episodes；
- 711 个 selected frames；
- 其中 16 个独立 positive sources；
- public + SANPO 合计 1,147 feature rows；
- 评价仍只看 11 个 held-out public-video sources。

结果：

| 指标 | SANPO support canary |
|---|---:|
| frame alert recall | 0.0000 |
| frame no-alert recall | 0.8788 |
| frame balanced accuracy | 0.4394 |
| segment alert recall | 0.0000 |
| segment no-alert recall | 1.0000 |
| segment balanced accuracy | 0.5000 |

每折 final train loss 都接近 `0`，但所有 held-out intervention segments 仍被判为
no-alert。这说明当前固定 HFTF grid 可以记忆训练来源，却没有可由该非线性局部头
提取的跨来源 actionability relation。

## 失败分类

本轮不是工程 invalid：

- 绑定的 11 个源视频全部存在并成功解码；
- 42-segment 库存和标签计数与预期一致；
- SANPO 推理、阈值回放和 source-heldout folds 全部完成；
- 没有 parser、path、scanner、JSON size 或 interruption 导致的证据损失。

三个新抓取的 Wikimedia 视频因第三人称或 crowd-only 视角不适合
phone-egocentric relation evaluation，只是 source qualification negative，
不计为算法失败，也不烧掉未来同来源的其他用途。

因此本轮负结果属于**有效科学负结果**：现有 fixed-backbone representation
缺少跨来源 actionability relation 可迁移性。

## 下一步

不再扩：

- 当前 fixed HFTF spatial head 的 checkpoint sweep；
- 同一 SANPO 30-event cohort 上的 L2、阈值和 head 搜索；
- 更多 discovery-only candidate 混入。
- fixed feature 上的 source centering 或其他线性 delta rescue。
- fixed HFTF grid 上的 nonlinear relation head 或增加 consumed SANPO support。

下一候选必须改变至少一个科学变量，而不是增加治理：

1. 直接训练/微调 relation-aware backbone，而不是冻结现有 HFTF grid；
2. 使用同一来源内 `clear → intervention → clear` 的成对目标，并把
   source-heldout actionability recall 作为训练前置筛选；
3. 补充真正 phone-egocentric、含 pre-event baseline 的独立实拍序列；
4. 只有 source-heldout relation 先超过 chance，才进入新的 outcome-unseen
   real-event 评价。

这条支线仍可继续，但下一阶段的研究问题已经从“更多监督能否救固定空间头”变为：

> relation-aware representation 是否能在来源外保留
> `障碍位置 × 可通行路径 × 当前干预需要` 的结构。

如果没有新增 backbone-level representation，D6 fixed-feature 路线保持关闭。

## 复现

```powershell
E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/research/hftf/run_stage_c_d6_provisional_relation_transfer.py `
  --feature-family spatial_grid_3x6 `
  --public-video-actionability-manifest artifacts.local/evidence/public-video-r789-actionability-manifest-20260719/actionability_manifest_r789.json `
  --public-video-feature-contract configs/public_video_actionability_linear_probe_contract_r790.json `
  --checkpoint artifacts.local/evidence/hftf/stage-c-d5-tartanground-cross-environment-v1/training/fold-0/directional-single-seed17/checkpoint.pt `
  --name directional-seed17-fold0 `
  --output artifacts.local/evidence/hftf/stage-c-d6-provisional-relation-spatial-public-video-actionability-v2/seed-17/fold-0.json
```

非线性 SANPO-support canary：

```powershell
E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/research/hftf/run_stage_c_d6_source_centered_relation_encoder_canary.py `
  --checkpoint artifacts.local/evidence/hftf/stage-c-d5-tartanground-cross-environment-v1/training/fold-0/directional-single-seed17/checkpoint.pt `
  --sanpo-support-manifest artifacts.local/evidence/riskseg-r0/event-eval/device-view-v2/manifest.json `
  --output-cache artifacts.local/evidence/hftf/stage-c-d6-source-centered-relation-encoder-sanpo-support-canary-v1/seed-17/fold-0-features.npz `
  --output artifacts.local/evidence/hftf/stage-c-d6-source-centered-relation-encoder-sanpo-support-canary-v1/seed-17/fold-0.json
```
