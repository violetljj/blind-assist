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

解冻 HFTF encoder tail 的 paired-RGB 训练仍未恢复 held-out intervention：

`PAIRED_RGB_TAIL_FINETUNE_RELATION_TRANSFER_NOT_SUPPORTED`

随后新增的配对预训练把“任务本身是否可学习”和“是否跨真实域迁移”拆开：

- TartanGround 6-parent 训练到 2 个 outcome-unseen parent 的 frame BA/AUROC
  为 `0.7098/0.7124`，episode BA/AUROC 均为 `1.0`：
  `TARTANGROUND_PAIRED_RELATION_OUTCOME_UNSEEN_SUPPORTED_IN_DEVELOPMENT`；
- 同一状态直接迁移到 public real 的 frame alert recall 为 `0.025`、AUROC
  `0.4053`，没有建立 synthetic-to-real transfer；
- SANPO-only 到 public 的直接迁移出现可复现的 Edmonton 局部正信号，但 3-source
  macro frame/episode AUROC 只有 `0.4604/0.4167`：
  `SANPO_TO_PUBLIC_PAIRED_RELATION_SIGNAL_SOURCE_LOCAL_ONLY`；
- TartanGround → SANPO 的课程没有改善 public 迁移：
  `TARTANGROUND_SANPO_CURRICULUM_PUBLIC_TRANSFER_NOT_SUPPORTED`。
- 仅 28,313 个可训练参数的 early joint-pair stem 把 pooled frame BA/AUROC
  提高到 `0.6185/0.6978`，并把 Edmonton frame/episode AUROC 提高到
  `0.8134/0.80`；但 source-macro AUROC 只有 `0.4184/0.4333`：
  `JOINT_PAIR_INTERACTION_POOLED_AND_EDMONTON_SIGNAL_SOURCE_LOCAL_ONLY`。

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

## Canary D：paired-RGB backbone tail fine-tune

最后一个 canary 真正改变 backbone 表征，而不是继续换 fixed-grid head：

- Siamese input：current RGB 与同来源 episode-balanced no-alert references；
- 解冻：MobileNetV3-small `encoder[9:] + pointwise`；
- trainable backbone parameters：810,472；
- relation head parameters：13,137；
- support：其余 public sources + 30 个 consumed SANPO sources；
- evaluation：只留出 3 个含 intervention 的 public sources；
- 10 epochs，backbone LR `1e-4`，head LR `3e-3`；
- threshold `0.5`，无超参数搜索。

第一次实现使用 CUDA adaptive-pool backward，完全相同配置两次出现不同结果。
这两次明确归类为：

`ENGINEERING_INVALID_NONDETERMINISTIC_BACKWARD_RETRY_ALLOWED`

它们没有烧掉 cohort，也不进入科学结论。修复为 deterministic algorithms、
关闭 TF32/benchmark，并把 `4×7 → 3×6` 改为 deterministic bilinear resize 后，
repeat A/B 的 folds、loss、逐 episode score 和 metrics 全部完全一致。

确定性结果：

| positive-source-heldout 指标 | 结果 |
|---|---:|
| frame alert recall | 0.0000 |
| frame no-alert recall | 0.8922 |
| frame balanced accuracy | 0.4461 |
| frame AUROC | 0.5034 |
| segment alert recall | 0.0000 |
| segment no-alert recall | 1.0000 |
| segment balanced accuracy | 0.5000 |
| segment AUROC | 0.3377 |

三折 final train loss 分别约为 `0.0013 / 0.0082 / 0.0017`，但 Bangkok、Ulm、
Edmonton 的 7 个 held-out intervention segments 全部未命中。当前配对 RGB
tail fine-tuning recipe 仍然是可拟合、不可迁移。

## Canary E：配对任务可学习性与 synthetic-to-real

为了避免把 public 迁移失败误写成“配对目标本身不可学习”，从已有 TartanGround
样本按当前 body 中央两个方向构造同 parent 的 clear/risk 配对：

- train：6 个 parents、193 帧；
- transfer：2 个 outcome-unseen parents、66 帧；
- ambiguous truth 排除；
- train/transfer parent 完全互斥；
- 与 Canary D 相同的 HFTF tail、relation head、10 epochs 和学习率。

结果为：

| outcome-unseen synthetic 指标 | 结果 |
|---|---:|
| frame alert recall | 0.6863 |
| frame no-alert recall | 0.7333 |
| frame balanced accuracy | 0.7098 |
| frame AUROC | 0.7124 |
| episode alert/no-alert recall | 1.0000 / 1.0000 |
| episode balanced accuracy / AUROC | 1.0000 / 1.0000 |

因此配对 relation 任务在合成环境中可学习并能跨 parent 迁移，正结果保留在
synthetic representation 层。把该确定性状态直接用于 public positive-source
held-out fine-tune 后，frame alert recall 只有 `0.025`、frame AUROC `0.4053`，
episode alert recall 仍为 `0`。这建立的是 synthetic-to-real domain gap，不是否定
synthetic paired learnability。

## Canary F：SANPO-only real-to-real 直接迁移

随后不使用任何 public frame 更新参数，只用 consumed SANPO support：

- train：46 episodes / 30 sources / 711 frames；
- transfer：Bangkok、Ulm、Edmonton 的 18 segments / 272 frames；
- public no-alert truth 只用于构造同来源 reference；
- public positive 和 no-alert frame 的训练使用数均为 0；
- 固定同一模型、10 epochs、threshold `0.5`，无参数搜索。

两次完整运行除时间戳外逐字段完全一致。pooled 结果为：

| SANPO → public 指标 | 结果 |
|---|---:|
| frame alert/no-alert recall | 0.2750 / 0.8621 |
| frame balanced accuracy / AUROC | 0.5685 / 0.5811 |
| episode alert/no-alert recall | 0.2857 / 0.9091 |
| episode balanced accuracy / AUROC | 0.5974 / 0.5844 |

但 pooled 指标混入 source score scale。逐来源结果为：

| source | frame BA | frame AUROC | episode BA | episode AUROC |
|---|---:|---:|---:|---:|
| Bangkok | 0.5625 | 0.5527 | 0.5000 | 0.5000 |
| Ulm | 0.3967 | 0.0326 | 0.5000 | 0.0000 |
| Edmonton | 0.6648 | 0.7958 | 0.6500 | 0.7500 |
| source macro | 0.5414 | 0.4604 | 0.5500 | 0.4167 |

因此 Edmonton 上存在真实、可复现的跨域 relation signal，不能被后续系统层失败
抹去；但它没有跨 3 个来源保持方向一致，不能称为 source-general transfer。
Ulm 的排序近乎完全反向，也说明仅做 source threshold/centering 不能解决问题。

最后固定测试 TartanGround paired state → SANPO fine-tune → public direct transfer。
该课程的 pooled frame/episode AUROC 为 `0.4920/0.4156`，source-macro AUROC 为
`0.4790/0.3278`；Edmonton frame AUROC 也从 `0.7958` 降到 `0.5019`。因此课程没有
建立增量，不能通过继续堆叠相同 encode-then-difference 预训练来救援。

## Canary G：early joint-pair interaction

现有 paired-RGB recipe 先分别编码 current/reference，再在 `128×3×6` embedding
上相减。为直接改变 frame interaction 发生的位置，新增一个小型 raw-pair stem：

- input：`current RGB / baseline RGB / signed delta / abs(delta)`，共 12 channels；
- 4 个轻量下采样 stage：`24/32/64/128` channels；
- 与冻结 HFTF current-context 在 `3×6` 网格拼接；
- trainable：pair stem + relation head，共 28,313 parameters；
- train/eval inventory、10 epochs、SANPO-only/public-zero-train 与 Canary F 相同；
- 无 threshold、architecture 或 seed 搜索。

两次运行除时间戳外逐字段一致。相对 encode-then-difference baseline：

| pooled frame 指标 | encode-then-difference | early joint-pair |
|---|---:|---:|
| alert recall | 0.2750 | 0.3750 |
| no-alert recall | 0.8621 | 0.8621 |
| balanced accuracy | 0.5685 | 0.6185 |
| AUROC | 0.5811 | 0.6978 |

这个 pooled frame-level 增量是真实、可复现的表示证据；但逐来源排序仍不一致：

| source | frame BA | frame AUROC | episode BA | episode AUROC |
|---|---:|---:|---:|---:|
| Bangkok | 0.5000 | 0.1836 | 0.6667 | 0.5000 |
| Ulm | 0.4891 | 0.2582 | 0.5000 | 0.0000 |
| Edmonton | 0.5000 | 0.8134 | 0.5000 | 0.8000 |
| source macro | 0.4964 | 0.4184 | 0.5556 | 0.4333 |

因此 early interaction 的科学结论是“对部分来源和 pooled frame discrimination
有增量”，不是“已经学会跨来源 actionability”。Bangkok 与 Ulm 的反向排序说明，
继续做 source threshold calibration 仍不能修复表示。下一实验应把 early
interaction 放回 HFTF 的 cell/lane future-risk teacher task，用结构化空间监督替代
单一 source-relative actionability 标签。

## 失败分类

上述有效结果不是工程 invalid：

- 绑定的 11 个源视频全部存在并成功解码；
- 42-segment 库存和标签计数与预期一致；
- SANPO 推理、阈值回放和 source-heldout folds 全部完成；
- 没有 parser、path、scanner、JSON size 或 interruption 导致的证据损失。

三个新抓取的 Wikimedia 视频因第三人称或 crowd-only 视角不适合
phone-egocentric relation evaluation，只是 source qualification negative，
不计为算法失败，也不烧掉未来同来源的其他用途。

因此结论按实际层级分开：

- synthetic paired relation outcome-unseen learnability：有效正结果；
- Edmonton real-to-real relation ranking：有效、来源局部正信号；
- early joint-pair 的 pooled frame discrimination 与 Edmonton ranking：
  有效、来源局部表示增量；
- 三来源宏平均 direct transfer、synthetic-to-real 与 synthetic→SANPO curriculum：
  有效科学负结果；
- 当前 `encode each frame → subtract embedding → relation head` 表示缺少跨真实来源
  方向一致性。

## 下一步

不再扩：

- 当前 fixed HFTF spatial head 的 checkpoint sweep；
- 同一 SANPO 30-event cohort 上的 L2、阈值和 head 搜索；
- 更多 discovery-only candidate 混入。
- fixed feature 上的 source centering 或其他线性 delta rescue。
- fixed HFTF grid 上的 nonlinear relation head 或增加 consumed SANPO support。
- 当前 `encoder[9:] + pointwise` paired-RGB tail fine-tuning recipe。
- TartanGround paired state 的直接 synthetic-to-real 使用。
- TartanGround → SANPO 的同构 encode-then-difference 课程。
- 当前 SANPO binary actionability 上的 joint-pair source-general claim。

下一候选必须改变至少一个科学变量，而不是增加治理：

1. 新增真正 phone-egocentric、含 pre/intervention/passed 的独立正来源；
2. 下一 representation 必须在 backbone 内联合比较 frame pair 或直接预测
   `身体包络 × 空间占用 × 短时未来风险`，不能继续分别编码后相减；
3. 训练目标优先回到 HFTF 的 cell/lane risk field，而不是把 actionability
   压成单一 source-relative 二分类；
4. 只有 source-heldout relation 先超过 chance，才进入新的 outcome-unseen
   real-event 评价。

这条支线仍可继续，但下一阶段的研究问题已经从“更多监督能否救固定空间头”变为：

> relation-aware representation 是否能在来源外保留
> `障碍位置 × 可通行路径 × 当前干预需要` 的结构。

如果没有新增 backbone-level representation，D6 fixed-feature 路线保持关闭。
当前 encode-then-difference paired-RGB recipe 保持关闭；synthetic positive 与
early joint-pair pooled/Edmonton 局部正信号作为下一代 structured-field
representation 的依据保留。

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

确定性 paired-RGB backbone canary：

```powershell
E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/research/hftf/run_stage_c_d6_paired_rgb_relation_backbone_canary.py `
  --checkpoint artifacts.local/evidence/hftf/stage-c-d5-tartanground-cross-environment-v1/training/fold-0/directional-single-seed17/checkpoint.pt `
  --output artifacts.local/evidence/hftf/stage-c-d6-paired-rgb-relation-backbone-positive-source-canary-v2-repeat-a/seed-17/result.json
```

TartanGround 配对预训练：

```powershell
E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/research/hftf/run_stage_c_d6_tartanground_paired_relation_pretraining_canary.py `
  --checkpoint artifacts.local/evidence/hftf/stage-c-d5-tartanground-cross-environment-v1/training/fold-0/directional-single-seed17/checkpoint.pt `
  --output-model artifacts.local/evidence/hftf/stage-c-d6-tartanground-paired-relation-pretraining-transfer-canary-v1/paired-relation-state.pt `
  --output artifacts.local/evidence/hftf/stage-c-d6-tartanground-paired-relation-pretraining-transfer-canary-v1/result.json
```

SANPO-only 到 public 直接迁移：

```powershell
E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/research/hftf/run_stage_c_d6_sanpo_paired_pretraining_public_transfer_canary.py `
  --checkpoint artifacts.local/evidence/hftf/stage-c-d5-tartanground-cross-environment-v1/training/fold-0/directional-single-seed17/checkpoint.pt `
  --output artifacts.local/evidence/hftf/stage-c-d6-sanpo-paired-pretraining-public-transfer-canary-v2-source-audit/result.json
```

TartanGround → SANPO 课程只在上一个命令增加：

```powershell
  --paired-pretrained-state artifacts.local/evidence/hftf/stage-c-d6-tartanground-paired-relation-pretraining-transfer-canary-v1/paired-relation-state.pt
```

early joint-pair interaction：

```powershell
E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/research/hftf/run_stage_c_d6_joint_pair_interaction_public_transfer_canary.py `
  --checkpoint artifacts.local/evidence/hftf/stage-c-d5-tartanground-cross-environment-v1/training/fold-0/directional-single-seed17/checkpoint.pt `
  --output artifacts.local/evidence/hftf/stage-c-d6-joint-pair-interaction-public-transfer-canary-v0/result.json
```
